//! Cluster runtime. Each device is a socket endpoint. A coordinator walks the
//! schedule over control connections. Tensor payloads move hop by hop on data
//! connections, following the path the scheduler computed from the links.

use crate::mixtral::{arch, evaluate, sample_batch, sgd, Batch, Config};
use crate::stage::{
    aux_cotangent, aux_value, mixtral_cluster, read_checkpoint, run_stage, save_random,
    schedule_on, Cluster, Device, DeviceKind, Phase, Schedule, Step, WeightMem,
};
use std::collections::HashMap;
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const RUN: u8 = 1;
const PUSH: u8 = 2;
const UPDATE: u8 = 3;
const PEERS: u8 = 4;
const STOP: u8 = 5;
const ACK: u8 = 6;
const ERR: u8 = 7;

struct MapMem {
    dir: PathBuf,
    slots: HashMap<usize, Vec<f32>>,
}

impl WeightMem for MapMem {
    fn read(&mut self, slot: usize) -> Result<Vec<f32>, String> {
        self.slots
            .get(&slot)
            .cloned()
            .ok_or_else(|| format!("device is missing slot {slot}"))
    }
    fn write(&mut self, slot: usize, data: &[f32]) -> Result<(), String> {
        self.slots.insert(slot, data.to_vec());
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for value in data {
            bytes.extend(value.to_le_bytes());
        }
        std::fs::write(self.dir.join(format!("{slot}.f32")), bytes).map_err(|e| e.to_string())
    }
}

fn configure(stream: &TcpStream, timeout: Duration) {
    let _ = stream.set_nodelay(true);
    let _ = stream.set_read_timeout(Some(timeout));
    let _ = stream.set_write_timeout(Some(timeout));
}

fn write_frame(stream: &mut TcpStream, tag: u8, body: &[u8]) -> Result<(), String> {
    let len = (body.len() + 1) as u32;
    stream
        .write_all(&len.to_be_bytes())
        .map_err(|e| e.to_string())?;
    stream.write_all(&[tag]).map_err(|e| e.to_string())?;
    stream.write_all(body).map_err(|e| e.to_string())?;
    stream.flush().map_err(|e| e.to_string())
}

fn read_frame(stream: &mut TcpStream) -> Result<(u8, Vec<u8>), String> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).map_err(|e| e.to_string())?;
    let len = u32::from_be_bytes(len_buf) as usize;
    if len == 0 {
        return Err("empty frame".into());
    }
    let mut body = vec![0u8; len];
    stream.read_exact(&mut body).map_err(|e| e.to_string())?;
    Ok((body[0], body[1..].to_vec()))
}

fn write_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend(value.to_be_bytes());
}

fn write_u64(buf: &mut Vec<u8>, value: u64) {
    buf.extend(value.to_be_bytes());
}

fn write_str(buf: &mut Vec<u8>, value: &str) {
    write_u32(buf, value.len() as u32);
    buf.extend(value.as_bytes());
}

fn read_u32(buf: &[u8], at: &mut usize) -> Result<u32, String> {
    if *at + 4 > buf.len() {
        return Err("truncated frame".into());
    }
    let value = u32::from_be_bytes(buf[*at..*at + 4].try_into().unwrap());
    *at += 4;
    Ok(value)
}

fn read_u64(buf: &[u8], at: &mut usize) -> Result<u64, String> {
    if *at + 8 > buf.len() {
        return Err("truncated frame".into());
    }
    let value = u64::from_be_bytes(buf[*at..*at + 8].try_into().unwrap());
    *at += 8;
    Ok(value)
}

fn read_f32(buf: &[u8], at: &mut usize) -> Result<f32, String> {
    if *at + 4 > buf.len() {
        return Err("truncated frame".into());
    }
    let bytes: [u8; 4] = buf[*at..*at + 4].try_into().unwrap();
    *at += 4;
    Ok(f32::from_le_bytes(bytes))
}

fn read_str(buf: &[u8], at: &mut usize) -> Result<String, String> {
    let len = read_u32(buf, at)? as usize;
    if *at + len > buf.len() {
        return Err("truncated string".into());
    }
    let text = String::from_utf8(buf[*at..*at + len].to_vec()).map_err(|e| e.to_string())?;
    *at += len;
    Ok(text)
}

fn read_f32s(buf: &[u8], at: &mut usize) -> Result<Vec<f32>, String> {
    let n = read_u32(buf, at)? as usize;
    if *at + n * 4 > buf.len() {
        return Err("truncated floats".into());
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(read_f32(buf, at)?);
    }
    Ok(out)
}

fn write_f32s(buf: &mut Vec<u8>, values: &[f32]) {
    write_u32(buf, values.len() as u32);
    for value in values {
        buf.extend(value.to_le_bytes());
    }
}

struct Shared {
    inbox: Mutex<HashMap<String, Vec<f32>>>,
    peers: Mutex<HashMap<String, std::net::SocketAddr>>,
}

struct Drain(Arc<AtomicUsize>);

impl Drop for Drain {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

fn relay(
    name: &str,
    peers: &HashMap<String, std::net::SocketAddr>,
    inbox: &Mutex<HashMap<String, Vec<f32>>>,
    path: &[String],
    tensor: &str,
    values: &[f32],
) -> Result<(), String> {
    if path.is_empty() {
        return Err("empty relay path".into());
    }
    if path.len() == 1 {
        if path[0] != name {
            return Err(format!(
                "tensor {tensor} delivered to {name}, addressed to {}",
                path[0]
            ));
        }
        inbox
            .lock()
            .unwrap()
            .insert(tensor.to_string(), values.to_vec());
        return Ok(());
    }
    let next = &path[1];
    let addr = *peers
        .get(next)
        .ok_or_else(|| format!("no address for {next}"))?;
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(10))
        .map_err(|e| format!("connect {next}: {e}"))?;
    configure(&stream, Duration::from_secs(30));
    let mut body = Vec::new();
    write_u32(&mut body, (path.len() - 1) as u32);
    for hop in &path[1..] {
        write_str(&mut body, hop);
    }
    write_str(&mut body, tensor);
    write_f32s(&mut body, values);
    write_frame(&mut stream, PUSH, &body)?;
    let (tag, payload) = read_frame(&mut stream)?;
    if tag == ERR {
        return Err(String::from_utf8_lossy(&payload).into_owned());
    }
    if tag != ACK {
        return Err(format!("relay of {tensor} missing ack"));
    }
    Ok(())
}

fn data_session(name: String, mut stream: TcpStream, shared: Arc<Shared>) -> Result<(), String> {
    let (tag, body) = read_frame(&mut stream)?;
    if tag != PUSH {
        write_frame(&mut stream, ERR, b"expected tensor")?;
        return Err("data plane expected a tensor".into());
    }
    let mut at = 0usize;
    let hops = read_u32(&body, &mut at)? as usize;
    let mut path = Vec::with_capacity(hops);
    for _ in 0..hops {
        path.push(read_str(&body, &mut at)?);
    }
    let tensor = read_str(&body, &mut at)?;
    let values = read_f32s(&body, &mut at)?;
    if path.first().map(String::as_str) != Some(name.as_str()) {
        write_frame(&mut stream, ERR, b"wrong hop")?;
        return Err(format!(
            "{name} received a packet for {}",
            path.first().map(String::as_str).unwrap_or("?")
        ));
    }
    let result = if path.len() == 1 {
        shared.inbox.lock().unwrap().insert(tensor, values);
        Ok(())
    } else {
        let peers = shared.peers.lock().unwrap().clone();
        relay(&name, &peers, &shared.inbox, &path, &tensor, &values)
    };
    match result {
        Ok(()) => write_frame(&mut stream, ACK, &[]),
        Err(err) => {
            write_frame(&mut stream, ERR, err.as_bytes())?;
            Err(err)
        }
    }
}

fn accept_loop(
    listener: TcpListener,
    stop: Arc<AtomicBool>,
    inflight: Arc<AtomicUsize>,
    shared: Arc<Shared>,
    name: String,
) {
    while !stop.load(Ordering::Relaxed) {
        match listener.accept() {
            Ok((stream, _)) => {
                let _ = stream.set_nonblocking(false);
                configure(&stream, Duration::from_secs(30));
                inflight.fetch_add(1, Ordering::SeqCst);
                let drain = Drain(Arc::clone(&inflight));
                let session_name = name.clone();
                let session_shared = Arc::clone(&shared);
                thread::spawn(move || {
                    let _drain = drain;
                    let _ = data_session(session_name, stream, session_shared);
                });
            }
            Err(err)
                if err.kind() == ErrorKind::WouldBlock || err.kind() == ErrorKind::TimedOut =>
            {
                thread::sleep(Duration::from_millis(2));
            }
            Err(_) => break,
        }
    }
}

fn slot_of(spec: &crate::mixtral::Arch, name: &str) -> Result<usize, String> {
    spec.slots
        .iter()
        .position(|slot| slot.name == name)
        .ok_or_else(|| format!("unknown parameter {name}"))
}

fn load_owned(
    spec: &crate::mixtral::Arch,
    plan: &Schedule,
    name: &str,
    dir: &Path,
) -> Result<HashMap<usize, Vec<f32>>, String> {
    let mut owned = HashMap::new();
    for stage in &plan.stages {
        if stage.device != name {
            continue;
        }
        for param in &stage.params {
            let slot = slot_of(spec, param)?;
            let bytes =
                std::fs::read(dir.join(format!("{slot}.f32"))).map_err(|e| e.to_string())?;
            if bytes.len() % 4 != 0 {
                return Err(format!("slot {slot} is not a sequence of f32"));
            }
            let values = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            owned.insert(slot, values);
        }
    }
    Ok(owned)
}

fn fail(stream: &mut TcpStream, err: String) -> Result<(), String> {
    let _ = write_frame(stream, ERR, err.as_bytes());
    Err(err)
}

fn member(
    name: String,
    rendezvous: std::net::SocketAddr,
    cfg: Config,
    batch: Batch,
    plan: Schedule,
    dir: PathBuf,
) -> Result<(), String> {
    let data_listener = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    data_listener
        .set_nonblocking(true)
        .map_err(|e| e.to_string())?;
    let data_addr = data_listener.local_addr().map_err(|e| e.to_string())?;
    let shared = Arc::new(Shared {
        inbox: Mutex::new(HashMap::new()),
        peers: Mutex::new(HashMap::new()),
    });
    let stop = Arc::new(AtomicBool::new(false));
    let inflight = Arc::new(AtomicUsize::new(0));
    let data_thread = {
        let stop = Arc::clone(&stop);
        let inflight = Arc::clone(&inflight);
        let shared = Arc::clone(&shared);
        let name = name.clone();
        thread::spawn(move || accept_loop(data_listener, stop, inflight, shared, name))
    };
    let outcome = member_loop(name, rendezvous, data_addr, cfg, batch, plan, dir, &shared);
    stop.store(true, Ordering::Relaxed);
    let joined = data_thread.join();
    let deadline = Instant::now() + Duration::from_secs(5);
    while inflight.load(Ordering::SeqCst) > 0 && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(1));
    }
    let drained = inflight.load(Ordering::SeqCst) == 0;
    match (outcome, joined, drained) {
        (Err(err), _, _) => Err(err),
        (_, Err(_), _) => Err("data plane thread panicked".into()),
        (_, _, false) => Err("data plane did not drain".into()),
        (Ok(()), Ok(()), true) => Ok(()),
    }
}

fn member_loop(
    name: String,
    rendezvous: std::net::SocketAddr,
    data_addr: std::net::SocketAddr,
    cfg: Config,
    batch: Batch,
    plan: Schedule,
    dir: PathBuf,
    shared: &Arc<Shared>,
) -> Result<(), String> {
    let spec = arch(&cfg);
    let owned = load_owned(&spec, &plan, &name, &dir)?;
    let mut weights = MapMem { dir, slots: owned };
    let mut grads: HashMap<usize, Vec<f32>> = HashMap::new();
    let mut ctrl = TcpStream::connect(rendezvous).map_err(|e| e.to_string())?;
    configure(&ctrl, Duration::from_secs(60));
    let mut hello = Vec::new();
    write_str(&mut hello, &name);
    write_str(&mut hello, &data_addr.to_string());
    write_frame(&mut ctrl, ACK, &hello)?;
    loop {
        let (tag, body) = read_frame(&mut ctrl)?;
        if tag == STOP {
            break;
        }
        if tag == PEERS {
            let mut at = 0usize;
            let n = read_u32(&body, &mut at)? as usize;
            let mut peers = HashMap::new();
            for _ in 0..n {
                let peer = read_str(&body, &mut at)?;
                let addr = read_str(&body, &mut at)?;
                let parsed: std::net::SocketAddr = addr
                    .parse()
                    .map_err(|e: std::net::AddrParseError| e.to_string())?;
                peers.insert(peer, parsed);
            }
            *shared.peers.lock().unwrap() = peers;
            write_frame(&mut ctrl, ACK, &[])?;
            continue;
        }
        if tag == PUSH {
            let mut at = 0usize;
            let tensor = read_str(&body, &mut at)?;
            let hops = read_u32(&body, &mut at)? as usize;
            let mut path = Vec::with_capacity(hops);
            for _ in 0..hops {
                path.push(read_str(&body, &mut at)?);
            }
            let values = shared.inbox.lock().unwrap().get(&tensor).cloned();
            let Some(values) = values else {
                return fail(
                    &mut ctrl,
                    format!("{name} cannot send missing tensor {tensor}"),
                );
            };
            let peers = shared.peers.lock().unwrap().clone();
            if let Err(err) = relay(&name, &peers, &shared.inbox, &path, &tensor, &values) {
                return fail(&mut ctrl, err);
            }
            write_frame(&mut ctrl, ACK, &[])?;
            continue;
        }
        // An adjoint run records gradients and leaves the checkpoint unchanged.
        // The update writes θ ← θ − η ∇L for the parameters that stage owns.
        if tag == UPDATE {
            let mut at = 0usize;
            let index = read_u32(&body, &mut at)? as usize;
            let lr = read_f32(&body, &mut at)?;
            let stage = plan
                .stages
                .get(index)
                .ok_or_else(|| format!("update index {index} is out of range"))?;
            if stage.device != name {
                return fail(
                    &mut ctrl,
                    format!("update for {} arrived at {name}", stage.device),
                );
            }
            for param in &stage.params {
                let slot = slot_of(&spec, param)?;
                let grad = grads
                    .get(&slot)
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; spec.slots[slot].shape.iter().product::<usize>()]);
                let mut value = weights.read(slot)?;
                if value.len() != grad.len() {
                    return fail(
                        &mut ctrl,
                        format!(
                            "slot {slot} has {} values and {} gradients",
                            value.len(),
                            grad.len()
                        ),
                    );
                }
                for (weight, g) in value.iter_mut().zip(&grad) {
                    *weight -= lr * *g;
                }
                weights.write(slot, &value)?;
            }
            write_frame(&mut ctrl, ACK, &[])?;
            continue;
        }
        if tag == RUN {
            let mut at = 0usize;
            let index = read_u32(&body, &mut at)? as usize;
            let phase = if read_u32(&body, &mut at)? == 0 {
                Phase::Forward
            } else {
                Phase::Adjoint
            };
            let aux = read_f32s(&body, &mut at)?;
            let budget = read_u64(&body, &mut at)?;
            let stage = plan
                .stages
                .get(index)
                .ok_or_else(|| format!("run index {index} is out of range"))?
                .clone();
            if stage.device != name {
                return fail(&mut ctrl, format!("run {} arrived at {name}", stage.device));
            }
            let inbox = shared.inbox.lock().unwrap().clone();
            let produced = match run_stage(
                &cfg,
                &batch,
                &stage,
                &plan.stages,
                phase,
                &mut weights,
                &inbox,
                &aux,
                plan.checkpoint_bytes,
                budget,
            ) {
                Ok(produced) => produced,
                Err(err) => return fail(&mut ctrl, format!("{}: {err}", stage.name)),
            };
            if phase == Phase::Adjoint {
                grads.extend(produced.grads);
            }
            {
                let mut box_ = shared.inbox.lock().unwrap();
                for (key, value) in produced.tensors {
                    box_.insert(key, value);
                }
            }
            let mut ack = Vec::new();
            if let Some(loss) = produced.loss {
                ack.push(1);
                ack.extend(loss.to_le_bytes());
            } else {
                ack.push(0);
                ack.extend(0f32.to_le_bytes());
            }
            write_u32(&mut ack, produced.counts.len() as u32);
            for count in produced.counts {
                write_u32(&mut ack, count);
            }
            write_f32s(&mut ack, &produced.prob_sum);
            write_frame(&mut ctrl, ACK, &ack)?;
            continue;
        }
        return fail(&mut ctrl, format!("unknown control tag {tag}"));
    }
    Ok(())
}

fn accept_members(
    gate: &TcpListener,
    controls: &mut HashMap<String, TcpStream>,
    peers: &mut HashMap<String, std::net::SocketAddr>,
    n: usize,
) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(30);
    while peers.len() < n {
        match gate.accept() {
            Ok((mut stream, _)) => {
                stream.set_nonblocking(false).map_err(|e| e.to_string())?;
                configure(&stream, Duration::from_secs(60));
                let (tag, body) = read_frame(&mut stream)?;
                if tag != ACK {
                    return Err("member hello failed".into());
                }
                let mut at = 0usize;
                let name = read_str(&body, &mut at)?;
                let addr = read_str(&body, &mut at)?;
                let parsed: std::net::SocketAddr = addr
                    .parse()
                    .map_err(|e: std::net::AddrParseError| e.to_string())?;
                if peers.contains_key(&name) {
                    return Err(format!("duplicate member {name}"));
                }
                peers.insert(name.clone(), parsed);
                controls.insert(name, stream);
            }
            Err(err)
                if err.kind() == ErrorKind::WouldBlock || err.kind() == ErrorKind::TimedOut =>
            {
                if Instant::now() >= deadline {
                    return Err(format!(
                        "timed out waiting for cluster members, joined {}",
                        peers.len()
                    ));
                }
                thread::sleep(Duration::from_millis(5));
            }
            Err(err) => return Err(err.to_string()),
        }
    }
    Ok(())
}

fn exchange_peers(
    controls: &mut HashMap<String, TcpStream>,
    peers: &HashMap<String, std::net::SocketAddr>,
) -> Result<(), String> {
    let mut peer_body = Vec::new();
    write_u32(&mut peer_body, peers.len() as u32);
    for (name, addr) in peers {
        write_str(&mut peer_body, name);
        write_str(&mut peer_body, &addr.to_string());
    }
    for stream in controls.values_mut() {
        write_frame(stream, PEERS, &peer_body)?;
        let (tag, payload) = read_frame(stream)?;
        if tag == ERR {
            return Err(String::from_utf8_lossy(&payload).into_owned());
        }
        if tag != ACK {
            return Err("peer exchange failed".into());
        }
    }
    Ok(())
}

fn device_budget(cluster: &Cluster, name: &str) -> Result<u64, String> {
    cluster
        .devices
        .iter()
        .find(|device| device.name == name)
        .map(|device| device.buffer_bytes)
        .ok_or_else(|| format!("unknown device {name}"))
}

fn drive(
    cfg: &Config,
    batch: &Batch,
    cluster: &Cluster,
    plan: &Schedule,
    controls: &mut HashMap<String, TcpStream>,
    lr: f32,
) -> Result<f32, String> {
    let mut counts = vec![vec![0u32; cfg.n_experts]; cfg.n_layers];
    let mut prob_sum = vec![0f32; cfg.n_experts];
    let mut rows = 0usize;
    let ntok = batch.batch * batch.seq;
    let mut ce = None;
    for step in &plan.steps {
        match step {
            Step::Run {
                stage,
                device,
                phase,
            } => {
                let index = plan
                    .stages
                    .iter()
                    .position(|item| item.name == *stage)
                    .ok_or_else(|| format!("schedule is missing stage {stage}"))?;
                let aux = if *phase == Phase::Adjoint {
                    aux_cotangent(cfg, &counts, ntok)
                } else {
                    Vec::new()
                };
                let mut body = Vec::new();
                write_u32(&mut body, index as u32);
                write_u32(&mut body, if *phase == Phase::Forward { 0 } else { 1 });
                write_f32s(&mut body, &aux);
                write_u64(&mut body, device_budget(cluster, device)?);
                let stream = controls
                    .get_mut(device)
                    .ok_or_else(|| format!("no control connection for {device}"))?;
                write_frame(stream, RUN, &body)
                    .map_err(|err| format!("run {stage} on {device}: {err}"))?;
                let (tag, ack) =
                    read_frame(stream).map_err(|err| format!("run {stage} on {device}: {err}"))?;
                if tag == ERR {
                    return Err(format!(
                        "run {stage} on {device}: {}",
                        String::from_utf8_lossy(&ack)
                    ));
                }
                if tag != ACK || ack.len() < 5 {
                    return Err(format!("run {stage} on {device} returned a short ack"));
                }
                if ack[0] == 1 {
                    ce = Some(f32::from_le_bytes(ack[1..5].try_into().unwrap()));
                }
                let mut at = 5usize;
                let ncounts = read_u32(&ack, &mut at)? as usize;
                let mut local_counts = Vec::with_capacity(ncounts);
                for _ in 0..ncounts {
                    local_counts.push(read_u32(&ack, &mut at)?);
                }
                let local_prob = read_f32s(&ack, &mut at)?;
                if !local_counts.is_empty() {
                    if let crate::stage::StageKind::Router(i) | crate::stage::StageKind::Layer(i) =
                        plan.stages[index].kind
                    {
                        counts[i] = local_counts;
                        for (slot, value) in prob_sum.iter_mut().zip(&local_prob) {
                            *slot += *value;
                        }
                        rows += ntok;
                    }
                }
            }
            Step::Send(message) => {
                let mut body = Vec::new();
                write_str(&mut body, &message.tensor);
                write_u32(&mut body, message.path.len() as u32);
                for hop in &message.path {
                    write_str(&mut body, hop);
                }
                let stream = controls
                    .get_mut(&message.src)
                    .ok_or_else(|| format!("no control connection for {}", message.src))?;
                write_frame(stream, PUSH, &body)
                    .map_err(|err| format!("send {} via {}: {err}", message.tensor, message.src))?;
                let (tag, ack) = read_frame(stream)
                    .map_err(|err| format!("send {} via {}: {err}", message.tensor, message.src))?;
                if tag == ERR {
                    return Err(format!(
                        "send {} via {}: {}",
                        message.tensor,
                        message.src,
                        String::from_utf8_lossy(&ack)
                    ));
                }
                if tag != ACK {
                    return Err(format!(
                        "send {} via {} was not acknowledged",
                        message.tensor, message.src
                    ));
                }
            }
            Step::Update { device, stage, .. } => {
                let index = plan
                    .stages
                    .iter()
                    .position(|item| item.name == *stage)
                    .ok_or_else(|| format!("schedule is missing stage {stage}"))?;
                let mut body = Vec::new();
                write_u32(&mut body, index as u32);
                body.extend(lr.to_le_bytes());
                let stream = controls
                    .get_mut(device)
                    .ok_or_else(|| format!("no control connection for {device}"))?;
                write_frame(stream, UPDATE, &body)
                    .map_err(|err| format!("update {stage} on {device}: {err}"))?;
                let (tag, ack) = read_frame(stream)
                    .map_err(|err| format!("update {stage} on {device}: {err}"))?;
                if tag == ERR {
                    return Err(format!(
                        "update {stage} on {device}: {}",
                        String::from_utf8_lossy(&ack)
                    ));
                }
                if tag != ACK {
                    return Err(format!("update {stage} on {device} was not acknowledged"));
                }
            }
        }
    }
    let ce = ce.ok_or("head did not report a loss")?;
    let expected_rows = cfg.n_layers * ntok;
    if rows != expected_rows {
        return Err(format!(
            "router reports covered {rows} token-rows, expected {expected_rows}"
        ));
    }
    Ok(ce + aux_value(cfg, &prob_sum, rows, &counts, ntok))
}

/// Execute one forward, adjoint, and parameter update on `cluster`.
///
/// Each device is a member thread bound to `127.0.0.1`. The coordinator walks
/// the schedule over control connections. Tensor values move along each
/// message's path on data connections. Parameter files in `dir` are
/// initialized from `seed` and rewritten by the members. `lr` is the step
/// size of `θ ← θ − η ∇L`.
pub fn run_on_cluster(
    cfg: &Config,
    batch: &Batch,
    cluster: &Cluster,
    dir: &Path,
    seed: u64,
    lr: f32,
) -> Result<f32, String> {
    let plan = schedule_on(cfg, batch.batch, batch.seq, cluster)?;
    save_random(dir, cfg, seed)?;
    let gate = TcpListener::bind("127.0.0.1:0").map_err(|e| e.to_string())?;
    gate.set_nonblocking(true).map_err(|e| e.to_string())?;
    let gate_addr = gate.local_addr().map_err(|e| e.to_string())?;
    let mut joins = Vec::new();
    for device in &cluster.devices {
        let name = device.name.clone();
        let cfg = cfg.clone();
        let batch = batch.clone();
        let plan = plan.clone();
        let dir = dir.to_path_buf();
        joins.push(thread::spawn(move || {
            member(name, gate_addr, cfg, batch, plan, dir)
        }));
    }
    let mut controls: HashMap<String, TcpStream> = HashMap::new();
    let mut peers: HashMap<String, std::net::SocketAddr> = HashMap::new();
    let result = (|| {
        accept_members(&gate, &mut controls, &mut peers, cluster.devices.len())?;
        exchange_peers(&mut controls, &peers)?;
        drive(cfg, batch, cluster, &plan, &mut controls, lr)
    })();
    for stream in controls.values_mut() {
        let _ = write_frame(stream, STOP, &[]);
    }
    let mut result = result;
    for join in joins {
        match join.join() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                if result.is_ok() {
                    result = Err(err);
                }
            }
            Err(_) => {
                if result.is_ok() {
                    result = Err("member thread panicked".into());
                }
            }
        }
    }
    result
}

fn compare_step(
    cfg: &Config,
    loss: f32,
    expected: f32,
    mono: &[f32],
    got: &[f32],
) -> Result<(), String> {
    if !loss.is_finite() || (loss - expected).abs() > 1e-3 {
        return Err(format!(
            "distributed loss {loss} vs single-buffer {expected}"
        ));
    }
    if mono.len() != got.len() {
        return Err(format!(
            "checkpoint has {} values, single-buffer has {}",
            got.len(),
            mono.len()
        ));
    }
    let spec = arch(cfg);
    let mut offset = 0usize;
    let mut worst = 0f32;
    let mut worst_name = String::new();
    for slot in &spec.slots {
        let n: usize = slot.shape.iter().product();
        let err = mono[offset..offset + n]
            .iter()
            .zip(&got[offset..offset + n])
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        if err > worst {
            worst = err;
            worst_name = slot.name.clone();
        }
        offset += n;
    }
    if worst > 1e-3 {
        return Err(format!("parameter {worst_name} differs by {worst}"));
    }
    Ok(())
}

/// One distributed step of the demonstration Mixtral must match the
/// single-buffer adjoint, on the heterogeneous star and on one device.
pub fn verify_distributed_step(dir: &Path) -> Result<(), String> {
    let cfg = Config::demo_smooth();
    let batch = sample_batch(&cfg, 2, 6, 7);
    let seed = 0x4D49_5854;
    let lr = 0.01;
    save_random(&dir.join("init"), &cfg, seed)?;
    let base = read_checkpoint(&dir.join("init"), &cfg)?;
    let ev = evaluate(&cfg, &base, &batch);
    let mut mono = base;
    sgd(&mut mono, &ev.grads, lr);
    let cluster = mixtral_cluster();
    let plan = schedule_on(&cfg, batch.batch, batch.seq, &cluster)?;
    if !plan.messages.iter().any(|message| message.path.len() >= 3) {
        return Err("cluster schedule has no multi-hop relay".into());
    }
    let loss = run_on_cluster(&cfg, &batch, &cluster, dir, seed, lr)?;
    compare_step(&cfg, loss, ev.loss, &mono, &read_checkpoint(dir, &cfg)?)?;
    let one = Cluster {
        devices: vec![Device {
            name: "gpu0".into(),
            kind: DeviceKind::Gpu,
            code_bytes: 1 << 20,
            buffer_bytes: 1 << 40,
        }],
        links: Vec::new(),
    };
    let one_dir = dir.join("one");
    let loss_one = run_on_cluster(&cfg, &batch, &one, &one_dir, seed, lr)?;
    compare_step(
        &cfg,
        loss_one,
        ev.loss,
        &mono,
        &read_checkpoint(&one_dir, &cfg)?,
    )?;
    Ok(())
}
