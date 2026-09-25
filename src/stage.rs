//! Memory-bounded schedule for Mixtral parameter loads and adjoint updates.
//!
//! `Memory` is a compilation parameter with two budgets. Code memory holds the
//! kernel images a stage launches. Buffer memory holds checkpoints, parameter
//! tensors, their adjoints, and the scratch tape. A stage loads only its
//! kernels and its buffers, applies `θ ← θ − η ∇L`, and drops both.

use crate::ad::Graph;
use crate::mixtral::{arch, attention, each_init, moe, router, sgd, swiglu, AttnW, Batch, Config};
use std::fs;
use std::path::Path;

/// Working memory available to one stage. Kernel images and data buffers
/// are separate address spaces and are scheduled against separate budgets.
#[derive(Clone, Copy, Debug)]
pub struct Memory {
    pub code_bytes: u64,
    pub buffer_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferRole {
    Checkpoint,
    Parameter,
    Adjoint,
    Scratch,
}

#[derive(Clone, Debug)]
pub struct Buffer {
    pub name: String,
    pub role: BufferRole,
    pub bytes: u64,
}

#[derive(Clone, Debug)]
pub(crate) enum StageKind {
    Embed,
    Layer(usize),
    Attention(usize),
    Router(usize),
    Expert { layer: usize, expert: usize },
    Head,
}

#[derive(Clone, Debug)]
pub struct Stage {
    pub name: String,
    pub params: Vec<String>,
    pub kernels: Vec<&'static str>,
    pub code_bytes: u64,
    pub buffers: Vec<Buffer>,
    pub buffer_bytes: u64,
    /// Resident buffer bytes. Kernel images are not added into this figure.
    pub peak_bytes: u64,
    /// Device in the cluster that runs this stage.
    pub device: String,
    pub(crate) kind: StageKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    Cpu,
    Gpu,
}

/// One machine in a heterogeneous cluster. Code and buffer budgets are the
/// memory on that device, not a shared pool.
#[derive(Clone, Debug)]
pub struct Device {
    pub name: String,
    pub kind: DeviceKind,
    pub code_bytes: u64,
    pub buffer_bytes: u64,
}

/// Undirected link. The scheduler routes messages along these edges.
#[derive(Clone, Debug)]
pub struct Link {
    pub a: String,
    pub b: String,
}

#[derive(Clone, Debug)]
pub struct Cluster {
    pub devices: Vec<Device>,
    pub links: Vec<Link>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Forward,
    Adjoint,
}

#[derive(Clone, Debug)]
pub struct Message {
    pub phase: Phase,
    pub src: String,
    pub dst: String,
    pub tensor: String,
    pub bytes: u64,
    pub path: Vec<String>,
}

#[derive(Clone, Debug)]
pub enum Step {
    Run {
        stage: String,
        device: String,
        phase: Phase,
    },
    Send(Message),
    Update {
        stage: String,
        device: String,
        params: Vec<String>,
    },
}

#[derive(Clone, Debug)]
pub struct Schedule {
    /// Buffer budget. Kernel images are accounted in `code_budget`.
    pub budget: u64,
    pub code_budget: u64,
    pub checkpoint_bytes: u64,
    pub stages: Vec<Stage>,
    pub peak_bytes: u64,
    pub peak_code_bytes: u64,
    pub messages: Vec<Message>,
    pub steps: Vec<Step>,
}

pub trait WeightMem {
    fn read(&mut self, slot: usize) -> Result<Vec<f32>, String>;
    fn write(&mut self, slot: usize, data: &[f32]) -> Result<(), String>;
}

struct DiskWeights {
    dir: std::path::PathBuf,
}

impl WeightMem for DiskWeights {
    fn read(&mut self, slot: usize) -> Result<Vec<f32>, String> {
        let bytes = fs::read(self.dir.join(format!("{slot}.f32"))).map_err(|e| e.to_string())?;
        if bytes.len() % 4 != 0 {
            return Err(format!("slot {slot} is not a sequence of f32"));
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }
    fn write(&mut self, slot: usize, data: &[f32]) -> Result<(), String> {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for x in data {
            bytes.extend(x.to_le_bytes());
        }
        fs::write(self.dir.join(format!("{slot}.f32")), bytes).map_err(|e| e.to_string())
    }
}

fn nbytes(shape: &[usize]) -> u64 {
    shape.iter().map(|v| *v as u64).product::<u64>() * 4
}

fn add_shape(total: &mut u64, shape: &[usize]) {
    *total += nbytes(shape);
}

fn attention_values(cfg: &Config, batch: usize, seq: usize) -> u64 {
    let d = cfg.dim;
    let hq = cfg.n_heads;
    let hkv = cfg.n_kv_heads;
    let hd = cfg.head_dim();
    let (b, t) = (batch, seq);
    let mut n = 0u64;
    add_shape(&mut n, &[b, t, d]);
    add_shape(&mut n, &[d]);
    add_shape(&mut n, &[d, hq * hd]);
    add_shape(&mut n, &[d, hkv * hd]);
    add_shape(&mut n, &[d, hkv * hd]);
    add_shape(&mut n, &[hq * hd, d]);
    add_shape(&mut n, &[b, t, d]);
    add_shape(&mut n, &[b, t, hq * hd]);
    add_shape(&mut n, &[b, t, hq, hd]);
    add_shape(&mut n, &[b, t, hkv * hd]);
    add_shape(&mut n, &[b, t, hkv, hd]);
    add_shape(&mut n, &[b, t, hkv * hd]);
    add_shape(&mut n, &[b, t, hkv, hd]);
    add_shape(&mut n, &[b, t, hq, hd]);
    add_shape(&mut n, &[b, t, hkv, hd]);
    add_shape(&mut n, &[b, hq, t, hd]);
    add_shape(&mut n, &[b, hkv, t, hd]);
    add_shape(&mut n, &[b, hkv, t, hd]);
    add_shape(&mut n, &[b, hq, t, hd]);
    add_shape(&mut n, &[b, hq, t, hd]);
    add_shape(&mut n, &[b, hq, hd, t]);
    add_shape(&mut n, &[b, hq, t, t]);
    add_shape(&mut n, &[b, hq, t, t]);
    add_shape(&mut n, &[b, hq, t, t]);
    add_shape(&mut n, &[b, hq, t, hd]);
    add_shape(&mut n, &[b, t, hq, hd]);
    add_shape(&mut n, &[b, t, d]);
    add_shape(&mut n, &[b, t, d]);
    add_shape(&mut n, &[b, t, d]);
    n
}

fn moe_extra(cfg: &Config, batch: usize, seq: usize, experts: usize) -> u64 {
    let d = cfg.dim;
    let e = cfg.n_experts;
    let inter = cfg.intermediate;
    let ntok = batch * seq;
    let (b, t) = (batch, seq);
    let mut n = 0u64;
    add_shape(&mut n, &[d]);
    add_shape(&mut n, &[d, e]);
    add_shape(&mut n, &[b, t, d]);
    add_shape(&mut n, &[ntok, d]);
    add_shape(&mut n, &[ntok, e]);
    add_shape(&mut n, &[ntok, e]);
    add_shape(&mut n, &[ntok, e]);
    add_shape(&mut n, &[ntok, e]);
    add_shape(&mut n, &[ntok, 1]);
    add_shape(&mut n, &[ntok, e]);
    add_shape(&mut n, &[ntok, d]);
    for _ in 0..experts {
        add_shape(&mut n, &[d, inter]);
        add_shape(&mut n, &[d, inter]);
        add_shape(&mut n, &[inter, d]);
        add_shape(&mut n, &[ntok, inter]);
        add_shape(&mut n, &[ntok, inter]);
        add_shape(&mut n, &[ntok, inter]);
        add_shape(&mut n, &[ntok, inter]);
        add_shape(&mut n, &[ntok, d]);
        add_shape(&mut n, &[ntok, 1]);
        add_shape(&mut n, &[ntok, d]);
        add_shape(&mut n, &[ntok, d]);
    }
    add_shape(&mut n, &[b, t, d]);
    add_shape(&mut n, &[b, t, d]);
    n
}

fn expert_values(cfg: &Config, batch: usize, seq: usize) -> u64 {
    let d = cfg.dim;
    let inter = cfg.intermediate;
    let ntok = batch * seq;
    let mut n = 0u64;
    add_shape(&mut n, &[ntok, d]);
    add_shape(&mut n, &[ntok, 1]);
    add_shape(&mut n, &[d, inter]);
    add_shape(&mut n, &[d, inter]);
    add_shape(&mut n, &[inter, d]);
    add_shape(&mut n, &[ntok, inter]);
    add_shape(&mut n, &[ntok, inter]);
    add_shape(&mut n, &[ntok, inter]);
    add_shape(&mut n, &[ntok, inter]);
    add_shape(&mut n, &[ntok, d]);
    add_shape(&mut n, &[ntok, d]);
    n
}

fn router_values(cfg: &Config, batch: usize, seq: usize) -> u64 {
    let d = cfg.dim;
    let e = cfg.n_experts;
    let ntok = batch * seq;
    let mut n = 0u64;
    add_shape(&mut n, &[batch, seq, d]);
    add_shape(&mut n, &[d]);
    add_shape(&mut n, &[d, e]);
    add_shape(&mut n, &[batch, seq, d]);
    add_shape(&mut n, &[ntok, d]);
    add_shape(&mut n, &[ntok, e]);
    add_shape(&mut n, &[ntok, e]);
    add_shape(&mut n, &[ntok, e]);
    add_shape(&mut n, &[ntok, e]);
    add_shape(&mut n, &[ntok, 1]);
    add_shape(&mut n, &[ntok, e]);
    n
}

fn head_values(cfg: &Config, batch: usize, seq: usize) -> u64 {
    let (b, t, d, v) = (batch, seq, cfg.dim, cfg.vocab);
    let mut n = 0u64;
    add_shape(&mut n, &[b, t, d]);
    add_shape(&mut n, &[d]);
    add_shape(&mut n, &[d, v]);
    add_shape(&mut n, &[b, t, d]);
    add_shape(&mut n, &[b, t, v]);
    add_shape(&mut n, &[b, t - 1, v]);
    add_shape(&mut n, &[b * (t - 1), v]);
    n + 64
}

fn embed_values(cfg: &Config, batch: usize, seq: usize) -> u64 {
    nbytes(&[cfg.vocab, cfg.dim]) + nbytes(&[batch, seq, cfg.dim])
}

fn saved_bytes(cfg: &Config, batch: usize, seq: usize, n_split: usize) -> u64 {
    let act = nbytes(&[batch, seq, cfg.dim]);
    let ntok = (batch * seq) as u64;
    let e = cfg.n_experts as u64;
    let d = cfg.dim as u64;
    act * (cfg.n_layers as u64 + 1)
        + n_split as u64 * act
        + n_split as u64 * ntok * d * 4
        + n_split as u64 * ntok * e * 4
        + act * 4
}

fn peak(saved: u64, values: u64) -> u64 {
    saved + values * 2 + 65_536
}

fn kernel_names(kind: &StageKind) -> Vec<&'static str> {
    match kind {
        StageKind::Embed => vec!["embed", "embed_bwd"],
        StageKind::Attention(_) => vec![
            "gemm",
            "gemm_dgrad",
            "gemm_wgrad",
            "rmsnorm_fwd",
            "rmsnorm_bwd",
            "rope",
            "softmax_fwd",
            "softmax_bwd",
        ],
        StageKind::Router(_) => vec![
            "gemm",
            "gemm_dgrad",
            "gemm_wgrad",
            "rmsnorm_fwd",
            "rmsnorm_bwd",
            "softmax_fwd",
            "softmax_bwd",
        ],
        StageKind::Expert { .. } => {
            vec!["gemm", "gemm_dgrad", "gemm_wgrad", "silu_fwd", "silu_bwd"]
        }
        StageKind::Head => vec![
            "gemm",
            "gemm_dgrad",
            "gemm_wgrad",
            "rmsnorm_fwd",
            "rmsnorm_bwd",
            "ce_grad",
        ],
        StageKind::Layer(_) => vec![
            "gemm",
            "gemm_dgrad",
            "gemm_wgrad",
            "rmsnorm_fwd",
            "rmsnorm_bwd",
            "rope",
            "softmax_fwd",
            "softmax_bwd",
            "silu_fwd",
            "silu_bwd",
        ],
    }
}

fn code_bytes(kind: &StageKind) -> u64 {
    kernel_names(kind)
        .iter()
        .map(|name| crate::cuda_emit::kernel_code_bytes(name))
        .sum()
}

fn make_stage(
    spec: &crate::mixtral::Arch,
    name: String,
    kind: StageKind,
    param_ids: &[usize],
    values: u64,
    saved: u64,
) -> Stage {
    let kernels = kernel_names(&kind);
    let code_bytes = code_bytes(&kind);
    let mut buffers = Vec::new();
    let mut param_bytes = 0u64;
    let mut params = Vec::new();
    for id in param_ids {
        let bytes = nbytes(&spec.slots[*id].shape);
        param_bytes += bytes;
        let slot = spec.slots[*id].name.clone();
        buffers.push(Buffer {
            name: slot.clone(),
            role: BufferRole::Parameter,
            bytes,
        });
        buffers.push(Buffer {
            name: format!("{slot} adjoint"),
            role: BufferRole::Adjoint,
            bytes,
        });
        params.push(slot);
    }
    buffers.push(Buffer {
        name: "checkpoints".into(),
        role: BufferRole::Checkpoint,
        bytes: saved,
    });
    let scratch = values * 2 + 65_536 - param_bytes * 2;
    buffers.push(Buffer {
        name: "tape".into(),
        role: BufferRole::Scratch,
        bytes: scratch,
    });
    let buffer_bytes = buffers.iter().map(|b| b.bytes).sum();
    Stage {
        name,
        params,
        kernels,
        code_bytes,
        buffers,
        buffer_bytes,
        peak_bytes: buffer_bytes,
        device: String::new(),
        kind,
    }
}

/// A training cluster for Mixtral: one CPU with a large host buffer, two GPUs
/// that can hold an expert, and one GPU that is too small for any stage.
pub fn mixtral_cluster() -> Cluster {
    Cluster {
        devices: vec![
            Device {
                name: "cpu0".into(),
                kind: DeviceKind::Cpu,
                code_bytes: 1 << 20,
                buffer_bytes: 64 << 30,
            },
            Device {
                name: "gpu0".into(),
                kind: DeviceKind::Gpu,
                code_bytes: 1 << 20,
                buffer_bytes: 8 << 30,
            },
            Device {
                name: "gpu1".into(),
                kind: DeviceKind::Gpu,
                code_bytes: 1 << 20,
                buffer_bytes: 24 << 30,
            },
            Device {
                name: "gpu2".into(),
                kind: DeviceKind::Gpu,
                code_bytes: 1 << 20,
                buffer_bytes: 256 << 20,
            },
        ],
        links: vec![
            Link {
                a: "cpu0".into(),
                b: "gpu0".into(),
            },
            Link {
                a: "cpu0".into(),
                b: "gpu1".into(),
            },
            Link {
                a: "cpu0".into(),
                b: "gpu2".into(),
            },
        ],
    }
}

/// Smallest code and buffer budgets that can hold one parameter group, its
/// adjoint, and the kernels that group launches.
pub fn minimum_budget(cfg: &Config, batch: usize, seq: usize) -> Result<Memory, String> {
    if seq < 2 {
        return Err("sequence length must leave a next-token target".into());
    }
    let saved = saved_bytes(cfg, batch, seq, cfg.n_layers);
    let buffer_bytes = peak(saved, attention_values(cfg, batch, seq))
        .max(peak(saved, expert_values(cfg, batch, seq)))
        .max(peak(saved, router_values(cfg, batch, seq)))
        .max(peak(saved, head_values(cfg, batch, seq)))
        .max(peak(saved, embed_values(cfg, batch, seq)));
    let code_bytes = code_bytes(&StageKind::Attention(0))
        .max(code_bytes(&StageKind::Expert {
            layer: 0,
            expert: 0,
        }))
        .max(code_bytes(&StageKind::Router(0)))
        .max(code_bytes(&StageKind::Head))
        .max(code_bytes(&StageKind::Embed));
    Ok(Memory {
        code_bytes,
        buffer_bytes,
    })
}

pub fn schedule(
    cfg: &Config,
    batch: usize,
    seq: usize,
    memory: Memory,
) -> Result<Schedule, String> {
    schedule_on(
        cfg,
        batch,
        seq,
        &Cluster {
            devices: vec![Device {
                name: "gpu0".into(),
                kind: DeviceKind::Gpu,
                code_bytes: memory.code_bytes,
                buffer_bytes: memory.buffer_bytes,
            }],
            links: Vec::new(),
        },
    )
}

/// Place every stage on a device in `cluster`. A stage runs only on a device
/// whose code budget holds that device's kernel images and whose buffer budget
/// holds the stage's checkpoints, parameters, adjoints, and scratch.
pub fn schedule_on(
    cfg: &Config,
    batch: usize,
    seq: usize,
    cluster: &Cluster,
) -> Result<Schedule, String> {
    if cluster.devices.is_empty() {
        return Err("cluster has no devices".into());
    }
    if seq < 2 {
        return Err("sequence length must leave a next-token target".into());
    }
    let spec = arch(cfg);
    let mut stages = Vec::new();
    let mut n_split = 0usize;
    for layer in &spec.layers {
        let li = layer_index_from_spec(&spec, layer);
        let joint = attention_values(cfg, batch, seq) + moe_extra(cfg, batch, seq, cfg.n_experts);
        let saved_if_split = saved_bytes(cfg, batch, seq, n_split + 1);
        let joint_peak = peak(saved_bytes(cfg, batch, seq, n_split), joint);
        let joint_kind = StageKind::Layer(li);
        let expert_peak_for_spread = peak(saved_if_split, expert_values(cfg, batch, seq));
        let expert_kind_for_spread = StageKind::Expert {
            layer: li,
            expert: 0,
        };
        let gpus_for_expert = cluster
            .devices
            .iter()
            .filter(|device| {
                device.kind == DeviceKind::Gpu
                    && device_fits(device, expert_peak_for_spread, &expert_kind_for_spread)
            })
            .count();
        if some_device_fits(cluster, joint_peak, &joint_kind) && gpus_for_expert < 2 {
            let mut ids = vec![
                layer.attn_norm,
                layer.wq,
                layer.wk,
                layer.wv,
                layer.wo,
                layer.ffn_norm,
                layer.gate,
            ];
            for e in &layer.experts {
                ids.extend(e);
            }
            stages.push(make_stage(
                &spec,
                format!("layers.{li}"),
                joint_kind,
                &ids,
                joint,
                saved_bytes(cfg, batch, seq, n_split),
            ));
        } else {
            n_split += 1;
            let saved = saved_if_split;
            let attn_peak = peak(saved, attention_values(cfg, batch, seq));
            let expert_peak = peak(saved, expert_values(cfg, batch, seq));
            let router_peak = peak(saved, router_values(cfg, batch, seq));
            let need = attn_peak.max(expert_peak).max(router_peak);
            let attn_kind = StageKind::Attention(li);
            let expert_kind = StageKind::Expert {
                layer: li,
                expert: 0,
            };
            let router_kind = StageKind::Router(li);
            let need_code = code_bytes(&attn_kind)
                .max(code_bytes(&expert_kind))
                .max(code_bytes(&router_kind));
            if !some_device_fits(cluster, attn_peak, &attn_kind)
                || !some_device_fits(cluster, expert_peak, &expert_kind)
                || !some_device_fits(cluster, router_peak, &router_kind)
            {
                return Err(format!(
                    "no device can hold a Mixtral group (need {need} buffer bytes and {need_code} code bytes on one device)"
                ));
            }
            stages.push(make_stage(
                &spec,
                format!("layers.{li}.attn"),
                attn_kind,
                &[layer.attn_norm, layer.wq, layer.wk, layer.wv, layer.wo],
                attention_values(cfg, batch, seq),
                saved,
            ));
            stages.push(make_stage(
                &spec,
                format!("layers.{li}.router"),
                router_kind,
                &[layer.ffn_norm, layer.gate],
                router_values(cfg, batch, seq),
                saved,
            ));
            for (e, ws) in layer.experts.iter().enumerate() {
                stages.push(make_stage(
                    &spec,
                    format!("layers.{li}.experts.{e}"),
                    StageKind::Expert {
                        layer: li,
                        expert: e,
                    },
                    ws,
                    expert_values(cfg, batch, seq),
                    saved,
                ));
            }
        }
    }
    let saved = saved_bytes(cfg, batch, seq, n_split);
    let embed_peak = peak(saved, embed_values(cfg, batch, seq));
    let head_peak = peak(saved, head_values(cfg, batch, seq));
    if !some_device_fits(cluster, embed_peak, &StageKind::Embed)
        || !some_device_fits(cluster, head_peak, &StageKind::Head)
    {
        return Err("no device can hold the embedding or the output head".into());
    }
    stages.insert(
        0,
        make_stage(
            &spec,
            "embed".into(),
            StageKind::Embed,
            &[spec.embed],
            embed_values(cfg, batch, seq),
            saved,
        ),
    );
    stages.push(make_stage(
        &spec,
        "head".into(),
        StageKind::Head,
        &[spec.norm, spec.head],
        head_values(cfg, batch, seq),
        saved,
    ));
    // Joint stages were numbered with a helper that may be wrong once embed is inserted.
    // Rename joint stages from their kind.
    let mut seen = std::collections::BTreeSet::new();
    for stage in &stages {
        for name in &stage.params {
            if !seen.insert(name.clone()) {
                return Err(format!("parameter `{name}` is updated in two stages"));
            }
        }
    }
    if seen.len() != spec.slots.len() {
        return Err(format!(
            "schedule covers {} of {} parameters",
            seen.len(),
            spec.slots.len()
        ));
    }
    place(&mut stages, cluster)?;
    let steps = coordinate(cfg, batch, seq, &stages, cluster)?;
    let messages = steps
        .iter()
        .filter_map(|step| match step {
            Step::Send(message) => Some(message.clone()),
            _ => None,
        })
        .collect();
    let peak_bytes = stages.iter().map(|s| s.buffer_bytes).max().unwrap_or(0);
    let peak_code_bytes = stages.iter().map(|s| s.code_bytes).max().unwrap_or(0);
    Ok(Schedule {
        budget: cluster
            .devices
            .iter()
            .map(|d| d.buffer_bytes)
            .max()
            .unwrap_or(0),
        code_budget: cluster
            .devices
            .iter()
            .map(|d| d.code_bytes)
            .max()
            .unwrap_or(0),
        checkpoint_bytes: saved,
        stages,
        peak_bytes,
        peak_code_bytes,
        messages,
        steps,
    })
}

fn hidden_bytes(cfg: &Config, batch: usize, seq: usize) -> u64 {
    (batch * seq * cfg.dim) as u64 * 4
}

fn column_bytes(batch: usize, seq: usize) -> u64 {
    (batch * seq) as u64 * 4
}

fn is_split(stages: &[Stage], layer: usize) -> bool {
    stages
        .iter()
        .any(|stage| matches!(stage.kind, StageKind::Attention(i) if i == layer))
}

fn h_parts(stages: &[Stage], layer: usize, n_experts: usize) -> Vec<String> {
    if layer == 0 {
        return vec!["h.0".into()];
    }
    let prev = layer - 1;
    if !is_split(stages, prev) {
        return vec![format!("h.{layer}")];
    }
    let mut parts = vec![format!("mid.{prev}")];
    for expert in 0..n_experts {
        parts.push(format!("partial.{prev}.{expert}"));
    }
    parts
}

fn tensor_bytes(name: &str, hidden: u64, column: u64) -> u64 {
    if name.contains("wcol.") {
        column
    } else {
        hidden
    }
}

fn forward_inputs(
    stage: &Stage,
    stages: &[Stage],
    n_layers: usize,
    n_experts: usize,
) -> Vec<String> {
    match stage.kind {
        StageKind::Embed => Vec::new(),
        StageKind::Attention(i) | StageKind::Layer(i) => h_parts(stages, i, n_experts),
        StageKind::Router(i) => vec![format!("mid.{i}")],
        StageKind::Expert { layer, expert } => {
            vec![format!("flat.{layer}"), format!("wcol.{layer}.{expert}")]
        }
        StageKind::Head => h_parts(stages, n_layers, n_experts),
    }
}

fn forward_outputs(stage: &Stage, n_experts: usize) -> Vec<String> {
    match stage.kind {
        StageKind::Embed => vec!["h.0".into()],
        StageKind::Attention(i) => vec![format!("mid.{i}")],
        StageKind::Router(i) => {
            let mut out = vec![format!("flat.{i}")];
            for expert in 0..n_experts {
                out.push(format!("wcol.{i}.{expert}"));
            }
            out
        }
        StageKind::Expert { layer, expert } => vec![format!("partial.{layer}.{expert}")],
        StageKind::Layer(i) => vec![format!("h.{}", i + 1)],
        StageKind::Head => Vec::new(),
    }
}

fn adjoint_consumes(stage: &Stage, n_experts: usize) -> Vec<String> {
    match stage.kind {
        StageKind::Embed => vec!["d.h.0".into()],
        StageKind::Attention(i) => vec![format!("d.mid.{i}"), format!("d.mid.router.{i}")],
        StageKind::Router(i) => {
            let mut out = Vec::new();
            for expert in 0..n_experts {
                out.push(format!("d.flat.{i}.{expert}"));
                out.push(format!("d.wcol.{i}.{expert}"));
            }
            out
        }
        StageKind::Expert { layer, expert } => vec![format!("d.partial.{layer}.{expert}")],
        StageKind::Layer(i) => vec![format!("d.h.{}", i + 1)],
        StageKind::Head => Vec::new(),
    }
}

fn adjoint_produces(
    stage: &Stage,
    stages: &[Stage],
    n_layers: usize,
    n_experts: usize,
) -> Vec<String> {
    match stage.kind {
        StageKind::Embed => Vec::new(),
        StageKind::Head => h_parts(stages, n_layers, n_experts)
            .into_iter()
            .map(|name| format!("d.{name}"))
            .collect(),
        StageKind::Attention(i) | StageKind::Layer(i) => h_parts(stages, i, n_experts)
            .into_iter()
            .map(|name| format!("d.{name}"))
            .collect(),
        StageKind::Router(i) => vec![format!("d.mid.router.{i}")],
        StageKind::Expert { layer, expert } => vec![
            format!("d.flat.{layer}.{expert}"),
            format!("d.wcol.{layer}.{expert}"),
        ],
    }
}

fn route(cluster: &Cluster, src: &str, dst: &str) -> Result<Vec<String>, String> {
    if src == dst {
        return Ok(vec![src.to_string()]);
    }
    let mut adjacent: std::collections::HashMap<&str, Vec<&str>> = std::collections::HashMap::new();
    for link in &cluster.links {
        adjacent
            .entry(link.a.as_str())
            .or_default()
            .push(link.b.as_str());
        adjacent
            .entry(link.b.as_str())
            .or_default()
            .push(link.a.as_str());
    }
    let mut queue = std::collections::VecDeque::new();
    let mut previous: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    queue.push_back(src.to_string());
    previous.insert(src.to_string(), String::new());
    while let Some(current) = queue.pop_front() {
        if current == dst {
            break;
        }
        for next in adjacent.get(current.as_str()).into_iter().flatten() {
            if !previous.contains_key(*next) {
                previous.insert((*next).to_string(), current.clone());
                queue.push_back((*next).to_string());
            }
        }
    }
    if !previous.contains_key(dst) {
        return Err(format!("no route from {src} to {dst}"));
    }
    let mut path = vec![dst.to_string()];
    while path.last().unwrap() != src {
        path.push(previous.get(path.last().unwrap()).unwrap().clone());
    }
    path.reverse();
    Ok(path)
}

fn send(
    steps: &mut Vec<Step>,
    seen: &mut std::collections::BTreeSet<(String, String, String)>,
    cluster: &Cluster,
    phase: Phase,
    src: &str,
    dst: &str,
    tensor: &str,
    bytes: u64,
) -> Result<(), String> {
    if src == dst {
        return Ok(());
    }
    let key = (
        format!("{phase:?}"),
        format!("{src}>{dst}"),
        tensor.to_string(),
    );
    if !seen.insert(key) {
        return Ok(());
    }
    let path = route(cluster, src, dst)?;
    steps.push(Step::Send(Message {
        phase,
        src: src.to_string(),
        dst: dst.to_string(),
        tensor: tensor.to_string(),
        bytes,
        path,
    }));
    Ok(())
}

fn coordinate(
    cfg: &Config,
    batch: usize,
    seq: usize,
    stages: &[Stage],
    cluster: &Cluster,
) -> Result<Vec<Step>, String> {
    let hidden = hidden_bytes(cfg, batch, seq);
    let column = column_bytes(batch, seq);
    let n_experts = cfg.n_experts;
    let n_layers = cfg.n_layers;
    let mut produced: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut steps = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for stage in stages {
        for tensor in forward_inputs(stage, stages, n_layers, n_experts) {
            let src = produced.get(&tensor).cloned().ok_or_else(|| {
                format!(
                    "forward stage {} needs {tensor} before it is produced",
                    stage.name
                )
            })?;
            send(
                &mut steps,
                &mut seen,
                cluster,
                Phase::Forward,
                &src,
                &stage.device,
                &tensor,
                tensor_bytes(&tensor, hidden, column),
            )?;
        }
        steps.push(Step::Run {
            stage: stage.name.clone(),
            device: stage.device.clone(),
            phase: Phase::Forward,
        });
        for tensor in forward_outputs(stage, n_experts) {
            produced.insert(tensor, stage.device.clone());
        }
    }
    let mut adjoint_at: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for stage in stages.iter().rev() {
        for tensor in adjoint_consumes(stage, n_experts) {
            let src = adjoint_at.get(&tensor).cloned().ok_or_else(|| {
                format!(
                    "adjoint stage {} needs {tensor} before it is produced",
                    stage.name
                )
            })?;
            send(
                &mut steps,
                &mut seen,
                cluster,
                Phase::Adjoint,
                &src,
                &stage.device,
                &tensor,
                tensor_bytes(&tensor, hidden, column),
            )?;
        }
        steps.push(Step::Run {
            stage: stage.name.clone(),
            device: stage.device.clone(),
            phase: Phase::Adjoint,
        });
        steps.push(Step::Update {
            stage: stage.name.clone(),
            device: stage.device.clone(),
            params: stage.params.clone(),
        });
        for tensor in adjoint_produces(stage, stages, n_layers, n_experts) {
            adjoint_at.insert(tensor, stage.device.clone());
        }
    }
    Ok(steps)
}

fn code_on(kind: &StageKind, device: DeviceKind) -> u64 {
    kernel_names(kind)
        .iter()
        .map(|name| match device {
            DeviceKind::Gpu => crate::cuda_emit::kernel_code_bytes(name),
            DeviceKind::Cpu => crate::cuda_emit::kernel_cpu_code_bytes(name),
        })
        .sum()
}

fn device_fits(device: &Device, buffer: u64, kind: &StageKind) -> bool {
    buffer <= device.buffer_bytes && code_on(kind, device.kind) <= device.code_bytes
}

fn some_device_fits(cluster: &Cluster, buffer: u64, kind: &StageKind) -> bool {
    cluster
        .devices
        .iter()
        .any(|device| device_fits(device, buffer, kind))
}

fn place(stages: &mut [Stage], cluster: &Cluster) -> Result<(), String> {
    let mut expert_n = 0usize;
    for stage in stages {
        let fitting: Vec<&Device> = cluster
            .devices
            .iter()
            .filter(|device| device_fits(device, stage.buffer_bytes, &stage.kind))
            .collect();
        if fitting.is_empty() {
            return Err(format!(
                "no device can hold stage {} ({} buffer bytes)",
                stage.name, stage.buffer_bytes
            ));
        }
        let chosen = match &stage.kind {
            StageKind::Expert { .. } => {
                let gpus: Vec<&Device> = fitting
                    .iter()
                    .copied()
                    .filter(|device| device.kind == DeviceKind::Gpu)
                    .collect();
                let pool = if gpus.is_empty() { fitting } else { gpus };
                let device = pool[expert_n % pool.len()];
                expert_n += 1;
                device
            }
            StageKind::Embed | StageKind::Head => fitting
                .iter()
                .copied()
                .find(|device| device.kind == DeviceKind::Cpu)
                .unwrap_or(fitting[0]),
            _ => fitting
                .iter()
                .copied()
                .find(|device| device.kind == DeviceKind::Gpu)
                .unwrap_or(fitting[0]),
        };
        stage.device = chosen.name.clone();
        stage.code_bytes = code_on(&stage.kind, chosen.kind);
    }
    Ok(())
}

fn layer_index_from_spec(spec: &crate::mixtral::Arch, layer: &crate::mixtral::LayerSpec) -> usize {
    spec.layers
        .iter()
        .position(|l| l.attn_norm == layer.attn_norm)
        .unwrap()
}

fn load(
    g: &mut Graph,
    mem: &mut dyn WeightMem,
    slot: usize,
    shape: &[usize],
) -> Result<usize, String> {
    Ok(g.leaf(shape, mem.read(slot)?))
}

thread_local! {
    static DEFER_GRADS: std::cell::RefCell<Option<std::collections::HashMap<usize, Vec<f32>>>> =
        std::cell::RefCell::new(None);
}

fn update(mem: &mut dyn WeightMem, slot: usize, grad: &[f32], lr: f32) -> Result<(), String> {
    let deferred = DEFER_GRADS.with(|slot_map| slot_map.borrow().is_some());
    if deferred {
        DEFER_GRADS.with(|slot_map| {
            if let Some(map) = slot_map.borrow_mut().as_mut() {
                map.insert(slot, grad.to_vec());
            }
        });
        return Ok(());
    }
    let mut w = mem.read(slot)?;
    for (value, g) in w.iter_mut().zip(grad) {
        *value -= lr * *g;
    }
    mem.write(slot, &w)
}

pub(crate) fn begin_deferred_grads() {
    DEFER_GRADS.with(|slot_map| *slot_map.borrow_mut() = Some(std::collections::HashMap::new()));
}

pub(crate) fn take_deferred_grads() -> std::collections::HashMap<usize, Vec<f32>> {
    DEFER_GRADS.with(|slot_map| slot_map.borrow_mut().take().unwrap_or_default())
}

fn fit(g: &Graph, reserved: u64, budget: u64, name: &str) -> Result<(), String> {
    let resident = reserved + g.bytes();
    if resident > budget {
        Err(format!(
            "{name} keeps {resident} bytes resident, budget is {budget}"
        ))
    } else {
        Ok(())
    }
}

fn targets(batch: &Batch) -> Vec<u32> {
    let mut out = Vec::with_capacity(batch.batch * (batch.seq - 1));
    for b in 0..batch.batch {
        for t in 1..batch.seq {
            out.push(batch.tokens[b * batch.seq + t]);
        }
    }
    out
}

fn fractions(cfg: &Config, counts: &[Vec<u32>], ntok: usize) -> Vec<f32> {
    let rows = (counts.len() * ntok) as f32;
    let mut fraction = vec![0f32; cfg.n_experts];
    for layer in counts {
        for (e, n) in layer.iter().enumerate() {
            fraction[e] += *n as f32;
        }
    }
    for value in &mut fraction {
        *value /= rows;
    }
    fraction
}

pub(crate) fn aux_cotangent(cfg: &Config, counts: &[Vec<u32>], ntok: usize) -> Vec<f32> {
    let rows = counts.len() * ntok;
    let fraction = fractions(cfg, counts, ntok);
    let scale = cfg.aux_loss_coef * cfg.n_experts as f32 / rows as f32;
    let mut cot = vec![0f32; ntok * cfg.n_experts];
    for n in 0..ntok {
        for e in 0..cfg.n_experts {
            cot[n * cfg.n_experts + e] = scale * fraction[e];
        }
    }
    cot
}

pub(crate) fn aux_value(
    cfg: &Config,
    prob_sum: &[f32],
    rows: usize,
    counts: &[Vec<u32>],
    ntok: usize,
) -> f32 {
    let fraction = fractions(cfg, counts, ntok);
    let mut s = 0f32;
    for e in 0..cfg.n_experts {
        s += fraction[e] * (prob_sum[e] / rows as f32);
    }
    cfg.aux_loss_coef * cfg.n_experts as f32 * s
}

fn shape_of<'a>(spec: &'a crate::mixtral::Arch, slot: usize) -> &'a [usize] {
    &spec.slots[slot].shape
}

struct ForwardTape {
    inputs: Vec<Vec<f32>>,
    mids: Vec<Option<Vec<f32>>>,
    flats: Vec<Option<Vec<f32>>>,
    weights: Vec<Option<Vec<f32>>>,
    counts: Vec<Vec<u32>>,
    loss: f32,
}

fn seal_split(
    i: usize,
    mids: &[Option<Vec<f32>>],
    accum: &mut [Option<Vec<f32>>],
    hidden: &mut Option<Vec<f32>>,
    inputs: &mut Vec<Vec<f32>>,
) {
    if let Some(sum) = accum[i].take() {
        let mid = mids[i].clone().unwrap();
        let mut h2 = mid;
        for (a, b) in h2.iter_mut().zip(&sum) {
            *a += *b;
        }
        *hidden = Some(h2);
        inputs.push(hidden.clone().unwrap());
    }
}

fn run_forward(
    cfg: &Config,
    batch: &Batch,
    plan: &Schedule,
    mem: &mut dyn WeightMem,
) -> Result<ForwardTape, String> {
    let spec = arch(cfg);
    let ntok = batch.batch * batch.seq;
    let mut hidden: Option<Vec<f32>> = None;
    let mut inputs = Vec::new();
    let mut mids = vec![None; cfg.n_layers];
    let mut flats = vec![None; cfg.n_layers];
    let mut weights = vec![None; cfg.n_layers];
    let mut counts = vec![vec![0u32; cfg.n_experts]; cfg.n_layers];
    let mut accum = vec![None; cfg.n_layers];
    let mut prob_sum = vec![0f32; cfg.n_experts];
    let mut rows = 0usize;
    let mut open: Option<usize> = None;

    for stage in &plan.stages {
        let stage_layer = match stage.kind {
            StageKind::Layer(i) | StageKind::Attention(i) | StageKind::Router(i) => Some(i),
            StageKind::Expert { layer, .. } => Some(layer),
            StageKind::Embed | StageKind::Head => None,
        };
        if let Some(prev) = open {
            if stage_layer != Some(prev) {
                seal_split(prev, &mids, &mut accum, &mut hidden, &mut inputs);
                open = None;
            }
        }
        match stage.kind {
            StageKind::Embed => {
                let mut g = Graph::new();
                let table = load(&mut g, mem, spec.embed, shape_of(&spec, spec.embed))?;
                let y = g.embed(table, &batch.tokens, batch.batch, batch.seq);
                hidden = Some(g.value(y).to_vec());
                inputs.push(hidden.clone().unwrap());
            }
            StageKind::Layer(i) => {
                let h_in = hidden.clone().unwrap();
                let mut g = Graph::new();
                let x = g.leaf(&[batch.batch, batch.seq, cfg.dim], h_in);
                let layer = &spec.layers[i];
                let aw = AttnW {
                    norm: load(
                        &mut g,
                        mem,
                        layer.attn_norm,
                        shape_of(&spec, layer.attn_norm),
                    )?,
                    q: load(&mut g, mem, layer.wq, shape_of(&spec, layer.wq))?,
                    k: load(&mut g, mem, layer.wk, shape_of(&spec, layer.wk))?,
                    v: load(&mut g, mem, layer.wv, shape_of(&spec, layer.wv))?,
                    o: load(&mut g, mem, layer.wo, shape_of(&spec, layer.wo))?,
                };
                let mid = attention(cfg, &mut g, aw, x, batch.seq);
                let experts: Vec<[usize; 3]> = layer
                    .experts
                    .iter()
                    .map(|e| {
                        Ok([
                            load(&mut g, mem, e[0], shape_of(&spec, e[0]))?,
                            load(&mut g, mem, e[1], shape_of(&spec, e[1]))?,
                            load(&mut g, mem, e[2], shape_of(&spec, e[2]))?,
                        ])
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let norm = load(&mut g, mem, layer.ffn_norm, shape_of(&spec, layer.ffn_norm))?;
                let gate = load(&mut g, mem, layer.gate, shape_of(&spec, layer.gate))?;
                let (y, probs, count) = moe(cfg, &mut g, norm, gate, &experts, mid);
                let pv = g.value(probs).to_vec();
                for (k, v) in pv.iter().enumerate() {
                    prob_sum[k % cfg.n_experts] += *v;
                }
                rows += ntok;
                counts[i] = count;
                hidden = Some(g.value(y).to_vec());
                inputs.push(hidden.clone().unwrap());
            }
            StageKind::Attention(i) => {
                let h_in = hidden.clone().unwrap();
                let mut g = Graph::new();
                let x = g.leaf(&[batch.batch, batch.seq, cfg.dim], h_in);
                let layer = &spec.layers[i];
                let aw = AttnW {
                    norm: load(
                        &mut g,
                        mem,
                        layer.attn_norm,
                        shape_of(&spec, layer.attn_norm),
                    )?,
                    q: load(&mut g, mem, layer.wq, shape_of(&spec, layer.wq))?,
                    k: load(&mut g, mem, layer.wk, shape_of(&spec, layer.wk))?,
                    v: load(&mut g, mem, layer.wv, shape_of(&spec, layer.wv))?,
                    o: load(&mut g, mem, layer.wo, shape_of(&spec, layer.wo))?,
                };
                let y = attention(cfg, &mut g, aw, x, batch.seq);
                mids[i] = Some(g.value(y).to_vec());
                hidden = mids[i].clone();
                open = Some(i);
            }
            StageKind::Router(i) => {
                let h_in = hidden.clone().unwrap();
                let mut g = Graph::new();
                let x = g.leaf(&[batch.batch, batch.seq, cfg.dim], h_in);
                let layer = &spec.layers[i];
                let norm = load(&mut g, mem, layer.ffn_norm, shape_of(&spec, layer.ffn_norm))?;
                let gate = load(&mut g, mem, layer.gate, shape_of(&spec, layer.gate))?;
                let route = router(cfg, &mut g, norm, gate, x);
                let pv = g.value(route.probs).to_vec();
                for (k, v) in pv.iter().enumerate() {
                    prob_sum[k % cfg.n_experts] += *v;
                }
                rows += ntok;
                flats[i] = Some(g.value(route.flat).to_vec());
                weights[i] = Some(g.value(route.weights).to_vec());
                counts[i] = route.counts;
                accum[i] = Some(vec![0f32; ntok * cfg.dim]);
            }
            StageKind::Expert { layer, expert } => {
                if counts[layer][expert] == 0 {
                    continue;
                }
                let flat_v = flats[layer].clone().unwrap();
                let w = weights[layer].clone().unwrap();
                let mut column = Vec::with_capacity(ntok);
                for n in 0..ntok {
                    column.push(w[n * cfg.n_experts + expert]);
                }
                let mut g = Graph::new();
                let flat = g.leaf(&[ntok, cfg.dim], flat_v);
                let ws = spec.layers[layer].experts[expert];
                let w1 = load(&mut g, mem, ws[0], shape_of(&spec, ws[0]))?;
                let w3 = load(&mut g, mem, ws[1], shape_of(&spec, ws[1]))?;
                let w2 = load(&mut g, mem, ws[2], shape_of(&spec, ws[2]))?;
                let col = g.leaf(&[ntok, 1], column);
                let hidden_e = swiglu(&mut g, flat, w1, w3, w2);
                let scaled = g.mul(hidden_e, col);
                let part = g.value(scaled).to_vec();
                let acc = accum[layer].as_mut().unwrap();
                for (a, b) in acc.iter_mut().zip(&part) {
                    *a += *b;
                }
            }
            StageKind::Head => {}
        }
    }
    if let Some(prev) = open {
        seal_split(prev, &mids, &mut accum, &mut hidden, &mut inputs);
    }
    let h = hidden.unwrap();
    if inputs.len() != cfg.n_layers + 1 {
        return Err(format!(
            "stored {} activations, expected {}",
            inputs.len(),
            cfg.n_layers + 1
        ));
    }
    let mut g = Graph::new();
    let x = g.leaf(&[batch.batch, batch.seq, cfg.dim], h);
    let norm = load(&mut g, mem, spec.norm, shape_of(&spec, spec.norm))?;
    let head = load(&mut g, mem, spec.head, shape_of(&spec, spec.head))?;
    let hidden_n = g.rmsnorm(x, norm, cfg.rms_norm_eps);
    let logits = g.matmul(hidden_n, head);
    let shifted = g.slice(logits, 1, 0, batch.seq - 1);
    let pred = g.reshape(shifted, &[batch.batch * (batch.seq - 1), cfg.vocab]);
    let ce = g.cross_entropy(pred, &targets(batch));
    let ce_v = g.scalar(ce);
    let loss = ce_v + aux_value(cfg, &prob_sum, rows, &counts, ntok);
    Ok(ForwardTape {
        inputs,
        mids,
        flats,
        weights,
        counts,
        loss,
    })
}

fn run_backward(
    cfg: &Config,
    batch: &Batch,
    plan: &Schedule,
    tape: &ForwardTape,
    mem: &mut dyn WeightMem,
    lr: f32,
) -> Result<(), String> {
    let spec = arch(cfg);
    let ntok = batch.batch * batch.seq;
    let reserved = plan.checkpoint_bytes;
    let aux = aux_cotangent(cfg, &tape.counts, ntok);
    let mut dh = head_backward(
        cfg,
        batch,
        &spec,
        mem,
        &tape.inputs[cfg.n_layers],
        lr,
        reserved,
        plan.budget,
    )?;
    let mut d_flat: Option<Vec<f32>> = None;
    let mut d_weights: Option<Vec<f32>> = None;
    let mut layer_dy: Option<Vec<f32>> = None;
    for stage in plan.stages.iter().rev() {
        match stage.kind {
            StageKind::Head => {}
            StageKind::Layer(i) => {
                dh = joint_backward(
                    cfg,
                    batch,
                    &spec,
                    i,
                    mem,
                    &tape.inputs[i],
                    &dh,
                    &aux,
                    lr,
                    reserved,
                    plan.budget,
                )?;
            }
            StageKind::Expert { layer, expert } => {
                if layer_dy.is_none() {
                    layer_dy = Some(dh.clone());
                    d_flat = Some(vec![0f32; ntok * cfg.dim]);
                    d_weights = Some(vec![0f32; ntok * cfg.n_experts]);
                }
                if tape.counts[layer][expert] == 0 {
                    continue;
                }
                let (df, dw) = expert_backward(
                    cfg,
                    batch,
                    &spec,
                    layer,
                    expert,
                    mem,
                    tape.flats[layer].as_ref().unwrap(),
                    tape.weights[layer].as_ref().unwrap(),
                    layer_dy.as_ref().unwrap(),
                    lr,
                    reserved,
                    plan.budget,
                )?;
                for (a, b) in d_flat.as_mut().unwrap().iter_mut().zip(&df) {
                    *a += *b;
                }
                for n in 0..ntok {
                    d_weights.as_mut().unwrap()[n * cfg.n_experts + expert] = dw[n];
                }
            }
            StageKind::Router(i) => {
                let dy = layer_dy.take().unwrap_or_else(|| dh.clone());
                let d_mid = router_backward(
                    cfg,
                    batch,
                    &spec,
                    i,
                    mem,
                    tape.mids[i].as_ref().unwrap(),
                    d_flat.take().unwrap(),
                    d_weights.take().unwrap(),
                    &aux,
                    &dy,
                    lr,
                    reserved,
                    plan.budget,
                )?;
                dh = d_mid;
            }
            StageKind::Attention(i) => {
                dh = attn_backward(
                    cfg,
                    batch,
                    &spec,
                    i,
                    mem,
                    &tape.inputs[i],
                    &dh,
                    lr,
                    reserved,
                    plan.budget,
                )?;
            }
            StageKind::Embed => {
                embed_backward(cfg, batch, &spec, mem, &dh, lr, reserved, plan.budget)?;
            }
        }
    }
    Ok(())
}

pub(crate) fn head_backward(
    cfg: &Config,
    batch: &Batch,
    spec: &crate::mixtral::Arch,
    mem: &mut dyn WeightMem,
    h: &[f32],
    lr: f32,
    reserved: u64,
    budget: u64,
) -> Result<Vec<f32>, String> {
    let mut g = Graph::new();
    let x = g.leaf(&[batch.batch, batch.seq, cfg.dim], h.to_vec());
    let norm = load(&mut g, mem, spec.norm, shape_of(spec, spec.norm))?;
    let head = load(&mut g, mem, spec.head, shape_of(spec, spec.head))?;
    let hidden = g.rmsnorm(x, norm, cfg.rms_norm_eps);
    let logits = g.matmul(hidden, head);
    let shifted = g.slice(logits, 1, 0, batch.seq - 1);
    let pred = g.reshape(shifted, &[batch.batch * (batch.seq - 1), cfg.vocab]);
    let ce = g.cross_entropy(pred, &targets(batch));
    g.backward(ce);
    fit(&g, reserved, budget, "head")?;
    let dh = g.grad(x).to_vec();
    update(mem, spec.norm, g.grad(norm), lr)?;
    update(mem, spec.head, g.grad(head), lr)?;
    Ok(dh)
}

pub(crate) fn joint_backward(
    cfg: &Config,
    batch: &Batch,
    spec: &crate::mixtral::Arch,
    layer_i: usize,
    mem: &mut dyn WeightMem,
    h_in: &[f32],
    dy: &[f32],
    aux: &[f32],
    lr: f32,
    reserved: u64,
    budget: u64,
) -> Result<Vec<f32>, String> {
    let layer = &spec.layers[layer_i];
    let mut g = Graph::new();
    let x = g.leaf(&[batch.batch, batch.seq, cfg.dim], h_in.to_vec());
    let aw = AttnW {
        norm: load(
            &mut g,
            mem,
            layer.attn_norm,
            shape_of(spec, layer.attn_norm),
        )?,
        q: load(&mut g, mem, layer.wq, shape_of(spec, layer.wq))?,
        k: load(&mut g, mem, layer.wk, shape_of(spec, layer.wk))?,
        v: load(&mut g, mem, layer.wv, shape_of(spec, layer.wv))?,
        o: load(&mut g, mem, layer.wo, shape_of(spec, layer.wo))?,
    };
    let ids = [aw.norm, aw.q, aw.k, aw.v, aw.o];
    let slots = [layer.attn_norm, layer.wq, layer.wk, layer.wv, layer.wo];
    let mid = attention(cfg, &mut g, aw, x, batch.seq);
    let mut expert_nodes = Vec::new();
    let mut expert_slots = Vec::new();
    for e in &layer.experts {
        let nodes = [
            load(&mut g, mem, e[0], shape_of(spec, e[0]))?,
            load(&mut g, mem, e[1], shape_of(spec, e[1]))?,
            load(&mut g, mem, e[2], shape_of(spec, e[2]))?,
        ];
        expert_nodes.push(nodes);
        expert_slots.push(*e);
    }
    let norm = load(&mut g, mem, layer.ffn_norm, shape_of(spec, layer.ffn_norm))?;
    let gate = load(&mut g, mem, layer.gate, shape_of(spec, layer.gate))?;
    let (y, probs, _) = moe(cfg, &mut g, norm, gate, &expert_nodes, mid);
    let y1 = g.reshape(y, &[dy.len()]);
    let p1 = g.reshape(probs, &[aux.len()]);
    let s1 = g.dot_const(y1, dy);
    let s2 = g.dot_const(p1, aux);
    let loss = g.add(s1, s2);
    g.backward(loss);
    fit(&g, reserved, budget, &format!("layers.{layer_i}"))?;
    let dx = g.grad(x).to_vec();
    for (slot, node) in slots.iter().zip(ids) {
        update(mem, *slot, g.grad(node), lr)?;
    }
    update(mem, layer.ffn_norm, g.grad(norm), lr)?;
    update(mem, layer.gate, g.grad(gate), lr)?;
    for (slots_e, nodes) in expert_slots.iter().zip(&expert_nodes) {
        for k in 0..3 {
            update(mem, slots_e[k], g.grad(nodes[k]), lr)?;
        }
    }
    Ok(dx)
}

pub(crate) fn attn_backward(
    cfg: &Config,
    batch: &Batch,
    spec: &crate::mixtral::Arch,
    layer_i: usize,
    mem: &mut dyn WeightMem,
    h_in: &[f32],
    dy: &[f32],
    lr: f32,
    reserved: u64,
    budget: u64,
) -> Result<Vec<f32>, String> {
    let layer = &spec.layers[layer_i];
    let mut g = Graph::new();
    let x = g.leaf(&[batch.batch, batch.seq, cfg.dim], h_in.to_vec());
    let aw = AttnW {
        norm: load(
            &mut g,
            mem,
            layer.attn_norm,
            shape_of(spec, layer.attn_norm),
        )?,
        q: load(&mut g, mem, layer.wq, shape_of(spec, layer.wq))?,
        k: load(&mut g, mem, layer.wk, shape_of(spec, layer.wk))?,
        v: load(&mut g, mem, layer.wv, shape_of(spec, layer.wv))?,
        o: load(&mut g, mem, layer.wo, shape_of(spec, layer.wo))?,
    };
    let nodes = [aw.norm, aw.q, aw.k, aw.v, aw.o];
    let slots = [layer.attn_norm, layer.wq, layer.wk, layer.wv, layer.wo];
    let y = attention(cfg, &mut g, aw, x, batch.seq);
    g.backward_cotangent(y, dy);
    fit(&g, reserved, budget, &format!("layers.{layer_i}.attn"))?;
    let dx = g.grad(x).to_vec();
    for (slot, node) in slots.iter().zip(nodes) {
        update(mem, *slot, g.grad(node), lr)?;
    }
    Ok(dx)
}

pub(crate) fn expert_backward(
    cfg: &Config,
    batch: &Batch,
    spec: &crate::mixtral::Arch,
    layer_i: usize,
    expert: usize,
    mem: &mut dyn WeightMem,
    flat_v: &[f32],
    weights_v: &[f32],
    dy: &[f32],
    lr: f32,
    reserved: u64,
    budget: u64,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let ntok = batch.batch * batch.seq;
    let mut column = Vec::with_capacity(ntok);
    for n in 0..ntok {
        column.push(weights_v[n * cfg.n_experts + expert]);
    }
    let mut g = Graph::new();
    let flat = g.leaf(&[ntok, cfg.dim], flat_v.to_vec());
    let ws = spec.layers[layer_i].experts[expert];
    let w1 = load(&mut g, mem, ws[0], shape_of(spec, ws[0]))?;
    let w3 = load(&mut g, mem, ws[1], shape_of(spec, ws[1]))?;
    let w2 = load(&mut g, mem, ws[2], shape_of(spec, ws[2]))?;
    let col = g.leaf(&[ntok, 1], column);
    let hidden = swiglu(&mut g, flat, w1, w3, w2);
    let scaled = g.mul(hidden, col);
    g.backward_cotangent(scaled, dy);
    fit(
        &g,
        reserved,
        budget,
        &format!("layers.{layer_i}.experts.{expert}"),
    )?;
    update(mem, ws[0], g.grad(w1), lr)?;
    update(mem, ws[1], g.grad(w3), lr)?;
    update(mem, ws[2], g.grad(w2), lr)?;
    Ok((g.grad(flat).to_vec(), g.grad(col).to_vec()))
}

pub(crate) fn router_backward(
    cfg: &Config,
    batch: &Batch,
    spec: &crate::mixtral::Arch,
    layer_i: usize,
    mem: &mut dyn WeightMem,
    h_mid: &[f32],
    d_flat: Vec<f32>,
    d_weights: Vec<f32>,
    aux: &[f32],
    dy: &[f32],
    lr: f32,
    reserved: u64,
    budget: u64,
) -> Result<Vec<f32>, String> {
    let layer = &spec.layers[layer_i];
    let mut g = Graph::new();
    let x = g.leaf(&[batch.batch, batch.seq, cfg.dim], h_mid.to_vec());
    let norm = load(&mut g, mem, layer.ffn_norm, shape_of(spec, layer.ffn_norm))?;
    let gate = load(&mut g, mem, layer.gate, shape_of(spec, layer.gate))?;
    let route = router(cfg, &mut g, norm, gate, x);
    let flat_1 = g.reshape(route.flat, &[d_flat.len()]);
    let w_1 = g.reshape(route.weights, &[d_weights.len()]);
    let p_1 = g.reshape(route.probs, &[aux.len()]);
    let s1 = g.dot_const(flat_1, &d_flat);
    let s2 = g.dot_const(w_1, &d_weights);
    let s3 = g.dot_const(p_1, aux);
    let s12 = g.add(s1, s2);
    let loss = g.add(s12, s3);
    g.backward(loss);
    fit(&g, reserved, budget, &format!("layers.{layer_i}.router"))?;
    let mut dh = g.grad(x).to_vec();
    for (a, b) in dh.iter_mut().zip(dy) {
        *a += *b;
    }
    update(mem, layer.ffn_norm, g.grad(norm), lr)?;
    update(mem, layer.gate, g.grad(gate), lr)?;
    Ok(dh)
}

pub(crate) fn embed_backward(
    _cfg: &Config,
    batch: &Batch,
    spec: &crate::mixtral::Arch,
    mem: &mut dyn WeightMem,
    dh: &[f32],
    lr: f32,
    reserved: u64,
    budget: u64,
) -> Result<(), String> {
    let mut g = Graph::new();
    let table = load(&mut g, mem, spec.embed, shape_of(spec, spec.embed))?;
    let y = g.embed(table, &batch.tokens, batch.batch, batch.seq);
    g.backward_cotangent(y, dh);
    fit(&g, reserved, budget, "embed")?;
    update(mem, spec.embed, g.grad(table), lr)
}

pub fn staged_update(
    cfg: &Config,
    batch: &Batch,
    memory: Memory,
    lr: f32,
    mem: &mut dyn WeightMem,
) -> Result<f32, String> {
    let plan = schedule(cfg, batch.batch, batch.seq, memory)?;
    let tape = run_forward(cfg, batch, &plan, mem)?;
    let loss = tape.loss;
    run_backward(cfg, batch, &plan, &tape, mem, lr)?;
    Ok(loss)
}

pub fn save_random(dir: &Path, cfg: &Config, seed: u64) -> Result<(), String> {
    fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let mut error: Option<String> = None;
    each_init(cfg, seed, |slot, values| {
        if error.is_some() {
            return;
        }
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for x in values {
            bytes.extend(x.to_le_bytes());
        }
        if let Err(e) = fs::write(dir.join(format!("{slot}.f32")), bytes) {
            error = Some(e.to_string());
        }
    });
    match error {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

pub fn read_checkpoint(dir: &Path, cfg: &Config) -> Result<Vec<f32>, String> {
    let spec = arch(cfg);
    let mut mem = DiskWeights {
        dir: dir.to_path_buf(),
    };
    let mut out = Vec::new();
    for slot in 0..spec.slots.len() {
        out.extend(mem.read(slot)?);
    }
    Ok(out)
}

pub fn staged_update_dir(
    dir: &Path,
    cfg: &Config,
    batch: &Batch,
    memory: Memory,
    lr: f32,
) -> Result<f32, String> {
    let mut mem = DiskWeights {
        dir: dir.to_path_buf(),
    };
    staged_update(cfg, batch, memory, lr, &mut mem)
}

/// One staged step from a randomly initialized checkpoint must match the
/// single-buffer adjoint at `lr`.
pub fn verify_staged_checkpoint(dir: &Path) -> Result<(), String> {
    let cfg = Config::demo_smooth();
    let batch = crate::mixtral::sample_batch(&cfg, 2, 6, 7);
    let seed = 0x4D49_5854;
    save_random(dir, &cfg, seed)?;
    let base = read_checkpoint(dir, &cfg)?;
    let budgets = [
        minimum_budget(&cfg, batch.batch, batch.seq)?,
        Memory {
            code_bytes: u64::MAX / 8,
            buffer_bytes: u64::MAX / 8,
        },
    ];
    for budget in budgets {
        write_flat(dir, &cfg, &base)?;
        let mut mono = base.clone();
        let ev = crate::mixtral::evaluate(&cfg, &mono, &batch);
        sgd(&mut mono, &ev.grads, 0.01);
        let loss = staged_update_dir(dir, &cfg, &batch, budget, 0.01)?;
        if (loss - ev.loss).abs() > 1e-3 {
            return Err(format!("staged loss {loss} vs monolithic {}", ev.loss));
        }
        let got = read_checkpoint(dir, &cfg)?;
        let err = mono
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        if err > 1e-3 {
            return Err(format!(
                "staged parameters differ by {err} at buffer budget {}",
                budget.buffer_bytes
            ));
        }
    }
    Ok(())
}

fn write_flat(dir: &Path, cfg: &Config, params: &[f32]) -> Result<(), String> {
    let spec = arch(cfg);
    let mut offset = 0usize;
    for (slot, s) in spec.slots.iter().enumerate() {
        let n: usize = s.shape.iter().product();
        let mut mem = DiskWeights {
            dir: dir.to_path_buf(),
        };
        mem.write(slot, &params[offset..offset + n])?;
        offset += n;
    }
    Ok(())
}

pub(crate) struct StageOutput {
    pub tensors: std::collections::HashMap<String, Vec<f32>>,
    pub loss: Option<f32>,
    pub counts: Vec<u32>,
    pub prob_sum: Vec<f32>,
    pub grads: std::collections::HashMap<usize, Vec<f32>>,
}

fn assemble(
    names: &[String],
    inputs: &std::collections::HashMap<String, Vec<f32>>,
) -> Result<Vec<f32>, String> {
    let mut acc = inputs
        .get(&names[0])
        .cloned()
        .ok_or_else(|| format!("missing tensor {}", names[0]))?;
    for name in &names[1..] {
        let part = inputs
            .get(name)
            .ok_or_else(|| format!("missing tensor {name}"))?;
        if part.len() != acc.len() {
            return Err(format!("tensor {name} has the wrong length"));
        }
        for (slot, value) in acc.iter_mut().zip(part) {
            *slot += *value;
        }
    }
    Ok(acc)
}

pub(crate) fn run_stage(
    cfg: &Config,
    batch: &Batch,
    stage: &Stage,
    stages: &[Stage],
    phase: Phase,
    mem: &mut dyn WeightMem,
    inputs: &std::collections::HashMap<String, Vec<f32>>,
    aux: &[f32],
    reserved: u64,
    budget: u64,
) -> Result<StageOutput, String> {
    let spec = arch(cfg);
    let ntok = batch.batch * batch.seq;
    let mut tensors = std::collections::HashMap::new();
    let mut loss = None;
    let mut counts = Vec::new();
    let mut prob_sum = vec![0f32; cfg.n_experts];
    if phase == Phase::Adjoint {
        begin_deferred_grads();
    }
    match (&stage.kind, phase) {
        (StageKind::Embed, Phase::Forward) => {
            let mut g = Graph::new();
            let table = load(&mut g, mem, spec.embed, shape_of(&spec, spec.embed))?;
            let y = g.embed(table, &batch.tokens, batch.batch, batch.seq);
            tensors.insert("h.0".into(), g.value(y).to_vec());
        }
        (StageKind::Embed, Phase::Adjoint) => {
            let dh = inputs.get("d.h.0").ok_or("missing d.h.0")?;
            embed_backward(cfg, batch, &spec, mem, dh, 0.0, reserved, budget)?;
        }
        (StageKind::Attention(i), Phase::Forward) => {
            let h_in = assemble(&h_parts(stages, *i, cfg.n_experts), inputs)?;
            let mut g = Graph::new();
            let x = g.leaf(&[batch.batch, batch.seq, cfg.dim], h_in);
            let layer = &spec.layers[*i];
            let aw = AttnW {
                norm: load(
                    &mut g,
                    mem,
                    layer.attn_norm,
                    shape_of(&spec, layer.attn_norm),
                )?,
                q: load(&mut g, mem, layer.wq, shape_of(&spec, layer.wq))?,
                k: load(&mut g, mem, layer.wk, shape_of(&spec, layer.wk))?,
                v: load(&mut g, mem, layer.wv, shape_of(&spec, layer.wv))?,
                o: load(&mut g, mem, layer.wo, shape_of(&spec, layer.wo))?,
            };
            let y = attention(cfg, &mut g, aw, x, batch.seq);
            tensors.insert(format!("mid.{i}"), g.value(y).to_vec());
        }
        (StageKind::Attention(i), Phase::Adjoint) => {
            let h_in = assemble(&h_parts(stages, *i, cfg.n_experts), inputs)?;
            let mut dy = inputs
                .get(&format!("d.mid.{i}"))
                .cloned()
                .ok_or("missing residual cotangent")?;
            let extra = inputs
                .get(&format!("d.mid.router.{i}"))
                .ok_or("missing router cotangent")?;
            for (slot, value) in dy.iter_mut().zip(extra) {
                *slot += *value;
            }
            let dx = attn_backward(
                cfg, batch, &spec, *i, mem, &h_in, &dy, 0.0, reserved, budget,
            )?;
            for name in h_parts(stages, *i, cfg.n_experts) {
                tensors.insert(format!("d.{name}"), dx.clone());
            }
        }
        (StageKind::Router(i), Phase::Forward) => {
            let mid = inputs
                .get(&format!("mid.{i}"))
                .cloned()
                .ok_or("missing mid")?;
            let mut g = Graph::new();
            let x = g.leaf(&[batch.batch, batch.seq, cfg.dim], mid);
            let layer = &spec.layers[*i];
            let norm = load(&mut g, mem, layer.ffn_norm, shape_of(&spec, layer.ffn_norm))?;
            let gate = load(&mut g, mem, layer.gate, shape_of(&spec, layer.gate))?;
            let route = router(cfg, &mut g, norm, gate, x);
            let probs = g.value(route.probs).to_vec();
            for (k, value) in probs.iter().enumerate() {
                prob_sum[k % cfg.n_experts] += *value;
            }
            counts = route.counts;
            tensors.insert(format!("flat.{i}"), g.value(route.flat).to_vec());
            let weights = g.value(route.weights).to_vec();
            for expert in 0..cfg.n_experts {
                let mut column = Vec::with_capacity(ntok);
                for n in 0..ntok {
                    column.push(weights[n * cfg.n_experts + expert]);
                }
                tensors.insert(format!("wcol.{i}.{expert}"), column);
            }
        }
        (StageKind::Router(i), Phase::Adjoint) => {
            let mid = inputs
                .get(&format!("mid.{i}"))
                .cloned()
                .ok_or("missing mid")?;
            let mut d_flat = vec![0f32; ntok * cfg.dim];
            let mut d_weights = vec![0f32; ntok * cfg.n_experts];
            for expert in 0..cfg.n_experts {
                let df = inputs
                    .get(&format!("d.flat.{i}.{expert}"))
                    .ok_or("missing expert flat cotangent")?;
                for (slot, value) in d_flat.iter_mut().zip(df) {
                    *slot += *value;
                }
                let dw = inputs
                    .get(&format!("d.wcol.{i}.{expert}"))
                    .ok_or("missing expert weight cotangent")?;
                for n in 0..ntok {
                    d_weights[n * cfg.n_experts + expert] = dw[n];
                }
            }
            let zeros = vec![0f32; mid.len()];
            let d_mid = router_backward(
                cfg, batch, &spec, *i, mem, &mid, d_flat, d_weights, aux, &zeros, 0.0, reserved,
                budget,
            )?;
            tensors.insert(format!("d.mid.router.{i}"), d_mid);
        }
        (StageKind::Expert { layer, expert }, Phase::Forward) => {
            let flat = inputs
                .get(&format!("flat.{layer}"))
                .cloned()
                .ok_or("missing flat")?;
            let column = inputs
                .get(&format!("wcol.{layer}.{expert}"))
                .cloned()
                .ok_or("missing weight column")?;
            let mut g = Graph::new();
            let flat_n = g.leaf(&[ntok, cfg.dim], flat);
            let ws = spec.layers[*layer].experts[*expert];
            let w1 = load(&mut g, mem, ws[0], shape_of(&spec, ws[0]))?;
            let w3 = load(&mut g, mem, ws[1], shape_of(&spec, ws[1]))?;
            let w2 = load(&mut g, mem, ws[2], shape_of(&spec, ws[2]))?;
            let hidden = swiglu(&mut g, flat_n, w1, w3, w2);
            let col = g.leaf(&[ntok, 1], column);
            let scaled = g.mul(hidden, col);
            tensors.insert(
                format!("partial.{layer}.{expert}"),
                g.value(scaled).to_vec(),
            );
        }
        (StageKind::Expert { layer, expert }, Phase::Adjoint) => {
            let flat = inputs
                .get(&format!("flat.{layer}"))
                .cloned()
                .ok_or("missing flat")?;
            let mut weights = vec![0f32; ntok * cfg.n_experts];
            let column = inputs
                .get(&format!("wcol.{layer}.{expert}"))
                .ok_or("missing weight column")?;
            for n in 0..ntok {
                weights[n * cfg.n_experts + expert] = column[n];
            }
            let dy = inputs
                .get(&format!("d.partial.{layer}.{expert}"))
                .cloned()
                .ok_or("missing partial cotangent")?;
            let (df, dw) = expert_backward(
                cfg, batch, &spec, *layer, *expert, mem, &flat, &weights, &dy, 0.0, reserved,
                budget,
            )?;
            tensors.insert(format!("d.flat.{layer}.{expert}"), df);
            tensors.insert(format!("d.wcol.{layer}.{expert}"), dw);
        }
        (StageKind::Layer(i), Phase::Forward) => {
            let h_in = assemble(&h_parts(stages, *i, cfg.n_experts), inputs)?;
            let mut g = Graph::new();
            let x = g.leaf(&[batch.batch, batch.seq, cfg.dim], h_in);
            let layer = &spec.layers[*i];
            let aw = AttnW {
                norm: load(
                    &mut g,
                    mem,
                    layer.attn_norm,
                    shape_of(&spec, layer.attn_norm),
                )?,
                q: load(&mut g, mem, layer.wq, shape_of(&spec, layer.wq))?,
                k: load(&mut g, mem, layer.wk, shape_of(&spec, layer.wk))?,
                v: load(&mut g, mem, layer.wv, shape_of(&spec, layer.wv))?,
                o: load(&mut g, mem, layer.wo, shape_of(&spec, layer.wo))?,
            };
            let mid = attention(cfg, &mut g, aw, x, batch.seq);
            let experts: Vec<[usize; 3]> = layer
                .experts
                .iter()
                .map(|e| {
                    Ok([
                        load(&mut g, mem, e[0], shape_of(&spec, e[0]))?,
                        load(&mut g, mem, e[1], shape_of(&spec, e[1]))?,
                        load(&mut g, mem, e[2], shape_of(&spec, e[2]))?,
                    ])
                })
                .collect::<Result<_, String>>()?;
            let norm = load(&mut g, mem, layer.ffn_norm, shape_of(&spec, layer.ffn_norm))?;
            let gate = load(&mut g, mem, layer.gate, shape_of(&spec, layer.gate))?;
            let (y, probs, count) = moe(cfg, &mut g, norm, gate, &experts, mid);
            let pv = g.value(probs).to_vec();
            for (k, value) in pv.iter().enumerate() {
                prob_sum[k % cfg.n_experts] += *value;
            }
            counts = count;
            tensors.insert(format!("h.{}", i + 1), g.value(y).to_vec());
        }
        (StageKind::Layer(i), Phase::Adjoint) => {
            let h_in = assemble(&h_parts(stages, *i, cfg.n_experts), inputs)?;
            let dy = inputs
                .get(&format!("d.h.{}", i + 1))
                .cloned()
                .ok_or("missing layer cotangent")?;
            let dx = joint_backward(
                cfg, batch, &spec, *i, mem, &h_in, &dy, aux, 0.0, reserved, budget,
            )?;
            for name in h_parts(stages, *i, cfg.n_experts) {
                tensors.insert(format!("d.{name}"), dx.clone());
            }
        }
        (StageKind::Head, Phase::Forward) => {
            let h = assemble(&h_parts(stages, cfg.n_layers, cfg.n_experts), inputs)?;
            let mut g = Graph::new();
            let x = g.leaf(&[batch.batch, batch.seq, cfg.dim], h);
            let norm = load(&mut g, mem, spec.norm, shape_of(&spec, spec.norm))?;
            let head = load(&mut g, mem, spec.head, shape_of(&spec, spec.head))?;
            let hidden = g.rmsnorm(x, norm, cfg.rms_norm_eps);
            let logits = g.matmul(hidden, head);
            let shifted = g.slice(logits, 1, 0, batch.seq - 1);
            let pred = g.reshape(shifted, &[batch.batch * (batch.seq - 1), cfg.vocab]);
            let mut targets = Vec::new();
            for b in 0..batch.batch {
                for t in 1..batch.seq {
                    targets.push(batch.tokens[b * batch.seq + t]);
                }
            }
            let ce = g.cross_entropy(pred, &targets);
            loss = Some(g.scalar(ce));
        }
        (StageKind::Head, Phase::Adjoint) => {
            let h = assemble(&h_parts(stages, cfg.n_layers, cfg.n_experts), inputs)?;
            let dh = head_backward(cfg, batch, &spec, mem, &h, 0.0, reserved, budget)?;
            for name in h_parts(stages, cfg.n_layers, cfg.n_experts) {
                tensors.insert(format!("d.{name}"), dh.clone());
            }
        }
    }
    let grads = if phase == Phase::Adjoint {
        take_deferred_grads()
    } else {
        std::collections::HashMap::new()
    };
    Ok(StageOutput {
        tensors,
        loss,
        counts,
        prob_sum,
        grads,
    })
}
