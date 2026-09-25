use gtt::{
    check_source, compile_mixtral, minimum_budget, parameter_count, schedule, verify_staged_checkpoint,
    Config, Memory,
};
use std::fs;
use std::process::Command;

#[test]
fn mixtral_signature_typechecks() {
    let src = fs::read_to_string("examples/mixtral/signature.gtt").unwrap();
    check_source(&src).unwrap_or_else(|e| panic!("{}", e.render(&src)));
}

#[test]
fn mixtral_forward_adjoint_and_sgd() {
    let paper = parameter_count(&Config::mixtral_8x7b());
    assert_eq!(paper, 46_702_792_704, "published Mixtral-8x7B parameter count");
    let report = compile_mixtral();
    assert_eq!(report.paper_parameters, paper);
    assert_eq!(report.active_parameters, 12_879_925_248);
    assert!(
        report.max_abs_fd_error < 5e-3,
        "adjoint disagrees with finite differences by {}",
        report.max_abs_fd_error
    );
    assert!(report.loss_after < report.loss_before, "SGD did not descend");
    assert!(report.learning_rate > 0.0);
    assert!(report.sparse_loss.is_finite());
    assert!(
        report.unused_expert_grad < 1e-8,
        "idle expert received a gradient {}",
        report.unused_expert_grad
    );
    assert!(
        report.used_expert_grad > 1e-8,
        "routed expert gradient vanished"
    );
    assert!(report.full_model_peak <= report.full_model_budget);
    assert!(report.full_model_stages > 32);
}

#[test]
fn mixtral_stages_parameters_under_a_memory_budget() {
    let paper = Config::mixtral_8x7b();
    let fit = schedule(
        &paper,
        1,
        512,
        Memory {
            code_bytes: 1 << 20,
            buffer_bytes: 8 << 30,
        },
    )
    .unwrap();
    assert!(fit.peak_bytes <= fit.budget);
    assert!(fit.peak_code_bytes <= fit.code_budget);
    assert!(fit.stages.len() > paper.n_layers);
    let catalog: u64 = gtt::kernel_catalog().iter().map(|k| k.code_bytes).sum();
    assert!(fit.peak_code_bytes < catalog);
    for stage in &fit.stages {
        let code: u64 = stage
            .kernels
            .iter()
            .map(|name| gtt::kernel_code_bytes(name))
            .sum();
        assert_eq!(stage.code_bytes, code);
        let buffers: u64 = stage.buffers.iter().map(|b| b.bytes).sum();
        assert_eq!(stage.buffer_bytes, buffers);
        assert!(stage.kernels.iter().any(|k| k.starts_with("gemm") || *k == "embed" || *k == "ce_grad"));
        assert!(stage.buffers.iter().any(|b| b.role == gtt::BufferRole::Parameter));
        assert!(stage.buffers.iter().any(|b| b.role == gtt::BufferRole::Adjoint));
        assert!(stage.buffers.iter().any(|b| b.role == gtt::BufferRole::Checkpoint));
        assert!(stage.buffers.iter().any(|b| b.role == gtt::BufferRole::Scratch));
    }
    assert!(schedule(
        &paper,
        1,
        512,
        Memory {
            code_bytes: 1 << 20,
            buffer_bytes: 1 << 30,
        },
    )
    .is_err());
    assert!(schedule(
        &paper,
        1,
        512,
        Memory {
            code_bytes: 64,
            buffer_bytes: 8 << 30,
        },
    )
    .is_err());
    let demo = Config::demo_smooth();
    let min = minimum_budget(&demo, 2, 6).unwrap();
    let tight = schedule(&demo, 2, 6, min).unwrap();
    assert!(tight.peak_bytes <= min.buffer_bytes);
    assert!(tight.peak_code_bytes <= min.code_bytes);
    assert!(tight.stages.len() > demo.n_layers + 2);
    let dir = std::env::temp_dir().join(format!("mixtral-stage-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    verify_staged_checkpoint(&dir).unwrap_or_else(|e| panic!("{e}"));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn mixtral_places_stages_on_a_heterogeneous_cluster() {
    let cluster = gtt::mixtral_cluster();
    let plan = gtt::schedule_on(&Config::mixtral_8x7b(), 1, 512, &cluster).unwrap();
    let embed = plan.stages.iter().find(|s| s.name == "embed").unwrap();
    assert_eq!(embed.device, "cpu0");
    let head = plan.stages.iter().find(|s| s.name == "head").unwrap();
    assert_eq!(head.device, "cpu0");
    let mut expert_devices = std::collections::BTreeSet::new();
    for stage in &plan.stages {
        let device = cluster
            .devices
            .iter()
            .find(|d| d.name == stage.device)
            .unwrap();
        assert!(stage.buffer_bytes <= device.buffer_bytes);
        assert!(stage.code_bytes <= device.code_bytes);
        assert_ne!(stage.device, "gpu2");
        if stage.name.contains("experts") {
            expert_devices.insert(stage.device.clone());
            assert_eq!(device.kind, gtt::DeviceKind::Gpu);
        }
        if stage.name.ends_with(".attn") {
            assert_eq!(device.kind, gtt::DeviceKind::Gpu);
        }
    }
    assert!(expert_devices.len() >= 2);
    assert!(!plan.messages.is_empty());
    let mut saw_forward = false;
    let mut saw_adjoint = false;
    let mut saw_host_hop = false;
    for message in &plan.messages {
        assert!(message.path.first().unwrap() == &message.src);
        assert!(message.path.last().unwrap() == &message.dst);
        for pair in message.path.windows(2) {
            let linked = cluster.links.iter().any(|link| {
                (link.a == pair[0] && link.b == pair[1]) || (link.a == pair[1] && link.b == pair[0])
            });
            assert!(linked, "hop {} -> {} is not a link", pair[0], pair[1]);
        }
        assert!(!message.path.contains(&"gpu2".to_string()));
        if message.path.len() >= 3 && message.path.contains(&"cpu0".to_string()) {
            saw_host_hop = true;
        }
        match message.phase {
            gtt::Phase::Forward => saw_forward = true,
            gtt::Phase::Adjoint => saw_adjoint = true,
        }
    }
    assert!(saw_forward && saw_adjoint && saw_host_hop);
    let h0 = plan.messages.iter().find(|m| m.tensor == "h.0").unwrap();
    assert_eq!(h0.src, "cpu0");
    assert_eq!(h0.phase, gtt::Phase::Forward);
    for step in &plan.steps {
        if let gtt::Step::Update { stage, device, params } = step {
            let owner = plan.stages.iter().find(|s| s.name == *stage).unwrap();
            assert_eq!(owner.device, *device);
            assert_eq!(owner.params, *params);
        }
    }
}

#[test]
fn mixtral_cuda_kernels_match_the_adjoint() {
    let src = gtt::kernel_source();
    let path = "examples/mixtral/kernels.cu";
    fs::write(path, &src).unwrap();
    let nvcc = Command::new("/usr/local/cuda/bin/nvcc")
        .args([
            "-ccbin",
            "/usr/bin/g++-11",
            "-arch=sm_75",
            "-O2",
            path,
            "-o",
            "/tmp/mixtral_kernels",
        ])
        .output()
        .expect("nvcc");
    assert!(
        nvcc.status.success(),
        "nvcc\n{}{}",
        String::from_utf8_lossy(&nvcc.stdout),
        String::from_utf8_lossy(&nvcc.stderr)
    );
    let run = Command::new("/tmp/mixtral_kernels").output().expect("run");
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stdout)
    );
    assert!(String::from_utf8_lossy(&run.stdout).contains("cuda adjoint ok"));
}
