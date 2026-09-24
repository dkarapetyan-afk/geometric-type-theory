use gtt::{check_source, compile_mixtral, parameter_count, Config};
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
