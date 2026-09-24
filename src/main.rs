use gtt::check_source;
use gtt::{compile_mixtral, kernel_source};
use std::env;
use std::fs;
use std::process::{Command, ExitCode};

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| "check".into());
    if cmd == "compile" {
        return compile();
    }
    if cmd != "check" {
        eprintln!("usage: gtt check <file.gtt>");
        eprintln!("       gtt compile");
        return ExitCode::from(2);
    }
    let mut failed = false;
    let files: Vec<String> = args.collect();
    if files.is_empty() {
        eprintln!("usage: gtt check <file.gtt>");
        return ExitCode::from(2);
    }
    for path in files {
        let src = match fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("{path}: {e}");
                failed = true;
                continue;
            }
        };
        match check_source(&src) {
            Ok(()) => println!("{path}: ok"),
            Err(e) => {
                eprintln!("{path}:\n{}", e.render(&src));
                failed = true;
            }
        }
    }
    if failed {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

fn compile() -> ExitCode {
    let report = compile_mixtral();
    println!("Mixtral-8x7B parameters: {}", report.paper_parameters);
    println!("active parameters:       {}", report.active_parameters);
    println!(
        "demo loss {:.6} -> {:.6} at lr {}",
        report.loss_before, report.loss_after, report.learning_rate
    );
    println!("max finite-difference error: {}", report.max_abs_fd_error);
    println!(
        "sparse loss {:.6}, idle expert grad {}, routed expert grad {}",
        report.sparse_loss, report.unused_expert_grad, report.used_expert_grad
    );
    let path = "examples/mixtral/kernels.cu";
    if let Err(e) = fs::write(path, kernel_source()) {
        eprintln!("{path}: {e}");
        return ExitCode::from(1);
    }
    println!("{path}: wrote");
    let nvcc = "/usr/local/cuda/bin/nvcc";
    let status = Command::new(nvcc)
        .args([
            "-ccbin",
            "/usr/bin/g++-11",
            "-arch=sm_75",
            "-O2",
            path,
            "-o",
            "/tmp/mixtral_kernels",
        ])
        .status();
    match status {
        Ok(code) if code.success() => {}
        Ok(code) => {
            eprintln!("nvcc exited {code}");
            return ExitCode::from(1);
        }
        Err(e) => {
            eprintln!("nvcc: {e}");
            return ExitCode::from(1);
        }
    }
    match Command::new("/tmp/mixtral_kernels").status() {
        Ok(code) if code.success() => ExitCode::SUCCESS,
        Ok(code) => {
            eprintln!("kernel driver exited {code}");
            ExitCode::from(1)
        }
        Err(e) => {
            eprintln!("kernel driver: {e}");
            ExitCode::from(1)
        }
    }
}
