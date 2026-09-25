//! Kernel and checker for geometric type theory.
//!
//! The two styles are Mitchell Riley’s, from “Geometric Type Theory, Done Two
//! Ways” (Topos Institute, 2026-06-15). Section 3 of that note — quasicoherence,
//! the `Const` modality, and the multimodal sketch — is not implemented.

mod ad;
mod cuda_emit;
mod dist;
mod elab;
mod error;
mod mixtral;
mod nbe;
mod parser;
mod stage;
mod surface;

pub use cuda_emit::{kernel_catalog, kernel_code_bytes, kernel_source, KernelSpec};
pub use dist::{run_on_cluster, verify_distributed_step};
pub use error::{Error, Span};
pub use mixtral::{compile_mixtral, parameter_count, CompileReport, Config};
pub use stage::{
    minimum_budget, mixtral_cluster, read_checkpoint, save_random, schedule, schedule_on,
    staged_update_dir, verify_staged_checkpoint, Buffer, BufferRole, Cluster, Device, DeviceKind,
    Link, Memory, Message, Phase, Schedule, Stage, Step,
};

pub fn check_source(src: &str) -> Result<(), Error> {
    let items = parser::parse_file(src)?;
    elab::check_items(&items)
}
