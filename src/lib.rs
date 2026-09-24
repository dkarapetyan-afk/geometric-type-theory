//! Kernel and checker for geometric type theory.
//!
//! The two styles are Mitchell Riley’s, from “Geometric Type Theory, Done Two
//! Ways” (Topos Institute, 2026-06-15). Section 3 of that note — quasicoherence,
//! the `Const` modality, and the multimodal sketch — is not implemented.

mod ad;
mod cuda_emit;
mod elab;
mod error;
mod mixtral;
mod nbe;
mod parser;
mod surface;

pub use cuda_emit::kernel_source;
pub use error::{Error, Span};
pub use mixtral::{compile_mixtral, parameter_count, CompileReport, Config};

pub fn check_source(src: &str) -> Result<(), Error> {
    let items = parser::parse_file(src)?;
    elab::check_items(&items)
}
