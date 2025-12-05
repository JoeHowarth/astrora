//! Astrora Core - Rust-backed astrodynamics library
//!
//! This crate provides high-performance orbital mechanics calculations.
//!
//! Formerly known as poliastro (archived 2023), astrora is a modern
//! reimplementation with significant performance improvements.

// Module declarations
pub mod core;
pub mod propagators;
pub mod coordinates;
pub mod maneuvers;
pub mod satellite;
pub mod utils;

// Test utilities (only compiled in test mode)
#[cfg(test)]
pub mod test_utils;

// Re-export commonly used types for Rust API
pub use core::{PoliastroError, PoliastroResult};
