//! MAKER Framework - Zero-Error LLM Agent Execution
//!
//! The MAKER (Massively decomposed Agentic processes with K-margin Error Reduction)
//! framework enables zero-error execution of long-horizon LLM agent tasks through:
//!
//! - **Microagent Architecture (m=1)**: Single-subtask agents minimize context burden
//! - **SPRT Voting**: First-to-ahead-by-k protocol for mathematical error correction
//! - **Red-Flagging**: Discard malformed outputs to maintain error decorrelation
//!
//! # Key Insight
//!
//! Even with 99% per-step accuracy, a 1M-step task has 0% success rate.
//! MAKER transforms this into Θ(s ln s) cost scaling with provable reliability.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use maker::core::calculate_kmin;
//!
//! // Calculate k_min for 10-disk Towers of Hanoi (1,023 steps)
//! let k = calculate_kmin(
//!     0.85,   // p: per-step success probability
//!     0.95,   // t: target task reliability
//!     1_023,  // s: total steps
//!     1,      // m: steps per agent (must be 1)
//! ).unwrap();
//!
//! println!("Required k-margin: {}", k);
//! ```

pub mod core;
pub mod events;
pub mod llm;
pub mod mcp;

// Re-export commonly used items at crate root
pub use core::{calculate_kmin, KminError};
pub use events::observers::{LoggingObserver, MetricsObserver};
pub use events::{EventBus, MakerEvent};
pub use mcp::{MakerServer, ServerConfig};
