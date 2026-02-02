//! MCP Tool Definitions for MAKER Framework
//!
//! Each tool is defined with serde/schemars for automatic JSON schema generation.

pub mod calibrate;
pub mod configure;
pub mod health;
pub mod validate;
pub mod vote;

pub use calibrate::{CalibrateRequest, CalibrateResponse};
pub use configure::{ConfigRequest, ConfigResponse};
pub use health::{HealthRequest, HealthResponse};
pub use validate::{ValidateRequest, ValidateResponse};
pub use vote::{VoteRequest, VoteResponse};
