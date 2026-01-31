//! MCP Server Implementation for MAKER Framework
//!
//! Exposes MAKER capabilities as MCP tools for integration with AI assistants
//! like Claude Code. Uses rmcp SDK with stdio transport.
//!
//! # Available Tools
//!
//! - `maker/vote` - Execute SPRT voting on a prompt
//! - `maker/validate` - Check if a response passes red-flagging
//! - `maker/calibrate` - Estimate per-step success rate (p)
//! - `maker/configure` - Set default voting parameters
//!
//! # Health Check
//!
//! The server provides a health check endpoint via the `health` module.

pub mod health;
pub mod server;
pub mod tools;

pub use health::{validate_config, HealthChecker, HealthState, HealthStatus};
pub use server::{MakerServer, ServerConfig};
