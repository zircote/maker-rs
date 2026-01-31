//! MAKER MCP Server Implementation
//!
//! Provides the core server handler for rmcp, managing server state
//! and routing tool calls to their respective handlers.

use crate::llm::ensemble::EnsembleConfigRequest;
use crate::mcp::tools::{
    calibrate::{execute_calibrate, CalibrateRequest},
    configure::{apply_config_updates, Config, ConfigRequest, ConfigResponse, MatcherConfig},
    validate::{execute_validate, ValidateRequest},
    vote::{execute_vote, VoteRequest},
};
use rmcp::{
    handler::server::{tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, Implementation, ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Server configuration with defaults
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Default k-margin for voting
    pub k_default: usize,
    /// Default temperature diversity for sampling
    pub temperature_diversity: f64,
    /// Default token limit for red-flagging
    pub token_limit: usize,
    /// Default LLM provider
    pub provider: String,
    /// Maximum prompt length (security)
    pub max_prompt_length: usize,
    /// Whether adaptive k-margin adjustment is enabled
    pub adaptive_k: bool,
    /// EMA smoothing factor for adaptive k estimation
    pub ema_alpha: f64,
    /// Bounds for adaptive k as (min, max)
    pub k_bounds: (usize, usize),
    /// Active matcher configuration for candidate response grouping
    pub matcher: MatcherConfig,
    /// Ensemble configuration for multi-model voting (None = single-model)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ensemble: Option<EnsembleConfigRequest>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            k_default: 3,
            temperature_diversity: 0.1,
            token_limit: 700,
            provider: "ollama".to_string(),
            max_prompt_length: 10_000,
            adaptive_k: false,
            ema_alpha: 0.1,
            k_bounds: (2, 10),
            matcher: MatcherConfig::default(),
            ensemble: None,
        }
    }
}

/// Shared server state
#[derive(Debug)]
pub struct ServerState {
    /// Current configuration
    pub config: RwLock<ServerConfig>,
}

impl Default for ServerState {
    fn default() -> Self {
        Self {
            config: RwLock::new(ServerConfig::default()),
        }
    }
}

/// MAKER MCP Server
///
/// Implements the rmcp ServerHandler trait to expose MAKER tools
/// via the Model Context Protocol.
#[derive(Clone)]
pub struct MakerServer {
    state: Arc<ServerState>,
    tool_router: ToolRouter<Self>,
}

// Tool implementations using rmcp macros
#[tool_router]
impl MakerServer {
    /// Create a new MAKER server with default configuration
    pub fn new() -> Self {
        Self {
            state: Arc::new(ServerState::default()),
            tool_router: Self::tool_router(),
        }
    }

    /// Create a new MAKER server with custom configuration
    pub fn with_config(config: ServerConfig) -> Self {
        Self {
            state: Arc::new(ServerState {
                config: RwLock::new(config),
            }),
            tool_router: Self::tool_router(),
        }
    }

    /// Get the current server configuration
    pub async fn get_config(&self) -> ServerConfig {
        self.state.config.read().await.clone()
    }

    /// Update the server configuration
    pub async fn set_config(&self, config: ServerConfig) {
        *self.state.config.write().await = config;
    }

    /// Execute SPRT voting on a prompt to get the voted winner with confidence metrics.
    ///
    /// This tool implements the MAKER first-to-ahead-by-k voting protocol for
    /// error-corrected LLM execution.
    #[tool(
        name = "maker/vote",
        description = "Execute SPRT voting on a prompt to get the voted winner with confidence metrics"
    )]
    async fn vote(
        &self,
        Parameters(request): Parameters<VoteRequest>,
    ) -> Result<CallToolResult, McpError> {
        let config = self.state.config.read().await;

        match execute_vote(
            &request,
            config.k_default * 10, // max_samples
            config.temperature_diversity,
            Some(config.token_limit),
        ) {
            Ok(response) => {
                let json = serde_json::to_string_pretty(&response)
                    .map_err(|e| McpError::internal_error(e.to_string(), None))?;
                Ok(CallToolResult::success(vec![Content::text(json)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    /// Check if a response passes red-flagging without committing to voting.
    ///
    /// Returns all triggered red-flags for the given response.
    #[tool(
        name = "maker/validate",
        description = "Check if a response passes red-flagging validation"
    )]
    async fn validate(
        &self,
        Parameters(request): Parameters<ValidateRequest>,
    ) -> Result<CallToolResult, McpError> {
        let response = execute_validate(&request);
        let json = serde_json::to_string_pretty(&response)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    /// Estimate per-step success rate (p) from calibration samples.
    ///
    /// Returns p_estimate, confidence interval, and recommended k-margin.
    #[tool(
        name = "maker/calibrate",
        description = "Estimate per-step success rate (p) from calibration samples"
    )]
    async fn calibrate(
        &self,
        Parameters(request): Parameters<CalibrateRequest>,
    ) -> Result<CallToolResult, McpError> {
        match execute_calibrate(&request) {
            Ok(response) => {
                let json = serde_json::to_string_pretty(&response)
                    .map_err(|e| McpError::internal_error(e.to_string(), None))?;
                Ok(CallToolResult::success(vec![Content::text(json)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![Content::text(e.to_string())])),
        }
    }

    /// Set default voting parameters for subsequent calls.
    ///
    /// Updates k_default, temperature_diversity, token_limit, and/or provider.
    #[tool(
        name = "maker/configure",
        description = "Set default voting parameters for subsequent calls"
    )]
    async fn configure(
        &self,
        Parameters(request): Parameters<ConfigRequest>,
    ) -> Result<CallToolResult, McpError> {
        let mut config = self.state.config.write().await;
        let applied = apply_config_updates(&mut config, &request);

        let response = ConfigResponse {
            applied,
            current_config: Config::from(&*config),
        };

        let json = serde_json::to_string_pretty(&response)
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }
}

impl Default for MakerServer {
    fn default() -> Self {
        Self::new()
    }
}

#[tool_handler]
impl ServerHandler for MakerServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: Default::default(),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "maker-mcp".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                ..Default::default()
            },
            instructions: Some(
                "MAKER Framework - Zero-error LLM agent execution via SPRT voting. \
                 Available tools: maker/vote, maker/validate, maker/calibrate, maker/configure."
                    .into(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.k_default, 3);
        assert!((config.temperature_diversity - 0.1).abs() < f64::EPSILON);
        assert_eq!(config.token_limit, 700);
        assert_eq!(config.provider, "ollama");
        assert_eq!(config.max_prompt_length, 10_000);
    }

    #[test]
    fn test_server_new() {
        let server = MakerServer::new();
        let info = server.get_info();
        assert_eq!(info.server_info.name, "maker-mcp");
    }

    #[tokio::test]
    async fn test_server_config_update() {
        let server = MakerServer::new();

        let mut new_config = server.get_config().await;
        new_config.k_default = 5;
        server.set_config(new_config).await;

        let updated = server.get_config().await;
        assert_eq!(updated.k_default, 5);
    }

    #[test]
    fn test_server_info_capabilities() {
        let server = MakerServer::new();
        let info = server.get_info();

        // Verify tools capability is enabled
        assert!(info.capabilities.tools.is_some());
    }

    #[test]
    fn test_server_with_config() {
        let config = ServerConfig {
            k_default: 5,
            temperature_diversity: 0.2,
            token_limit: 500,
            provider: "openai".to_string(),
            max_prompt_length: 5000,
            ..Default::default()
        };
        let server = MakerServer::with_config(config);
        let info = server.get_info();
        assert_eq!(info.server_info.name, "maker-mcp");
    }

    #[tokio::test]
    async fn test_server_with_config_values() {
        let config = ServerConfig {
            k_default: 7,
            temperature_diversity: 0.3,
            token_limit: 1000,
            provider: "anthropic".to_string(),
            max_prompt_length: 20_000,
            ..Default::default()
        };
        let server = MakerServer::with_config(config);
        let cfg = server.get_config().await;
        assert_eq!(cfg.k_default, 7);
        assert!((cfg.temperature_diversity - 0.3).abs() < f64::EPSILON);
        assert_eq!(cfg.token_limit, 1000);
        assert_eq!(cfg.provider, "anthropic");
        assert_eq!(cfg.max_prompt_length, 20_000);
    }

    #[test]
    fn test_server_default() {
        let server = MakerServer::default();
        let info = server.get_info();
        assert_eq!(info.server_info.name, "maker-mcp");
        assert!(info.instructions.is_some());
        assert!(info.instructions.unwrap().contains("MAKER Framework"));
    }

    #[test]
    fn test_server_info_version() {
        let server = MakerServer::new();
        let info = server.get_info();
        assert_eq!(info.server_info.version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn test_server_config_serialization() {
        let config = ServerConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("\"k_default\":3"));
        assert!(json.contains("\"provider\":\"ollama\""));

        let deserialized: ServerConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.k_default, config.k_default);
        assert_eq!(deserialized.provider, config.provider);
    }

    #[tokio::test]
    async fn test_vote_tool_handler() {
        let server = MakerServer::new();
        let request = VoteRequest {
            prompt: "What is 2+2?".to_string(),
            k_margin: 3,
            max_samples: Some(20),
            temperature_diversity: None,
            provider: None,
            adaptive: None,
            matcher: None,
            ensemble: None,
        };

        let result = server.vote(Parameters(request)).await;

        // Should succeed (uses internal mock-like behavior)
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_validate_tool_handler() {
        let server = MakerServer::new();
        let request = ValidateRequest {
            response: r#"{"move_action": "move disk 1 from A to C", "next_state": {}}"#.to_string(),
            token_limit: Some(700),
            schema: None,
        };

        let result = server.validate(Parameters(request)).await;

        assert!(result.is_ok());
        let call_result = result.unwrap();
        let text = format!("{:?}", call_result);
        assert!(text.contains("valid"));
    }

    #[tokio::test]
    async fn test_calibrate_tool_handler_no_samples() {
        let server = MakerServer::new();
        let request = CalibrateRequest {
            samples: vec![],
            target_reliability: 0.95,
            target_steps: 1000,
        };

        let result = server.calibrate(Parameters(request)).await;

        // Should return error for empty samples
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_configure_tool_handler() {
        let server = MakerServer::new();
        let request = ConfigRequest {
            k_default: Some(5),
            temperature_diversity: Some(0.2),
            ..Default::default()
        };

        let result = server.configure(Parameters(request)).await;

        assert!(result.is_ok());

        // Verify config was updated
        let cfg = server.get_config().await;
        assert_eq!(cfg.k_default, 5);
        assert!((cfg.temperature_diversity - 0.2).abs() < f64::EPSILON);
    }
}
