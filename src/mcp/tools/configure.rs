//! maker/configure Tool Implementation
//!
//! Set default voting parameters for subsequent calls.

use crate::mcp::server::ServerConfig;
use rmcp::schemars::{self, JsonSchema};
use serde::{Deserialize, Serialize};

/// Matcher configuration for candidate response grouping.
///
/// Determines how responses are compared during voting:
/// - `exact`: String equality after whitespace normalization (default)
/// - `embedding`: Cosine similarity of embedding vectors
/// - `code`: AST-based comparison (requires `code-matcher` feature)
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum MatcherConfig {
    /// Exact string matching after whitespace normalization (default)
    #[default]
    Exact,
    /// Embedding-based semantic similarity matching
    Embedding {
        /// Similarity threshold for equivalence (0.0 to 1.0, default 0.92)
        #[serde(default = "default_embedding_threshold")]
        threshold: f64,
        /// Embedding provider ("ollama" or "openai")
        #[serde(default = "default_embedding_provider")]
        provider: String,
    },
    /// Code AST matching via tree-sitter (requires `code-matcher` feature)
    Code {
        /// Programming language ("rust", "python", "javascript")
        language: String,
        /// Similarity threshold for equivalence (0.0 to 1.0, default 0.95)
        #[serde(default = "default_code_threshold")]
        threshold: f64,
    },
}

fn default_embedding_threshold() -> f64 {
    0.92
}

fn default_embedding_provider() -> String {
    "ollama".to_string()
}

fn default_code_threshold() -> f64 {
    0.95
}



impl MatcherConfig {
    /// Get a human-readable type name for this matcher configuration.
    pub fn type_name(&self) -> &str {
        match self {
            MatcherConfig::Exact => "exact",
            MatcherConfig::Embedding { .. } => "embedding",
            MatcherConfig::Code { .. } => "code",
        }
    }

    /// Validate the matcher configuration.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            MatcherConfig::Exact => Ok(()),
            MatcherConfig::Embedding {
                threshold,
                provider,
            } => {
                if !(0.0..=1.0).contains(threshold) {
                    return Err(format!(
                        "embedding threshold must be 0.0..1.0, got {}",
                        threshold
                    ));
                }
                if provider != "ollama" && provider != "openai" {
                    return Err(format!(
                        "embedding provider must be 'ollama' or 'openai', got '{}'",
                        provider
                    ));
                }
                Ok(())
            }
            MatcherConfig::Code {
                language,
                threshold,
            } => {
                if !(0.0..=1.0).contains(threshold) {
                    return Err(format!(
                        "code threshold must be 0.0..1.0, got {}",
                        threshold
                    ));
                }
                let valid = ["rust", "rs", "python", "py", "javascript", "js"];
                if !valid.contains(&language.to_lowercase().as_str()) {
                    return Err(format!(
                        "code language must be rust/python/javascript, got '{}'",
                        language
                    ));
                }
                Ok(())
            }
        }
    }
}

/// Request for maker/configure tool
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ConfigRequest {
    /// Default k-margin for voting (optional)
    #[serde(default)]
    pub k_default: Option<usize>,
    /// Default temperature diversity for sampling (optional)
    #[serde(default)]
    pub temperature_diversity: Option<f64>,
    /// Default token limit for red-flagging (optional)
    #[serde(default)]
    pub token_limit: Option<usize>,
    /// Default LLM provider (optional)
    #[serde(default)]
    pub provider: Option<String>,
    /// Enable adaptive k-margin adjustment (optional)
    #[serde(default)]
    pub adaptive_k: Option<bool>,
    /// EMA smoothing factor for adaptive k estimation (optional, default 0.1)
    #[serde(default)]
    pub ema_alpha: Option<f64>,
    /// Bounds for adaptive k as [min, max] (optional, default [2, 10])
    #[serde(default)]
    pub k_bounds: Option<(usize, usize)>,
    /// Matcher configuration for candidate response grouping (optional)
    #[serde(default)]
    pub matcher: Option<MatcherConfig>,
}

impl ConfigRequest {
    /// Check if any configuration values are provided
    pub fn has_updates(&self) -> bool {
        self.k_default.is_some()
            || self.temperature_diversity.is_some()
            || self.token_limit.is_some()
            || self.provider.is_some()
            || self.adaptive_k.is_some()
            || self.ema_alpha.is_some()
            || self.k_bounds.is_some()
            || self.matcher.is_some()
    }
}

/// Current server configuration (exported for MCP response)
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct Config {
    /// Default k-margin for voting
    pub k_default: usize,
    /// Default temperature diversity for sampling
    pub temperature_diversity: f64,
    /// Default token limit for red-flagging
    pub token_limit: usize,
    /// Default LLM provider
    pub provider: String,
    /// Whether adaptive k-margin is enabled
    pub adaptive_k: bool,
    /// Active matcher configuration
    pub matcher: MatcherConfig,
}

impl From<&ServerConfig> for Config {
    fn from(config: &ServerConfig) -> Self {
        Config {
            k_default: config.k_default,
            temperature_diversity: config.temperature_diversity,
            token_limit: config.token_limit,
            provider: config.provider.clone(),
            adaptive_k: config.adaptive_k,
            matcher: config.matcher.clone(),
        }
    }
}

/// Response from maker/configure tool
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ConfigResponse {
    /// Whether configuration was applied
    pub applied: bool,
    /// Current configuration after update
    pub current_config: Config,
}

/// Apply configuration updates to a ServerConfig
///
/// Returns the updated config and whether any changes were made.
pub fn apply_config_updates(config: &mut ServerConfig, request: &ConfigRequest) -> bool {
    let mut applied = false;

    if let Some(k) = request.k_default {
        if k >= 1 {
            config.k_default = k;
            applied = true;
        }
    }

    if let Some(temp) = request.temperature_diversity {
        if (0.0..=1.0).contains(&temp) {
            config.temperature_diversity = temp;
            applied = true;
        }
    }

    if let Some(limit) = request.token_limit {
        if limit > 0 {
            config.token_limit = limit;
            applied = true;
        }
    }

    if let Some(ref provider) = request.provider {
        if !provider.is_empty() {
            config.provider = provider.clone();
            applied = true;
        }
    }

    if let Some(adaptive) = request.adaptive_k {
        config.adaptive_k = adaptive;
        applied = true;
    }

    if let Some(alpha) = request.ema_alpha {
        if (0.0..=1.0).contains(&alpha) {
            config.ema_alpha = alpha;
            applied = true;
        }
    }

    if let Some((min, max)) = request.k_bounds {
        if min >= 1 && max >= min {
            config.k_bounds = (min, max);
            applied = true;
        }
    }

    if let Some(ref matcher) = request.matcher {
        if matcher.validate().is_ok() {
            config.matcher = matcher.clone();
            applied = true;
        }
    }

    applied
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_request_partial() {
        let json = r#"{"k_default": 5}"#;
        let request: ConfigRequest = serde_json::from_str(json).unwrap();
        assert_eq!(request.k_default, Some(5));
        assert!(request.temperature_diversity.is_none());
        assert!(request.token_limit.is_none());
        assert!(request.provider.is_none());
    }

    #[test]
    fn test_config_request_full() {
        let request = ConfigRequest {
            k_default: Some(4),
            temperature_diversity: Some(0.2),
            token_limit: Some(1000),
            provider: Some("openai".to_string()),
            ..Default::default()
        };

        let json = serde_json::to_string(&request).unwrap();
        let parsed: ConfigRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.k_default, Some(4));
        assert_eq!(parsed.provider, Some("openai".to_string()));
    }

    #[test]
    fn test_config_request_deny_unknown_fields() {
        let json = r#"{"k_default": 5, "unknown": true}"#;
        let result: Result<ConfigRequest, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_response_serialization() {
        let response = ConfigResponse {
            applied: true,
            current_config: Config {
                k_default: 4,
                temperature_diversity: 0.15,
                token_limit: 800,
                provider: "anthropic".to_string(),
                adaptive_k: false,
                matcher: MatcherConfig::default(),
            },
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("applied"));
        assert!(json.contains("anthropic"));
    }

    #[test]
    fn test_config_request_has_updates() {
        let empty = ConfigRequest::default();
        assert!(!empty.has_updates());

        let with_k = ConfigRequest {
            k_default: Some(5),
            ..Default::default()
        };
        assert!(with_k.has_updates());
    }

    #[test]
    fn test_apply_config_updates_k_default() {
        let mut config = ServerConfig::default();
        let request = ConfigRequest {
            k_default: Some(5),
            ..Default::default()
        };

        let applied = apply_config_updates(&mut config, &request);
        assert!(applied);
        assert_eq!(config.k_default, 5);
    }

    #[test]
    fn test_apply_config_updates_temperature() {
        let mut config = ServerConfig::default();
        let request = ConfigRequest {
            temperature_diversity: Some(0.25),
            ..Default::default()
        };

        let applied = apply_config_updates(&mut config, &request);
        assert!(applied);
        assert!((config.temperature_diversity - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_apply_config_updates_invalid_values() {
        let mut config = ServerConfig::default();
        let original_k = config.k_default;
        let original_temp = config.temperature_diversity;

        // k_default = 0 should not be applied
        let request = ConfigRequest {
            k_default: Some(0),
            temperature_diversity: Some(1.5), // Out of range
            token_limit: Some(0),             // Invalid
            provider: Some("".to_string()),   // Empty
            ..Default::default()
        };

        let applied = apply_config_updates(&mut config, &request);
        assert!(!applied);
        assert_eq!(config.k_default, original_k);
        assert!((config.temperature_diversity - original_temp).abs() < 0.001);
    }

    #[test]
    fn test_apply_config_updates_multiple() {
        let mut config = ServerConfig::default();
        let request = ConfigRequest {
            k_default: Some(4),
            temperature_diversity: Some(0.2),
            token_limit: Some(1000),
            provider: Some("anthropic".to_string()),
            ..Default::default()
        };

        let applied = apply_config_updates(&mut config, &request);
        assert!(applied);
        assert_eq!(config.k_default, 4);
        assert!((config.temperature_diversity - 0.2).abs() < 0.001);
        assert_eq!(config.token_limit, 1000);
        assert_eq!(config.provider, "anthropic");
    }

    #[test]
    fn test_config_from_server_config() {
        let server_config = ServerConfig::default();
        let config = Config::from(&server_config);

        assert_eq!(config.k_default, server_config.k_default);
        assert_eq!(
            config.temperature_diversity,
            server_config.temperature_diversity
        );
        assert_eq!(config.token_limit, server_config.token_limit);
        assert_eq!(config.provider, server_config.provider);
    }
}
