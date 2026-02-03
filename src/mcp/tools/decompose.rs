//! maker/decompose Tool Implementation
//!
//! Execute recursive task decomposition to split complex tasks into manageable subtasks.

use crate::core::decomposition::{
    CompositionFunction, DecompositionAgent, DecompositionError, LlmAgentConfig,
    LlmDecompositionAgent,
};
use crate::core::executor::LlmClient;
use crate::llm::adapter::setup_provider_client;
use rmcp::schemars::{self, JsonSchema};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Request for maker/decompose tool
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DecomposeRequest {
    /// The task description to decompose
    pub task: String,
    /// Maximum recursion depth (default: 10)
    #[serde(default = "default_depth_limit")]
    pub depth_limit: usize,
    /// Timeout in milliseconds (default: 60000)
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
    /// LLM provider to use (optional, defaults to server config)
    #[serde(default)]
    pub provider: Option<String>,
    /// Model name override (optional)
    #[serde(default)]
    pub model: Option<String>,
}

fn default_depth_limit() -> usize {
    10
}

fn default_timeout_ms() -> u64 {
    60_000
}

impl DecomposeRequest {
    /// Validate the request parameters
    pub fn validate(&self) -> Result<(), DecomposeToolError> {
        if self.task.is_empty() {
            return Err(DecomposeToolError::EmptyTask);
        }
        if self.depth_limit == 0 {
            return Err(DecomposeToolError::InvalidDepthLimit {
                provided: 0,
                min: 1,
            });
        }
        if self.timeout_ms == 0 {
            return Err(DecomposeToolError::InvalidTimeout { provided: 0 });
        }
        Ok(())
    }
}

/// Response from maker/decompose tool
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct DecomposeResponse {
    /// Unique identifier for this decomposition proposal
    pub proposal_id: String,
    /// The subtasks resulting from decomposition
    pub subtasks: Vec<SubtaskInfo>,
    /// Composition strategy for combining subtask results
    pub composition: String,
    /// Depth of decomposition (0 for atomic tasks)
    pub depth: usize,
    /// Confidence score for this decomposition (0.0-1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// Rationale for this decomposition
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
}

/// Information about a single subtask
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SubtaskInfo {
    /// Unique identifier for this subtask
    pub id: String,
    /// Description of what this subtask does
    pub description: String,
    /// Whether this is a leaf node (atomic, cannot be decomposed further)
    pub is_leaf: bool,
    /// Execution order hint (lower = earlier)
    #[serde(default)]
    pub order: usize,
}

/// Errors that can occur during decompose tool execution
#[derive(Debug, Clone, PartialEq)]
pub enum DecomposeToolError {
    /// Task description cannot be empty
    EmptyTask,
    /// Invalid depth limit
    InvalidDepthLimit { provided: usize, min: usize },
    /// Invalid timeout
    InvalidTimeout { provided: u64 },
    /// Decomposition failed
    DecompositionFailed { message: String },
    /// LLM provider error
    ProviderError { message: String },
    /// Timeout exceeded
    Timeout { elapsed_ms: u64, limit_ms: u64 },
}

impl std::fmt::Display for DecomposeToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyTask => write!(f, "task description cannot be empty"),
            Self::InvalidDepthLimit { provided, min } => {
                write!(f, "depth_limit must be >= {}, provided: {}", min, provided)
            }
            Self::InvalidTimeout { provided } => {
                write!(f, "timeout_ms must be > 0, provided: {}", provided)
            }
            Self::DecompositionFailed { message } => {
                write!(f, "decomposition failed: {}", message)
            }
            Self::ProviderError { message } => write!(f, "LLM provider error: {}", message),
            Self::Timeout {
                elapsed_ms,
                limit_ms,
            } => {
                write!(
                    f,
                    "decomposition timed out after {}ms (limit: {}ms)",
                    elapsed_ms, limit_ms
                )
            }
        }
    }
}

impl std::error::Error for DecomposeToolError {}

/// Execute task decomposition
///
/// This function wraps the LlmDecompositionAgent for use in the MCP tool.
pub fn execute_decompose(
    request: &DecomposeRequest,
    default_provider: &str,
) -> Result<DecomposeResponse, DecomposeToolError> {
    request.validate()?;

    // Create provider
    let provider_name = request.provider.as_deref().unwrap_or(default_provider);
    let client: Arc<dyn LlmClient> = Arc::from(
        setup_provider_client(provider_name, request.model.clone())
            .map_err(|e| DecomposeToolError::ProviderError { message: e })?,
    );

    // Create LLM decomposition agent
    let agent_config = LlmAgentConfig::default().with_max_subtasks(request.depth_limit);
    let agent = LlmDecompositionAgent::new(client, agent_config);

    // Track timeout
    let start = std::time::Instant::now();
    let timeout_duration = std::time::Duration::from_millis(request.timeout_ms);

    // Generate task ID
    let task_id = format!(
        "task_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );

    // Propose decomposition
    let proposal = agent
        .propose_decomposition(&task_id, &request.task, &serde_json::Value::Null, 0)
        .map_err(|e| match e {
            DecompositionError::Timeout {
                elapsed_ms,
                limit_ms,
            } => DecomposeToolError::Timeout {
                elapsed_ms,
                limit_ms,
            },
            DecompositionError::AgentError { message } => {
                DecomposeToolError::DecompositionFailed { message }
            }
            other => DecomposeToolError::DecompositionFailed {
                message: other.to_string(),
            },
        })?;

    // Check timeout after decomposition
    let elapsed = start.elapsed();
    if elapsed > timeout_duration {
        return Err(DecomposeToolError::Timeout {
            elapsed_ms: elapsed.as_millis() as u64,
            limit_ms: request.timeout_ms,
        });
    }

    // Convert subtasks to response format
    let subtasks: Vec<SubtaskInfo> = proposal
        .subtasks
        .iter()
        .map(|st| SubtaskInfo {
            id: st.task_id.clone(),
            description: st.description.clone(),
            is_leaf: st.is_leaf,
            order: st.order,
        })
        .collect();

    // Convert composition function to string
    let composition = match proposal.composition_fn {
        CompositionFunction::Sequential => "sequential",
        CompositionFunction::Parallel { .. } => "parallel",
        CompositionFunction::Conditional { .. } => "conditional",
        CompositionFunction::Custom { ref name, .. } => name.as_str(),
    };

    Ok(DecomposeResponse {
        proposal_id: proposal.proposal_id,
        subtasks,
        composition: composition.to_string(),
        depth: if proposal.subtasks.is_empty() { 0 } else { 1 },
        confidence: Some(proposal.confidence),
        rationale: proposal.rationale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(task: &str) -> DecomposeRequest {
        DecomposeRequest {
            task: task.to_string(),
            depth_limit: 10,
            timeout_ms: 60_000,
            provider: None,
            model: None,
        }
    }

    #[test]
    fn test_request_serialization() {
        let request = DecomposeRequest {
            task: "Build a web server".to_string(),
            depth_limit: 5,
            timeout_ms: 30_000,
            provider: Some("ollama".to_string()),
            model: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        let parsed: DecomposeRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.task, request.task);
        assert_eq!(parsed.depth_limit, request.depth_limit);
        assert_eq!(parsed.timeout_ms, request.timeout_ms);
    }

    #[test]
    fn test_request_deny_unknown_fields() {
        let json = r#"{"task": "test", "unknown_field": true}"#;
        let result: Result<DecomposeRequest, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_empty_task() {
        let request = make_request("");
        assert_eq!(request.validate(), Err(DecomposeToolError::EmptyTask));
    }

    #[test]
    fn test_validate_zero_depth() {
        let mut request = make_request("test");
        request.depth_limit = 0;
        assert!(matches!(
            request.validate(),
            Err(DecomposeToolError::InvalidDepthLimit { .. })
        ));
    }

    #[test]
    fn test_validate_zero_timeout() {
        let mut request = make_request("test");
        request.timeout_ms = 0;
        assert!(matches!(
            request.validate(),
            Err(DecomposeToolError::InvalidTimeout { .. })
        ));
    }

    #[test]
    fn test_validate_valid_request() {
        let request = make_request("Build a calculator");
        assert!(request.validate().is_ok());
    }

    #[test]
    fn test_response_serialization() {
        let response = DecomposeResponse {
            proposal_id: "prop-123".to_string(),
            subtasks: vec![
                SubtaskInfo {
                    id: "sub-1".to_string(),
                    description: "First step".to_string(),
                    is_leaf: true,
                    order: 0,
                },
                SubtaskInfo {
                    id: "sub-2".to_string(),
                    description: "Second step".to_string(),
                    is_leaf: true,
                    order: 1,
                },
            ],
            composition: "sequential".to_string(),
            depth: 1,
            confidence: Some(0.85),
            rationale: Some("Good decomposition".to_string()),
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("prop-123"));
        assert!(json.contains("sequential"));
    }

    #[test]
    fn test_error_display() {
        let errors = vec![
            DecomposeToolError::EmptyTask,
            DecomposeToolError::InvalidDepthLimit {
                provided: 0,
                min: 1,
            },
            DecomposeToolError::InvalidTimeout { provided: 0 },
            DecomposeToolError::DecompositionFailed {
                message: "test error".to_string(),
            },
            DecomposeToolError::ProviderError {
                message: "connection failed".to_string(),
            },
            DecomposeToolError::Timeout {
                elapsed_ms: 65000,
                limit_ms: 60000,
            },
        ];

        for error in errors {
            let display = format!("{}", error);
            assert!(!display.is_empty());
        }
    }

    #[test]
    fn test_default_values() {
        assert_eq!(default_depth_limit(), 10);
        assert_eq!(default_timeout_ms(), 60_000);
    }

    #[test]
    #[ignore] // Requires running Ollama instance
    fn test_execute_decompose_with_ollama() {
        let request = DecomposeRequest {
            task: "Build a simple calculator".to_string(),
            depth_limit: 5,
            timeout_ms: 30_000,
            provider: Some("ollama".to_string()),
            model: None,
        };

        let result = execute_decompose(&request, "ollama");
        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(!response.proposal_id.is_empty());
        assert!(!response.subtasks.is_empty() || response.depth == 0);
    }

    #[test]
    fn test_execute_decompose_unknown_provider() {
        let request = DecomposeRequest {
            task: "Test task".to_string(),
            depth_limit: 5,
            timeout_ms: 30_000,
            provider: Some("unknown-provider".to_string()),
            model: None,
        };

        let result = execute_decompose(&request, "ollama");
        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(DecomposeToolError::ProviderError { .. })
        ));
    }

    #[test]
    fn test_execute_decompose_invalid_request() {
        let request = make_request("");
        let result = execute_decompose(&request, "ollama");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), DecomposeToolError::EmptyTask);
    }
}
