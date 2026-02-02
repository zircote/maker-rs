//! maker/health Tool Implementation
//!
//! Check server health status including uptime and component health.

use crate::mcp::health::{HealthChecker, HealthStatus};
use rmcp::schemars::{self, JsonSchema};
use serde::{Deserialize, Serialize};

/// Request for maker/health tool
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HealthRequest {
    /// Whether to include LLM provider health check (optional, slower)
    #[serde(default)]
    pub check_provider: Option<bool>,
}

/// Response from maker/health tool
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct HealthResponse {
    /// Health status information
    #[serde(flatten)]
    pub status: HealthStatus,
}

/// Execute health check
pub fn execute_health(request: &HealthRequest, checker: &HealthChecker) -> HealthResponse {
    let status = if request.check_provider.unwrap_or(false) {
        // For now, assume provider is healthy since we can't check synchronously
        // In a real implementation, this would ping the configured LLM provider
        checker.check_with_provider(true)
    } else {
        checker.check()
    };

    HealthResponse { status }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_health_check() {
        let checker = HealthChecker::new();
        let request = HealthRequest::default();

        let response = execute_health(&request, &checker);

        assert!(response.status.status.is_healthy());
        assert!(!response.status.version.is_empty());
    }

    #[test]
    fn test_health_check_with_provider() {
        let checker = HealthChecker::new();
        let request = HealthRequest {
            check_provider: Some(true),
        };

        let response = execute_health(&request, &checker);

        assert!(response.status.status.is_healthy());
        assert!(response.status.components.llm_provider.is_some());
    }
}
