//! maker/validate Tool Implementation
//!
//! Check if a response passes red-flagging without committing to voting.

use crate::core::{validate_token_length, RedFlag};
use rmcp::schemars::{self, JsonSchema};
use serde::{Deserialize, Serialize};

/// Request for maker/validate tool
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ValidateRequest {
    /// The response content to validate
    pub response: String,
    /// Maximum token count (optional)
    #[serde(default)]
    pub token_limit: Option<usize>,
    /// Expected JSON schema (optional)
    #[serde(default)]
    pub schema: Option<serde_json::Value>,
}

/// Information about a triggered red-flag
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct RedFlagInfo {
    /// Type of red-flag (e.g., "TokenLengthExceeded", "FormatViolation")
    pub flag_type: String,
    /// Human-readable details
    pub details: String,
}

impl From<&RedFlag> for RedFlagInfo {
    fn from(flag: &RedFlag) -> Self {
        match flag {
            RedFlag::TokenLengthExceeded { actual, limit } => RedFlagInfo {
                flag_type: "TokenLengthExceeded".to_string(),
                details: format!("Token count {} exceeds limit {}", actual, limit),
            },
            RedFlag::FormatViolation { message } => RedFlagInfo {
                flag_type: "FormatViolation".to_string(),
                details: message.clone(),
            },
            RedFlag::LogicLoop { pattern } => RedFlagInfo {
                flag_type: "LogicLoop".to_string(),
                details: format!("Detected repetitive pattern: {}", pattern),
            },
        }
    }
}

/// Response from maker/validate tool
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ValidateResponse {
    /// Whether the response passed all validation
    pub valid: bool,
    /// List of triggered red-flags (empty if valid)
    pub red_flags: Vec<RedFlagInfo>,
}

/// Execute validation on a response
///
/// Returns all triggered red-flags, not just the first one.
pub fn execute_validate(request: &ValidateRequest) -> ValidateResponse {
    let mut flags = Vec::new();

    // Check token length if limit specified
    if let Some(limit) = request.token_limit {
        if let Err(flag) = validate_token_length(&request.response, limit) {
            flags.push(RedFlagInfo::from(&flag));
        }
    }

    // Check JSON schema if provided
    if let Some(schema) = &request.schema {
        // Try to parse as JSON and check structure
        match serde_json::from_str::<serde_json::Value>(&request.response) {
            Ok(parsed) => {
                // Check if parsed value has expected fields from schema
                if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
                    for field in required {
                        if let Some(field_name) = field.as_str() {
                            if parsed.get(field_name).is_none() {
                                flags.push(RedFlagInfo {
                                    flag_type: "FormatViolation".to_string(),
                                    details: format!("Missing required field: {}", field_name),
                                });
                            }
                        }
                    }
                }
            }
            Err(e) => {
                flags.push(RedFlagInfo {
                    flag_type: "FormatViolation".to_string(),
                    details: format!("Invalid JSON: {}", e),
                });
            }
        }
    }

    ValidateResponse {
        valid: flags.is_empty(),
        red_flags: flags,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_request_serialization() {
        let request = ValidateRequest {
            response: "test response".to_string(),
            token_limit: Some(700),
            schema: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        let parsed: ValidateRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.response, request.response);
    }

    #[test]
    fn test_validate_request_deny_unknown_fields() {
        let json = r#"{"response": "test", "extra": true}"#;
        let result: Result<ValidateRequest, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_response_valid() {
        let response = ValidateResponse {
            valid: true,
            red_flags: vec![],
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("true"));
    }

    #[test]
    fn test_validate_response_invalid() {
        let response = ValidateResponse {
            valid: false,
            red_flags: vec![RedFlagInfo {
                flag_type: "TokenLengthExceeded".to_string(),
                details: "Token count 750 exceeds limit 700".to_string(),
            }],
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("TokenLengthExceeded"));
    }

    #[test]
    fn test_execute_validate_valid_response() {
        let request = ValidateRequest {
            response: "short response".to_string(),
            token_limit: Some(700),
            schema: None,
        };

        let result = execute_validate(&request);
        assert!(result.valid);
        assert!(result.red_flags.is_empty());
    }

    #[test]
    fn test_execute_validate_token_exceeded() {
        let request = ValidateRequest {
            response: "x".repeat(750),
            token_limit: Some(700),
            schema: None,
        };

        let result = execute_validate(&request);
        assert!(!result.valid);
        assert_eq!(result.red_flags.len(), 1);
        assert_eq!(result.red_flags[0].flag_type, "TokenLengthExceeded");
    }

    #[test]
    fn test_execute_validate_invalid_json() {
        let schema = serde_json::json!({
            "required": ["field1", "field2"]
        });

        let request = ValidateRequest {
            response: "not json".to_string(),
            token_limit: None,
            schema: Some(schema),
        };

        let result = execute_validate(&request);
        assert!(!result.valid);
        assert_eq!(result.red_flags.len(), 1);
        assert_eq!(result.red_flags[0].flag_type, "FormatViolation");
    }

    #[test]
    fn test_execute_validate_missing_required_field() {
        let schema = serde_json::json!({
            "required": ["field1", "field2"]
        });

        let request = ValidateRequest {
            response: r#"{"field1": "value"}"#.to_string(),
            token_limit: None,
            schema: Some(schema),
        };

        let result = execute_validate(&request);
        assert!(!result.valid);
        assert_eq!(result.red_flags.len(), 1);
        assert!(result.red_flags[0].details.contains("field2"));
    }

    #[test]
    fn test_execute_validate_valid_json() {
        let schema = serde_json::json!({
            "required": ["field1", "field2"]
        });

        let request = ValidateRequest {
            response: r#"{"field1": "value1", "field2": "value2"}"#.to_string(),
            token_limit: None,
            schema: Some(schema),
        };

        let result = execute_validate(&request);
        assert!(result.valid);
        assert!(result.red_flags.is_empty());
    }

    #[test]
    fn test_execute_validate_multiple_flags() {
        let schema = serde_json::json!({
            "required": ["field1"]
        });

        let request = ValidateRequest {
            response: "x".repeat(750), // Also not valid JSON
            token_limit: Some(700),
            schema: Some(schema),
        };

        let result = execute_validate(&request);
        assert!(!result.valid);
        assert_eq!(result.red_flags.len(), 2); // Token exceeded + invalid JSON
    }
}
