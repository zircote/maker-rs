//! Red-Flagging Parsers for MAKER Framework
//!
//! Red-flagging is a critical protocol for identifying and discarding malformed
//! LLM outputs without attempting repair. This maintains error decorrelation,
//! which is essential for the mathematical efficiency of SPRT-based voting.
//!
//! # Philosophy
//!
//! > "In zero-error environments, discarding a sample is superior to 'repairing'
//! > its format. Attempting to post-process a misformatted response masks
//! > underlying logical instability."
//!
//! # Red Flag Types
//!
//! - **Token Length Exceeded**: Response exceeds configured token limit (e.g., >700 tokens)
//! - **Format Violation**: Response doesn't match expected JSON schema
//! - **Logic Loop**: Repetitive semantic content (future implementation)

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tracing::warn;

/// Strict agent output schema with deny_unknown_fields
///
/// This is used for security-critical validation where extra fields
/// could indicate malicious input or confused model output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct StrictAgentOutput {
    /// The action/move performed by the agent (alias: "action" or "move")
    #[serde(alias = "action", alias = "move")]
    pub move_action: String,
    /// The state after executing the action
    pub next_state: serde_json::Value,
}

/// Validate agent output with strict schema enforcement
///
/// Checks for:
/// - Missing required fields (move_action, next_state)
/// - Unexpected extra fields (in strict mode)
/// - Schema violations logged at WARN level
///
/// # Arguments
///
/// * `content` - JSON string of agent output
/// * `strict_mode` - If true, reject outputs with unexpected fields
///
/// # Returns
///
/// * `Ok(StrictAgentOutput)` if valid
/// * `Err(RedFlag::FormatViolation)` with details if invalid
pub fn validate_agent_output(
    content: &str,
    strict_mode: bool,
) -> Result<StrictAgentOutput, RedFlag> {
    // First try strict parsing
    let strict_result: Result<StrictAgentOutput, _> = serde_json::from_str(content);

    match strict_result {
        Ok(output) => {
            // Validate next_state is not null/empty
            if output.next_state.is_null() {
                let msg = "next_state field is null".to_string();
                warn!(content_hash = %hash_content(content), "Schema violation: {}", msg);
                return Err(RedFlag::FormatViolation { message: msg });
            }
            Ok(output)
        }
        Err(e) => {
            let error_msg = e.to_string();

            // Check for specific missing field errors
            if error_msg.contains("missing field") {
                let field = if error_msg.contains("next_state") {
                    "next_state"
                } else if error_msg.contains("move_action")
                    || error_msg.contains("action")
                    || error_msg.contains("move")
                {
                    "move_action (or alias: action, move)"
                } else {
                    "required field"
                };
                let msg = format!("Agent output missing required field: {}", field);
                warn!(content_hash = %hash_content(content), "Schema violation: {}", msg);
                return Err(RedFlag::FormatViolation { message: msg });
            }

            // Check for unknown field errors (strict mode)
            if error_msg.contains("unknown field") {
                if strict_mode {
                    let msg = format!("Agent output contains unexpected fields: {}", error_msg);
                    warn!(content_hash = %hash_content(content), "Schema violation (strict): {}", msg);
                    return Err(RedFlag::FormatViolation { message: msg });
                } else {
                    // In non-strict mode, try lenient parsing
                    #[derive(Deserialize)]
                    struct LenientAgentOutput {
                        #[serde(alias = "action", alias = "move")]
                        move_action: String,
                        next_state: serde_json::Value,
                    }

                    let lenient: Result<LenientAgentOutput, _> = serde_json::from_str(content);
                    match lenient {
                        Ok(output) => {
                            if output.next_state.is_null() {
                                let msg = "next_state field is null".to_string();
                                warn!(content_hash = %hash_content(content), "Schema violation: {}", msg);
                                return Err(RedFlag::FormatViolation { message: msg });
                            }
                            return Ok(StrictAgentOutput {
                                move_action: output.move_action,
                                next_state: output.next_state,
                            });
                        }
                        Err(e2) => {
                            let msg = format!("Invalid agent output format: {}", e2);
                            warn!(content_hash = %hash_content(content), "Schema violation: {}", msg);
                            return Err(RedFlag::FormatViolation { message: msg });
                        }
                    }
                }
            }

            // Generic parse error
            let msg = format!("Invalid agent output JSON: {}", error_msg);
            warn!(content_hash = %hash_content(content), "Schema violation: {}", msg);
            Err(RedFlag::FormatViolation { message: msg })
        }
    }
}

/// Hash content for secure logging (don't log full content for privacy)
fn hash_content(content: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Red flag indicating why a response was rejected
#[derive(Debug, Clone, PartialEq)]
pub enum RedFlag {
    /// Response exceeded the maximum token/character limit
    TokenLengthExceeded {
        /// Actual length of the response
        actual: usize,
        /// Configured limit
        limit: usize,
    },
    /// Response didn't match expected format/schema
    FormatViolation {
        /// Description of the violation
        message: String,
    },
    /// Response contains repetitive/looping content (future)
    LogicLoop {
        /// Description of the detected loop
        pattern: String,
    },
}

impl std::fmt::Display for RedFlag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RedFlag::TokenLengthExceeded { actual, limit } => {
                write!(f, "Token length exceeded: {} > {} limit", actual, limit)
            }
            RedFlag::FormatViolation { message } => {
                write!(f, "Format violation: {}", message)
            }
            RedFlag::LogicLoop { pattern } => {
                write!(f, "Logic loop detected: {}", pattern)
            }
        }
    }
}

impl std::error::Error for RedFlag {}

/// Result of validation - either the validated content or a red flag
pub type ValidationResult<T> = Result<T, RedFlag>;

/// Validate that content doesn't exceed the token/character limit
///
/// In MAKER, responses exceeding ~700 tokens often indicate pathological
/// reasoning or confusion loops.
///
/// # Arguments
///
/// * `content` - The response content to validate
/// * `limit` - Maximum allowed length (in characters/tokens)
///
/// # Returns
///
/// * `Ok(())` if content is within limit
/// * `Err(RedFlag::TokenLengthExceeded)` if content exceeds limit
///
/// # Example
///
/// ```
/// use maker::core::redflag::validate_token_length;
///
/// assert!(validate_token_length("short response", 700).is_ok());
/// assert!(validate_token_length("x".repeat(701).as_str(), 700).is_err());
/// ```
pub fn validate_token_length(content: &str, limit: usize) -> ValidationResult<()> {
    let actual = content.len();
    if actual > limit {
        Err(RedFlag::TokenLengthExceeded { actual, limit })
    } else {
        Ok(())
    }
}

/// Validate and parse content as JSON matching a schema
///
/// This performs strict schema validation - any missing required fields
/// or malformed JSON triggers a red flag.
///
/// # Type Parameters
///
/// * `T` - The expected type (must implement `DeserializeOwned`)
///
/// # Arguments
///
/// * `content` - JSON string to validate and parse
///
/// # Returns
///
/// * `Ok(T)` if content parses successfully
/// * `Err(RedFlag::FormatViolation)` if parsing fails
///
/// # Example
///
/// ```
/// use maker::core::redflag::validate_json_schema;
/// use serde::Deserialize;
///
/// #[derive(Deserialize)]
/// struct Move {
///     action: String,
///     next_state: String,
/// }
///
/// let valid = r#"{"action": "move disk", "next_state": "[[1,2],[3],[]]"}"#;
/// assert!(validate_json_schema::<Move>(valid).is_ok());
///
/// let missing_field = r#"{"action": "move disk"}"#;
/// assert!(validate_json_schema::<Move>(missing_field).is_err());
/// ```
pub fn validate_json_schema<T: DeserializeOwned>(content: &str) -> ValidationResult<T> {
    serde_json::from_str(content).map_err(|e| RedFlag::FormatViolation {
        message: e.to_string(),
    })
}

/// Validate content against multiple rules, collecting all violations
///
/// Unlike individual validators that fail fast, this collects all red flags
/// for comprehensive error reporting.
///
/// # Arguments
///
/// * `content` - The content to validate
/// * `token_limit` - Optional maximum token/character limit
///
/// # Returns
///
/// Vector of all triggered red flags (empty if valid)
pub fn validate_all(content: &str, token_limit: Option<usize>) -> Vec<RedFlag> {
    let mut flags = Vec::new();

    if let Some(limit) = token_limit {
        if let Err(flag) = validate_token_length(content, limit) {
            flags.push(flag);
        }
    }

    // Future: Add logic loop detection here

    flags
}

/// Red-flag validator with configurable rules
///
/// Provides a builder pattern for configuring validation rules.
#[derive(Debug, Clone)]
pub struct RedFlagValidator {
    /// Maximum token/character limit (None = no limit)
    pub token_limit: Option<usize>,
    /// Whether to check for logic loops (future)
    pub check_logic_loops: bool,
}

impl Default for RedFlagValidator {
    fn default() -> Self {
        Self {
            token_limit: Some(700), // Default from paper (gpt-4.1-mini)
            check_logic_loops: false,
        }
    }
}

impl RedFlagValidator {
    /// Create a new validator with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the token limit
    pub fn with_token_limit(mut self, limit: usize) -> Self {
        self.token_limit = Some(limit);
        self
    }

    /// Disable token limit checking
    pub fn without_token_limit(mut self) -> Self {
        self.token_limit = None;
        self
    }

    /// Enable logic loop detection (future)
    pub fn with_logic_loop_detection(mut self) -> Self {
        self.check_logic_loops = true;
        self
    }

    /// Validate content and return all triggered red flags
    pub fn validate(&self, content: &str) -> Vec<RedFlag> {
        validate_all(content, self.token_limit)
    }

    /// Validate content, returning Ok if no red flags, Err with first flag otherwise
    pub fn validate_strict(&self, content: &str) -> ValidationResult<()> {
        let flags = self.validate(content);
        if let Some(flag) = flags.into_iter().next() {
            Err(flag)
        } else {
            Ok(())
        }
    }

    /// Validate and parse JSON content
    pub fn validate_json<T: DeserializeOwned>(&self, content: &str) -> ValidationResult<T> {
        // First check token limit
        self.validate_strict(content)?;
        // Then parse JSON
        validate_json_schema(content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    // ==========================================
    // Token Length Validation Tests
    // ==========================================

    #[test]
    fn test_accepts_content_within_limit() {
        let content = "a".repeat(700);
        assert!(validate_token_length(&content, 700).is_ok());
    }

    #[test]
    fn test_accepts_content_under_limit() {
        let content = "short response";
        assert!(validate_token_length(content, 700).is_ok());
    }

    #[test]
    fn test_rejects_content_exceeding_limit() {
        let content = "a".repeat(701);
        let result = validate_token_length(&content, 700);
        assert!(result.is_err());

        match result.unwrap_err() {
            RedFlag::TokenLengthExceeded { actual, limit } => {
                assert_eq!(actual, 701);
                assert_eq!(limit, 700);
            }
            _ => panic!("Expected TokenLengthExceeded"),
        }
    }

    #[test]
    fn test_rejects_exactly_one_over_limit() {
        // Edge case: 701 tokens when limit is 700
        let content = "x".repeat(701);
        let result = validate_token_length(&content, 700);
        assert!(result.is_err());
    }

    #[test]
    fn test_accepts_empty_content() {
        assert!(validate_token_length("", 700).is_ok());
    }

    // ==========================================
    // JSON Schema Validation Tests
    // ==========================================

    #[derive(Debug, Deserialize, PartialEq)]
    struct TestMove {
        action: String,
        next_state: String,
    }

    #[test]
    fn test_accepts_valid_json() {
        let json = r#"{"action": "move disk 1 from A to C", "next_state": "[[2,3],[],[1]]"}"#;
        let result: ValidationResult<TestMove> = validate_json_schema(json);
        assert!(result.is_ok());

        let parsed = result.unwrap();
        assert_eq!(parsed.action, "move disk 1 from A to C");
        assert_eq!(parsed.next_state, "[[2,3],[],[1]]");
    }

    #[test]
    fn test_rejects_missing_required_field() {
        let json = r#"{"action": "move disk"}"#;
        let result: ValidationResult<TestMove> = validate_json_schema(json);
        assert!(result.is_err());

        match result.unwrap_err() {
            RedFlag::FormatViolation { message } => {
                assert!(message.contains("next_state") || message.contains("missing field"));
            }
            _ => panic!("Expected FormatViolation"),
        }
    }

    #[test]
    fn test_rejects_invalid_json_syntax() {
        let json = r#"{"action": "move disk", "next_state": }"#; // Invalid JSON
        let result: ValidationResult<TestMove> = validate_json_schema(json);
        assert!(result.is_err());

        match result.unwrap_err() {
            RedFlag::FormatViolation { message } => {
                assert!(!message.is_empty());
            }
            _ => panic!("Expected FormatViolation"),
        }
    }

    #[test]
    fn test_rejects_wrong_type() {
        let json = r#"{"action": 123, "next_state": "state"}"#; // action should be string
        let result: ValidationResult<TestMove> = validate_json_schema(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_accepts_valid_with_extra_fields() {
        // By default, serde allows extra fields
        let json = r#"{"action": "move", "next_state": "state", "extra": "ignored"}"#;
        let result: ValidationResult<TestMove> = validate_json_schema(json);
        assert!(result.is_ok());
    }

    // ==========================================
    // RedFlagValidator Tests
    // ==========================================

    #[test]
    fn test_validator_default_token_limit() {
        let validator = RedFlagValidator::new();
        assert_eq!(validator.token_limit, Some(700));
    }

    #[test]
    fn test_validator_custom_token_limit() {
        let validator = RedFlagValidator::new().with_token_limit(1000);
        assert_eq!(validator.token_limit, Some(1000));
    }

    #[test]
    fn test_validator_no_token_limit() {
        let validator = RedFlagValidator::new().without_token_limit();
        assert_eq!(validator.token_limit, None);

        // Should accept any length
        let long_content = "x".repeat(10000);
        assert!(validator.validate(&long_content).is_empty());
    }

    #[test]
    fn test_validator_validate_returns_all_flags() {
        let validator = RedFlagValidator::new().with_token_limit(10);
        let content = "x".repeat(20);

        let flags = validator.validate(&content);
        assert_eq!(flags.len(), 1);
        assert!(matches!(flags[0], RedFlag::TokenLengthExceeded { .. }));
    }

    #[test]
    fn test_validator_validate_json_checks_both() {
        let validator = RedFlagValidator::new().with_token_limit(10);

        // Content exceeds limit but is valid JSON
        let json = r#"{"action": "very long action text", "next_state": "state"}"#;
        let result: ValidationResult<TestMove> = validator.validate_json(json);

        // Should fail on token limit first
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RedFlag::TokenLengthExceeded { .. }
        ));
    }

    #[test]
    fn test_validator_validate_json_valid() {
        let validator = RedFlagValidator::new().with_token_limit(1000);

        let json = r#"{"action": "move", "next_state": "state"}"#;
        let result: ValidationResult<TestMove> = validator.validate_json(json);

        assert!(result.is_ok());
    }

    // ==========================================
    // Display Tests
    // ==========================================

    #[test]
    fn test_red_flag_display() {
        let flag = RedFlag::TokenLengthExceeded {
            actual: 800,
            limit: 700,
        };
        assert_eq!(flag.to_string(), "Token length exceeded: 800 > 700 limit");

        let flag = RedFlag::FormatViolation {
            message: "missing field".to_string(),
        };
        assert_eq!(flag.to_string(), "Format violation: missing field");

        let flag = RedFlag::LogicLoop {
            pattern: "repeated phrase".to_string(),
        };
        assert_eq!(flag.to_string(), "Logic loop detected: repeated phrase");
    }

    // ==========================================
    // Agent Output Schema Enforcement Tests (STORY-008-01)
    // ==========================================

    #[test]
    fn test_agent_output_valid_strict() {
        let json =
            r#"{"move_action": "move disk 1 from A to C", "next_state": {"rods": [[2,3],[],[1]]}}"#;
        let result = validate_agent_output(json, true);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.move_action, "move disk 1 from A to C");
    }

    #[test]
    fn test_agent_output_accepts_action_alias() {
        // "action" should be accepted as alias for "move_action"
        let json = r#"{"action": "move disk", "next_state": {"step": 1}}"#;
        let result = validate_agent_output(json, true);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().move_action, "move disk");
    }

    #[test]
    fn test_agent_output_accepts_move_alias() {
        // "move" should be accepted as alias for "move_action"
        let json = r#"{"move": "disk 1 A->C", "next_state": [1,2,3]}"#;
        let result = validate_agent_output(json, true);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().move_action, "disk 1 A->C");
    }

    #[test]
    fn test_agent_output_missing_next_state() {
        let json = r#"{"move_action": "move disk"}"#;
        let result = validate_agent_output(json, true);
        assert!(result.is_err());

        match result.unwrap_err() {
            RedFlag::FormatViolation { message } => {
                assert!(message.contains("next_state") || message.contains("required field"));
            }
            _ => panic!("Expected FormatViolation"),
        }
    }

    #[test]
    fn test_agent_output_missing_move_action() {
        let json = r#"{"next_state": {"step": 1}}"#;
        let result = validate_agent_output(json, true);
        assert!(result.is_err());

        match result.unwrap_err() {
            RedFlag::FormatViolation { message } => {
                assert!(message.contains("move_action") || message.contains("required field"));
            }
            _ => panic!("Expected FormatViolation"),
        }
    }

    #[test]
    fn test_agent_output_null_next_state() {
        let json = r#"{"move_action": "test", "next_state": null}"#;
        let result = validate_agent_output(json, true);
        assert!(result.is_err());

        match result.unwrap_err() {
            RedFlag::FormatViolation { message } => {
                assert!(message.contains("null"));
            }
            _ => panic!("Expected FormatViolation"),
        }
    }

    #[test]
    fn test_agent_output_strict_rejects_extra_fields() {
        let json = r#"{"move_action": "test", "next_state": {}, "malicious_field": "injection"}"#;
        let result = validate_agent_output(json, true);
        assert!(result.is_err());

        match result.unwrap_err() {
            RedFlag::FormatViolation { message } => {
                assert!(message.contains("unexpected") || message.contains("unknown"));
            }
            _ => panic!("Expected FormatViolation"),
        }
    }

    #[test]
    fn test_agent_output_lenient_allows_extra_fields() {
        let json = r#"{"move_action": "test", "next_state": {"valid": true}, "extra": "allowed"}"#;
        let result = validate_agent_output(json, false);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.move_action, "test");
    }

    #[test]
    fn test_agent_output_invalid_json() {
        let json = r#"{"move_action": "test", next_state: broken}"#;
        let result = validate_agent_output(json, true);
        assert!(result.is_err());

        match result.unwrap_err() {
            RedFlag::FormatViolation { message } => {
                assert!(message.contains("JSON") || message.contains("parse"));
            }
            _ => panic!("Expected FormatViolation"),
        }
    }

    #[test]
    fn test_agent_output_wrong_type() {
        // move_action must be string, not number
        let json = r#"{"move_action": 123, "next_state": {}}"#;
        let result = validate_agent_output(json, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_hash_content_deterministic() {
        let content = "test content";
        let hash1 = hash_content(content);
        let hash2 = hash_content(content);
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 16); // 16 hex chars = 64 bits
    }
}
