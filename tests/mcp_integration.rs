//! MCP Protocol Compliance Tests (STORY-005-03)
//!
//! Integration tests validating MCP protocol compliance and tool functionality.
//! These tests ensure interoperability with Claude Code and other MCP clients.

use maker::mcp::server::{MakerServer, ServerConfig};
use maker::mcp::tools::calibrate::CalibrationSample;
use maker::mcp::tools::validate::RedFlagInfo;
use maker::mcp::tools::{
    CalibrateRequest, CalibrateResponse, ConfigRequest, ConfigResponse, ValidateRequest,
    ValidateResponse, VoteRequest, VoteResponse,
};
use rmcp::ServerHandler;
use std::collections::HashMap;

// ============================================
// Server Lifecycle Tests
// ============================================

#[test]
fn test_server_initialization() {
    let server = MakerServer::new();
    let info = server.get_info();

    assert_eq!(info.server_info.name, "maker-mcp");
    assert!(!info.server_info.version.is_empty());
    assert!(info.capabilities.tools.is_some());
    assert!(info.instructions.is_some());
}

#[test]
fn test_server_custom_config() {
    let config = ServerConfig {
        k_default: 5,
        temperature_diversity: 0.2,
        token_limit: 1000,
        provider: "anthropic".to_string(),
        max_prompt_length: 5000,
    };

    let server = MakerServer::with_config(config.clone());

    // Verify server was created (further config access requires async)
    let info = server.get_info();
    assert_eq!(info.server_info.name, "maker-mcp");
}

#[tokio::test]
async fn test_server_config_persistence() {
    let server = MakerServer::new();

    // Get initial config
    let initial = server.get_config().await;
    assert_eq!(initial.k_default, 3);

    // Update config
    let mut updated = initial.clone();
    updated.k_default = 7;
    server.set_config(updated).await;

    // Verify persistence
    let retrieved = server.get_config().await;
    assert_eq!(retrieved.k_default, 7);
}

// ============================================
// Vote Tool Tests
// ============================================

#[test]
fn test_vote_request_schema_valid() {
    let json = r#"{
        "prompt": "What is 2+2?",
        "k_margin": 3,
        "max_samples": 50,
        "temperature_diversity": 0.1,
        "provider": "ollama"
    }"#;

    let request: VoteRequest = serde_json::from_str(json).unwrap();
    assert_eq!(request.prompt, "What is 2+2?");
    assert_eq!(request.k_margin, 3);
    assert_eq!(request.max_samples, Some(50));
}

#[test]
fn test_vote_request_schema_minimal() {
    let json = r#"{
        "prompt": "What is 2+2?",
        "k_margin": 3
    }"#;

    let request: VoteRequest = serde_json::from_str(json).unwrap();
    assert_eq!(request.prompt, "What is 2+2?");
    assert!(request.max_samples.is_none());
}

#[test]
fn test_vote_request_rejects_unknown_fields() {
    let json = r#"{
        "prompt": "test",
        "k_margin": 3,
        "malicious_field": "injection attempt"
    }"#;

    let result: Result<VoteRequest, _> = serde_json::from_str(json);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("unknown field"));
}

#[test]
fn test_vote_request_validation_k_margin_zero() {
    let request = VoteRequest {
        prompt: "test".to_string(),
        k_margin: 0,
        max_samples: None,
        temperature_diversity: None,
        provider: None,
    };

    assert!(request.validate().is_err());
}

#[test]
fn test_vote_request_validation_empty_prompt() {
    let request = VoteRequest {
        prompt: "".to_string(),
        k_margin: 3,
        max_samples: None,
        temperature_diversity: None,
        provider: None,
    };

    assert!(request.validate().is_err());
}

#[test]
fn test_vote_request_validation_prompt_too_long() {
    let request = VoteRequest {
        prompt: "x".repeat(10_001),
        k_margin: 3,
        max_samples: None,
        temperature_diversity: None,
        provider: None,
    };

    assert!(request.validate().is_err());
}

#[test]
fn test_vote_response_schema() {
    let mut vote_counts = HashMap::new();
    vote_counts.insert("answer_a".to_string(), 5);
    vote_counts.insert("answer_b".to_string(), 2);

    let response = VoteResponse {
        winner: "answer_a".to_string(),
        vote_counts,
        total_samples: 7,
        red_flags: 0,
        cost_tokens: 1500,
        cost_usd: 0.0015,
        latency_ms: 2500,
    };

    let json = serde_json::to_string(&response).unwrap();
    let parsed: VoteResponse = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.winner, "answer_a");
    assert_eq!(parsed.total_samples, 7);
}

// ============================================
// Validate Tool Tests
// ============================================

#[test]
fn test_validate_request_schema_valid() {
    let json = r#"{
        "response": "test response",
        "token_limit": 700,
        "schema": {"required": ["field1"]}
    }"#;

    let request: ValidateRequest = serde_json::from_str(json).unwrap();
    assert_eq!(request.response, "test response");
    assert_eq!(request.token_limit, Some(700));
}

#[test]
fn test_validate_request_minimal() {
    let json = r#"{"response": "test"}"#;

    let request: ValidateRequest = serde_json::from_str(json).unwrap();
    assert_eq!(request.response, "test");
    assert!(request.token_limit.is_none());
    assert!(request.schema.is_none());
}

#[test]
fn test_validate_request_rejects_unknown_fields() {
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
    assert!(json.contains("\"valid\":true"));
    assert!(json.contains("\"red_flags\":[]"));
}

#[test]
fn test_validate_response_with_flags() {
    let response = ValidateResponse {
        valid: false,
        red_flags: vec![RedFlagInfo {
            flag_type: "TokenLengthExceeded".to_string(),
            details: "750 > 700".to_string(),
        }],
    };

    let json = serde_json::to_string(&response).unwrap();
    let parsed: ValidateResponse = serde_json::from_str(&json).unwrap();

    assert!(!parsed.valid);
    assert_eq!(parsed.red_flags.len(), 1);
    assert_eq!(parsed.red_flags[0].flag_type, "TokenLengthExceeded");
}

// ============================================
// Calibrate Tool Tests
// ============================================

#[test]
fn test_calibrate_request_schema_valid() {
    let json = r#"{
        "samples": [
            {"prompt": "Q1", "ground_truth": "A1", "response": "A1"},
            {"prompt": "Q2", "ground_truth": "A2", "response": "A2"}
        ],
        "target_reliability": 0.95,
        "target_steps": 1000
    }"#;

    let request: CalibrateRequest = serde_json::from_str(json).unwrap();
    assert_eq!(request.samples.len(), 2);
    assert!((request.target_reliability - 0.95).abs() < f64::EPSILON);
}

#[test]
fn test_calibrate_request_defaults() {
    let json = r#"{"samples": []}"#;

    let request: CalibrateRequest = serde_json::from_str(json).unwrap();
    assert!((request.target_reliability - 0.95).abs() < f64::EPSILON);
    assert_eq!(request.target_steps, 1000);
}

#[test]
fn test_calibrate_request_rejects_unknown_fields() {
    let json = r#"{"samples": [], "unknown": true}"#;

    let result: Result<CalibrateRequest, _> = serde_json::from_str(json);
    assert!(result.is_err());
}

#[test]
fn test_calibrate_response_schema() {
    let response = CalibrateResponse {
        p_estimate: 0.85,
        confidence_interval: (0.78, 0.92),
        sample_count: 50,
        recommended_k: 4,
    };

    let json = serde_json::to_string(&response).unwrap();
    let parsed: CalibrateResponse = serde_json::from_str(&json).unwrap();

    assert!((parsed.p_estimate - 0.85).abs() < f64::EPSILON);
    assert_eq!(parsed.sample_count, 50);
    assert_eq!(parsed.recommended_k, 4);
}

#[test]
fn test_calibration_sample_schema() {
    let sample = CalibrationSample {
        prompt: "What is 2+2?".to_string(),
        ground_truth: "4".to_string(),
        response: Some("4".to_string()),
    };

    let json = serde_json::to_string(&sample).unwrap();
    let parsed: CalibrationSample = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.prompt, "What is 2+2?");
    assert_eq!(parsed.ground_truth, "4");
    assert_eq!(parsed.response, Some("4".to_string()));
}

// ============================================
// Configure Tool Tests
// ============================================

#[test]
fn test_configure_request_schema_valid() {
    let json = r#"{
        "k_default": 5,
        "temperature_diversity": 0.2,
        "token_limit": 1000,
        "provider": "openai"
    }"#;

    let request: ConfigRequest = serde_json::from_str(json).unwrap();
    assert_eq!(request.k_default, Some(5));
    assert_eq!(request.provider, Some("openai".to_string()));
}

#[test]
fn test_configure_request_partial() {
    let json = r#"{"k_default": 4}"#;

    let request: ConfigRequest = serde_json::from_str(json).unwrap();
    assert_eq!(request.k_default, Some(4));
    assert!(request.temperature_diversity.is_none());
    assert!(request.token_limit.is_none());
    assert!(request.provider.is_none());
}

#[test]
fn test_configure_request_empty() {
    let json = r#"{}"#;

    let request: ConfigRequest = serde_json::from_str(json).unwrap();
    assert!(request.k_default.is_none());
    assert!(!request.has_updates());
}

#[test]
fn test_configure_request_rejects_unknown_fields() {
    let json = r#"{"k_default": 5, "unknown": true}"#;

    let result: Result<ConfigRequest, _> = serde_json::from_str(json);
    assert!(result.is_err());
}

#[test]
fn test_configure_response_schema() {
    use maker::mcp::tools::configure::Config;

    let response = ConfigResponse {
        applied: true,
        current_config: Config {
            k_default: 5,
            temperature_diversity: 0.2,
            token_limit: 1000,
            provider: "openai".to_string(),
        },
    };

    let json = serde_json::to_string(&response).unwrap();
    let parsed: ConfigResponse = serde_json::from_str(&json).unwrap();

    assert!(parsed.applied);
    assert_eq!(parsed.current_config.k_default, 5);
}

// ============================================
// Schema Validation Comprehensive Tests
// ============================================

#[test]
fn test_all_requests_reject_null_required_fields() {
    // VoteRequest requires prompt and k_margin
    let vote_null_prompt: Result<VoteRequest, _> =
        serde_json::from_str(r#"{"prompt": null, "k_margin": 3}"#);
    assert!(vote_null_prompt.is_err());

    // ValidateRequest requires response
    let validate_null: Result<ValidateRequest, _> = serde_json::from_str(r#"{"response": null}"#);
    assert!(validate_null.is_err());

    // CalibrateRequest requires samples
    let calibrate_null: Result<CalibrateRequest, _> = serde_json::from_str(r#"{"samples": null}"#);
    assert!(calibrate_null.is_err());
}

#[test]
fn test_all_requests_reject_wrong_types() {
    // k_margin must be number
    let vote_wrong_type: Result<VoteRequest, _> =
        serde_json::from_str(r#"{"prompt": "test", "k_margin": "three"}"#);
    assert!(vote_wrong_type.is_err());

    // token_limit must be number
    let validate_wrong_type: Result<ValidateRequest, _> =
        serde_json::from_str(r#"{"response": "test", "token_limit": "seven hundred"}"#);
    assert!(validate_wrong_type.is_err());

    // samples must be array
    let calibrate_wrong_type: Result<CalibrateRequest, _> =
        serde_json::from_str(r#"{"samples": "not an array"}"#);
    assert!(calibrate_wrong_type.is_err());
}

#[test]
fn test_json_injection_prevention() {
    // Test various injection attempts
    let injections = vec![
        // Extra fields (rejected by deny_unknown_fields)
        r#"{"prompt": "test", "k_margin": 3, "__proto__": {}}"#,
        r#"{"prompt": "test", "k_margin": 3, "constructor": {}}"#,
        r#"{"response": "test", "prototype": {}}"#,
    ];

    for injection in injections {
        let vote_result: Result<VoteRequest, _> = serde_json::from_str(injection);
        let validate_result: Result<ValidateRequest, _> = serde_json::from_str(injection);

        // At least one should fail (depending on which request type)
        assert!(vote_result.is_err() || validate_result.is_err());
    }
}

// ============================================
// Tool Execution Integration Tests
// ============================================

#[test]
fn test_vote_tool_execution_with_mock() {
    use maker::mcp::tools::vote::execute_vote;

    let request = VoteRequest {
        prompt: "What is 2+2?".to_string(),
        k_margin: 3,
        max_samples: Some(50),
        temperature_diversity: None,
        provider: None,
    };

    let result = execute_vote(&request, 100, 0.1, Some(700));
    assert!(result.is_ok());

    let response = result.unwrap();
    assert!(!response.winner.is_empty());
    assert!(response.total_samples > 0);
}

#[test]
fn test_validate_tool_execution() {
    use maker::mcp::tools::validate::execute_validate;

    // Valid response
    let request = ValidateRequest {
        response: "short valid response".to_string(),
        token_limit: Some(700),
        schema: None,
    };

    let result = execute_validate(&request);
    assert!(result.valid);
    assert!(result.red_flags.is_empty());

    // Invalid response (too long)
    let long_request = ValidateRequest {
        response: "x".repeat(750),
        token_limit: Some(700),
        schema: None,
    };

    let long_result = execute_validate(&long_request);
    assert!(!long_result.valid);
    assert!(!long_result.red_flags.is_empty());
}

#[test]
fn test_calibrate_tool_execution() {
    use maker::mcp::tools::calibrate::execute_calibrate;

    // 8 correct out of 10 (p = 0.8)
    let mut samples = Vec::new();
    for i in 0..8 {
        samples.push(CalibrationSample {
            prompt: format!("Q{}", i),
            ground_truth: "correct".to_string(),
            response: Some("correct".to_string()),
        });
    }
    for i in 8..10 {
        samples.push(CalibrationSample {
            prompt: format!("Q{}", i),
            ground_truth: "correct".to_string(),
            response: Some("wrong".to_string()),
        });
    }

    let request = CalibrateRequest {
        samples,
        target_reliability: 0.95,
        target_steps: 1000,
    };

    let result = execute_calibrate(&request);
    assert!(result.is_ok());

    let response = result.unwrap();
    assert!((response.p_estimate - 0.8).abs() < 0.001);
    assert_eq!(response.sample_count, 10);
    assert!(response.recommended_k >= 1);
}

#[test]
fn test_configure_tool_execution() {
    use maker::mcp::tools::configure::apply_config_updates;

    let mut config = ServerConfig::default();
    let request = ConfigRequest {
        k_default: Some(5),
        temperature_diversity: Some(0.2),
        token_limit: Some(1000),
        provider: Some("openai".to_string()),
    };

    let applied = apply_config_updates(&mut config, &request);

    assert!(applied);
    assert_eq!(config.k_default, 5);
    assert!((config.temperature_diversity - 0.2).abs() < 0.001);
    assert_eq!(config.token_limit, 1000);
    assert_eq!(config.provider, "openai");
}

// ============================================
// Error Response Tests
// ============================================

#[test]
fn test_calibrate_error_no_samples() {
    use maker::mcp::tools::calibrate::{execute_calibrate, CalibrateError};

    let request = CalibrateRequest {
        samples: vec![],
        target_reliability: 0.95,
        target_steps: 1000,
    };

    let result = execute_calibrate(&request);
    assert_eq!(result, Err(CalibrateError::NoSamples));
}

#[test]
fn test_calibrate_error_missing_responses() {
    use maker::mcp::tools::calibrate::{execute_calibrate, CalibrateError};

    let request = CalibrateRequest {
        samples: vec![CalibrationSample {
            prompt: "Q1".to_string(),
            ground_truth: "A1".to_string(),
            response: None,
        }],
        target_reliability: 0.95,
        target_steps: 1000,
    };

    let result = execute_calibrate(&request);
    assert_eq!(result, Err(CalibrateError::MissingResponses));
}

#[test]
fn test_calibrate_error_low_accuracy() {
    use maker::mcp::tools::calibrate::{execute_calibrate, CalibrateError};

    // 3 correct out of 10 (p = 0.3 < 0.5)
    let mut samples = Vec::new();
    for i in 0..3 {
        samples.push(CalibrationSample {
            prompt: format!("Q{}", i),
            ground_truth: "correct".to_string(),
            response: Some("correct".to_string()),
        });
    }
    for i in 3..10 {
        samples.push(CalibrationSample {
            prompt: format!("Q{}", i),
            ground_truth: "correct".to_string(),
            response: Some("wrong".to_string()),
        });
    }

    let request = CalibrateRequest {
        samples,
        target_reliability: 0.95,
        target_steps: 1000,
    };

    let result = execute_calibrate(&request);
    assert!(matches!(
        result,
        Err(CalibrateError::InsufficientAccuracy { .. })
    ));
}
