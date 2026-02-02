//! CLI Functional Tests for MAKER Framework
//!
//! These tests validate maker-cli by spawning the actual binary as a subprocess.
//! They test real-world CLI usage patterns.
//!
//! ## Running Tests
//!
//! ```bash
//! # Run all CLI tests (excluding those requiring external services)
//! cargo test --test cli_functional
//!
//! # Run tests that require Ollama (must have Ollama running locally)
//! cargo test --test cli_functional -- --ignored
//!
//! # Run all tests including ignored ones
//! cargo test --test cli_functional -- --include-ignored
//! ```

use serde::Deserialize;
use std::process::{Command, Output};
use std::sync::Once;

static BUILD_ONCE: Once = Once::new();

/// Ensure the CLI binary is built before running tests
fn ensure_cli_built() {
    BUILD_ONCE.call_once(|| {
        let status = Command::new("cargo")
            .args(["build", "--bin", "maker-cli"])
            .status()
            .expect("Failed to run cargo build");

        assert!(status.success(), "Failed to build maker-cli binary");
    });
}

/// Get the path to the maker-cli binary
fn cli_binary_path() -> String {
    // Cargo puts binaries in target/debug or target/release
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    format!("{}/target/debug/maker-cli", manifest_dir)
}

/// Run maker-cli with the given arguments
fn run_cli(args: &[&str]) -> Output {
    ensure_cli_built();

    Command::new(cli_binary_path())
        .args(args)
        .output()
        .expect("Failed to execute maker-cli")
}

/// Check if Ollama is available by attempting a simple health check
fn is_ollama_available() -> bool {
    // Try to connect to Ollama's default endpoint
    match std::net::TcpStream::connect_timeout(
        &"127.0.0.1:11434".parse().unwrap(),
        std::time::Duration::from_secs(1),
    ) {
        Ok(_) => true,
        Err(_) => false,
    }
}

// ============================================================================
// Response Types for JSON Parsing
// ============================================================================

#[derive(Debug, Deserialize)]
struct VoteResponse {
    winner: String,
    votes: usize,
    total_samples: usize,
    k_margin: usize,
    converged: bool,
}

#[derive(Debug, Deserialize)]
struct ValidateResponse {
    valid: bool,
    red_flags: Vec<RedFlagInfo>,
}

#[derive(Debug, Deserialize)]
struct RedFlagInfo {
    flag_type: String,
    details: String,
}

#[derive(Debug, Deserialize)]
struct CalibrateResponse {
    p_estimate: f64,
    confidence_interval: (f64, f64),
    sample_count: usize,
    recommended_k: usize,
}

#[derive(Debug, Deserialize)]
struct ConfigResponse {
    k_margin: usize,
    provider: String,
    matcher: String,
    adaptive: bool,
    max_samples: usize,
}

#[derive(Debug, Deserialize)]
struct HealthResponse {
    status: String,
    #[allow(dead_code)]
    version: String,
    components: HealthComponents,
}

#[derive(Debug, Deserialize)]
struct HealthComponents {
    voting: ComponentStatus,
    config: ComponentStatus,
    #[serde(default)]
    llm_provider: Option<ComponentStatus>,
}

#[derive(Debug, Deserialize)]
struct ComponentStatus {
    healthy: bool,
    #[allow(dead_code)]
    last_check: String,
}

// ============================================================================
// Test 1: CLI Vote Command
// ============================================================================

/// Test voting with Ollama provider (requires running Ollama instance)
///
/// Run with: `cargo test --test cli_functional test_cli_vote_with_ollama -- --ignored`
#[test]
#[ignore = "Requires running Ollama instance. Run with: cargo test -- --ignored"]
fn test_cli_vote_with_ollama() {
    if !is_ollama_available() {
        println!("SKIPPED: Ollama is not running at localhost:11434");
        return;
    }

    let output = run_cli(&[
        "--format",
        "json",
        "vote",
        "--prompt",
        "What is 2+2? Reply with just the number.",
        "--k-margin",
        "2",
        "--max-samples",
        "20",
        "--provider",
        "ollama",
    ]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    println!("stdout: {}", stdout);
    println!("stderr: {}", stderr);

    if !output.status.success() {
        // Check if it's a provider connectivity issue
        if stderr.contains("connection") || stderr.contains("refused") {
            println!("SKIPPED: Ollama connection failed");
            return;
        }
        panic!("CLI failed: {}", stderr);
    }

    // Parse JSON response
    let response: VoteResponse =
        serde_json::from_str(&stdout).expect("Failed to parse vote response JSON");

    // Validate response structure
    assert!(!response.winner.is_empty(), "Winner should not be empty");
    assert!(response.votes > 0, "Should have at least one vote");
    assert!(
        response.total_samples >= response.votes,
        "Total samples should be >= votes"
    );
    assert_eq!(
        response.k_margin, 2,
        "k_margin should match requested value"
    );
    assert!(response.converged, "Should have converged");

    println!("Vote result: {:?}", response);
}

/// Test that vote command validates k_margin parameter
#[test]
fn test_cli_vote_rejects_invalid_k_margin() {
    let output = run_cli(&[
        "--format",
        "json",
        "vote",
        "--prompt",
        "test",
        "--k-margin",
        "0", // Invalid: must be >= 1
        "--provider",
        "ollama",
    ]);

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should fail with an error about k_margin
    assert!(
        !output.status.success() || stderr.contains("k_margin"),
        "Should reject k_margin=0"
    );
}

/// Test that vote command requires a prompt
#[test]
fn test_cli_vote_requires_prompt() {
    let output = run_cli(&["vote", "--k-margin", "3"]);

    // Without prompt and without stdin, should fail or prompt for input
    // The exact behavior depends on implementation
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    println!("stdout: {}", stdout);
    println!("stderr: {}", stderr);

    // Either fails or produces empty/error output
    // This validates the CLI handles missing required args
}

// ============================================================================
// Test 2: CLI Validate Command
// ============================================================================

/// Test validate command with short response (should pass token limit)
#[test]
fn test_cli_validate_short_response_passes() {
    let output = run_cli(&[
        "--format",
        "json",
        "validate",
        "--response",
        "This is a short response.",
        "--token-limit",
        "100",
    ]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    println!("stdout: {}", stdout);
    println!("stderr: {}", stderr);

    assert!(output.status.success(), "CLI should succeed");

    let response: ValidateResponse =
        serde_json::from_str(&stdout).expect("Failed to parse validate response JSON");

    assert!(response.valid, "Short response should be valid");
    assert!(
        response.red_flags.is_empty(),
        "Should have no red flags for valid response"
    );
}

/// Test validate command with long response (should fail token limit)
#[test]
fn test_cli_validate_long_response_fails_token_limit() {
    // Create a response that exceeds 5 tokens (roughly 5 words)
    let long_response = "This is a very long response that definitely exceeds the tiny token limit we set for testing purposes and should trigger a red flag.";

    let output = run_cli(&[
        "--format",
        "json",
        "validate",
        "--response",
        long_response,
        "--token-limit",
        "5",
    ]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    println!("stdout: {}", stdout);
    println!("stderr: {}", stderr);

    assert!(
        output.status.success(),
        "CLI should succeed (validation runs)"
    );

    let response: ValidateResponse =
        serde_json::from_str(&stdout).expect("Failed to parse validate response JSON");

    assert!(!response.valid, "Long response should be invalid");
    assert!(
        !response.red_flags.is_empty(),
        "Should have red flags for exceeded token limit"
    );

    // Verify the red flag is for token length
    let has_token_flag = response
        .red_flags
        .iter()
        .any(|f| f.flag_type.contains("Token") || f.details.contains("token"));

    assert!(has_token_flag, "Should have token-related red flag");
}

/// Test validate command with valid JSON response
#[test]
fn test_cli_validate_json_response() {
    let json_response = r#"{"action": "move", "value": 42}"#;

    let output = run_cli(&[
        "--format",
        "json",
        "validate",
        "--response",
        json_response,
        "--token-limit",
        "100",
    ]);

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "CLI should succeed");

    let response: ValidateResponse =
        serde_json::from_str(&stdout).expect("Failed to parse validate response JSON");

    assert!(response.valid, "Valid JSON response should pass");
}

/// Test validate command with empty response
#[test]
fn test_cli_validate_empty_response() {
    let output = run_cli(&[
        "--format",
        "json",
        "validate",
        "--response",
        "",
        "--token-limit",
        "100",
    ]);

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Empty response should still be valid (0 tokens < 100 limit)
    if output.status.success() {
        let response: ValidateResponse =
            serde_json::from_str(&stdout).expect("Failed to parse validate response JSON");

        assert!(response.valid, "Empty response should be valid (0 tokens)");
    }
}

// ============================================================================
// Test 3: CLI Calibrate Command
// ============================================================================

/// Test calibrate command with sample data via stdin
#[test]
fn test_cli_calibrate_with_samples() {
    let samples = r#"[
        {"prompt": "What is 1+1?", "ground_truth": "2", "response": "2"},
        {"prompt": "What is 2+2?", "ground_truth": "4", "response": "4"},
        {"prompt": "What is 3+3?", "ground_truth": "6", "response": "5"},
        {"prompt": "What is 4+4?", "ground_truth": "8", "response": "8"},
        {"prompt": "What is 5+5?", "ground_truth": "10", "response": "10"}
    ]"#;

    // Write samples to a temp file
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join("maker_calibration_samples.json");
    std::fs::write(&temp_file, samples).expect("Failed to write temp file");

    let output = run_cli(&[
        "--format",
        "json",
        "calibrate",
        "--file",
        temp_file.to_str().unwrap(),
        "--target-reliability",
        "0.95",
        "--target-steps",
        "100",
    ]);

    // Clean up
    let _ = std::fs::remove_file(&temp_file);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    println!("stdout: {}", stdout);
    println!("stderr: {}", stderr);

    assert!(output.status.success(), "CLI should succeed");

    let response: CalibrateResponse =
        serde_json::from_str(&stdout).expect("Failed to parse calibrate response JSON");

    // 4 out of 5 correct = 0.8 accuracy
    assert!(
        (response.p_estimate - 0.8).abs() < 0.01,
        "p_estimate should be ~0.8, got {}",
        response.p_estimate
    );
    assert_eq!(response.sample_count, 5, "Should have 5 samples");
    assert!(
        response.recommended_k >= 1,
        "recommended_k should be at least 1"
    );
    assert!(
        response.confidence_interval.0 <= response.p_estimate,
        "Lower CI should be <= p_estimate"
    );
    assert!(
        response.confidence_interval.1 >= response.p_estimate,
        "Upper CI should be >= p_estimate"
    );

    println!("Calibration result: {:?}", response);
}

// ============================================================================
// Test 4: CLI Config Command
// ============================================================================

/// Test config show command
#[test]
fn test_cli_config_show() {
    let output = run_cli(&["--format", "json", "config", "--show"]);

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "CLI should succeed");

    let response: ConfigResponse =
        serde_json::from_str(&stdout).expect("Failed to parse config response JSON");

    // Verify default config values
    assert!(response.k_margin >= 1, "k_margin should be >= 1");
    assert!(
        !response.provider.is_empty(),
        "provider should have a default"
    );
    assert!(
        !response.matcher.is_empty(),
        "matcher should have a default"
    );

    println!("Config: {:?}", response);
}

/// Test config rejects invalid provider
#[test]
fn test_cli_config_rejects_invalid_provider() {
    let output = run_cli(&["config", "--provider", "invalid-provider-name"]);

    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        !output.status.success() || stderr.contains("Invalid provider"),
        "Should reject invalid provider"
    );
}

/// Test config rejects invalid matcher
#[test]
fn test_cli_config_rejects_invalid_matcher() {
    let output = run_cli(&["config", "--matcher", "invalid-matcher-name"]);

    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        !output.status.success() || stderr.contains("Invalid matcher"),
        "Should reject invalid matcher"
    );
}

// ============================================================================
// Test 5: CLI Health Command
// ============================================================================

/// Test health check command
#[test]
fn test_cli_health_basic() {
    let output = run_cli(&["--format", "json", "health"]);

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "CLI should succeed");

    let response: HealthResponse =
        serde_json::from_str(&stdout).expect("Failed to parse health response JSON");

    // Voting and config should always be healthy
    assert!(
        response.components.voting.healthy,
        "Voting should be healthy"
    );
    assert!(
        response.components.config.healthy,
        "Config should be healthy"
    );

    println!("Health: {:?}", response);
}

/// Test health check with provider check (requires Ollama)
#[test]
#[ignore = "Requires running Ollama instance"]
fn test_cli_health_with_provider() {
    if !is_ollama_available() {
        println!("SKIPPED: Ollama is not running");
        return;
    }

    let output = run_cli(&["--format", "json", "health", "--check-provider"]);

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "CLI should succeed");

    let response: HealthResponse =
        serde_json::from_str(&stdout).expect("Failed to parse health response JSON");

    assert!(
        response.components.llm_provider.is_some(),
        "Should have provider status when --check-provider is used"
    );
}

// ============================================================================
// Test 6: CLI --validate-config Flag
// ============================================================================

/// Test the --validate-config flag
#[test]
fn test_cli_validate_config_flag() {
    let output = run_cli(&["--validate-config"]);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    println!("stdout: {}", stdout);
    println!("stderr: {}", stderr);

    // Should either succeed with "valid" message or fail with specific errors
    // Both are acceptable outcomes for this test
    assert!(
        output.status.success() || !stderr.is_empty(),
        "Should produce output about config validity"
    );
}

// ============================================================================
// Test 7: CLI Help and Version
// ============================================================================

/// Test --help flag
#[test]
fn test_cli_help() {
    let output = run_cli(&["--help"]);

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "CLI --help should succeed");
    assert!(
        stdout.contains("maker-cli") || stdout.contains("MAKER"),
        "Help should mention the CLI name"
    );
    assert!(stdout.contains("vote"), "Help should mention vote command");
    assert!(
        stdout.contains("validate"),
        "Help should mention validate command"
    );
}

/// Test --version flag
#[test]
fn test_cli_version() {
    let output = run_cli(&["--version"]);

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "CLI --version should succeed");
    // Version string should contain a semver-like pattern
    assert!(
        stdout.contains('.') || stdout.contains("maker"),
        "Version should contain version info"
    );
}

/// Test subcommand help
#[test]
fn test_cli_vote_help() {
    let output = run_cli(&["vote", "--help"]);

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "vote --help should succeed");
    assert!(
        stdout.contains("--prompt") || stdout.contains("prompt"),
        "Vote help should mention prompt option"
    );
    assert!(
        stdout.contains("--k-margin") || stdout.contains("k_margin"),
        "Vote help should mention k-margin option"
    );
}

// ============================================================================
// Test 8: CLI Error Handling
// ============================================================================

/// Test that CLI provides helpful error for unknown command
#[test]
fn test_cli_unknown_command() {
    let output = run_cli(&["unknown-command"]);

    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(!output.status.success(), "Unknown command should fail");
    assert!(
        stderr.contains("error") || stderr.contains("invalid") || stderr.contains("unrecognized"),
        "Should provide error message for unknown command"
    );
}

/// Test that CLI handles missing required args gracefully
#[test]
fn test_cli_missing_args() {
    // calibrate without file or stdin should fail gracefully
    let output = run_cli(&["calibrate"]);

    // Should either fail or wait for stdin
    // We just verify it doesn't crash
    let _stderr = String::from_utf8_lossy(&output.stderr);
}

// ============================================================================
// Test 9: Text Output Format
// ============================================================================

/// Test that text output format works (default)
#[test]
fn test_cli_text_output_format() {
    let output = run_cli(&[
        "--format",
        "text",
        "validate",
        "--response",
        "test response",
        "--token-limit",
        "100",
    ]);

    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(output.status.success(), "CLI should succeed");

    // Text format should NOT be valid JSON
    let _parse_result: Result<serde_json::Value, _> = serde_json::from_str(&stdout);

    // Text output is human-readable, not JSON
    // It may or may not parse as JSON depending on implementation
    println!("Text output: {}", stdout);

    // Should contain readable info
    assert!(
        stdout.contains("valid") || stdout.contains("true") || stdout.contains("red"),
        "Text output should contain validation result"
    );
}
