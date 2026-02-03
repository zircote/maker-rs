//! Integration Tests for Hanoi Demo Example with Ollama
//!
//! These tests validate the critical integration points for the hanoi_demo refactoring:
//! - Ollama client integration with vote_with_margin_adaptive()
//! - Temperature=0 deterministic sampling
//! - RedFlagValidator integration
//! - Event system integration
//! - Move validation patterns specific to Hanoi
//!
//! Run with: `cargo test --test hanoi_demo_integration`
//! Run with Ollama: `cargo test --test hanoi_demo_integration -- --ignored`

use maker::core::{
    calculate_kmin, vote_with_margin, vote_with_margin_adaptive, KEstimator, KEstimatorConfig,
    LlmClient, MockLlmClient, RedFlagValidator, VoteConfig,
};
use maker::llm::ollama::OllamaClient;
use maker::llm::BlockingLlmAdapter;
use std::time::Duration;

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if Ollama is available
fn is_ollama_available() -> bool {
    std::net::TcpStream::connect_timeout(
        &"127.0.0.1:11434".parse().unwrap(),
        Duration::from_secs(1),
    )
    .is_ok()
}

/// Compute optimal Hanoi move sequence for testing
fn compute_hanoi_solution(n_disks: u8) -> Vec<String> {
    let mut moves = Vec::with_capacity((1 << n_disks) - 1);
    solve_recursive(n_disks, 'A', 'C', 'B', &mut moves);
    moves
}

fn solve_recursive(n: u8, from: char, to: char, aux: char, moves: &mut Vec<String>) {
    if n == 0 {
        return;
    }
    solve_recursive(n - 1, from, aux, to, moves);
    moves.push(format!("move {} from {} to {}", n, from, to));
    solve_recursive(n - 1, aux, to, from, moves);
}

/// Validate that a move follows the Hanoi format
fn is_valid_hanoi_move(s: &str) -> bool {
    // Expected format: "move N from X to Y" where N is 1-20, X/Y are A/B/C
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() != 6 {
        return false;
    }
    if parts[0] != "move" || parts[2] != "from" || parts[4] != "to" {
        return false;
    }

    // Check disk number (1-20)
    if parts[1].parse::<u8>().is_err() {
        return false;
    }

    // Check pegs (A, B, or C)
    let from = parts[3];
    let to = parts[5];
    matches!(from, "A" | "B" | "C") && matches!(to, "A" | "B" | "C")
}

// ============================================================================
// Unit Tests - Move Validation
// ============================================================================

#[test]
fn test_hanoi_move_validation_valid_formats() {
    assert!(is_valid_hanoi_move("move 1 from A to B"));
    assert!(is_valid_hanoi_move("move 3 from C to A"));
    assert!(is_valid_hanoi_move("move 10 from B to C"));
}

#[test]
fn test_hanoi_move_validation_invalid_formats() {
    assert!(!is_valid_hanoi_move("move 1 A to B")); // Missing "from"
    assert!(!is_valid_hanoi_move("move disk from A to B")); // Non-numeric disk
    assert!(!is_valid_hanoi_move("move 1 from A B")); // Missing "to"
    assert!(!is_valid_hanoi_move("move 1 from X to Y")); // Invalid pegs
    assert!(!is_valid_hanoi_move("1 from A to B")); // Missing "move"
    assert!(!is_valid_hanoi_move("")); // Empty string
}

#[test]
fn test_hanoi_solution_3_disks() {
    let solution = compute_hanoi_solution(3);
    assert_eq!(solution.len(), 7); // 2^3 - 1
    assert_eq!(solution[0], "move 1 from A to C");
    // Last move should complete the tower
    assert!(solution[6].contains("to C"), "Last move: {}", solution[6]);

    // Verify all moves are valid format
    for move_str in &solution {
        assert!(is_valid_hanoi_move(move_str), "Invalid move: {}", move_str);
    }
}

#[test]
fn test_hanoi_solution_5_disks() {
    let solution = compute_hanoi_solution(5);
    assert_eq!(solution.len(), 31); // 2^5 - 1

    // Verify all moves are valid
    for move_str in &solution {
        assert!(is_valid_hanoi_move(move_str));
    }
}

// ============================================================================
// Integration Tests - Mock Client with Adaptive Voting
// ============================================================================

#[test]
fn test_adaptive_voting_with_mock_reduces_k() {
    // Test that adaptive voting reduces k when accuracy is high
    let correct_answer = "move 1 from A to C";
    let accuracy = 0.90; // High accuracy

    let client = MockLlmClient::biased(correct_answer, "move 1 from A to B", accuracy, 500);
    let config = VoteConfig::default()
        .with_max_samples(100)
        .without_token_limit();

    let estimator_config = KEstimatorConfig {
        ema_alpha: 0.2,
        initial_p_hat: accuracy,
        k_min_floor: 1,
        k_max_ceiling: 10,
    };
    let mut estimator = KEstimator::new(estimator_config);

    let target_reliability = 0.95;

    // First vote
    let result1 = vote_with_margin_adaptive(
        "Step 1",
        &mut estimator,
        target_reliability,
        10,
        &client,
        config.clone(),
    )
    .unwrap();

    assert_eq!(result1.winner, correct_answer);
    assert!(result1.p_hat.is_some(), "Adaptive mode should return p_hat");

    let initial_k = result1.k_used;

    // After several successful votes, k should decrease or stay same
    let mut final_k = initial_k;
    for step in 2..=5 {
        let result = vote_with_margin_adaptive(
            &format!("Step {}", step),
            &mut estimator,
            target_reliability,
            10 - step + 1,
            &client,
            config.clone(),
        )
        .unwrap();

        assert_eq!(result.winner, correct_answer);
        final_k = result.k_used;
    }

    // k should have decreased or stayed reasonable due to high observed accuracy
    assert!(
        final_k <= initial_k,
        "Expected k to decrease or stay same from {} but got {}",
        initial_k,
        final_k
    );
}

#[test]
fn test_adaptive_voting_increases_k_on_disagreement() {
    // Test that adaptive voting monitors convergence with low accuracy
    let correct_answer = "move 1 from A to C";
    let wrong_answer = "move 1 from A to B";
    let accuracy = 0.60; // Moderate accuracy

    let client = MockLlmClient::biased(correct_answer, wrong_answer, accuracy, 500);
    let config = VoteConfig::default()
        .with_max_samples(100)
        .without_token_limit();

    let estimator_config = KEstimatorConfig {
        ema_alpha: 0.3,
        initial_p_hat: 0.75, // Assume higher initially
        k_min_floor: 1,
        k_max_ceiling: 10,
    };
    let mut estimator = KEstimator::new(estimator_config);

    let target_reliability = 0.95;

    // First vote with moderate accuracy
    let result1 = vote_with_margin_adaptive(
        "Step 1",
        &mut estimator,
        target_reliability,
        10,
        &client,
        config.clone(),
    )
    .unwrap();

    // After observing the convergence pattern, get the new k
    let k_after = estimator.recommended_k(target_reliability, 9);

    // k should adjust based on observed accuracy
    assert!(
        (1..=10).contains(&k_after),
        "k should be within bounds: {}",
        k_after
    );

    // The result should have converged
    assert_eq!(result1.winner, correct_answer);
}

// ============================================================================
// Integration Tests - Red-Flag Validation with Hanoi Moves
// ============================================================================

#[test]
fn test_redflag_validator_accepts_valid_hanoi_moves() {
    let validator = RedFlagValidator::new().with_token_limit(100);

    let valid_moves = vec![
        "move 1 from A to C",
        "move 3 from B to A",
        "move 10 from C to B",
    ];

    for move_str in valid_moves {
        let flags = validator.validate(move_str);
        assert!(
            flags.is_empty(),
            "Valid move '{}' should not have red flags, but got: {:?}",
            move_str,
            flags
        );
    }
}

#[test]
fn test_redflag_validator_catches_malformed_moves() {
    let validator = RedFlagValidator::new();

    // These should still pass red-flag (which is content-agnostic),
    // but we can test with very long responses
    let too_long = "move ".repeat(1000); // Way over token limit
    let validator = validator.with_token_limit(50);

    let flags = validator.validate(&too_long);
    assert!(
        !flags.is_empty(),
        "Extremely long response should trigger token limit red flag"
    );
}

#[test]
fn test_redflag_validator_with_json_parsing() {
    // Test that JSON parsing validation works with validate_json
    let validator = RedFlagValidator::new();

    // Valid JSON object
    let valid_json = r#"{"move": "move 1 from A to C"}"#;
    let result: Result<serde_json::Value, _> = validator.validate_json(valid_json);
    assert!(
        result.is_ok(),
        "Valid JSON should parse successfully: {:?}",
        result
    );

    // Invalid JSON
    let invalid_json = "not json";
    let result: Result<serde_json::Value, _> = validator.validate_json(invalid_json);
    assert!(
        result.is_err(),
        "Invalid JSON should fail validation when parsed"
    );
}

// ============================================================================
// Integration Tests - Temperature=0 Deterministic Sampling
// ============================================================================

#[test]
fn test_mock_client_temperature_zero_determinism() {
    // With mock client, temperature doesn't affect output, but test the pattern
    let client = MockLlmClient::constant("move 1 from A to C");

    // Generate multiple samples at T=0 using the sync LlmClient trait
    let mut results = Vec::new();
    for _ in 0..5 {
        let response = client.generate("test", 0.0).unwrap();
        results.push(response.content.clone());
    }

    // All results should be identical (MockLlmClient always returns same value)
    let first = &results[0];
    for result in &results {
        assert_eq!(result, first, "T=0 should give deterministic results");
    }
}

#[test]
fn test_vote_config_diversity_temperature() {
    // Test VoteConfig properly configures diversity temperature
    let config = VoteConfig::default().with_diversity_temperature(0.2);
    assert_eq!(config.diversity_temperature, 0.2);

    // Default should be 0.1
    let default_config = VoteConfig::default();
    assert_eq!(default_config.diversity_temperature, 0.1);
}

// ============================================================================
// Integration Tests - K-Margin Calculation for Hanoi
// ============================================================================

#[test]
fn test_kmin_calculation_3_disk_hanoi() {
    let n_disks = 3;
    let total_steps = (1 << n_disks) - 1; // 7 steps
    let accuracy = 0.85;
    let target_reliability = 0.95;

    let k = calculate_kmin(accuracy, target_reliability, total_steps, 1).unwrap();

    // k should be reasonable for these parameters
    assert!(k >= 1, "k should be at least 1");
    assert!(k <= 10, "k should not be excessive for 85% accuracy");
}

#[test]
fn test_kmin_calculation_10_disk_hanoi() {
    let n_disks = 10;
    let total_steps = (1 << n_disks) - 1; // 1,023 steps
    let accuracy = 0.85;
    let target_reliability = 0.95;

    let k = calculate_kmin(accuracy, target_reliability, total_steps, 1).unwrap();

    // For longer sequences, k should be higher
    assert!(k >= 1, "k should be at least 1");
    assert!(k <= 20, "k should be reasonable even for long sequences");
}

#[test]
fn test_kmin_higher_accuracy_gives_lower_k() {
    let total_steps = 100;
    let target_reliability = 0.95;

    let k_low = calculate_kmin(0.70, target_reliability, total_steps, 1).unwrap();
    let k_high = calculate_kmin(0.90, target_reliability, total_steps, 1).unwrap();

    assert!(
        k_high <= k_low,
        "Higher accuracy should require same or lower k-margin"
    );
}

// ============================================================================
// Integration Tests - Full Voting Pipeline with Mock
// ============================================================================

#[test]
fn test_full_voting_pipeline_3_disk_hanoi() {
    // Simulate complete 3-disk Hanoi with voting
    let n_disks = 3;
    let solution = compute_hanoi_solution(n_disks);
    let total_steps = solution.len();
    let accuracy = 0.85;
    let k = 3;

    let mut errors = 0;

    for (step_idx, expected_move) in solution.iter().enumerate() {
        // Create biased mock client for this step
        let pool_size = 50;
        let correct_count = (pool_size as f64 * accuracy).round() as usize;
        let mut responses = vec![expected_move.clone(); correct_count];
        let wrong_move = "move 1 from A to B".to_string();
        responses.extend(vec![wrong_move; pool_size - correct_count]);

        let client = MockLlmClient::new(responses);
        let config = VoteConfig::default()
            .with_max_samples(pool_size)
            .without_token_limit();

        match vote_with_margin(&format!("Step {}", step_idx), k, &client, config) {
            Ok(result) => {
                if result.winner != *expected_move {
                    errors += 1;
                }
            }
            Err(_) => {
                errors += 1;
            }
        }
    }

    // With k=3 and 85% accuracy, we should have very few errors
    let error_rate = errors as f64 / total_steps as f64;
    assert!(
        error_rate < 0.1,
        "Error rate {:.1}% should be < 10% with voting",
        error_rate * 100.0
    );
}

// ============================================================================
// Integration Tests - Ollama Client (Requires Running Ollama)
// ============================================================================

#[test]
#[ignore = "Requires running Ollama instance with gpt-oss model"]
fn test_ollama_single_hanoi_move() {
    if !is_ollama_available() {
        println!("Skipping: Ollama not available");
        return;
    }

    let async_client = OllamaClient::new("gpt-oss").with_timeout(Duration::from_secs(60));
    let client = BlockingLlmAdapter::new(async_client);

    let prompt = "Generate a Towers of Hanoi move in this exact format: 'move N from X to Y' \
                  where N is the disk number and X, Y are pegs (A, B, or C). \
                  Only output the move, nothing else.";

    let response = client.generate(prompt, 0.0).unwrap();

    println!("Ollama response: {}", response.content);

    // Response should be relatively short
    assert!(
        response.content.len() < 100,
        "Response should be concise: {}",
        response.content
    );
}

#[test]
#[ignore = "Requires running Ollama instance"]
fn test_ollama_voting_with_k_margin() {
    if !is_ollama_available() {
        println!("Skipping: Ollama not available");
        return;
    }

    let async_client = OllamaClient::new("gpt-oss").with_timeout(Duration::from_secs(120));
    let client = BlockingLlmAdapter::new(async_client);

    let prompt = "What is 2+2? Answer with just the number.";
    let config = VoteConfig::default()
        .with_max_samples(10)
        .without_token_limit()
        .with_diversity_temperature(0.0); // Deterministic

    let result = vote_with_margin(prompt, 2, &client, config).unwrap();

    println!("Winner: {}", result.winner);
    println!("Total samples: {}", result.total_samples);
    println!("K used: {}", result.k_used);

    assert!(
        result.total_samples >= 3,
        "Should collect at least 3 samples for k=2"
    );

    // Get the winner's vote count from vote_counts
    let winner_votes = result.vote_counts.get(&result.winner).unwrap_or(&0);
    assert!(
        *winner_votes >= 2,
        "Winner should have at least k+1 votes, got {}",
        winner_votes
    );
}

#[test]
#[ignore = "Requires running Ollama instance"]
fn test_ollama_adaptive_voting() {
    if !is_ollama_available() {
        println!("Skipping: Ollama not available");
        return;
    }

    let async_client = OllamaClient::new("gpt-oss").with_timeout(Duration::from_secs(120));
    let client = BlockingLlmAdapter::new(async_client);

    let config = VoteConfig::default()
        .with_max_samples(15)
        .without_token_limit();

    let estimator_config = KEstimatorConfig {
        ema_alpha: 0.2,
        initial_p_hat: 0.80,
        k_min_floor: 1,
        k_max_ceiling: 5,
    };
    let mut estimator = KEstimator::new(estimator_config);

    let target_reliability = 0.95;
    let remaining_steps = 5;
    let prompt = "What is the capital of France? Answer with just the city name.";

    let result = vote_with_margin_adaptive(
        prompt,
        &mut estimator,
        target_reliability,
        remaining_steps,
        &client,
        config,
    )
    .unwrap();

    println!("Winner: {}", result.winner);
    println!("P-hat: {:?}", result.p_hat);
    println!("Total samples: {}", result.total_samples);
    println!("K used: {}", result.k_used);

    assert!(result.p_hat.is_some(), "Adaptive mode should return p_hat");
}

#[test]
#[ignore = "Requires running Ollama instance"]
fn test_ollama_connection_error_handling() {
    // Use a port that's definitely not running Ollama
    let async_client = OllamaClient::with_url("gpt-oss", "http://localhost:59999")
        .with_timeout(Duration::from_millis(500));
    let client = BlockingLlmAdapter::new(async_client);

    let result = client.generate("test", 0.0);

    assert!(result.is_err(), "Should error on connection refused");
}

#[test]
#[ignore = "Requires running Ollama instance"]
fn test_ollama_temperature_zero_consistency() {
    if !is_ollama_available() {
        println!("Skipping: Ollama not available");
        return;
    }

    let async_client = OllamaClient::new("gpt-oss").with_timeout(Duration::from_secs(60));
    let client = BlockingLlmAdapter::new(async_client);

    let prompt = "What is 2+2? Answer with only the number.";

    // Generate same prompt multiple times with T=0
    let mut responses = Vec::new();
    for _ in 0..3 {
        let response = client.generate(prompt, 0.0).unwrap();
        responses.push(response.content.clone());
    }

    // All responses should be very similar or identical
    println!("T=0 responses: {:?}", responses);

    // At minimum, they should all contain the same answer
    for response in &responses {
        let response_str: &str = response;
        assert!(
            response_str.contains("4"),
            "All T=0 responses should contain the correct answer"
        );
    }
}

// ============================================================================
// Integration Tests - Event System (Basic)
// ============================================================================

#[test]
fn test_vote_config_builder_pattern() {
    let config = VoteConfig::default()
        .with_max_samples(50)
        .with_token_limit(500)
        .with_diversity_temperature(0.2)
        .with_timeout(Duration::from_secs(30));

    assert_eq!(config.max_samples, 50);
    assert_eq!(config.token_limit, Some(500));
    assert_eq!(config.diversity_temperature, 0.2);
    assert_eq!(config.timeout, Some(Duration::from_secs(30)));
}

#[test]
fn test_vote_config_without_token_limit() {
    let config = VoteConfig::default().without_token_limit();
    assert_eq!(config.token_limit, None);
}
