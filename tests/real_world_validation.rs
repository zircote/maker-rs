//! Real-World Integration Tests for MAKER Framework
//!
//! These tests validate MAKER's core functionality using realistic scenarios
//! with mock LLM clients for offline CI execution.
//!
//! Test 1: Code validation with SPRT voting - demonstrates k-margin consensus
//! Test 2: Task decomposition and aggregation - demonstrates recursive workflow

use maker::core::decomposition::{
    AggregatorConfig, CompositionFunction, DecompositionAgent, DecompositionProposal,
    DecompositionSubtask, IdentityDecomposer, MergeStrategy, SolutionDiscriminator, SubtaskResult,
};
use maker::core::{
    calculate_kmin, validate_token_length, vote_with_margin, MockLlmClient, VoteConfig, VoteRace,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

// ============================================================================
// Test 1: Code Validation with SPRT Voting
// ============================================================================

/// Simulates voting on code solutions for "write a function to check if a number is prime"
///
/// This test demonstrates:
/// - SPRT-based voting mechanism reaching consensus
/// - k-margin voting dynamics
/// - Red-flag validation (token length)
/// - Winner selection with correct margin
#[test]
fn test_code_validation_with_voting() {
    // Define the "correct" solution that should win (single line to avoid whitespace issues)
    let correct_solution = "fn is_prime(n: u64) -> bool { if n < 2 { false } else if n == 2 { true } else { (2..=((n as f64).sqrt() as u64)).all(|i| n % i != 0) } }";

    // Define an "incorrect" solution (has a bug - returns true for 1)
    let incorrect_solution = "fn is_prime(n: u64) -> bool { n >= 1 }";

    // Calculate optimal k-margin for 85% model accuracy, 95% target reliability
    let p = 0.85; // Model accuracy
    let t = 0.95; // Target reliability
    let total_steps = 10; // Number of voting steps to simulate
    let k = calculate_kmin(p, t, total_steps, 1).expect("Valid k-min parameters");

    assert!(
        k >= 2,
        "k-margin should be at least 2 for meaningful voting"
    );
    println!("Calculated k-margin: {} for p={}, t={}", k, p, t);

    // Create biased mock client: 85% correct, 15% incorrect
    let client = MockLlmClient::biased(correct_solution, incorrect_solution, p, 100);

    // Configure voting
    let config = VoteConfig::default()
        .with_max_samples(100)
        .without_token_limit(); // Code can be long

    // Execute voting
    let prompt = "Write a Rust function to check if a number is prime";
    let result = vote_with_margin(prompt, k, &client, config);

    assert!(result.is_ok(), "Voting should succeed");
    let vote_result = result.unwrap();

    // Verify winner is the correct solution
    assert_eq!(
        vote_result.winner, correct_solution,
        "Winner should be the correct prime check implementation"
    );

    // Verify k-margin was achieved
    let winner_votes = vote_result
        .vote_counts
        .get(&vote_result.winner)
        .copied()
        .unwrap_or(0);
    let runner_up_votes = vote_result
        .vote_counts
        .values()
        .filter(|&&v| v != winner_votes)
        .max()
        .copied()
        .unwrap_or(0);
    let margin = winner_votes.saturating_sub(runner_up_votes);

    assert!(
        margin >= k,
        "Winner should lead by at least k={} votes, got margin={}",
        k,
        margin
    );

    // Verify vote counts are tracked
    assert!(
        vote_result.total_samples >= k,
        "Should have at least k samples"
    );

    println!("Voting results:");
    println!("  Winner votes: {}", winner_votes);
    println!("  Runner-up votes: {}", runner_up_votes);
    println!("  Margin: {} (required: {})", margin, k);
    println!("  Total samples: {}", vote_result.total_samples);
}

/// Test red-flag validation on code solutions
#[test]
fn test_code_validation_red_flags() {
    // Test 1: Token length validation
    let short_code = "fn f() {}";
    let long_code = "fn f() { ".to_string() + &"x + ".repeat(1000) + "}";

    assert!(
        validate_token_length(short_code, 100).is_ok(),
        "Short code should pass token limit"
    );
    assert!(
        validate_token_length(&long_code, 100).is_err(),
        "Long code should fail token limit"
    );

    // Test 2: JSON schema validation using type-based deserialization
    // validate_json_schema<T> validates by attempting to deserialize into T
    #[derive(Deserialize, Serialize)]
    struct CodeResponse {
        code: String,
        language: String,
        #[serde(default)]
        tests_pass: Option<bool>,
    }

    let valid_json = r#"{"code": "fn is_prime(n: u64) -> bool { n > 1 }", "language": "rust", "tests_pass": true}"#;
    let invalid_json = r#"{"code": "fn is_prime(n: u64) -> bool { n > 1 }"}"#; // missing "language"

    // Use maker's validate_json_schema which deserializes to check validity
    let valid_result: Result<CodeResponse, _> =
        maker::core::validate_json_schema::<CodeResponse>(valid_json);
    let invalid_result: Result<CodeResponse, _> =
        maker::core::validate_json_schema::<CodeResponse>(invalid_json);

    assert!(
        valid_result.is_ok(),
        "Valid JSON should pass schema validation"
    );
    assert!(
        invalid_result.is_err(),
        "Invalid JSON (missing required field) should fail validation"
    );
}

/// Test VoteRace directly with deterministic inputs
#[test]
fn test_vote_race_k_margin_dynamics() {
    // Create a vote race with k=3
    let k = 3;
    let race = VoteRace::new(k).expect("Valid k-margin");

    // Cast votes to simulate convergence
    // Candidate A: 5 votes, Candidate B: 2 votes → margin = 3 = k (winner!)
    for _ in 0..5 {
        race.cast_vote("correct_solution".into());
    }
    for _ in 0..2 {
        race.cast_vote("incorrect_solution".into());
    }

    // Check winner
    let result = race.check_winner();
    match result {
        maker::core::VoteCheckResult::Winner {
            candidate, margin, ..
        } => {
            assert_eq!(candidate.as_str(), "correct_solution");
            assert_eq!(margin, 3, "Margin should equal k");
        }
        maker::core::VoteCheckResult::Ongoing { .. } => {
            panic!("Should have a winner with margin >= k");
        }
    }
}

/// Test that voting handles ties correctly (no premature winner)
#[test]
fn test_vote_race_no_premature_winner() {
    let k = 3;
    let race = VoteRace::new(k).expect("Valid k-margin");

    // Equal votes: no winner
    for _ in 0..3 {
        race.cast_vote("A".into());
        race.cast_vote("B".into());
    }

    match race.check_winner() {
        maker::core::VoteCheckResult::Ongoing {
            current_margin,
            leader,
            ..
        } => {
            assert_eq!(current_margin, 0, "Margin should be 0 with tied votes");
            assert!(leader.is_some(), "Should have a leader even when tied");
        }
        maker::core::VoteCheckResult::Winner { .. } => {
            panic!("Should not have a winner when votes are tied");
        }
    }
}

// ============================================================================
// Test 2: Task Decomposition and Aggregation
// ============================================================================

/// Custom decomposition agent for REST API endpoint task
///
/// Decomposes "build a REST API endpoint" into:
/// 1. Define route handler
/// 2. Implement request validation
/// 3. Add error handling
/// 4. Write response serialization
#[derive(Debug)]
struct RestApiDecomposer;

impl DecompositionAgent for RestApiDecomposer {
    fn propose_decomposition(
        &self,
        task_id: &str,
        _description: &str,
        context: &serde_json::Value,
        _depth: usize,
    ) -> Result<DecompositionProposal, maker::core::decomposition::DecompositionError> {
        // Create subtasks for building a REST API endpoint
        let subtasks = vec![
            DecompositionSubtask::leaf(
                format!("{}-route", task_id),
                "Define route handler with HTTP method and path",
            )
            .with_context(context.clone())
            .with_order(0),
            DecompositionSubtask::leaf(
                format!("{}-validation", task_id),
                "Implement request body and query parameter validation",
            )
            .with_context(context.clone())
            .with_order(1),
            DecompositionSubtask::leaf(
                format!("{}-error", task_id),
                "Add error handling with appropriate HTTP status codes",
            )
            .with_context(context.clone())
            .with_order(2),
            DecompositionSubtask::leaf(
                format!("{}-response", task_id),
                "Implement response serialization to JSON",
            )
            .with_context(context.clone())
            .with_order(3),
        ];

        Ok(DecompositionProposal::new(
            format!("rest-api-proposal-{}", task_id),
            task_id,
            subtasks,
            CompositionFunction::Sequential, // Steps must be done in order
        )
        .with_confidence(0.9)
        .with_rationale("Standard REST API endpoint implementation pattern"))
    }

    fn is_atomic(&self, _task_id: &str, _description: &str) -> bool {
        false // Always decompose
    }

    fn name(&self) -> &str {
        "rest-api-decomposer"
    }
}

/// Test task decomposition for a REST API endpoint
#[test]
fn test_rest_api_task_decomposition() {
    let decomposer = RestApiDecomposer;

    let task_id = "create-user-endpoint";
    let description = "Build a REST API endpoint for user creation";
    let context = json!({
        "method": "POST",
        "path": "/api/users",
        "request_schema": {
            "name": "string",
            "email": "string"
        }
    });

    let proposal = decomposer
        .propose_decomposition(task_id, description, &context, 0)
        .expect("Decomposition should succeed");

    // Verify proposal structure
    assert_eq!(proposal.source_task_id, task_id);
    assert_eq!(proposal.subtasks.len(), 4, "Should have 4 subtasks");

    // Verify all subtasks are leaf nodes with m=1
    for subtask in &proposal.subtasks {
        assert!(subtask.is_leaf, "All subtasks should be leaf nodes");
        assert_eq!(subtask.m_value, 1, "All leaf nodes must have m=1");
    }

    // Verify ordering
    let orders: Vec<usize> = proposal.subtasks.iter().map(|s| s.order).collect();
    assert_eq!(orders, vec![0, 1, 2, 3], "Subtasks should be ordered 0-3");

    // Verify composition function
    assert_eq!(
        proposal.composition_fn,
        CompositionFunction::Sequential,
        "Should use sequential composition"
    );

    // Validate proposal
    assert!(proposal.validate().is_ok(), "Proposal should be valid");

    println!("Decomposition proposal:");
    println!("  Task: {}", proposal.source_task_id);
    println!("  Subtasks: {}", proposal.subtasks.len());
    for subtask in &proposal.subtasks {
        println!(
            "    [{}] {} (m={})",
            subtask.order, subtask.description, subtask.m_value
        );
    }
}

/// Test result aggregation using SolutionDiscriminator
#[test]
fn test_solution_aggregation_sequential() {
    // Create mock subtask results simulating executed leaf nodes
    let results = vec![
        SubtaskResult::success(
            "task-route".to_string(),
            "Route handler: GET /api/users".to_string(),
            json!({"route": "/api/users", "method": "GET"}),
        ),
        SubtaskResult::success(
            "task-validation".to_string(),
            "Validation: email format check".to_string(),
            json!({"validated": true, "fields": ["email"]}),
        ),
        SubtaskResult::success(
            "task-error".to_string(),
            "Error handling: 400, 404, 500".to_string(),
            json!({"status_codes": [400, 404, 500]}),
        ),
        SubtaskResult::success(
            "task-response".to_string(),
            "Response: JSON serialization".to_string(),
            json!({"content_type": "application/json"}),
        ),
    ];

    // Create aggregator with default config
    let config = AggregatorConfig::default();
    let discriminator = SolutionDiscriminator::with_config(config);

    // Create a proposal for aggregation
    let proposal = DecompositionProposal::new(
        "test-proposal",
        "api-endpoint",
        vec![
            DecompositionSubtask::leaf("task-route", "Route").with_order(0),
            DecompositionSubtask::leaf("task-validation", "Validation").with_order(1),
            DecompositionSubtask::leaf("task-error", "Errors").with_order(2),
            DecompositionSubtask::leaf("task-response", "Response").with_order(3),
        ],
        CompositionFunction::Sequential,
    );

    // Aggregate results (takes owned Vec, depth=0)
    let aggregated = discriminator
        .aggregate(&proposal, results, 0)
        .expect("Aggregation should succeed");

    // Verify aggregation using correct field names
    assert!(aggregated.all_succeeded, "All subtasks should succeed");
    assert_eq!(
        aggregated.metrics.subtask_count, 4,
        "Should aggregate 4 subtask results"
    );
    assert_eq!(
        aggregated.success_count(),
        4,
        "All 4 subtasks should be successful"
    );

    // Verify aggregation produced output (sequential composition returns final result)
    assert!(!aggregated.output.is_empty(), "Output should not be empty");
    // Sequential composition uses the last subtask's output as the final output
    // This is by design - state is threaded through, final output is the composition
    println!("Sequential aggregation output: {}", aggregated.output);

    println!("Aggregated result:");
    println!("  All succeeded: {}", aggregated.all_succeeded);
    println!(
        "  Subtasks: {}/{}",
        aggregated.success_count(),
        aggregated.metrics.subtask_count
    );
    println!("  Output length: {} chars", aggregated.output.len());
}

/// Test parallel composition with merge strategy
#[test]
fn test_solution_aggregation_parallel() {
    // Simulate parallel execution results (independent tasks)
    let results = vec![
        SubtaskResult::success(
            "fetch-users".to_string(),
            "Fetched 10 users".to_string(),
            json!({"users": [{"id": 1}, {"id": 2}]}),
        ),
        SubtaskResult::success(
            "fetch-posts".to_string(),
            "Fetched 25 posts".to_string(),
            json!({"posts": [{"id": 101}, {"id": 102}]}),
        ),
        SubtaskResult::success(
            "fetch-comments".to_string(),
            "Fetched 50 comments".to_string(),
            json!({"comments": [{"id": 1001}]}),
        ),
    ];

    let config = AggregatorConfig::default();
    let discriminator = SolutionDiscriminator::with_config(config);

    let proposal = DecompositionProposal::new(
        "parallel-fetch",
        "fetch-all-data",
        vec![
            DecompositionSubtask::leaf("fetch-users", "Fetch users"),
            DecompositionSubtask::leaf("fetch-posts", "Fetch posts"),
            DecompositionSubtask::leaf("fetch-comments", "Fetch comments"),
        ],
        CompositionFunction::Parallel {
            merge_strategy: MergeStrategy::Concatenate,
        },
    );

    let aggregated = discriminator
        .aggregate(&proposal, results, 0)
        .expect("Parallel aggregation should succeed");

    assert!(aggregated.all_succeeded);
    assert_eq!(aggregated.metrics.subtask_count, 3);
    assert_eq!(aggregated.success_count(), 3);

    // Verify all outputs are present in concatenated result
    assert!(aggregated.output.contains("users"));
    assert!(aggregated.output.contains("posts"));
    assert!(aggregated.output.contains("comments"));
}

/// Test aggregation handles partial failures correctly
#[test]
fn test_aggregation_with_partial_failure() {
    let results = vec![
        SubtaskResult::success(
            "task-1".to_string(),
            "Success".to_string(),
            json!({"ok": true}),
        ),
        SubtaskResult::failure(
            "task-2".to_string(),
            "Database connection failed".to_string(),
        ),
        SubtaskResult::success(
            "task-3".to_string(),
            "Also succeeded".to_string(),
            json!({"ok": true}),
        ),
    ];

    let config = AggregatorConfig::default();
    let discriminator = SolutionDiscriminator::with_config(config);

    let proposal = DecompositionProposal::new(
        "mixed-results",
        "source",
        vec![
            DecompositionSubtask::leaf("task-1", "First"),
            DecompositionSubtask::leaf("task-2", "Second (will fail)"),
            DecompositionSubtask::leaf("task-3", "Third"),
        ],
        CompositionFunction::Sequential,
    );

    let aggregated = discriminator
        .aggregate(&proposal, results, 0)
        .expect("Aggregation should handle partial failures");

    // With sequential composition, partial failure should still produce a result
    // but the aggregated result tracks the failure
    assert!(!aggregated.all_succeeded, "Not all subtasks succeeded");
    assert_eq!(aggregated.metrics.subtask_count, 3);
    assert_eq!(aggregated.success_count(), 2, "Only 2 of 3 tasks succeeded");
    assert_eq!(aggregated.failure_count(), 1, "1 task failed");

    println!("Partial failure aggregation:");
    println!("  Total: {}", aggregated.metrics.subtask_count);
    println!("  Successful: {}", aggregated.success_count());
    println!("  Failed: {}", aggregated.failure_count());
}

/// Test the complete decomposition → execution → aggregation workflow
#[test]
fn test_end_to_end_decomposition_workflow() {
    // Step 1: Decompose the task
    let decomposer = RestApiDecomposer;
    let task_id = "build-api";
    let context = json!({"endpoint": "/api/items"});

    let proposal = decomposer
        .propose_decomposition(task_id, "Build REST API", &context, 0)
        .expect("Decomposition should succeed");

    // Step 2: Simulate leaf execution with voting
    // In production, each leaf would use vote_with_margin() with a real LLM
    let mut results = Vec::new();

    for subtask in &proposal.subtasks {
        // Create a mock client that always returns a "correct" implementation
        let mock_response = format!("Implementation for: {}", subtask.description);
        let client = MockLlmClient::constant(&mock_response);

        // Use simple k=2 for testing
        let vote_config = VoteConfig::default()
            .with_max_samples(10)
            .without_token_limit();

        let vote_result =
            vote_with_margin(&subtask.description, 2, &client, vote_config).expect("Voting works");

        results.push(SubtaskResult::success(
            subtask.task_id.clone(),
            vote_result.winner.clone(),
            json!({"voted": true, "samples": vote_result.total_samples}),
        ));
    }

    // Step 3: Aggregate results
    let aggregator = SolutionDiscriminator::with_config(AggregatorConfig::default());
    let final_result = aggregator
        .aggregate(&proposal, results, 0)
        .expect("Aggregation should succeed");

    // Verify end-to-end success
    assert!(
        final_result.all_succeeded,
        "End-to-end workflow should succeed"
    );
    assert_eq!(final_result.metrics.subtask_count, 4);
    assert_eq!(final_result.success_count(), 4);

    // Verify output is non-empty and contains implementation text
    assert!(
        !final_result.output.is_empty(),
        "Final output should not be empty"
    );
    assert!(
        final_result.output.contains("Implementation for:"),
        "Final output should contain implementation markers. Got: {}",
        final_result.output
    );

    println!("\n=== End-to-End Workflow Complete ===");
    println!("Task: {}", task_id);
    println!("Subtasks executed: {}", proposal.subtasks.len());
    println!("All succeeded: {}", final_result.all_succeeded);
    println!(
        "Output preview: {}",
        &final_result.output[..200.min(final_result.output.len())]
    );
}

// ============================================================================
// Edge Cases and Robustness Tests
// ============================================================================

/// Test voting with very high accuracy (should converge quickly)
#[test]
fn test_voting_high_accuracy_fast_convergence() {
    let client = MockLlmClient::biased("correct", "wrong", 0.99, 100);
    let config = VoteConfig::default().with_max_samples(50);

    let result = vote_with_margin("test", 3, &client, config).expect("Should converge");

    assert_eq!(result.winner, "correct");
    // With 99% accuracy, should converge very quickly
    assert!(
        result.total_samples < 20,
        "High accuracy should converge in <20 samples, got {}",
        result.total_samples
    );
}

/// Test voting with moderate accuracy (takes more samples)
#[test]
fn test_voting_moderate_accuracy() {
    let client = MockLlmClient::biased("correct", "wrong", 0.70, 200);
    let config = VoteConfig::default().with_max_samples(200);

    let result = vote_with_margin("test", 3, &client, config).expect("Should eventually converge");

    assert_eq!(result.winner, "correct");
    // With 70% accuracy and k=3, needs at least k samples to establish margin
    assert!(
        result.total_samples >= 3,
        "Voting needs at least k samples, got {}",
        result.total_samples
    );
}

/// Test decomposition validation catches invalid proposals
#[test]
fn test_decomposition_validation_catches_errors() {
    // Empty subtasks
    let empty_proposal =
        DecompositionProposal::new("empty", "source", vec![], CompositionFunction::Sequential);
    assert!(
        empty_proposal.validate().is_err(),
        "Empty proposal should fail validation"
    );

    // Duplicate task IDs
    let duplicate_proposal = DecompositionProposal::new(
        "dup",
        "source",
        vec![
            DecompositionSubtask::leaf("same-id", "First"),
            DecompositionSubtask::leaf("same-id", "Second"),
        ],
        CompositionFunction::Sequential,
    );
    assert!(
        duplicate_proposal.validate().is_err(),
        "Duplicate IDs should fail validation"
    );

    // Invalid m-value on leaf
    let mut invalid_subtask = DecompositionSubtask::leaf("leaf", "Leaf task");
    invalid_subtask.m_value = 5; // Invalid: leaf must have m=1

    let invalid_proposal = DecompositionProposal::new(
        "invalid-m",
        "source",
        vec![invalid_subtask],
        CompositionFunction::Sequential,
    );
    assert!(
        invalid_proposal.validate().is_err(),
        "Invalid m-value should fail validation"
    );
}

/// Test identity decomposer (pass-through)
#[test]
fn test_identity_decomposer() {
    let decomposer = IdentityDecomposer;

    assert!(
        decomposer.is_atomic("any", "any"),
        "Identity decomposer treats all tasks as atomic"
    );

    let proposal = decomposer
        .propose_decomposition("task-1", "Simple task", &json!({}), 0)
        .expect("Identity decomposition always succeeds");

    assert_eq!(proposal.subtasks.len(), 1);
    assert!(proposal.subtasks[0].is_leaf);
    assert_eq!(proposal.subtasks[0].m_value, 1);
    assert!((proposal.confidence - 1.0).abs() < f64::EPSILON);
}
