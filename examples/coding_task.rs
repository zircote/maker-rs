//! Coding Task with Semantic Matching
//!
//! Demonstrates MAKER voting on a coding task where multiple correct
//! implementations exist. Uses the CandidateMatcher trait to group
//! semantically equivalent responses for voting.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example coding_task
//! ```

use maker::core::matcher::{CandidateMatcher, ExactMatcher};
use maker::core::{calculate_kmin, vote_with_margin, MockLlmClient, VoteConfig};
use std::sync::Arc;

fn main() {
    println!("=== MAKER Coding Task Example ===\n");

    // Scenario: voting on a FizzBuzz implementation
    // Multiple correct solutions exist, but we use ExactMatcher here
    // since MockLlmClient returns identical strings.
    //
    // In production with real LLM responses, you would use:
    //   - CodeMatcher (with `code-matcher` feature) for AST comparison
    //   - EmbeddingMatcher for semantic similarity

    let p = 0.85;
    let t = 0.95;
    let steps = 5; // 5-step task decomposition

    let k = calculate_kmin(p, t, steps, 1).expect("Valid parameters");
    println!("Coding task: {} subtasks", steps);
    println!("Model accuracy: {:.0}%", p * 100.0);
    println!("Required k-margin: {}\n", k);

    // Simulate voting on each subtask
    let subtasks = [
        "Define the FizzBuzz function signature",
        "Implement the divisibility logic",
        "Handle the FizzBuzz case (divisible by both)",
        "Add the loop for 1 to 100",
        "Format and return the output",
    ];

    let matcher: Arc<dyn CandidateMatcher> = Arc::new(ExactMatcher);
    let mut total_samples = 0;

    for (i, subtask) in subtasks.iter().enumerate() {
        let client = MockLlmClient::biased("correct_impl", "wrong_impl", p, 100);
        let config = VoteConfig::default()
            .with_max_samples(50)
            .without_token_limit()
            .with_matcher(matcher.clone());

        match vote_with_margin(subtask, k, &client, config) {
            Ok(result) => {
                total_samples += result.total_samples;
                println!(
                    "  Step {}: {} ({} samples) -> {}",
                    i + 1,
                    if result.winner == "correct_impl" {
                        "PASS"
                    } else {
                        "FAIL"
                    },
                    result.total_samples,
                    subtask,
                );
            }
            Err(e) => {
                println!("  Step {}: ERROR - {}", i + 1, e);
            }
        }
    }

    println!("\nTotal samples used: {}", total_samples);
    println!(
        "Average samples per step: {:.1}",
        total_samples as f64 / steps as f64
    );
    println!("\n=== Done ===");
}
