//! Custom Task Integration Template
//!
//! Shows how to integrate MAKER's voting engine with a custom task.
//! Replace the mock LLM client with a real provider for production use.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example custom_task
//! ```

use maker::core::{calculate_kmin, vote_with_margin, MockLlmClient, VoteConfig};

fn main() {
    println!("=== MAKER Custom Task Example ===\n");

    // Step 1: Estimate your model's per-step success probability
    let p = 0.85; // 85% accuracy per step

    // Step 2: Set your target reliability
    let t = 0.95; // 95% overall task success

    // Step 3: Define your task
    let total_steps = 50;

    // Step 4: Calculate the required k-margin
    let k = calculate_kmin(p, t, total_steps, 1).expect("Valid parameters");
    println!("Task: {} steps", total_steps);
    println!("Model accuracy: {:.0}%", p * 100.0);
    println!("Target reliability: {:.0}%", t * 100.0);
    println!("Required k-margin: {}\n", k);

    // Step 5: For each step, use voting to get error-corrected output
    //
    // In production, replace MockLlmClient with a real provider:
    //   use maker::llm::OllamaClient;
    //   let client = OllamaClient::new("http://localhost:11434", "llama3");
    let client = MockLlmClient::biased("correct_answer", "wrong_answer", p, 200);
    let config = VoteConfig::default()
        .with_max_samples(50)
        .without_token_limit();

    // Execute a single voting round
    match vote_with_margin("What is the capital of France?", k, &client, config) {
        Ok(result) => {
            println!("Winner: {}", result.winner);
            println!("Total samples: {}", result.total_samples);
            println!("Red-flagged: {}", result.red_flagged);
            println!("Elapsed: {:?}", result.elapsed);
            println!("\nVote distribution:");
            for (candidate, count) in &result.vote_counts {
                println!("  {}: {} votes", candidate, count);
            }
        }
        Err(e) => {
            eprintln!("Voting failed: {}", e);
        }
    }

    println!("\n=== Done ===");
}
