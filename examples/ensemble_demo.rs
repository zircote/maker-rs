//! Multi-Model Ensemble Voting Demo
//!
//! Demonstrates configuring MAKER with multiple models for ensemble
//! voting, where error decorrelation across model architectures
//! improves reliability beyond any single model.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example ensemble_demo
//! ```

use maker::core::{calculate_kmin, vote_with_margin, MockLlmClient, VoteConfig};
use maker::llm::ensemble::{CostTier, EnsembleStrategy};

fn main() {
    println!("=== MAKER Ensemble Voting Demo ===\n");

    // Scenario: compare single-model vs ensemble voting
    let p_cheap = 0.80; // Cheap model accuracy
    let p_expensive = 0.90; // Expensive model accuracy
    let t = 0.95;
    let steps = 50;

    println!(
        "Task: {} steps, target reliability: {:.0}%\n",
        steps,
        t * 100.0
    );

    // --- Single model (expensive) ---
    let k_single = calculate_kmin(p_expensive, t, steps, 1).unwrap();
    println!("Single model (p={:.2}):", p_expensive);
    println!("  k-margin: {}", k_single);

    let client = MockLlmClient::biased("correct", "wrong", p_expensive, 200);
    let config = VoteConfig::default()
        .with_max_samples(200)
        .without_token_limit();
    let result = vote_with_margin("test", k_single, &client, config).unwrap();
    let single_cost = result.total_samples as f64 * 0.01; // $0.01/sample
    println!("  Samples for 1 step: {}", result.total_samples);
    println!("  Cost per step: ${:.4}\n", single_cost);

    // --- Ensemble (cheap + expensive) ---
    // With ensemble, effective p is higher due to error decorrelation
    let p_ensemble = 1.0 - (1.0 - p_cheap) * (1.0 - p_expensive); // ≈ 0.98
    let k_ensemble = calculate_kmin(p_ensemble.min(0.95), t, steps, 1).unwrap();
    println!(
        "Ensemble (p_cheap={:.2}, p_expensive={:.2}):",
        p_cheap, p_expensive
    );
    println!("  Effective p (independence): {:.4}", p_ensemble);
    println!("  k-margin: {}", k_ensemble);

    // Cost-aware: cheap model samples first
    let client_cheap = MockLlmClient::biased("correct", "wrong", p_cheap, 200);
    let config = VoteConfig::default()
        .with_max_samples(200)
        .without_token_limit();
    let result = vote_with_margin("test", k_ensemble, &client_cheap, config).unwrap();
    let ensemble_cost = result.total_samples as f64 * 0.001; // $0.001/sample for cheap
    println!("  Samples (cheap model): {}", result.total_samples);
    println!("  Cost per step: ${:.4}", ensemble_cost);

    let savings = 1.0 - ensemble_cost / single_cost;
    println!("\n--- Cost Comparison ---");
    println!("  Single expensive: ${:.4}/step", single_cost);
    println!("  Ensemble (cost-aware): ${:.4}/step", ensemble_cost);
    println!("  Savings: {:.1}%", savings * 100.0);

    // --- Demonstrate EnsembleConfig construction ---
    println!("\n--- Ensemble Configuration ---");

    // Note: EnsembleConfig requires real LlmClient instances.
    // This demonstrates the configuration API structure.
    println!("  EnsembleConfig supports 2-5 models");
    println!("  Strategies: RoundRobin, CostAware, ReliabilityWeighted");
    println!("  CostTiers: Cheap, Medium, Expensive");
    println!("  Configure via MCP: maker/configure with 'ensemble' field");

    // Show the strategy descriptions
    let strategies = [
        (EnsembleStrategy::RoundRobin, "Distribute samples evenly"),
        (
            EnsembleStrategy::CostAware,
            "Cheap first, escalate on disagreement",
        ),
        (
            EnsembleStrategy::ReliabilityWeighted,
            "Weight toward reliable models",
        ),
    ];

    println!("\n  Available strategies:");
    for (strategy, desc) in &strategies {
        println!("    {:?}: {}", strategy, desc);
    }

    println!("\n  Cost tiers:");
    for tier in &[CostTier::Cheap, CostTier::Medium, CostTier::Expensive] {
        println!("    {:?} (order: {})", tier, tier.cost_order());
    }

    println!("\n=== Done ===");
}
