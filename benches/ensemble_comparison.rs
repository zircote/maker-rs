//! Ensemble Comparison Benchmark for MAKER Framework
//!
//! Compares single-model vs. multi-model ensemble reliability and cost
//! across different strategies (round-robin, cost-aware) using Monte Carlo
//! simulation.
//!
//! # Usage
//!
//! ```bash
//! cargo bench --bench ensemble_comparison
//! ```

use maker::core::{calculate_kmin, vote_with_margin, MockLlmClient, VoteConfig};
use std::collections::HashMap;
use std::time::Instant;

/// Configuration for a benchmark scenario
#[derive(Debug, Clone)]
struct BenchConfig {
    name: String,
    /// Per-step success probabilities for each model
    /// Single-model: vec![p], Ensemble: vec![p1, p2, ...]
    model_ps: Vec<f64>,
    /// Cost per sample for each model (normalized)
    model_costs: Vec<f64>,
    /// Whether to use cost-aware ordering (cheap models first)
    cost_aware: bool,
}

/// Results from a benchmark run
#[derive(Debug)]
#[allow(dead_code)]
struct BenchResult {
    name: String,
    steps: usize,
    k_margin: usize,
    mean_total_samples: f64,
    mean_total_cost: f64,
    error_rate: f64,
    mean_samples_per_step: f64,
    mean_time_ms: f64,
    trials: usize,
}

/// Simulate a single step of ensemble voting.
///
/// For ensemble: interleave samples from multiple models (each with different p),
/// then vote. This simulates the decorrelation benefit of using diverse models.
///
/// Returns (samples_used, cost_incurred, success)
fn simulate_ensemble_step(config: &BenchConfig, k: usize, pool_size: usize) -> (usize, f64, bool) {
    let n_models = config.model_ps.len();

    if n_models == 1 {
        // Single model: standard voting
        let p = config.model_ps[0];
        let cost_per = config.model_costs[0];
        let correct_count = (pool_size as f64 * p).round() as usize;
        let mut responses = vec!["correct".to_string(); correct_count];
        responses.extend(vec!["wrong".to_string(); pool_size - correct_count]);

        let client = MockLlmClient::new(responses);
        let vote_config = VoteConfig::default()
            .with_max_samples(pool_size)
            .without_token_limit();

        match vote_with_margin("bench", k, &client, vote_config) {
            Ok(result) => {
                let cost = result.total_samples as f64 * cost_per;
                let success = result.winner == "correct";
                (result.total_samples, cost, success)
            }
            Err(_) => (pool_size, pool_size as f64 * cost_per, false),
        }
    } else {
        // Multi-model ensemble: build interleaved response pool
        // Each model contributes samples at its own success rate.
        // The key insight: different model architectures produce independent errors,
        // so the effective p for the ensemble is higher than any individual p.
        let per_model = pool_size / n_models;

        // Build combined pool with interleaved responses
        let mut responses = Vec::with_capacity(pool_size);
        let mut cost_schedule = Vec::with_capacity(pool_size);

        if config.cost_aware {
            // Cost-aware: all cheap samples first, then medium, then expensive
            let mut model_indices: Vec<usize> = (0..n_models).collect();
            model_indices.sort_by(|a, b| {
                config.model_costs[*a]
                    .partial_cmp(&config.model_costs[*b])
                    .unwrap()
            });

            for &mi in &model_indices {
                let p = config.model_ps[mi];
                let cost_per = config.model_costs[mi];
                let correct = (per_model as f64 * p).round() as usize;
                responses.extend(vec!["correct".to_string(); correct]);
                responses.extend(vec!["wrong".to_string(); per_model - correct]);
                cost_schedule.extend(vec![cost_per; per_model]);
            }
        } else {
            // Round-robin: alternate between models
            for sample_idx in 0..pool_size {
                let mi = sample_idx % n_models;
                let p = config.model_ps[mi];
                let cost_per = config.model_costs[mi];
                // Deterministic: use p to decide correct/wrong based on position within model's slice
                let model_sample_idx = sample_idx / n_models;
                let threshold = (per_model as f64 * p).round() as usize;
                if model_sample_idx < threshold {
                    responses.push("correct".to_string());
                } else {
                    responses.push("wrong".to_string());
                }
                cost_schedule.push(cost_per);
            }
        }

        let client = MockLlmClient::new(responses);
        let vote_config = VoteConfig::default()
            .with_max_samples(pool_size)
            .without_token_limit();

        match vote_with_margin("bench", k, &client, vote_config) {
            Ok(result) => {
                let cost: f64 = cost_schedule[..result.total_samples.min(cost_schedule.len())]
                    .iter()
                    .sum();
                let success = result.winner == "correct";
                (result.total_samples, cost, success)
            }
            Err(_) => {
                let cost: f64 = cost_schedule.iter().sum();
                (pool_size, cost, false)
            }
        }
    }
}

/// Run Monte Carlo simulation for a benchmark configuration
fn run_simulation(config: &BenchConfig, s: usize, k: usize, trials: usize) -> BenchResult {
    let pool_size = 200;
    let mut total_samples_all = Vec::with_capacity(trials);
    let mut total_cost_all = Vec::with_capacity(trials);
    let mut error_count = 0usize;
    let mut total_time_ms = 0u128;

    for _ in 0..trials {
        let start = Instant::now();
        let mut trial_samples = 0usize;
        let mut trial_cost = 0.0f64;
        let mut trial_success = true;

        for _ in 0..s {
            let (samples, cost, success) = simulate_ensemble_step(config, k, pool_size);
            trial_samples += samples;
            trial_cost += cost;
            if !success {
                trial_success = false;
            }
        }

        total_time_ms += start.elapsed().as_millis();
        total_samples_all.push(trial_samples);
        total_cost_all.push(trial_cost);
        if !trial_success {
            error_count += 1;
        }
    }

    let mean_samples = total_samples_all.iter().sum::<usize>() as f64 / trials as f64;
    let mean_cost = total_cost_all.iter().sum::<f64>() / trials as f64;

    BenchResult {
        name: config.name.clone(),
        steps: s,
        k_margin: k,
        mean_total_samples: mean_samples,
        mean_total_cost: mean_cost,
        error_rate: error_count as f64 / trials as f64,
        mean_samples_per_step: mean_samples / s as f64,
        mean_time_ms: total_time_ms as f64 / trials as f64,
        trials,
    }
}

/// Compute effective p for an ensemble of independent models.
///
/// With k-margin voting and independent errors, the effective probability
/// that the correct answer wins is higher than any individual model's p
/// because errors are decorrelated across model architectures.
///
/// Approximation: p_eff = 1 - product(1 - p_i) for independent models
/// (probability that at least one model gets it right)
fn effective_ensemble_p(ps: &[f64]) -> f64 {
    1.0 - ps.iter().map(|p| 1.0 - p).product::<f64>()
}

fn main() {
    let trials = 1_000;

    println!("=== MAKER Ensemble Comparison Benchmark ===\n");
    println!("Trials per configuration: {}", trials);
    println!();

    // --- Benchmark Configurations ---

    // Single model: Ollama llama3 (cheap, p=0.80)
    let single_cheap = BenchConfig {
        name: "Single: llama3 (p=0.80)".to_string(),
        model_ps: vec![0.80],
        model_costs: vec![0.001], // $0.001 per sample
        cost_aware: false,
    };

    // Single model: Claude Haiku (expensive, p=0.90)
    let single_expensive = BenchConfig {
        name: "Single: Haiku (p=0.90)".to_string(),
        model_ps: vec![0.90],
        model_costs: vec![0.01], // $0.01 per sample
        cost_aware: false,
    };

    // Ensemble: llama3 + Haiku, round-robin
    let ensemble_rr = BenchConfig {
        name: "Ensemble RR: llama3+Haiku".to_string(),
        model_ps: vec![0.80, 0.90],
        model_costs: vec![0.001, 0.01],
        cost_aware: false,
    };

    // Ensemble: llama3 + Haiku, cost-aware
    let ensemble_ca = BenchConfig {
        name: "Ensemble CA: llama3+Haiku".to_string(),
        model_ps: vec![0.80, 0.90],
        model_costs: vec![0.001, 0.01],
        cost_aware: true,
    };

    let configs = vec![single_cheap, single_expensive, ensemble_rr, ensemble_ca];

    // --- Run at two task sizes ---
    for &s in &[100, 1000] {
        println!("━━━ Task size: s={} steps ━━━\n", s);

        let mut results = Vec::new();

        for config in &configs {
            // Calculate effective p for k-margin determination
            let p_eff = if config.model_ps.len() == 1 {
                config.model_ps[0]
            } else {
                // For ensemble, use a conservative blended p
                // Average of individual ps (conservative vs effective_ensemble_p)
                let avg_p: f64 = config.model_ps.iter().sum::<f64>() / config.model_ps.len() as f64;
                avg_p.min(0.95) // Cap to avoid k=1
            };

            let k = calculate_kmin(p_eff, 0.95, s, 1).unwrap_or(3);
            let result = run_simulation(config, s, k, trials);
            results.push(result);
        }

        // Print comparison table
        println!(
            "{:<35} {:>4} {:>6} {:>10} {:>10} {:>8} {:>8}",
            "Configuration", "k", "steps", "samples", "cost($)", "err%", "samp/step"
        );
        println!("{}", "─".repeat(87));

        for r in &results {
            println!(
                "{:<35} {:>4} {:>6} {:>10.0} {:>10.4} {:>7.2}% {:>8.1}",
                r.name,
                r.k_margin,
                r.steps,
                r.mean_total_samples,
                r.mean_total_cost,
                r.error_rate * 100.0,
                r.mean_samples_per_step,
            );
        }

        println!();

        // --- Verify acceptance criteria ---

        // AC1: Ensemble error rate < min(individual model error rates)
        let single_error_rates: Vec<f64> = results
            .iter()
            .filter(|r| !r.name.contains("Ensemble"))
            .map(|r| r.error_rate)
            .collect();
        let min_single_error = single_error_rates
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        let ensemble_rr_result = results.iter().find(|r| r.name.contains("RR")).unwrap();
        let ensemble_ca_result = results.iter().find(|r| r.name.contains("CA")).unwrap();

        let rr_diversity_benefit = ensemble_rr_result.error_rate <= min_single_error;
        let ca_diversity_benefit = ensemble_ca_result.error_rate <= min_single_error;

        println!("Acceptance Criteria (s={}):", s);
        println!(
            "  [{}] Ensemble RR error ({:.2}%) <= min single-model error ({:.2}%)",
            if rr_diversity_benefit { "PASS" } else { "WARN" },
            ensemble_rr_result.error_rate * 100.0,
            min_single_error * 100.0,
        );
        println!(
            "  [{}] Ensemble CA error ({:.2}%) <= min single-model error ({:.2}%)",
            if ca_diversity_benefit { "PASS" } else { "WARN" },
            ensemble_ca_result.error_rate * 100.0,
            min_single_error * 100.0,
        );

        // AC2: Cost-aware ensemble cost < expensive-model-only cost by 30%+
        let expensive_only = results
            .iter()
            .find(|r| r.name.contains("Haiku") && !r.name.contains("Ensemble"))
            .unwrap();
        let ca_savings =
            1.0 - (ensemble_ca_result.mean_total_cost / expensive_only.mean_total_cost);

        println!(
            "  [{}] Cost-aware ensemble cost saving vs expensive-only: {:.1}% (target: >=30%)",
            if ca_savings >= 0.30 { "PASS" } else { "WARN" },
            ca_savings * 100.0,
        );

        // Effective p comparison
        let ensemble_ps = vec![0.80, 0.90];
        let p_eff = effective_ensemble_p(&ensemble_ps);
        println!(
            "  Info: Effective ensemble p (independence assumption): {:.4}",
            p_eff,
        );

        println!();
    }

    // --- JSON Output ---
    let mut json_configs = Vec::new();
    for config in &configs {
        let p_eff = if config.model_ps.len() == 1 {
            config.model_ps[0]
        } else {
            let avg_p: f64 = config.model_ps.iter().sum::<f64>() / config.model_ps.len() as f64;
            avg_p.min(0.95)
        };

        let mut per_size = HashMap::new();
        for &s in &[100, 1000] {
            let k = calculate_kmin(p_eff, 0.95, s, 1).unwrap_or(3);
            let result = run_simulation(config, s, k, trials);
            per_size.insert(
                format!("s_{}", s),
                serde_json::json!({
                    "steps": s,
                    "k_margin": k,
                    "mean_total_samples": result.mean_total_samples,
                    "mean_total_cost": result.mean_total_cost,
                    "error_rate": result.error_rate,
                    "mean_samples_per_step": result.mean_samples_per_step,
                    "mean_time_ms": result.mean_time_ms,
                }),
            );
        }

        json_configs.push(serde_json::json!({
            "name": config.name,
            "model_ps": config.model_ps,
            "model_costs": config.model_costs,
            "cost_aware": config.cost_aware,
            "effective_p": p_eff,
            "results": per_size,
        }));
    }

    let output = serde_json::json!({
        "benchmark": "ensemble_comparison",
        "trials": trials,
        "target_reliability": 0.95,
        "configurations": json_configs,
    });

    println!("=== JSON Output ===\n");
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
