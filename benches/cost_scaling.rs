//! Cost Scaling Benchmark for MAKER Framework
//!
//! Validates that MAKER's cost scales as Θ(s ln s) by running simulations
//! across multiple task sizes and fitting the data.
//!
//! # Usage
//!
//! ```bash
//! cargo bench --bench cost_scaling
//! ```

use maker::core::{calculate_kmin, vote_with_margin, MockLlmClient, VoteConfig};
use std::time::Instant;

/// Run MAKER simulation for a given disk count and collect metrics
fn run_hanoi_simulation(n_disks: u8, p: f64, t: f64, trials: usize) -> BenchmarkResult {
    let s = (1usize << n_disks) - 1; // 2^n - 1 steps
    let k = calculate_kmin(p, t, s, 1).unwrap();

    let mut total_samples_all = Vec::with_capacity(trials);
    let mut total_time_ms = 0u128;

    for _ in 0..trials {
        let start = Instant::now();
        let mut step_samples = 0usize;

        for _ in 0..s {
            let pool_size = 200;
            let correct_count = (pool_size as f64 * p).round() as usize;
            let mut responses = vec!["correct".to_string(); correct_count];
            responses.extend(vec!["wrong".to_string(); pool_size - correct_count]);

            let client = MockLlmClient::new(responses);
            let config = VoteConfig::default()
                .with_max_samples(pool_size)
                .without_token_limit();

            match vote_with_margin("bench", k, &client, config) {
                Ok(result) => step_samples += result.total_samples,
                Err(_) => step_samples += pool_size,
            }
        }

        total_time_ms += start.elapsed().as_millis();
        total_samples_all.push(step_samples);
    }

    let mean_samples = total_samples_all.iter().sum::<usize>() as f64 / trials as f64;

    BenchmarkResult {
        n_disks,
        steps: s,
        k_margin: k,
        mean_total_samples: mean_samples,
        mean_time_ms: total_time_ms as f64 / trials as f64,
        trials,
    }
}

#[derive(Debug)]
struct BenchmarkResult {
    n_disks: u8,
    steps: usize,
    k_margin: usize,
    mean_total_samples: f64,
    mean_time_ms: f64,
    trials: usize,
}

/// Compute R² for a linear fit y = a*x + b
fn linear_r_squared(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let sum_x2: f64 = x.iter().map(|xi| xi * xi).sum();

    let mean_y = sum_y / n;
    let ss_tot: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();

    // Linear regression: a = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x^2)
    let denom = n * sum_x2 - sum_x * sum_x;
    if denom.abs() < 1e-10 {
        return 0.0;
    }

    let a = (n * sum_xy - sum_x * sum_y) / denom;
    let b = (sum_y - a * sum_x) / n;

    let ss_res: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(xi, yi)| (yi - (a * xi + b)).powi(2))
        .sum();

    if ss_tot.abs() < 1e-10 {
        return 1.0;
    }

    1.0 - ss_res / ss_tot
}

fn main() {
    let p = 0.85;
    let t = 0.95;
    let trials = 20;

    println!("=== MAKER Cost Scaling Benchmark ===\n");
    println!("Parameters: p={}, t={}, trials={}", p, t, trials);
    println!();

    // Run benchmarks for different task sizes
    let disk_counts: Vec<u8> = vec![3, 5, 7];

    let mut results = Vec::new();

    for &n in &disk_counts {
        let result = run_hanoi_simulation(n, p, t, trials);
        println!(
            "n={:2} | steps={:>5} | k={} | mean_samples={:>8.0} | time={:>6.1}ms",
            result.n_disks,
            result.steps,
            result.k_margin,
            result.mean_total_samples,
            result.mean_time_ms,
        );
        results.push(result);
    }

    println!();

    // Fit to Θ(s ln s) model
    // Transform: x = s * ln(s), y = total_samples
    // If y = a * s * ln(s) + b, then regress y on x = s*ln(s)
    let x_vals: Vec<f64> = results
        .iter()
        .map(|r| r.steps as f64 * (r.steps as f64).ln())
        .collect();
    let y_vals: Vec<f64> = results.iter().map(|r| r.mean_total_samples).collect();

    let r_squared = linear_r_squared(&x_vals, &y_vals);

    println!("=== Cost Scaling Analysis ===\n");
    println!("Fitting cost to Θ(s ln s) model:");
    for (i, r) in results.iter().enumerate() {
        println!(
            "  s={:>5}, s·ln(s)={:>10.1}, samples={:>8.0}",
            r.steps, x_vals[i], y_vals[i]
        );
    }
    println!();
    println!("R² = {:.4}", r_squared);

    if r_squared > 0.95 {
        println!("✓ Cost scaling matches Θ(s ln s) (R² > 0.95)");
    } else if r_squared > 0.80 {
        println!("~ Cost scaling approximately matches Θ(s ln s) (R² > 0.80)");
    } else {
        println!("✗ Cost scaling deviates from Θ(s ln s) (R² < 0.80)");
    }

    // Export results as JSON
    let json_results: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "n_disks": r.n_disks,
                "steps": r.steps,
                "k_margin": r.k_margin,
                "mean_total_samples": r.mean_total_samples,
                "mean_time_ms": r.mean_time_ms,
                "trials": r.trials,
            })
        })
        .collect();

    let output = serde_json::json!({
        "parameters": { "p": p, "t": t, "trials": trials },
        "results": json_results,
        "analysis": {
            "r_squared": r_squared,
            "model": "Θ(s ln s)",
        }
    });

    println!("\n=== JSON Output ===\n");
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
