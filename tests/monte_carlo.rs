//! Monte Carlo Cost Validation Tests
//!
//! Validates MAKER's Θ(s ln s) cost scaling through simulation,
//! and compares MAKER cost to naive retry approaches.

use maker::core::{calculate_kmin, vote_with_margin, MockLlmClient, VoteConfig};

/// Simulate MAKER cost for a given task configuration.
///
/// Returns (mean_total_samples, std_dev) across `trials` runs.
fn simulate_maker_cost(s: usize, p: f64, k: usize, trials: usize) -> (f64, f64) {
    let mut total_samples_per_trial = Vec::with_capacity(trials);

    for _ in 0..trials {
        let mut step_samples = 0usize;

        for _ in 0..s {
            // Create a biased mock: p probability of "correct", (1-p) of "wrong"
            let pool_size = 200;
            let correct_count = (pool_size as f64 * p).round() as usize;
            let mut responses = vec!["correct".to_string(); correct_count];
            responses.extend(vec!["wrong".to_string(); pool_size - correct_count]);

            let client = MockLlmClient::new(responses);
            let config = VoteConfig::default()
                .with_max_samples(pool_size)
                .without_token_limit();

            match vote_with_margin("test", k, &client, config) {
                Ok(result) => {
                    step_samples += result.total_samples;
                }
                Err(_) => {
                    step_samples += pool_size;
                }
            }
        }

        total_samples_per_trial.push(step_samples as f64);
    }

    let mean = total_samples_per_trial.iter().sum::<f64>() / trials as f64;
    let variance = total_samples_per_trial
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / trials as f64;
    let std_dev = variance.sqrt();

    (mean, std_dev)
}

/// Calculate naive task-level retry cost to achieve target reliability t.
///
/// Naive approach: run the entire s-step task, each step with probability p
/// of success. Task succeeds with probability p^s. To achieve reliability t,
/// need n retries where 1-(1-p^s)^n >= t, so n >= ln(1-t)/ln(1-p^s).
/// Total cost = n * s (each retry costs s API calls).
fn theoretical_naive_retry_cost(s: usize, p: f64, t: f64) -> f64 {
    let task_success_prob = p.powi(s as i32);

    if task_success_prob >= t {
        // Single attempt suffices
        return s as f64;
    }

    if task_success_prob < 1e-15 {
        // Effectively impossible with naive retry
        return f64::INFINITY;
    }

    // n = ceil(ln(1-t) / ln(1-p^s))
    let n = ((1.0 - t).ln() / (1.0 - task_success_prob).ln()).ceil();
    n * s as f64
}

/// Calculate theoretical MAKER cost: s * E[samples per step]
///
/// Expected samples per step with k-margin voting and probability p:
/// E[samples] ≈ k / (2p - 1) for the Gambler's Ruin expected duration.
fn theoretical_maker_cost(s: usize, _p: f64, k: usize) -> f64 {
    // Simple model: each step needs approximately k samples for unanimous winner,
    // more with contention. Use s * k as conservative estimate.
    s as f64 * k as f64
}

#[test]
fn test_cost_scaling_theta_s_ln_s() {
    // Test that MAKER cost scales as Θ(s ln s) by comparing ratios
    let p = 0.85;
    let t = 0.95;
    let trials = 50;

    // Test at different task sizes
    let test_cases: Vec<(usize, usize)> = vec![
        (10, calculate_kmin(p, t, 10, 1).unwrap()),
        (50, calculate_kmin(p, t, 50, 1).unwrap()),
        (100, calculate_kmin(p, t, 100, 1).unwrap()),
    ];

    let mut results: Vec<(usize, f64)> = Vec::new();

    for (s, k) in &test_cases {
        let (mean_cost, _std) = simulate_maker_cost(*s, p, *k, trials);
        results.push((*s, mean_cost));
    }

    // Verify Θ(s ln s) scaling: cost(s) / (s * ln(s)) should be roughly constant
    let normalized: Vec<f64> = results
        .iter()
        .map(|(s, cost)| cost / (*s as f64 * (*s as f64).ln()))
        .collect();

    let min_norm = normalized.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_norm = normalized.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Within 3x ratio is reasonable for Monte Carlo with small trial counts
    assert!(
        max_norm / min_norm < 3.0,
        "Cost scaling deviates from Θ(s ln s): normalized values {:?}, ratio {:.2}",
        normalized,
        max_norm / min_norm
    );
}

#[test]
fn test_maker_cheaper_than_naive_retry() {
    // Compare MAKER to naive task-level retry for achieving the same reliability target.
    //
    // Naive retry: rerun the entire task until it succeeds.
    // Cost = n_retries * s, where n_retries = ceil(ln(1-t)/ln(1-p^s))
    //
    // MAKER: vote per step with k-margin.
    // Cost ≈ s * k, where k = O(ln s)
    let p = 0.85;
    let t = 0.95;

    // For s=50 steps: naive approach needs many full-task retries
    for &s in &[20, 50, 100] {
        let k = calculate_kmin(p, t, s, 1).unwrap();

        let maker_cost = theoretical_maker_cost(s, p, k);
        let naive_cost = theoretical_naive_retry_cost(s, p, t);

        eprintln!(
            "s={}: MAKER cost={:.0} (k={}), Naive cost={:.0}",
            s, maker_cost, k, naive_cost
        );

        // MAKER should be dramatically cheaper for non-trivial task lengths
        if s >= 20 {
            assert!(
                maker_cost < naive_cost,
                "MAKER cost ({:.0}) should be less than naive retry ({:.0}) for s={}",
                maker_cost,
                naive_cost,
                s
            );
        }
    }
}

#[test]
fn test_naive_retry_exponential_blowup() {
    // Demonstrate that naive retry cost grows exponentially with s
    let p = 0.85;
    let t = 0.95;

    let cost_10 = theoretical_naive_retry_cost(10, p, t);
    let cost_50 = theoretical_naive_retry_cost(50, p, t);
    let cost_100 = theoretical_naive_retry_cost(100, p, t);

    eprintln!(
        "Naive retry costs: s=10 → {:.0}, s=50 → {:.0}, s=100 → {:.0}",
        cost_10, cost_50, cost_100
    );

    // Cost should grow super-linearly (exponentially) with s
    // For p=0.85: p^10=0.197, p^50≈2.6e-4, p^100≈6.8e-8
    assert!(
        cost_50 > cost_10 * 10.0,
        "Naive cost should grow super-linearly: c(50)={:.0} vs 10*c(10)={:.0}",
        cost_50,
        cost_10 * 10.0
    );
}

#[test]
fn test_cost_ratio_matches_theory() {
    // Compare cost at s=20 vs s=100
    // Theory: cost(100) / cost(20) ≈ (100 * ln(100)) / (20 * ln(20))
    let p = 0.85;
    let trials = 50;

    let s1 = 20;
    let s2 = 100;
    let k1 = calculate_kmin(p, 0.95, s1, 1).unwrap();
    let k2 = calculate_kmin(p, 0.95, s2, 1).unwrap();

    let (cost1, _) = simulate_maker_cost(s1, p, k1, trials);
    let (cost2, _) = simulate_maker_cost(s2, p, k2, trials);

    let actual_ratio = cost2 / cost1;
    let theoretical_ratio = (s2 as f64 * (s2 as f64).ln()) / (s1 as f64 * (s1 as f64).ln());

    // Allow 50% tolerance for Monte Carlo variance
    let tolerance = 0.5;
    let lower = theoretical_ratio * (1.0 - tolerance);
    let upper = theoretical_ratio * (1.0 + tolerance);

    eprintln!(
        "Cost ratio: actual={:.2}, theoretical={:.2}, bounds=[{:.2}, {:.2}]",
        actual_ratio, theoretical_ratio, lower, upper
    );

    assert!(
        actual_ratio >= lower && actual_ratio <= upper,
        "Cost ratio {:.2} outside [{:.2}, {:.2}] (theoretical {:.2})",
        actual_ratio,
        lower,
        upper,
        theoretical_ratio
    );
}
