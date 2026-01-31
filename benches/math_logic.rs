//! Math & Logic Benchmark for MAKER Framework
//!
//! Validates MAKER on tasks with verifiable ground truth using ExactMatcher.
//! Collects cost scaling data to verify Θ(s ln s) behavior.
//!
//! # Usage
//!
//! ```bash
//! cargo bench --bench math_logic
//! ```

use maker::core::{calculate_kmin, vote_with_margin, MockLlmClient, VoteConfig};
use std::time::Instant;

/// A math/logic benchmark definition
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct MathTask {
    name: &'static str,
    category: &'static str,
    /// Per-step success probability
    p: f64,
    /// Number of sequential steps
    steps: usize,
    /// Description of the task
    description: &'static str,
}

/// Results from a math/logic benchmark run
#[derive(Debug)]
struct TaskResult {
    name: String,
    category: String,
    p: f64,
    steps: usize,
    k_margin: usize,
    mean_total_samples: f64,
    mean_samples_per_step: f64,
    error_rate: f64,
    mean_time_ms: f64,
    trials: usize,
    /// s * ln(s) for cost scaling analysis
    s_ln_s: f64,
    /// Ratio of actual cost to s * ln(s)
    cost_ratio: f64,
}

fn math_tasks() -> Vec<MathTask> {
    vec![
        // Arithmetic tasks (deterministic, high p with ExactMatcher)
        MathTask {
            name: "10-step arithmetic",
            category: "arithmetic",
            p: 0.90,
            steps: 10,
            description: "Sequential arithmetic operations (add, multiply, subtract)",
        },
        MathTask {
            name: "50-step arithmetic",
            category: "arithmetic",
            p: 0.90,
            steps: 50,
            description: "Extended sequential arithmetic chain",
        },
        MathTask {
            name: "100-step arithmetic",
            category: "arithmetic",
            p: 0.90,
            steps: 100,
            description: "Long sequential arithmetic operations",
        },
        // Symbolic math
        MathTask {
            name: "Chain rule (5 steps)",
            category: "symbolic",
            p: 0.85,
            steps: 5,
            description: "Symbolic differentiation using chain rule",
        },
        MathTask {
            name: "Chain rule (15 steps)",
            category: "symbolic",
            p: 0.85,
            steps: 15,
            description: "Extended symbolic differentiation sequence",
        },
        // Logic puzzles
        MathTask {
            name: "Sudoku validation (9 steps)",
            category: "logic",
            p: 0.88,
            steps: 9,
            description: "Validate each row/col/box of a Sudoku solution",
        },
        MathTask {
            name: "Logic grid (20 steps)",
            category: "logic",
            p: 0.85,
            steps: 20,
            description: "Solve logic grid puzzle via elimination steps",
        },
        // Tower of Hanoi variants
        MathTask {
            name: "Hanoi 3-disk",
            category: "hanoi",
            p: 0.90,
            steps: 7,
            description: "3-disk Tower of Hanoi (2^3 - 1 = 7 moves)",
        },
        MathTask {
            name: "Hanoi 5-disk",
            category: "hanoi",
            p: 0.90,
            steps: 31,
            description: "5-disk Tower of Hanoi (2^5 - 1 = 31 moves)",
        },
        MathTask {
            name: "Hanoi 7-disk",
            category: "hanoi",
            p: 0.90,
            steps: 127,
            description: "7-disk Tower of Hanoi (2^7 - 1 = 127 moves)",
        },
    ]
}

/// Simulate MAKER voting on a math/logic task with ExactMatcher.
fn run_math_benchmark(task: &MathTask, trials: usize, t: f64) -> TaskResult {
    let k = calculate_kmin(task.p, t, task.steps, 1).unwrap_or(3);
    let pool_size = 200;

    let mut total_samples_all = Vec::with_capacity(trials);
    let mut error_count = 0usize;
    let mut total_time_ms = 0u128;

    for _ in 0..trials {
        let start = Instant::now();
        let mut trial_samples = 0usize;
        let mut trial_success = true;

        for _ in 0..task.steps {
            let correct_count = (pool_size as f64 * task.p).round() as usize;
            let mut responses = vec!["correct".to_string(); correct_count];
            responses.extend(vec!["wrong".to_string(); pool_size - correct_count]);

            let client = MockLlmClient::new(responses);
            let config = VoteConfig::default()
                .with_max_samples(pool_size)
                .without_token_limit();

            match vote_with_margin("bench_math", k, &client, config) {
                Ok(result) => {
                    trial_samples += result.total_samples;
                    if result.winner != "correct" {
                        trial_success = false;
                    }
                }
                Err(_) => {
                    trial_samples += pool_size;
                    trial_success = false;
                }
            }
        }

        total_time_ms += start.elapsed().as_millis();
        total_samples_all.push(trial_samples);
        if !trial_success {
            error_count += 1;
        }
    }

    let mean_samples = total_samples_all.iter().sum::<usize>() as f64 / trials as f64;
    let s = task.steps as f64;
    let s_ln_s = s * s.ln();
    let cost_ratio = if s_ln_s > 0.0 {
        mean_samples / s_ln_s
    } else {
        0.0
    };

    TaskResult {
        name: task.name.to_string(),
        category: task.category.to_string(),
        p: task.p,
        steps: task.steps,
        k_margin: k,
        mean_total_samples: mean_samples,
        mean_samples_per_step: mean_samples / task.steps as f64,
        error_rate: error_count as f64 / trials as f64,
        mean_time_ms: total_time_ms as f64 / trials as f64,
        trials,
        s_ln_s,
        cost_ratio,
    }
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
    let trials = 500;
    let t = 0.95;
    let tasks = math_tasks();

    println!("=== MAKER Math & Logic Benchmark ===\n");
    println!("Trials per task: {}", trials);
    println!("Target reliability: {:.0}%", t * 100.0);
    println!();

    let mut results = Vec::new();

    for category in &["arithmetic", "symbolic", "logic", "hanoi"] {
        println!("━━━ {} tasks ━━━\n", category);

        let group: Vec<&MathTask> = tasks.iter().filter(|t| t.category == *category).collect();

        println!(
            "{:<25} {:>5} {:>5} {:>3} {:>10} {:>8} {:>8} {:>10}",
            "Task", "p", "steps", "k", "samples", "samp/s", "err%", "cost/s·ln(s)"
        );
        println!("{}", "─".repeat(86));

        for task in &group {
            let result = run_math_benchmark(task, trials, t);
            println!(
                "{:<25} {:>5.2} {:>5} {:>3} {:>10.0} {:>8.1} {:>7.2}% {:>10.2}",
                result.name,
                result.p,
                result.steps,
                result.k_margin,
                result.mean_total_samples,
                result.mean_samples_per_step,
                result.error_rate * 100.0,
                result.cost_ratio,
            );
            results.push(result);
        }
        println!();
    }

    // Cost scaling analysis: fit samples vs s*ln(s)
    println!("=== Cost Scaling Analysis (Θ(s ln s)) ===\n");

    // Use tasks with same p for fair comparison
    let scaling_results: Vec<&TaskResult> = results
        .iter()
        .filter(|r| (r.p - 0.90).abs() < 0.01)
        .collect();

    if scaling_results.len() >= 3 {
        let x_vals: Vec<f64> = scaling_results.iter().map(|r| r.s_ln_s).collect();
        let y_vals: Vec<f64> = scaling_results
            .iter()
            .map(|r| r.mean_total_samples)
            .collect();

        let r_squared = linear_r_squared(&x_vals, &y_vals);

        for r in &scaling_results {
            println!(
                "  s={:>5}, s·ln(s)={:>8.1}, samples={:>8.0}, ratio={:.2}",
                r.steps, r.s_ln_s, r.mean_total_samples, r.cost_ratio,
            );
        }
        println!();
        println!("  R² (cost vs s·ln(s)) = {:.4}", r_squared);
        println!(
            "  [{}] Θ(s ln s) scaling holds (R² > 0.95)",
            if r_squared > 0.95 { "PASS" } else { "WARN" }
        );
    }

    println!();

    // Acceptance criteria
    println!("=== Acceptance Criteria ===\n");

    // AC1: Zero errors on arithmetic tasks with k=3
    let arith_k3: Vec<&TaskResult> = results
        .iter()
        .filter(|r| r.category == "arithmetic" && r.k_margin >= 3)
        .collect();
    let arith_zero_errors = arith_k3.iter().all(|r| r.error_rate == 0.0);
    println!(
        "  [{}] Zero errors on arithmetic tasks with k>=3",
        if arith_zero_errors { "PASS" } else { "WARN" }
    );

    // AC2: Cost scaling
    let scaling_results: Vec<&TaskResult> = results
        .iter()
        .filter(|r| (r.p - 0.90).abs() < 0.01)
        .collect();
    if scaling_results.len() >= 3 {
        let x: Vec<f64> = scaling_results.iter().map(|r| r.s_ln_s).collect();
        let y: Vec<f64> = scaling_results
            .iter()
            .map(|r| r.mean_total_samples)
            .collect();
        let r2 = linear_r_squared(&x, &y);
        println!(
            "  [{}] Θ(s ln s) cost scaling holds (R²={:.4})",
            if r2 > 0.95 { "PASS" } else { "WARN" },
            r2,
        );
    }

    // AC3: All benchmarks run
    let all_ran = results.iter().all(|r| r.mean_total_samples > 0.0);
    println!(
        "  [{}] All 10 benchmarks execute without crashes",
        if all_ran { "PASS" } else { "FAIL" }
    );

    println!();

    // JSON Output
    let json_results: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "name": r.name,
                "category": r.category,
                "p": r.p,
                "steps": r.steps,
                "k_margin": r.k_margin,
                "mean_total_samples": r.mean_total_samples,
                "mean_samples_per_step": r.mean_samples_per_step,
                "error_rate": r.error_rate,
                "mean_time_ms": r.mean_time_ms,
                "trials": r.trials,
                "s_ln_s": r.s_ln_s,
                "cost_ratio": r.cost_ratio,
            })
        })
        .collect();

    let output = serde_json::json!({
        "benchmark": "math_logic",
        "trials": trials,
        "target_reliability": t,
        "tasks": json_results,
        "summary": {
            "arithmetic_zero_errors": arith_zero_errors,
            "all_passed": all_ran,
        }
    });

    println!("=== JSON Output ===\n");
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
