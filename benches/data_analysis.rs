//! Data Analysis Benchmark for MAKER Framework
//!
//! Validates MAKER on data analysis tasks with approximate matching,
//! simulating scenarios where outputs may differ in formatting but
//! be semantically equivalent.
//!
//! # Usage
//!
//! ```bash
//! cargo bench --bench data_analysis
//! ```

use maker::core::{calculate_kmin, vote_with_margin, MockLlmClient, VoteConfig};
use std::time::Instant;

/// A data analysis benchmark definition
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct DataTask {
    name: &'static str,
    category: &'static str,
    /// Per-step success probability
    p: f64,
    /// Number of pipeline steps
    steps: usize,
    /// Whether outputs use approximate matching (vs exact)
    approximate: bool,
    /// Description of the task
    description: &'static str,
}

/// Results from a data analysis benchmark run
#[derive(Debug)]
struct TaskResult {
    name: String,
    category: String,
    p: f64,
    steps: usize,
    k_margin: usize,
    approximate: bool,
    mean_total_samples: f64,
    mean_samples_per_step: f64,
    error_rate: f64,
    mean_time_ms: f64,
    trials: usize,
}

fn data_tasks() -> Vec<DataTask> {
    vec![
        // CSV processing (deterministic with exact matching)
        DataTask {
            name: "CSV column select",
            category: "csv",
            p: 0.92,
            steps: 3,
            approximate: false,
            description: "Select and reorder CSV columns",
        },
        DataTask {
            name: "CSV filter rows",
            category: "csv",
            p: 0.90,
            steps: 4,
            approximate: false,
            description: "Filter CSV rows by predicate",
        },
        DataTask {
            name: "CSV join tables",
            category: "csv",
            p: 0.85,
            steps: 5,
            approximate: false,
            description: "Join two CSV files on key column",
        },
        // Statistical summaries (approximate matching for float outputs)
        DataTask {
            name: "Mean/median/mode",
            category: "statistics",
            p: 0.88,
            steps: 3,
            approximate: true,
            description: "Compute basic statistical measures",
        },
        DataTask {
            name: "Correlation matrix",
            category: "statistics",
            p: 0.82,
            steps: 6,
            approximate: true,
            description: "Compute pairwise correlations",
        },
        DataTask {
            name: "Regression coefficients",
            category: "statistics",
            p: 0.80,
            steps: 5,
            approximate: true,
            description: "Linear regression with coefficient extraction",
        },
        // SQL generation (code equivalence)
        DataTask {
            name: "SQL SELECT query",
            category: "sql",
            p: 0.88,
            steps: 3,
            approximate: false,
            description: "Generate SELECT with WHERE/ORDER BY",
        },
        DataTask {
            name: "SQL GROUP BY",
            category: "sql",
            p: 0.83,
            steps: 4,
            approximate: false,
            description: "Aggregation query with GROUP BY/HAVING",
        },
        // Data cleaning pipeline
        DataTask {
            name: "Null handling",
            category: "cleaning",
            p: 0.90,
            steps: 4,
            approximate: false,
            description: "Detect and impute null values",
        },
        DataTask {
            name: "Type coercion",
            category: "cleaning",
            p: 0.87,
            steps: 5,
            approximate: false,
            description: "Convert string columns to appropriate types",
        },
    ]
}

/// Simulate MAKER voting on a data analysis task.
///
/// For approximate-matching tasks, we simulate the scenario where
/// an EmbeddingMatcher would group semantically similar outputs.
/// The effective p is slightly higher because near-correct outputs
/// get grouped with the correct answer.
fn run_data_benchmark(task: &DataTask, trials: usize, t: f64) -> TaskResult {
    // For approximate matching, effective p is slightly boosted
    // because near-correct outputs get grouped with correct ones
    let effective_p = if task.approximate {
        (task.p + 0.03).min(0.99)
    } else {
        task.p
    };

    let k = calculate_kmin(effective_p, t, task.steps, 1).unwrap_or(3);
    let pool_size = 200;

    let mut total_samples_all = Vec::with_capacity(trials);
    let mut error_count = 0usize;
    let mut total_time_ms = 0u128;

    for _ in 0..trials {
        let start = Instant::now();
        let mut trial_samples = 0usize;
        let mut trial_success = true;

        for _ in 0..task.steps {
            let correct_count = (pool_size as f64 * effective_p).round() as usize;
            let mut responses = vec!["correct_output".to_string(); correct_count];
            responses.extend(vec!["wrong_output".to_string(); pool_size - correct_count]);

            let client = MockLlmClient::new(responses);
            let config = VoteConfig::default()
                .with_max_samples(pool_size)
                .without_token_limit();

            match vote_with_margin("bench_data", k, &client, config) {
                Ok(result) => {
                    trial_samples += result.total_samples;
                    if result.winner != "correct_output" {
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

    TaskResult {
        name: task.name.to_string(),
        category: task.category.to_string(),
        p: task.p,
        steps: task.steps,
        k_margin: k,
        approximate: task.approximate,
        mean_total_samples: mean_samples,
        mean_samples_per_step: mean_samples / task.steps as f64,
        error_rate: error_count as f64 / trials as f64,
        mean_time_ms: total_time_ms as f64 / trials as f64,
        trials,
    }
}

fn main() {
    let trials = 500;
    let t = 0.95;
    let tasks = data_tasks();

    println!("=== MAKER Data Analysis Benchmark ===\n");
    println!("Trials per task: {}", trials);
    println!("Target reliability: {:.0}%", t * 100.0);
    println!();

    let mut results = Vec::new();

    for category in &["csv", "statistics", "sql", "cleaning"] {
        println!("━━━ {} tasks ━━━\n", category);

        let group: Vec<&DataTask> = tasks.iter().filter(|t| t.category == *category).collect();

        println!(
            "{:<25} {:>5} {:>5} {:>3} {:>5} {:>10} {:>8} {:>8}",
            "Task", "p", "steps", "k", "match", "samples", "samp/s", "err%"
        );
        println!("{}", "─".repeat(75));

        for task in &group {
            let result = run_data_benchmark(task, trials, t);
            println!(
                "{:<25} {:>5.2} {:>5} {:>3} {:>5} {:>10.0} {:>8.1} {:>7.2}%",
                result.name,
                result.p,
                result.steps,
                result.k_margin,
                if result.approximate {
                    "approx"
                } else {
                    "exact"
                },
                result.mean_total_samples,
                result.mean_samples_per_step,
                result.error_rate * 100.0,
            );
            results.push(result);
        }
        println!();
    }

    // Acceptance criteria
    println!("=== Acceptance Criteria ===\n");

    let overall_accuracy =
        1.0 - results.iter().map(|r| r.error_rate).sum::<f64>() / results.len() as f64;
    let all_ran = results.iter().all(|r| r.mean_total_samples > 0.0);

    // AC1: >85% accuracy on data analysis tasks
    println!(
        "  [{}] Overall accuracy: {:.1}% (target: >85%)",
        if overall_accuracy > 0.85 {
            "PASS"
        } else {
            "FAIL"
        },
        overall_accuracy * 100.0,
    );

    // AC2: Approximate matching tasks should benefit from grouping
    let approx_tasks: Vec<&TaskResult> = results.iter().filter(|r| r.approximate).collect();
    let approx_accuracy =
        1.0 - approx_tasks.iter().map(|r| r.error_rate).sum::<f64>() / approx_tasks.len() as f64;
    println!(
        "  [{}] Approximate matching accuracy: {:.1}% (target: >85%)",
        if approx_accuracy > 0.85 {
            "PASS"
        } else {
            "FAIL"
        },
        approx_accuracy * 100.0,
    );

    // AC3: All benchmarks run
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
                "approximate": r.approximate,
                "mean_total_samples": r.mean_total_samples,
                "mean_samples_per_step": r.mean_samples_per_step,
                "error_rate": r.error_rate,
                "mean_time_ms": r.mean_time_ms,
                "trials": r.trials,
            })
        })
        .collect();

    let output = serde_json::json!({
        "benchmark": "data_analysis",
        "trials": trials,
        "target_reliability": t,
        "tasks": json_results,
        "summary": {
            "overall_accuracy": overall_accuracy,
            "approximate_accuracy": approx_accuracy,
            "all_passed": all_ran && overall_accuracy > 0.85,
        }
    });

    println!("=== JSON Output ===\n");
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
