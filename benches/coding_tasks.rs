//! Coding Task Benchmark for MAKER Framework
//!
//! Validates MAKER's reliability on coding tasks of varying difficulty,
//! using simulated LLM responses with domain-appropriate success rates.
//!
//! # Usage
//!
//! ```bash
//! cargo bench --bench coding_tasks
//! ```

use maker::core::{calculate_kmin, vote_with_margin, MockLlmClient, VoteConfig};
use std::time::Instant;

/// A coding task benchmark definition
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CodingTask {
    name: &'static str,
    difficulty: &'static str,
    /// Simulated per-step success probability for this task difficulty
    p: f64,
    /// Number of subtask steps in decomposition
    steps: usize,
    /// Description of what the task involves
    description: &'static str,
}

/// Results from running a coding task benchmark
#[derive(Debug)]
struct TaskResult {
    name: String,
    difficulty: String,
    p: f64,
    steps: usize,
    k_margin: usize,
    mean_total_samples: f64,
    mean_samples_per_step: f64,
    error_rate: f64,
    red_flag_rate: f64,
    mean_time_ms: f64,
    trials: usize,
}

/// Coding tasks spanning trivial to complex difficulty
fn coding_tasks() -> Vec<CodingTask> {
    vec![
        // Trivial tasks (p ≈ 0.95)
        CodingTask {
            name: "FizzBuzz",
            difficulty: "trivial",
            p: 0.95,
            steps: 3,
            description: "Generate FizzBuzz for 1-100",
        },
        CodingTask {
            name: "Hello World variants",
            difficulty: "trivial",
            p: 0.96,
            steps: 2,
            description: "Hello World in multiple languages",
        },
        // Moderate tasks (p ≈ 0.85)
        CodingTask {
            name: "Binary Search",
            difficulty: "moderate",
            p: 0.85,
            steps: 5,
            description: "Implement binary search with edge cases",
        },
        CodingTask {
            name: "Linked List Reversal",
            difficulty: "moderate",
            p: 0.83,
            steps: 4,
            description: "Reverse a singly linked list in-place",
        },
        CodingTask {
            name: "Stack Calculator",
            difficulty: "moderate",
            p: 0.84,
            steps: 6,
            description: "Evaluate postfix expressions with a stack",
        },
        CodingTask {
            name: "Merge Sort",
            difficulty: "moderate",
            p: 0.82,
            steps: 5,
            description: "Implement merge sort with proper splitting",
        },
        // Complex tasks (p ≈ 0.65-0.75)
        CodingTask {
            name: "JSON Parser",
            difficulty: "complex",
            p: 0.70,
            steps: 10,
            description: "Recursive descent JSON parser with error handling",
        },
        CodingTask {
            name: "SQL Query Generator",
            difficulty: "complex",
            p: 0.65,
            steps: 8,
            description: "Generate SQL from natural language spec",
        },
        CodingTask {
            name: "Regex Engine",
            difficulty: "complex",
            p: 0.68,
            steps: 12,
            description: "Simple regex matcher with *, +, ? operators",
        },
        CodingTask {
            name: "HTTP Router",
            difficulty: "complex",
            p: 0.72,
            steps: 7,
            description: "Path-based HTTP router with parameter extraction",
        },
    ]
}

/// Simulate MAKER voting on a coding task.
///
/// Uses MockLlmClient with a configurable success rate to simulate
/// how MAKER would perform on tasks where a code matcher groups
/// equivalent implementations.
fn run_coding_benchmark(task: &CodingTask, trials: usize, t: f64) -> TaskResult {
    let k = calculate_kmin(task.p, t, task.steps, 1).unwrap_or(3);
    let pool_size = 200;
    let red_flag_rate = match task.difficulty {
        "trivial" => 0.02,
        "moderate" => 0.05,
        "complex" => 0.10,
        _ => 0.05,
    };

    let mut total_samples_all = Vec::with_capacity(trials);
    let mut error_count = 0usize;
    let mut total_red_flags = 0usize;
    let mut total_sample_count = 0usize;
    let mut total_time_ms = 0u128;

    for _ in 0..trials {
        let start = Instant::now();
        let mut trial_samples = 0usize;
        let mut trial_success = true;

        for _ in 0..task.steps {
            // Build response pool: correct, incorrect, and red-flagged
            let red_flag_count = (pool_size as f64 * red_flag_rate).round() as usize;
            let remaining = pool_size - red_flag_count;
            let correct_count = (remaining as f64 * task.p).round() as usize;
            let incorrect_count = remaining - correct_count;

            let mut responses = Vec::with_capacity(pool_size);
            responses.extend(vec!["correct_impl".to_string(); correct_count]);
            responses.extend(vec!["incorrect_impl".to_string(); incorrect_count]);
            // Red-flagged responses are those that would be discarded
            // (in real use, these would fail validation). MockLlmClient
            // returns them but they would be caught by RedFlagValidator.
            // For simulation, we treat very short responses as red-flagged.
            responses.extend(vec!["".to_string(); red_flag_count]);

            let client = MockLlmClient::new(responses);
            let config = VoteConfig::default()
                .with_max_samples(pool_size)
                .without_token_limit();

            match vote_with_margin("bench_coding", k, &client, config) {
                Ok(result) => {
                    trial_samples += result.total_samples;
                    if result.winner != "correct_impl" {
                        trial_success = false;
                    }
                }
                Err(_) => {
                    trial_samples += pool_size;
                    trial_success = false;
                }
            }

            total_red_flags += red_flag_count;
            total_sample_count += pool_size;
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
        difficulty: task.difficulty.to_string(),
        p: task.p,
        steps: task.steps,
        k_margin: k,
        mean_total_samples: mean_samples,
        mean_samples_per_step: mean_samples / task.steps as f64,
        error_rate: error_count as f64 / trials as f64,
        red_flag_rate: total_red_flags as f64 / total_sample_count as f64,
        mean_time_ms: total_time_ms as f64 / trials as f64,
        trials,
    }
}

fn main() {
    let trials = 500;
    let t = 0.95;
    let tasks = coding_tasks();

    println!("=== MAKER Coding Task Benchmark ===\n");
    println!("Trials per task: {}", trials);
    println!("Target reliability: {:.0}%", t * 100.0);
    println!();

    let mut results = Vec::new();

    // Run benchmarks by difficulty group
    for difficulty in &["trivial", "moderate", "complex"] {
        println!("━━━ {} tasks ━━━\n", difficulty);

        let group: Vec<&CodingTask> = tasks
            .iter()
            .filter(|t| t.difficulty == *difficulty)
            .collect();

        println!(
            "{:<25} {:>5} {:>5} {:>3} {:>10} {:>8} {:>8} {:>7}",
            "Task", "p", "steps", "k", "samples", "samp/s", "err%", "rf%"
        );
        println!("{}", "─".repeat(77));

        for task in &group {
            let result = run_coding_benchmark(task, trials, t);
            println!(
                "{:<25} {:>5.2} {:>5} {:>3} {:>10.0} {:>8.1} {:>7.2}% {:>6.1}%",
                result.name,
                result.p,
                result.steps,
                result.k_margin,
                result.mean_total_samples,
                result.mean_samples_per_step,
                result.error_rate * 100.0,
                result.red_flag_rate * 100.0,
            );
            results.push(result);
        }
        println!();
    }

    // Acceptance criteria verification
    println!("=== Acceptance Criteria ===\n");

    let trivial_moderate: Vec<&TaskResult> = results
        .iter()
        .filter(|r| r.difficulty == "trivial" || r.difficulty == "moderate")
        .collect();
    let complex: Vec<&TaskResult> = results
        .iter()
        .filter(|r| r.difficulty == "complex")
        .collect();

    let tm_accuracy = 1.0
        - trivial_moderate.iter().map(|r| r.error_rate).sum::<f64>()
            / trivial_moderate.len() as f64;
    let complex_accuracy =
        1.0 - complex.iter().map(|r| r.error_rate).sum::<f64>() / complex.len() as f64;

    let no_crashes = results.iter().all(|r| r.mean_total_samples > 0.0);

    println!(
        "  [{}] All 10 benchmarks execute without crashes",
        if no_crashes { "PASS" } else { "FAIL" }
    );
    println!(
        "  [{}] Trivial/moderate accuracy: {:.1}% (target: >90%)",
        if tm_accuracy > 0.90 { "PASS" } else { "FAIL" },
        tm_accuracy * 100.0,
    );
    println!(
        "  [{}] Complex accuracy: {:.1}% (target: >80%)",
        if complex_accuracy > 0.80 {
            "PASS"
        } else {
            "FAIL"
        },
        complex_accuracy * 100.0,
    );
    println!();

    // JSON Output
    let json_results: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "name": r.name,
                "difficulty": r.difficulty,
                "p": r.p,
                "steps": r.steps,
                "k_margin": r.k_margin,
                "mean_total_samples": r.mean_total_samples,
                "mean_samples_per_step": r.mean_samples_per_step,
                "error_rate": r.error_rate,
                "red_flag_rate": r.red_flag_rate,
                "mean_time_ms": r.mean_time_ms,
                "trials": r.trials,
            })
        })
        .collect();

    let output = serde_json::json!({
        "benchmark": "coding_tasks",
        "trials": trials,
        "target_reliability": t,
        "tasks": json_results,
        "summary": {
            "trivial_moderate_accuracy": tm_accuracy,
            "complex_accuracy": complex_accuracy,
            "all_passed": no_crashes && tm_accuracy > 0.90 && complex_accuracy > 0.80,
        }
    });

    println!("=== JSON Output ===\n");
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
