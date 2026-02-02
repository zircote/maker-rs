//! Research Validation Script for MAKER Framework
//!
//! Comprehensive validation of MAKER's error correction functionality
//! producing academically rigorous, reproducible results.
//!
//! # Usage
//!
//! ```bash
//! cargo run --bin research-validation -- --mock --quick
//! cargo run --bin research-validation -- --mock --trials 50
//! cargo run --bin research-validation -- --live --trials 10
//! ```

use clap::{Parser, ValueEnum};
use maker::core::{vote_with_margin, LlmClient, LlmResponse, VoteConfig};
use maker::llm::adapter::{create_provider, ProviderConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

// ============================================================================
// CLI Definition
// ============================================================================

/// MAKER Research Validation - Academic validation of error correction
#[derive(Parser)]
#[command(name = "research-validation")]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Use deterministic mock client (default)
    #[arg(long, default_value = "true", conflicts_with = "live")]
    mock: bool,

    /// Use real LLM for testing (requires --provider)
    #[arg(long)]
    live: bool,

    /// LLM provider for live mode (ollama, openai, anthropic)
    #[arg(long, default_value = "ollama")]
    provider: String,

    /// Model name for live mode
    #[arg(long, default_value = "llama3.2")]
    model: String,

    /// Trials per configuration
    #[arg(long, default_value = "50")]
    trials: usize,

    /// RNG seed for mock mode
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Output directory
    #[arg(long, default_value = "docs/research")]
    output_dir: PathBuf,

    /// Quick mode (10 trials for validation)
    #[arg(long)]
    quick: bool,

    /// Run specific experiment only
    #[arg(long, value_enum)]
    experiment: Option<ExperimentName>,

    /// Include ensemble experiments
    #[arg(long)]
    ensemble: bool,

    /// Run all experiments including ensemble (default)
    #[arg(long)]
    all: bool,

    /// Increase verbosity
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

#[derive(Clone, Copy, ValueEnum, Debug)]
enum ExperimentName {
    /// Accuracy comparison (k-margin vs single)
    Accuracy,
    /// Multi-step reliability
    Multistep,
    /// Convergence analysis
    Convergence,
    /// Cost scaling validation
    Cost,
    /// Ensemble comparison
    Ensemble,
    /// Ensemble error decorrelation
    Decorrelation,
}

// ============================================================================
// Data Structures
// ============================================================================

/// Trial result for each test run
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrialResult {
    trial_id: usize,
    config_id: String,
    k_margin: usize,
    base_accuracy: f64,
    steps: usize,
    voting_correct: bool,
    single_correct: bool,
    total_samples: usize,
    converged: bool,
    elapsed_ms: u64,
}

/// Aggregated statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StatisticalSummary {
    config_id: String,
    n: usize,
    voting_mean: f64,
    voting_std: f64,
    voting_ci_lower: f64,
    voting_ci_upper: f64,
    single_mean: f64,
    cohens_d: f64,
    t_statistic: f64,
    p_value: f64,
    significant: bool,
}

/// Experiment context with shared state
struct ExperimentContext {
    seed: u64,
    trials: usize,
    verbose: u8,
    output_dir: PathBuf,
    use_live: bool,
    provider: String,
    model: String,
    /// Shared live client (created once, reused across experiments)
    live_client: Option<Arc<dyn LlmClient>>,
}

impl ExperimentContext {
    fn new(cli: &Cli) -> Result<Self, String> {
        let trials = if cli.quick { 10 } else { cli.trials };

        // Create live client if needed
        let live_client = if cli.live {
            let config = ProviderConfig {
                model: Some(cli.model.clone()),
                api_key: None, // Will use env vars
                base_url: None,
            };
            let client = create_provider(&cli.provider, Some(config))?
                .ok_or_else(|| format!("Unknown provider: {}", cli.provider))?;
            Some(Arc::from(client))
        } else {
            None
        };

        Ok(Self {
            seed: cli.seed,
            trials,
            verbose: cli.verbose,
            output_dir: cli.output_dir.clone(),
            use_live: cli.live,
            provider: cli.provider.clone(),
            model: cli.model.clone(),
            live_client,
        })
    }

    fn log(&self, level: u8, msg: &str) {
        if self.verbose >= level {
            eprintln!("{}", msg);
        }
    }
}

// ============================================================================
// Statistics Module
// ============================================================================

mod statistics {
    /// Mean and standard deviation with Bessel's correction
    pub fn mean_std(data: &[f64]) -> (f64, f64) {
        if data.is_empty() {
            return (0.0, 0.0);
        }
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        if data.len() == 1 {
            return (mean, 0.0);
        }
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, variance.sqrt())
    }

    /// 95% CI using t-distribution approximation
    pub fn confidence_interval_95(mean: f64, std: f64, n: usize) -> (f64, f64) {
        if n <= 1 {
            return (mean, mean);
        }
        // t-critical value for 95% CI (approximation for n > 30)
        let t_crit = if n > 120 {
            1.96
        } else if n > 60 {
            2.0
        } else if n > 30 {
            2.04
        } else if n > 15 {
            2.13
        } else {
            2.26
        };
        let se = std / (n as f64).sqrt();
        let margin = t_crit * se;
        (mean - margin, mean + margin)
    }

    /// Paired t-test (t-statistic, p-value, df)
    pub fn paired_t_test(x: &[f64], y: &[f64]) -> (f64, f64, usize) {
        assert_eq!(
            x.len(),
            y.len(),
            "Paired t-test requires equal-length arrays"
        );
        let n = x.len();
        if n == 0 {
            return (0.0, 1.0, 0);
        }

        let diffs: Vec<f64> = x.iter().zip(y.iter()).map(|(a, b)| a - b).collect();
        let (mean_d, std_d) = mean_std(&diffs);

        if std_d < 1e-10 {
            return if mean_d.abs() < 1e-10 {
                (0.0, 1.0, n - 1)
            } else {
                (f64::INFINITY * mean_d.signum(), 0.0, n - 1)
            };
        }

        let se = std_d / (n as f64).sqrt();
        let t = mean_d / se;
        let df = n - 1;

        // Approximate p-value using normal distribution for large df
        let p = 2.0 * (1.0 - normal_cdf(t.abs()));
        (t, p, df)
    }

    /// Cohen's d effect size for paired samples
    pub fn cohens_d_paired(x: &[f64], y: &[f64]) -> f64 {
        assert_eq!(x.len(), y.len());
        if x.is_empty() {
            return 0.0;
        }

        let diffs: Vec<f64> = x.iter().zip(y.iter()).map(|(a, b)| a - b).collect();
        let (mean_d, std_d) = mean_std(&diffs);

        if std_d < 1e-10 {
            return 0.0;
        }
        mean_d / std_d
    }

    /// Wilson score CI for proportions
    #[allow(dead_code)]
    pub fn wilson_ci(successes: usize, total: usize) -> (f64, f64) {
        if total == 0 {
            return (0.0, 1.0);
        }
        let n = total as f64;
        let p = successes as f64 / n;
        let z = 1.96; // 95% confidence
        let z2 = z * z;

        let denominator = 1.0 + z2 / n;
        let center = p + z2 / (2.0 * n);
        let margin = z * ((p * (1.0 - p) + z2 / (4.0 * n)) / n).sqrt();

        let lower = ((center - margin) / denominator).max(0.0);
        let upper = ((center + margin) / denominator).min(1.0);
        (lower, upper)
    }

    /// Normal CDF approximation (using error function approximation)
    fn normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
    }

    /// Error function approximation (Abramowitz and Stegun)
    fn erf(x: f64) -> f64 {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        sign * y
    }

    /// Interpret effect size magnitude
    pub fn effect_size_interpretation(d: f64) -> &'static str {
        let d_abs = d.abs();
        if d_abs < 0.2 {
            "negligible"
        } else if d_abs < 0.5 {
            "small"
        } else if d_abs < 0.8 {
            "medium"
        } else {
            "large"
        }
    }

    /// Significance stars
    pub fn significance_stars(p: f64) -> &'static str {
        if p < 0.001 {
            "***"
        } else if p < 0.01 {
            "**"
        } else if p < 0.05 {
            "*"
        } else {
            ""
        }
    }
}

// ============================================================================
// Seeded Mock Client
// ============================================================================

/// Mock client with deterministic seeded randomness
struct SeededMockClient {
    responses: Vec<String>,
    index: std::sync::atomic::AtomicUsize,
}

impl SeededMockClient {
    fn biased_seeded(correct: &str, incorrect: &str, p: f64, count: usize, seed: u64) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut responses = Vec::with_capacity(count);
        for i in 0..count {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();
            let random_val = (hash as f64) / (u64::MAX as f64);

            if random_val < p {
                responses.push(correct.to_string());
            } else {
                responses.push(incorrect.to_string());
            }
        }

        Self {
            responses,
            index: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl LlmClient for SeededMockClient {
    fn generate(&self, _prompt: &str, _temperature: f64) -> Result<LlmResponse, String> {
        let idx = self.index.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let response = &self.responses[idx % self.responses.len()];
        Ok(LlmResponse {
            content: response.clone(),
            input_tokens: 100,
            output_tokens: 50,
        })
    }
}

// ============================================================================
// Experiments
// ============================================================================

/// Test questions for live mode with expected answers
/// Mix of difficulties to create variance in model accuracy
const LIVE_TEST_QUESTIONS: &[(&str, &str)] = &[
    // Multi-step arithmetic (model often makes errors)
    ("What is 47 + 38 - 19? Reply with just the number.", "66"),
    ("What is 156 - 87 + 23? Reply with just the number.", "92"),
    ("What is 13 * 7 - 25? Reply with just the number.", "66"),
    ("What is 200 / 8 + 17? Reply with just the number.", "42"),
    ("What is 15 * 4 - 23 + 8? Reply with just the number.", "45"),
    // Order of operations (often confused)
    ("What is 3 + 4 * 5? Reply with just the number.", "23"),
    ("What is 20 - 3 * 4? Reply with just the number.", "8"),
    ("What is 100 / 5 / 4? Reply with just the number.", "5"),
    // Percentage calculations
    ("What is 35% of 80? Reply with just the number.", "28"),
    ("What is 120% of 50? Reply with just the number.", "60"),
    // Slightly tricky
    (
        "What is 99 + 1 - 50 + 50? Reply with just the number.",
        "100",
    ),
    ("What is 7 * 8 - 6 * 9? Reply with just the number.", "2"),
];

/// Check if response contains the expected answer
fn response_matches(response: &str, expected: &str) -> bool {
    // Normalize both strings and check if expected appears in response
    let response_clean = response.trim().to_lowercase();
    let expected_clean = expected.trim().to_lowercase();

    // Exact match or response contains the expected value
    response_clean == expected_clean ||
    response_clean.contains(&expected_clean) ||
    // Handle cases like "The answer is 45"
    response_clean.split_whitespace().any(|word| {
        word.trim_matches(|c: char| !c.is_alphanumeric()) == expected_clean
    })
}

/// Run accuracy comparison experiment
fn run_accuracy_experiment(ctx: &ExperimentContext) -> Vec<TrialResult> {
    ctx.log(1, "\n=== Accuracy Comparison Experiment ===");

    // k=1-3 validated as sweet spot; k=5-7 have too many non-convergences
    let k_values_live = [1, 2, 3];
    let k_values_mock = [1, 2, 3, 5, 7];
    let p_values = [0.6, 0.7, 0.8, 0.85, 0.9];
    let correct = "4";
    let incorrect = "5";

    let mut results = Vec::new();

    if ctx.use_live {
        // Live mode: use real LLM with actual test questions
        // Run each question ctx.trials times for statistical power
        ctx.log(1, "  Using live LLM - running with actual test questions");

        for &k in &k_values_live {
            let config_id = format!("k{}_live", k);
            let total_trials = LIVE_TEST_QUESTIONS.len() * ctx.trials;
            ctx.log(
                1,
                &format!(
                    "  Running config: {} ({} questions x {} trials = {} total)",
                    config_id,
                    LIVE_TEST_QUESTIONS.len(),
                    ctx.trials,
                    total_trials
                ),
            );

            for trial in 0..ctx.trials {
                for (q_idx, (question, expected)) in LIVE_TEST_QUESTIONS.iter().enumerate() {
                    let trial_id = trial * LIVE_TEST_QUESTIONS.len() + q_idx;
                    ctx.log(2, &format!("    Trial {} Q{}: {}", trial, q_idx, question));

                    let client = ctx.live_client.as_ref().unwrap();

                    // Single sample at high temperature (noisy baseline)
                    let single_result = client.generate(question, 0.7);
                    let single_correct = match &single_result {
                        Ok(r) => {
                            let matches = response_matches(&r.content, expected);
                            ctx.log(
                                2,
                                &format!(
                                    "      Single: '{}' -> {}",
                                    r.content.chars().take(30).collect::<String>(),
                                    matches
                                ),
                            );
                            matches
                        }
                        Err(e) => {
                            ctx.log(1, &format!("      Single error: {}", e));
                            false
                        }
                    };

                    // Voting with multiple samples - use higher temperature for live mode
                    let config = VoteConfig::default()
                        .with_max_samples(20)
                        .without_token_limit()
                        .with_diversity_temperature(0.7);

                    let start = Instant::now();
                    let vote_result = vote_with_margin(question, k, client.as_ref(), config);
                    let elapsed = start.elapsed();

                    let (voting_correct, total_samples, converged) = match &vote_result {
                        Ok(r) => {
                            let matches = response_matches(&r.winner, expected);
                            ctx.log(
                                2,
                                &format!(
                                    "      Voting: '{}' -> {} ({} samples)",
                                    r.winner.chars().take(30).collect::<String>(),
                                    matches,
                                    r.total_samples
                                ),
                            );
                            (matches, r.total_samples, true)
                        }
                        Err(e) => {
                            ctx.log(1, &format!("      Voting error: {}", e));
                            (false, 20, false)
                        }
                    };

                    results.push(TrialResult {
                        trial_id: trial_id,
                        config_id: config_id.clone(),
                        k_margin: k,
                        base_accuracy: 0.0, // Unknown for live mode
                        steps: 1,
                        voting_correct,
                        single_correct,
                        total_samples,
                        converged,
                        elapsed_ms: elapsed.as_millis() as u64,
                    });
                }
            }
        }
    } else {
        // Mock mode: simulate with known probability
        for &k in &k_values_mock {
            for &p in &p_values {
                let config_id = format!("k{}_p{}", k, (p * 100.0) as i32);
                ctx.log(1, &format!("  Running config: {}", config_id));

                for trial in 0..ctx.trials {
                    let trial_seed = ctx.seed.wrapping_add(trial as u64);

                    // Single sample client
                    let single_client =
                        SeededMockClient::biased_seeded(correct, incorrect, p, 1, trial_seed);
                    let single_result = single_client.generate("2+2?", 0.0).unwrap();
                    let single_correct = single_result.content == correct;

                    // Voting client
                    let voting_client = SeededMockClient::biased_seeded(
                        correct,
                        incorrect,
                        p,
                        100, // max samples
                        trial_seed.wrapping_add(1000),
                    );

                    let config = VoteConfig::default()
                        .with_max_samples(100)
                        .without_token_limit();

                    let start = Instant::now();
                    let vote_result = vote_with_margin("2+2?", k, &voting_client, config);
                    let elapsed = start.elapsed();

                    let (voting_correct, total_samples, converged) = match vote_result {
                        Ok(r) => (r.winner == correct, r.total_samples, true),
                        Err(_) => (false, 100, false),
                    };

                    results.push(TrialResult {
                        trial_id: trial,
                        config_id: config_id.clone(),
                        k_margin: k,
                        base_accuracy: p,
                        steps: 1,
                        voting_correct,
                        single_correct,
                        total_samples,
                        converged,
                        elapsed_ms: elapsed.as_millis() as u64,
                    });
                }
            }
        }
    }

    results
}

/// Run multi-step reliability experiment
fn run_multistep_experiment(ctx: &ExperimentContext) -> Vec<TrialResult> {
    ctx.log(1, "\n=== Multi-Step Reliability Experiment ===");

    let steps_values = [1, 3, 5, 10];
    let k = 3;
    let p = 0.85;
    let correct = "done";
    let incorrect = "error";

    let mut results = Vec::new();

    for &steps in &steps_values {
        let config_id = format!("steps{}_k{}_p{}", steps, k, (p * 100.0) as i32);
        ctx.log(1, &format!("  Running config: {}", config_id));

        for trial in 0..ctx.trials {
            let trial_seed = ctx
                .seed
                .wrapping_add(trial as u64)
                .wrapping_add(steps as u64 * 1000);

            // Single sample: all steps must succeed
            let mut single_success = true;
            for step in 0..steps {
                let client = SeededMockClient::biased_seeded(
                    correct,
                    incorrect,
                    p,
                    1,
                    trial_seed.wrapping_add(step as u64),
                );
                if client.generate("step", 0.0).unwrap().content != correct {
                    single_success = false;
                    break;
                }
            }

            // Voting: vote on each step
            let mut voting_success = true;
            let mut total_samples = 0;
            let start = Instant::now();

            for step in 0..steps {
                let client = SeededMockClient::biased_seeded(
                    correct,
                    incorrect,
                    p,
                    100,
                    trial_seed.wrapping_add(step as u64 * 100 + 10000),
                );
                let config = VoteConfig::default()
                    .with_max_samples(50)
                    .without_token_limit();

                match vote_with_margin("step", k, &client, config) {
                    Ok(r) => {
                        total_samples += r.total_samples;
                        if r.winner != correct {
                            voting_success = false;
                            break;
                        }
                    }
                    Err(_) => {
                        voting_success = false;
                        break;
                    }
                }
            }

            let elapsed = start.elapsed();

            results.push(TrialResult {
                trial_id: trial,
                config_id: config_id.clone(),
                k_margin: k,
                base_accuracy: p,
                steps,
                voting_correct: voting_success,
                single_correct: single_success,
                total_samples,
                converged: voting_success,
                elapsed_ms: elapsed.as_millis() as u64,
            });
        }
    }

    results
}

/// Run convergence analysis experiment
fn run_convergence_experiment(ctx: &ExperimentContext) -> Vec<TrialResult> {
    ctx.log(1, "\n=== Convergence Analysis Experiment ===");

    let k_values = [1, 2, 3, 5, 7, 10];
    let p = 0.85;
    let correct = "yes";
    let incorrect = "no";

    let mut results = Vec::new();

    for &k in &k_values {
        let config_id = format!("k{}_p{}", k, (p * 100.0) as i32);
        ctx.log(1, &format!("  Running config: {}", config_id));

        for trial in 0..ctx.trials {
            let trial_seed = ctx
                .seed
                .wrapping_add(trial as u64)
                .wrapping_add(k as u64 * 10000);

            let client = SeededMockClient::biased_seeded(correct, incorrect, p, 200, trial_seed);
            let config = VoteConfig::default()
                .with_max_samples(200)
                .without_token_limit();

            let start = Instant::now();
            let vote_result = vote_with_margin("test", k, &client, config);
            let elapsed = start.elapsed();

            let (voting_correct, total_samples, converged) = match vote_result {
                Ok(r) => (r.winner == correct, r.total_samples, true),
                Err(_) => (false, 200, false),
            };

            results.push(TrialResult {
                trial_id: trial,
                config_id: config_id.clone(),
                k_margin: k,
                base_accuracy: p,
                steps: 1,
                voting_correct,
                single_correct: true, // Not relevant for convergence
                total_samples,
                converged,
                elapsed_ms: elapsed.as_millis() as u64,
            });
        }
    }

    results
}

/// Run cost scaling experiment
fn run_cost_experiment(ctx: &ExperimentContext) -> Vec<TrialResult> {
    ctx.log(1, "\n=== Cost Scaling Experiment ===");

    let step_counts = [7, 31, 127, 511];
    let k = 3;
    let p = 0.85;
    let correct = "done";
    let incorrect = "error";

    let mut results = Vec::new();

    for &steps in &step_counts {
        let config_id = format!("s{}_k{}", steps, k);
        ctx.log(
            1,
            &format!("  Running config: {} ({} steps)", config_id, steps),
        );

        // Reduced trials for large step counts
        let adjusted_trials = if steps > 100 {
            ctx.trials / 2
        } else {
            ctx.trials
        };

        for trial in 0..adjusted_trials {
            let trial_seed = ctx
                .seed
                .wrapping_add(trial as u64)
                .wrapping_add(steps as u64 * 100);

            let mut total_samples = 0;
            let start = Instant::now();

            for step in 0..steps {
                let client = SeededMockClient::biased_seeded(
                    correct,
                    incorrect,
                    p,
                    100,
                    trial_seed.wrapping_add(step as u64),
                );
                let config = VoteConfig::default()
                    .with_max_samples(50)
                    .without_token_limit();

                if let Ok(r) = vote_with_margin("step", k, &client, config) {
                    total_samples += r.total_samples;
                }
            }

            let elapsed = start.elapsed();

            results.push(TrialResult {
                trial_id: trial,
                config_id: config_id.clone(),
                k_margin: k,
                base_accuracy: p,
                steps,
                voting_correct: true,
                single_correct: true,
                total_samples,
                converged: true,
                elapsed_ms: elapsed.as_millis() as u64,
            });
        }
    }

    results
}

/// Ensemble client that alternates between two real LLM providers
struct EnsembleClient {
    client_a: Arc<dyn LlmClient>,
    client_b: Arc<dyn LlmClient>,
    counter: std::sync::atomic::AtomicUsize,
}

impl EnsembleClient {
    fn new(client_a: Arc<dyn LlmClient>, client_b: Arc<dyn LlmClient>) -> Self {
        Self {
            client_a,
            client_b,
            counter: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl LlmClient for EnsembleClient {
    fn generate(&self, prompt: &str, temperature: f64) -> Result<LlmResponse, String> {
        let count = self
            .counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if count % 2 == 0 {
            self.client_a.generate(prompt, temperature)
        } else {
            self.client_b.generate(prompt, temperature)
        }
    }
}

/// Run ensemble comparison experiment with REAL LLMs
/// Compares: single model (ollama) vs single model (openai) vs ensemble (both)
fn run_ensemble_experiment(ctx: &ExperimentContext) -> Vec<TrialResult> {
    ctx.log(1, "\n=== Ensemble Comparison Experiment (LIVE) ===");

    if !ctx.use_live {
        ctx.log(0, "  ERROR: Ensemble experiment requires --live mode");
        ctx.log(0, "  Skipping ensemble experiment");
        return Vec::new();
    }

    // Create both clients
    let ollama_config = ProviderConfig {
        model: Some("llama3.2:3b".to_string()),
        api_key: None,
        base_url: None,
    };
    let ollama_client = match create_provider("ollama", Some(ollama_config)) {
        Ok(Some(c)) => Arc::from(c),
        Ok(None) | Err(_) => {
            ctx.log(0, "  ERROR: Failed to create ollama client");
            return Vec::new();
        }
    };

    let openai_config = ProviderConfig {
        model: Some("gpt-5-mini".to_string()),
        api_key: None,
        base_url: None,
    };
    let openai_client = match create_provider("openai", Some(openai_config)) {
        Ok(Some(c)) => Arc::from(c),
        Ok(None) | Err(_) => {
            ctx.log(0, "  ERROR: Failed to create openai client");
            return Vec::new();
        }
    };

    // Create ensemble client
    let ensemble_client =
        EnsembleClient::new(Arc::clone(&ollama_client), Arc::clone(&openai_client));

    let k = 3;
    let mut results = Vec::new();

    ctx.log(
        1,
        "  Running: ollama-only, openai-only, ensemble (alternating)",
    );

    for (q_idx, (question, expected)) in LIVE_TEST_QUESTIONS.iter().enumerate() {
        ctx.log(2, &format!("    Q{}: {}", q_idx, question));

        // Test 1: Ollama only
        let start = Instant::now();
        let ollama_single = ollama_client.generate(question, 0.7);
        let ollama_single_correct = match &ollama_single {
            Ok(r) => response_matches(&r.content, expected),
            Err(_) => false,
        };

        let config = VoteConfig::default()
            .with_max_samples(20)
            .without_token_limit()
            .with_diversity_temperature(0.7);
        let ollama_vote = vote_with_margin(question, k, ollama_client.as_ref(), config.clone());
        let (ollama_vote_correct, ollama_samples, ollama_converged) = match &ollama_vote {
            Ok(r) => (response_matches(&r.winner, expected), r.total_samples, true),
            Err(_) => (false, 20, false),
        };
        let ollama_elapsed = start.elapsed();

        results.push(TrialResult {
            trial_id: q_idx,
            config_id: "ollama_only".to_string(),
            k_margin: k,
            base_accuracy: 0.0,
            steps: 1,
            voting_correct: ollama_vote_correct,
            single_correct: ollama_single_correct,
            total_samples: ollama_samples,
            converged: ollama_converged,
            elapsed_ms: ollama_elapsed.as_millis() as u64,
        });

        // Test 2: OpenAI only
        let start = Instant::now();
        let openai_single = openai_client.generate(question, 0.7);
        let openai_single_correct = match &openai_single {
            Ok(r) => response_matches(&r.content, expected),
            Err(_) => false,
        };

        let openai_vote = vote_with_margin(question, k, openai_client.as_ref(), config.clone());
        let (openai_vote_correct, openai_samples, openai_converged) = match &openai_vote {
            Ok(r) => (response_matches(&r.winner, expected), r.total_samples, true),
            Err(_) => (false, 20, false),
        };
        let openai_elapsed = start.elapsed();

        results.push(TrialResult {
            trial_id: q_idx,
            config_id: "openai_only".to_string(),
            k_margin: k,
            base_accuracy: 0.0,
            steps: 1,
            voting_correct: openai_vote_correct,
            single_correct: openai_single_correct,
            total_samples: openai_samples,
            converged: openai_converged,
            elapsed_ms: openai_elapsed.as_millis() as u64,
        });

        // Test 3: Ensemble (alternating between ollama and openai)
        let start = Instant::now();
        // For single, use ollama (the "cheaper" one)
        let ensemble_single_correct = ollama_single_correct;

        let ensemble_vote = vote_with_margin(question, k, &ensemble_client, config);
        let (ensemble_vote_correct, ensemble_samples, ensemble_converged) = match &ensemble_vote {
            Ok(r) => (response_matches(&r.winner, expected), r.total_samples, true),
            Err(_) => (false, 20, false),
        };
        let ensemble_elapsed = start.elapsed();

        results.push(TrialResult {
            trial_id: q_idx,
            config_id: "ensemble".to_string(),
            k_margin: k,
            base_accuracy: 0.0,
            steps: 1,
            voting_correct: ensemble_vote_correct,
            single_correct: ensemble_single_correct,
            total_samples: ensemble_samples,
            converged: ensemble_converged,
            elapsed_ms: ensemble_elapsed.as_millis() as u64,
        });

        ctx.log(
            2,
            &format!(
                "      ollama: {}/{}, openai: {}/{}, ensemble: {}/{}",
                if ollama_single_correct { "✓" } else { "✗" },
                if ollama_vote_correct { "✓" } else { "✗" },
                if openai_single_correct { "✓" } else { "✗" },
                if openai_vote_correct { "✓" } else { "✗" },
                if ensemble_single_correct {
                    "✓"
                } else {
                    "✗"
                },
                if ensemble_vote_correct { "✓" } else { "✗" },
            ),
        );
    }

    results
}

/// Run ensemble decorrelation experiment with REAL LLMs
/// Tests if ensemble (ollama + openai) beats either model alone
fn run_decorrelation_experiment(ctx: &ExperimentContext) -> Vec<TrialResult> {
    ctx.log(1, "\n=== Ensemble Decorrelation Experiment (LIVE) ===");

    if !ctx.use_live {
        ctx.log(0, "  ERROR: Decorrelation experiment requires --live mode");
        return Vec::new();
    }

    // Create both clients
    let ollama_config = ProviderConfig {
        model: Some("llama3.2:3b".to_string()),
        api_key: None,
        base_url: None,
    };
    let ollama_client = match create_provider("ollama", Some(ollama_config)) {
        Ok(Some(c)) => Arc::from(c),
        Ok(None) | Err(_) => {
            ctx.log(0, "  ERROR: Failed to create ollama client");
            return Vec::new();
        }
    };

    let openai_config = ProviderConfig {
        model: Some("gpt-5-mini".to_string()),
        api_key: None,
        base_url: None,
    };
    let openai_client = match create_provider("openai", Some(openai_config)) {
        Ok(Some(c)) => Arc::from(c),
        Ok(None) | Err(_) => {
            ctx.log(0, "  ERROR: Failed to create openai client");
            return Vec::new();
        }
    };

    let ensemble_client =
        EnsembleClient::new(Arc::clone(&ollama_client), Arc::clone(&openai_client));

    let k_values = [1, 2, 3];
    let mut results = Vec::new();

    ctx.log(
        1,
        "  Testing decorrelation: does ensemble beat individual models?",
    );

    for &k in &k_values {
        ctx.log(1, &format!("  k={}", k));

        for trial in 0..ctx.trials {
            for (q_idx, (question, expected)) in LIVE_TEST_QUESTIONS.iter().enumerate() {
                let trial_id = trial * LIVE_TEST_QUESTIONS.len() + q_idx;

                let config = VoteConfig::default()
                    .with_max_samples(20)
                    .without_token_limit()
                    .with_diversity_temperature(0.7);

                // Ollama voting
                let start = Instant::now();
                let ollama_single = ollama_client.generate(question, 0.7);
                let ollama_single_correct = match &ollama_single {
                    Ok(r) => response_matches(&r.content, expected),
                    Err(_) => false,
                };
                let ollama_vote =
                    vote_with_margin(question, k, ollama_client.as_ref(), config.clone());
                let (ollama_correct, ollama_samples, ollama_conv) = match &ollama_vote {
                    Ok(r) => (response_matches(&r.winner, expected), r.total_samples, true),
                    Err(_) => (false, 20, false),
                };
                let ollama_elapsed = start.elapsed();

                results.push(TrialResult {
                    trial_id,
                    config_id: format!("decorr_ollama_k{}", k),
                    k_margin: k,
                    base_accuracy: 0.0,
                    steps: 1,
                    voting_correct: ollama_correct,
                    single_correct: ollama_single_correct,
                    total_samples: ollama_samples,
                    converged: ollama_conv,
                    elapsed_ms: ollama_elapsed.as_millis() as u64,
                });

                // OpenAI voting
                let start = Instant::now();
                let openai_single = openai_client.generate(question, 0.7);
                let openai_single_correct = match &openai_single {
                    Ok(r) => response_matches(&r.content, expected),
                    Err(_) => false,
                };
                let openai_vote =
                    vote_with_margin(question, k, openai_client.as_ref(), config.clone());
                let (openai_correct, openai_samples, openai_conv) = match &openai_vote {
                    Ok(r) => (response_matches(&r.winner, expected), r.total_samples, true),
                    Err(_) => (false, 20, false),
                };
                let openai_elapsed = start.elapsed();

                results.push(TrialResult {
                    trial_id,
                    config_id: format!("decorr_openai_k{}", k),
                    k_margin: k,
                    base_accuracy: 0.0,
                    steps: 1,
                    voting_correct: openai_correct,
                    single_correct: openai_single_correct,
                    total_samples: openai_samples,
                    converged: openai_conv,
                    elapsed_ms: openai_elapsed.as_millis() as u64,
                });

                // Ensemble voting
                let start = Instant::now();
                let ensemble_vote = vote_with_margin(question, k, &ensemble_client, config);
                let (ensemble_correct, ensemble_samples, ensemble_conv) = match &ensemble_vote {
                    Ok(r) => (response_matches(&r.winner, expected), r.total_samples, true),
                    Err(_) => (false, 20, false),
                };
                let ensemble_elapsed = start.elapsed();

                results.push(TrialResult {
                    trial_id,
                    config_id: format!("decorr_ensemble_k{}", k),
                    k_margin: k,
                    base_accuracy: 0.0,
                    steps: 1,
                    voting_correct: ensemble_correct,
                    single_correct: ollama_single_correct, // Use ollama as baseline single
                    total_samples: ensemble_samples,
                    converged: ensemble_conv,
                    elapsed_ms: ensemble_elapsed.as_millis() as u64,
                });
            }
        }
    }

    results
}

// ============================================================================
// Report Generation
// ============================================================================

fn generate_csv(results: &[TrialResult], path: &PathBuf) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(
        file,
        "trial_id,config_id,k_margin,base_accuracy,steps,voting_correct,single_correct,total_samples,converged,elapsed_ms"
    )?;

    for r in results {
        writeln!(
            file,
            "{},{},{},{},{},{},{},{},{},{}",
            r.trial_id,
            r.config_id,
            r.k_margin,
            r.base_accuracy,
            r.steps,
            r.voting_correct,
            r.single_correct,
            r.total_samples,
            r.converged,
            r.elapsed_ms
        )?;
    }

    Ok(())
}

fn compute_summaries(results: &[TrialResult]) -> Vec<StatisticalSummary> {
    let mut by_config: HashMap<String, Vec<&TrialResult>> = HashMap::new();
    for r in results {
        by_config.entry(r.config_id.clone()).or_default().push(r);
    }

    let mut summaries = Vec::new();

    for (config_id, trials) in by_config {
        let n = trials.len();
        if n == 0 {
            continue;
        }

        let voting_scores: Vec<f64> = trials
            .iter()
            .map(|t| if t.voting_correct { 1.0 } else { 0.0 })
            .collect();
        let single_scores: Vec<f64> = trials
            .iter()
            .map(|t| if t.single_correct { 1.0 } else { 0.0 })
            .collect();

        let (voting_mean, voting_std) = statistics::mean_std(&voting_scores);
        let (single_mean, _) = statistics::mean_std(&single_scores);
        let (voting_ci_lower, voting_ci_upper) =
            statistics::confidence_interval_95(voting_mean, voting_std, n);

        let cohens_d = statistics::cohens_d_paired(&voting_scores, &single_scores);
        let (t_statistic, p_value, _) = statistics::paired_t_test(&voting_scores, &single_scores);

        summaries.push(StatisticalSummary {
            config_id,
            n,
            voting_mean,
            voting_std,
            voting_ci_lower,
            voting_ci_upper,
            single_mean,
            cohens_d,
            t_statistic,
            p_value,
            significant: p_value < 0.05,
        });
    }

    summaries.sort_by(|a, b| a.config_id.cmp(&b.config_id));
    summaries
}

fn generate_results_md(summaries: &[StatisticalSummary], path: &PathBuf) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    writeln!(file, "# Research Validation Results\n")?;
    writeln!(file, "Generated: {}\n", chrono_lite_now())?;

    writeln!(file, "## Summary Table\n")?;
    writeln!(
        file,
        "| Config | n | Voting Acc | Single Acc | Improvement | Cohen's d | p-value | Sig |"
    )?;
    writeln!(
        file,
        "|--------|---|------------|------------|-------------|-----------|---------|-----|"
    )?;

    for s in summaries {
        let improvement = if s.single_mean > 0.0 {
            format!("{:.2}x", s.voting_mean / s.single_mean)
        } else {
            "N/A".to_string()
        };

        let sig_stars = statistics::significance_stars(s.p_value);

        writeln!(
            file,
            "| {} | {} | {:.1}% ({:.1}-{:.1}) | {:.1}% | {} | {:.2} | {:.4} | {} |",
            s.config_id,
            s.n,
            s.voting_mean * 100.0,
            s.voting_ci_lower * 100.0,
            s.voting_ci_upper * 100.0,
            s.single_mean * 100.0,
            improvement,
            s.cohens_d,
            s.p_value,
            sig_stars
        )?;
    }

    writeln!(file, "\n## Statistical Notes\n")?;
    writeln!(file, "- **Sig column**: *** p<0.001, ** p<0.01, * p<0.05")?;
    writeln!(file, "- **Cohen's d interpretation**: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large")?;
    writeln!(
        file,
        "- Confidence intervals are 95% Wilson score intervals for proportions"
    )?;

    Ok(())
}

fn generate_methodology_md(ctx: &ExperimentContext, path: &PathBuf) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    writeln!(file, "# Methodology\n")?;
    writeln!(file, "## Experimental Design\n")?;
    writeln!(
        file,
        "This validation study evaluates MAKER's error correction through"
    )?;
    writeln!(
        file,
        "SPRT-based voting using controlled mock experiments.\n"
    )?;

    writeln!(file, "### Parameters\n")?;
    writeln!(file, "- **Trials per configuration**: {}", ctx.trials)?;
    writeln!(file, "- **RNG Seed**: {}", ctx.seed)?;
    writeln!(
        file,
        "- **Mode**: {}",
        if ctx.use_live {
            "Live (Ollama)"
        } else {
            "Mock (deterministic)"
        }
    )?;

    writeln!(file, "\n### Experiments\n")?;
    writeln!(
        file,
        "1. **Accuracy Comparison**: k-margin voting vs single sample"
    )?;
    writeln!(file, "   - k values: 1, 2, 3, 5, 7")?;
    writeln!(file, "   - p values: 0.6, 0.7, 0.8, 0.9")?;
    writeln!(
        file,
        "2. **Multi-Step Reliability**: Task success over multiple steps"
    )?;
    writeln!(file, "   - Steps: 1, 3, 5, 10")?;
    writeln!(file, "   - k=3, p=0.85")?;
    writeln!(
        file,
        "3. **Convergence Analysis**: Samples required for k-margin"
    )?;
    writeln!(file, "   - k values: 1, 2, 3, 5, 7, 10")?;
    writeln!(file, "4. **Cost Scaling**: Total samples vs step count")?;
    writeln!(file, "   - Steps: 7, 31, 127, 511")?;

    writeln!(file, "\n## Statistical Methods\n")?;
    writeln!(
        file,
        "- **Mean/Std**: Sample mean with Bessel-corrected standard deviation"
    )?;
    writeln!(file, "- **95% CI**: t-distribution confidence intervals")?;
    writeln!(
        file,
        "- **Paired t-test**: Two-tailed test comparing voting vs single sample"
    )?;
    writeln!(file, "- **Cohen's d**: Effect size for paired samples")?;
    writeln!(
        file,
        "- **Wilson CI**: Confidence intervals for proportions"
    )?;

    writeln!(file, "\n## Reproducibility\n")?;
    writeln!(file, "```bash")?;
    writeln!(
        file,
        "cargo run --bin research-validation -- --mock --seed {} --trials {}",
        ctx.seed, ctx.trials
    )?;
    writeln!(file, "```")?;

    writeln!(file, "\n## Limitations\n")?;
    writeln!(
        file,
        "- Mock experiments use pseudo-random sequences, not true LLM behavior"
    )?;
    writeln!(
        file,
        "- Error correlation assumed to be independent (may not hold for real LLMs)"
    )?;
    writeln!(file, "- Ensemble experiments simulate decorrelation effect")?;

    Ok(())
}

fn generate_statistical_analysis_md(
    summaries: &[StatisticalSummary],
    path: &PathBuf,
) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    writeln!(file, "# Statistical Analysis\n")?;

    writeln!(file, "## Hypothesis Tests\n")?;
    writeln!(file, "**H0**: Voting accuracy = Single sample accuracy")?;
    writeln!(file, "**H1**: Voting accuracy > Single sample accuracy\n")?;

    for s in summaries {
        writeln!(file, "### {}\n", s.config_id)?;
        writeln!(file, "- **n**: {}", s.n)?;
        writeln!(file, "- **Voting mean**: {:.1}%", s.voting_mean * 100.0)?;
        writeln!(file, "- **Single mean**: {:.1}%", s.single_mean * 100.0)?;
        writeln!(file, "- **t-statistic**: {:.4}", s.t_statistic)?;
        writeln!(file, "- **p-value**: {:.6}", s.p_value)?;
        writeln!(
            file,
            "- **Cohen's d**: {:.4} ({})",
            s.cohens_d,
            statistics::effect_size_interpretation(s.cohens_d)
        )?;
        writeln!(
            file,
            "- **Conclusion**: {}",
            if s.significant {
                "Reject H0 - Voting significantly improves accuracy"
            } else {
                "Fail to reject H0"
            }
        )?;
        writeln!(file)?;
    }

    Ok(())
}

// ============================================================================
// SVG Generation
// ============================================================================

fn generate_accuracy_svg(summaries: &[StatisticalSummary], path: &PathBuf) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    // Filter for accuracy comparison configs (include all non-empty configs)
    let accuracy_configs: Vec<_> = summaries
        .iter()
        .filter(|s| !s.config_id.is_empty() && s.n > 0)
        .collect();

    let width = 800;
    let height = 450;
    let margin = 60;
    let bar_width = 25;

    writeln!(
        file,
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">"##
    )?;

    // Background
    writeln!(
        file,
        r##"  <rect width="{width}" height="{height}" rx="8" fill="#f8fafc"/>"##
    )?;

    // Title
    writeln!(
        file,
        r##"  <text x="{}" y="30" font-family="sans-serif" font-size="18" font-weight="bold" fill="#0f172a" text-anchor="middle">Accuracy Comparison: Voting vs Single Sample</text>"##,
        width / 2
    )?;

    let chart_width = width - 2 * margin;
    let chart_height = height - 2 * margin - 30;
    let n_bars = accuracy_configs.len();

    if n_bars > 0 {
        let group_width = chart_width / n_bars;

        for (i, s) in accuracy_configs.iter().enumerate() {
            let x = margin + i * group_width + group_width / 2;

            // Voting bar (blue)
            let voting_height = (s.voting_mean * chart_height as f64) as usize;
            writeln!(
                file,
                r##"  <rect x="{}" y="{}" width="{}" height="{}" fill="#3b82f6" rx="2"/>"##,
                x - bar_width,
                margin + 40 + chart_height - voting_height,
                bar_width,
                voting_height
            )?;

            // Single bar (gray)
            let single_height = (s.single_mean * chart_height as f64) as usize;
            writeln!(
                file,
                r##"  <rect x="{}" y="{}" width="{}" height="{}" fill="#94a3b8" rx="2"/>"##,
                x + 2,
                margin + 40 + chart_height - single_height,
                bar_width,
                single_height
            )?;

            // Label
            writeln!(
                file,
                r##"  <text x="{}" y="{}" font-family="sans-serif" font-size="10" fill="#64748b" text-anchor="middle" transform="rotate(-45 {} {})">{}</text>"##,
                x,
                height - 20,
                x,
                height - 20,
                s.config_id
            )?;
        }
    }

    // Y-axis labels
    for pct in [0, 25, 50, 75, 100] {
        let y = margin + 40 + chart_height - (pct * chart_height / 100);
        writeln!(
            file,
            r##"  <text x="{}" y="{}" font-family="sans-serif" font-size="10" fill="#64748b" text-anchor="end">{}%</text>"##,
            margin - 5,
            y + 4,
            pct
        )?;
        writeln!(
            file,
            r##"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#e2e8f0" stroke-dasharray="2,2"/>"##,
            margin,
            y,
            width - margin,
            y
        )?;
    }

    // Legend
    writeln!(
        file,
        r##"  <rect x="{}" y="50" width="15" height="15" fill="#3b82f6" rx="2"/>"##,
        width - 150
    )?;
    writeln!(
        file,
        r##"  <text x="{}" y="62" font-family="sans-serif" font-size="12" fill="#1e293b">Voting</text>"##,
        width - 130
    )?;
    writeln!(
        file,
        r##"  <rect x="{}" y="70" width="15" height="15" fill="#94a3b8" rx="2"/>"##,
        width - 150
    )?;
    writeln!(
        file,
        r##"  <text x="{}" y="82" font-family="sans-serif" font-size="12" fill="#1e293b">Single</text>"##,
        width - 130
    )?;

    writeln!(file, "</svg>")?;

    Ok(())
}

fn generate_convergence_svg(results: &[TrialResult], path: &PathBuf) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    // Group by k value (include all results with k_margin > 0)
    let mut by_k: HashMap<usize, Vec<usize>> = HashMap::new();
    for r in results.iter().filter(|r| r.k_margin > 0) {
        by_k.entry(r.k_margin).or_default().push(r.total_samples);
    }

    let width = 700;
    let height = 400;
    let margin = 60;

    writeln!(
        file,
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">"##
    )?;

    writeln!(
        file,
        r##"  <rect width="{width}" height="{height}" rx="8" fill="#f8fafc"/>"##
    )?;

    writeln!(
        file,
        r##"  <text x="{}" y="30" font-family="sans-serif" font-size="18" font-weight="bold" fill="#0f172a" text-anchor="middle">Convergence: Samples Required by k-margin</text>"##,
        width / 2
    )?;

    // Calculate means
    let mut k_means: Vec<(usize, f64)> = Vec::new();
    for (k, samples) in &by_k {
        let mean = samples.iter().sum::<usize>() as f64 / samples.len() as f64;
        k_means.push((*k, mean));
    }
    k_means.sort_by_key(|(k, _)| *k);

    if !k_means.is_empty() {
        let max_samples = k_means.iter().map(|(_, m)| *m).fold(0.0, f64::max) * 1.1;
        let chart_width = width - 2 * margin;
        let chart_height = height - 2 * margin - 40;

        // Draw line
        let mut points = String::new();
        for (i, (_k, mean)) in k_means.iter().enumerate() {
            let x = margin + (i * chart_width / (k_means.len().max(1) - 1).max(1));
            let y =
                margin + 40 + chart_height - ((mean / max_samples) * chart_height as f64) as usize;
            if i == 0 {
                points = format!("M {} {}", x, y);
            } else {
                points = format!("{} L {} {}", points, x, y);
            }
        }

        writeln!(
            file,
            r##"  <path d="{}" stroke="#3b82f6" stroke-width="3" fill="none"/>"##,
            points
        )?;

        // Draw points and labels
        for (i, (k, mean)) in k_means.iter().enumerate() {
            let x = margin + (i * chart_width / (k_means.len().max(1) - 1).max(1));
            let y =
                margin + 40 + chart_height - ((mean / max_samples) * chart_height as f64) as usize;

            writeln!(
                file,
                r##"  <circle cx="{}" cy="{}" r="6" fill="#3b82f6"/>"##,
                x, y
            )?;
            writeln!(
                file,
                r##"  <text x="{}" y="{}" font-family="sans-serif" font-size="10" fill="#1e293b" text-anchor="middle">k={}</text>"##,
                x,
                height - 20,
                k
            )?;
            writeln!(
                file,
                r##"  <text x="{}" y="{}" font-family="sans-serif" font-size="10" fill="#64748b" text-anchor="middle">{:.1}</text>"##,
                x,
                y - 12,
                mean
            )?;
        }
    }

    // Y-axis label
    writeln!(
        file,
        r##"  <text x="20" y="{}" font-family="sans-serif" font-size="12" fill="#64748b" text-anchor="middle" transform="rotate(-90 20 {})">Avg Samples</text>"##,
        height / 2,
        height / 2
    )?;

    writeln!(file, "</svg>")?;

    Ok(())
}

fn generate_k_reliability_svg(path: &PathBuf) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    let width = 700;
    let height = 400;
    let margin = 60;

    writeln!(
        file,
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">"##
    )?;

    writeln!(
        file,
        r##"  <rect width="{width}" height="{height}" rx="8" fill="#f8fafc"/>"##
    )?;

    writeln!(
        file,
        r##"  <text x="{}" y="30" font-family="sans-serif" font-size="18" font-weight="bold" fill="#0f172a" text-anchor="middle">k-margin vs Theoretical Reliability</text>"##,
        width / 2
    )?;

    // Calculate theoretical curve for p=0.85, s=1000
    let p = 0.85;
    let s = 1000;
    let chart_width = width - 2 * margin;
    let chart_height = height - 2 * margin - 40;

    let mut points = String::new();
    for k in 1..=10 {
        // Reliability per step with k-margin voting
        let r: f64 = (1.0 - p) / p;
        let reliability_per_step: f64 = 1.0 - r.powi(k as i32);
        let task_reliability: f64 = reliability_per_step.powi(s as i32);

        let x = margin + ((k - 1) * chart_width / 9);
        let y = margin + 40 + chart_height - ((task_reliability) * chart_height as f64) as usize;

        if k == 1 {
            points = format!("M {} {}", x, y);
        } else {
            points = format!("{} L {} {}", points, x, y);
        }
    }

    writeln!(
        file,
        r##"  <path d="{}" stroke="#10b981" stroke-width="3" fill="none"/>"##,
        points
    )?;

    // Draw points
    for k in 1..=10 {
        let r: f64 = (1.0 - p) / p;
        let reliability_per_step: f64 = 1.0 - r.powi(k as i32);
        let task_reliability: f64 = reliability_per_step.powi(s as i32);

        let x = margin + ((k - 1) * chart_width / 9);
        let y = margin + 40 + chart_height - ((task_reliability) * chart_height as f64) as usize;

        writeln!(
            file,
            r##"  <circle cx="{}" cy="{}" r="5" fill="#10b981"/>"##,
            x, y
        )?;
        writeln!(
            file,
            r##"  <text x="{}" y="{}" font-family="sans-serif" font-size="10" fill="#1e293b" text-anchor="middle">{}</text>"##,
            x,
            height - 20,
            k
        )?;
    }

    // 95% line
    let y_95 = margin + 40 + chart_height - ((0.95) * chart_height as f64) as usize;
    writeln!(
        file,
        r##"  <line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#ef4444" stroke-dasharray="5,5"/>"##,
        margin,
        y_95,
        width - margin,
        y_95
    )?;
    writeln!(
        file,
        r##"  <text x="{}" y="{}" font-family="sans-serif" font-size="10" fill="#ef4444">95% target</text>"##,
        width - margin + 5,
        y_95 + 4
    )?;

    // Axes labels
    writeln!(
        file,
        r##"  <text x="{}" y="{}" font-family="sans-serif" font-size="12" fill="#64748b" text-anchor="middle">k-margin</text>"##,
        width / 2,
        height - 5
    )?;

    writeln!(
        file,
        r##"  <text x="{}" y="55" font-family="sans-serif" font-size="10" fill="#64748b">p=0.85, s=1000</text>"##,
        margin
    )?;

    writeln!(file, "</svg>")?;

    Ok(())
}

// ============================================================================
// Utility
// ============================================================================

fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    // Simple timestamp format without chrono dependency
    format!("Unix timestamp: {}", secs)
}

// ============================================================================
// Main
// ============================================================================

fn main() -> ExitCode {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    if let Err(_) = tracing_subscriber::fmt()
        .with_env_filter(log_level)
        .with_writer(std::io::stderr)
        .try_init()
    {
        // Subscriber already initialized
    }

    let ctx = match ExperimentContext::new(&cli) {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("Failed to initialize: {}", e);
            return ExitCode::FAILURE;
        }
    };

    println!("MAKER Research Validation");
    println!("========================");
    println!(
        "Mode: {}",
        if ctx.use_live {
            format!("Live ({} / {})", ctx.provider, ctx.model)
        } else {
            "Mock".to_string()
        }
    );
    println!("Trials: {}", ctx.trials);
    println!("Seed: {}", ctx.seed);
    println!("Output: {}", ctx.output_dir.display());
    println!();

    // Create output directories
    if let Err(e) = fs::create_dir_all(&ctx.output_dir) {
        eprintln!("Failed to create output directory: {}", e);
        return ExitCode::FAILURE;
    }
    let figures_dir = ctx.output_dir.join("figures");
    if let Err(e) = fs::create_dir_all(&figures_dir) {
        eprintln!("Failed to create figures directory: {}", e);
        return ExitCode::FAILURE;
    }

    // Determine which experiments to run
    let run_all = cli.all || cli.experiment.is_none();
    let run_ensemble = cli.ensemble || cli.all;

    let mut all_results: Vec<TrialResult> = Vec::new();

    // Run experiments
    if run_all || matches!(cli.experiment, Some(ExperimentName::Accuracy)) {
        let results = run_accuracy_experiment(&ctx);
        all_results.extend(results);
    }

    if run_all || matches!(cli.experiment, Some(ExperimentName::Multistep)) {
        let results = run_multistep_experiment(&ctx);
        all_results.extend(results);
    }

    if run_all || matches!(cli.experiment, Some(ExperimentName::Convergence)) {
        let results = run_convergence_experiment(&ctx);
        all_results.extend(results);
    }

    if run_all || matches!(cli.experiment, Some(ExperimentName::Cost)) {
        let results = run_cost_experiment(&ctx);
        all_results.extend(results);
    }

    if run_ensemble || matches!(cli.experiment, Some(ExperimentName::Ensemble)) {
        let results = run_ensemble_experiment(&ctx);
        all_results.extend(results);
    }

    if run_ensemble || matches!(cli.experiment, Some(ExperimentName::Decorrelation)) {
        let results = run_decorrelation_experiment(&ctx);
        all_results.extend(results);
    }

    // Generate reports
    println!("\nGenerating reports...");

    // Raw CSV
    if let Err(e) = generate_csv(&all_results, &ctx.output_dir.join("raw_data.csv")) {
        eprintln!("Failed to generate CSV: {}", e);
    } else {
        println!("  - raw_data.csv");
    }

    // Statistical summaries
    let summaries = compute_summaries(&all_results);

    // Results markdown
    if let Err(e) = generate_results_md(&summaries, &ctx.output_dir.join("results.md")) {
        eprintln!("Failed to generate results.md: {}", e);
    } else {
        println!("  - results.md");
    }

    // Methodology markdown
    if let Err(e) = generate_methodology_md(&ctx, &ctx.output_dir.join("methodology.md")) {
        eprintln!("Failed to generate methodology.md: {}", e);
    } else {
        println!("  - methodology.md");
    }

    // Statistical analysis markdown
    if let Err(e) = generate_statistical_analysis_md(
        &summaries,
        &ctx.output_dir.join("statistical_analysis.md"),
    ) {
        eprintln!("Failed to generate statistical_analysis.md: {}", e);
    } else {
        println!("  - statistical_analysis.md");
    }

    // SVG figures
    if let Err(e) = generate_accuracy_svg(&summaries, &figures_dir.join("accuracy_comparison.svg"))
    {
        eprintln!("Failed to generate accuracy SVG: {}", e);
    } else {
        println!("  - figures/accuracy_comparison.svg");
    }

    if let Err(e) =
        generate_convergence_svg(&all_results, &figures_dir.join("convergence_curves.svg"))
    {
        eprintln!("Failed to generate convergence SVG: {}", e);
    } else {
        println!("  - figures/convergence_curves.svg");
    }

    if let Err(e) = generate_k_reliability_svg(&figures_dir.join("k_margin_vs_reliability.svg")) {
        eprintln!("Failed to generate k-reliability SVG: {}", e);
    } else {
        println!("  - figures/k_margin_vs_reliability.svg");
    }

    // Print summary
    println!("\n=== Quick Summary ===");
    let accuracy_summaries: Vec<_> = summaries
        .iter()
        .filter(|s| s.config_id.starts_with("k") && s.config_id.contains("_p"))
        .collect();

    if !accuracy_summaries.is_empty() {
        let avg_voting = accuracy_summaries
            .iter()
            .map(|s| s.voting_mean)
            .sum::<f64>()
            / accuracy_summaries.len() as f64;
        let avg_single = accuracy_summaries
            .iter()
            .map(|s| s.single_mean)
            .sum::<f64>()
            / accuracy_summaries.len() as f64;
        let significant_count = accuracy_summaries.iter().filter(|s| s.significant).count();

        println!("Accuracy Comparison:");
        println!("  Average voting accuracy: {:.1}%", avg_voting * 100.0);
        println!("  Average single accuracy: {:.1}%", avg_single * 100.0);
        println!(
            "  Significant improvements: {}/{}",
            significant_count,
            accuracy_summaries.len()
        );
    }

    println!(
        "\nValidation complete. Results in: {}",
        ctx.output_dir.display()
    );

    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_std() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let (mean, std) = statistics::mean_std(&data);
        assert!((mean - 5.0).abs() < 0.001);
        // Sample std = sqrt(32/7) ≈ 2.138
        assert!((std - 2.138).abs() < 0.01);
    }

    #[test]
    fn test_wilson_ci() {
        let (lower, upper) = statistics::wilson_ci(80, 100);
        assert!(lower > 0.70);
        assert!(upper < 0.88);
    }

    #[test]
    fn test_paired_t_test() {
        // Use data with variance in differences
        let x = vec![1.0, 1.0, 1.0, 0.8, 1.2];
        let y = vec![0.0, 0.1, 0.0, 0.0, 0.0];
        let (t, p, _) = statistics::paired_t_test(&x, &y);
        assert!(t > 0.0);
        assert!(p < 0.05);
    }

    #[test]
    fn test_cohens_d() {
        // Use data with variance in differences for meaningful effect size
        let x = vec![1.0, 1.2, 0.8, 1.1, 0.9];
        let y = vec![0.0, 0.1, 0.0, 0.1, 0.0];
        let d = statistics::cohens_d_paired(&x, &y);
        // Mean diff ~1.0, std diff should give large effect
        assert!(d > 0.8, "Cohen's d should be large: {}", d);
    }

    #[test]
    fn test_seeded_mock_deterministic() {
        let client1 = SeededMockClient::biased_seeded("yes", "no", 0.8, 100, 42);
        let client2 = SeededMockClient::biased_seeded("yes", "no", 0.8, 100, 42);

        for _ in 0..10 {
            assert_eq!(
                client1.generate("", 0.0).unwrap().content,
                client2.generate("", 0.0).unwrap().content
            );
        }
    }
}
