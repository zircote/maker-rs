//! End-to-End Towers of Hanoi with Voting
//!
//! Demonstrates MAKER's zero-error execution on the Towers of Hanoi using
//! the complete voting engine with either real LLM providers or mock responses.
//!
//! # Execution Modes
//!
//! ## LLM Mode (Default)
//! Uses adaptive k-margin voting with a real LLM provider. The k-margin adjusts
//! dynamically based on observed model accuracy to minimize token usage while
//! maintaining the target reliability (95% by default).
//!
//! ## Mock Mode (CI/Testing)
//! Controlled simulation with fixed accuracy for reproducible testing. Useful
//! for CI pipelines and validating the voting algorithm with known accuracy.
//! Activated via `MAKER_USE_MOCK=1` environment variable.
//!
//! # Usage
//!
//! ```bash
//! # Run with 3 disks (7 steps) using default provider (Ollama)
//! cargo run --example hanoi_demo -- --disks 3
//!
//! # Run with 10 disks (1,023 steps) using Ollama with specific model
//! cargo run --example hanoi_demo -- --disks 10 --provider ollama --model mistral
//!
//! # Run with OpenAI (requires OPENAI_API_KEY environment variable)
//! cargo run --example hanoi_demo -- --disks 5 --provider openai
//!
//! # Run with Anthropic (requires ANTHROPIC_API_KEY environment variable)
//! cargo run --example hanoi_demo -- --disks 5 --provider anthropic --model claude-3-5-sonnet-20241022
//!
//! # Run in mock mode (for CI/testing)
//! MAKER_USE_MOCK=1 cargo run --example hanoi_demo -- --disks 10 --accuracy 0.85
//! ```
//!
//! # Expected Output
//!
//! The demo prints each step's result (or progress updates for large runs),
//! showing the winning move, number of samples required, current k-margin,
//! and estimated model accuracy (p_hat). Final statistics include:
//! - Total errors (should be 0 for zero-error guarantee)
//! - Average samples per step (lower is better)
//! - Red-flagged responses (rejected for format violations)
//! - Total execution time and estimated token cost

use clap::Parser;
use maker::core::{
    calculate_kmin, vote_with_margin, vote_with_margin_adaptive, KEstimator, KEstimatorConfig,
    LlmClient, LlmResponse, MockLlmClient, VoteConfig,
};
use maker::llm::adapter::setup_provider_client;
use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

// Default model names per provider
const DEFAULT_OLLAMA_MODEL: &str = "gpt-oss";
const DEFAULT_OPENAI_MODEL: &str = "gpt-5-mini";
const DEFAULT_ANTHROPIC_MODEL: &str = "claude-3-5-sonnet-20241022";

/// Configuration for the Hanoi demo execution
#[derive(Clone)]
struct HanoiDemoConfig {
    // Execution settings
    target_reliability: f64,
    voting_temperature: f64,
    progress_interval: usize,

    // Mock mode settings
    pool_size: usize,

    // K-Estimator settings
    ema_alpha: f64,
    initial_p_hat: f64,
    k_min_floor: usize,
    k_max_ceiling: usize,
}

impl Default for HanoiDemoConfig {
    fn default() -> Self {
        Self {
            target_reliability: 0.95,
            voting_temperature: 0.0,
            progress_interval: 100,
            pool_size: 100,
            ema_alpha: 0.2,
            initial_p_hat: 0.80,
            k_min_floor: 1,
            k_max_ceiling: 10,
        }
    }
}

/// Statistics tracked during Hanoi execution
///
/// # Fields
/// - `errors`: Number of steps where the voted winner did not match expected move
/// - `total_samples`: Cumulative LLM samples requested across all steps
/// - `total_red_flagged`: Samples rejected due to format/validation failures
#[derive(Default, Clone)]
struct HanoiStats {
    errors: usize,
    total_samples: usize,
    total_red_flagged: usize,
}

/// Towers of Hanoi demonstration with MAKER's adaptive voting
#[derive(Parser, Debug)]
#[command(
    name = "hanoi_demo",
    about = "Demonstrates MAKER's zero-error execution on Towers of Hanoi"
)]
struct Args {
    /// Number of disks (1-20)
    #[arg(short, long, default_value = "3", value_parser = clap::value_parser!(u8).range(1..=20))]
    disks: u8,

    /// Model accuracy 0.51-0.99 (mock mode only)
    #[arg(short, long, default_value = "0.85")]
    accuracy: f64,

    /// LLM provider: ollama, openai, or anthropic
    #[arg(short, long, default_value = "ollama")]
    provider: String,

    /// Model name (defaults to provider-specific: gpt-oss for ollama, gpt-4o for openai, claude-3-5-sonnet-20241022 for anthropic)
    #[arg(short, long)]
    model: Option<String>,

    /// Enable ensemble mode: use multiple providers with round-robin sampling
    /// Combines --provider with a second provider for error decorrelation
    #[arg(short, long)]
    ensemble: bool,

    /// Second provider for ensemble mode (default: openai if primary is ollama, else ollama)
    #[arg(long)]
    ensemble_provider: Option<String>,

    /// Strict mode: halt on first error (true zero-error execution)
    #[arg(long, default_value = "false")]
    strict: bool,
}

/// Ensemble client that alternates between two LLM providers
/// This decorrelates errors across model architectures
struct EnsembleClient {
    client_a: Box<dyn LlmClient>,
    client_b: Box<dyn LlmClient>,
    counter: AtomicUsize,
}

impl EnsembleClient {
    fn new(client_a: Box<dyn LlmClient>, client_b: Box<dyn LlmClient>) -> Self {
        Self {
            client_a,
            client_b,
            counter: AtomicUsize::new(0),
        }
    }
}

impl LlmClient for EnsembleClient {
    fn generate(&self, prompt: &str, temperature: f64) -> Result<LlmResponse, String> {
        let count = self.counter.fetch_add(1, Ordering::SeqCst);
        if count.is_multiple_of(2) {
            self.client_a.generate(prompt, temperature)
        } else {
            self.client_b.generate(prompt, temperature)
        }
    }
}

/// Determine if we should print this step based on disk count and step number
///
/// # Parameters
/// - `n_disks`: Number of disks in the puzzle (controls verbosity)
/// - `step`: Current step index (0-based)
/// - `total_steps`: Total number of steps in the solution
/// - `config`: Configuration containing progress_interval
///
/// # Returns
/// `true` if this step should be printed, `false` to suppress output
///
/// Prints all steps for small puzzles (≤5 disks), or only periodic progress
/// updates plus the final step for larger puzzles.
fn should_print_step(
    n_disks: u8,
    step: usize,
    total_steps: usize,
    config: &HanoiDemoConfig,
) -> bool {
    n_disks <= 5 || step.is_multiple_of(config.progress_interval) || step == total_steps - 1
}

/// Compute the optimal Hanoi move sequence
///
/// Generates the ground truth solution for the Towers of Hanoi puzzle
/// using the standard recursive algorithm. This solution is used to
/// validate the LLM's voted responses at each step.
///
/// # Parameters
/// - `n_disks`: Number of disks in the puzzle
///
/// # Returns
/// Vector of move strings in the format "move N from X to Y"
fn compute_hanoi_solution(n_disks: u8) -> Vec<String> {
    let mut moves = Vec::with_capacity((1 << n_disks) - 1);
    solve_recursive(n_disks, 'A', 'C', 'B', &mut moves);
    moves
}

/// Recursive helper for Hanoi solution generation
///
/// # Parameters
/// - `n`: Number of disks to move in this subproblem
/// - `from`: Source tower (A, B, or C)
/// - `to`: Destination tower (A, B, or C)
/// - `aux`: Auxiliary tower (A, B, or C)
/// - `moves`: Accumulator for move sequence
fn solve_recursive(n: u8, from: char, to: char, aux: char, moves: &mut Vec<String>) {
    if n == 0 {
        return;
    }
    solve_recursive(n - 1, from, aux, to, moves);
    moves.push(format!("move {} from {} to {}", n, from, to));
    solve_recursive(n - 1, aux, to, from, moves);
}

fn main() {
    // === Setup: Parse arguments and configure execution ===
    let args = Args::parse();
    let config = HanoiDemoConfig::default();

    // Validate accuracy range (clap doesn't support f64 range validation)
    if args.accuracy <= 0.5 || args.accuracy >= 1.0 {
        eprintln!(
            "Error: accuracy must be in (0.5, 1.0), got {}",
            args.accuracy
        );
        std::process::exit(1);
    }

    let total_steps = (1usize << args.disks) - 1;

    // Check if we should use mock mode (for CI)
    let use_mock = env::var("MAKER_USE_MOCK").unwrap_or_default() == "1";

    // Determine model name
    let model_name =
        args.model
            .clone()
            .unwrap_or_else(|| match args.provider.to_lowercase().as_str() {
                "ollama" => DEFAULT_OLLAMA_MODEL.to_string(),
                "openai" => DEFAULT_OPENAI_MODEL.to_string(),
                "anthropic" => DEFAULT_ANTHROPIC_MODEL.to_string(),
                _ => DEFAULT_OLLAMA_MODEL.to_string(),
            });

    println!("=== MAKER Towers of Hanoi Demo (with Adaptive Voting) ===\n");
    println!("Disks:      {}", args.disks);
    println!("Steps:      {} (2^{} - 1)", total_steps, args.disks);
    println!(
        "Target:     {:.0}% task reliability",
        config.target_reliability * 100.0
    );

    // Determine ensemble configuration
    let ensemble_provider = if args.ensemble {
        Some(args.ensemble_provider.clone().unwrap_or_else(|| {
            // Default: pair with a different provider
            match args.provider.to_lowercase().as_str() {
                "ollama" => "openai".to_string(),
                "openai" => "ollama".to_string(),
                "anthropic" => "openai".to_string(),
                _ => "openai".to_string(),
            }
        }))
    } else {
        None
    };

    if use_mock {
        println!("Mode:       Mock (accuracy: {:.0}%)", args.accuracy * 100.0);
        let k = calculate_kmin(args.accuracy, config.target_reliability, total_steps, 1)
            .expect("Valid parameters");
        println!("k-margin:   {} (fixed)", k);
    } else if let Some(ref ens_provider) = ensemble_provider {
        println!(
            "Mode:       Ensemble ({} + {})",
            args.provider, ens_provider
        );
        println!(
            "k-margin:   Adaptive (min={}, max={})",
            config.k_min_floor, config.k_max_ceiling
        );
    } else {
        println!("Mode:       {} ({})", args.provider, model_name);
        println!(
            "k-margin:   Adaptive (min={}, max={})",
            config.k_min_floor, config.k_max_ceiling
        );
    }
    println!();

    // Compute ground truth solution
    let solution = compute_hanoi_solution(args.disks);
    assert_eq!(solution.len(), total_steps);

    let start = Instant::now();

    // === Execution: Run voting loop in selected mode ===
    if args.strict {
        println!("Strict:     ON (halt on first error)");
    }

    let stats = if use_mock {
        // Original mock-based implementation
        run_mock_mode(
            args.disks,
            &solution,
            args.accuracy,
            total_steps,
            &config,
            args.strict,
        )
    } else {
        // LLM-based implementation with adaptive voting
        match run_llm_mode(
            args.disks,
            &solution,
            total_steps,
            &args.provider,
            &model_name,
            ensemble_provider.as_deref(),
            &config,
            args.strict,
        ) {
            Ok(stats) => stats,
            Err(e) => {
                eprintln!("\n{}", e);
                std::process::exit(1);
            }
        }
    };

    let elapsed = start.elapsed();
    let avg_samples = stats.total_samples as f64 / total_steps as f64;

    // === Results: Display statistics and exit status ===
    println!();
    println!("=== Results ===");
    println!("Total steps:      {}", total_steps);
    println!("Errors:           {}", stats.errors);
    println!("Total samples:    {}", stats.total_samples);
    println!("Avg samples/step: {:.1}", avg_samples);
    println!("Red-flagged:      {}", stats.total_red_flagged);
    println!("Elapsed:          {:.2?}", elapsed);
    println!(
        "Cost (tokens):    ~{} (input) + ~{} (output)",
        stats.total_samples * 100,
        stats.total_samples * 50
    );

    if stats.errors == 0 {
        println!(
            "\n[SUCCESS] Zero errors on {}-disk Hanoi ({} steps)!",
            args.disks, total_steps
        );
    } else {
        println!(
            "\n[FAILURE] {} errors on {}-disk Hanoi ({} steps)",
            stats.errors, args.disks, total_steps
        );
        std::process::exit(1);
    }
}

fn run_mock_mode(
    n_disks: u8,
    solution: &[String],
    accuracy: f64,
    total_steps: usize,
    config: &HanoiDemoConfig,
    strict: bool,
) -> HanoiStats {
    // === Setup: Calculate k-margin and initialize statistics ===
    let k = calculate_kmin(accuracy, config.target_reliability, total_steps, 1)
        .expect("Valid parameters");
    let mut stats = HanoiStats::default();

    // === Voting Loop: Process each step with mock client ===
    for (step, expected_move) in solution.iter().enumerate() {
        // Create a biased mock client for this step:
        // - accuracy% of responses are the correct move
        // - (1-accuracy)% are a wrong move
        let correct_count = (config.pool_size as f64 * accuracy).round() as usize;
        let mut responses = vec![expected_move.clone(); correct_count];
        let wrong_move = "move 1 from A to B".to_string(); // Deterministic wrong answer
        responses.extend(vec![wrong_move; config.pool_size - correct_count]);

        let client = MockLlmClient::new(responses);
        let vote_config = VoteConfig::default()
            .with_max_samples(config.pool_size)
            .without_token_limit();

        match vote_with_margin(&format!("Step {}", step), k, &client, vote_config) {
            Ok(result) => {
                stats.total_samples += result.total_samples;
                stats.total_red_flagged += result.red_flagged;

                if result.winner != *expected_move {
                    stats.errors += 1;
                    if n_disks <= 5 {
                        eprintln!(
                            "  ERROR Step {}: expected '{}', got '{}'",
                            step + 1,
                            expected_move,
                            result.winner
                        );
                    }
                    if strict {
                        eprintln!("  HALTING: Strict mode enabled");
                        return stats;
                    }
                } else if should_print_step(n_disks, step, total_steps, config) {
                    println!(
                        "Step {:>4}/{}: {} (samples: {})",
                        step + 1,
                        total_steps,
                        result.winner,
                        result.total_samples
                    );
                }
            }
            Err(e) => {
                stats.errors += 1;
                eprintln!("  FAIL Step {}: {}", step + 1, e);
                if strict {
                    eprintln!("  HALTING: Strict mode enabled");
                    return stats;
                }
            }
        }
    }

    stats
}

/// Represents the current state of the three Hanoi towers
#[derive(Clone, Debug)]
struct HanoiTowers {
    /// Tower A (source) - disks stored bottom-to-top (largest first)
    a: Vec<u8>,
    /// Tower B (auxiliary)
    b: Vec<u8>,
    /// Tower C (destination)
    c: Vec<u8>,
}

impl HanoiTowers {
    /// Create initial state with all disks on tower A
    fn new(n_disks: u8) -> Self {
        // Disks numbered 1 (smallest) to n (largest), stored bottom-to-top
        let a: Vec<u8> = (1..=n_disks).rev().collect();
        Self {
            a,
            b: Vec::new(),
            c: Vec::new(),
        }
    }

    /// Format tower showing all disks (for genuine prompts)
    fn format_tower_full(disks: &[u8]) -> String {
        if disks.is_empty() {
            "empty".to_string()
        } else {
            format!("{:?}", disks)
        }
    }

    /// Apply a move to the towers, returns Ok if valid, Err if invalid
    fn apply_move(&mut self, move_str: &str) -> Result<(), String> {
        // Parse "move N from X to Y"
        let parts: Vec<&str> = move_str.split_whitespace().collect();
        if parts.len() != 6 || parts[0] != "move" || parts[2] != "from" || parts[4] != "to" {
            return Err(format!("Invalid move format: {}", move_str));
        }

        let disk: u8 = parts[1]
            .parse()
            .map_err(|_| format!("Invalid disk number: {}", parts[1]))?;
        let from = parts[3].to_uppercase();
        let to = parts[5].to_uppercase();

        // Get source and destination towers
        let source = match from.as_str() {
            "A" => &mut self.a,
            "B" => &mut self.b,
            "C" => &mut self.c,
            _ => return Err(format!("Invalid source tower: {}", from)),
        };

        // Check that the disk is on top of the source tower
        if source.last() != Some(&disk) {
            return Err(format!(
                "Disk {} is not on top of tower {} (top is {:?})",
                disk,
                from,
                source.last()
            ));
        }

        source.pop();

        let dest = match to.as_str() {
            "A" => &mut self.a,
            "B" => &mut self.b,
            "C" => &mut self.c,
            _ => return Err(format!("Invalid destination tower: {}", to)),
        };

        // Check that we're not placing larger disk on smaller
        if let Some(&top) = dest.last() {
            if disk > top {
                return Err(format!(
                    "Cannot place disk {} on top of disk {} (larger on smaller)",
                    disk, top
                ));
            }
        }

        dest.push(disk);
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn run_llm_mode(
    n_disks: u8,
    solution: &[String],
    total_steps: usize,
    provider: &str,
    model: &str,
    ensemble_provider: Option<&str>,
    config: &HanoiDemoConfig,
    strict: bool,
) -> Result<HanoiStats, String> {
    // === Client Setup: Initialize LLM client(s) and verify connection ===

    // Helper to format error messages
    let format_error = |e: String| -> String {
        if e.contains("OPENAI_API_KEY") {
            format!(
                "Error: OpenAI requires OPENAI_API_KEY environment variable.\n\
                 Set it with: export OPENAI_API_KEY=your-key-here\n\
                 Original error: {}",
                e
            )
        } else if e.contains("ANTHROPIC_API_KEY") {
            format!(
                "Error: Anthropic requires ANTHROPIC_API_KEY environment variable.\n\
                 Set it with: export ANTHROPIC_API_KEY=your-key-here\n\
                 Original error: {}",
                e
            )
        } else {
            e
        }
    };

    // Create primary provider client
    let primary_client =
        setup_provider_client(provider, Some(model.to_string())).map_err(&format_error)?;

    // Test primary connection
    println!("Testing {} connection...", provider);
    if let Err(e) = primary_client.generate("test", 0.0) {
        let error_msg = match provider.to_lowercase().as_str() {
            "ollama" => format!(
                "Error: Cannot connect to Ollama at http://localhost:11434.\n\
                 Please ensure Ollama is running with: ollama serve\n\
                 And that the '{}' model is available.\n\
                 Connection error: {}",
                model, e
            ),
            _ => format!(
                "Error: Cannot connect to {} provider.\n\
                 Connection error: {}",
                provider, e
            ),
        };
        return Err(error_msg);
    }

    // Create ensemble client if secondary provider specified
    let client: Box<dyn LlmClient> = if let Some(ens_provider) = ensemble_provider {
        let secondary_client = setup_provider_client(ens_provider, None).map_err(&format_error)?;

        println!("Testing {} connection...", ens_provider);
        if let Err(e) = secondary_client.generate("test", 0.0) {
            return Err(format!(
                "Error: Cannot connect to ensemble provider {}.\n\
                 Connection error: {}",
                ens_provider, e
            ));
        }
        println!("Ensemble mode: {} + {}", provider, ens_provider);

        Box::new(EnsembleClient::new(primary_client, secondary_client))
    } else {
        primary_client
    };
    println!("{} connection successful.\n", provider);

    // === K-Estimator Setup: Configure adaptive margin calculation ===

    // Create adaptive K estimator with custom config
    let k_config = KEstimatorConfig {
        ema_alpha: config.ema_alpha,
        initial_p_hat: config.initial_p_hat,
        k_min_floor: config.k_min_floor,
        k_max_ceiling: config.k_max_ceiling,
    };
    let mut k_estimator = KEstimator::new(k_config);
    let mut stats = HanoiStats::default();

    // === Tower State: Track current disk positions ===
    let mut towers = HanoiTowers::new(n_disks);

    // === Execution Loop: Adaptive voting for each Hanoi step ===

    for (step, expected_move) in solution.iter().enumerate() {
        let remaining_steps = total_steps - step;

        // Use Temperature for deterministic sampling
        let vote_config =
            VoteConfig::default().with_diversity_temperature(config.voting_temperature);

        // Build prompt with few-shot examples and chain-of-thought
        let prompt = format!(
            "Tower of Hanoi: Move all disks from A to C using B as auxiliary.

RULES:
1. Move one disk at a time (only the top disk)
2. Never place larger disk on smaller disk
3. Use the OPTIMAL algorithm: On odd steps move disk 1, on even steps make the only legal non-disk-1 move

EXAMPLE (2 disks):
State: A:[2,1] B:[] C:[] → Step 1 (odd): move 1 from A to B
State: A:[2] B:[1] C:[] → Step 2 (even): move 2 from A to C
State: A:[] B:[1] C:[2] → Step 3 (odd): move 1 from B to C
Done: A:[] B:[] C:[2,1]

NOW SOLVE:
Disks: {}
Step: {} ({})
State: A:{} B:{} C:{}

Think step by step:
1. Is this an odd or even step?
2. If odd: move disk 1 to the next tower in its cycle
3. If even: find the only legal move that doesn't involve disk 1

Answer with ONLY: move N from X to Y",
            n_disks,
            step + 1,
            if (step + 1) % 2 == 1 { "odd" } else { "even" },
            HanoiTowers::format_tower_full(&towers.a),
            HanoiTowers::format_tower_full(&towers.b),
            HanoiTowers::format_tower_full(&towers.c),
        );

        match vote_with_margin_adaptive(
            &prompt,
            &mut k_estimator,
            config.target_reliability,
            remaining_steps,
            client.as_ref(),
            vote_config,
        ) {
            Ok(result) => {
                stats.total_samples += result.total_samples;
                stats.total_red_flagged += result.red_flagged;

                let k_used = result.k_used;
                let p_hat = k_estimator.p_hat();

                if result.winner != *expected_move {
                    stats.errors += 1;
                    if n_disks <= 5 {
                        eprintln!(
                            "  ERROR Step {}: expected '{}', got '{}' (k={}, p_hat={:.3})",
                            step + 1,
                            expected_move,
                            result.winner,
                            k_used,
                            p_hat
                        );
                    }
                    if strict {
                        eprintln!("  HALTING: Strict mode enabled");
                        return Ok(stats);
                    }
                } else if should_print_step(n_disks, step, total_steps, config) {
                    println!(
                        "Step {:>4}/{}: {} (samples: {}, k={}, p_hat={:.3})",
                        step + 1,
                        total_steps,
                        result.winner,
                        result.total_samples,
                        k_used,
                        p_hat
                    );
                }
            }
            Err(e) => {
                stats.errors += 1;
                eprintln!("  FAIL Step {}: {}", step + 1, e);
                if strict {
                    eprintln!("  HALTING: Strict mode enabled");
                    return Ok(stats);
                }
            }
        }

        // Update tower state using expected move (keeps state accurate for next prompt)
        if let Err(e) = towers.apply_move(expected_move) {
            eprintln!(
                "  WARNING: Failed to apply expected move '{}': {}",
                expected_move, e
            );
        }
    }

    Ok(stats)
}
