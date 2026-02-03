//! Arithmetic Demo - Testing MAKER's Error Correction with Random Errors
//!
//! This demo validates MAKER's voting-based error correction using arithmetic
//! problems where LLMs make occasional random errors (not systematic ones).
//!
//! Unlike Hanoi (which produces systematic reasoning errors), arithmetic problems
//! produce random calculation errors that voting can effectively correct.
//!
//! # Usage
//!
//! ```bash
//! # Run 20 arithmetic problems with OpenAI
//! cargo run --example arithmetic_demo -- --problems 20 --provider openai
//!
//! # Run with strict mode (halt on first error)
//! cargo run --example arithmetic_demo -- --problems 50 --provider openai --strict
//!
//! # Run with mock mode for testing
//! MAKER_USE_MOCK=1 cargo run --example arithmetic_demo -- --problems 100 --accuracy 0.85
//! ```

use clap::Parser;
use maker::core::{
    calculate_kmin, vote_with_margin, vote_with_margin_adaptive, KEstimator, KEstimatorConfig,
    MockLlmClient, VoteConfig,
};
use maker::llm::adapter::setup_provider_client;
use rand::Rng;
use std::env;
use std::time::Instant;

// Default model names per provider
const DEFAULT_OLLAMA_MODEL: &str = "gpt-oss";
const DEFAULT_OPENAI_MODEL: &str = "gpt-5-mini";
const DEFAULT_ANTHROPIC_MODEL: &str = "claude-3-5-sonnet-20241022";

/// Arithmetic problem types
#[derive(Debug, Clone, Copy)]
enum Operation {
    Add,
    Subtract,
    Multiply,
}

impl Operation {
    fn symbol(&self) -> char {
        match self {
            Operation::Add => '+',
            Operation::Subtract => '-',
            Operation::Multiply => '*',
        }
    }

    fn apply(&self, a: i64, b: i64) -> i64 {
        match self {
            Operation::Add => a + b,
            Operation::Subtract => a - b,
            Operation::Multiply => a * b,
        }
    }
}

/// An arithmetic problem with its expected answer
#[derive(Debug, Clone)]
struct Problem {
    a: i64,
    b: i64,
    op: Operation,
    answer: i64,
}

impl Problem {
    fn generate(rng: &mut impl Rng, difficulty: u32) -> Self {
        // Scale numbers based on difficulty (1-5)
        let max = 10i64.pow(difficulty.min(5));
        let a = rng.gen_range(10..max);
        let b = rng.gen_range(10..max);

        // Weight towards addition/subtraction (multiplication is harder)
        let op = match rng.gen_range(0..10) {
            0..=4 => Operation::Add,
            5..=8 => Operation::Subtract,
            _ => Operation::Multiply,
        };

        // For multiplication, use smaller numbers
        let (a, b) = if matches!(op, Operation::Multiply) {
            (a % 100, b % 100)
        } else {
            (a, b)
        };

        let answer = op.apply(a, b);

        Self { a, b, op, answer }
    }

    fn prompt(&self) -> String {
        format!(
            "Calculate: {} {} {}\n\nReply with ONLY the numeric answer (no explanation, no equals sign, just the number).",
            self.a,
            self.op.symbol(),
            self.b
        )
    }

    fn answer_str(&self) -> String {
        self.answer.to_string()
    }
}

/// Configuration for the arithmetic demo
#[derive(Clone)]
struct DemoConfig {
    target_reliability: f64,
    voting_temperature: f64,
    pool_size: usize,
    ema_alpha: f64,
    initial_p_hat: f64,
    k_min_floor: usize,
    k_max_ceiling: usize,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self {
            target_reliability: 0.95,
            voting_temperature: 0.0,
            pool_size: 100,
            ema_alpha: 0.2,
            initial_p_hat: 0.90, // Arithmetic is generally easier
            k_min_floor: 1,
            k_max_ceiling: 10,
        }
    }
}

/// Statistics tracked during execution
#[derive(Default, Clone)]
struct DemoStats {
    errors: usize,
    total_samples: usize,
    total_red_flagged: usize,
}

/// CLI arguments
#[derive(Parser, Debug)]
#[command(
    name = "arithmetic_demo",
    about = "Demonstrates MAKER's error correction with arithmetic problems"
)]
struct Args {
    /// Number of problems to solve
    #[arg(short = 'n', long, default_value = "20")]
    problems: usize,

    /// Problem difficulty 1-5 (affects number magnitude)
    #[arg(short, long, default_value = "3", value_parser = clap::value_parser!(u32).range(1..=5))]
    difficulty: u32,

    /// Model accuracy 0.51-0.99 (mock mode only)
    #[arg(short, long, default_value = "0.90")]
    accuracy: f64,

    /// LLM provider: ollama, openai, or anthropic
    #[arg(short, long, default_value = "ollama")]
    provider: String,

    /// Model name
    #[arg(short, long)]
    model: Option<String>,

    /// Strict mode: halt on first error
    #[arg(long, default_value = "false")]
    strict: bool,

    /// Random seed for reproducible problems
    #[arg(long)]
    seed: Option<u64>,
}

fn main() {
    let args = Args::parse();
    let config = DemoConfig::default();

    // Validate accuracy
    if args.accuracy <= 0.5 || args.accuracy >= 1.0 {
        eprintln!(
            "Error: accuracy must be in (0.5, 1.0), got {}",
            args.accuracy
        );
        std::process::exit(1);
    }

    let use_mock = env::var("MAKER_USE_MOCK").unwrap_or_default() == "1";

    let model_name =
        args.model
            .clone()
            .unwrap_or_else(|| match args.provider.to_lowercase().as_str() {
                "ollama" => DEFAULT_OLLAMA_MODEL.to_string(),
                "openai" => DEFAULT_OPENAI_MODEL.to_string(),
                "anthropic" => DEFAULT_ANTHROPIC_MODEL.to_string(),
                _ => DEFAULT_OLLAMA_MODEL.to_string(),
            });

    // Generate problems
    let mut rng: rand::rngs::StdRng = match args.seed {
        Some(seed) => rand::SeedableRng::seed_from_u64(seed),
        None => rand::SeedableRng::from_entropy(),
    };
    let problems: Vec<Problem> = (0..args.problems)
        .map(|_| Problem::generate(&mut rng, args.difficulty))
        .collect();

    println!("=== MAKER Arithmetic Demo (Error Correction Validation) ===\n");
    println!("Problems:   {}", args.problems);
    println!(
        "Difficulty: {} (numbers up to {})",
        args.difficulty,
        10i64.pow(args.difficulty.min(5))
    );
    println!(
        "Target:     {:.0}% task reliability",
        config.target_reliability * 100.0
    );

    if use_mock {
        println!("Mode:       Mock (accuracy: {:.0}%)", args.accuracy * 100.0);
        let k = calculate_kmin(args.accuracy, config.target_reliability, args.problems, 1)
            .expect("Valid parameters");
        println!("k-margin:   {} (fixed)", k);
    } else {
        println!("Mode:       {} ({})", args.provider, model_name);
        println!(
            "k-margin:   Adaptive (min={}, max={})",
            config.k_min_floor, config.k_max_ceiling
        );
    }
    if args.strict {
        println!("Strict:     ON (halt on first error)");
    }
    println!();

    let start = Instant::now();

    let stats = if use_mock {
        run_mock_mode(&problems, args.accuracy, &config, args.strict)
    } else {
        match run_llm_mode(&problems, &args.provider, &model_name, &config, args.strict) {
            Ok(stats) => stats,
            Err(e) => {
                eprintln!("\n{}", e);
                std::process::exit(1);
            }
        }
    };

    let elapsed = start.elapsed();
    let completed = if args.strict && stats.errors > 0 {
        args.problems - (args.problems - stats.total_samples / 2) // Approximate
    } else {
        args.problems
    };

    println!();
    println!("=== Results ===");
    println!("Problems:       {}", args.problems);
    println!("Completed:      {}", completed);
    println!("Errors:         {}", stats.errors);
    println!("Total samples:  {}", stats.total_samples);
    println!(
        "Avg samples:    {:.1}",
        stats.total_samples as f64 / completed.max(1) as f64
    );
    println!("Red-flagged:    {}", stats.total_red_flagged);
    println!("Elapsed:        {:.2?}", elapsed);

    if stats.errors == 0 {
        println!(
            "\n[SUCCESS] Zero errors on {} arithmetic problems!",
            args.problems
        );
    } else {
        println!(
            "\n[FAILURE] {} errors on {} arithmetic problems",
            stats.errors, args.problems
        );
        std::process::exit(1);
    }
}

fn run_mock_mode(
    problems: &[Problem],
    accuracy: f64,
    config: &DemoConfig,
    strict: bool,
) -> DemoStats {
    let k = calculate_kmin(accuracy, config.target_reliability, problems.len(), 1)
        .expect("Valid parameters");
    let mut stats = DemoStats::default();

    for (i, problem) in problems.iter().enumerate() {
        let correct_answer = problem.answer_str();
        let correct_count = (config.pool_size as f64 * accuracy).round() as usize;
        let mut responses = vec![correct_answer.clone(); correct_count];
        // Wrong answers are off by a random amount
        let wrong_answer = (problem.answer + 1).to_string();
        responses.extend(vec![wrong_answer; config.pool_size - correct_count]);

        let client = MockLlmClient::new(responses);
        let vote_config = VoteConfig::default()
            .with_max_samples(config.pool_size)
            .without_token_limit();

        match vote_with_margin(&format!("Problem {}", i + 1), k, &client, vote_config) {
            Ok(result) => {
                stats.total_samples += result.total_samples;
                stats.total_red_flagged += result.red_flagged;

                // Normalize answers for comparison (strip whitespace)
                let voted = result.winner.trim();
                let expected = correct_answer.trim();

                if voted != expected {
                    stats.errors += 1;
                    eprintln!(
                        "  ERROR Problem {}: {} {} {} = {} (got '{}')",
                        i + 1,
                        problem.a,
                        problem.op.symbol(),
                        problem.b,
                        expected,
                        voted
                    );
                    if strict {
                        eprintln!("  HALTING: Strict mode enabled");
                        return stats;
                    }
                } else {
                    println!(
                        "Problem {:>3}/{}: {} {} {} = {} (samples: {})",
                        i + 1,
                        problems.len(),
                        problem.a,
                        problem.op.symbol(),
                        problem.b,
                        voted,
                        result.total_samples
                    );
                }
            }
            Err(e) => {
                stats.errors += 1;
                eprintln!("  FAIL Problem {}: {}", i + 1, e);
                if strict {
                    eprintln!("  HALTING: Strict mode enabled");
                    return stats;
                }
            }
        }
    }

    stats
}

fn run_llm_mode(
    problems: &[Problem],
    provider: &str,
    model: &str,
    config: &DemoConfig,
    strict: bool,
) -> Result<DemoStats, String> {
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

    let client = setup_provider_client(provider, Some(model.to_string())).map_err(&format_error)?;

    println!("Testing {} connection...", provider);
    if let Err(e) = client.generate("test", 0.0) {
        return Err(format!(
            "Error: Cannot connect to {} provider.\nConnection error: {}",
            provider, e
        ));
    }
    println!("{} connection successful.\n", provider);

    let k_config = KEstimatorConfig {
        ema_alpha: config.ema_alpha,
        initial_p_hat: config.initial_p_hat,
        k_min_floor: config.k_min_floor,
        k_max_ceiling: config.k_max_ceiling,
    };
    let mut k_estimator = KEstimator::new(k_config);
    let mut stats = DemoStats::default();

    for (i, problem) in problems.iter().enumerate() {
        let remaining = problems.len() - i;
        let vote_config =
            VoteConfig::default().with_diversity_temperature(config.voting_temperature);

        let prompt = problem.prompt();
        let expected = problem.answer_str();

        match vote_with_margin_adaptive(
            &prompt,
            &mut k_estimator,
            config.target_reliability,
            remaining,
            client.as_ref(),
            vote_config,
        ) {
            Ok(result) => {
                stats.total_samples += result.total_samples;
                stats.total_red_flagged += result.red_flagged;

                let k_used = result.k_used;
                let p_hat = k_estimator.p_hat();

                // Normalize and compare
                let voted = result.winner.trim();

                if voted != expected {
                    stats.errors += 1;
                    eprintln!(
                        "  ERROR Problem {}: {} {} {} = {} (got '{}', k={}, p_hat={:.3})",
                        i + 1,
                        problem.a,
                        problem.op.symbol(),
                        problem.b,
                        expected,
                        voted,
                        k_used,
                        p_hat
                    );
                    if strict {
                        eprintln!("  HALTING: Strict mode enabled");
                        return Ok(stats);
                    }
                } else {
                    println!(
                        "Problem {:>3}/{}: {} {} {} = {} (samples: {}, k={}, p_hat={:.3})",
                        i + 1,
                        problems.len(),
                        problem.a,
                        problem.op.symbol(),
                        problem.b,
                        voted,
                        result.total_samples,
                        k_used,
                        p_hat
                    );
                }
            }
            Err(e) => {
                stats.errors += 1;
                eprintln!("  FAIL Problem {}: {}", i + 1, e);
                if strict {
                    eprintln!("  HALTING: Strict mode enabled");
                    return Ok(stats);
                }
            }
        }
    }

    Ok(stats)
}
