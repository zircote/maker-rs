//! End-to-End Towers of Hanoi with Voting
//!
//! Demonstrates MAKER's zero-error execution on the Towers of Hanoi using
//! the complete voting engine with mock LLM responses at configurable accuracy.
//!
//! # Usage
//!
//! ```bash
//! # Run with 3 disks (7 steps) and default accuracy
//! cargo run --example hanoi_demo -- --disks 3
//!
//! # Run with 10 disks (1,023 steps) and 85% accuracy
//! cargo run --example hanoi_demo -- --disks 10 --accuracy 0.85
//! ```

use maker::core::{calculate_kmin, vote_with_margin, MockLlmClient, VoteConfig};
use std::env;
use std::time::Instant;

/// Compute the optimal Hanoi move sequence
fn compute_hanoi_solution(n_disks: u8) -> Vec<String> {
    let mut moves = Vec::with_capacity((1 << n_disks) - 1);
    solve_recursive(n_disks, 'A', 'C', 'B', &mut moves);
    moves
}

fn solve_recursive(n: u8, from: char, to: char, aux: char, moves: &mut Vec<String>) {
    if n == 0 {
        return;
    }
    solve_recursive(n - 1, from, aux, to, moves);
    moves.push(format!("move {} from {} to {}", n, from, to));
    solve_recursive(n - 1, aux, to, from, moves);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let (n_disks, accuracy) = parse_args(&args);

    let total_steps = (1usize << n_disks) - 1;
    let t = 0.95;
    let k = calculate_kmin(accuracy, t, total_steps, 1).expect("Valid parameters");

    println!("=== MAKER Towers of Hanoi Demo (with Voting) ===\n");
    println!("Disks:      {}", n_disks);
    println!("Steps:      {} (2^{} - 1)", total_steps, n_disks);
    println!("Accuracy:   {:.0}%", accuracy * 100.0);
    println!("Target:     {:.0}% task reliability", t * 100.0);
    println!("k-margin:   {}", k);
    println!();

    // Compute ground truth solution
    let solution = compute_hanoi_solution(n_disks);
    assert_eq!(solution.len(), total_steps);

    let start = Instant::now();
    let mut errors = 0;
    let mut total_samples = 0;
    let mut total_red_flagged = 0;

    for (step, expected_move) in solution.iter().enumerate() {
        // Create a biased mock client for this step:
        // - accuracy% of responses are the correct move
        // - (1-accuracy)% are a wrong move
        let pool_size = 100;
        let correct_count = (pool_size as f64 * accuracy).round() as usize;
        let mut responses = vec![expected_move.clone(); correct_count];
        let wrong_move = "move 1 from A to B".to_string(); // Deterministic wrong answer
        responses.extend(vec![wrong_move; pool_size - correct_count]);

        let client = MockLlmClient::new(responses);
        let config = VoteConfig::default()
            .with_max_samples(pool_size)
            .without_token_limit();

        match vote_with_margin(&format!("Step {}", step), k, &client, config) {
            Ok(result) => {
                total_samples += result.total_samples;
                total_red_flagged += result.red_flagged;

                if result.winner != *expected_move {
                    errors += 1;
                    if n_disks <= 5 {
                        eprintln!(
                            "  ERROR Step {}: expected '{}', got '{}'",
                            step + 1,
                            expected_move,
                            result.winner
                        );
                    }
                } else if n_disks <= 5 || step % 100 == 0 || step == total_steps - 1 {
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
                errors += 1;
                eprintln!("  FAIL Step {}: {}", step + 1, e);
            }
        }
    }

    let elapsed = start.elapsed();
    let avg_samples = total_samples as f64 / total_steps as f64;

    println!();
    println!("=== Results ===");
    println!("Total steps:      {}", total_steps);
    println!("Errors:           {}", errors);
    println!("Total samples:    {}", total_samples);
    println!("Avg samples/step: {:.1}", avg_samples);
    println!("Red-flagged:      {}", total_red_flagged);
    println!("Elapsed:          {:.2?}", elapsed);
    println!(
        "Cost (tokens):    ~{} (input) + ~{} (output)",
        total_samples * 100,
        total_samples * 50
    );

    if errors == 0 {
        println!(
            "\n[SUCCESS] Zero errors on {}-disk Hanoi ({} steps)!",
            n_disks, total_steps
        );
    } else {
        println!(
            "\n[FAILURE] {} errors on {}-disk Hanoi ({} steps)",
            errors, n_disks, total_steps
        );
        std::process::exit(1);
    }
}

fn parse_args(args: &[String]) -> (u8, f64) {
    let mut n_disks: u8 = 3;
    let mut accuracy: f64 = 0.85;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--disks" | "-d" => {
                if i + 1 < args.len() {
                    n_disks = args[i + 1].parse().expect("--disks requires a number 1-20");
                    i += 1;
                }
            }
            "--accuracy" | "-a" => {
                if i + 1 < args.len() {
                    accuracy = args[i + 1]
                        .parse()
                        .expect("--accuracy requires a float 0.51-0.99");
                    i += 1;
                }
            }
            "--help" | "-h" => {
                println!("Usage: hanoi_demo [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --disks, -d <N>       Number of disks (1-20, default: 3)");
                println!("  --accuracy, -a <P>    Model accuracy 0.51-0.99 (default: 0.85)");
                println!("  --help, -h            Show this help");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    if n_disks == 0 || n_disks > 20 {
        eprintln!("Error: disks must be 1-20");
        std::process::exit(1);
    }
    if accuracy <= 0.5 || accuracy >= 1.0 {
        eprintln!("Error: accuracy must be in (0.5, 1.0)");
        std::process::exit(1);
    }

    (n_disks, accuracy)
}
