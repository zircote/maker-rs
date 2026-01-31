//! Towers of Hanoi Demo for MAKER Framework
//!
//! Demonstrates MAKER's zero-error execution on the canonical
//! Towers of Hanoi problem.
//!
//! # Usage
//!
//! ```bash
//! # Run with 3 disks (7 steps)
//! cargo run --example hanoi -- --disks 3
//!
//! # Run with 10 disks (1,023 steps)
//! cargo run --example hanoi -- --disks 10
//! ```

mod decomposer;
mod state;

pub use decomposer::HanoiDecomposer;
pub use state::{compute_solution, HanoiMove, HanoiState, ROD_NAMES};

use maker::core::orchestration::{AgentOutput, TaskDecomposer, TaskOrchestrator};
use std::env;

fn main() {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let n_disks = parse_args(&args);

    println!("=== MAKER Towers of Hanoi Demo ===\n");
    println!("Disks: {}", n_disks);

    let decomposer = HanoiDecomposer::new(n_disks);
    let total_steps = decomposer.total_steps();
    println!("Total steps: {} (2^{} - 1)\n", total_steps, n_disks);

    // Create orchestrator
    let mut orchestrator = TaskOrchestrator::new(decomposer);

    println!("Initial state:");
    print_state(&HanoiState::new(n_disks));
    println!();

    // Execute all steps (simulating perfect agent responses)
    let mut errors = 0;

    while !orchestrator.is_complete() {
        let step = orchestrator.current_step_index();

        // Get expected move and state (simulating LLM response)
        // Clone values to avoid holding borrow across process_output
        let expected_move = *orchestrator.decomposer().expected_move(step).unwrap();
        let expected_state = orchestrator
            .decomposer()
            .expected_state_after(step)
            .unwrap()
            .clone();
        let move_str = expected_move.to_string();

        // Create agent output (in real usage, this comes from voting)
        let output = AgentOutput::new(move_str.clone(), expected_state.to_state());

        // Process the output
        match orchestrator.process_output(output) {
            Ok(()) => {
                if n_disks <= 5 || step.is_multiple_of(100) || step == total_steps - 1 {
                    println!("Step {}/{}: {}", step + 1, total_steps, move_str);
                }
            }
            Err(e) => {
                eprintln!("Error at step {}: {}", step, e);
                errors += 1;
            }
        }
    }

    println!();
    println!("=== Results ===");
    println!("Total steps: {}", total_steps);
    println!("Errors: {}", errors);

    // Verify final state
    let final_state =
        HanoiState::from_state(orchestrator.current_state()).expect("Valid final state");
    println!("\nFinal state:");
    print_state(&final_state);

    if final_state.is_solved() {
        println!("\n[SUCCESS] Puzzle solved correctly!");
    } else {
        println!("\n[FAILURE] Puzzle not solved correctly!");
        std::process::exit(1);
    }
}

fn parse_args(args: &[String]) -> u8 {
    let mut n_disks: u8 = 3;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--disks" | "-d" => {
                if i + 1 < args.len() {
                    n_disks = args[i + 1].parse().expect("--disks requires a number 1-20");
                    i += 1;
                }
            }
            "--help" | "-h" => {
                println!("Usage: hanoi [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --disks, -d <N>  Number of disks (1-20, default: 3)");
                println!("  --help, -h       Show this help");
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

    n_disks
}

fn print_state(state: &HanoiState) {
    for (i, rod) in state.rods.iter().enumerate() {
        println!("  Rod {}: {:?}", ROD_NAMES[i], rod);
    }
}
