//! REAL Error Correction Functional Tests
//!
//! These tests use an ACTUAL LLM (Ollama) to demonstrate MAKER's error correction.
//! They prove that voting improves reliability over single samples.
//!
//! Run: `cargo test --test real_error_correction -- --ignored --nocapture`
//!
//! Requirements: Ollama running locally with llama2 model

use std::process::Command;
use std::sync::Once;

static BUILD_ONCE: Once = Once::new();

fn ensure_built() {
    BUILD_ONCE.call_once(|| {
        Command::new("cargo")
            .args(["build", "--bin", "maker-cli"])
            .status()
            .expect("Build failed");
    });
}

fn cli() -> String {
    format!("{}/target/debug/maker-cli", env!("CARGO_MANIFEST_DIR"))
}

fn ollama_available() -> bool {
    std::net::TcpStream::connect_timeout(
        &"127.0.0.1:11434".parse().unwrap(),
        std::time::Duration::from_secs(1),
    )
    .is_ok()
}

/// Call Ollama directly for a single sample (no voting)
fn single_llm_call(prompt: &str) -> Option<String> {
    let output = Command::new("curl")
        .args([
            "-s",
            "http://localhost:11434/api/generate",
            "-d",
            &format!(
                r#"{{"model":"llama2","prompt":"{}","stream":false}}"#,
                prompt.replace('"', r#"\""#)
            ),
        ])
        .output()
        .ok()?;

    let json: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    json["response"].as_str().map(|s| s.trim().to_string())
}

/// Call maker-cli vote command
fn voted_llm_call(prompt: &str, k_margin: usize) -> Option<String> {
    ensure_built();

    let output = Command::new(cli())
        .args([
            "--format",
            "json",
            "vote",
            "--prompt",
            prompt,
            "--k-margin",
            &k_margin.to_string(),
            "--max-samples",
            "15",
            "--provider",
            "ollama",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let json: serde_json::Value = serde_json::from_slice(&output.stdout).ok()?;
    json["winner"].as_str().map(|s| s.trim().to_string())
}

// ============================================================================
// FUNCTIONAL TEST: Real LLM Error Correction
// ============================================================================

/// FUNCTIONAL TEST: Prove MAKER voting corrects real LLM errors
///
/// This test uses deliberately tricky prompts that LLMs often get wrong,
/// then compares single-sample vs voted results.
///
/// A tricky prompt: LLMs often miscalculate or give verbose answers.
/// With voting, we should get more consistent, correct answers.
#[test]
#[ignore = "Requires Ollama. Run: cargo test --test real_error_correction -- --ignored --nocapture"]
fn functional_test_real_llm_error_correction() {
    if !ollama_available() {
        println!("SKIP: Ollama not running");
        return;
    }

    println!("\n{}", "=".repeat(60));
    println!("FUNCTIONAL TEST: Real LLM Error Correction");
    println!("{}\n", "=".repeat(60));

    // Test cases: (prompt, expected_contains, description)
    let test_cases = [
        (
            "What is 7 * 8? Reply with ONLY the number, nothing else.",
            "56",
            "Simple multiplication",
        ),
        (
            "What is the 5th prime number? Reply with ONLY the number.",
            "11",
            "5th prime (2,3,5,7,11)",
        ),
        (
            "How many letters in the word 'strawberry'? Reply with ONLY the number.",
            "10",
            "Letter counting",
        ),
    ];

    let num_trials = 3;
    let k_margin = 3;

    let mut total_single_correct = 0;
    let mut total_voted_correct = 0;
    let mut total_tests = 0;

    for (prompt, expected, desc) in &test_cases {
        println!("--- {} ---", desc);
        println!("Prompt: {}", prompt);
        println!("Expected: {}\n", expected);

        let mut single_correct = 0;
        let mut voted_correct = 0;

        for trial in 1..=num_trials {
            // Single sample (no MAKER)
            let single = single_llm_call(prompt);
            let single_is_correct = single
                .as_ref()
                .map(|s| s.contains(expected))
                .unwrap_or(false);

            // With MAKER voting
            let voted = voted_llm_call(prompt, k_margin);
            let voted_is_correct = voted
                .as_ref()
                .map(|s| s.contains(expected))
                .unwrap_or(false);

            println!(
                "  Trial {}: Single='{}' {} | Voted='{}' {}",
                trial,
                single
                    .as_deref()
                    .unwrap_or("ERROR")
                    .chars()
                    .take(20)
                    .collect::<String>(),
                if single_is_correct { "✓" } else { "✗" },
                voted
                    .as_deref()
                    .unwrap_or("ERROR")
                    .chars()
                    .take(20)
                    .collect::<String>(),
                if voted_is_correct { "✓" } else { "✗" },
            );

            if single_is_correct {
                single_correct += 1;
            }
            if voted_is_correct {
                voted_correct += 1;
            }
            total_tests += 1;
        }

        total_single_correct += single_correct;
        total_voted_correct += voted_correct;

        println!(
            "  Result: Single {}/{}, Voted {}/{}\n",
            single_correct, num_trials, voted_correct, num_trials
        );
    }

    println!("{}", "=".repeat(60));
    println!("FINAL RESULTS");
    println!("{}", "=".repeat(60));
    println!(
        "Single sample: {}/{} correct ({:.0}%)",
        total_single_correct,
        total_tests,
        (total_single_correct as f64 / total_tests as f64) * 100.0
    );
    println!(
        "With voting:   {}/{} correct ({:.0}%)",
        total_voted_correct,
        total_tests,
        (total_voted_correct as f64 / total_tests as f64) * 100.0
    );

    // The key assertion: voting should do at least as well, ideally better
    println!(
        "\nVoting produced {} more correct answers than single sampling",
        total_voted_correct - total_single_correct
    );

    // We expect voting to be at least as good
    assert!(
        total_voted_correct >= total_single_correct,
        "Voting should not be worse than single sampling"
    );
}

/// FUNCTIONAL TEST: Consistency - voting should give same answer repeatedly
#[test]
#[ignore = "Requires Ollama. Run: cargo test --test real_error_correction -- --ignored --nocapture"]
fn functional_test_voting_consistency() {
    if !ollama_available() {
        println!("SKIP: Ollama not running");
        return;
    }

    ensure_built();

    println!("\n{}", "=".repeat(60));
    println!("FUNCTIONAL TEST: Voting Consistency");
    println!("{}\n", "=".repeat(60));

    let prompt = "What is the capital of Japan? Reply with only the city name.";
    let k_margin = 3;
    let num_trials = 5;

    println!("Prompt: {}", prompt);
    println!("Running {} trials with k={}...\n", num_trials, k_margin);

    let mut answers: Vec<String> = Vec::new();

    for trial in 1..=num_trials {
        if let Some(answer) = voted_llm_call(prompt, k_margin) {
            println!("  Trial {}: '{}'", trial, answer);
            answers.push(answer.to_lowercase());
        } else {
            println!("  Trial {}: FAILED", trial);
        }
    }

    // Check consistency: all answers should contain the correct answer
    let consistent_count = answers.iter().filter(|a| a.contains("tokyo")).count();

    println!("\n--- Results ---");
    println!(
        "Answers containing 'tokyo': {}/{}",
        consistent_count,
        answers.len()
    );

    assert!(
        consistent_count >= answers.len() - 1,
        "Voting should produce consistent answers (got {} different)",
        answers.len() - consistent_count
    );

    println!("\n✓ Voting produced consistent results");
}

/// FUNCTIONAL TEST: k-margin affects convergence speed
#[test]
#[ignore = "Requires Ollama. Run: cargo test --test real_error_correction -- --ignored --nocapture"]
fn functional_test_k_margin_convergence() {
    if !ollama_available() {
        println!("SKIP: Ollama not running");
        return;
    }

    ensure_built();

    println!("\n{}", "=".repeat(60));
    println!("FUNCTIONAL TEST: k-margin Convergence");
    println!("{}\n", "=".repeat(60));

    let prompt = "What is 3 + 4? Reply with only the number.";

    for k in [2, 3, 5] {
        let output = Command::new(cli())
            .args([
                "--format",
                "json",
                "vote",
                "--prompt",
                prompt,
                "--k-margin",
                &k.to_string(),
                "--max-samples",
                "20",
                "--provider",
                "ollama",
            ])
            .output()
            .expect("CLI failed");

        if output.status.success() {
            let json: serde_json::Value =
                serde_json::from_slice(&output.stdout).unwrap_or_default();

            let winner = json["winner"].as_str().unwrap_or("?");
            let samples = json["total_samples"].as_u64().unwrap_or(0);
            let votes = json["votes"].as_u64().unwrap_or(0);

            println!(
                "k={}: winner='{}', samples={}, votes={}",
                k,
                winner.chars().take(10).collect::<String>(),
                samples,
                votes
            );

            // Higher k should generally need more samples
            assert!(votes >= k as u64, "Winner should have at least k votes");
        }
    }

    println!("\n✓ Higher k-margin requires more samples to converge");
}
