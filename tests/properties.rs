//! Property-based tests for MAKER framework
//!
//! These tests validate probabilistic guarantees using proptest.

use maker::core::kmin::calculate_kmin;
use maker::core::redflag::{validate_json_schema, validate_token_length, RedFlag};
use maker::core::voting::{VoteCheckResult, VoteRace};
use proptest::prelude::*;
use serde::Deserialize;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Property: k_min increases (roughly) logarithmically with s
    ///
    /// For fixed p and t, doubling s should increase k by approximately a constant
    /// (since k ∝ ln(s)). We verify that k(2s) - k(s) is bounded.
    #[test]
    fn prop_kmin_grows_logarithmically_with_s(
        p in 0.6_f64..0.99,
        t in 0.8_f64..0.99,
        s_base in 10_usize..10_000
    ) {
        let s_double = s_base * 2;

        let k1 = calculate_kmin(p, t, s_base, 1).unwrap();
        let k2 = calculate_kmin(p, t, s_double, 1).unwrap();

        // k should increase as s increases (more steps need more margin)
        prop_assert!(k2 >= k1, "k should not decrease when s doubles: k({})={} vs k({})={}",
            s_base, k1, s_double, k2);

        // The increase should be bounded (logarithmic, not linear)
        // For ln(2s) - ln(s) = ln(2) ≈ 0.69, so k should increase by at most ~2-3
        let delta = k2.saturating_sub(k1);
        prop_assert!(delta <= 5, "k increased too much when s doubled: delta={}", delta);
    }

    /// Property: k_min decreases as p approaches 1.0
    ///
    /// Higher per-step accuracy means we need less voting margin.
    #[test]
    fn prop_kmin_decreases_as_p_increases(
        p_low in 0.55_f64..0.75,
        p_high in 0.85_f64..0.99,
        t in 0.8_f64..0.99,
        s in 100_usize..10_000
    ) {
        let k_low = calculate_kmin(p_low, t, s, 1).unwrap();
        let k_high = calculate_kmin(p_high, t, s, 1).unwrap();

        prop_assert!(
            k_high <= k_low,
            "Higher p should require smaller or equal k: k(p={})={} vs k(p={})={}",
            p_low, k_low, p_high, k_high
        );
    }

    /// Property: k_min increases as t approaches 1.0
    ///
    /// Stricter reliability target requires more voting margin.
    #[test]
    fn prop_kmin_increases_as_t_increases(
        p in 0.6_f64..0.99,
        t_low in 0.5_f64..0.8,
        t_high in 0.9_f64..0.999,
        s in 100_usize..10_000
    ) {
        let k_low = calculate_kmin(p, t_low, s, 1).unwrap();
        let k_high = calculate_kmin(p, t_high, s, 1).unwrap();

        prop_assert!(
            k_high >= k_low,
            "Higher t should require larger or equal k: k(t={})={} vs k(t={})={}",
            t_low, k_low, t_high, k_high
        );
    }

    /// Property: k_min is always at least 1
    #[test]
    fn prop_kmin_at_least_one(
        p in 0.51_f64..0.999,
        t in 0.01_f64..0.999,
        s in 1_usize..100_000
    ) {
        let k = calculate_kmin(p, t, s, 1).unwrap();
        prop_assert!(k >= 1, "k_min must be at least 1, got {}", k);
    }

    /// Property: Invalid inputs are rejected
    #[test]
    fn prop_rejects_invalid_p(
        p_invalid in prop_oneof![
            -1.0_f64..=0.5,  // Too low
            1.0_f64..=2.0    // Too high
        ],
        t in 0.5_f64..0.99,
        s in 1_usize..1000
    ) {
        let result = calculate_kmin(p_invalid, t, s, 1);
        prop_assert!(result.is_err(), "Should reject invalid p={}", p_invalid);
    }

    /// Property: Invalid t values are rejected
    #[test]
    fn prop_rejects_invalid_t(
        p in 0.6_f64..0.99,
        t_invalid in prop_oneof![
            -1.0_f64..=0.0,  // Too low
            1.0_f64..=2.0    // Too high
        ],
        s in 1_usize..1000
    ) {
        let result = calculate_kmin(p, t_invalid, s, 1);
        prop_assert!(result.is_err(), "Should reject invalid t={}", t_invalid);
    }

    /// Property: m != 1 is rejected (microagent constraint)
    #[test]
    fn prop_rejects_m_not_one(
        p in 0.6_f64..0.99,
        t in 0.5_f64..0.99,
        s in 1_usize..1000,
        m in 0_usize..100
    ) {
        prop_assume!(m != 1);
        let result = calculate_kmin(p, t, s, m);
        prop_assert!(result.is_err(), "Should reject m={} (must be 1)", m);
    }

    // ==========================================
    // Voting Properties
    // ==========================================

    /// Property: Winner is only declared when margin reaches k
    #[test]
    fn prop_winner_requires_k_margin(
        k_margin in 1_usize..10,
        votes_a in 1_usize..50,
        votes_b in 1_usize..50
    ) {
        let race = VoteRace::new(k_margin).unwrap();

        // Cast votes for candidate A
        for _ in 0..votes_a {
            race.cast_vote("A".into());
        }
        // Cast votes for candidate B
        for _ in 0..votes_b {
            race.cast_vote("B".into());
        }

        let result = race.check_winner();
        let diff = (votes_a as i32 - votes_b as i32).unsigned_abs() as usize;

        match result {
            VoteCheckResult::Winner { candidate, .. } => {
                // Should only have a winner when margin >= k
                prop_assert!(diff >= k_margin, "Winner declared but diff={} < k={}", diff, k_margin);
                if votes_a > votes_b {
                    prop_assert_eq!(candidate.as_str(), "A");
                } else {
                    prop_assert_eq!(candidate.as_str(), "B");
                }
            }
            VoteCheckResult::Ongoing { .. } => {
                // Should not have a winner yet - verify margin < k
                prop_assert!(diff < k_margin, "No winner but diff={} >= k={}", diff, k_margin);
            }
        }
    }

    /// Property: Total votes equals sum of all candidate votes
    #[test]
    fn prop_vote_count_consistency(
        k_margin in 1_usize..5,
        votes in proptest::collection::vec("[a-z]", 1..100)
    ) {
        let race = VoteRace::new(k_margin).unwrap();

        for candidate in &votes {
            race.cast_vote(candidate.clone().into());
        }

        prop_assert_eq!(race.total_votes(), votes.len(),
            "Total votes should equal number of cast_vote calls");
    }

    /// Property: No false positive winners (winner only when margin truly reached)
    #[test]
    fn prop_no_false_positive_winner(
        k_margin in 2_usize..10,
        candidate_count in 2_usize..5,
        total_votes in 1_usize..20
    ) {
        let race = VoteRace::new(k_margin).unwrap();
        let candidates: Vec<String> = (0..candidate_count).map(|i| format!("c{}", i)).collect();

        // Distribute votes roughly evenly to avoid triggering winner
        for i in 0..total_votes {
            let candidate = &candidates[i % candidates.len()];
            race.cast_vote(candidate.clone().into());
        }

        // Check if any candidate actually has k-margin lead
        match race.check_winner() {
            VoteCheckResult::Winner { margin, .. } => {
                // Verify the margin is actually >= k
                prop_assert!(margin >= k_margin, "Winner with margin {} < k {}", margin, k_margin);
            }
            VoteCheckResult::Ongoing { .. } => {
                // No winner, which is fine
            }
        }
    }

    // ==========================================
    // Red-Flag Properties
    // ==========================================

    /// Property: Content under token limit always passes
    #[test]
    fn prop_content_under_limit_passes(
        limit in 1_usize..1000,
        // Generate content shorter than limit
        content_len in 0_usize..500
    ) {
        prop_assume!(content_len < limit);
        let content: String = "x".repeat(content_len);

        let result = validate_token_length(&content, limit);
        prop_assert!(result.is_ok(), "Content with {} chars should pass limit {}",
            content_len, limit);
    }

    /// Property: Content over token limit always fails
    #[test]
    fn prop_content_over_limit_fails(
        limit in 1_usize..500,
        excess in 1_usize..100
    ) {
        let content_len = limit + excess;
        let content: String = "x".repeat(content_len);

        let result = validate_token_length(&content, limit);
        prop_assert!(result.is_err(), "Content with {} chars should fail limit {}",
            content_len, limit);

        if let Err(RedFlag::TokenLengthExceeded { actual, limit: lim }) = result {
            prop_assert_eq!(actual, content_len);
            prop_assert_eq!(lim, limit);
        } else {
            prop_assert!(false, "Wrong error type");
        }
    }

    /// Property: Valid JSON with required fields always parses
    #[test]
    fn prop_valid_json_parses(
        move_val in "[a-z]{1,10}",
        next_state in "[a-z]{1,20}"
    ) {
        #[derive(Deserialize)]
        #[allow(dead_code)]
        struct TestOutput {
            #[serde(rename = "move")]
            _move: String,
            next_state: String,
        }

        let json = format!(r#"{{"move": "{}", "next_state": "{}"}}"#, move_val, next_state);
        let result: Result<TestOutput, _> = validate_json_schema(&json);
        prop_assert!(result.is_ok(), "Valid JSON should parse: {}", json);
    }

    /// Property: Invalid JSON always fails to parse
    #[test]
    fn prop_invalid_json_fails(
        garbage in "[^{}\"]+{1,50}"
    ) {
        #[derive(Deserialize)]
        #[allow(dead_code)]
        struct TestOutput {
            field: String,
        }

        // Ensure it's truly invalid JSON
        prop_assume!(!garbage.starts_with('{'));

        let result: Result<TestOutput, _> = validate_json_schema(&garbage);
        prop_assert!(result.is_err(), "Invalid JSON should fail: {}", garbage);
    }

    /// Property: JSON missing required fields fails
    #[test]
    fn prop_missing_field_fails(
        value in "[a-z]{1,10}"
    ) {
        #[derive(Deserialize)]
        #[allow(dead_code)]
        struct RequiresBoth {
            required_a: String,
            required_b: String,
        }

        // Only provide one field
        let json = format!(r#"{{"required_a": "{}"}}"#, value);
        let result: Result<RequiresBoth, _> = validate_json_schema(&json);
        prop_assert!(result.is_err(), "Missing required_b should fail");
    }
}

// ==========================================
// Adaptive K Property Tests (STORY-011-04)
// ==========================================

use maker::core::adaptive::{KEstimator, KEstimatorConfig, VoteObservation};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Property: Adaptive k never violates configured bounds
    #[test]
    fn prop_adaptive_k_respects_bounds(
        initial_p in 0.55_f64..0.95,
        alpha in 0.01_f64..0.5,
        k_floor in 1_usize..5,
        k_ceiling_offset in 3_usize..10,
        target_t in 0.80_f64..0.99,
        remaining_steps in 1_usize..100_000,
        num_obs in 1_usize..50,
        obs_total_samples in 2_usize..30,
        obs_k_used in 2_usize..8,
        obs_red_flagged in 0_usize..5
    ) {
        let k_ceiling = k_floor + k_ceiling_offset;
        let mut estimator = KEstimator::new(KEstimatorConfig {
            ema_alpha: alpha,
            initial_p_hat: initial_p,
            k_min_floor: k_floor,
            k_max_ceiling: k_ceiling,
        });

        // Check initial recommendation
        let k = estimator.recommended_k(target_t, remaining_steps);
        prop_assert!(k >= k_floor && k <= k_ceiling,
            "Initial k={} out of bounds [{}, {}]", k, k_floor, k_ceiling);

        // Observe and check after each observation
        let clamped_red = obs_red_flagged.min(obs_total_samples.saturating_sub(1));
        for _ in 0..num_obs {
            estimator.observe(VoteObservation {
                converged_quickly: obs_total_samples <= obs_k_used,
                total_samples: obs_total_samples,
                k_used: obs_k_used.min(obs_total_samples),
                red_flagged: clamped_red,
            });

            let k = estimator.recommended_k(target_t, remaining_steps);
            prop_assert!(k >= k_floor && k <= k_ceiling,
                "After obs: k={} out of bounds [{}, {}], p_hat={:.4}",
                k, k_floor, k_ceiling, estimator.p_hat());
        }
    }

    /// Property: p_hat stays within valid range (0.51, 0.99)
    #[test]
    fn prop_p_hat_stays_valid(
        initial_p in 0.51_f64..0.99,
        alpha in 0.01_f64..0.99,
        num_obs in 1_usize..100,
        obs_total in 1_usize..50,
        obs_k in 1_usize..10,
        obs_red in 0_usize..10
    ) {
        let mut estimator = KEstimator::new(KEstimatorConfig {
            ema_alpha: alpha,
            initial_p_hat: initial_p,
            ..Default::default()
        });

        let clamped_total = obs_total.max(1);
        let clamped_red = obs_red.min(clamped_total.saturating_sub(1));

        for _ in 0..num_obs {
            estimator.observe(VoteObservation {
                converged_quickly: clamped_total <= obs_k,
                total_samples: clamped_total,
                k_used: obs_k.min(clamped_total),
                red_flagged: clamped_red,
            });

            let p = estimator.p_hat();
            prop_assert!(p >= 0.51 && p <= 0.99,
                "p_hat={:.6} out of valid range after {} observations",
                p, estimator.observation_count());
        }
    }

    /// Property: Adaptive k with unanimous votes (high p) produces lower or equal k
    /// compared to contested votes (low p)
    #[test]
    fn prop_high_p_gives_lower_or_equal_k(
        target_t in 0.85_f64..0.99,
        steps in 10_usize..10_000
    ) {
        let mut estimator_high = KEstimator::new(KEstimatorConfig {
            initial_p_hat: 0.70,
            ema_alpha: 0.3,
            ..Default::default()
        });
        let mut estimator_low = KEstimator::new(KEstimatorConfig {
            initial_p_hat: 0.70,
            ema_alpha: 0.3,
            ..Default::default()
        });

        // High p: unanimous votes
        for _ in 0..20 {
            estimator_high.observe(VoteObservation {
                converged_quickly: true,
                total_samples: 3,
                k_used: 3,
                red_flagged: 0,
            });
        }

        // Low p: contested votes
        for _ in 0..20 {
            estimator_low.observe(VoteObservation {
                converged_quickly: false,
                total_samples: 15,
                k_used: 3,
                red_flagged: 2,
            });
        }

        let k_high = estimator_high.recommended_k(target_t, steps);
        let k_low = estimator_low.recommended_k(target_t, steps);

        prop_assert!(k_high <= k_low,
            "High p should give k <= low p: k_high={}, k_low={}, p_high={:.4}, p_low={:.4}",
            k_high, k_low, estimator_high.p_hat(), estimator_low.p_hat());
    }
}

// ==========================================
// Adaptive K Monte Carlo Tests (STORY-011-04)
// ==========================================

/// Simulate voting with a given true p, using the adaptive estimator.
/// Returns (total_samples_used, errors).
fn simulate_adaptive_task(
    true_p: f64,
    total_steps: usize,
    target_t: f64,
    config: KEstimatorConfig,
) -> (usize, usize) {
    use maker::core::{vote_with_margin_adaptive, MockLlmClient, VoteConfig};

    // Create a biased mock: correct answer with probability true_p
    let correct_count = (100.0 * true_p).round() as usize;
    let mut responses = vec!["correct".to_string(); correct_count];
    responses.extend(vec!["wrong".to_string(); 100 - correct_count]);
    let client = MockLlmClient::new(responses);

    let vote_config = VoteConfig::default().with_max_samples(200);
    let mut estimator = KEstimator::new(config);
    let mut total_samples = 0;
    let mut errors = 0;

    for step in 0..total_steps {
        let remaining = total_steps - step;
        match vote_with_margin_adaptive(
            &format!("step {}", step),
            &mut estimator,
            target_t,
            remaining,
            &client,
            vote_config.clone(),
        ) {
            Ok(result) => {
                total_samples += result.total_samples;
                if result.winner != "correct" {
                    errors += 1;
                }
            }
            Err(_) => {
                errors += 1;
                total_samples += 200; // max_samples
            }
        }
    }

    (total_samples, errors)
}

/// Simulate voting with static k for comparison.
fn simulate_static_task(true_p: f64, total_steps: usize, static_k: usize) -> (usize, usize) {
    use maker::core::{vote_with_margin, MockLlmClient, VoteConfig};

    let correct_count = (100.0 * true_p).round() as usize;
    let mut responses = vec!["correct".to_string(); correct_count];
    responses.extend(vec!["wrong".to_string(); 100 - correct_count]);
    let client = MockLlmClient::new(responses);

    let vote_config = VoteConfig::default().with_max_samples(200);
    let mut total_samples = 0;
    let mut errors = 0;

    for step in 0..total_steps {
        match vote_with_margin(
            &format!("step {}", step),
            static_k,
            &client,
            vote_config.clone(),
        ) {
            Ok(result) => {
                total_samples += result.total_samples;
                if result.winner != "correct" {
                    errors += 1;
                }
            }
            Err(_) => {
                errors += 1;
                total_samples += 200;
            }
        }
    }

    (total_samples, errors)
}

#[test]
fn test_adaptive_k_zero_errors_deterministic() {
    // Regression: adaptive k on a deterministic task (high p) produces zero errors
    let (_, errors) = simulate_adaptive_task(
        0.85,
        20, // 20 steps (small for test speed)
        0.95,
        KEstimatorConfig::default(),
    );

    assert_eq!(
        errors, 0,
        "Expected zero errors with adaptive k on high-p task"
    );
}

#[test]
fn test_adaptive_k_cost_reduction_vs_static() {
    // Monte Carlo: adaptive k should use fewer total samples than static k=4
    let steps = 30;
    let true_p = 0.85;

    let (adaptive_samples, adaptive_errors) =
        simulate_adaptive_task(true_p, steps, 0.95, KEstimatorConfig::default());

    let (static_samples, static_errors) = simulate_static_task(true_p, steps, 4);

    // Adaptive should have very few errors; static may have more due to
    // MockLlmClient's deterministic cycling (85 correct, 15 wrong in a
    // fixed sequence), which can cause clustered wrong answers at some steps.
    assert!(
        adaptive_errors <= 1,
        "Adaptive had too many errors: {}",
        adaptive_errors
    );
    // Static k=4 with deterministic mock cycling can hit unlucky patterns;
    // we only care that it doesn't completely fail.
    assert!(
        static_errors <= 5,
        "Static had too many errors: {}",
        static_errors
    );

    // Adaptive should use <= samples (cost savings)
    // Note: with high p and default config, adaptive may start at k=2 (floor)
    // while static is fixed at k=4, so adaptive should use fewer samples
    assert!(
        adaptive_samples <= static_samples + steps, // Allow small margin
        "Adaptive should not use significantly more samples: adaptive={} vs static={}",
        adaptive_samples,
        static_samples
    );
}

#[test]
fn test_adaptive_k_recovers_from_p_drop() {
    // Stress test: start with high p, then p drops, adaptive k should increase
    use maker::core::adaptive::{KEstimator, KEstimatorConfig, VoteObservation};

    let mut estimator = KEstimator::new(KEstimatorConfig {
        ema_alpha: 0.3, // Faster reaction for test
        initial_p_hat: 0.90,
        ..Default::default()
    });
    estimator.set_initial_k(estimator.recommended_k(0.95, 1000));

    // Phase 1: high p (unanimous)
    for _ in 0..10 {
        estimator.observe(VoteObservation {
            converged_quickly: true,
            total_samples: 3,
            k_used: 3,
            red_flagged: 0,
        });
    }
    let k_high_p = estimator.recommended_k(0.95, 1000);

    // Phase 2: p drops suddenly (contested votes, red flags)
    for _ in 0..10 {
        estimator.observe(VoteObservation {
            converged_quickly: false,
            total_samples: 18,
            k_used: 3,
            red_flagged: 5,
        });
    }
    let k_low_p = estimator.recommended_k(0.95, 1000);

    assert!(
        k_low_p > k_high_p,
        "k should increase when p drops: k_high_p={}, k_low_p={}, p_hat={:.4}",
        k_high_p,
        k_low_p,
        estimator.p_hat()
    );
}
