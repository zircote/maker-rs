//! Adaptive K-Margin Estimation for MAKER Framework
//!
//! Dynamically adjusts the k-margin based on observed vote convergence data,
//! minimizing API calls without sacrificing reliability guarantees.
//!
//! # Algorithm
//!
//! Uses an exponential moving average (EMA) to estimate the true per-step
//! success probability `p_hat` from observed vote outcomes:
//!
//! ```text
//! p_hat = α * p_sample + (1 - α) * p_hat_prev
//! ```
//!
//! The estimated `p_hat` is then fed into the k_min formula to compute
//! the recommended k-margin for subsequent steps.
//!
//! # Bounds
//!
//! k is bounded by `[k_min_floor, k_max_ceiling]` (default `[2, 10]`) to
//! prevent runaway adjustment in either direction.

use crate::core::kmin::calculate_kmin;

/// Observation from a completed vote round, used to update the estimator.
#[derive(Debug, Clone)]
pub struct VoteObservation {
    /// Whether the winning candidate was decided within the initial k samples
    /// (indicates high effective p when true)
    pub converged_quickly: bool,
    /// Total samples needed to reach a decision
    pub total_samples: usize,
    /// The k-margin that was used for this vote
    pub k_used: usize,
    /// Number of red-flagged (discarded) samples
    pub red_flagged: usize,
}

/// Event emitted when the adaptive k-margin changes.
#[derive(Debug, Clone, PartialEq)]
pub struct KAdjusted {
    /// Previous k-margin value
    pub old_k: usize,
    /// New k-margin value
    pub new_k: usize,
    /// Current p-hat estimate
    pub p_hat: f64,
    /// Reason for the adjustment
    pub reason: String,
}

/// Adaptive k-margin estimator using EMA-based p-hat estimation.
///
/// Tracks observed vote outcomes and recommends k-margin adjustments
/// based on the estimated per-step success probability.
///
/// # Example
///
/// ```rust
/// use maker::core::adaptive::{KEstimator, KEstimatorConfig, VoteObservation};
///
/// let mut estimator = KEstimator::new(KEstimatorConfig::default());
///
/// // After each vote, observe the result
/// estimator.observe(VoteObservation {
///     converged_quickly: true,
///     total_samples: 3,
///     k_used: 3,
///     red_flagged: 0,
/// });
///
/// // Get recommended k for remaining steps
/// let k = estimator.recommended_k(0.95, 100);
/// assert!(k >= 2); // Never below floor
/// ```
#[derive(Debug, Clone)]
pub struct KEstimator {
    /// Current p-hat estimate
    p_hat: f64,
    /// Configuration
    config: KEstimatorConfig,
    /// Number of observations received
    observation_count: usize,
    /// Last recommended k (for change detection)
    last_k: Option<usize>,
    /// History of adjustments
    adjustments: Vec<KAdjusted>,
}

/// Configuration for the adaptive k-margin estimator.
#[derive(Debug, Clone)]
pub struct KEstimatorConfig {
    /// EMA smoothing factor (α). Higher values react faster to changes.
    /// Default: 0.1
    pub ema_alpha: f64,
    /// Initial p-hat estimate before any observations.
    /// Default: 0.85
    pub initial_p_hat: f64,
    /// Minimum allowed k-margin.
    /// Default: 2
    pub k_min_floor: usize,
    /// Maximum allowed k-margin.
    /// Default: 10
    pub k_max_ceiling: usize,
}

impl Default for KEstimatorConfig {
    fn default() -> Self {
        Self {
            ema_alpha: 0.1,
            initial_p_hat: 0.85,
            k_min_floor: 2,
            k_max_ceiling: 10,
        }
    }
}

impl KEstimator {
    /// Create a new estimator with the given configuration.
    pub fn new(config: KEstimatorConfig) -> Self {
        let p_hat = config.initial_p_hat;
        Self {
            p_hat,
            config,
            observation_count: 0,
            last_k: None,
            adjustments: Vec::new(),
        }
    }

    /// Observe the outcome of a vote round and update the p-hat estimate.
    ///
    /// The observed p is estimated from the vote outcome:
    /// - If the vote converged in exactly k samples (unanimous), p is high (~0.95)
    /// - If it took many more samples, p is lower
    /// - Red-flagged samples reduce the effective p
    pub fn observe(&mut self, obs: VoteObservation) -> Option<KAdjusted> {
        let p_sample = self.estimate_p_from_observation(&obs);

        // EMA update: p_hat = α * p_sample + (1 - α) * p_hat_prev
        self.p_hat = self.config.ema_alpha * p_sample + (1.0 - self.config.ema_alpha) * self.p_hat;

        // Clamp p_hat to valid range (0.5, 1.0) — voting requires p > 0.5
        self.p_hat = self.p_hat.clamp(0.51, 0.99);

        self.observation_count += 1;

        // Check if k changed
        let old_k = self.last_k;
        // Use a reasonable default for change detection
        if let Some(old) = old_k {
            let new_k = self.recommended_k(0.95, 1000);
            if new_k != old {
                let adjustment = KAdjusted {
                    old_k: old,
                    new_k,
                    p_hat: self.p_hat,
                    reason: if new_k > old {
                        format!(
                            "p_hat decreased to {:.4}, increasing k for safety",
                            self.p_hat
                        )
                    } else {
                        format!(
                            "p_hat increased to {:.4}, decreasing k for cost savings",
                            self.p_hat
                        )
                    },
                };
                self.adjustments.push(adjustment.clone());
                self.last_k = Some(new_k);
                return Some(adjustment);
            }
        }

        None
    }

    /// Get the recommended k-margin for the given target reliability and remaining steps.
    ///
    /// Uses the current p_hat estimate in the k_min formula, then clamps
    /// to `[k_min_floor, k_max_ceiling]`.
    pub fn recommended_k(&self, target_t: f64, remaining_steps: usize) -> usize {
        // Use at least 1 remaining step
        let steps = remaining_steps.max(1);

        // Calculate k_min using current p_hat estimate
        let k = match calculate_kmin(self.p_hat, target_t, steps, 1) {
            Ok(k) => k,
            Err(_) => {
                // If p_hat is out of range, use the ceiling as a safe default
                self.config.k_max_ceiling
            }
        };

        // Clamp to bounds
        k.clamp(self.config.k_min_floor, self.config.k_max_ceiling)
    }

    /// Get the current p-hat estimate.
    pub fn p_hat(&self) -> f64 {
        self.p_hat
    }

    /// Get the number of observations received.
    pub fn observation_count(&self) -> usize {
        self.observation_count
    }

    /// Get the history of k-margin adjustments.
    pub fn adjustments(&self) -> &[KAdjusted] {
        &self.adjustments
    }

    /// Reset the estimator to its initial state.
    pub fn reset(&mut self) {
        self.p_hat = self.config.initial_p_hat;
        self.observation_count = 0;
        self.last_k = None;
        self.adjustments.clear();
    }

    /// Initialize last_k for change tracking. Call this before the first
    /// vote round with the initial k value being used.
    pub fn set_initial_k(&mut self, k: usize) {
        self.last_k = Some(k);
    }

    /// Estimate per-step success probability from a vote observation.
    ///
    /// Heuristic: if vote converged in exactly k samples (unanimous),
    /// the model's effective p is very high. As total_samples / k_used grows,
    /// the estimated p decreases. Red-flagged samples further reduce the estimate.
    fn estimate_p_from_observation(&self, obs: &VoteObservation) -> f64 {
        let valid_samples = obs.total_samples.saturating_sub(obs.red_flagged);
        if obs.total_samples == 0 || valid_samples == 0 {
            return 0.51; // Minimum valid p
        }

        // Ratio of k_used to total samples gives an efficiency metric.
        // Using total_samples (not valid) so red flags naturally reduce efficiency.
        // If we needed exactly k samples (all the same, no red flags), p is very high.
        // If we needed many more, p is lower.
        let efficiency = obs.k_used as f64 / obs.total_samples as f64;

        // Map efficiency to p estimate:
        // efficiency = 1.0 → all samples agreed → p ≈ 0.95
        // efficiency = 0.5 → needed 2x samples → p ≈ 0.75
        // efficiency < 0.3 → lots of disagreement → p ≈ 0.60
        let p_from_efficiency = 0.55 + 0.40 * efficiency.clamp(0.0, 1.0);

        // Additional penalty for red flags (on top of efficiency reduction)
        let red_flag_penalty = obs.red_flagged as f64 / obs.total_samples as f64 * 0.1;

        (p_from_efficiency - red_flag_penalty).clamp(0.51, 0.99)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================
    // Construction and Configuration Tests
    // ==========================================

    #[test]
    fn test_default_config() {
        let config = KEstimatorConfig::default();
        assert!((config.ema_alpha - 0.1).abs() < f64::EPSILON);
        assert!((config.initial_p_hat - 0.85).abs() < f64::EPSILON);
        assert_eq!(config.k_min_floor, 2);
        assert_eq!(config.k_max_ceiling, 10);
    }

    #[test]
    fn test_new_estimator_uses_initial_p_hat() {
        let config = KEstimatorConfig {
            initial_p_hat: 0.90,
            ..Default::default()
        };
        let estimator = KEstimator::new(config);
        assert!((estimator.p_hat() - 0.90).abs() < f64::EPSILON);
        assert_eq!(estimator.observation_count(), 0);
    }

    // ==========================================
    // EMA Convergence Tests
    // ==========================================

    #[test]
    fn test_p_hat_converges_to_high_p_with_quick_convergence() {
        let mut estimator = KEstimator::new(KEstimatorConfig::default());

        // Simulate 30 observations of quick convergence (unanimous votes)
        for _ in 0..30 {
            estimator.observe(VoteObservation {
                converged_quickly: true,
                total_samples: 3,
                k_used: 3,
                red_flagged: 0,
            });
        }

        // p_hat should be high (> 0.90) after many unanimous observations
        assert!(
            estimator.p_hat() > 0.90,
            "Expected p_hat > 0.90 after quick convergence, got {:.4}",
            estimator.p_hat()
        );
    }

    #[test]
    fn test_p_hat_decreases_with_slow_convergence() {
        let mut estimator = KEstimator::new(KEstimatorConfig {
            initial_p_hat: 0.90,
            ..Default::default()
        });

        // Simulate observations where votes take many samples
        for _ in 0..30 {
            estimator.observe(VoteObservation {
                converged_quickly: false,
                total_samples: 12,
                k_used: 3,
                red_flagged: 0,
            });
        }

        // p_hat should decrease below initial
        assert!(
            estimator.p_hat() < 0.90,
            "Expected p_hat < 0.90 after slow convergence, got {:.4}",
            estimator.p_hat()
        );
    }

    #[test]
    fn test_p_hat_converges_within_20_observations() {
        let mut estimator = KEstimator::new(KEstimatorConfig {
            ema_alpha: 0.1,
            initial_p_hat: 0.70, // Start far from true p
            ..Default::default()
        });

        // True p is ~0.95 (unanimous votes with k=3)
        for _ in 0..20 {
            estimator.observe(VoteObservation {
                converged_quickly: true,
                total_samples: 3,
                k_used: 3,
                red_flagged: 0,
            });
        }

        let p_sample = 0.55 + 0.40 * 1.0; // efficiency=1.0 → p_sample=0.95
                                          // After 20 EMA updates from 0.70 toward 0.95, should be within 5%
        let target = p_sample;
        let tolerance = target * 0.05;
        let diff = (estimator.p_hat() - target).abs();
        assert!(
            diff < tolerance,
            "Expected p_hat within 5% of {:.2} after 20 obs, got {:.4} (diff={:.4})",
            target,
            estimator.p_hat(),
            diff
        );
    }

    // ==========================================
    // Recommended K Tests
    // ==========================================

    #[test]
    fn test_recommended_k_respects_floor() {
        let estimator = KEstimator::new(KEstimatorConfig {
            initial_p_hat: 0.99, // Very high p → k_min would be 1
            k_min_floor: 2,
            ..Default::default()
        });

        let k = estimator.recommended_k(0.95, 10);
        assert!(k >= 2, "k={} should be >= floor of 2", k);
    }

    #[test]
    fn test_recommended_k_respects_ceiling() {
        let estimator = KEstimator::new(KEstimatorConfig {
            initial_p_hat: 0.51, // Very low p → k_min would be huge
            k_max_ceiling: 10,
            ..Default::default()
        });

        let k = estimator.recommended_k(0.95, 1000);
        assert!(k <= 10, "k={} should be <= ceiling of 10", k);
    }

    #[test]
    fn test_recommended_k_increases_when_p_drops() {
        let mut estimator = KEstimator::new(KEstimatorConfig {
            initial_p_hat: 0.90,
            ..Default::default()
        });

        let k_initial = estimator.recommended_k(0.95, 1000);

        // Simulate many observations of slow convergence (dropping p)
        for _ in 0..30 {
            estimator.observe(VoteObservation {
                converged_quickly: false,
                total_samples: 15,
                k_used: 3,
                red_flagged: 2,
            });
        }

        let k_after_drop = estimator.recommended_k(0.95, 1000);
        assert!(
            k_after_drop >= k_initial,
            "k should increase when p drops: {} -> {}",
            k_initial,
            k_after_drop
        );
    }

    #[test]
    fn test_recommended_k_decreases_when_p_rises() {
        let mut estimator = KEstimator::new(KEstimatorConfig {
            initial_p_hat: 0.60,
            ..Default::default()
        });

        let k_initial = estimator.recommended_k(0.95, 1000);

        // Simulate many observations of quick convergence (rising p)
        for _ in 0..30 {
            estimator.observe(VoteObservation {
                converged_quickly: true,
                total_samples: 3,
                k_used: 3,
                red_flagged: 0,
            });
        }

        let k_after_rise = estimator.recommended_k(0.95, 1000);
        assert!(
            k_after_rise <= k_initial,
            "k should decrease when p rises: {} -> {}",
            k_initial,
            k_after_rise
        );
    }

    #[test]
    fn test_k_never_violates_bounds() {
        let config = KEstimatorConfig {
            k_min_floor: 2,
            k_max_ceiling: 10,
            ..Default::default()
        };
        let mut estimator = KEstimator::new(config);

        // Test across a range of scenarios
        for t in [0.90, 0.95, 0.99] {
            for s in [1, 10, 100, 1000, 100000] {
                let k = estimator.recommended_k(t, s);
                assert!(k >= 2 && k <= 10, "k={} out of bounds [2, 10]", k);
            }
        }

        // Also test after many observations
        for _ in 0..50 {
            estimator.observe(VoteObservation {
                converged_quickly: true,
                total_samples: 3,
                k_used: 3,
                red_flagged: 0,
            });
            let k = estimator.recommended_k(0.95, 1000);
            assert!(k >= 2 && k <= 10, "k={} out of bounds after observation", k);
        }
    }

    // ==========================================
    // Observation Estimation Tests
    // ==========================================

    #[test]
    fn test_unanimous_vote_gives_high_p() {
        let estimator = KEstimator::new(KEstimatorConfig::default());
        let p = estimator.estimate_p_from_observation(&VoteObservation {
            converged_quickly: true,
            total_samples: 3,
            k_used: 3,
            red_flagged: 0,
        });
        assert!(p > 0.90, "Unanimous vote should give high p, got {:.4}", p);
    }

    #[test]
    fn test_contested_vote_gives_lower_p() {
        let estimator = KEstimator::new(KEstimatorConfig::default());
        let p = estimator.estimate_p_from_observation(&VoteObservation {
            converged_quickly: false,
            total_samples: 15,
            k_used: 3,
            red_flagged: 0,
        });
        assert!(p < 0.85, "Contested vote should give lower p, got {:.4}", p);
    }

    #[test]
    fn test_red_flags_reduce_p_estimate() {
        let estimator = KEstimator::new(KEstimatorConfig::default());

        let p_no_flags = estimator.estimate_p_from_observation(&VoteObservation {
            converged_quickly: false,
            total_samples: 10,
            k_used: 3,
            red_flagged: 0,
        });

        let p_with_flags = estimator.estimate_p_from_observation(&VoteObservation {
            converged_quickly: false,
            total_samples: 10,
            k_used: 3,
            red_flagged: 4,
        });

        assert!(
            p_with_flags < p_no_flags,
            "Red flags should reduce p: {:.4} vs {:.4}",
            p_with_flags,
            p_no_flags
        );
    }

    #[test]
    fn test_zero_valid_samples_gives_minimum_p() {
        let estimator = KEstimator::new(KEstimatorConfig::default());
        let p = estimator.estimate_p_from_observation(&VoteObservation {
            converged_quickly: false,
            total_samples: 5,
            k_used: 3,
            red_flagged: 5, // All red-flagged
        });
        assert!(
            (p - 0.51).abs() < f64::EPSILON,
            "All-red-flagged should give minimum p, got {:.4}",
            p
        );
    }

    // ==========================================
    // Reset and State Management Tests
    // ==========================================

    #[test]
    fn test_reset_restores_initial_state() {
        let mut estimator = KEstimator::new(KEstimatorConfig::default());

        // Make some observations
        for _ in 0..10 {
            estimator.observe(VoteObservation {
                converged_quickly: true,
                total_samples: 3,
                k_used: 3,
                red_flagged: 0,
            });
        }

        assert!(estimator.observation_count() > 0);

        estimator.reset();

        assert_eq!(estimator.observation_count(), 0);
        assert!((estimator.p_hat() - 0.85).abs() < f64::EPSILON);
        assert!(estimator.adjustments().is_empty());
    }

    #[test]
    fn test_set_initial_k() {
        let mut estimator = KEstimator::new(KEstimatorConfig::default());
        estimator.set_initial_k(4);

        // Now observe — should be able to detect changes from k=4
        estimator.observe(VoteObservation {
            converged_quickly: true,
            total_samples: 3,
            k_used: 3,
            red_flagged: 0,
        });
    }

    // ==========================================
    // KAdjusted Event Tests
    // ==========================================

    #[test]
    fn test_adjustment_emitted_on_k_change() {
        let mut estimator = KEstimator::new(KEstimatorConfig {
            initial_p_hat: 0.90,
            ema_alpha: 0.5, // High alpha for faster changes
            ..Default::default()
        });
        estimator.set_initial_k(estimator.recommended_k(0.95, 1000));

        // Drive p_hat down significantly with bad observations
        let mut got_adjustment = false;
        for _ in 0..20 {
            if let Some(_adj) = estimator.observe(VoteObservation {
                converged_quickly: false,
                total_samples: 20,
                k_used: 3,
                red_flagged: 5,
            }) {
                got_adjustment = true;
            }
        }

        assert!(
            got_adjustment || !estimator.adjustments().is_empty(),
            "Expected at least one KAdjusted event"
        );
    }

    // ==========================================
    // Edge Case Tests
    // ==========================================

    #[test]
    fn test_p_hat_stays_in_valid_range() {
        let mut estimator = KEstimator::new(KEstimatorConfig {
            ema_alpha: 0.9, // Very reactive
            initial_p_hat: 0.51,
            ..Default::default()
        });

        // Push p_hat to extremes
        for _ in 0..50 {
            estimator.observe(VoteObservation {
                converged_quickly: true,
                total_samples: 3,
                k_used: 3,
                red_flagged: 0,
            });
        }
        assert!(estimator.p_hat() <= 0.99, "p_hat should not exceed 0.99");
        assert!(
            estimator.p_hat() >= 0.51,
            "p_hat should not drop below 0.51"
        );

        // Now push down
        for _ in 0..50 {
            estimator.observe(VoteObservation {
                converged_quickly: false,
                total_samples: 50,
                k_used: 3,
                red_flagged: 40,
            });
        }
        assert!(
            estimator.p_hat() >= 0.51,
            "p_hat should not drop below 0.51"
        );
    }

    #[test]
    fn test_recommended_k_with_single_step_remaining() {
        let estimator = KEstimator::new(KEstimatorConfig::default());
        let k = estimator.recommended_k(0.95, 1);
        assert!(k >= 2, "k should respect floor even for 1 step");
    }

    #[test]
    fn test_recommended_k_with_zero_steps_remaining() {
        let estimator = KEstimator::new(KEstimatorConfig::default());
        // 0 steps should be treated as 1
        let k = estimator.recommended_k(0.95, 0);
        assert!(k >= 2, "k should respect floor even for 0 steps");
    }
}
