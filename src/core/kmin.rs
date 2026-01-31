//! k_min Calculation for MAKER Framework
//!
//! Calculates the minimum k-margin required for target reliability in the
//! first-to-ahead-by-k voting protocol.
//!
//! # Mathematical Background
//!
//! The MAKER framework uses Sequential Probability Ratio Test (SPRT) based
//! voting to achieve error correction. The minimum margin k_min is calculated as:
//!
//! ```text
//! k_min = ⌈ln(t^(-m/s) - 1) / ln((1-p)/p)⌉
//! ```
//!
//! Where:
//! - `p`: Per-step success probability (p ∈ (0.5, 1.0))
//! - `t`: Target task reliability (t ∈ (0, 1))
//! - `s`: Total number of steps in the task
//! - `m`: Steps per agent (always 1 for microagents)
//!
//! # Key Insight
//!
//! k_min grows logarithmically with s (Θ(ln s)), making million-step tasks
//! economically feasible. For example, 20-disk Towers of Hanoi (1,048,575 steps)
//! requires only k_min=3-4 with p=0.85 and t=0.95.

/// Error type for k_min calculation
#[derive(Debug, Clone, PartialEq)]
pub enum KminError {
    /// p must be in (0.5, 1.0) - voting requires better than random chance
    InvalidSuccessProbability { p: f64 },
    /// t must be in (0, 1) - target reliability as a probability
    InvalidTargetReliability { t: f64 },
    /// s must be > 0 - task must have at least one step
    InvalidStepCount { s: usize },
    /// m must equal 1 for microagent architecture
    InvalidStepsPerAgent { m: usize },
}

impl std::fmt::Display for KminError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KminError::InvalidSuccessProbability { p } => {
                write!(f, "Success probability p={} must be in (0.5, 1.0)", p)
            }
            KminError::InvalidTargetReliability { t } => {
                write!(f, "Target reliability t={} must be in (0, 1)", t)
            }
            KminError::InvalidStepCount { s } => {
                write!(f, "Step count s={} must be > 0", s)
            }
            KminError::InvalidStepsPerAgent { m } => {
                write!(
                    f,
                    "Steps per agent m={} must be 1 (microagent constraint)",
                    m
                )
            }
        }
    }
}

impl std::error::Error for KminError {}

/// Calculates the minimum k-margin required for target reliability.
///
/// # Arguments
///
/// * `p` - Per-step success probability, must be in (0.5, 1.0)
/// * `t` - Target task reliability, must be in (0, 1)
/// * `s` - Total number of steps in the task, must be > 0
/// * `m` - Steps per agent, must be 1 (microagent constraint)
///
/// # Returns
///
/// The minimum k-margin (k_min) as a positive integer, or an error if
/// inputs are invalid.
///
/// # Formula
///
/// ```text
/// k_min = ⌈ln(t^(-m/s) - 1) / ln((1-p)/p)⌉
/// ```
///
/// # Examples
///
/// ```
/// use maker::core::kmin::calculate_kmin;
///
/// // 20-disk Towers of Hanoi: 1,048,575 steps with high-accuracy model (p=0.99)
/// let k = calculate_kmin(0.99, 0.95, 1_048_575, 1).unwrap();
/// assert!(k >= 3 && k <= 4);
///
/// // With moderate accuracy model (p=0.85), more margin is needed
/// let k_moderate = calculate_kmin(0.85, 0.95, 1_048_575, 1).unwrap();
/// assert!(k_moderate >= 8 && k_moderate <= 12);
/// ```
pub fn calculate_kmin(p: f64, t: f64, s: usize, m: usize) -> Result<usize, KminError> {
    // Input validation
    if p <= 0.5 || p >= 1.0 {
        return Err(KminError::InvalidSuccessProbability { p });
    }
    if t <= 0.0 || t >= 1.0 {
        return Err(KminError::InvalidTargetReliability { t });
    }
    if s == 0 {
        return Err(KminError::InvalidStepCount { s });
    }
    if m != 1 {
        return Err(KminError::InvalidStepsPerAgent { m });
    }

    // Derivation from Gambler's Ruin / SPRT:
    //
    // In first-to-ahead-by-k voting with per-vote success probability p > 0.5,
    // the probability that the correct answer wins is (Gambler's Ruin):
    //
    //   P(correct wins) = (1 - r^k) / (1 - r^(2k))  where r = (1-p)/p < 1
    //
    // For r << 1 (i.e., p close to 1), this simplifies to approximately:
    //   P(correct wins) ≈ 1 - r^k
    //
    // To achieve overall task reliability t over s steps, each step needs
    // per-step reliability at least t^(1/s). So we need:
    //
    //   1 - r^k >= t^(1/s)
    //   r^k <= 1 - t^(1/s)
    //   k * ln(r) <= ln(1 - t^(1/s))
    //
    // Since r < 1 means ln(r) < 0, dividing flips the inequality:
    //   k >= ln(1 - t^(1/s)) / ln(r)
    //
    // Therefore: k_min = ceil(ln(1 - t^(1/s)) / ln((1-p)/p))
    //
    // Note: This formula is equivalent to the paper's formula when rearranged.

    let r = (1.0 - p) / p; // r < 1 for p > 0.5
    let per_step_target = t.powf(1.0 / (s as f64)); // t^(1/s)
    let one_minus_target = 1.0 - per_step_target; // 1 - t^(1/s)

    // Handle edge case: if one_minus_target is extremely small (near 0),
    // ln will be very negative, but the formula still works
    let numerator = one_minus_target.ln();
    let denominator = r.ln();

    // Both numerator and denominator are negative (since both arguments < 1)
    // so their ratio is positive
    let k_raw = numerator / denominator;
    let k_min = k_raw.ceil() as usize;

    // k_min must be at least 1
    Ok(k_min.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================
    // Input Validation Tests
    // ==========================================

    #[test]
    fn test_rejects_p_less_than_or_equal_to_half() {
        // p = 0.5 is random chance - voting can't help
        assert_eq!(
            calculate_kmin(0.5, 0.95, 100, 1),
            Err(KminError::InvalidSuccessProbability { p: 0.5 })
        );

        // p < 0.5 means model is worse than random
        assert_eq!(
            calculate_kmin(0.3, 0.95, 100, 1),
            Err(KminError::InvalidSuccessProbability { p: 0.3 })
        );
    }

    #[test]
    fn test_rejects_p_greater_than_or_equal_to_one() {
        // p = 1.0 is perfect - voting unnecessary
        assert_eq!(
            calculate_kmin(1.0, 0.95, 100, 1),
            Err(KminError::InvalidSuccessProbability { p: 1.0 })
        );

        // p > 1.0 is impossible
        assert_eq!(
            calculate_kmin(1.1, 0.95, 100, 1),
            Err(KminError::InvalidSuccessProbability { p: 1.1 })
        );
    }

    #[test]
    fn test_rejects_invalid_target_reliability() {
        // t <= 0 is invalid
        assert_eq!(
            calculate_kmin(0.85, 0.0, 100, 1),
            Err(KminError::InvalidTargetReliability { t: 0.0 })
        );

        assert_eq!(
            calculate_kmin(0.85, -0.1, 100, 1),
            Err(KminError::InvalidTargetReliability { t: -0.1 })
        );

        // t >= 1 is invalid (100% reliability is impossible)
        assert_eq!(
            calculate_kmin(0.85, 1.0, 100, 1),
            Err(KminError::InvalidTargetReliability { t: 1.0 })
        );
    }

    #[test]
    fn test_rejects_zero_steps() {
        assert_eq!(
            calculate_kmin(0.85, 0.95, 0, 1),
            Err(KminError::InvalidStepCount { s: 0 })
        );
    }

    #[test]
    fn test_rejects_m_not_equal_to_one() {
        // MAKER requires m=1 (microagent constraint)
        assert_eq!(
            calculate_kmin(0.85, 0.95, 100, 2),
            Err(KminError::InvalidStepsPerAgent { m: 2 })
        );

        assert_eq!(
            calculate_kmin(0.85, 0.95, 100, 0),
            Err(KminError::InvalidStepsPerAgent { m: 0 })
        );
    }

    // ==========================================
    // Correctness Tests (from paper)
    // ==========================================

    #[test]
    fn test_20_disk_hanoi_kmin_high_accuracy_model() {
        // From the paper: 20-disk Hanoi (1,048,575 steps) with gpt-4.1-mini
        // achieves k_min = 3-4 with t=0.95. This requires p ≈ 0.99.
        let k = calculate_kmin(0.99, 0.95, 1_048_575, 1).unwrap();
        assert!(
            (3..=4).contains(&k),
            "Expected k_min 3-4 for p=0.99, got {}",
            k
        );
    }

    #[test]
    fn test_20_disk_hanoi_kmin_moderate_accuracy() {
        // With p=0.85 (lower accuracy model), we need more margin
        let k = calculate_kmin(0.85, 0.95, 1_048_575, 1).unwrap();
        // Formula gives k ≈ 10 for p=0.85
        assert!(
            (8..=12).contains(&k),
            "Expected k_min ~10 for p=0.85, got {}",
            k
        );
    }

    #[test]
    fn test_3_disk_hanoi_kmin() {
        // 3-disk Hanoi: 7 steps (2^3 - 1)
        // With p=0.85, t=0.95, per-step target is high, k should be small
        let k = calculate_kmin(0.85, 0.95, 7, 1).unwrap();
        assert!(k >= 1, "k_min must be at least 1");
        // Small task with 7 steps needs lower per-step reliability
        assert!(k <= 3, "Expected small k for 7 steps, got {}", k);
    }

    #[test]
    fn test_10_disk_hanoi_kmin() {
        // 10-disk Hanoi: 1,023 steps (2^10 - 1)
        let k = calculate_kmin(0.85, 0.95, 1_023, 1).unwrap();
        // With p=0.85 and 1023 steps, formula gives k ≈ 6
        assert!(
            (5..=7).contains(&k),
            "Expected k_min 5-7 for 1023 steps with p=0.85, got {}",
            k
        );
    }

    #[test]
    fn test_returns_at_least_one() {
        // k_min should always be at least 1
        let k = calculate_kmin(0.99, 0.5, 1, 1).unwrap();
        assert!(k >= 1, "k_min must be at least 1");
    }

    // ==========================================
    // Edge Case Tests
    // ==========================================

    #[test]
    fn test_p_approaching_one_gives_small_k() {
        // As p → 1, k_min decreases (near-perfect model needs less voting margin)
        let k_high_p = calculate_kmin(0.95, 0.95, 1000, 1).unwrap();
        let k_low_p = calculate_kmin(0.6, 0.95, 1000, 1).unwrap();

        assert!(
            k_high_p < k_low_p,
            "Higher p should require smaller k: k(p=0.95)={} vs k(p=0.6)={}",
            k_high_p,
            k_low_p
        );
    }

    #[test]
    fn test_higher_reliability_requires_larger_k() {
        // As t → 1, k_min increases (stricter reliability needs more margin)
        let k_low_t = calculate_kmin(0.85, 0.90, 1000, 1).unwrap();
        let k_high_t = calculate_kmin(0.85, 0.99, 1000, 1).unwrap();

        assert!(
            k_high_t >= k_low_t,
            "Higher t should require larger k: k(t=0.99)={} vs k(t=0.90)={}",
            k_high_t,
            k_low_t
        );
    }

    #[test]
    fn test_single_step_task() {
        // s=1 is a valid edge case
        let k = calculate_kmin(0.85, 0.95, 1, 1).unwrap();
        assert!(k >= 1, "k_min must be at least 1 even for single step");
    }

    // ==========================================
    // Error Display Tests
    // ==========================================

    #[test]
    fn test_error_display_all_variants() {
        assert!(KminError::InvalidSuccessProbability { p: 0.3 }
            .to_string()
            .contains("0.3"));
        assert!(KminError::InvalidTargetReliability { t: 1.5 }
            .to_string()
            .contains("1.5"));
        assert!(KminError::InvalidStepCount { s: 0 }
            .to_string()
            .contains("0"));
        assert!(KminError::InvalidStepsPerAgent { m: 5 }
            .to_string()
            .contains("5"));
    }

    #[test]
    fn test_error_is_error_trait() {
        let err: Box<dyn std::error::Error> =
            Box::new(KminError::InvalidSuccessProbability { p: 0.3 });
        assert!(err.to_string().contains("0.3"));
    }

    // ==========================================
    // Additional Correctness Tests
    // ==========================================

    #[test]
    fn test_large_step_count() {
        // 1 million steps
        let k = calculate_kmin(0.85, 0.95, 1_000_000, 1).unwrap();
        assert!(k >= 8, "Large tasks need bigger k: got {}", k);
        assert!(k <= 15, "k should be reasonable for p=0.85: got {}", k);
    }

    #[test]
    fn test_very_high_reliability() {
        let k = calculate_kmin(0.85, 0.999, 100, 1).unwrap();
        let k_lower = calculate_kmin(0.85, 0.9, 100, 1).unwrap();
        assert!(k > k_lower, "t=0.999 should need more k than t=0.9");
    }

    #[test]
    fn test_barely_above_half() {
        // p=0.51, barely above chance - needs very large k
        let k = calculate_kmin(0.51, 0.95, 10, 1).unwrap();
        assert!(k > 10, "p=0.51 should require large k: got {}", k);
    }

    #[test]
    fn test_very_close_to_one() {
        // p=0.999, near perfect
        let k = calculate_kmin(0.999, 0.95, 1_000_000, 1).unwrap();
        assert!(k <= 3, "p=0.999 should need tiny k: got {}", k);
    }

    #[test]
    fn test_kmin_monotonic_with_steps() {
        // k should increase (or stay same) as step count increases
        let k_10 = calculate_kmin(0.85, 0.95, 10, 1).unwrap();
        let k_100 = calculate_kmin(0.85, 0.95, 100, 1).unwrap();
        let k_10000 = calculate_kmin(0.85, 0.95, 10000, 1).unwrap();

        assert!(k_100 >= k_10, "More steps should need >= k");
        assert!(k_10000 >= k_100, "More steps should need >= k");
    }
}
