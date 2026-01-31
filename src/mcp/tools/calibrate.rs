//! maker/calibrate Tool Implementation
//!
//! Estimate per-step success rate (p) from calibration samples.

use crate::core::calculate_kmin;
use rmcp::schemars::{self, JsonSchema};
use serde::{Deserialize, Serialize};

/// A single calibration sample with prompt and expected answer
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CalibrationSample {
    /// The prompt to send to the LLM
    pub prompt: String,
    /// The expected correct answer (ground truth)
    pub ground_truth: String,
    /// The actual LLM response (for offline calibration)
    #[serde(default)]
    pub response: Option<String>,
}

/// Request for maker/calibrate tool
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CalibrateRequest {
    /// Calibration samples with prompts and ground truth
    pub samples: Vec<CalibrationSample>,
    /// Target task reliability (default 0.95)
    #[serde(default = "default_target_reliability")]
    pub target_reliability: f64,
    /// Target step count for k recommendation (default 1000)
    #[serde(default = "default_step_count")]
    pub target_steps: usize,
}

fn default_target_reliability() -> f64 {
    0.95
}

fn default_step_count() -> usize {
    1000
}

/// Response from maker/calibrate tool
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct CalibrateResponse {
    /// Estimated per-step success probability
    pub p_estimate: f64,
    /// 95% Wilson score confidence interval (lower, upper)
    pub confidence_interval: (f64, f64),
    /// Number of samples used
    pub sample_count: usize,
    /// Recommended k-margin for target reliability and steps
    pub recommended_k: usize,
}

/// Errors that can occur during calibration
#[derive(Debug, Clone, PartialEq)]
pub enum CalibrateError {
    /// No samples provided
    NoSamples,
    /// Samples without responses (need to run LLM first)
    MissingResponses,
    /// Estimated p is too low for voting (p <= 0.5)
    InsufficientAccuracy { p: f64 },
}

impl std::fmt::Display for CalibrateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CalibrateError::NoSamples => write!(f, "No calibration samples provided"),
            CalibrateError::MissingResponses => {
                write!(
                    f,
                    "Samples missing LLM responses - run samples through LLM first"
                )
            }
            CalibrateError::InsufficientAccuracy { p } => {
                write!(
                    f,
                    "Estimated accuracy p={:.2} is <= 0.5, voting requires p > 0.5",
                    p
                )
            }
        }
    }
}

impl std::error::Error for CalibrateError {}

/// Calculate Wilson score confidence interval for a proportion
///
/// Uses the Wilson score interval which has better coverage for small samples
/// and proportions near 0 or 1 compared to the normal approximation.
fn wilson_score_interval(successes: usize, total: usize, confidence: f64) -> (f64, f64) {
    if total == 0 {
        return (0.0, 1.0);
    }

    let n = total as f64;
    let p = successes as f64 / n;

    // z-score for confidence level (95% -> 1.96)
    let z = match confidence {
        c if (c - 0.99).abs() < 0.001 => 2.576,
        c if (c - 0.95).abs() < 0.001 => 1.96,
        c if (c - 0.90).abs() < 0.001 => 1.645,
        _ => 1.96, // Default to 95%
    };

    let z2 = z * z;

    // Wilson score formula
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half_width = (z / denominator) * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();

    let lower = (center - half_width).max(0.0);
    let upper = (center + half_width).min(1.0);

    (lower, upper)
}

/// Execute calibration on provided samples
///
/// Calculates p_estimate, confidence interval, and recommended k.
pub fn execute_calibrate(request: &CalibrateRequest) -> Result<CalibrateResponse, CalibrateError> {
    if request.samples.is_empty() {
        return Err(CalibrateError::NoSamples);
    }

    // Count correct responses (exact match for MVP)
    let mut correct = 0;
    let mut total = 0;

    for sample in &request.samples {
        if let Some(response) = &sample.response {
            total += 1;
            if response.trim() == sample.ground_truth.trim() {
                correct += 1;
            }
        }
    }

    if total == 0 {
        return Err(CalibrateError::MissingResponses);
    }

    let p_estimate = correct as f64 / total as f64;

    // Check if p is sufficient for voting
    if p_estimate <= 0.5 {
        return Err(CalibrateError::InsufficientAccuracy { p: p_estimate });
    }

    // Calculate 95% Wilson score confidence interval
    let confidence_interval = wilson_score_interval(correct, total, 0.95);

    // Calculate recommended k using k_min formula
    // Use lower bound of confidence interval for more conservative recommendation
    let p_conservative = confidence_interval.0.max(0.51); // Must be > 0.5
    let recommended_k = calculate_kmin(
        p_conservative,
        request.target_reliability,
        request.target_steps,
        1, // m=1 for microagents
    )
    .unwrap_or(10); // Fallback if calculation fails

    Ok(CalibrateResponse {
        p_estimate,
        confidence_interval,
        sample_count: total,
        recommended_k,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_sample_serialization() {
        let sample = CalibrationSample {
            prompt: "What is 2+2?".to_string(),
            ground_truth: "4".to_string(),
            response: Some("4".to_string()),
        };

        let json = serde_json::to_string(&sample).unwrap();
        let parsed: CalibrationSample = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.prompt, sample.prompt);
        assert_eq!(parsed.ground_truth, sample.ground_truth);
    }

    #[test]
    fn test_calibrate_request_serialization() {
        let request = CalibrateRequest {
            samples: vec![
                CalibrationSample {
                    prompt: "Q1".to_string(),
                    ground_truth: "A1".to_string(),
                    response: Some("A1".to_string()),
                },
                CalibrationSample {
                    prompt: "Q2".to_string(),
                    ground_truth: "A2".to_string(),
                    response: Some("A2".to_string()),
                },
            ],
            target_reliability: 0.95,
            target_steps: 1000,
        };

        let json = serde_json::to_string(&request).unwrap();
        let parsed: CalibrateRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.samples.len(), 2);
    }

    #[test]
    fn test_calibrate_request_deny_unknown_fields() {
        let json = r#"{"samples": [], "extra": true}"#;
        let result: Result<CalibrateRequest, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_calibrate_response_serialization() {
        let response = CalibrateResponse {
            p_estimate: 0.85,
            confidence_interval: (0.78, 0.90),
            sample_count: 50,
            recommended_k: 4,
        };

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("0.85"));
        assert!(json.contains("recommended_k"));
    }

    #[test]
    fn test_wilson_score_interval() {
        // 80 correct out of 100
        let (lower, upper) = wilson_score_interval(80, 100, 0.95);
        assert!(lower > 0.7);
        assert!(upper < 0.9);
        assert!(lower < 0.8);
        assert!(upper > 0.8);
    }

    #[test]
    fn test_wilson_score_interval_edge_cases() {
        // All correct
        let (lower, upper) = wilson_score_interval(10, 10, 0.95);
        assert!(lower > 0.7);
        assert!((upper - 1.0).abs() < 0.001);

        // None correct
        let (lower, upper) = wilson_score_interval(0, 10, 0.95);
        assert!((lower - 0.0).abs() < 0.001);
        assert!(upper < 0.3);

        // Empty
        let (lower, upper) = wilson_score_interval(0, 0, 0.95);
        assert!((lower - 0.0).abs() < 0.001);
        assert!((upper - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_execute_calibrate_no_samples() {
        let request = CalibrateRequest {
            samples: vec![],
            target_reliability: 0.95,
            target_steps: 1000,
        };

        let result = execute_calibrate(&request);
        assert_eq!(result, Err(CalibrateError::NoSamples));
    }

    #[test]
    fn test_execute_calibrate_missing_responses() {
        let request = CalibrateRequest {
            samples: vec![CalibrationSample {
                prompt: "Q1".to_string(),
                ground_truth: "A1".to_string(),
                response: None,
            }],
            target_reliability: 0.95,
            target_steps: 1000,
        };

        let result = execute_calibrate(&request);
        assert_eq!(result, Err(CalibrateError::MissingResponses));
    }

    #[test]
    fn test_execute_calibrate_high_accuracy() {
        // 9 correct out of 10
        let mut samples = Vec::new();
        for i in 0..9 {
            samples.push(CalibrationSample {
                prompt: format!("Q{}", i),
                ground_truth: "correct".to_string(),
                response: Some("correct".to_string()),
            });
        }
        samples.push(CalibrationSample {
            prompt: "Q9".to_string(),
            ground_truth: "correct".to_string(),
            response: Some("wrong".to_string()),
        });

        let request = CalibrateRequest {
            samples,
            target_reliability: 0.95,
            target_steps: 1000,
        };

        let result = execute_calibrate(&request).unwrap();
        assert!((result.p_estimate - 0.9).abs() < 0.001);
        assert_eq!(result.sample_count, 10);
        assert!(result.recommended_k >= 1);
    }

    #[test]
    fn test_execute_calibrate_low_accuracy() {
        // 4 correct out of 10 (p = 0.4 < 0.5)
        let mut samples = Vec::new();
        for i in 0..4 {
            samples.push(CalibrationSample {
                prompt: format!("Q{}", i),
                ground_truth: "correct".to_string(),
                response: Some("correct".to_string()),
            });
        }
        for i in 4..10 {
            samples.push(CalibrationSample {
                prompt: format!("Q{}", i),
                ground_truth: "correct".to_string(),
                response: Some("wrong".to_string()),
            });
        }

        let request = CalibrateRequest {
            samples,
            target_reliability: 0.95,
            target_steps: 1000,
        };

        let result = execute_calibrate(&request);
        assert!(matches!(
            result,
            Err(CalibrateError::InsufficientAccuracy { .. })
        ));
    }
}
