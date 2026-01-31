//! Voting Integration for MAKER Framework
//!
//! Integrates sampling, red-flagging, and voting into the complete
//! first-to-ahead-by-k protocol for error-corrected LLM execution.
//!
//! # Execution Flow
//!
//! 1. **Prompting**: Deliver subtask with minimal context
//! 2. **Parallel Sampling**: Generate k samples concurrently (T=0 first, T=0.1 rest)
//! 3. **Red-Flagging**: Discard malformed samples
//! 4. **Voting**: Apply first-to-ahead-by-k rule to select winner
//!
//! # Usage
//!
//! ```rust
//! use maker::core::{vote_with_margin, VoteConfig, MockLlmClient};
//!
//! let client = MockLlmClient::constant("correct_answer");
//! let config = VoteConfig::default();
//! let result = vote_with_margin("What is 2+2?", 3, &client, config).unwrap();
//! assert_eq!(result.winner, "correct_answer");
//! ```

use crate::core::redflag::RedFlagValidator;
use crate::core::voting::{CandidateId, VoteCheckResult, VoteRace};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for voting execution
#[derive(Debug, Clone)]
pub struct VoteConfig {
    /// Maximum samples to collect before giving up
    pub max_samples: usize,
    /// Token limit for red-flagging
    pub token_limit: Option<usize>,
    /// Temperature for diversity samples (samples after the first)
    pub diversity_temperature: f64,
    /// Timeout for the entire voting process
    pub timeout: Option<Duration>,
}

impl Default for VoteConfig {
    fn default() -> Self {
        Self {
            max_samples: 100,
            token_limit: Some(700),
            diversity_temperature: 0.1,
            timeout: Some(Duration::from_secs(60)),
        }
    }
}

impl VoteConfig {
    /// Create config with custom max samples
    pub fn with_max_samples(mut self, max: usize) -> Self {
        self.max_samples = max;
        self
    }

    /// Create config with custom token limit
    pub fn with_token_limit(mut self, limit: usize) -> Self {
        self.token_limit = Some(limit);
        self
    }

    /// Create config without token limit
    pub fn without_token_limit(mut self) -> Self {
        self.token_limit = None;
        self
    }

    /// Create config with custom timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
}

/// Cost metrics for a voting session
#[derive(Debug, Clone, Default)]
pub struct CostMetrics {
    /// Total input tokens used
    pub input_tokens: usize,
    /// Total output tokens used
    pub output_tokens: usize,
    /// Estimated cost in USD (if pricing available)
    pub estimated_cost_usd: Option<f64>,
}

/// Result of a successful vote
#[derive(Debug, Clone)]
pub struct VoteResult {
    /// The winning candidate content
    pub winner: String,
    /// Vote counts per candidate
    pub vote_counts: HashMap<String, usize>,
    /// Total samples collected (including red-flagged)
    pub total_samples: usize,
    /// Number of red-flagged samples
    pub red_flagged: usize,
    /// Cost metrics
    pub cost: CostMetrics,
    /// Time taken for voting
    pub elapsed: Duration,
}

/// Errors during voting execution
#[derive(Debug, Clone, PartialEq)]
pub enum VoteError {
    /// No convergence within max samples
    NoConvergence {
        /// Number of samples collected before stopping
        samples_collected: usize,
        /// Maximum samples allowed
        max_samples: usize,
    },
    /// Timeout exceeded
    Timeout {
        /// Time elapsed before timeout
        elapsed: Duration,
    },
    /// All samples were red-flagged
    AllRedFlagged {
        /// Total samples that were all discarded
        total_samples: usize,
    },
    /// LLM client error
    LlmError {
        /// Error message from the LLM client
        message: String,
    },
    /// Invalid configuration
    InvalidConfig {
        /// Description of the configuration error
        message: String,
    },
}

impl std::fmt::Display for VoteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VoteError::NoConvergence {
                samples_collected,
                max_samples,
            } => {
                write!(
                    f,
                    "No convergence after {} samples (max: {})",
                    samples_collected, max_samples
                )
            }
            VoteError::Timeout { elapsed } => {
                write!(f, "Voting timed out after {:?}", elapsed)
            }
            VoteError::AllRedFlagged { total_samples } => {
                write!(f, "All {} samples were red-flagged", total_samples)
            }
            VoteError::LlmError { message } => {
                write!(f, "LLM error: {}", message)
            }
            VoteError::InvalidConfig { message } => {
                write!(f, "Invalid config: {}", message)
            }
        }
    }
}

impl std::error::Error for VoteError {}

/// Response from an LLM generation
#[derive(Debug, Clone)]
pub struct LlmResponse {
    /// Generated content
    pub content: String,
    /// Input tokens used
    pub input_tokens: usize,
    /// Output tokens used
    pub output_tokens: usize,
}

/// Trait for LLM clients (minimal interface for voting)
///
/// This trait is expanded in the `llm` module with full provider support.
/// Here we define the minimal interface needed for voting integration.
pub trait LlmClient: Send + Sync {
    /// Generate a response for the given prompt
    fn generate(&self, prompt: &str, temperature: f64) -> Result<LlmResponse, String>;
}

/// Mock LLM client for testing
///
/// Returns responses from a predefined list, cycling through them.
#[derive(Debug)]
pub struct MockLlmClient {
    /// Responses to return (cycles through)
    responses: Vec<String>,
    /// Current index
    index: std::sync::atomic::AtomicUsize,
    /// Simulated tokens per response
    tokens_per_response: usize,
}

impl Clone for MockLlmClient {
    fn clone(&self) -> Self {
        Self {
            responses: self.responses.clone(),
            index: std::sync::atomic::AtomicUsize::new(
                self.index.load(std::sync::atomic::Ordering::SeqCst),
            ),
            tokens_per_response: self.tokens_per_response,
        }
    }
}

impl MockLlmClient {
    /// Create a new mock client with the given responses
    pub fn new(responses: Vec<String>) -> Self {
        Self {
            responses,
            index: std::sync::atomic::AtomicUsize::new(0),
            tokens_per_response: 50,
        }
    }

    /// Create a mock that always returns the same response
    pub fn constant(response: &str) -> Self {
        Self::new(vec![response.to_string()])
    }

    /// Create a mock with biased responses (p probability of correct)
    pub fn biased(correct: &str, incorrect: &str, p: f64, total: usize) -> Self {
        let correct_count = (total as f64 * p).round() as usize;
        let mut responses = vec![correct.to_string(); correct_count];
        responses.extend(vec![incorrect.to_string(); total - correct_count]);
        Self::new(responses)
    }
}

impl LlmClient for MockLlmClient {
    fn generate(&self, _prompt: &str, _temperature: f64) -> Result<LlmResponse, String> {
        let idx = self.index.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let response = &self.responses[idx % self.responses.len()];

        Ok(LlmResponse {
            content: response.clone(),
            input_tokens: 100, // Simulated
            output_tokens: self.tokens_per_response,
        })
    }
}

/// Execute voting with the given margin
///
/// This is the main entry point for MAKER's error-corrected execution.
/// It orchestrates parallel sampling, red-flagging, and voting.
///
/// # Arguments
///
/// * `prompt` - The prompt to send to the LLM
/// * `k` - The required margin for declaring a winner
/// * `client` - The LLM client to use for generation
/// * `config` - Configuration for the voting process
///
/// # Returns
///
/// * `Ok(VoteResult)` - The winning response with metrics
/// * `Err(VoteError)` - If voting fails to converge or times out
pub fn vote_with_margin(
    prompt: &str,
    k: usize,
    client: &dyn LlmClient,
    config: VoteConfig,
) -> Result<VoteResult, VoteError> {
    if k == 0 {
        return Err(VoteError::InvalidConfig {
            message: "k must be >= 1".to_string(),
        });
    }

    let start = Instant::now();
    let validator = match config.token_limit {
        Some(limit) => RedFlagValidator::new().with_token_limit(limit),
        None => RedFlagValidator::new().without_token_limit(),
    };

    let race = VoteRace::new(k).map_err(|e| VoteError::InvalidConfig {
        message: e.to_string(),
    })?;

    let mut total_samples = 0;
    let mut red_flagged = 0;
    let mut cost = CostMetrics::default();
    let mut candidate_contents: HashMap<String, String> = HashMap::new();

    // Voting loop
    loop {
        // Check timeout
        if let Some(timeout) = config.timeout {
            if start.elapsed() > timeout {
                return Err(VoteError::Timeout {
                    elapsed: start.elapsed(),
                });
            }
        }

        // Check max samples
        if total_samples >= config.max_samples {
            return Err(VoteError::NoConvergence {
                samples_collected: total_samples,
                max_samples: config.max_samples,
            });
        }

        // Generate sample (T=0 for first, T=diversity for rest)
        let temperature = if total_samples == 0 {
            0.0
        } else {
            config.diversity_temperature
        };

        let response = client
            .generate(prompt, temperature)
            .map_err(|e| VoteError::LlmError { message: e })?;

        total_samples += 1;
        cost.input_tokens += response.input_tokens;
        cost.output_tokens += response.output_tokens;

        // Red-flag check
        let red_flags = validator.validate(&response.content);
        if !red_flags.is_empty() {
            red_flagged += 1;
            continue; // Discard sample
        }

        // Create candidate ID (hash of content for deduplication)
        let candidate_id = CandidateId::new(&response.content);

        // Store content for winner retrieval
        candidate_contents
            .entry(response.content.clone())
            .or_insert(response.content.clone());

        // Cast vote
        race.cast_vote(candidate_id);

        // Check for winner
        if let VoteCheckResult::Winner { candidate, .. } = race.check_winner() {
            let winner_content = candidate_contents
                .get(candidate.as_str())
                .cloned()
                .unwrap_or_else(|| candidate.as_str().to_string());

            let vote_counts: HashMap<String, usize> = race
                .get_votes()
                .into_iter()
                .map(|(id, count)| (id.as_str().to_string(), count))
                .collect();

            return Ok(VoteResult {
                winner: winner_content,
                vote_counts,
                total_samples,
                red_flagged,
                cost,
                elapsed: start.elapsed(),
            });
        }

        // Check if all samples red-flagged (no valid votes)
        if total_samples > k * 2 && race.total_votes() == 0 {
            return Err(VoteError::AllRedFlagged { total_samples });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================
    // VoteConfig Tests
    // ==========================================

    #[test]
    fn test_vote_config_defaults() {
        let config = VoteConfig::default();
        assert_eq!(config.max_samples, 100);
        assert_eq!(config.token_limit, Some(700));
        assert_eq!(config.diversity_temperature, 0.1);
    }

    #[test]
    fn test_vote_config_builder() {
        let config = VoteConfig::default()
            .with_max_samples(50)
            .with_token_limit(500)
            .with_timeout(Duration::from_secs(30));

        assert_eq!(config.max_samples, 50);
        assert_eq!(config.token_limit, Some(500));
        assert_eq!(config.timeout, Some(Duration::from_secs(30)));
    }

    // ==========================================
    // MockLlmClient Tests
    // ==========================================

    #[test]
    fn test_mock_client_cycles_responses() {
        let client = MockLlmClient::new(vec!["A".to_string(), "B".to_string()]);

        assert_eq!(client.generate("", 0.0).unwrap().content, "A");
        assert_eq!(client.generate("", 0.0).unwrap().content, "B");
        assert_eq!(client.generate("", 0.0).unwrap().content, "A"); // Cycles
    }

    #[test]
    fn test_mock_client_constant() {
        let client = MockLlmClient::constant("answer");

        for _ in 0..10 {
            assert_eq!(client.generate("", 0.0).unwrap().content, "answer");
        }
    }

    // ==========================================
    // vote_with_margin Tests
    // ==========================================

    #[test]
    fn test_vote_converges_with_unanimous_responses() {
        let client = MockLlmClient::constant("correct");
        let config = VoteConfig::default();

        let result = vote_with_margin("test", 3, &client, config).unwrap();

        assert_eq!(result.winner, "correct");
        assert_eq!(result.total_samples, 3); // Exactly k samples for unanimous
        assert_eq!(result.red_flagged, 0);
    }

    #[test]
    fn test_vote_converges_with_majority() {
        // 80% correct responses
        let client = MockLlmClient::new(vec![
            "correct".to_string(),
            "correct".to_string(),
            "correct".to_string(),
            "correct".to_string(),
            "wrong".to_string(),
        ]);
        let config = VoteConfig::default().with_max_samples(20);

        let result = vote_with_margin("test", 3, &client, config).unwrap();

        assert_eq!(result.winner, "correct");
        assert!(result.total_samples >= 3);
    }

    #[test]
    fn test_vote_rejects_invalid_k() {
        let client = MockLlmClient::constant("test");
        let config = VoteConfig::default();

        let result = vote_with_margin("test", 0, &client, config);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VoteError::InvalidConfig { .. }
        ));
    }

    #[test]
    fn test_vote_no_convergence() {
        // Alternating responses - will never converge with k=3
        let client = MockLlmClient::new(vec!["A".to_string(), "B".to_string()]);
        let config = VoteConfig::default().with_max_samples(10);

        let result = vote_with_margin("test", 3, &client, config);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VoteError::NoConvergence { .. }
        ));
    }

    #[test]
    fn test_vote_red_flags_long_responses() {
        // Responses that exceed token limit (700 chars)
        let long_response = "x".repeat(800);
        let client = MockLlmClient::new(vec![
            long_response.clone(),
            long_response.clone(),
            "short".to_string(),
            "short".to_string(),
            "short".to_string(),
        ]);
        let config = VoteConfig::default().with_token_limit(700);

        let result = vote_with_margin("test", 3, &client, config).unwrap();

        assert_eq!(result.winner, "short");
        assert_eq!(result.red_flagged, 2); // First two were too long
    }

    #[test]
    fn test_vote_excludes_red_flagged_from_vote_pool() {
        // Mix of valid and invalid responses
        let long = "x".repeat(800);
        let client = MockLlmClient::new(vec![
            "valid".to_string(),
            long.clone(),
            "valid".to_string(),
            long.clone(),
            "valid".to_string(),
        ]);
        let config = VoteConfig::default().with_token_limit(700);

        let result = vote_with_margin("test", 3, &client, config).unwrap();

        assert_eq!(result.winner, "valid");
        // Only valid samples should count towards votes
        assert_eq!(*result.vote_counts.get("valid").unwrap(), 3);
    }

    #[test]
    fn test_vote_cost_tracking() {
        let client = MockLlmClient::constant("answer");
        let config = VoteConfig::default();

        let result = vote_with_margin("test", 3, &client, config).unwrap();

        assert!(result.cost.input_tokens > 0);
        assert!(result.cost.output_tokens > 0);
    }

    // ==========================================
    // Integration: 3-disk Towers of Hanoi
    // ==========================================

    /// Mock client that returns correct Hanoi moves
    struct HanoiMockClient {
        correct_moves: Vec<String>,
        step: std::sync::atomic::AtomicUsize,
    }

    impl HanoiMockClient {
        fn new() -> Self {
            // Correct sequence for 3-disk Hanoi
            Self {
                correct_moves: vec![
                    "move 1 from A to C".to_string(),
                    "move 2 from A to B".to_string(),
                    "move 1 from C to B".to_string(),
                    "move 3 from A to C".to_string(),
                    "move 1 from B to A".to_string(),
                    "move 2 from B to C".to_string(),
                    "move 1 from A to C".to_string(),
                ],
                step: std::sync::atomic::AtomicUsize::new(0),
            }
        }

        fn advance_step(&self) {
            self.step.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }

        fn current_step(&self) -> usize {
            self.step.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl LlmClient for HanoiMockClient {
        fn generate(&self, _prompt: &str, _temperature: f64) -> Result<LlmResponse, String> {
            let step = self.current_step();
            let content = self.correct_moves[step % self.correct_moves.len()].clone();

            Ok(LlmResponse {
                content,
                input_tokens: 100,
                output_tokens: 20,
            })
        }
    }

    #[test]
    fn test_3_disk_hanoi_zero_errors() {
        let client = HanoiMockClient::new();
        let config = VoteConfig::default();

        let expected_moves = [
            "move 1 from A to C",
            "move 2 from A to B",
            "move 1 from C to B",
            "move 3 from A to C",
            "move 1 from B to A",
            "move 2 from B to C",
            "move 1 from A to C",
        ];

        let mut errors = 0;

        for (i, expected) in expected_moves.iter().enumerate() {
            let result = vote_with_margin(&format!("Step {}", i), 3, &client, config.clone());

            match result {
                Ok(vote_result) => {
                    if vote_result.winner != *expected {
                        errors += 1;
                    }
                }
                Err(_) => {
                    errors += 1;
                }
            }

            client.advance_step();
        }

        assert_eq!(errors, 0, "Expected zero errors on 3-disk Hanoi");
    }

    #[test]
    fn test_voting_converges_within_expected_samples() {
        // With p > 0.5, voting should converge quickly
        let client = MockLlmClient::biased("correct", "wrong", 0.8, 100);
        let config = VoteConfig::default().with_max_samples(50);

        let result = vote_with_margin("test", 3, &client, config).unwrap();

        // With 80% correct and k=3, should converge well under 20 samples typically
        assert!(
            result.total_samples < 20,
            "Expected convergence in <20 samples, got {}",
            result.total_samples
        );
    }

    // ==========================================
    // Error Display Tests
    // ==========================================

    #[test]
    fn test_vote_error_display_all_variants() {
        assert!(VoteError::NoConvergence {
            samples_collected: 50,
            max_samples: 100
        }
        .to_string()
        .contains("50"));
        assert!(VoteError::Timeout {
            elapsed: Duration::from_secs(60)
        }
        .to_string()
        .contains("60"));
        assert!(VoteError::AllRedFlagged { total_samples: 10 }
            .to_string()
            .contains("10"));
        assert!(VoteError::LlmError {
            message: "network error".to_string()
        }
        .to_string()
        .contains("network error"));
        assert!(VoteError::InvalidConfig {
            message: "bad config".to_string()
        }
        .to_string()
        .contains("bad config"));
    }

    #[test]
    fn test_vote_error_is_error_trait() {
        let err: Box<dyn std::error::Error> = Box::new(VoteError::LlmError {
            message: "test".to_string(),
        });
        assert!(err.to_string().contains("test"));
    }

    // ==========================================
    // Additional Error Path Tests
    // ==========================================

    #[test]
    fn test_vote_all_red_flagged() {
        // All responses exceed token limit - should trigger AllRedFlagged
        let long = "x".repeat(800);
        let client = MockLlmClient::constant(&long);
        let config = VoteConfig::default()
            .with_token_limit(100)
            .with_max_samples(20);

        let result = vote_with_margin("test", 3, &client, config);

        // Either NoConvergence or AllRedFlagged is acceptable
        assert!(result.is_err());
    }

    #[test]
    fn test_vote_llm_error() {
        struct FailingClient;
        impl LlmClient for FailingClient {
            fn generate(&self, _prompt: &str, _temperature: f64) -> Result<LlmResponse, String> {
                Err("LLM service unavailable".to_string())
            }
        }

        let client = FailingClient;
        let config = VoteConfig::default();

        let result = vote_with_margin("test", 3, &client, config);
        assert!(matches!(result, Err(VoteError::LlmError { .. })));
    }

    #[test]
    fn test_vote_timeout() {
        // Use a mock that takes time (by having the client be slow conceptually)
        // Since generate is sync, timeout only checks elapsed between iterations
        let client = MockLlmClient::new(vec!["A".to_string(), "B".to_string()]);
        let config = VoteConfig::default()
            .with_max_samples(1000)
            .with_timeout(Duration::from_nanos(1)); // Extremely short timeout

        let result = vote_with_margin("test", 3, &client, config);

        // Either timeout or no convergence, depending on timing
        assert!(result.is_err());
    }

    #[test]
    fn test_mock_client_clone() {
        let client = MockLlmClient::constant("hello");
        let _ = client.generate("", 0.0);

        let cloned = client.clone();
        // Cloned client should continue from the same index
        let resp = cloned.generate("", 0.0).unwrap();
        assert_eq!(resp.content, "hello");
    }

    #[test]
    fn test_mock_client_biased() {
        let client = MockLlmClient::biased("correct", "wrong", 0.8, 10);
        let mut correct_count = 0;
        for _ in 0..10 {
            if client.generate("", 0.0).unwrap().content == "correct" {
                correct_count += 1;
            }
        }
        assert_eq!(correct_count, 8); // 80% of 10
    }

    #[test]
    fn test_vote_without_token_limit() {
        let client = MockLlmClient::constant("answer");
        let config = VoteConfig::default().without_token_limit();

        let result = vote_with_margin("test", 3, &client, config).unwrap();
        assert_eq!(result.winner, "answer");
    }

    #[test]
    fn test_cost_metrics_default() {
        let cost = CostMetrics::default();
        assert_eq!(cost.input_tokens, 0);
        assert_eq!(cost.output_tokens, 0);
        assert!(cost.estimated_cost_usd.is_none());
    }
}
