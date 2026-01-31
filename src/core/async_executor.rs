//! Async Voting Executor for MAKER Framework
//!
//! Provides async versions of the voting functions for efficient parallel sampling.
//! Uses tokio for concurrent LLM calls and supports cancellation.

use crate::core::adaptive::{KEstimator, VoteObservation};
use crate::core::executor::{CostMetrics, LlmResponse, VoteConfig, VoteError, VoteResult};
use crate::core::redflag::RedFlagValidator;
use crate::core::voting::{CandidateId, VoteCheckResult, VoteRace};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// Async LLM client trait
///
/// Async version of `LlmClient` for use with tokio runtime.
pub trait AsyncLlmClient: Send + Sync {
    /// Generate a response asynchronously
    fn generate_async(
        &self,
        prompt: &str,
        temperature: f64,
    ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, String>> + Send + '_>>;
}

/// Cancellation token for async operations
///
/// Allows graceful cancellation of ongoing voting operations.
#[derive(Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Create a new cancellation token
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Request cancellation
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Check if cancellation has been requested
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

/// Configuration for async voting
#[derive(Clone)]
pub struct AsyncVoteConfig {
    /// Base vote configuration
    pub base: VoteConfig,
    /// Maximum concurrent samples (for batch processing)
    pub max_concurrent: usize,
    /// Cancellation token
    pub cancel_token: Option<CancellationToken>,
}

impl Default for AsyncVoteConfig {
    fn default() -> Self {
        Self {
            base: VoteConfig::default(),
            max_concurrent: 4,
            cancel_token: None,
        }
    }
}

impl AsyncVoteConfig {
    /// Create from base config
    pub fn from_base(base: VoteConfig) -> Self {
        Self {
            base,
            max_concurrent: 4,
            cancel_token: None,
        }
    }

    /// Set maximum concurrent samples
    pub fn with_max_concurrent(mut self, max: usize) -> Self {
        self.max_concurrent = max.max(1);
        self
    }

    /// Set cancellation token
    pub fn with_cancel_token(mut self, token: CancellationToken) -> Self {
        self.cancel_token = Some(token);
        self
    }
}

/// Async voting executor
///
/// Manages async LLM sampling with proper rate limiting and cancellation.
pub struct AsyncVotingExecutor<C: AsyncLlmClient> {
    client: Arc<C>,
    config: AsyncVoteConfig,
}

impl<C: AsyncLlmClient + 'static> AsyncVotingExecutor<C> {
    /// Create a new async voting executor
    pub fn new(client: C, config: AsyncVoteConfig) -> Self {
        Self {
            client: Arc::new(client),
            config,
        }
    }

    /// Execute voting asynchronously
    pub async fn vote(&self, prompt: &str, k: usize) -> Result<VoteResult, VoteError> {
        vote_with_margin_async(prompt, k, self.client.as_ref(), self.config.clone()).await
    }
}

/// Execute voting asynchronously with the given margin
///
/// Async version of `vote_with_margin` that allows concurrent I/O.
/// Samples are collected sequentially but each sample generation is async.
///
/// # Arguments
///
/// * `prompt` - The prompt to send to the LLM
/// * `k` - The required margin for declaring a winner
/// * `client` - The async LLM client to use
/// * `config` - Async voting configuration
///
/// # Returns
///
/// * `Ok(VoteResult)` - The winning response with metrics
/// * `Err(VoteError)` - If voting fails, times out, or is cancelled
pub async fn vote_with_margin_async<C: AsyncLlmClient>(
    prompt: &str,
    k: usize,
    client: &C,
    config: AsyncVoteConfig,
) -> Result<VoteResult, VoteError> {
    if k == 0 {
        return Err(VoteError::InvalidConfig {
            message: "k must be >= 1".to_string(),
        });
    }

    let start = Instant::now();
    let validator = match config.base.token_limit {
        Some(limit) => RedFlagValidator::new().with_token_limit(limit),
        None => RedFlagValidator::new().without_token_limit(),
    };

    let race = VoteRace::new(k)
        .map_err(|e| VoteError::InvalidConfig {
            message: e.to_string(),
        })?
        .with_matcher(config.base.matcher.clone());

    let mut total_samples = 0;
    let mut red_flagged = 0;
    let mut cost = CostMetrics::default();
    let mut candidate_contents: HashMap<String, String> = HashMap::new();

    // Voting loop - sequential but async
    loop {
        // Check cancellation
        if let Some(ref token) = config.cancel_token {
            if token.is_cancelled() {
                return Err(VoteError::Timeout {
                    elapsed: start.elapsed(),
                });
            }
        }

        // Check timeout
        if let Some(timeout_duration) = config.base.timeout {
            if start.elapsed() > timeout_duration {
                return Err(VoteError::Timeout {
                    elapsed: start.elapsed(),
                });
            }
        }

        // Check max samples
        if total_samples >= config.base.max_samples {
            return Err(VoteError::NoConvergence {
                samples_collected: total_samples,
                max_samples: config.base.max_samples,
            });
        }

        // Generate sample (T=0 for first, T=diversity for rest)
        let temperature = if total_samples == 0 {
            0.0
        } else {
            config.base.diversity_temperature
        };

        let response = client
            .generate_async(prompt, temperature)
            .await
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

        // Create candidate ID
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
                k_used: k,
                p_hat: None,
                cost_by_model: None,
                ensemble_metrics: None,
            });
        }

        // Check if all samples red-flagged
        if total_samples > k * 2 && race.total_votes() == 0 {
            return Err(VoteError::AllRedFlagged { total_samples });
        }
    }
}

/// Execute adaptive async voting with dynamic k-margin adjustment.
pub async fn vote_with_margin_adaptive_async<C: AsyncLlmClient>(
    prompt: &str,
    estimator: &mut KEstimator,
    target_t: f64,
    remaining_steps: usize,
    client: &C,
    config: AsyncVoteConfig,
) -> Result<VoteResult, VoteError> {
    let k = estimator.recommended_k(target_t, remaining_steps);

    let mut result = vote_with_margin_async(prompt, k, client, config).await?;

    // Feed observation back to estimator
    let observation = VoteObservation {
        converged_quickly: result.total_samples <= k,
        total_samples: result.total_samples,
        k_used: k,
        red_flagged: result.red_flagged,
    };
    estimator.observe(observation);

    // Annotate result with adaptive info
    result.k_used = k;
    result.p_hat = Some(estimator.p_hat());

    Ok(result)
}

/// Execute voting with concurrent batch sampling
///
/// Collects multiple samples concurrently for improved throughput.
/// Uses join_all to await multiple futures simultaneously.
pub async fn vote_with_margin_concurrent<C: AsyncLlmClient + Clone + 'static>(
    prompt: &str,
    k: usize,
    client: C,
    config: AsyncVoteConfig,
) -> Result<VoteResult, VoteError> {
    if k == 0 {
        return Err(VoteError::InvalidConfig {
            message: "k must be >= 1".to_string(),
        });
    }

    let start = Instant::now();
    let validator = match config.base.token_limit {
        Some(limit) => RedFlagValidator::new().with_token_limit(limit),
        None => RedFlagValidator::new().without_token_limit(),
    };

    let race = Arc::new(
        VoteRace::new(k)
            .map_err(|e| VoteError::InvalidConfig {
                message: e.to_string(),
            })?
            .with_matcher(config.base.matcher.clone()),
    );

    let total_samples = Arc::new(AtomicUsize::new(0));
    let red_flagged = Arc::new(AtomicUsize::new(0));
    let cost = Arc::new(Mutex::new(CostMetrics::default()));
    let candidate_contents = Arc::new(Mutex::new(HashMap::<String, String>::new()));
    let client = Arc::new(client);

    // Concurrent voting loop
    loop {
        // Check cancellation
        if let Some(ref token) = config.cancel_token {
            if token.is_cancelled() {
                return Err(VoteError::Timeout {
                    elapsed: start.elapsed(),
                });
            }
        }

        // Check timeout
        if let Some(timeout_duration) = config.base.timeout {
            if start.elapsed() > timeout_duration {
                return Err(VoteError::Timeout {
                    elapsed: start.elapsed(),
                });
            }
        }

        let current_samples = total_samples.load(Ordering::SeqCst);

        // Check max samples
        if current_samples >= config.base.max_samples {
            return Err(VoteError::NoConvergence {
                samples_collected: current_samples,
                max_samples: config.base.max_samples,
            });
        }

        // Check for winner
        if let VoteCheckResult::Winner { candidate, .. } = race.check_winner() {
            let contents = candidate_contents.lock().await;
            let winner_content = contents
                .get(candidate.as_str())
                .cloned()
                .unwrap_or_else(|| candidate.as_str().to_string());

            let vote_counts: HashMap<String, usize> = race
                .get_votes()
                .into_iter()
                .map(|(id, count)| (id.as_str().to_string(), count))
                .collect();

            let total = total_samples.load(Ordering::SeqCst);
            let red = red_flagged.load(Ordering::SeqCst);
            let cost_snapshot = cost.lock().await.clone();

            return Ok(VoteResult {
                winner: winner_content,
                vote_counts,
                total_samples: total,
                red_flagged: red,
                cost: cost_snapshot,
                elapsed: start.elapsed(),
                k_used: k,
                p_hat: None,
                cost_by_model: None,
                ensemble_metrics: None,
            });
        }

        // Calculate batch size
        let remaining = config.base.max_samples - current_samples;
        let batch_size = config.max_concurrent.min(remaining);

        // Spawn concurrent sample tasks
        let mut handles = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let client_clone = Arc::clone(&client);
            let race_clone = Arc::clone(&race);
            let total_clone = Arc::clone(&total_samples);
            let red_clone = Arc::clone(&red_flagged);
            let cost_clone = Arc::clone(&cost);
            let contents_clone = Arc::clone(&candidate_contents);
            let validator_clone = validator.clone();
            let prompt = prompt.to_string();
            let diversity_temp = config.base.diversity_temperature;

            let sample_idx = current_samples + i;
            let temperature = if sample_idx == 0 { 0.0 } else { diversity_temp };

            let handle = tokio::spawn(async move {
                // Generate sample
                let response = client_clone.generate_async(&prompt, temperature).await?;

                // Update counters
                total_clone.fetch_add(1, Ordering::SeqCst);
                {
                    let mut c = cost_clone.lock().await;
                    c.input_tokens += response.input_tokens;
                    c.output_tokens += response.output_tokens;
                }

                // Red-flag check
                let flags = validator_clone.validate(&response.content);
                if !flags.is_empty() {
                    red_clone.fetch_add(1, Ordering::SeqCst);
                    return Ok::<_, String>(false);
                }

                // Create candidate ID and store content
                let candidate_id = CandidateId::new(&response.content);
                {
                    let mut contents = contents_clone.lock().await;
                    contents
                        .entry(response.content.clone())
                        .or_insert(response.content);
                }

                // Cast vote
                race_clone.cast_vote(candidate_id);

                Ok(true)
            });

            handles.push(handle);
        }

        // Wait for all tasks in batch
        for handle in handles {
            match handle.await {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => return Err(VoteError::LlmError { message: e }),
                Err(e) => {
                    return Err(VoteError::LlmError {
                        message: format!("Task panicked: {}", e),
                    })
                }
            }
        }

        // Check if all samples red-flagged
        let total = total_samples.load(Ordering::SeqCst);
        if total > k * 2 && race.total_votes() == 0 {
            return Err(VoteError::AllRedFlagged {
                total_samples: total,
            });
        }
    }
}

/// Mock async LLM client for testing
#[derive(Debug, Clone)]
pub struct MockAsyncLlmClient {
    responses: Vec<String>,
    index: Arc<AtomicUsize>,
    tokens_per_response: usize,
    delay: Option<Duration>,
}

impl MockAsyncLlmClient {
    /// Create a new mock async client
    pub fn new(responses: Vec<String>) -> Self {
        Self {
            responses,
            index: Arc::new(AtomicUsize::new(0)),
            tokens_per_response: 50,
            delay: None,
        }
    }

    /// Create a mock that always returns the same response
    pub fn constant(response: &str) -> Self {
        Self::new(vec![response.to_string()])
    }

    /// Add simulated delay
    pub fn with_delay(mut self, delay: Duration) -> Self {
        self.delay = Some(delay);
        self
    }

    /// Create a mock with biased responses
    pub fn biased(correct: &str, incorrect: &str, p: f64, total: usize) -> Self {
        let correct_count = (total as f64 * p).round() as usize;
        let mut responses = vec![correct.to_string(); correct_count];
        responses.extend(vec![incorrect.to_string(); total - correct_count]);
        Self::new(responses)
    }
}

impl AsyncLlmClient for MockAsyncLlmClient {
    fn generate_async(
        &self,
        _prompt: &str,
        _temperature: f64,
    ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, String>> + Send + '_>> {
        let idx = self.index.fetch_add(1, Ordering::SeqCst);
        let response = self.responses[idx % self.responses.len()].clone();
        let tokens = self.tokens_per_response;
        let delay = self.delay;

        Box::pin(async move {
            if let Some(d) = delay {
                tokio::time::sleep(d).await;
            }
            Ok(LlmResponse {
                content: response,
                input_tokens: 100,
                output_tokens: tokens,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::adaptive::KEstimatorConfig;

    #[tokio::test]
    async fn test_async_vote_basic() {
        let client = MockAsyncLlmClient::constant("test response");
        let config = AsyncVoteConfig::default();

        let result = vote_with_margin_async("prompt", 3, &client, config).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.winner, "test response");
        assert!(result.total_samples >= 3);
    }

    #[tokio::test]
    async fn test_async_vote_concurrent() {
        let client = MockAsyncLlmClient::constant("concurrent response")
            .with_delay(Duration::from_millis(10));
        let config = AsyncVoteConfig::default().with_max_concurrent(4);

        let start = Instant::now();
        let result = vote_with_margin_concurrent("prompt", 3, client, config).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        // With 4 concurrent, should be faster than sequential
        assert!(
            elapsed < Duration::from_millis(100),
            "Elapsed: {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_async_vote_cancellation() {
        let client =
            MockAsyncLlmClient::constant("response").with_delay(Duration::from_millis(100));
        let cancel = CancellationToken::new();
        let config = AsyncVoteConfig::default().with_cancel_token(cancel.clone());

        // Cancel immediately
        cancel.cancel();

        let result = vote_with_margin_async("prompt", 3, &client, config).await;
        assert!(matches!(result, Err(VoteError::Timeout { .. })));
    }

    #[tokio::test]
    async fn test_async_vote_timeout() {
        let client =
            MockAsyncLlmClient::constant("response").with_delay(Duration::from_millis(100));
        let config = AsyncVoteConfig {
            base: VoteConfig::default().with_timeout(Duration::from_millis(50)),
            max_concurrent: 1,
            cancel_token: None,
        };

        let result = vote_with_margin_async("prompt", 3, &client, config).await;
        assert!(matches!(result, Err(VoteError::Timeout { .. })));
    }

    #[tokio::test]
    async fn test_async_executor() {
        let client = MockAsyncLlmClient::constant("executor response");
        let config = AsyncVoteConfig::default();
        let executor = AsyncVotingExecutor::new(client, config);

        let result = executor.vote("test prompt", 3).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().winner, "executor response");
    }

    #[tokio::test]
    async fn test_cancellation_token() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());

        token.cancel();
        assert!(token.is_cancelled());

        // Clone should share state
        let token2 = token.clone();
        assert!(token2.is_cancelled());
    }

    #[tokio::test]
    async fn test_async_vote_k_zero_rejected() {
        let client = MockAsyncLlmClient::constant("response");
        let config = AsyncVoteConfig::default();

        let result = vote_with_margin_async("prompt", 0, &client, config).await;
        assert!(matches!(result, Err(VoteError::InvalidConfig { .. })));
    }

    #[tokio::test]
    async fn test_async_vote_results_match_sync_semantics() {
        let client = MockAsyncLlmClient::biased("correct", "incorrect", 0.8, 100);
        let config = AsyncVoteConfig::default();

        let result = vote_with_margin_async("prompt", 3, &client, config).await;
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.winner, "correct");
    }

    #[tokio::test]
    async fn test_mock_async_client_delay() {
        let client = MockAsyncLlmClient::constant("delayed").with_delay(Duration::from_millis(50));

        let start = Instant::now();
        let result = client.generate_async("test", 0.0).await;
        let elapsed = start.elapsed();

        assert!(result.is_ok());
        assert!(elapsed >= Duration::from_millis(50));
    }

    #[tokio::test]
    async fn test_async_adaptive_voting() {
        let client = MockAsyncLlmClient::constant("adaptive response");
        let config = AsyncVoteConfig::default();
        let estimator_config = KEstimatorConfig::default();
        let mut estimator = KEstimator::new(estimator_config);

        let result =
            vote_with_margin_adaptive_async("prompt", &mut estimator, 0.95, 100, &client, config)
                .await;

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.k_used >= 1);
        assert!(result.p_hat.is_some());
    }

    #[tokio::test]
    async fn test_concurrent_vote_basic() {
        let client = MockAsyncLlmClient::constant("concurrent test");
        let config = AsyncVoteConfig::default().with_max_concurrent(2);

        let result = vote_with_margin_concurrent("prompt", 3, client, config).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().winner, "concurrent test");
    }
}
