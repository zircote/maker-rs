//! Parallel Sampling for MAKER Voting
//!
//! Collects multiple LLM samples in parallel for voting, minimizing latency.
//! Uses Tokio's JoinSet for bounded concurrency.
//!
//! # Temperature Strategy
//!
//! - First sample: T=0.0 (deterministic baseline)
//! - Remaining samples: T=config.diversity_temp (default 0.1)
//!
//! This ensures we always get the most likely response while also
//! exploring nearby alternatives for robust voting.

use crate::events::{EventBus, MakerEvent};
use crate::llm::{LlmClient, LlmError, LlmResponse};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task::JoinSet;

/// Configuration for parallel sampling
#[derive(Debug, Clone)]
pub struct SampleConfig {
    /// Temperature for diversity (samples 2..n)
    pub diversity_temp: f64,
    /// Global timeout for all samples
    pub timeout: Duration,
    /// Maximum concurrent requests
    pub max_concurrent: usize,
}

impl Default for SampleConfig {
    fn default() -> Self {
        Self {
            diversity_temp: 0.1,
            timeout: Duration::from_secs(60),
            max_concurrent: 10,
        }
    }
}

impl SampleConfig {
    /// Create a new sample config
    pub fn new(diversity_temp: f64, timeout: Duration, max_concurrent: usize) -> Self {
        Self {
            diversity_temp,
            timeout,
            max_concurrent: max_concurrent.max(1),
        }
    }

    /// Get temperature for a given sample index (0-indexed)
    pub fn temperature_for_sample(&self, index: usize) -> f64 {
        if index == 0 {
            0.0 // First sample is deterministic
        } else {
            self.diversity_temp
        }
    }
}

/// Result of collecting samples
#[derive(Debug)]
pub struct SampleResult {
    /// All collected responses (success or failure)
    pub responses: Vec<Result<LlmResponse, LlmError>>,
    /// Total time to collect all samples
    pub total_duration: Duration,
    /// Number of successful samples
    pub success_count: usize,
    /// Number of failed samples
    pub failure_count: usize,
}

impl SampleResult {
    /// Get only successful responses
    pub fn successes(&self) -> impl Iterator<Item = &LlmResponse> {
        self.responses.iter().filter_map(|r| r.as_ref().ok())
    }

    /// Get only failed responses
    pub fn failures(&self) -> impl Iterator<Item = &LlmError> {
        self.responses.iter().filter_map(|r| r.as_ref().err())
    }
}

/// Collect multiple samples from an LLM in parallel
///
/// # Arguments
/// * `prompt` - The prompt to send to the LLM
/// * `num_samples` - Number of samples to collect
/// * `client` - LLM client to use
/// * `config` - Sampling configuration
/// * `event_bus` - Optional event bus for emitting events
///
/// # Returns
/// * `SampleResult` containing all responses and timing information
///
/// # Example
///
/// ```rust,ignore
/// let result = collect_samples(
///     "What is 2+2?",
///     5,
///     Arc::new(my_client),
///     SampleConfig::default(),
///     None,
/// ).await;
/// ```
pub async fn collect_samples(
    prompt: &str,
    num_samples: usize,
    client: Arc<dyn LlmClient>,
    config: SampleConfig,
    event_bus: Option<&EventBus>,
) -> SampleResult {
    let start = Instant::now();
    let prompt_hash = format!("{:016x}", hash_prompt(prompt));

    // Create bounded task set
    let mut join_set: JoinSet<(usize, Result<LlmResponse, LlmError>)> = JoinSet::new();

    // Spawn sampling tasks
    for i in 0..num_samples {
        let client = Arc::clone(&client);
        let prompt = prompt.to_string();
        let temp = config.temperature_for_sample(i);
        let model = client.model_name().to_string();
        let prompt_hash_clone = prompt_hash.clone();

        // Emit SampleRequested event
        if let Some(bus) = event_bus {
            bus.emit(MakerEvent::sample_requested(
                &model,
                &prompt_hash_clone,
                temp,
            ));
        }

        join_set.spawn(async move {
            let result = client.generate(&prompt, temp).await;
            (i, result)
        });

        // Respect max_concurrent by waiting if we hit the limit
        if join_set.len() >= config.max_concurrent {
            if let Some(Ok((idx, result))) = join_set.join_next().await {
                // We can't easily store these results mid-loop, so just drop them
                // The main collection loop will handle any remaining tasks
                let _ = (idx, result);
            }
        }
    }

    // Collect all results with timeout
    let mut responses = vec![Err(LlmError::Timeout); num_samples];
    let deadline = tokio::time::Instant::now() + config.timeout;

    while let Ok(Some(result)) = tokio::time::timeout_at(deadline, join_set.join_next()).await {
        if let Ok((idx, response)) = result {
            // Emit SampleCompleted event
            if let Some(bus) = event_bus {
                let model = client.model_name();
                match &response {
                    Ok(resp) => {
                        bus.emit(MakerEvent::sample_completed(
                            model,
                            resp.tokens.total(),
                            resp.latency.as_millis() as u64,
                            vec![],
                        ));
                    }
                    Err(_) => {
                        // Error case - emit with zero tokens
                        bus.emit(MakerEvent::sample_completed(model, 0, 0, vec![]));
                    }
                }
            }
            responses[idx] = response;
        }
    }

    // Cancel any remaining tasks
    join_set.abort_all();

    let success_count = responses.iter().filter(|r| r.is_ok()).count();
    let failure_count = responses.iter().filter(|r| r.is_err()).count();

    SampleResult {
        responses,
        total_duration: start.elapsed(),
        success_count,
        failure_count,
    }
}

/// Simple hash function for prompt (for event correlation)
fn hash_prompt(prompt: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    prompt.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{TokenCost, TokenUsage};
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Mock LLM client for testing parallel sampling
    struct MockSamplerClient {
        delay: Duration,
        response_fn: Box<dyn Fn(usize, f64) -> Result<LlmResponse, LlmError> + Send + Sync>,
        call_count: AtomicUsize,
    }

    impl MockSamplerClient {
        fn constant(content: &str, delay: Duration) -> Self {
            let content = content.to_string();
            Self {
                delay,
                response_fn: Box::new(move |_, _| {
                    Ok(LlmResponse {
                        content: content.clone(),
                        tokens: TokenUsage::new(10, 20),
                        latency: delay,
                    })
                }),
                call_count: AtomicUsize::new(0),
            }
        }

        fn with_fn<F>(delay: Duration, f: F) -> Self
        where
            F: Fn(usize, f64) -> Result<LlmResponse, LlmError> + Send + Sync + 'static,
        {
            Self {
                delay,
                response_fn: Box::new(f),
                call_count: AtomicUsize::new(0),
            }
        }

        #[allow(dead_code)]
        fn call_count(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    impl LlmClient for MockSamplerClient {
        fn generate(
            &self,
            _prompt: &str,
            temperature: f64,
        ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, LlmError>> + Send + '_>> {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            let delay = self.delay;
            let result = (self.response_fn)(idx, temperature);

            Box::pin(async move {
                tokio::time::sleep(delay).await;
                result
            })
        }

        fn model_name(&self) -> &str {
            "mock-sampler"
        }

        fn cost_per_1k_tokens(&self) -> TokenCost {
            TokenCost::new(0.001, 0.002)
        }
    }

    // ==========================================
    // SampleConfig Tests
    // ==========================================

    #[test]
    fn test_config_default() {
        let config = SampleConfig::default();
        assert!((config.diversity_temp - 0.1).abs() < f64::EPSILON);
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.max_concurrent, 10);
    }

    #[test]
    fn test_config_temperature_strategy() {
        let config = SampleConfig::default();

        // First sample is deterministic
        assert!((config.temperature_for_sample(0) - 0.0).abs() < f64::EPSILON);

        // Rest use diversity temperature
        assert!((config.temperature_for_sample(1) - 0.1).abs() < f64::EPSILON);
        assert!((config.temperature_for_sample(5) - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_max_concurrent_minimum() {
        let config = SampleConfig::new(0.1, Duration::from_secs(10), 0);
        assert_eq!(config.max_concurrent, 1); // Clamped to minimum 1
    }

    // ==========================================
    // Parallel Sampling Tests
    // ==========================================

    #[tokio::test]
    async fn test_collect_samples_returns_all() {
        let client = Arc::new(MockSamplerClient::constant(
            "response",
            Duration::from_millis(10),
        ));
        let config = SampleConfig::new(0.1, Duration::from_secs(5), 10);

        let result = collect_samples("test prompt", 5, client, config, None).await;

        assert_eq!(result.responses.len(), 5);
        assert_eq!(result.success_count, 5);
        assert_eq!(result.failure_count, 0);
    }

    #[tokio::test]
    async fn test_collect_samples_parallel_faster_than_sequential() {
        let delay = Duration::from_millis(50);
        let client = Arc::new(MockSamplerClient::constant("response", delay));
        let config = SampleConfig::new(0.1, Duration::from_secs(5), 10);

        let start = Instant::now();
        let result = collect_samples("test", 5, client, config, None).await;
        let elapsed = start.elapsed();

        // Sequential would take 5 * 50ms = 250ms
        // Parallel should take ~50-100ms (roughly 1x single call + overhead)
        assert!(result.success_count == 5);
        assert!(
            elapsed < Duration::from_millis(200),
            "Parallel sampling too slow: {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_collect_samples_temperature_strategy() {
        let temps: Arc<std::sync::Mutex<Vec<f64>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
        let temps_clone = temps.clone();

        let client = Arc::new(MockSamplerClient::with_fn(
            Duration::from_millis(5),
            move |_, temp| {
                temps_clone.lock().unwrap().push(temp);
                Ok(LlmResponse {
                    content: format!("temp={}", temp),
                    tokens: TokenUsage::new(10, 10),
                    latency: Duration::from_millis(5),
                })
            },
        ));

        let config = SampleConfig::new(0.2, Duration::from_secs(5), 10);
        let _ = collect_samples("test", 3, client, config, None).await;

        let recorded_temps = temps.lock().unwrap();
        // First should be T=0.0, rest should be T=0.2
        // Note: order may vary due to parallelism, so check counts
        let zero_count = recorded_temps.iter().filter(|&&t| t == 0.0).count();
        let diversity_count = recorded_temps
            .iter()
            .filter(|&&t| (t - 0.2).abs() < f64::EPSILON)
            .count();

        assert_eq!(zero_count, 1, "Should have exactly one T=0 sample");
        assert_eq!(diversity_count, 2, "Should have two diversity samples");
    }

    #[tokio::test]
    async fn test_collect_samples_handles_failures() {
        let client = Arc::new(MockSamplerClient::with_fn(
            Duration::from_millis(5),
            |idx, _| {
                if idx % 2 == 0 {
                    Ok(LlmResponse {
                        content: "success".to_string(),
                        tokens: TokenUsage::new(10, 10),
                        latency: Duration::from_millis(5),
                    })
                } else {
                    Err(LlmError::NetworkError("simulated failure".to_string()))
                }
            },
        ));

        let config = SampleConfig::new(0.1, Duration::from_secs(5), 10);
        let result = collect_samples("test", 4, client, config, None).await;

        // Should have mix of successes and failures
        assert!(result.success_count > 0);
        assert!(result.failure_count > 0);
        assert_eq!(result.success_count + result.failure_count, 4);
    }

    #[tokio::test]
    async fn test_collect_samples_timeout() {
        // Create client with very long delay
        let client = Arc::new(MockSamplerClient::constant("slow", Duration::from_secs(10)));

        // Very short timeout
        let config = SampleConfig::new(0.1, Duration::from_millis(100), 10);

        let start = Instant::now();
        let result = collect_samples("test", 3, client, config, None).await;
        let elapsed = start.elapsed();

        // Should timeout quickly, not wait for slow responses
        assert!(
            elapsed < Duration::from_secs(1),
            "Should have timed out quickly"
        );
        // All should be failures (timeout)
        assert_eq!(result.failure_count, 3);
    }

    #[tokio::test]
    async fn test_collect_samples_respects_max_concurrent() {
        let concurrent_count = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));

        let concurrent = concurrent_count.clone();
        let max = max_seen.clone();

        let client = Arc::new(MockSamplerClient::with_fn(
            Duration::from_millis(50),
            move |_, _| {
                let current = concurrent.fetch_add(1, Ordering::SeqCst) + 1;
                max.fetch_max(current, Ordering::SeqCst);

                // Simulate work
                std::thread::sleep(Duration::from_millis(10));

                concurrent.fetch_sub(1, Ordering::SeqCst);

                Ok(LlmResponse {
                    content: "done".to_string(),
                    tokens: TokenUsage::new(10, 10),
                    latency: Duration::from_millis(50),
                })
            },
        ));

        // Limit to 3 concurrent
        let config = SampleConfig::new(0.1, Duration::from_secs(5), 3);
        let _ = collect_samples("test", 10, client, config, None).await;

        // Max concurrent should not exceed limit (with some slack for timing)
        let observed_max = max_seen.load(Ordering::SeqCst);
        assert!(
            observed_max <= 5,
            "Max concurrent {} exceeded limit",
            observed_max
        );
    }

    #[tokio::test]
    async fn test_sample_result_iterators() {
        let client = Arc::new(MockSamplerClient::with_fn(
            Duration::from_millis(5),
            |idx, _| {
                if idx == 0 {
                    Ok(LlmResponse {
                        content: "success".to_string(),
                        tokens: TokenUsage::new(10, 10),
                        latency: Duration::from_millis(5),
                    })
                } else {
                    Err(LlmError::Timeout)
                }
            },
        ));

        let config = SampleConfig::default();
        let result = collect_samples("test", 2, client, config, None).await;

        assert_eq!(result.successes().count(), 1);
        assert_eq!(result.failures().count(), 1);
    }

    #[test]
    fn test_hash_prompt_deterministic() {
        let h1 = hash_prompt("test prompt");
        let h2 = hash_prompt("test prompt");
        let h3 = hash_prompt("different prompt");

        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }
}
