//! Exponential Backoff Retry Strategy
//!
//! Provides automatic retry with exponential backoff for transient LLM API failures.
//! Handles rate limits, network errors, and server errors gracefully.
//!
//! # Strategy
//!
//! - Delay doubles each attempt: base_delay * 2^attempt
//! - Jitter prevents thundering herd: delay * (1 + random(0, jitter_factor))
//! - Retry-After header overrides calculated delay
//! - Non-retryable errors fail immediately (4xx except 429)

use crate::llm::LlmError;
use std::future::Future;
use std::time::Duration;
use tokio::time::sleep;

/// Configuration for retry behavior
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (0 = no retries)
    pub max_retries: usize,
    /// Initial delay before first retry
    pub base_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Random jitter factor (0.0 - 1.0)
    pub jitter_factor: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            base_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(60),
            jitter_factor: 0.25,
        }
    }
}

impl RetryConfig {
    /// Create a new retry config with custom values
    pub fn new(
        max_retries: usize,
        base_delay: Duration,
        max_delay: Duration,
        jitter_factor: f64,
    ) -> Self {
        Self {
            max_retries,
            base_delay,
            max_delay,
            jitter_factor: jitter_factor.clamp(0.0, 1.0),
        }
    }

    /// Calculate delay for a given attempt (0-indexed)
    ///
    /// Uses exponential backoff: delay = min(base * 2^attempt, max)
    /// Then applies jitter: delay * (1 + random(0, jitter_factor))
    pub fn calculate_delay(&self, attempt: usize) -> Duration {
        let exp_delay = self.base_delay.as_millis() as u64 * 2u64.saturating_pow(attempt as u32);
        let capped = exp_delay.min(self.max_delay.as_millis() as u64);

        // Apply jitter deterministically for testing (use attempt as seed)
        let jitter = if self.jitter_factor > 0.0 {
            let pseudo_random = ((attempt as f64 * 0.618033988749895) % 1.0) * self.jitter_factor;
            1.0 + pseudo_random
        } else {
            1.0
        };

        Duration::from_millis((capped as f64 * jitter) as u64)
    }

    /// Override delay with Retry-After header if present
    pub fn delay_with_retry_after(
        &self,
        attempt: usize,
        retry_after: Option<Duration>,
    ) -> Duration {
        retry_after.unwrap_or_else(|| self.calculate_delay(attempt))
    }
}

/// Execute an async operation with retry logic
///
/// # Arguments
/// * `operation` - Async closure that returns Result<T, LlmError>
/// * `config` - Retry configuration
///
/// # Returns
/// * `Ok(T)` - Operation succeeded (possibly after retries)
/// * `Err(LlmError)` - All retries exhausted or non-retryable error
///
/// # Example
///
/// ```rust,ignore
/// let result = call_with_retry(
///     || async { client.generate("prompt", 0.0).await },
///     RetryConfig::default(),
/// ).await;
/// ```
pub async fn call_with_retry<F, Fut, T>(
    mut operation: F,
    config: RetryConfig,
) -> Result<T, LlmError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, LlmError>>,
{
    let mut attempt = 0;

    loop {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(err) => {
                // Check if error is retryable
                if !err.is_retryable() {
                    return Err(err);
                }

                // Check if we've exhausted retries
                if attempt >= config.max_retries {
                    return Err(err);
                }

                // Get retry-after if rate limited
                let retry_after = if let LlmError::RateLimited { retry_after } = &err {
                    *retry_after
                } else {
                    None
                };

                // Calculate and apply delay
                let delay = config.delay_with_retry_after(attempt, retry_after);
                sleep(delay).await;

                attempt += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // ==========================================
    // RetryConfig Tests
    // ==========================================

    #[test]
    fn test_default_config() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.base_delay, Duration::from_secs(1));
        assert_eq!(config.max_delay, Duration::from_secs(60));
        assert!((config.jitter_factor - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_config_new() {
        let config = RetryConfig::new(3, Duration::from_millis(500), Duration::from_secs(30), 0.1);
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.base_delay, Duration::from_millis(500));
    }

    #[test]
    fn test_jitter_factor_clamped() {
        let config = RetryConfig::new(1, Duration::from_secs(1), Duration::from_secs(10), 2.0);
        assert!((config.jitter_factor - 1.0).abs() < f64::EPSILON);

        let config2 = RetryConfig::new(1, Duration::from_secs(1), Duration::from_secs(10), -0.5);
        assert!((config2.jitter_factor - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_calculate_delay_exponential() {
        let config = RetryConfig::new(5, Duration::from_secs(1), Duration::from_secs(60), 0.0);

        // Without jitter, delays should double each attempt
        assert_eq!(config.calculate_delay(0), Duration::from_secs(1));
        assert_eq!(config.calculate_delay(1), Duration::from_secs(2));
        assert_eq!(config.calculate_delay(2), Duration::from_secs(4));
        assert_eq!(config.calculate_delay(3), Duration::from_secs(8));
    }

    #[test]
    fn test_calculate_delay_capped() {
        let config = RetryConfig::new(10, Duration::from_secs(1), Duration::from_secs(10), 0.0);

        // Should cap at max_delay
        assert_eq!(config.calculate_delay(5), Duration::from_secs(10)); // 2^5 = 32, capped to 10
        assert_eq!(config.calculate_delay(10), Duration::from_secs(10));
    }

    #[test]
    fn test_delay_with_retry_after_override() {
        let config = RetryConfig::default();
        let retry_after = Some(Duration::from_secs(30));

        let delay = config.delay_with_retry_after(0, retry_after);
        assert_eq!(delay, Duration::from_secs(30));
    }

    #[test]
    fn test_delay_with_retry_after_none() {
        let config = RetryConfig::new(5, Duration::from_secs(1), Duration::from_secs(60), 0.0);

        let delay = config.delay_with_retry_after(2, None);
        assert_eq!(delay, Duration::from_secs(4)); // 2^2 = 4
    }

    // ==========================================
    // call_with_retry Tests
    // ==========================================

    #[tokio::test]
    async fn test_retry_success_first_attempt() {
        let config = RetryConfig::new(3, Duration::from_millis(1), Duration::from_millis(10), 0.0);

        let result = call_with_retry(|| async { Ok::<_, LlmError>("success") }, config).await;

        assert_eq!(result.unwrap(), "success");
    }

    #[tokio::test]
    async fn test_retry_success_after_failures() {
        let attempt_count = Arc::new(AtomicUsize::new(0));
        let count = attempt_count.clone();

        let config = RetryConfig::new(5, Duration::from_millis(1), Duration::from_millis(10), 0.0);

        let result = call_with_retry(
            || {
                let count = count.clone();
                async move {
                    let attempt = count.fetch_add(1, Ordering::SeqCst);
                    if attempt < 2 {
                        Err(LlmError::Timeout)
                    } else {
                        Ok("success after retries")
                    }
                }
            },
            config,
        )
        .await;

        assert_eq!(result.unwrap(), "success after retries");
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3); // 2 failures + 1 success
    }

    #[tokio::test]
    async fn test_retry_exhausted() {
        let attempt_count = Arc::new(AtomicUsize::new(0));
        let count = attempt_count.clone();

        let config = RetryConfig::new(2, Duration::from_millis(1), Duration::from_millis(10), 0.0);

        let result: Result<(), LlmError> = call_with_retry(
            || {
                let count = count.clone();
                async move {
                    count.fetch_add(1, Ordering::SeqCst);
                    Err(LlmError::NetworkError("always fails".to_string()))
                }
            },
            config,
        )
        .await;

        assert!(result.is_err());
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3); // 1 initial + 2 retries
    }

    #[tokio::test]
    async fn test_retry_non_retryable_fails_immediately() {
        let attempt_count = Arc::new(AtomicUsize::new(0));
        let count = attempt_count.clone();

        let config = RetryConfig::new(5, Duration::from_millis(1), Duration::from_millis(10), 0.0);

        let result: Result<(), LlmError> = call_with_retry(
            || {
                let count = count.clone();
                async move {
                    count.fetch_add(1, Ordering::SeqCst);
                    Err(LlmError::ApiError {
                        status: 400,
                        message: "Bad request".to_string(),
                    })
                }
            },
            config,
        )
        .await;

        assert!(result.is_err());
        assert_eq!(attempt_count.load(Ordering::SeqCst), 1); // Only 1 attempt, no retries
    }

    #[tokio::test]
    async fn test_retry_rate_limited_uses_retry_after() {
        let attempt_count = Arc::new(AtomicUsize::new(0));
        let count = attempt_count.clone();

        // Use very short delays for testing
        let config = RetryConfig::new(2, Duration::from_millis(100), Duration::from_secs(1), 0.0);

        let start = std::time::Instant::now();

        let result: Result<&str, LlmError> = call_with_retry(
            || {
                let count = count.clone();
                async move {
                    let attempt = count.fetch_add(1, Ordering::SeqCst);
                    if attempt == 0 {
                        Err(LlmError::RateLimited {
                            retry_after: Some(Duration::from_millis(50)),
                        })
                    } else {
                        Ok("success")
                    }
                }
            },
            config,
        )
        .await;

        let elapsed = start.elapsed();

        assert!(result.is_ok());
        // Should have waited ~50ms (the retry_after value), not 100ms (base_delay)
        assert!(elapsed >= Duration::from_millis(40)); // Allow some slack
        assert!(elapsed < Duration::from_millis(150)); // But not too long
    }

    #[tokio::test]
    async fn test_retry_500_error_is_retried() {
        let attempt_count = Arc::new(AtomicUsize::new(0));
        let count = attempt_count.clone();

        let config = RetryConfig::new(3, Duration::from_millis(1), Duration::from_millis(10), 0.0);

        let result: Result<&str, LlmError> = call_with_retry(
            || {
                let count = count.clone();
                async move {
                    let attempt = count.fetch_add(1, Ordering::SeqCst);
                    if attempt < 2 {
                        Err(LlmError::ApiError {
                            status: 503,
                            message: "Service unavailable".to_string(),
                        })
                    } else {
                        Ok("recovered")
                    }
                }
            },
            config,
        )
        .await;

        assert_eq!(result.unwrap(), "recovered");
        assert_eq!(attempt_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_zero_retries() {
        let attempt_count = Arc::new(AtomicUsize::new(0));
        let count = attempt_count.clone();

        let config = RetryConfig::new(0, Duration::from_millis(1), Duration::from_millis(10), 0.0);

        let result: Result<(), LlmError> = call_with_retry(
            || {
                let count = count.clone();
                async move {
                    count.fetch_add(1, Ordering::SeqCst);
                    Err(LlmError::Timeout)
                }
            },
            config,
        )
        .await;

        assert!(result.is_err());
        assert_eq!(attempt_count.load(Ordering::SeqCst), 1); // Only initial attempt
    }
}
