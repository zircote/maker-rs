//! LLM Provider Abstraction for MAKER Framework
//!
//! Provides a unified async trait for LLM API calls, enabling provider-agnostic
//! voting logic. Supports Ollama (local), OpenAI, and Anthropic.
//!
//! # Architecture
//!
//! ```text
//! vote_with_margin() → LlmClient trait → [OllamaClient, OpenAiClient, AnthropicClient]
//! ```

pub mod anthropic;
pub mod ollama;
pub mod openai;
pub mod retry;
pub mod sampler;

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

/// Response from an LLM API call
#[derive(Debug, Clone, PartialEq)]
pub struct LlmResponse {
    /// Generated text content
    pub content: String,
    /// Token usage for cost calculation
    pub tokens: TokenUsage,
    /// API call latency
    pub latency: Duration,
}

/// Token usage breakdown for cost calculation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TokenUsage {
    /// Input/prompt tokens
    pub input: usize,
    /// Output/completion tokens
    pub output: usize,
}

impl TokenUsage {
    /// Create new token usage
    pub fn new(input: usize, output: usize) -> Self {
        Self { input, output }
    }

    /// Total tokens (input + output)
    pub fn total(&self) -> usize {
        self.input + self.output
    }
}

/// Errors that can occur during LLM API calls
#[derive(Debug, Clone, PartialEq)]
pub enum LlmError {
    /// Rate limited by the API (429)
    RateLimited {
        /// Suggested retry delay from Retry-After header
        retry_after: Option<Duration>,
    },
    /// Request timed out
    Timeout,
    /// Network connectivity issue
    NetworkError(String),
    /// API returned an error response
    ApiError {
        /// HTTP status code
        status: u16,
        /// Error message from API
        message: String,
    },
    /// Response could not be parsed
    InvalidResponse(String),
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmError::RateLimited { retry_after } => {
                if let Some(d) = retry_after {
                    write!(f, "Rate limited, retry after {:?}", d)
                } else {
                    write!(f, "Rate limited")
                }
            }
            LlmError::Timeout => write!(f, "Request timed out"),
            LlmError::NetworkError(msg) => write!(f, "Network error: {}", msg),
            LlmError::ApiError { status, message } => {
                write!(f, "API error {}: {}", status, message)
            }
            LlmError::InvalidResponse(msg) => write!(f, "Invalid response: {}", msg),
        }
    }
}

impl std::error::Error for LlmError {}

impl LlmError {
    /// Check if this error is retryable
    ///
    /// Retryable errors: RateLimited, Timeout, NetworkError, 5xx ApiErrors
    /// Non-retryable: 4xx ApiErrors (except 429), InvalidResponse
    pub fn is_retryable(&self) -> bool {
        match self {
            LlmError::RateLimited { .. } => true,
            LlmError::Timeout => true,
            LlmError::NetworkError(_) => true,
            LlmError::ApiError { status, .. } => *status >= 500 || *status == 429,
            LlmError::InvalidResponse(_) => false,
        }
    }
}

/// Cost per 1K tokens (input, output) in USD
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenCost {
    /// Cost per 1K input tokens
    pub input_per_1k: f64,
    /// Cost per 1K output tokens
    pub output_per_1k: f64,
}

impl TokenCost {
    /// Create new token cost
    pub fn new(input_per_1k: f64, output_per_1k: f64) -> Self {
        Self {
            input_per_1k,
            output_per_1k,
        }
    }

    /// Calculate cost for given token usage
    pub fn calculate(&self, usage: TokenUsage) -> f64 {
        (usage.input as f64 * self.input_per_1k / 1000.0)
            + (usage.output as f64 * self.output_per_1k / 1000.0)
    }
}

/// Unified trait for LLM API clients
///
/// All LLM providers implement this trait, allowing the voting engine
/// to be provider-agnostic. The trait is object-safe through explicit
/// boxing of the async return type.
///
/// # Example
///
/// ```rust,ignore
/// async fn sample(client: &dyn LlmClient, prompt: &str) -> Result<String, LlmError> {
///     let response = client.generate(prompt, 0.1).await?;
///     Ok(response.content)
/// }
/// ```
pub trait LlmClient: Send + Sync {
    /// Generate a completion for the given prompt
    ///
    /// # Arguments
    /// * `prompt` - The input prompt
    /// * `temperature` - Sampling temperature (0.0 = deterministic, higher = more random)
    ///
    /// # Returns
    /// * `Ok(LlmResponse)` - Successful generation with content and metadata
    /// * `Err(LlmError)` - API call failed
    fn generate(
        &self,
        prompt: &str,
        temperature: f64,
    ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, LlmError>> + Send + '_>>;

    /// Get the model name/identifier
    fn model_name(&self) -> &str;

    /// Get cost per 1K tokens (input, output)
    fn cost_per_1k_tokens(&self) -> TokenCost;
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================
    // TokenUsage Tests
    // ==========================================

    #[test]
    fn test_token_usage_new() {
        let usage = TokenUsage::new(100, 50);
        assert_eq!(usage.input, 100);
        assert_eq!(usage.output, 50);
    }

    #[test]
    fn test_token_usage_total() {
        let usage = TokenUsage::new(100, 50);
        assert_eq!(usage.total(), 150);
    }

    #[test]
    fn test_token_usage_default() {
        let usage = TokenUsage::default();
        assert_eq!(usage.input, 0);
        assert_eq!(usage.output, 0);
    }

    // ==========================================
    // LlmError Tests
    // ==========================================

    #[test]
    fn test_error_display_rate_limited() {
        let err = LlmError::RateLimited {
            retry_after: Some(Duration::from_secs(30)),
        };
        assert!(err.to_string().contains("Rate limited"));
        assert!(err.to_string().contains("30"));
    }

    #[test]
    fn test_error_display_timeout() {
        let err = LlmError::Timeout;
        assert_eq!(err.to_string(), "Request timed out");
    }

    #[test]
    fn test_error_display_network() {
        let err = LlmError::NetworkError("connection refused".to_string());
        assert!(err.to_string().contains("connection refused"));
    }

    #[test]
    fn test_error_display_api() {
        let err = LlmError::ApiError {
            status: 400,
            message: "Bad request".to_string(),
        };
        assert!(err.to_string().contains("400"));
        assert!(err.to_string().contains("Bad request"));
    }

    #[test]
    fn test_error_is_retryable_rate_limited() {
        let err = LlmError::RateLimited { retry_after: None };
        assert!(err.is_retryable());
    }

    #[test]
    fn test_error_is_retryable_timeout() {
        let err = LlmError::Timeout;
        assert!(err.is_retryable());
    }

    #[test]
    fn test_error_is_retryable_network() {
        let err = LlmError::NetworkError("test".to_string());
        assert!(err.is_retryable());
    }

    #[test]
    fn test_error_is_retryable_500() {
        let err = LlmError::ApiError {
            status: 500,
            message: "Internal error".to_string(),
        };
        assert!(err.is_retryable());
    }

    #[test]
    fn test_error_not_retryable_400() {
        let err = LlmError::ApiError {
            status: 400,
            message: "Bad request".to_string(),
        };
        assert!(!err.is_retryable());
    }

    #[test]
    fn test_error_not_retryable_invalid_response() {
        let err = LlmError::InvalidResponse("parse error".to_string());
        assert!(!err.is_retryable());
    }

    // ==========================================
    // TokenCost Tests
    // ==========================================

    #[test]
    fn test_token_cost_new() {
        let cost = TokenCost::new(0.01, 0.03);
        assert!((cost.input_per_1k - 0.01).abs() < f64::EPSILON);
        assert!((cost.output_per_1k - 0.03).abs() < f64::EPSILON);
    }

    #[test]
    fn test_token_cost_calculate() {
        // $0.01 per 1K input, $0.03 per 1K output
        let cost = TokenCost::new(0.01, 0.03);
        let usage = TokenUsage::new(1000, 500);

        // Expected: 1000 * 0.01/1000 + 500 * 0.03/1000 = 0.01 + 0.015 = 0.025
        let total = cost.calculate(usage);
        assert!((total - 0.025).abs() < 1e-10);
    }

    #[test]
    fn test_token_cost_calculate_zero() {
        let cost = TokenCost::new(0.01, 0.03);
        let usage = TokenUsage::default();
        assert!((cost.calculate(usage) - 0.0).abs() < f64::EPSILON);
    }

    // ==========================================
    // LlmResponse Tests
    // ==========================================

    #[test]
    fn test_llm_response_creation() {
        let response = LlmResponse {
            content: "Hello, world!".to_string(),
            tokens: TokenUsage::new(10, 5),
            latency: Duration::from_millis(100),
        };

        assert_eq!(response.content, "Hello, world!");
        assert_eq!(response.tokens.total(), 15);
        assert_eq!(response.latency, Duration::from_millis(100));
    }

    #[test]
    fn test_llm_response_clone() {
        let response = LlmResponse {
            content: "test".to_string(),
            tokens: TokenUsage::new(1, 1),
            latency: Duration::from_millis(50),
        };

        let cloned = response.clone();
        assert_eq!(response, cloned);
    }

    // ==========================================
    // LlmClient Trait Tests (with Mock)
    // ==========================================

    /// Mock LLM client for testing
    struct MockLlmClient {
        responses: std::sync::Mutex<Vec<Result<LlmResponse, LlmError>>>,
        model: String,
        cost: TokenCost,
    }

    impl MockLlmClient {
        fn new(responses: Vec<Result<LlmResponse, LlmError>>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses),
                model: "mock-model".to_string(),
                cost: TokenCost::new(0.001, 0.002),
            }
        }
    }

    impl LlmClient for MockLlmClient {
        fn generate(
            &self,
            _prompt: &str,
            _temperature: f64,
        ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, LlmError>> + Send + '_>> {
            let result =
                self.responses
                    .lock()
                    .unwrap()
                    .pop()
                    .unwrap_or(Err(LlmError::InvalidResponse(
                        "No more responses".to_string(),
                    )));

            Box::pin(async move { result })
        }

        fn model_name(&self) -> &str {
            &self.model
        }

        fn cost_per_1k_tokens(&self) -> TokenCost {
            self.cost
        }
    }

    #[tokio::test]
    async fn test_mock_client_returns_response() {
        let response = LlmResponse {
            content: "mocked response".to_string(),
            tokens: TokenUsage::new(10, 20),
            latency: Duration::from_millis(50),
        };

        let client = MockLlmClient::new(vec![Ok(response.clone())]);
        let result = client.generate("test prompt", 0.5).await;

        assert!(result.is_ok());
        assert_eq!(result.unwrap().content, "mocked response");
    }

    #[tokio::test]
    async fn test_mock_client_returns_error() {
        let client = MockLlmClient::new(vec![Err(LlmError::Timeout)]);
        let result = client.generate("test", 0.0).await;

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), LlmError::Timeout);
    }

    #[test]
    fn test_trait_is_object_safe() {
        // This test verifies LlmClient can be used as a trait object
        fn _accepts_trait_object(_client: &dyn LlmClient) {}

        let client = MockLlmClient::new(vec![]);
        _accepts_trait_object(&client);
    }

    #[test]
    fn test_client_model_name() {
        let client = MockLlmClient::new(vec![]);
        assert_eq!(client.model_name(), "mock-model");
    }

    #[test]
    fn test_client_cost() {
        let client = MockLlmClient::new(vec![]);
        let cost = client.cost_per_1k_tokens();
        assert!((cost.input_per_1k - 0.001).abs() < f64::EPSILON);
        assert!((cost.output_per_1k - 0.002).abs() < f64::EPSILON);
    }
}
