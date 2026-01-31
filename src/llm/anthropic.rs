//! Anthropic LLM Client
//!
//! Client for Anthropic's Claude API. Supports Claude 3 models including
//! Haiku (lowest cost), Sonnet, and Opus.
//!
//! # Requirements
//!
//! - `ANTHROPIC_API_KEY` environment variable must be set
//!
//! # Example
//!
//! ```rust,ignore
//! let client = AnthropicClient::new("claude-3-haiku-20240307")?;
//! let response = client.generate("What is 2+2?", 0.0).await?;
//! println!("{}", response.content);
//! ```

use crate::llm::{LlmClient, LlmError, LlmResponse, TokenCost, TokenUsage};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, Instant};

/// Default Anthropic API URL
pub const DEFAULT_ANTHROPIC_URL: &str = "https://api.anthropic.com/v1";

/// Anthropic API version header
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic LLM client
#[derive(Debug, Clone)]
pub struct AnthropicClient {
    /// Base URL for Anthropic API
    base_url: String,
    /// Model name (e.g., "claude-3-haiku-20240307")
    model: String,
    /// Maximum tokens to generate
    max_tokens: usize,
    /// Request timeout
    timeout: Duration,
    /// HTTP client with auth headers
    client: Client,
    /// Cost per 1K tokens (depends on model)
    cost: TokenCost,
}

impl AnthropicClient {
    /// Create a new Anthropic client, reading API key from ANTHROPIC_API_KEY env var
    pub fn new(model: &str) -> Result<Self, LlmError> {
        let api_key = env::var("ANTHROPIC_API_KEY").map_err(|_| LlmError::ApiError {
            status: 401,
            message: "ANTHROPIC_API_KEY environment variable not set".to_string(),
        })?;

        Self::with_api_key(model, &api_key)
    }

    /// Create a new Anthropic client with explicit API key
    pub fn with_api_key(model: &str, api_key: &str) -> Result<Self, LlmError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(api_key)
                .map_err(|e| LlmError::InvalidResponse(format!("Invalid API key format: {}", e)))?,
        );
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static(ANTHROPIC_VERSION),
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let client = Client::builder()
            .default_headers(headers)
            .build()
            .map_err(|e| LlmError::NetworkError(format!("Failed to create client: {}", e)))?;

        Ok(Self {
            base_url: DEFAULT_ANTHROPIC_URL.to_string(),
            model: model.to_string(),
            max_tokens: 1024,
            timeout: Duration::from_secs(120),
            client,
            cost: Self::model_cost(model),
        })
    }

    /// Set custom base URL
    pub fn with_base_url(mut self, url: &str) -> Self {
        self.base_url = url.trim_end_matches('/').to_string();
        self
    }

    /// Set maximum tokens to generate
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set request timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Get cost based on model name (USD per 1K tokens as of 2024)
    fn model_cost(model: &str) -> TokenCost {
        match model {
            m if m.contains("haiku") => TokenCost::new(0.00025, 0.00125),
            m if m.contains("sonnet") => TokenCost::new(0.003, 0.015),
            m if m.contains("opus") => TokenCost::new(0.015, 0.075),
            _ => TokenCost::new(0.001, 0.005), // Default fallback
        }
    }
}

/// Request body for Anthropic messages API
#[derive(Debug, Serialize)]
struct MessagesRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: usize,
    temperature: f64,
}

#[derive(Debug, Serialize)]
struct Message {
    role: String,
    content: String,
}

/// Response from Anthropic messages API
#[derive(Debug, Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
    usage: Usage,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    text: String,
}

#[derive(Debug, Deserialize)]
struct Usage {
    input_tokens: usize,
    output_tokens: usize,
}

/// Error response from Anthropic
#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Deserialize)]
struct ErrorDetail {
    message: String,
}

impl LlmClient for AnthropicClient {
    fn generate(
        &self,
        prompt: &str,
        temperature: f64,
    ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, LlmError>> + Send + '_>> {
        let prompt = prompt.to_string();

        Box::pin(async move {
            let start = Instant::now();

            let request = MessagesRequest {
                model: self.model.clone(),
                messages: vec![Message {
                    role: "user".to_string(),
                    content: prompt,
                }],
                max_tokens: self.max_tokens,
                temperature,
            };

            let url = format!("{}/messages", self.base_url);

            let response = self
                .client
                .post(&url)
                .json(&request)
                .timeout(self.timeout)
                .send()
                .await
                .map_err(|e| {
                    if e.is_timeout() {
                        LlmError::Timeout
                    } else if e.is_connect() {
                        LlmError::NetworkError(format!("Connection failed: {}", e))
                    } else {
                        LlmError::NetworkError(e.to_string())
                    }
                })?;

            let status = response.status();

            // Handle rate limiting
            if status.as_u16() == 429 {
                let retry_after = response
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|s| s.parse::<u64>().ok())
                    .map(Duration::from_secs);

                return Err(LlmError::RateLimited { retry_after });
            }

            if !status.is_success() {
                let error_body: Result<ErrorResponse, _> = response.json().await;
                let message = error_body
                    .map(|e| e.error.message)
                    .unwrap_or_else(|_| "Unknown error".to_string());

                return Err(LlmError::ApiError {
                    status: status.as_u16(),
                    message,
                });
            }

            let messages_response: MessagesResponse = response.json().await.map_err(|e| {
                LlmError::InvalidResponse(format!("Failed to parse response: {}", e))
            })?;

            let content = messages_response
                .content
                .first()
                .map(|c| c.text.clone())
                .unwrap_or_default();

            let latency = start.elapsed();

            Ok(LlmResponse {
                content,
                tokens: TokenUsage::new(
                    messages_response.usage.input_tokens,
                    messages_response.usage.output_tokens,
                ),
                latency,
            })
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn cost_per_1k_tokens(&self) -> TokenCost {
        self.cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================
    // Client Configuration Tests
    // ==========================================

    #[test]
    fn test_client_with_api_key() {
        let client = AnthropicClient::with_api_key("claude-3-haiku-20240307", "test-key").unwrap();
        assert_eq!(client.model, "claude-3-haiku-20240307");
        assert_eq!(client.base_url, DEFAULT_ANTHROPIC_URL);
    }

    #[test]
    fn test_client_with_base_url() {
        let client = AnthropicClient::with_api_key("claude-3-sonnet-20240229", "test-key")
            .unwrap()
            .with_base_url("https://custom.anthropic.proxy/v1/");

        assert_eq!(client.base_url, "https://custom.anthropic.proxy/v1");
    }

    #[test]
    fn test_client_with_max_tokens() {
        let client = AnthropicClient::with_api_key("claude-3-haiku-20240307", "test-key")
            .unwrap()
            .with_max_tokens(2048);

        assert_eq!(client.max_tokens, 2048);
    }

    #[test]
    fn test_client_with_timeout() {
        let client = AnthropicClient::with_api_key("claude-3-haiku-20240307", "test-key")
            .unwrap()
            .with_timeout(Duration::from_secs(30));

        assert_eq!(client.timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_client_model_name() {
        let client = AnthropicClient::with_api_key("claude-3-opus-20240229", "test-key").unwrap();
        assert_eq!(client.model_name(), "claude-3-opus-20240229");
    }

    // ==========================================
    // Model Cost Tests
    // ==========================================

    #[test]
    fn test_model_cost_haiku() {
        let cost = AnthropicClient::model_cost("claude-3-haiku-20240307");
        assert!((cost.input_per_1k - 0.00025).abs() < 1e-10);
        assert!((cost.output_per_1k - 0.00125).abs() < 1e-10);
    }

    #[test]
    fn test_model_cost_sonnet() {
        let cost = AnthropicClient::model_cost("claude-3-sonnet-20240229");
        assert!((cost.input_per_1k - 0.003).abs() < 1e-10);
        assert!((cost.output_per_1k - 0.015).abs() < 1e-10);
    }

    #[test]
    fn test_model_cost_opus() {
        let cost = AnthropicClient::model_cost("claude-3-opus-20240229");
        assert!((cost.input_per_1k - 0.015).abs() < 1e-10);
        assert!((cost.output_per_1k - 0.075).abs() < 1e-10);
    }

    #[test]
    fn test_model_cost_unknown() {
        let cost = AnthropicClient::model_cost("unknown-model");
        assert!((cost.input_per_1k - 0.001).abs() < 1e-10);
    }

    // ==========================================
    // Request/Response Serialization Tests
    // ==========================================

    #[test]
    fn test_request_serialization() {
        let request = MessagesRequest {
            model: "claude-3-haiku-20240307".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            max_tokens: 1024,
            temperature: 0.5,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"model\":\"claude-3-haiku-20240307\""));
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"Hello\""));
        assert!(json.contains("\"max_tokens\":1024"));
        assert!(json.contains("\"temperature\":0.5"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "content": [
                {
                    "type": "text",
                    "text": "The answer is 4"
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5
            },
            "stop_reason": "end_turn"
        }"#;

        let response: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.content[0].text, "The answer is 4");
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
    }

    #[test]
    fn test_error_response_deserialization() {
        let json = r#"{
            "error": {
                "type": "rate_limit_error",
                "message": "Rate limit exceeded"
            }
        }"#;

        let response: ErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.error.message, "Rate limit exceeded");
    }

    // ==========================================
    // Integration Tests (require API key)
    // ==========================================

    #[tokio::test]
    #[ignore] // Run with: cargo test --ignored anthropic
    async fn test_anthropic_generate_integration() {
        let client = match AnthropicClient::new("claude-3-haiku-20240307") {
            Ok(c) => c,
            Err(e) => {
                println!("Skipping: {:?}", e);
                return;
            }
        };

        let result = client.generate("Say 'hello' and nothing else.", 0.0).await;

        match result {
            Ok(response) => {
                assert!(!response.content.is_empty());
                println!("Response: {}", response.content);
                println!("Tokens: {:?}", response.tokens);
                println!("Latency: {:?}", response.latency);
            }
            Err(e) => panic!("Error: {:?}", e),
        }
    }

    // ==========================================
    // Mock HTTP Server Tests
    // ==========================================

    #[tokio::test]
    async fn test_generate_success_with_mock() {
        let server = wiremock::MockServer::start().await;

        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/messages"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "content": [{"type": "text", "text": "4"}],
                    "usage": {"input_tokens": 5, "output_tokens": 2},
                    "stop_reason": "end_turn"
                })),
            )
            .mount(&server)
            .await;

        let client = AnthropicClient::with_api_key("claude-3-haiku-20240307", "test-key")
            .unwrap()
            .with_base_url(&server.uri());

        let response = client.generate("What is 2+2?", 0.0).await.unwrap();
        assert_eq!(response.content, "4");
        assert_eq!(response.tokens.input, 5);
        assert_eq!(response.tokens.output, 2);
    }

    #[tokio::test]
    async fn test_generate_rate_limited_with_retry_after() {
        let server = wiremock::MockServer::start().await;

        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(wiremock::ResponseTemplate::new(429).append_header("retry-after", "15"))
            .mount(&server)
            .await;

        let client = AnthropicClient::with_api_key("claude-3-haiku-20240307", "test-key")
            .unwrap()
            .with_base_url(&server.uri());

        let result = client.generate("test", 0.0).await;
        match result {
            Err(LlmError::RateLimited { retry_after }) => {
                assert_eq!(retry_after, Some(Duration::from_secs(15)));
            }
            other => panic!("Expected RateLimited, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_generate_rate_limited_without_retry_after() {
        let server = wiremock::MockServer::start().await;

        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(wiremock::ResponseTemplate::new(429))
            .mount(&server)
            .await;

        let client = AnthropicClient::with_api_key("claude-3-haiku-20240307", "test-key")
            .unwrap()
            .with_base_url(&server.uri());

        let result = client.generate("test", 0.0).await;
        match result {
            Err(LlmError::RateLimited { retry_after }) => {
                assert_eq!(retry_after, None);
            }
            other => panic!("Expected RateLimited, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_generate_api_error_with_body() {
        let server = wiremock::MockServer::start().await;

        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(wiremock::ResponseTemplate::new(400).set_body_json(
                serde_json::json!({"error": {"type": "invalid_request_error", "message": "Bad request"}}),
            ))
            .mount(&server)
            .await;

        let client = AnthropicClient::with_api_key("claude-3-haiku-20240307", "test-key")
            .unwrap()
            .with_base_url(&server.uri());

        let result = client.generate("test", 0.0).await;
        match result {
            Err(LlmError::ApiError { status, message }) => {
                assert_eq!(status, 400);
                assert_eq!(message, "Bad request");
            }
            other => panic!("Expected ApiError, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_generate_api_error_unparseable_body() {
        let server = wiremock::MockServer::start().await;

        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(wiremock::ResponseTemplate::new(500).set_body_string("internal error"))
            .mount(&server)
            .await;

        let client = AnthropicClient::with_api_key("claude-3-haiku-20240307", "test-key")
            .unwrap()
            .with_base_url(&server.uri());

        let result = client.generate("test", 0.0).await;
        match result {
            Err(LlmError::ApiError { status, message }) => {
                assert_eq!(status, 500);
                assert_eq!(message, "Unknown error");
            }
            other => panic!("Expected ApiError, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_generate_invalid_response_body() {
        let server = wiremock::MockServer::start().await;

        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_string("not json"))
            .mount(&server)
            .await;

        let client = AnthropicClient::with_api_key("claude-3-haiku-20240307", "test-key")
            .unwrap()
            .with_base_url(&server.uri());

        let result = client.generate("test", 0.0).await;
        assert!(matches!(result, Err(LlmError::InvalidResponse(_))));
    }

    #[tokio::test]
    async fn test_generate_timeout() {
        let server = wiremock::MockServer::start().await;

        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_delay(Duration::from_secs(5)))
            .mount(&server)
            .await;

        let client = AnthropicClient::with_api_key("claude-3-haiku-20240307", "test-key")
            .unwrap()
            .with_base_url(&server.uri())
            .with_timeout(Duration::from_millis(100));

        let result = client.generate("test", 0.0).await;
        assert!(matches!(result, Err(LlmError::Timeout)));
    }

    #[tokio::test]
    async fn test_generate_empty_content() {
        let server = wiremock::MockServer::start().await;

        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "content": [],
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                    "stop_reason": "end_turn"
                })),
            )
            .mount(&server)
            .await;

        let client = AnthropicClient::with_api_key("claude-3-haiku-20240307", "test-key")
            .unwrap()
            .with_base_url(&server.uri());

        let response = client.generate("test", 0.0).await.unwrap();
        assert_eq!(response.content, "");
    }

    #[test]
    fn test_new_without_env_var() {
        let original = env::var("ANTHROPIC_API_KEY").ok();
        unsafe { env::remove_var("ANTHROPIC_API_KEY") };

        let result = AnthropicClient::new("claude-3-haiku-20240307");
        assert!(matches!(
            result,
            Err(LlmError::ApiError { status: 401, .. })
        ));

        if let Some(key) = original {
            unsafe { env::set_var("ANTHROPIC_API_KEY", key) };
        }
    }
}
