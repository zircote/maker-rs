//! OpenAI LLM Client
//!
//! Client for OpenAI's chat completions API. Supports GPT-4, GPT-3.5-turbo,
//! and other chat models.
//!
//! # Requirements
//!
//! - `OPENAI_API_KEY` environment variable must be set
//!
//! # Example
//!
//! ```rust,ignore
//! let client = OpenAiClient::new("gpt-4o-mini")?;
//! let response = client.generate("What is 2+2?", 0.0).await?;
//! println!("{}", response.content);
//! ```

use crate::llm::{LlmClient, LlmError, LlmResponse, TokenCost, TokenUsage};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::env;
use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, Instant};

/// Default OpenAI API URL
pub const DEFAULT_OPENAI_URL: &str = "https://api.openai.com/v1";

/// OpenAI LLM client
#[derive(Debug, Clone)]
pub struct OpenAiClient {
    /// Base URL for OpenAI API
    base_url: String,
    /// Model name (e.g., "gpt-4o-mini", "gpt-4o")
    model: String,
    /// Request timeout
    timeout: Duration,
    /// HTTP client with auth headers
    client: Client,
    /// Cost per 1K tokens (depends on model)
    cost: TokenCost,
}

impl OpenAiClient {
    /// Create a new OpenAI client, reading API key from OPENAI_API_KEY env var
    pub fn new(model: &str) -> Result<Self, LlmError> {
        let api_key = env::var("OPENAI_API_KEY").map_err(|_| LlmError::ApiError {
            status: 401,
            message: "OPENAI_API_KEY environment variable not set".to_string(),
        })?;

        Self::with_api_key(model, &api_key)
    }

    /// Create a new OpenAI client with explicit API key
    pub fn with_api_key(model: &str, api_key: &str) -> Result<Self, LlmError> {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", api_key))
                .map_err(|e| LlmError::InvalidResponse(format!("Invalid API key format: {}", e)))?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let client = Client::builder()
            .default_headers(headers)
            .build()
            .map_err(|e| LlmError::NetworkError(format!("Failed to create client: {}", e)))?;

        Ok(Self {
            base_url: DEFAULT_OPENAI_URL.to_string(),
            model: model.to_string(),
            timeout: Duration::from_secs(120),
            client,
            cost: Self::model_cost(model),
        })
    }

    /// Set custom base URL (for proxies or compatible APIs)
    pub fn with_base_url(mut self, url: &str) -> Self {
        self.base_url = url.trim_end_matches('/').to_string();
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
            m if m.starts_with("gpt-4o-mini") => TokenCost::new(0.00015, 0.0006),
            m if m.starts_with("gpt-4o") => TokenCost::new(0.005, 0.015),
            m if m.starts_with("gpt-4-turbo") => TokenCost::new(0.01, 0.03),
            m if m.starts_with("gpt-4") => TokenCost::new(0.03, 0.06),
            m if m.starts_with("gpt-3.5-turbo") => TokenCost::new(0.0005, 0.0015),
            _ => TokenCost::new(0.001, 0.002), // Default fallback
        }
    }
}

/// Request body for OpenAI chat completions
#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f64,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

/// Response from OpenAI chat completions
#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    usage: ChatUsage,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
}

/// Error response from OpenAI
#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Deserialize)]
struct ErrorDetail {
    message: String,
}

impl LlmClient for OpenAiClient {
    fn generate(
        &self,
        prompt: &str,
        temperature: f64,
    ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, LlmError>> + Send + '_>> {
        let prompt = prompt.to_string();

        Box::pin(async move {
            let start = Instant::now();

            let request = ChatRequest {
                model: self.model.clone(),
                messages: vec![ChatMessage {
                    role: "user".to_string(),
                    content: prompt,
                }],
                temperature,
            };

            let url = format!("{}/chat/completions", self.base_url);

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

            let chat_response: ChatResponse = response.json().await.map_err(|e| {
                LlmError::InvalidResponse(format!("Failed to parse response: {}", e))
            })?;

            let content = chat_response
                .choices
                .first()
                .map(|c| c.message.content.clone())
                .unwrap_or_default();

            let latency = start.elapsed();

            Ok(LlmResponse {
                content,
                tokens: TokenUsage::new(
                    chat_response.usage.prompt_tokens,
                    chat_response.usage.completion_tokens,
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
        let client = OpenAiClient::with_api_key("gpt-4o-mini", "test-key").unwrap();
        assert_eq!(client.model, "gpt-4o-mini");
        assert_eq!(client.base_url, DEFAULT_OPENAI_URL);
    }

    #[test]
    fn test_client_with_base_url() {
        let client = OpenAiClient::with_api_key("gpt-4o", "test-key")
            .unwrap()
            .with_base_url("https://custom.openai.proxy/v1/");

        assert_eq!(client.base_url, "https://custom.openai.proxy/v1");
    }

    #[test]
    fn test_client_with_timeout() {
        let client = OpenAiClient::with_api_key("gpt-4o", "test-key")
            .unwrap()
            .with_timeout(Duration::from_secs(30));

        assert_eq!(client.timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_client_model_name() {
        let client = OpenAiClient::with_api_key("gpt-3.5-turbo", "test-key").unwrap();
        assert_eq!(client.model_name(), "gpt-3.5-turbo");
    }

    // ==========================================
    // Model Cost Tests
    // ==========================================

    #[test]
    fn test_model_cost_gpt4o_mini() {
        let cost = OpenAiClient::model_cost("gpt-4o-mini");
        assert!((cost.input_per_1k - 0.00015).abs() < 1e-10);
        assert!((cost.output_per_1k - 0.0006).abs() < 1e-10);
    }

    #[test]
    fn test_model_cost_gpt4o() {
        let cost = OpenAiClient::model_cost("gpt-4o");
        assert!((cost.input_per_1k - 0.005).abs() < 1e-10);
        assert!((cost.output_per_1k - 0.015).abs() < 1e-10);
    }

    #[test]
    fn test_model_cost_gpt4() {
        let cost = OpenAiClient::model_cost("gpt-4");
        assert!((cost.input_per_1k - 0.03).abs() < 1e-10);
        assert!((cost.output_per_1k - 0.06).abs() < 1e-10);
    }

    #[test]
    fn test_model_cost_gpt35() {
        let cost = OpenAiClient::model_cost("gpt-3.5-turbo");
        assert!((cost.input_per_1k - 0.0005).abs() < 1e-10);
    }

    #[test]
    fn test_model_cost_unknown() {
        let cost = OpenAiClient::model_cost("unknown-model");
        // Should use default pricing
        assert!((cost.input_per_1k - 0.001).abs() < 1e-10);
    }

    // ==========================================
    // Request/Response Serialization Tests
    // ==========================================

    #[test]
    fn test_request_serialization() {
        let request = ChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            temperature: 0.5,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"model\":\"gpt-4o\""));
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"content\":\"Hello\""));
        assert!(json.contains("\"temperature\":0.5"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The answer is 4"
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }"#;

        let response: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.choices[0].message.content, "The answer is 4");
        assert_eq!(response.usage.prompt_tokens, 10);
        assert_eq!(response.usage.completion_tokens, 5);
    }

    #[test]
    fn test_error_response_deserialization() {
        let json = r#"{
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_error"
            }
        }"#;

        let response: ErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.error.message, "Rate limit exceeded");
    }

    // ==========================================
    // Integration Tests (require API key)
    // ==========================================

    #[tokio::test]
    #[ignore] // Run with: cargo test --ignored openai
    async fn test_openai_generate_integration() {
        let client = match OpenAiClient::new("gpt-4o-mini") {
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
            .and(wiremock::matchers::path("/chat/completions"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "choices": [{"message": {"role": "assistant", "content": "4"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
                })),
            )
            .mount(&server)
            .await;

        let client = OpenAiClient::with_api_key("gpt-4o-mini", "test-key")
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
            .respond_with(wiremock::ResponseTemplate::new(429).append_header("retry-after", "30"))
            .mount(&server)
            .await;

        let client = OpenAiClient::with_api_key("gpt-4o-mini", "test-key")
            .unwrap()
            .with_base_url(&server.uri());

        let result = client.generate("test", 0.0).await;
        match result {
            Err(LlmError::RateLimited { retry_after }) => {
                assert_eq!(retry_after, Some(Duration::from_secs(30)));
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

        let client = OpenAiClient::with_api_key("gpt-4o-mini", "test-key")
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
                serde_json::json!({"error": {"message": "Bad request", "type": "invalid_request_error"}}),
            ))
            .mount(&server)
            .await;

        let client = OpenAiClient::with_api_key("gpt-4o-mini", "test-key")
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

        let client = OpenAiClient::with_api_key("gpt-4o-mini", "test-key")
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

        let client = OpenAiClient::with_api_key("gpt-4o-mini", "test-key")
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

        let client = OpenAiClient::with_api_key("gpt-4o-mini", "test-key")
            .unwrap()
            .with_base_url(&server.uri())
            .with_timeout(Duration::from_millis(100));

        let result = client.generate("test", 0.0).await;
        assert!(matches!(result, Err(LlmError::Timeout)));
    }

    #[tokio::test]
    async fn test_generate_empty_choices() {
        let server = wiremock::MockServer::start().await;

        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "choices": [],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5}
                })),
            )
            .mount(&server)
            .await;

        let client = OpenAiClient::with_api_key("gpt-4o-mini", "test-key")
            .unwrap()
            .with_base_url(&server.uri());

        let response = client.generate("test", 0.0).await.unwrap();
        assert_eq!(response.content, "");
    }

    #[test]
    fn test_model_cost_gpt4_turbo() {
        let cost = OpenAiClient::model_cost("gpt-4-turbo-preview");
        assert!((cost.input_per_1k - 0.01).abs() < 1e-10);
        assert!((cost.output_per_1k - 0.03).abs() < 1e-10);
    }

    #[test]
    fn test_new_without_env_var() {
        let original = env::var("OPENAI_API_KEY").ok();
        unsafe { env::remove_var("OPENAI_API_KEY") };

        let result = OpenAiClient::new("gpt-4o-mini");
        assert!(matches!(
            result,
            Err(LlmError::ApiError { status: 401, .. })
        ));

        if let Some(key) = original {
            unsafe { env::set_var("OPENAI_API_KEY", key) };
        }
    }
}
