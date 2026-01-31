//! Ollama LLM Client
//!
//! Client for local Ollama inference. Useful for development and testing
//! without cloud API costs.
//!
//! # Requirements
//!
//! - Ollama must be running locally (default: http://localhost:11434)
//! - A model must be pulled (e.g., `ollama pull llama2`)
//!
//! # Example
//!
//! ```rust,ignore
//! let client = OllamaClient::new("llama2");
//! let response = client.generate("What is 2+2?", 0.0).await?;
//! println!("{}", response.content);
//! ```

use crate::llm::{LlmClient, LlmError, LlmResponse, TokenCost, TokenUsage};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::pin::Pin;
use std::time::{Duration, Instant};

/// Default Ollama server URL
pub const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

/// Ollama LLM client for local inference
#[derive(Debug, Clone)]
pub struct OllamaClient {
    /// Base URL for Ollama API
    base_url: String,
    /// Model name (e.g., "llama2", "mistral", "codellama")
    model: String,
    /// Request timeout
    timeout: Duration,
    /// HTTP client
    client: Client,
}

impl OllamaClient {
    /// Create a new Ollama client with default URL
    pub fn new(model: &str) -> Self {
        Self::with_url(model, DEFAULT_OLLAMA_URL)
    }

    /// Create a new Ollama client with custom URL
    pub fn with_url(model: &str, base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            timeout: Duration::from_secs(120),
            client: Client::new(),
        }
    }

    /// Set request timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Get the base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
}

/// Request body for Ollama /api/generate endpoint
#[derive(Debug, Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    temperature: f64,
}

/// Response from Ollama /api/generate endpoint
#[derive(Debug, Deserialize)]
struct OllamaResponse {
    response: String,
    #[serde(default)]
    prompt_eval_count: Option<usize>,
    #[serde(default)]
    eval_count: Option<usize>,
}

impl LlmClient for OllamaClient {
    fn generate(
        &self,
        prompt: &str,
        temperature: f64,
    ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, LlmError>> + Send + '_>> {
        let prompt = prompt.to_string();

        Box::pin(async move {
            let start = Instant::now();

            let request = OllamaRequest {
                model: self.model.clone(),
                prompt,
                stream: false,
                options: OllamaOptions { temperature },
            };

            let url = format!("{}/api/generate", self.base_url);

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
            if !status.is_success() {
                let message = response.text().await.unwrap_or_default();
                return Err(LlmError::ApiError {
                    status: status.as_u16(),
                    message,
                });
            }

            let ollama_response: OllamaResponse = response.json().await.map_err(|e| {
                LlmError::InvalidResponse(format!("Failed to parse response: {}", e))
            })?;

            let latency = start.elapsed();

            Ok(LlmResponse {
                content: ollama_response.response,
                tokens: TokenUsage::new(
                    ollama_response.prompt_eval_count.unwrap_or(0),
                    ollama_response.eval_count.unwrap_or(0),
                ),
                latency,
            })
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    fn cost_per_1k_tokens(&self) -> TokenCost {
        // Ollama is free (local inference)
        TokenCost::new(0.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================
    // Client Configuration Tests
    // ==========================================

    #[test]
    fn test_client_new() {
        let client = OllamaClient::new("llama2");
        assert_eq!(client.model, "llama2");
        assert_eq!(client.base_url, DEFAULT_OLLAMA_URL);
    }

    #[test]
    fn test_client_with_url() {
        let client = OllamaClient::with_url("mistral", "http://192.168.1.100:11434/");
        assert_eq!(client.model, "mistral");
        assert_eq!(client.base_url, "http://192.168.1.100:11434");
    }

    #[test]
    fn test_client_with_timeout() {
        let client = OllamaClient::new("llama2").with_timeout(Duration::from_secs(30));
        assert_eq!(client.timeout, Duration::from_secs(30));
    }

    #[test]
    fn test_client_model_name() {
        let client = OllamaClient::new("codellama");
        assert_eq!(client.model_name(), "codellama");
    }

    #[test]
    fn test_client_cost_is_free() {
        let client = OllamaClient::new("llama2");
        let cost = client.cost_per_1k_tokens();
        assert!((cost.input_per_1k - 0.0).abs() < f64::EPSILON);
        assert!((cost.output_per_1k - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_client_base_url_accessor() {
        let client = OllamaClient::new("llama2");
        assert_eq!(client.base_url(), DEFAULT_OLLAMA_URL);
    }

    // ==========================================
    // Request/Response Serialization Tests
    // ==========================================

    #[test]
    fn test_request_serialization() {
        let request = OllamaRequest {
            model: "llama2".to_string(),
            prompt: "Hello".to_string(),
            stream: false,
            options: OllamaOptions { temperature: 0.5 },
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"model\":\"llama2\""));
        assert!(json.contains("\"prompt\":\"Hello\""));
        assert!(json.contains("\"stream\":false"));
        assert!(json.contains("\"temperature\":0.5"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "response": "The answer is 4",
            "prompt_eval_count": 10,
            "eval_count": 5
        }"#;

        let response: OllamaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.response, "The answer is 4");
        assert_eq!(response.prompt_eval_count, Some(10));
        assert_eq!(response.eval_count, Some(5));
    }

    #[test]
    fn test_response_deserialization_missing_optional() {
        let json = r#"{"response": "Hello"}"#;

        let response: OllamaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.response, "Hello");
        assert_eq!(response.prompt_eval_count, None);
        assert_eq!(response.eval_count, None);
    }

    // ==========================================
    // Integration Tests (require Ollama running)
    // ==========================================

    #[tokio::test]
    #[ignore] // Run with: cargo test --ignored ollama
    async fn test_ollama_generate_integration() {
        let client = OllamaClient::new("llama2");
        let result = client.generate("Say 'hello' and nothing else.", 0.0).await;

        match result {
            Ok(response) => {
                assert!(!response.content.is_empty());
                println!("Response: {}", response.content);
                println!("Tokens: {:?}", response.tokens);
                println!("Latency: {:?}", response.latency);
            }
            Err(LlmError::NetworkError(msg)) => {
                println!("Ollama not running: {}", msg);
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    // ==========================================
    // Mock HTTP Server Tests
    // ==========================================

    #[tokio::test]
    async fn test_generate_success_with_mock() {
        let server = wiremock::MockServer::start().await;

        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/api/generate"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "response": "4",
                    "prompt_eval_count": 5,
                    "eval_count": 2
                })),
            )
            .mount(&server)
            .await;

        let client = OllamaClient::with_url("llama2", &server.uri());
        let response = client.generate("What is 2+2?", 0.0).await.unwrap();
        assert_eq!(response.content, "4");
        assert_eq!(response.tokens.input, 5);
        assert_eq!(response.tokens.output, 2);
    }

    #[tokio::test]
    async fn test_generate_success_missing_token_counts() {
        let server = wiremock::MockServer::start().await;

        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"response": "hello"})),
            )
            .mount(&server)
            .await;

        let client = OllamaClient::with_url("llama2", &server.uri());
        let response = client.generate("test", 0.0).await.unwrap();
        assert_eq!(response.content, "hello");
        assert_eq!(response.tokens.input, 0);
        assert_eq!(response.tokens.output, 0);
    }

    #[tokio::test]
    async fn test_generate_api_error() {
        let server = wiremock::MockServer::start().await;

        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(wiremock::ResponseTemplate::new(404).set_body_string("model not found"))
            .mount(&server)
            .await;

        let client = OllamaClient::with_url("nonexistent", &server.uri());
        let result = client.generate("test", 0.0).await;
        match result {
            Err(LlmError::ApiError { status, message }) => {
                assert_eq!(status, 404);
                assert_eq!(message, "model not found");
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

        let client = OllamaClient::with_url("llama2", &server.uri());
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

        let client = OllamaClient::with_url("llama2", &server.uri())
            .with_timeout(Duration::from_millis(100));
        let result = client.generate("test", 0.0).await;
        assert!(matches!(result, Err(LlmError::Timeout)));
    }

    #[tokio::test]
    async fn test_ollama_connection_refused() {
        // Use a port that's definitely not running Ollama
        let client = OllamaClient::with_url("llama2", "http://localhost:59999")
            .with_timeout(Duration::from_millis(500));

        let result = client.generate("test", 0.0).await;

        assert!(result.is_err());
        match result {
            Err(LlmError::NetworkError(msg)) => {
                assert!(
                    msg.contains("Connection") || msg.contains("connect"),
                    "Expected connection error, got: {}",
                    msg
                );
            }
            Err(LlmError::Timeout) => {
                // Also acceptable
            }
            Err(e) => panic!("Unexpected error type: {:?}", e),
            Ok(_) => panic!("Expected error"),
        }
    }
}
