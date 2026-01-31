//! OpenAI Embedding Client
//!
//! Implements `EmbeddingClient` for OpenAI's text-embedding API.
//! Uses the POST /v1/embeddings endpoint (text-embedding-3-small by default).

use super::embedding::EmbeddingClient;
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::env;
use std::time::Duration;

/// Default OpenAI API URL
pub const DEFAULT_OPENAI_URL: &str = "https://api.openai.com/v1";

/// Default embedding model
pub const DEFAULT_EMBEDDING_MODEL: &str = "text-embedding-3-small";

/// OpenAI embedding client.
///
/// Connects to OpenAI's embedding API to convert text into vector representations.
/// Requires an API key via constructor or `OPENAI_API_KEY` environment variable.
#[derive(Debug, Clone)]
pub struct OpenAiEmbeddingClient {
    /// Base URL for OpenAI API
    base_url: String,
    /// Model name (e.g., "text-embedding-3-small", "text-embedding-3-large")
    model: String,
    /// Request timeout
    timeout: Duration,
    /// HTTP client with auth headers
    client: Client,
    /// Expected embedding dimensions
    dims: usize,
}

impl OpenAiEmbeddingClient {
    /// Create a new OpenAI embedding client, reading API key from `OPENAI_API_KEY` env var.
    pub fn new() -> Result<Self, String> {
        let api_key = env::var("OPENAI_API_KEY")
            .map_err(|_| "OPENAI_API_KEY environment variable not set".to_string())?;
        Self::with_api_key(&api_key)
    }

    /// Create a new OpenAI embedding client with an explicit API key.
    pub fn with_api_key(api_key: &str) -> Result<Self, String> {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", api_key))
                .map_err(|e| format!("Invalid API key format: {}", e))?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let client = Client::builder()
            .default_headers(headers)
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        Ok(Self {
            base_url: DEFAULT_OPENAI_URL.to_string(),
            model: DEFAULT_EMBEDDING_MODEL.to_string(),
            timeout: Duration::from_secs(30),
            client,
            dims: 1536, // text-embedding-3-small default
        })
    }

    /// Set the embedding model.
    pub fn with_model(mut self, model: &str) -> Self {
        self.dims = model_dimensions(model);
        self.model = model.to_string();
        self
    }

    /// Set custom base URL (for proxies or compatible APIs).
    pub fn with_base_url(mut self, url: &str) -> Self {
        self.base_url = url.trim_end_matches('/').to_string();
        self
    }

    /// Set request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Get the model name.
    pub fn model(&self) -> &str {
        &self.model
    }
}

/// Get expected dimensions for an OpenAI embedding model.
fn model_dimensions(model: &str) -> usize {
    match model {
        "text-embedding-3-small" => 1536,
        "text-embedding-3-large" => 3072,
        "text-embedding-ada-002" => 1536,
        _ => 1536, // default fallback
    }
}

/// Request body for OpenAI /v1/embeddings endpoint
#[derive(Debug, Serialize)]
struct EmbeddingRequest {
    model: String,
    input: String,
}

/// Response from OpenAI /v1/embeddings endpoint
#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f64>,
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

impl EmbeddingClient for OpenAiEmbeddingClient {
    fn embed(&self, text: &str) -> Result<Vec<f64>, String> {
        let request = EmbeddingRequest {
            model: self.model.clone(),
            input: text.to_string(),
        };

        let url = format!("{}/embeddings", self.base_url);

        let response = self
            .client
            .post(&url)
            .json(&request)
            .timeout(self.timeout)
            .send()
            .map_err(|e| {
                if e.is_timeout() {
                    "OpenAI embedding request timed out".to_string()
                } else if e.is_connect() {
                    format!("Failed to connect to OpenAI API: {}", e)
                } else {
                    format!("OpenAI embedding request failed: {}", e)
                }
            })?;

        let status = response.status();

        if status.as_u16() == 429 {
            return Err("OpenAI rate limit exceeded".to_string());
        }

        if !status.is_success() {
            let error_body: Result<ErrorResponse, _> = response.json();
            let message = error_body
                .map(|e| e.error.message)
                .unwrap_or_else(|_| "Unknown error".to_string());
            return Err(format!("OpenAI API error ({}): {}", status.as_u16(), message));
        }

        let embedding_response: EmbeddingResponse = response
            .json()
            .map_err(|e| format!("Failed to parse OpenAI embedding response: {}", e))?;

        embedding_response
            .data
            .into_iter()
            .next()
            .map(|d| d.embedding)
            .ok_or_else(|| "OpenAI returned no embedding data".to_string())
    }

    fn dimensions(&self) -> usize {
        self.dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_with_api_key() {
        let client = OpenAiEmbeddingClient::with_api_key("test-key").unwrap();
        assert_eq!(client.model(), DEFAULT_EMBEDDING_MODEL);
        assert_eq!(client.dimensions(), 1536);
    }

    #[test]
    fn test_client_with_model() {
        let client = OpenAiEmbeddingClient::with_api_key("test-key")
            .unwrap()
            .with_model("text-embedding-3-large");
        assert_eq!(client.model(), "text-embedding-3-large");
        assert_eq!(client.dimensions(), 3072);
    }

    #[test]
    fn test_client_with_base_url() {
        let client = OpenAiEmbeddingClient::with_api_key("test-key")
            .unwrap()
            .with_base_url("https://proxy.example.com/v1/");
        assert_eq!(client.base_url, "https://proxy.example.com/v1");
    }

    #[test]
    fn test_client_with_timeout() {
        let client = OpenAiEmbeddingClient::with_api_key("test-key")
            .unwrap()
            .with_timeout(Duration::from_secs(10));
        assert_eq!(client.timeout, Duration::from_secs(10));
    }

    #[test]
    fn test_model_dimensions() {
        assert_eq!(model_dimensions("text-embedding-3-small"), 1536);
        assert_eq!(model_dimensions("text-embedding-3-large"), 3072);
        assert_eq!(model_dimensions("text-embedding-ada-002"), 1536);
        assert_eq!(model_dimensions("unknown"), 1536);
    }

    #[test]
    fn test_client_debug() {
        let client = OpenAiEmbeddingClient::with_api_key("test-key").unwrap();
        let debug = format!("{:?}", client);
        assert!(debug.contains("OpenAiEmbeddingClient"));
    }

    #[test]
    fn test_client_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OpenAiEmbeddingClient>();
    }

    #[test]
    fn test_request_serialization() {
        let request = EmbeddingRequest {
            model: "text-embedding-3-small".to_string(),
            input: "hello world".to_string(),
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("text-embedding-3-small"));
        assert!(json.contains("hello world"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{"data": [{"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"}], "model": "text-embedding-3-small", "usage": {"prompt_tokens": 3, "total_tokens": 3}}"#;
        let response: EmbeddingResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].embedding, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_error_response_deserialization() {
        let json = r#"{"error": {"message": "Invalid API key", "type": "invalid_request_error"}}"#;
        let response: ErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.error.message, "Invalid API key");
    }

    #[test]
    #[ignore] // Run with: cargo test --ignored openai_embedding
    fn test_openai_embedding_integration() {
        let client = match OpenAiEmbeddingClient::new() {
            Ok(c) => c,
            Err(e) => {
                println!("Skipping: {}", e);
                return;
            }
        };

        let result = client.embed("Hello, world!");
        match result {
            Ok(embedding) => {
                assert!(!embedding.is_empty());
                println!("Embedding dimensions: {}", embedding.len());
                println!("First 5 values: {:?}", &embedding[..5.min(embedding.len())]);
            }
            Err(e) => {
                println!("API error: {}", e);
            }
        }
    }
}
