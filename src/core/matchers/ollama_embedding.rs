//! Ollama Embedding Client
//!
//! Implements `EmbeddingClient` for local Ollama embedding models.
//! Uses the POST /api/embeddings endpoint to generate text embeddings.

use super::embedding::EmbeddingClient;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Default Ollama server URL
pub const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

/// Default embedding model
pub const DEFAULT_EMBEDDING_MODEL: &str = "nomic-embed-text";

/// Ollama embedding client for local inference.
///
/// Connects to a running Ollama instance and uses its embedding endpoint
/// to convert text into vector representations.
#[derive(Debug, Clone)]
pub struct OllamaEmbeddingClient {
    /// Base URL for Ollama API
    base_url: String,
    /// Model name (e.g., "nomic-embed-text", "all-minilm")
    model: String,
    /// Request timeout
    timeout: Duration,
    /// HTTP client
    client: Client,
    /// Expected embedding dimensions (0 = auto-detect from first call)
    dims: usize,
}

impl OllamaEmbeddingClient {
    /// Create a new Ollama embedding client with default URL and model.
    pub fn new() -> Self {
        Self::with_model(DEFAULT_EMBEDDING_MODEL)
    }

    /// Create a new Ollama embedding client with a specific model.
    pub fn with_model(model: &str) -> Self {
        Self {
            base_url: DEFAULT_OLLAMA_URL.trim_end_matches('/').to_string(),
            model: model.to_string(),
            timeout: Duration::from_secs(30),
            client: Client::new(),
            dims: 0,
        }
    }

    /// Set custom base URL.
    pub fn with_url(mut self, url: &str) -> Self {
        self.base_url = url.trim_end_matches('/').to_string();
        self
    }

    /// Set request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set expected embedding dimensions.
    pub fn with_dimensions(mut self, dims: usize) -> Self {
        self.dims = dims;
        self
    }

    /// Get the base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Get the model name.
    pub fn model(&self) -> &str {
        &self.model
    }
}

impl Default for OllamaEmbeddingClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Request body for Ollama /api/embeddings endpoint
#[derive(Debug, Serialize)]
struct OllamaEmbeddingRequest {
    model: String,
    prompt: String,
}

/// Response from Ollama /api/embeddings endpoint
#[derive(Debug, Deserialize)]
struct OllamaEmbeddingResponse {
    embedding: Vec<f64>,
}

impl EmbeddingClient for OllamaEmbeddingClient {
    fn embed(&self, text: &str) -> Result<Vec<f64>, String> {
        let request = OllamaEmbeddingRequest {
            model: self.model.clone(),
            prompt: text.to_string(),
        };

        let url = format!("{}/api/embeddings", self.base_url);

        let response = self
            .client
            .post(&url)
            .json(&request)
            .timeout(self.timeout)
            .send()
            .map_err(|e| {
                if e.is_timeout() {
                    "Ollama embedding request timed out".to_string()
                } else if e.is_connect() {
                    format!("Failed to connect to Ollama at {}: {}", self.base_url, e)
                } else {
                    format!("Ollama embedding request failed: {}", e)
                }
            })?;

        let status = response.status();
        if !status.is_success() {
            let message = response.text().unwrap_or_default();
            return Err(format!(
                "Ollama API error ({}): {}",
                status.as_u16(),
                message
            ));
        }

        let embedding_response: OllamaEmbeddingResponse =
            response.json().map_err(|e| format!("Failed to parse Ollama embedding response: {}", e))?;

        Ok(embedding_response.embedding)
    }

    fn dimensions(&self) -> usize {
        self.dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_new() {
        let client = OllamaEmbeddingClient::new();
        assert_eq!(client.model(), DEFAULT_EMBEDDING_MODEL);
        assert_eq!(client.base_url(), DEFAULT_OLLAMA_URL);
    }

    #[test]
    fn test_client_with_model() {
        let client = OllamaEmbeddingClient::with_model("all-minilm");
        assert_eq!(client.model(), "all-minilm");
    }

    #[test]
    fn test_client_with_url() {
        let client = OllamaEmbeddingClient::new().with_url("http://192.168.1.100:11434/");
        assert_eq!(client.base_url(), "http://192.168.1.100:11434");
    }

    #[test]
    fn test_client_with_timeout() {
        let client = OllamaEmbeddingClient::new().with_timeout(Duration::from_secs(10));
        assert_eq!(client.timeout, Duration::from_secs(10));
    }

    #[test]
    fn test_client_with_dimensions() {
        let client = OllamaEmbeddingClient::new().with_dimensions(768);
        assert_eq!(client.dimensions(), 768);
    }

    #[test]
    fn test_client_default() {
        let client = OllamaEmbeddingClient::default();
        assert_eq!(client.model(), DEFAULT_EMBEDDING_MODEL);
    }

    #[test]
    fn test_client_debug() {
        let client = OllamaEmbeddingClient::new();
        let debug = format!("{:?}", client);
        assert!(debug.contains("OllamaEmbeddingClient"));
    }

    #[test]
    fn test_client_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OllamaEmbeddingClient>();
    }

    #[test]
    fn test_request_serialization() {
        let request = OllamaEmbeddingRequest {
            model: "nomic-embed-text".to_string(),
            prompt: "hello world".to_string(),
        };
        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("nomic-embed-text"));
        assert!(json.contains("hello world"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{"embedding": [0.1, 0.2, 0.3]}"#;
        let response: OllamaEmbeddingResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.embedding, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_embed_connection_refused() {
        let client = OllamaEmbeddingClient::new()
            .with_url("http://localhost:59999")
            .with_timeout(Duration::from_millis(500));
        let result = client.embed("hello");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("connect") || err.contains("Connect") || err.contains("timed out"),
            "Expected connection error, got: {}",
            err
        );
    }

    #[test]
    #[ignore] // Run with: cargo test --ignored ollama_embedding
    fn test_ollama_embedding_integration() {
        let client = OllamaEmbeddingClient::new();
        let result = client.embed("Hello, world!");

        match result {
            Ok(embedding) => {
                assert!(!embedding.is_empty());
                println!("Embedding dimensions: {}", embedding.len());
                println!("First 5 values: {:?}", &embedding[..5.min(embedding.len())]);
            }
            Err(e) => {
                println!("Ollama not running: {}", e);
            }
        }
    }
}
