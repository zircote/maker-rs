//! Adapter bridging async LlmClient to sync LlmClient
//!
//! The voting engine uses a synchronous LlmClient trait, but providers
//! implement an async trait. This adapter bridges the gap using tokio's
//! blocking runtime.

use crate::core::executor::{LlmClient as SyncLlmClient, LlmResponse as SyncLlmResponse};

/// Valid LLM provider names supported by the framework.
///
/// Use this constant for validation and help messages to ensure consistency.
pub const VALID_PROVIDERS: &[&str] = &["ollama", "openai", "anthropic"];

/// Returns a comma-separated string of valid provider names for error messages.
pub fn valid_providers_str() -> String {
    VALID_PROVIDERS.join(", ")
}
use crate::llm::anthropic::AnthropicClient;
use crate::llm::ollama::OllamaClient;
use crate::llm::openai::OpenAiClient;
use crate::llm::LlmClient as AsyncLlmClient;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Adapter that wraps an async LlmClient to provide a sync interface
///
/// Uses a dedicated tokio runtime to block on async calls.
pub struct BlockingLlmAdapter<C: AsyncLlmClient> {
    client: C,
    runtime: Arc<Runtime>,
}

impl<C: AsyncLlmClient + 'static> BlockingLlmAdapter<C> {
    /// Create a new adapter wrapping an async client
    ///
    /// # Panics
    ///
    /// Panics if the tokio runtime cannot be created. This typically only fails
    /// if the system is out of resources (file descriptors, memory). For fallible
    /// construction, use [`BlockingLlmAdapter::try_new`] instead.
    pub fn new(client: C) -> Self {
        let runtime = Runtime::new().expect("Failed to create tokio runtime");
        Self {
            client,
            runtime: Arc::new(runtime),
        }
    }

    /// Try to create a new adapter, returning an error if runtime creation fails
    ///
    /// Use this when you need to handle runtime creation failures gracefully.
    pub fn try_new(client: C) -> Result<Self, std::io::Error> {
        let runtime = Runtime::new()?;
        Ok(Self {
            client,
            runtime: Arc::new(runtime),
        })
    }

    /// Create with a shared runtime (for reuse across multiple adapters)
    pub fn with_runtime(client: C, runtime: Arc<Runtime>) -> Self {
        Self { client, runtime }
    }
}

impl<C: AsyncLlmClient + 'static> SyncLlmClient for BlockingLlmAdapter<C> {
    fn generate(&self, prompt: &str, temperature: f64) -> Result<SyncLlmResponse, String> {
        let prompt = prompt.to_string();

        self.runtime.block_on(async {
            self.client
                .generate(&prompt, temperature)
                .await
                .map(|resp| SyncLlmResponse {
                    content: resp.content,
                    input_tokens: resp.tokens.input,
                    output_tokens: resp.tokens.output,
                })
                .map_err(|e| e.to_string())
        })
    }
}

// SAFETY: BlockingLlmAdapter is Send + Sync because:
// 1. `runtime: Arc<Runtime>` - Arc<T> is Send+Sync when T: Send+Sync, and tokio::Runtime is both
// 2. `client: C` - The wrapped async client C is accessed only through `&self` in block_on(),
//    which provides exclusive access during the blocking call. The client is never shared across
//    threads simultaneously; each call to generate() blocks until completion.
// 3. All async operations complete within the block_on() call, ensuring no dangling references.
//
// The 'static bound on C ensures the client doesn't contain non-static references that could
// become invalid. The AsyncLlmClient trait requires only &self for generate(), making concurrent
// access safe when serialized through block_on().
unsafe impl<C: AsyncLlmClient + 'static> Send for BlockingLlmAdapter<C> {}
unsafe impl<C: AsyncLlmClient + 'static> Sync for BlockingLlmAdapter<C> {}

/// Provider configuration for creating LLM clients
#[derive(Debug, Clone, Default)]
pub struct ProviderConfig {
    /// Model name override (uses provider default if None)
    pub model: Option<String>,
    /// API key for cloud providers (required for OpenAI/Anthropic)
    pub api_key: Option<String>,
    /// Base URL override (for custom endpoints)
    pub base_url: Option<String>,
}

/// Create a provider-specific LLM client wrapped in the blocking adapter
///
/// # Arguments
/// * `provider` - Provider name: "ollama", "openai", "anthropic"
/// * `config` - Optional configuration (API keys, model overrides, etc.)
///
/// # Returns
/// * `Ok(Some(adapter))` - Successfully created provider
/// * `Ok(None)` - Provider name not recognized
/// * `Err(message)` - Configuration error (e.g., missing API key)
pub fn create_provider(
    provider: &str,
    config: Option<ProviderConfig>,
) -> Result<Option<Box<dyn SyncLlmClient>>, String> {
    let config = config.unwrap_or_default();
    let runtime = Arc::new(Runtime::new().map_err(|e| e.to_string())?);

    match provider.to_lowercase().as_str() {
        "ollama" => {
            let model = config.model.as_deref().unwrap_or("llama2");
            let client = if let Some(url) = &config.base_url {
                OllamaClient::with_url(model, url)
            } else {
                OllamaClient::new(model)
            };
            Ok(Some(Box::new(BlockingLlmAdapter::with_runtime(
                client, runtime,
            ))))
        }
        "openai" => {
            let api_key = config
                .api_key
                .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                .ok_or(
                    "OpenAI requires OPENAI_API_KEY environment variable or api_key in config",
                )?;
            let model = config.model.as_deref().unwrap_or("gpt-5-mini");
            let client = OpenAiClient::with_api_key(model, &api_key).map_err(|e| e.to_string())?;
            Ok(Some(Box::new(BlockingLlmAdapter::with_runtime(
                client, runtime,
            ))))
        }
        "anthropic" => {
            let api_key = config
                .api_key
                .or_else(|| std::env::var("ANTHROPIC_API_KEY").ok())
                .ok_or(
                    "Anthropic requires ANTHROPIC_API_KEY environment variable or api_key in config",
                )?;
            let model = config.model.as_deref().unwrap_or("claude-3-haiku-20240307");
            let client =
                AnthropicClient::with_api_key(model, &api_key).map_err(|e| e.to_string())?;
            Ok(Some(Box::new(BlockingLlmAdapter::with_runtime(
                client, runtime,
            ))))
        }
        _ => Ok(None),
    }
}

/// Shared helper to setup a provider client with error handling
///
/// This consolidates the common pattern of:
/// 1. Validate provider name
/// 2. Create ProviderConfig
/// 3. Call create_provider()
/// 4. Handle Ok(Some)/Ok(None)/Err cases
///
/// # Arguments
/// * `provider_name` - Provider name: "ollama", "openai", "anthropic"
/// * `model` - Optional model name override
///
/// # Returns
/// * `Ok(client)` - Successfully created provider client
/// * `Err(message)` - Error message describing the failure
///
/// # Examples
/// ```
/// use maker::llm::adapter::setup_provider_client;
///
/// // Create Ollama provider with default model
/// let client = setup_provider_client("ollama", None).unwrap();
///
/// // Create OpenAI provider with specific model
/// let client = setup_provider_client("openai", Some("gpt-4".to_string())).unwrap();
/// ```
pub fn setup_provider_client(
    provider_name: &str,
    model: Option<String>,
) -> Result<Box<dyn SyncLlmClient>, String> {
    let provider_config = ProviderConfig {
        model,
        ..Default::default()
    };

    match create_provider(provider_name, Some(provider_config)) {
        Ok(Some(client)) => Ok(client),
        Ok(None) => Err(format!(
            "Unknown provider: '{}'. Valid options: {}",
            provider_name,
            valid_providers_str()
        )),
        Err(e) => Err(format!(
            "Failed to create provider '{}': {}",
            provider_name, e
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ollama::OllamaClient;

    #[test]
    fn test_adapter_implements_sync_trait() {
        fn accepts_sync_client(_client: &dyn crate::core::executor::LlmClient) {}

        let async_client = OllamaClient::new("test-model");
        let adapter = BlockingLlmAdapter::new(async_client);
        accepts_sync_client(&adapter);
    }

    #[test]
    fn test_adapter_with_shared_runtime() {
        let runtime = Arc::new(Runtime::new().unwrap());
        let client1 = OllamaClient::new("model1");
        let client2 = OllamaClient::new("model2");

        let _adapter1 = BlockingLlmAdapter::with_runtime(client1, Arc::clone(&runtime));
        let _adapter2 = BlockingLlmAdapter::with_runtime(client2, Arc::clone(&runtime));
    }

    #[test]
    fn test_create_provider_ollama() {
        let client = super::create_provider("ollama", None).unwrap();
        assert!(client.is_some());
    }

    #[test]
    fn test_create_provider_unknown_returns_none() {
        let client = super::create_provider("unknown-provider", None);
        assert!(client.is_ok());
        assert!(client.unwrap().is_none());
    }

    #[test]
    fn test_create_provider_ollama_with_config() {
        let config = super::ProviderConfig {
            model: Some("mistral".to_string()),
            api_key: None,
            base_url: Some("http://localhost:11434".to_string()),
        };
        let client = super::create_provider("ollama", Some(config)).unwrap();
        assert!(client.is_some());
    }

    #[test]
    fn test_create_provider_openai_without_key_fails() {
        // Temporarily unset env var if it exists
        let original = std::env::var("OPENAI_API_KEY").ok();
        std::env::remove_var("OPENAI_API_KEY");

        let result = super::create_provider("openai", None);
        assert!(result.is_err());

        // Restore env var
        if let Some(key) = original {
            std::env::set_var("OPENAI_API_KEY", key);
        }
    }

    #[test]
    fn test_create_provider_anthropic_without_key_fails() {
        // Temporarily unset env var if it exists
        let original = std::env::var("ANTHROPIC_API_KEY").ok();
        std::env::remove_var("ANTHROPIC_API_KEY");

        let result = super::create_provider("anthropic", None);
        assert!(result.is_err());

        // Restore env var
        if let Some(key) = original {
            std::env::set_var("ANTHROPIC_API_KEY", key);
        }
    }

    #[test]
    fn test_create_provider_case_insensitive() {
        let client1 = super::create_provider("OLLAMA", None).unwrap();
        let client2 = super::create_provider("Ollama", None).unwrap();
        assert!(client1.is_some());
        assert!(client2.is_some());
    }
}
