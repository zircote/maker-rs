//! Matcher implementations for candidate response grouping.
//!
//! This module provides concrete implementations of the `CandidateMatcher` trait:
//! - `EmbeddingMatcher`: Groups responses by cosine similarity of embeddings
//! - `OllamaEmbeddingClient`: Local Ollama embedding provider
//! - `OpenAiEmbeddingClient`: OpenAI embedding provider
//! - `CodeMatcher`: Groups code responses by AST structure (requires `code-matcher` feature)
//!
//! Domain presets and factory:
//! - `MatcherPreset`: Domain-specific configurations (CodeGeneration, Chat, etc.)
//! - `MatcherFactory`: Creates appropriate matchers based on presets and content

pub mod analyzer;
pub mod embedding;
pub mod factory;
pub mod ollama_embedding;
pub mod openai_embedding;
pub mod presets;

#[cfg(feature = "code-matcher")]
pub mod code;

pub use analyzer::{PromptAnalyzer, TaskCategory};
pub use embedding::{EmbeddingClient, EmbeddingMatcher, MockEmbeddingClient};
#[cfg(feature = "code-matcher")]
pub use factory::detect_language;
pub use factory::{looks_like_code, MatcherBuilder, MatcherFactory};
pub use ollama_embedding::OllamaEmbeddingClient;
pub use openai_embedding::OpenAiEmbeddingClient;
pub use presets::{CustomPreset, MatcherPreset, PreferredMatcher, PresetParseError};

#[cfg(feature = "code-matcher")]
pub use code::{CodeLanguage, CodeMatcher};

use crate::core::matcher::{CandidateMatcher, ExactMatcher};
use std::sync::Arc;

/// Error type for matcher creation failures
#[derive(Debug, Clone, PartialEq)]
pub enum MatcherError {
    /// Invalid matcher type specified
    InvalidMatcher {
        /// The invalid matcher value provided
        provided: String,
        /// Valid options
        valid: Vec<String>,
    },
    /// Code matcher feature not enabled
    CodeMatcherNotEnabled,
}

impl std::fmt::Display for MatcherError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatcherError::InvalidMatcher { provided, valid } => {
                write!(
                    f,
                    "invalid matcher type '{}', valid options: {:?}",
                    provided, valid
                )
            }
            MatcherError::CodeMatcherNotEnabled => {
                write!(f, "code matcher requires 'code-matcher' feature")
            }
        }
    }
}

impl std::error::Error for MatcherError {}

/// Create a matcher from a string type specification.
///
/// This is the shared implementation used by both CLI and MCP tools.
///
/// # Arguments
/// * `matcher_type` - Matcher type or preset name:
///   - Basic types: "exact", "embedding", "code"
///   - Presets: "code_generation", "summarization", "chat", "extraction",
///     "classification", "reasoning", "creative"
///   - Auto-detect: "auto" (analyzes prompt to choose preset)
///   - None defaults to "exact"
/// * `prompt` - The prompt text (used for auto-detection)
///
/// # Returns
/// * `Ok(Arc<dyn CandidateMatcher>)` - The configured matcher
/// * `Err(MatcherError)` - If the matcher type is invalid or unavailable
///
/// # Examples
/// ```
/// use maker::core::matchers::create_matcher_from_string;
///
/// // Create exact matcher (default)
/// let matcher = create_matcher_from_string(None, "test prompt").unwrap();
///
/// // Create embedding matcher
/// let matcher = create_matcher_from_string(Some("embedding"), "test prompt").unwrap();
///
/// // Use auto-detection
/// let matcher = create_matcher_from_string(Some("auto"), "Write a function to...").unwrap();
///
/// // Use preset
/// let matcher = create_matcher_from_string(Some("code_generation"), "test").unwrap();
/// ```
pub fn create_matcher_from_string(
    matcher_type: Option<&str>,
    prompt: &str,
) -> Result<Arc<dyn CandidateMatcher>, MatcherError> {
    let factory = MatcherFactory::new();

    match matcher_type.unwrap_or("exact") {
        // Basic matcher types
        "exact" => Ok(Arc::new(ExactMatcher::new())),
        "embedding" => {
            let client = Box::new(OllamaEmbeddingClient::new());
            Ok(Arc::new(EmbeddingMatcher::new(client, 0.92)))
        }
        #[cfg(feature = "code-matcher")]
        "code" => {
            use crate::core::matchers::code::{CodeLanguage, CodeMatcher};
            Ok(Arc::new(CodeMatcher::with_default_threshold(
                CodeLanguage::Rust,
            )))
        }
        #[cfg(not(feature = "code-matcher"))]
        "code" => Err(MatcherError::CodeMatcherNotEnabled),

        // Auto-detect from prompt
        "auto" => {
            let analyzer = PromptAnalyzer::new();
            let preset = analyzer.recommend_preset(prompt);
            let client = Box::new(OllamaEmbeddingClient::new());
            Ok(factory.create_for_preset(preset, Some(client)))
        }

        // Try to parse as a preset name
        other => {
            if let Ok(preset) = other.parse::<MatcherPreset>() {
                let client = Box::new(OllamaEmbeddingClient::new());
                Ok(factory.create_for_preset(preset, Some(client)))
            } else {
                Err(MatcherError::InvalidMatcher {
                    provided: other.to_string(),
                    valid: get_valid_matcher_types(),
                })
            }
        }
    }
}

/// Get list of valid matcher types for error messages.
fn get_valid_matcher_types() -> Vec<String> {
    #[cfg(feature = "code-matcher")]
    {
        vec![
            "exact".to_string(),
            "embedding".to_string(),
            "code".to_string(),
            "auto".to_string(),
            "code_generation".to_string(),
            "summarization".to_string(),
            "chat".to_string(),
            "extraction".to_string(),
            "classification".to_string(),
            "reasoning".to_string(),
            "creative".to_string(),
        ]
    }
    #[cfg(not(feature = "code-matcher"))]
    {
        vec![
            "exact".to_string(),
            "embedding".to_string(),
            "auto".to_string(),
            "code_generation".to_string(),
            "summarization".to_string(),
            "chat".to_string(),
            "extraction".to_string(),
            "classification".to_string(),
            "reasoning".to_string(),
            "creative".to_string(),
        ]
    }
}
