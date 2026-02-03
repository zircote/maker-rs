//! Factory for creating matchers based on presets and content analysis.
//!
//! The `MatcherFactory` creates appropriate `CandidateMatcher` implementations
//! based on domain presets and optional content analysis.

use std::sync::Arc;

use super::presets::{MatcherPreset, PreferredMatcher};
use super::{EmbeddingClient, EmbeddingMatcher};
use crate::core::matcher::{CandidateMatcher, ExactMatcher};

#[cfg(feature = "code-matcher")]
use super::{CodeLanguage, CodeMatcher};

/// Factory for creating matchers based on configuration.
///
/// The factory uses domain presets to determine appropriate thresholds
/// and matcher types, with optional runtime content analysis for auto-detection.
///
/// # Example
///
/// ```ignore
/// use maker::core::matchers::{MatcherFactory, MatcherPreset, MockEmbeddingClient};
///
/// let factory = MatcherFactory::new();
///
/// // Create exact matcher (no embedding client needed)
/// let exact = factory.create_exact();
///
/// // Create embedding matcher with client
/// let client = Box::new(MockEmbeddingClient::default());
/// let semantic = factory.create_embedding(client, MatcherPreset::Chat);
/// ```
#[derive(Debug, Clone, Default)]
pub struct MatcherFactory {
    /// Default preset when none specified
    default_preset: MatcherPreset,
}

impl MatcherFactory {
    /// Creates a new factory with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a factory with a specific default preset.
    pub fn with_default_preset(preset: MatcherPreset) -> Self {
        Self {
            default_preset: preset,
        }
    }

    /// Returns the default preset.
    pub fn default_preset(&self) -> MatcherPreset {
        self.default_preset
    }

    /// Creates an exact matcher.
    ///
    /// Exact matchers compare responses character-by-character after
    /// whitespace normalization.
    pub fn create_exact(&self) -> Arc<dyn CandidateMatcher> {
        Arc::new(ExactMatcher)
    }

    /// Creates an embedding-based semantic matcher.
    ///
    /// # Arguments
    ///
    /// * `client` - Embedding client for generating vectors
    /// * `preset` - Preset determining similarity threshold
    pub fn create_embedding(
        &self,
        client: Box<dyn EmbeddingClient>,
        preset: MatcherPreset,
    ) -> Arc<dyn CandidateMatcher> {
        Arc::new(EmbeddingMatcher::new(client, preset.threshold()))
    }

    /// Creates an embedding matcher with explicit threshold.
    pub fn create_embedding_with_threshold(
        &self,
        client: Box<dyn EmbeddingClient>,
        threshold: f64,
    ) -> Arc<dyn CandidateMatcher> {
        Arc::new(EmbeddingMatcher::new(client, threshold))
    }

    /// Creates a code matcher for AST-based comparison.
    ///
    /// Requires the `code-matcher` feature.
    #[cfg(feature = "code-matcher")]
    pub fn create_code(&self, language: CodeLanguage) -> Arc<dyn CandidateMatcher> {
        Arc::new(CodeMatcher::with_default_threshold(language))
    }

    /// Creates a code matcher with auto-detected language.
    ///
    /// Analyzes the content to determine the programming language.
    #[cfg(feature = "code-matcher")]
    pub fn create_code_auto(&self, content: &str) -> Arc<dyn CandidateMatcher> {
        let language = detect_language(content);
        Arc::new(CodeMatcher::with_default_threshold(language))
    }

    /// Creates a matcher based on preset and optional embedding client.
    ///
    /// If the preset requires embedding matching and no client is provided,
    /// falls back to exact matching.
    pub fn create_for_preset(
        &self,
        preset: MatcherPreset,
        embedding_client: Option<Box<dyn EmbeddingClient>>,
    ) -> Arc<dyn CandidateMatcher> {
        match preset.preferred_matcher() {
            PreferredMatcher::Exact => self.create_exact(),

            PreferredMatcher::Embedding => {
                if let Some(client) = embedding_client {
                    self.create_embedding(client, preset)
                } else {
                    self.create_exact()
                }
            }

            PreferredMatcher::Code => {
                #[cfg(feature = "code-matcher")]
                {
                    Arc::new(CodeMatcher::with_default_threshold(CodeLanguage::Rust))
                }
                #[cfg(not(feature = "code-matcher"))]
                {
                    if preset.fallback_to_embedding() {
                        if let Some(client) = embedding_client {
                            self.create_embedding(client, preset)
                        } else {
                            self.create_exact()
                        }
                    } else {
                        self.create_exact()
                    }
                }
            }

            PreferredMatcher::Auto => {
                if let Some(client) = embedding_client {
                    self.create_embedding(client, preset)
                } else {
                    self.create_exact()
                }
            }
        }
    }

    /// Creates a matcher based on content analysis and optional embedding client.
    ///
    /// Analyzes the content to determine the best matcher type, then creates it.
    pub fn create_for_content(
        &self,
        content: &str,
        preset: MatcherPreset,
        embedding_client: Option<Box<dyn EmbeddingClient>>,
    ) -> Arc<dyn CandidateMatcher> {
        // Check if content looks like code
        if looks_like_code(content) {
            #[cfg(feature = "code-matcher")]
            {
                return self.create_code_auto(content);
            }
            #[cfg(not(feature = "code-matcher"))]
            {
                // Fall through to preset-based creation
            }
        }

        self.create_for_preset(preset, embedding_client)
    }
}

/// Detects if text content looks like code.
pub fn looks_like_code(text: &str) -> bool {
    let code_indicators = [
        "fn ",
        "def ",
        "function ",
        "class ",
        "impl ",
        "struct ",
        "pub ",
        "async ",
        "const ",
        "let ",
        "var ",
        "import ",
        "from ",
        "require(",
        "export ",
        "module ",
        "package ",
        "return ",
        "if (",
        "for (",
        "while (",
        "match ",
        "switch ",
        "try {",
        "catch ",
        "=>",
        "->",
        "self.",
        "this.",
        "print(",
        "println!",
        "console.log",
        "__init__",
    ];

    let lower = text.to_lowercase();
    let indicator_count = code_indicators
        .iter()
        .filter(|&indicator| lower.contains(indicator))
        .count();

    let has_braces = text.contains('{') && text.contains('}');
    let has_parens = text.contains('(') && text.contains(')');
    let has_semicolons = text.matches(';').count() >= 2;
    let has_colon_indent = text.contains(':') && text.contains('\n');

    indicator_count >= 3
        || (has_braces && indicator_count >= 2)
        || (has_semicolons && indicator_count >= 1)
        || (has_parens && has_braces && indicator_count >= 1)
        || (has_parens && has_colon_indent && indicator_count >= 2) // Python-style
}

/// Detects programming language from content.
#[cfg(feature = "code-matcher")]
pub fn detect_language(content: &str) -> CodeLanguage {
    // Rust indicators
    if content.contains("fn ")
        || content.contains("impl ")
        || content.contains("pub fn")
        || content.contains("let mut")
        || content.contains("&self")
        || content.contains("-> Result")
        || content.contains("::new()")
    {
        return CodeLanguage::Rust;
    }

    // Python indicators
    if content.contains("def ")
        || content.contains("import ")
        || content.contains("from ")
        || content.contains("self.")
        || content.contains("__init__")
        || content.contains("print(")
    {
        return CodeLanguage::Python;
    }

    // JavaScript indicators
    if content.contains("const ")
        || content.contains("let ")
        || content.contains("function ")
        || content.contains("=>")
        || content.contains("require(")
        || content.contains("export ")
        || content.contains("async ")
        || content.contains("await ")
    {
        return CodeLanguage::JavaScript;
    }

    // Default to Rust (project language)
    CodeLanguage::Rust
}

/// Builder for configuring matcher creation with fluent API.
#[derive(Debug, Clone, Default)]
pub struct MatcherBuilder {
    preset: MatcherPreset,
    content_hint: Option<String>,
}

impl MatcherBuilder {
    /// Creates a new builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the preset for matcher creation.
    pub fn preset(mut self, preset: MatcherPreset) -> Self {
        self.preset = preset;
        self
    }

    /// Sets a content hint for auto-detection.
    pub fn content_hint(mut self, content: impl Into<String>) -> Self {
        self.content_hint = Some(content.into());
        self
    }

    /// Builds an exact matcher.
    pub fn build_exact(self) -> Arc<dyn CandidateMatcher> {
        Arc::new(ExactMatcher)
    }

    /// Builds an embedding matcher with the provided client.
    pub fn build_embedding(self, client: Box<dyn EmbeddingClient>) -> Arc<dyn CandidateMatcher> {
        let factory = MatcherFactory::new();
        factory.create_embedding(client, self.preset)
    }

    /// Builds a matcher, using embedding if client provided, else exact.
    pub fn build(
        self,
        embedding_client: Option<Box<dyn EmbeddingClient>>,
    ) -> Arc<dyn CandidateMatcher> {
        let factory = MatcherFactory::new();

        if let Some(content) = &self.content_hint {
            factory.create_for_content(content, self.preset, embedding_client)
        } else {
            factory.create_for_preset(self.preset, embedding_client)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::matchers::MockEmbeddingClient;

    #[test]
    fn test_factory_create_exact() {
        let factory = MatcherFactory::new();
        let matcher = factory.create_exact();
        assert_eq!(matcher.canonicalize("test"), "test");
        assert!(matcher.are_equivalent("hello world", "hello world"));
    }

    #[test]
    fn test_factory_create_embedding() {
        let factory = MatcherFactory::new();
        let client = Box::new(MockEmbeddingClient::default());
        let matcher = factory.create_embedding(client, MatcherPreset::Chat);
        assert!(matcher.similarity_score("hello", "hi") > 0.0);
    }

    #[test]
    fn test_factory_for_preset_with_client() {
        let factory = MatcherFactory::new();
        let client = Box::new(MockEmbeddingClient::default());
        let matcher = factory.create_for_preset(MatcherPreset::Summarization, Some(client));
        // Should create embedding matcher
        assert!(matcher.similarity_score("hello world", "hi world") > 0.0);
    }

    #[test]
    fn test_factory_for_preset_without_client() {
        let factory = MatcherFactory::new();
        let matcher = factory.create_for_preset(MatcherPreset::Chat, None);
        // Should fall back to exact matcher
        assert!(matcher.are_equivalent("test", "test"));
        assert!(!matcher.are_equivalent("test", "test2"));
    }

    #[test]
    fn test_factory_exact_preset() {
        let factory = MatcherFactory::new();
        let matcher = factory.create_for_preset(MatcherPreset::Reasoning, None);
        // Reasoning uses exact matching
        assert!(matcher.are_equivalent("42", "42"));
        assert!(!matcher.are_equivalent("42", "42.0"));
    }

    #[test]
    fn test_looks_like_code() {
        // Rust code
        let rust_code = r#"
            fn main() {
                let x = 42;
                println!("{}", x);
            }
        "#;
        assert!(looks_like_code(rust_code));

        // Python code
        let python_code = r#"
            def main():
                x = 42
                print(x)
        "#;
        assert!(looks_like_code(python_code));

        // Plain text
        let plain_text = "This is just a regular sentence about programming.";
        assert!(!looks_like_code(plain_text));
    }

    #[test]
    fn test_builder_exact() {
        let matcher = MatcherBuilder::new()
            .preset(MatcherPreset::Reasoning)
            .build_exact();
        assert!(matcher.are_equivalent("42", "42"));
    }

    #[test]
    fn test_builder_with_embedding() {
        let client = Box::new(MockEmbeddingClient::default());
        let matcher = MatcherBuilder::new()
            .preset(MatcherPreset::Summarization)
            .content_hint("Some text to summarize")
            .build_embedding(client);
        assert!(matcher.similarity_score("hello world", "hi world") > 0.0);
    }

    #[test]
    fn test_builder_auto_without_client() {
        let matcher = MatcherBuilder::new()
            .preset(MatcherPreset::Chat)
            .build(None);
        // Falls back to exact
        assert!(matcher.are_equivalent("test", "test"));
    }
}
