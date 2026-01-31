//! Matcher implementations for candidate response grouping.
//!
//! This module provides concrete implementations of the `CandidateMatcher` trait:
//! - `EmbeddingMatcher`: Groups responses by cosine similarity of embeddings
//! - `OllamaEmbeddingClient`: Local Ollama embedding provider
//! - `OpenAiEmbeddingClient`: OpenAI embedding provider
//! - `CodeMatcher`: Groups code responses by AST structure (requires `code-matcher` feature)

pub mod embedding;
pub mod ollama_embedding;
pub mod openai_embedding;

#[cfg(feature = "code-matcher")]
pub mod code;

pub use embedding::{EmbeddingClient, EmbeddingMatcher, MockEmbeddingClient};
pub use ollama_embedding::OllamaEmbeddingClient;
pub use openai_embedding::OpenAiEmbeddingClient;

#[cfg(feature = "code-matcher")]
pub use code::CodeMatcher;
