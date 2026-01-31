//! Embedding-Based Similarity Matcher
//!
//! Groups candidate responses by cosine similarity of their embedding vectors.
//! Supports pluggable embedding providers via the `EmbeddingClient` trait.
//!
//! # Architecture
//!
//! Responses are embedded into vector space, then compared via cosine similarity.
//! A configurable threshold (default 0.92) determines equivalence. An internal
//! cache prevents redundant embedding API calls during voting.

use crate::core::matcher::CandidateMatcher;
use std::collections::HashMap;
use std::sync::Mutex;

/// Trait for embedding text into vector space.
///
/// Implementations call an embedding API (Ollama, OpenAI, etc.) to convert
/// text into a fixed-dimensional vector for similarity comparison.
pub trait EmbeddingClient: Send + Sync + std::fmt::Debug {
    /// Embed a text string into a vector.
    ///
    /// Returns a vector of floats representing the text in embedding space.
    fn embed(&self, text: &str) -> Result<Vec<f64>, String>;

    /// The dimensionality of embeddings produced by this client.
    fn dimensions(&self) -> usize;
}

/// Embedding-based candidate matcher.
///
/// Groups responses whose embedding cosine similarity exceeds a threshold.
/// Maintains an internal cache of embeddings to minimize API calls.
pub struct EmbeddingMatcher {
    /// Similarity threshold for equivalence (0.0 to 1.0)
    threshold: f64,
    /// Embedding client for generating vectors
    client: Box<dyn EmbeddingClient>,
    /// Cache of computed embeddings (canonicalized text -> vector)
    cache: Mutex<HashMap<String, Vec<f64>>>,
}

impl std::fmt::Debug for EmbeddingMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbeddingMatcher")
            .field("threshold", &self.threshold)
            .field("client", &self.client)
            .field("cache_size", &self.cache.lock().unwrap().len())
            .finish()
    }
}

impl EmbeddingMatcher {
    /// Create a new EmbeddingMatcher with the given client and threshold.
    ///
    /// # Arguments
    ///
    /// * `client` - Embedding client for generating vectors
    /// * `threshold` - Cosine similarity threshold for equivalence (default: 0.92)
    pub fn new(client: Box<dyn EmbeddingClient>, threshold: f64) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 1.0),
            client,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Create with the default threshold of 0.92.
    pub fn with_default_threshold(client: Box<dyn EmbeddingClient>) -> Self {
        Self::new(client, 0.92)
    }

    /// Get the current cache size.
    pub fn cache_size(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Get the configured threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Get or compute the embedding for a text string.
    fn get_embedding(&self, text: &str) -> Result<Vec<f64>, String> {
        let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let mut cache = self.cache.lock().unwrap();

        if let Some(cached) = cache.get(&normalized) {
            return Ok(cached.clone());
        }

        let embedding = self.client.embed(&normalized)?;
        cache.insert(normalized, embedding.clone());
        Ok(embedding)
    }
}

/// Compute cosine similarity between two vectors.
///
/// Returns a value in [-1.0, 1.0] where 1.0 means identical direction.
/// Returns 0.0 if either vector has zero magnitude.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    dot / (mag_a * mag_b)
}

impl CandidateMatcher for EmbeddingMatcher {
    fn canonicalize(&self, response: &str) -> String {
        // For embedding matcher, canonicalization is whitespace normalization.
        // The actual grouping happens via are_equivalent/similarity_score.
        response.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    fn are_equivalent(&self, a: &str, b: &str) -> bool {
        self.similarity_score(a, b) >= self.threshold
    }

    fn similarity_score(&self, a: &str, b: &str) -> f64 {
        // Fast path: identical after canonicalization
        let canon_a = self.canonicalize(a);
        let canon_b = self.canonicalize(b);
        if canon_a == canon_b {
            return 1.0;
        }

        // Compute embeddings and cosine similarity
        let emb_a = match self.get_embedding(a) {
            Ok(v) => v,
            Err(_) => return 0.0,
        };
        let emb_b = match self.get_embedding(b) {
            Ok(v) => v,
            Err(_) => return 0.0,
        };

        cosine_similarity(&emb_a, &emb_b)
    }

    fn matcher_type(&self) -> &str {
        "embedding"
    }
}

/// Mock embedding client for testing.
///
/// Produces deterministic embeddings based on character frequency,
/// allowing controlled similarity testing without an actual API.
#[derive(Debug, Clone)]
pub struct MockEmbeddingClient {
    dims: usize,
}

impl MockEmbeddingClient {
    /// Create a new mock client with the given dimensionality.
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }
}

impl Default for MockEmbeddingClient {
    fn default() -> Self {
        Self::new(128)
    }
}

impl EmbeddingClient for MockEmbeddingClient {
    fn embed(&self, text: &str) -> Result<Vec<f64>, String> {
        if text.is_empty() {
            return Ok(vec![0.0; self.dims]);
        }

        // Generate a deterministic embedding based on character frequencies.
        // This creates vectors where similar text produces similar vectors.
        let mut vec = vec![0.0; self.dims];
        let bytes = text.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            let idx = (b as usize + i) % self.dims;
            vec[idx] += 1.0;
        }

        // Normalize to unit vector
        let mag: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
        if mag > 0.0 {
            for v in &mut vec {
                *v /= mag;
            }
        }

        Ok(vec)
    }

    fn dimensions(&self) -> usize {
        self.dims
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================
    // Cosine Similarity Tests
    // ==========================================

    #[test]
    fn test_cosine_identical_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_opposite_vectors() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similar_vectors() {
        let a = vec![1.0, 1.0, 0.0];
        let b = vec![1.0, 1.0, 0.1];
        let sim = cosine_similarity(&a, &b);
        assert!(sim > 0.99, "Expected high similarity, got {}", sim);
    }

    #[test]
    fn test_cosine_zero_vector() {
        let a = vec![1.0, 2.0];
        let b = vec![0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_empty_vectors() {
        let a: Vec<f64> = vec![];
        let b: Vec<f64> = vec![];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_different_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    // ==========================================
    // MockEmbeddingClient Tests
    // ==========================================

    #[test]
    fn test_mock_client_deterministic() {
        let client = MockEmbeddingClient::new(64);
        let a1 = client.embed("hello world").unwrap();
        let a2 = client.embed("hello world").unwrap();
        assert_eq!(a1, a2);
    }

    #[test]
    fn test_mock_client_dimensions() {
        let client = MockEmbeddingClient::new(256);
        assert_eq!(client.dimensions(), 256);
        let vec = client.embed("test").unwrap();
        assert_eq!(vec.len(), 256);
    }

    #[test]
    fn test_mock_client_unit_vector() {
        let client = MockEmbeddingClient::new(64);
        let vec = client.embed("test text").unwrap();
        let mag: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((mag - 1.0).abs() < 1e-10, "Expected unit vector, mag={}", mag);
    }

    #[test]
    fn test_mock_client_empty_text() {
        let client = MockEmbeddingClient::new(64);
        let vec = client.embed("").unwrap();
        assert_eq!(vec.len(), 64);
        assert!(vec.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_mock_client_similar_text_similar_embeddings() {
        let client = MockEmbeddingClient::new(128);
        let a = client.embed("hello world").unwrap();
        let b = client.embed("hello world!").unwrap();
        let sim = cosine_similarity(&a, &b);
        assert!(sim > 0.8, "Similar text should have high similarity: {}", sim);
    }

    #[test]
    fn test_mock_client_different_text_lower_similarity() {
        let client = MockEmbeddingClient::new(128);
        let a = client.embed("hello world").unwrap();
        let b = client.embed("completely different topic about quantum physics").unwrap();
        let similar = client.embed("hello world!").unwrap();

        let sim_different = cosine_similarity(&a, &b);
        let sim_similar = cosine_similarity(&a, &similar);

        assert!(
            sim_similar > sim_different,
            "Similar text ({:.4}) should score higher than different text ({:.4})",
            sim_similar,
            sim_different
        );
    }

    // ==========================================
    // EmbeddingMatcher Tests
    // ==========================================

    #[test]
    fn test_matcher_type() {
        let matcher = EmbeddingMatcher::with_default_threshold(
            Box::new(MockEmbeddingClient::default()),
        );
        assert_eq!(matcher.matcher_type(), "embedding");
    }

    #[test]
    fn test_matcher_identical_strings() {
        let matcher = EmbeddingMatcher::with_default_threshold(
            Box::new(MockEmbeddingClient::default()),
        );
        assert!(matcher.are_equivalent("hello", "hello"));
        assert_eq!(matcher.similarity_score("hello", "hello"), 1.0);
    }

    #[test]
    fn test_matcher_whitespace_equivalent() {
        let matcher = EmbeddingMatcher::with_default_threshold(
            Box::new(MockEmbeddingClient::default()),
        );
        // After canonicalization, these are identical
        assert!(matcher.are_equivalent("hello  world", "hello world"));
    }

    #[test]
    fn test_matcher_threshold() {
        let matcher = EmbeddingMatcher::new(
            Box::new(MockEmbeddingClient::default()),
            0.99,
        );
        assert_eq!(matcher.threshold(), 0.99);

        // With very high threshold, slightly different text may not be equivalent
        let sim = matcher.similarity_score("hello world", "hello world!");
        // Whether they pass depends on the mock's embedding similarity
        if sim < 0.99 {
            assert!(!matcher.are_equivalent("hello world", "hello world!"));
        }
    }

    #[test]
    fn test_matcher_threshold_clamped() {
        let matcher = EmbeddingMatcher::new(
            Box::new(MockEmbeddingClient::default()),
            1.5, // Should be clamped to 1.0
        );
        assert_eq!(matcher.threshold(), 1.0);

        let matcher = EmbeddingMatcher::new(
            Box::new(MockEmbeddingClient::default()),
            -0.5, // Should be clamped to 0.0
        );
        assert_eq!(matcher.threshold(), 0.0);
    }

    #[test]
    fn test_matcher_caching() {
        let matcher = EmbeddingMatcher::with_default_threshold(
            Box::new(MockEmbeddingClient::default()),
        );

        assert_eq!(matcher.cache_size(), 0);

        // First call embeds and caches
        let _ = matcher.similarity_score("hello", "world");
        assert_eq!(matcher.cache_size(), 2);

        // Second call should use cache (same result)
        let _ = matcher.similarity_score("hello", "world");
        assert_eq!(matcher.cache_size(), 2); // No new entries
    }

    #[test]
    fn test_matcher_cache_hit_on_repeated_text() {
        let matcher = EmbeddingMatcher::with_default_threshold(
            Box::new(MockEmbeddingClient::default()),
        );

        // Embed "hello" twice via different comparison pairs
        let s1 = matcher.similarity_score("hello", "world");
        let s2 = matcher.similarity_score("hello", "foo");

        // Cache should have 3 entries: hello, world, foo
        assert_eq!(matcher.cache_size(), 3);

        // Scores should be deterministic
        let s1_again = matcher.similarity_score("hello", "world");
        assert_eq!(s1, s1_again);
        let _ = s2;
    }

    #[test]
    fn test_matcher_reflexivity() {
        let matcher = EmbeddingMatcher::with_default_threshold(
            Box::new(MockEmbeddingClient::default()),
        );
        let samples = vec!["hello", "world", "test string", ""];
        for s in samples {
            assert!(
                matcher.are_equivalent(s, s),
                "Reflexivity violated for {:?}",
                s
            );
        }
    }

    #[test]
    fn test_matcher_symmetry() {
        let matcher = EmbeddingMatcher::with_default_threshold(
            Box::new(MockEmbeddingClient::default()),
        );
        let pairs = vec![
            ("hello", "world"),
            ("foo bar", "foo baz"),
            ("same", "same"),
        ];
        for (a, b) in pairs {
            assert_eq!(
                matcher.similarity_score(a, b),
                matcher.similarity_score(b, a),
                "Symmetry violated for ({:?}, {:?})",
                a,
                b
            );
        }
    }

    #[test]
    fn test_matcher_canonicalize() {
        let matcher = EmbeddingMatcher::with_default_threshold(
            Box::new(MockEmbeddingClient::default()),
        );
        assert_eq!(matcher.canonicalize("  hello   world  "), "hello world");
    }

    #[test]
    fn test_matcher_debug() {
        let matcher = EmbeddingMatcher::with_default_threshold(
            Box::new(MockEmbeddingClient::default()),
        );
        let debug = format!("{:?}", matcher);
        assert!(debug.contains("EmbeddingMatcher"));
        assert!(debug.contains("threshold"));
    }

    // ==========================================
    // Integration with CandidateMatcher trait
    // ==========================================

    #[test]
    fn test_as_candidate_matcher_trait_object() {
        let matcher: Box<dyn CandidateMatcher> = Box::new(
            EmbeddingMatcher::with_default_threshold(
                Box::new(MockEmbeddingClient::default()),
            ),
        );
        assert_eq!(matcher.matcher_type(), "embedding");
        assert!(matcher.are_equivalent("test", "test"));
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EmbeddingMatcher>();
        assert_send_sync::<MockEmbeddingClient>();
    }
}
