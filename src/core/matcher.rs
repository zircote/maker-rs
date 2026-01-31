//! Candidate Matching for MAKER Framework
//!
//! Provides a pluggable trait for comparing candidate responses during voting.
//! Different matchers enable voting to work with exact string equality, semantic
//! similarity, or domain-specific equivalence (e.g., AST-based code comparison).
//!
//! # Architecture
//!
//! The `CandidateMatcher` trait defines three operations:
//! - `canonicalize`: Normalize a response before comparison
//! - `are_equivalent`: Check if two responses should be grouped as the same candidate
//! - `similarity_score`: Return a numeric similarity (0.0 = unrelated, 1.0 = identical)
//!
//! # Default Behavior
//!
//! The `ExactMatcher` provides backward-compatible exact string matching after
//! whitespace normalization. This is the default matcher used when none is specified.

use std::sync::Arc;

/// Trait for comparing candidate responses during voting.
///
/// Implementations determine how responses are grouped into candidates.
/// The voting system uses `canonicalize` to normalize responses and
/// `are_equivalent` to decide whether two responses should share votes.
///
/// All implementations must be `Send + Sync` for thread-safe voting.
pub trait CandidateMatcher: Send + Sync + std::fmt::Debug {
    /// Normalize a response for comparison.
    ///
    /// The canonical form is used as the candidate key in the vote race.
    /// Two responses that should be grouped together must produce the same
    /// canonical form.
    fn canonicalize(&self, response: &str) -> String;

    /// Check if two responses are equivalent (should share votes).
    ///
    /// The default implementation compares canonical forms for equality.
    fn are_equivalent(&self, a: &str, b: &str) -> bool {
        self.canonicalize(a) == self.canonicalize(b)
    }

    /// Return a similarity score between two responses.
    ///
    /// Returns a value in [0.0, 1.0] where:
    /// - 1.0 = identical (after canonicalization)
    /// - 0.0 = completely different
    ///
    /// The default implementation returns 1.0 if equivalent, 0.0 otherwise.
    fn similarity_score(&self, a: &str, b: &str) -> f64 {
        if self.are_equivalent(a, b) {
            1.0
        } else {
            0.0
        }
    }

    /// Human-readable name for this matcher type.
    fn matcher_type(&self) -> &str;
}

/// Exact string matcher (default, backward-compatible).
///
/// Canonicalizes by trimming leading/trailing whitespace and collapsing
/// internal whitespace runs to single spaces. Two responses are equivalent
/// if and only if their canonical forms are identical.
#[derive(Debug, Clone)]
pub struct ExactMatcher;

impl ExactMatcher {
    /// Create a new ExactMatcher.
    pub fn new() -> Self {
        Self
    }
}

impl Default for ExactMatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl CandidateMatcher for ExactMatcher {
    fn canonicalize(&self, response: &str) -> String {
        response
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn matcher_type(&self) -> &str {
        "exact"
    }
}

/// Create the default matcher (ExactMatcher wrapped in Arc).
pub fn default_matcher() -> Arc<dyn CandidateMatcher> {
    Arc::new(ExactMatcher::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================
    // ExactMatcher Tests
    // ==========================================

    #[test]
    fn test_exact_matcher_identity() {
        let m = ExactMatcher::new();
        assert!(m.are_equivalent("hello world", "hello world"));
    }

    #[test]
    fn test_exact_matcher_whitespace_normalization() {
        let m = ExactMatcher::new();
        assert!(m.are_equivalent("hello  world", "hello world"));
        assert!(m.are_equivalent("  hello world  ", "hello world"));
        assert!(m.are_equivalent("hello\tworld", "hello world"));
        assert!(m.are_equivalent("hello\nworld", "hello world"));
    }

    #[test]
    fn test_exact_matcher_not_equivalent() {
        let m = ExactMatcher::new();
        assert!(!m.are_equivalent("hello", "world"));
        assert!(!m.are_equivalent("Hello", "hello")); // Case-sensitive
    }

    #[test]
    fn test_exact_matcher_canonicalize() {
        let m = ExactMatcher::new();
        assert_eq!(m.canonicalize("  hello   world  "), "hello world");
        assert_eq!(m.canonicalize("already clean"), "already clean");
        assert_eq!(m.canonicalize(""), "");
    }

    #[test]
    fn test_exact_matcher_similarity_score() {
        let m = ExactMatcher::new();
        assert_eq!(m.similarity_score("hello", "hello"), 1.0);
        assert_eq!(m.similarity_score("hello", "world"), 0.0);
        // Whitespace normalized
        assert_eq!(m.similarity_score("hello  world", "hello world"), 1.0);
    }

    #[test]
    fn test_exact_matcher_type() {
        let m = ExactMatcher::new();
        assert_eq!(m.matcher_type(), "exact");
    }

    #[test]
    fn test_exact_matcher_default() {
        let m = ExactMatcher::default();
        assert_eq!(m.matcher_type(), "exact");
    }

    #[test]
    fn test_default_matcher_is_exact() {
        let m = default_matcher();
        assert_eq!(m.matcher_type(), "exact");
    }

    // ==========================================
    // Trait Object Safety Tests
    // ==========================================

    #[test]
    fn test_matcher_as_trait_object() {
        let m: Arc<dyn CandidateMatcher> = Arc::new(ExactMatcher::new());
        assert!(m.are_equivalent("test", "test"));
        assert!(!m.are_equivalent("test", "other"));
    }

    #[test]
    fn test_matcher_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ExactMatcher>();
        assert_send_sync::<Arc<dyn CandidateMatcher>>();
    }

    // ==========================================
    // Edge Cases
    // ==========================================

    #[test]
    fn test_empty_strings() {
        let m = ExactMatcher::new();
        assert!(m.are_equivalent("", ""));
        assert!(m.are_equivalent("   ", ""));
        assert!(!m.are_equivalent("", "nonempty"));
    }

    #[test]
    fn test_unicode_preserved() {
        let m = ExactMatcher::new();
        assert!(m.are_equivalent("café", "café"));
        assert!(!m.are_equivalent("café", "cafe"));
        assert!(m.are_equivalent("日本語  テスト", "日本語 テスト"));
    }

    #[test]
    fn test_newlines_and_tabs_normalized() {
        let m = ExactMatcher::new();
        assert!(m.are_equivalent("line1\nline2", "line1 line2"));
        assert!(m.are_equivalent("col1\tcol2", "col1 col2"));
    }

    #[test]
    fn test_reflexivity() {
        let m = ExactMatcher::new();
        let samples = vec!["hello", "", "  spaces  ", "line\nbreak", "café"];
        for s in samples {
            assert!(
                m.are_equivalent(s, s),
                "Reflexivity violated for {:?}",
                s
            );
        }
    }

    #[test]
    fn test_symmetry() {
        let m = ExactMatcher::new();
        let pairs = vec![
            ("hello", "hello"),
            ("hello", "world"),
            ("a  b", "a b"),
        ];
        for (a, b) in pairs {
            assert_eq!(
                m.are_equivalent(a, b),
                m.are_equivalent(b, a),
                "Symmetry violated for ({:?}, {:?})",
                a,
                b
            );
        }
    }
}
