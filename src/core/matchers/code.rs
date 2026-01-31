//! Code AST Matcher
//!
//! Groups code responses by their Abstract Syntax Tree (AST) structure.
//! Uses tree-sitter for multi-language parsing, then normalizes the AST
//! by stripping comments, normalizing whitespace, and performing alpha-renaming
//! of identifiers. This allows structurally equivalent code with different
//! variable names or formatting to be recognized as equivalent.
//!
//! # Supported Languages
//!
//! - Rust
//! - Python
//! - JavaScript
//!
//! # Fallback
//!
//! If parsing fails (invalid syntax, unsupported language), the matcher
//! falls back to whitespace-normalized string comparison.

use crate::core::matcher::CandidateMatcher;
use std::collections::HashMap;
use tree_sitter::{Language, Parser};

/// Supported programming languages for AST matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodeLanguage {
    Rust,
    Python,
    JavaScript,
}

impl CodeLanguage {
    /// Get the tree-sitter Language for this code language.
    pub fn tree_sitter_language(&self) -> Language {
        match self {
            CodeLanguage::Rust => tree_sitter_rust::LANGUAGE.into(),
            CodeLanguage::Python => tree_sitter_python::LANGUAGE.into(),
            CodeLanguage::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
        }
    }

    /// Get identifier node types for this language.
    fn identifier_types(&self) -> &[&str] {
        match self {
            CodeLanguage::Rust => &["identifier", "field_identifier", "type_identifier"],
            CodeLanguage::Python => &["identifier"],
            CodeLanguage::JavaScript => &["identifier", "property_identifier"],
        }
    }

    /// Get comment node types for this language.
    fn comment_types(&self) -> &[&str] {
        match self {
            CodeLanguage::Rust => &["line_comment", "block_comment"],
            CodeLanguage::Python => &["comment"],
            CodeLanguage::JavaScript => &["comment"],
        }
    }

    /// Parse a string hint into a CodeLanguage.
    pub fn from_str_hint(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "rust" | "rs" => Some(CodeLanguage::Rust),
            "python" | "py" => Some(CodeLanguage::Python),
            "javascript" | "js" => Some(CodeLanguage::JavaScript),
            _ => None,
        }
    }
}

impl std::fmt::Display for CodeLanguage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodeLanguage::Rust => write!(f, "rust"),
            CodeLanguage::Python => write!(f, "python"),
            CodeLanguage::JavaScript => write!(f, "javascript"),
        }
    }
}

/// Code AST matcher using tree-sitter.
///
/// Compares code responses by parsing into ASTs, normalizing identifiers
/// (alpha-renaming), stripping comments, and comparing the normalized forms.
#[derive(Debug)]
pub struct CodeMatcher {
    /// Target programming language
    language: CodeLanguage,
    /// Similarity threshold for equivalence (0.0 to 1.0)
    threshold: f64,
}

impl CodeMatcher {
    /// Create a new CodeMatcher for the given language.
    pub fn new(language: CodeLanguage, threshold: f64) -> Self {
        Self {
            language,
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Create a CodeMatcher with the default threshold of 0.95.
    pub fn with_default_threshold(language: CodeLanguage) -> Self {
        Self::new(language, 0.95)
    }

    /// Get the configured language.
    pub fn language(&self) -> CodeLanguage {
        self.language
    }

    /// Get the configured threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Parse code into a tree-sitter tree.
    fn parse(&self, code: &str) -> Option<tree_sitter::Tree> {
        let mut parser = Parser::new();
        parser
            .set_language(&self.language.tree_sitter_language())
            .ok()?;
        parser.parse(code, None)
    }

    /// Normalize code by parsing to AST, alpha-renaming identifiers,
    /// stripping comments, and pretty-printing.
    fn normalize_ast(&self, code: &str) -> Option<String> {
        let tree = self.parse(code)?;
        let root = tree.root_node();

        // Check for parse errors — if there are errors, the code may be malformed
        if root.has_error() {
            return None;
        }

        let mut output = String::new();
        let mut renaming = AlphaRenamer::new();
        self.walk_and_normalize(root, code.as_bytes(), &mut renaming, &mut output);

        Some(output)
    }

    /// Recursively walk the AST, emitting normalized tokens.
    fn walk_and_normalize(
        &self,
        node: tree_sitter::Node,
        source: &[u8],
        renamer: &mut AlphaRenamer,
        output: &mut String,
    ) {
        let kind = node.kind();

        // Skip comment nodes
        if self.language.comment_types().contains(&kind) {
            return;
        }

        if node.child_count() == 0 {
            // Leaf node — emit normalized content
            let text = node.utf8_text(source).unwrap_or("");

            if self.language.identifier_types().contains(&kind) {
                // Alpha-rename identifiers
                let renamed = renamer.rename(text);
                output.push_str(&renamed);
            } else {
                output.push_str(text);
            }
            output.push(' ');
        } else {
            // Internal node — recurse into children
            for i in 0..node.child_count() {
                if let Some(child) = node.child(i) {
                    self.walk_and_normalize(child, source, renamer, output);
                }
            }
        }
    }

    /// Compute similarity between two normalized AST strings.
    /// Uses token-level Jaccard similarity.
    fn token_similarity(a: &str, b: &str) -> f64 {
        let tokens_a: Vec<&str> = a.split_whitespace().collect();
        let tokens_b: Vec<&str> = b.split_whitespace().collect();

        if tokens_a.is_empty() && tokens_b.is_empty() {
            return 1.0;
        }
        if tokens_a.is_empty() || tokens_b.is_empty() {
            return 0.0;
        }

        // Use longest common subsequence for similarity
        let lcs_len = lcs_length(&tokens_a, &tokens_b);
        let max_len = tokens_a.len().max(tokens_b.len());

        lcs_len as f64 / max_len as f64
    }
}

/// Compute length of longest common subsequence.
fn lcs_length(a: &[&str], b: &[&str]) -> usize {
    let m = a.len();
    let n = b.len();

    // Use two-row optimization for space efficiency
    let mut prev = vec![0usize; n + 1];
    let mut curr = vec![0usize; n + 1];

    for i in 1..=m {
        for j in 1..=n {
            if a[i - 1] == b[j - 1] {
                curr[j] = prev[j - 1] + 1;
            } else {
                curr[j] = prev[j].max(curr[j - 1]);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
        curr.fill(0);
    }

    prev.iter().copied().max().unwrap_or(0)
}

/// Alpha-renamer for consistent identifier normalization.
///
/// Maps each unique identifier to a canonical name (v0, v1, v2, ...).
/// This ensures that `def foo(x)` and `def bar(y)` produce the same
/// normalized output.
#[derive(Debug)]
struct AlphaRenamer {
    mapping: HashMap<String, String>,
    counter: usize,
}

impl AlphaRenamer {
    fn new() -> Self {
        Self {
            mapping: HashMap::new(),
            counter: 0,
        }
    }

    fn rename(&mut self, identifier: &str) -> String {
        if let Some(existing) = self.mapping.get(identifier) {
            return existing.clone();
        }

        let canonical = format!("v{}", self.counter);
        self.counter += 1;
        self.mapping
            .insert(identifier.to_string(), canonical.clone());
        canonical
    }
}

impl CandidateMatcher for CodeMatcher {
    fn canonicalize(&self, response: &str) -> String {
        // Try AST normalization; fall back to whitespace normalization
        self.normalize_ast(response)
            .unwrap_or_else(|| response.split_whitespace().collect::<Vec<_>>().join(" "))
    }

    fn are_equivalent(&self, a: &str, b: &str) -> bool {
        self.similarity_score(a, b) >= self.threshold
    }

    fn similarity_score(&self, a: &str, b: &str) -> f64 {
        let canon_a = self.canonicalize(a);
        let canon_b = self.canonicalize(b);

        // Fast path: identical canonical forms
        if canon_a == canon_b {
            return 1.0;
        }

        Self::token_similarity(&canon_a, &canon_b)
    }

    fn matcher_type(&self) -> &str {
        "code"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // ==========================================
    // CodeLanguage Tests
    // ==========================================

    #[test]
    fn test_language_from_str_hint() {
        assert_eq!(
            CodeLanguage::from_str_hint("rust"),
            Some(CodeLanguage::Rust)
        );
        assert_eq!(CodeLanguage::from_str_hint("rs"), Some(CodeLanguage::Rust));
        assert_eq!(
            CodeLanguage::from_str_hint("Python"),
            Some(CodeLanguage::Python)
        );
        assert_eq!(
            CodeLanguage::from_str_hint("py"),
            Some(CodeLanguage::Python)
        );
        assert_eq!(
            CodeLanguage::from_str_hint("JavaScript"),
            Some(CodeLanguage::JavaScript)
        );
        assert_eq!(
            CodeLanguage::from_str_hint("js"),
            Some(CodeLanguage::JavaScript)
        );
        assert_eq!(CodeLanguage::from_str_hint("unknown"), None);
    }

    #[test]
    fn test_language_display() {
        assert_eq!(CodeLanguage::Rust.to_string(), "rust");
        assert_eq!(CodeLanguage::Python.to_string(), "python");
        assert_eq!(CodeLanguage::JavaScript.to_string(), "javascript");
    }

    // ==========================================
    // AlphaRenamer Tests
    // ==========================================

    #[test]
    fn test_alpha_renamer_consistent() {
        let mut renamer = AlphaRenamer::new();
        assert_eq!(renamer.rename("foo"), "v0");
        assert_eq!(renamer.rename("bar"), "v1");
        assert_eq!(renamer.rename("foo"), "v0"); // Same as before
        assert_eq!(renamer.rename("baz"), "v2");
    }

    // ==========================================
    // Python AST Matching Tests
    // ==========================================

    #[test]
    fn test_python_identical() {
        let matcher = CodeMatcher::with_default_threshold(CodeLanguage::Python);
        let code = "def add(x, y):\n    return x + y\n";
        assert!(matcher.are_equivalent(code, code));
        assert_eq!(matcher.similarity_score(code, code), 1.0);
    }

    #[test]
    fn test_python_alpha_renaming() {
        let matcher = CodeMatcher::with_default_threshold(CodeLanguage::Python);
        let a = "def foo(x):\n    return x + 1\n";
        let b = "def bar(y):\n    return y + 1\n";

        let canon_a = matcher.canonicalize(a);
        let canon_b = matcher.canonicalize(b);
        assert_eq!(canon_a, canon_b, "Alpha-renamed code should be identical");
        assert!(matcher.are_equivalent(a, b));
    }

    #[test]
    fn test_python_comment_stripping() {
        let matcher = CodeMatcher::with_default_threshold(CodeLanguage::Python);
        let a = "def add(x, y):\n    # add two numbers\n    return x + y\n";
        let b = "def add(x, y):\n    return x + y\n";

        let canon_a = matcher.canonicalize(a);
        let canon_b = matcher.canonicalize(b);
        assert_eq!(
            canon_a, canon_b,
            "Code with/without comments should be identical"
        );
    }

    #[test]
    fn test_python_whitespace_irrelevant() {
        let matcher = CodeMatcher::with_default_threshold(CodeLanguage::Python);
        let a = "def add(x, y):\n    return x + y\n";
        let b = "def add(x,y):\n    return x+y\n";

        // These may parse differently due to Python whitespace sensitivity
        // but the similarity should still be high
        let sim = matcher.similarity_score(a, b);
        assert!(
            sim > 0.8,
            "Similar Python code should have high similarity: {}",
            sim
        );
    }

    #[test]
    fn test_python_different_logic() {
        let matcher = CodeMatcher::new(CodeLanguage::Python, 0.95);
        let a = "def add(x, y):\n    return x + y\n";
        let b = "def multiply(x, y):\n    return x * y\n";

        let sim = matcher.similarity_score(a, b);
        // Different operations — should be similar structure but not equivalent
        assert!(
            sim < 0.95,
            "Different logic should not be equivalent: {}",
            sim
        );
    }

    // ==========================================
    // Rust AST Matching Tests
    // ==========================================

    #[test]
    fn test_rust_alpha_renaming() {
        let matcher = CodeMatcher::with_default_threshold(CodeLanguage::Rust);
        let a = "fn foo(x: i32) -> i32 { x + 1 }";
        let b = "fn bar(y: i32) -> i32 { y + 1 }";

        let canon_a = matcher.canonicalize(a);
        let canon_b = matcher.canonicalize(b);
        assert_eq!(canon_a, canon_b, "Alpha-renamed Rust code should match");
    }

    #[test]
    fn test_rust_comment_stripping() {
        let matcher = CodeMatcher::with_default_threshold(CodeLanguage::Rust);
        let a = "// increment by one\nfn inc(x: i32) -> i32 { x + 1 }";
        let b = "fn inc(x: i32) -> i32 { x + 1 }";

        let canon_a = matcher.canonicalize(a);
        let canon_b = matcher.canonicalize(b);
        assert_eq!(
            canon_a, canon_b,
            "Rust code with/without comments should match"
        );
    }

    // ==========================================
    // JavaScript AST Matching Tests
    // ==========================================

    #[test]
    fn test_js_alpha_renaming() {
        let matcher = CodeMatcher::with_default_threshold(CodeLanguage::JavaScript);
        let a = "function add(a, b) { return a + b; }";
        let b = "function sum(x, y) { return x + y; }";

        let canon_a = matcher.canonicalize(a);
        let canon_b = matcher.canonicalize(b);
        assert_eq!(canon_a, canon_b, "Alpha-renamed JS code should match");
    }

    // ==========================================
    // Fallback Tests
    // ==========================================

    #[test]
    fn test_fallback_on_parse_error() {
        let matcher = CodeMatcher::with_default_threshold(CodeLanguage::Python);
        // Invalid Python that tree-sitter may have errors on
        let code = "this is not valid python code @@##$$";
        let canon = matcher.canonicalize(code);
        // Should fall back to whitespace normalization
        assert!(!canon.is_empty());
    }

    // ==========================================
    // Trait Object Tests
    // ==========================================

    #[test]
    fn test_as_candidate_matcher_trait_object() {
        let matcher: Arc<dyn CandidateMatcher> =
            Arc::new(CodeMatcher::with_default_threshold(CodeLanguage::Python));
        assert_eq!(matcher.matcher_type(), "code");
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CodeMatcher>();
    }

    // ==========================================
    // LCS Tests
    // ==========================================

    #[test]
    fn test_lcs_identical() {
        let a = vec!["a", "b", "c"];
        assert_eq!(lcs_length(&a, &a), 3);
    }

    #[test]
    fn test_lcs_completely_different() {
        let a = vec!["a", "b", "c"];
        let b = vec!["x", "y", "z"];
        assert_eq!(lcs_length(&a, &b), 0);
    }

    #[test]
    fn test_lcs_partial() {
        let a = vec!["a", "b", "c", "d"];
        let b = vec!["a", "x", "c", "d"];
        assert_eq!(lcs_length(&a, &b), 3);
    }

    #[test]
    fn test_lcs_empty() {
        let a: Vec<&str> = vec![];
        let b = vec!["a"];
        assert_eq!(lcs_length(&a, &b), 0);
    }

    // ==========================================
    // Configuration Tests
    // ==========================================

    #[test]
    fn test_matcher_type() {
        let matcher = CodeMatcher::with_default_threshold(CodeLanguage::Rust);
        assert_eq!(matcher.matcher_type(), "code");
    }

    #[test]
    fn test_threshold() {
        let matcher = CodeMatcher::new(CodeLanguage::Python, 0.85);
        assert_eq!(matcher.threshold(), 0.85);
    }

    #[test]
    fn test_threshold_clamped() {
        let matcher = CodeMatcher::new(CodeLanguage::Python, 1.5);
        assert_eq!(matcher.threshold(), 1.0);

        let matcher = CodeMatcher::new(CodeLanguage::Python, -0.5);
        assert_eq!(matcher.threshold(), 0.0);
    }

    #[test]
    fn test_language() {
        let matcher = CodeMatcher::with_default_threshold(CodeLanguage::Rust);
        assert_eq!(matcher.language(), CodeLanguage::Rust);
    }

    // ==========================================
    // Reflexivity and Symmetry
    // ==========================================

    #[test]
    fn test_reflexivity() {
        let matcher = CodeMatcher::with_default_threshold(CodeLanguage::Python);
        let samples = vec!["def f(x): return x", "x = 1 + 2", "print('hello')"];
        for s in samples {
            assert!(
                matcher.are_equivalent(s, s),
                "Reflexivity violated for: {}",
                s
            );
        }
    }

    #[test]
    fn test_symmetry() {
        let matcher = CodeMatcher::with_default_threshold(CodeLanguage::Python);
        let pairs = vec![
            ("def f(x): return x", "def g(y): return y"),
            ("x = 1", "y = 2"),
        ];
        for (a, b) in pairs {
            let sim_ab = matcher.similarity_score(a, b);
            let sim_ba = matcher.similarity_score(b, a);
            assert!(
                (sim_ab - sim_ba).abs() < 1e-10,
                "Symmetry violated for ({}, {}): {} vs {}",
                a,
                b,
                sim_ab,
                sim_ba
            );
        }
    }
}
