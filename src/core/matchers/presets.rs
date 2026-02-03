//! Domain-specific matcher presets with tuned thresholds.
//!
//! Provides predefined configurations for common use cases, ensuring optimal
//! similarity thresholds and matcher selection based on task domain.

use std::fmt;

/// Domain-specific matcher presets with optimized thresholds.
///
/// Each preset encodes domain knowledge about:
/// - How strict similarity matching should be
/// - Whether semantic (embedding) or structural (AST) matching is preferred
/// - Fallback behavior when primary matcher fails
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MatcherPreset {
    /// Code generation tasks requiring high structural fidelity.
    /// Uses CodeMatcher (AST) when available, falls back to high-threshold embedding.
    /// Threshold: 0.95 (strict - code must be functionally equivalent)
    CodeGeneration,

    /// Text summarization and paraphrasing tasks.
    /// Uses embedding similarity with moderate threshold.
    /// Threshold: 0.85 (allows stylistic variation while preserving meaning)
    Summarization,

    /// Conversational and chat-style responses.
    /// Uses embedding similarity with standard threshold.
    /// Threshold: 0.90 (balanced between variation and consistency)
    #[default]
    Chat,

    /// Data extraction and structured output tasks.
    /// Uses exact matching for structured fields, embedding for free text.
    /// Threshold: 0.92 (strict for data integrity)
    Extraction,

    /// Classification and labeling tasks.
    /// Uses exact matching for labels, embedding for explanations.
    /// Threshold: 0.98 (very strict - labels should match exactly)
    Classification,

    /// Mathematical and logical reasoning tasks.
    /// Uses exact matching with normalization for numeric values.
    /// Threshold: 1.0 (exact match required for correctness)
    Reasoning,

    /// Creative writing and open-ended generation.
    /// Uses embedding similarity with relaxed threshold.
    /// Threshold: 0.80 (allows creative variation)
    Creative,

    /// Custom threshold for specialized use cases.
    Custom(CustomPreset),
}

/// Custom preset configuration for specialized domains.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CustomPreset {
    /// Similarity threshold (0.0 to 1.0)
    pub threshold: f64,
    /// Preferred matcher type
    pub preferred_matcher: PreferredMatcher,
}

impl Eq for CustomPreset {}

impl std::hash::Hash for CustomPreset {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Convert f64 to bits for hashing (safe for our use case)
        self.threshold.to_bits().hash(state);
        self.preferred_matcher.hash(state);
    }
}

/// Preferred matcher type for a preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PreferredMatcher {
    /// Exact string matching with normalization
    Exact,
    /// Embedding-based semantic similarity
    #[default]
    Embedding,
    /// AST-based code structure matching
    Code,
    /// Automatic detection based on content
    Auto,
}

impl MatcherPreset {
    /// Returns the similarity threshold for this preset.
    ///
    /// The threshold determines when two responses are considered equivalent.
    /// Higher values require more similar responses.
    pub fn threshold(&self) -> f64 {
        match self {
            Self::CodeGeneration => 0.95,
            Self::Summarization => 0.85,
            Self::Chat => 0.90,
            Self::Extraction => 0.92,
            Self::Classification => 0.98,
            Self::Reasoning => 1.0,
            Self::Creative => 0.80,
            Self::Custom(preset) => preset.threshold,
        }
    }

    /// Returns the preferred matcher type for this preset.
    pub fn preferred_matcher(&self) -> PreferredMatcher {
        match self {
            Self::CodeGeneration => PreferredMatcher::Code,
            Self::Summarization => PreferredMatcher::Embedding,
            Self::Chat => PreferredMatcher::Embedding,
            Self::Extraction => PreferredMatcher::Exact,
            Self::Classification => PreferredMatcher::Exact,
            Self::Reasoning => PreferredMatcher::Exact,
            Self::Creative => PreferredMatcher::Embedding,
            Self::Custom(preset) => preset.preferred_matcher,
        }
    }

    /// Returns whether this preset should fall back to embedding matching
    /// when the preferred matcher cannot be used.
    pub fn fallback_to_embedding(&self) -> bool {
        match self {
            Self::CodeGeneration => true,  // Fall back if code parsing fails
            Self::Summarization => false,  // Already using embedding
            Self::Chat => false,           // Already using embedding
            Self::Extraction => true,      // Fall back for free-text fields
            Self::Classification => false, // Stay strict
            Self::Reasoning => false,      // Stay strict
            Self::Creative => false,       // Already using embedding
            Self::Custom(_) => true,       // Conservative default
        }
    }

    /// Creates a custom preset with the specified threshold.
    pub fn custom(threshold: f64) -> Self {
        Self::Custom(CustomPreset {
            threshold: threshold.clamp(0.0, 1.0),
            preferred_matcher: PreferredMatcher::Auto,
        })
    }

    /// Creates a custom preset with threshold and matcher preference.
    pub fn custom_with_matcher(threshold: f64, preferred: PreferredMatcher) -> Self {
        Self::Custom(CustomPreset {
            threshold: threshold.clamp(0.0, 1.0),
            preferred_matcher: preferred,
        })
    }

    /// Returns a human-readable description of this preset.
    pub fn description(&self) -> &'static str {
        match self {
            Self::CodeGeneration => "Code generation with AST-based matching",
            Self::Summarization => "Text summarization with semantic similarity",
            Self::Chat => "Conversational responses with balanced matching",
            Self::Extraction => "Data extraction with strict field matching",
            Self::Classification => "Classification with exact label matching",
            Self::Reasoning => "Mathematical reasoning with exact matching",
            Self::Creative => "Creative writing with relaxed similarity",
            Self::Custom(_) => "Custom domain configuration",
        }
    }
}

impl fmt::Display for MatcherPreset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CodeGeneration => write!(f, "code_generation"),
            Self::Summarization => write!(f, "summarization"),
            Self::Chat => write!(f, "chat"),
            Self::Extraction => write!(f, "extraction"),
            Self::Classification => write!(f, "classification"),
            Self::Reasoning => write!(f, "reasoning"),
            Self::Creative => write!(f, "creative"),
            Self::Custom(preset) => write!(f, "custom({})", preset.threshold),
        }
    }
}

impl std::str::FromStr for MatcherPreset {
    type Err = PresetParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "code_generation" | "code" | "codegeneration" => Ok(Self::CodeGeneration),
            "summarization" | "summary" => Ok(Self::Summarization),
            "chat" | "conversation" => Ok(Self::Chat),
            "extraction" | "extract" => Ok(Self::Extraction),
            "classification" | "classify" => Ok(Self::Classification),
            "reasoning" | "math" | "logic" => Ok(Self::Reasoning),
            "creative" | "writing" => Ok(Self::Creative),
            s if s.starts_with("custom(") && s.ends_with(')') => {
                let threshold_str = &s[7..s.len() - 1];
                let threshold: f64 = threshold_str
                    .parse()
                    .map_err(|_| PresetParseError::InvalidThreshold(threshold_str.to_string()))?;
                Ok(Self::custom(threshold))
            }
            _ => Err(PresetParseError::UnknownPreset(s.to_string())),
        }
    }
}

/// Error parsing a matcher preset from string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PresetParseError {
    /// Unknown preset name
    UnknownPreset(String),
    /// Invalid threshold value in custom preset
    InvalidThreshold(String),
}

impl fmt::Display for PresetParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownPreset(name) => write!(f, "unknown preset: {}", name),
            Self::InvalidThreshold(val) => write!(f, "invalid threshold value: {}", val),
        }
    }
}

impl std::error::Error for PresetParseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preset_thresholds() {
        assert!((MatcherPreset::CodeGeneration.threshold() - 0.95).abs() < f64::EPSILON);
        assert!((MatcherPreset::Summarization.threshold() - 0.85).abs() < f64::EPSILON);
        assert!((MatcherPreset::Chat.threshold() - 0.90).abs() < f64::EPSILON);
        assert!((MatcherPreset::Extraction.threshold() - 0.92).abs() < f64::EPSILON);
        assert!((MatcherPreset::Classification.threshold() - 0.98).abs() < f64::EPSILON);
        assert!((MatcherPreset::Reasoning.threshold() - 1.0).abs() < f64::EPSILON);
        assert!((MatcherPreset::Creative.threshold() - 0.80).abs() < f64::EPSILON);
    }

    #[test]
    fn test_custom_preset() {
        let preset = MatcherPreset::custom(0.88);
        assert!((preset.threshold() - 0.88).abs() < f64::EPSILON);
        assert_eq!(preset.preferred_matcher(), PreferredMatcher::Auto);
    }

    #[test]
    fn test_custom_preset_clamping() {
        let low = MatcherPreset::custom(-0.5);
        assert!((low.threshold() - 0.0).abs() < f64::EPSILON);

        let high = MatcherPreset::custom(1.5);
        assert!((high.threshold() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_preset_parsing() {
        assert_eq!(
            "code".parse::<MatcherPreset>().unwrap(),
            MatcherPreset::CodeGeneration
        );
        assert_eq!(
            "summarization".parse::<MatcherPreset>().unwrap(),
            MatcherPreset::Summarization
        );
        assert_eq!(
            "chat".parse::<MatcherPreset>().unwrap(),
            MatcherPreset::Chat
        );
        assert_eq!(
            "custom(0.75)".parse::<MatcherPreset>().unwrap().threshold(),
            0.75
        );
    }

    #[test]
    fn test_preset_display() {
        assert_eq!(MatcherPreset::CodeGeneration.to_string(), "code_generation");
        assert_eq!(MatcherPreset::custom(0.88).to_string(), "custom(0.88)");
    }

    #[test]
    fn test_preferred_matchers() {
        assert_eq!(
            MatcherPreset::CodeGeneration.preferred_matcher(),
            PreferredMatcher::Code
        );
        assert_eq!(
            MatcherPreset::Chat.preferred_matcher(),
            PreferredMatcher::Embedding
        );
        assert_eq!(
            MatcherPreset::Reasoning.preferred_matcher(),
            PreferredMatcher::Exact
        );
    }

    #[test]
    fn test_default_preset() {
        assert_eq!(MatcherPreset::default(), MatcherPreset::Chat);
    }
}
