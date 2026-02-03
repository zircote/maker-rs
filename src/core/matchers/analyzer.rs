//! Prompt analyzer for automatic matcher selection.
//!
//! Analyzes prompt content to determine the appropriate task category
//! and corresponding matcher preset.

use super::presets::MatcherPreset;

/// Categories of tasks based on prompt analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskCategory {
    /// Code generation, modification, or explanation
    CodeGeneration,
    /// Text summarization or paraphrasing
    Summarization,
    /// Conversational or Q&A tasks
    Conversation,
    /// Data extraction from text
    DataExtraction,
    /// Classification or labeling tasks
    Classification,
    /// Mathematical or logical reasoning
    Reasoning,
    /// Creative writing tasks
    CreativeWriting,
    /// Unknown or mixed task type
    General,
}

impl TaskCategory {
    /// Returns the recommended matcher preset for this task category.
    pub fn recommended_preset(&self) -> MatcherPreset {
        match self {
            Self::CodeGeneration => MatcherPreset::CodeGeneration,
            Self::Summarization => MatcherPreset::Summarization,
            Self::Conversation => MatcherPreset::Chat,
            Self::DataExtraction => MatcherPreset::Extraction,
            Self::Classification => MatcherPreset::Classification,
            Self::Reasoning => MatcherPreset::Reasoning,
            Self::CreativeWriting => MatcherPreset::Creative,
            Self::General => MatcherPreset::Chat,
        }
    }
}

impl std::fmt::Display for TaskCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CodeGeneration => write!(f, "code_generation"),
            Self::Summarization => write!(f, "summarization"),
            Self::Conversation => write!(f, "conversation"),
            Self::DataExtraction => write!(f, "data_extraction"),
            Self::Classification => write!(f, "classification"),
            Self::Reasoning => write!(f, "reasoning"),
            Self::CreativeWriting => write!(f, "creative_writing"),
            Self::General => write!(f, "general"),
        }
    }
}

/// Analyzes prompts to determine task category and recommend matchers.
///
/// Uses pattern matching and keyword detection to classify prompts
/// and recommend appropriate matching strategies.
#[derive(Debug, Clone, Default)]
pub struct PromptAnalyzer {
    /// Confidence threshold for classification (0.0 to 1.0)
    confidence_threshold: f64,
}

impl PromptAnalyzer {
    /// Creates a new analyzer with default settings.
    pub fn new() -> Self {
        Self {
            confidence_threshold: 0.6,
        }
    }

    /// Creates an analyzer with a custom confidence threshold.
    pub fn with_confidence_threshold(threshold: f64) -> Self {
        Self {
            confidence_threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Analyzes a prompt and returns the detected task category.
    pub fn analyze(&self, prompt: &str) -> TaskCategory {
        let (category, confidence) = self.analyze_with_confidence(prompt);
        if confidence >= self.confidence_threshold {
            category
        } else {
            TaskCategory::General
        }
    }

    /// Analyzes a prompt and returns category with confidence score.
    pub fn analyze_with_confidence(&self, prompt: &str) -> (TaskCategory, f64) {
        let lower = prompt.to_lowercase();
        let scores = self.compute_category_scores(&lower);

        // Find highest scoring category
        let mut best = (TaskCategory::General, 0.0);
        for (category, score) in scores {
            if score > best.1 {
                best = (category, score);
            }
        }

        best
    }

    /// Returns the recommended preset for a prompt.
    pub fn recommend_preset(&self, prompt: &str) -> MatcherPreset {
        self.analyze(prompt).recommended_preset()
    }

    /// Computes scores for each category based on keyword matching.
    fn compute_category_scores(&self, text: &str) -> Vec<(TaskCategory, f64)> {
        vec![
            (TaskCategory::CodeGeneration, self.code_score(text)),
            (TaskCategory::Summarization, self.summarization_score(text)),
            (TaskCategory::Conversation, self.conversation_score(text)),
            (TaskCategory::DataExtraction, self.extraction_score(text)),
            (
                TaskCategory::Classification,
                self.classification_score(text),
            ),
            (TaskCategory::Reasoning, self.reasoning_score(text)),
            (TaskCategory::CreativeWriting, self.creative_score(text)),
        ]
    }

    /// Scores likelihood of code generation task.
    fn code_score(&self, text: &str) -> f64 {
        let keywords = [
            ("write code", 0.9),
            ("implement", 0.8),
            ("function", 0.6),
            ("code", 0.5),
            ("program", 0.5),
            ("script", 0.6),
            ("fix bug", 0.8),
            ("fix the bug", 0.8),
            ("bug", 0.5),
            ("debug", 0.7),
            ("refactor", 0.8),
            ("class", 0.4),
            ("method", 0.4),
            ("api", 0.4),
            ("algorithm", 0.6),
            ("compile", 0.7),
            ("syntax", 0.6),
            ("rust", 0.6),
            ("python", 0.6),
            ("javascript", 0.6),
            ("typescript", 0.6),
        ];

        self.keyword_score(text, &keywords)
    }

    /// Scores likelihood of summarization task.
    fn summarization_score(&self, text: &str) -> f64 {
        let keywords = [
            ("summarize", 0.95),
            ("summary", 0.9),
            ("tldr", 0.9),
            ("brief", 0.6),
            ("condense", 0.8),
            ("shorten", 0.7),
            ("main points", 0.8),
            ("key points", 0.8),
            ("paraphrase", 0.85),
            ("rephrase", 0.7),
            ("simplify", 0.6),
            ("overview", 0.7),
            ("gist", 0.8),
        ];

        self.keyword_score(text, &keywords)
    }

    /// Scores likelihood of conversational task.
    fn conversation_score(&self, text: &str) -> f64 {
        let keywords = [
            ("explain", 0.6),
            ("tell me", 0.6),
            ("what is", 0.5),
            ("how do", 0.5),
            ("why", 0.4),
            ("help me understand", 0.7),
            ("describe", 0.5),
            ("discuss", 0.6),
            ("chat", 0.8),
            ("talk about", 0.7),
        ];

        self.keyword_score(text, &keywords)
    }

    /// Scores likelihood of data extraction task.
    fn extraction_score(&self, text: &str) -> f64 {
        let keywords = [
            ("extract", 0.9),
            ("parse", 0.8),
            ("find all", 0.7),
            ("list all", 0.7),
            ("identify", 0.6),
            ("pull out", 0.8),
            ("get the", 0.5),
            ("json", 0.7),
            ("structured", 0.6),
            ("field", 0.5),
            ("entity", 0.7),
            ("data from", 0.7),
        ];

        self.keyword_score(text, &keywords)
    }

    /// Scores likelihood of classification task.
    fn classification_score(&self, text: &str) -> f64 {
        let keywords = [
            ("classify", 0.95),
            ("categorize", 0.9),
            ("label", 0.8),
            ("tag", 0.7),
            ("which category", 0.9),
            ("sentiment", 0.85),
            ("positive or negative", 0.9),
            ("true or false", 0.85),
            ("yes or no", 0.8),
            ("is this", 0.5),
            ("type of", 0.5),
        ];

        self.keyword_score(text, &keywords)
    }

    /// Scores likelihood of reasoning task.
    fn reasoning_score(&self, text: &str) -> f64 {
        let keywords = [
            ("calculate", 0.9),
            ("compute", 0.9),
            ("solve", 0.8),
            ("math", 0.85),
            ("equation", 0.85),
            ("logic", 0.8),
            ("prove", 0.85),
            ("deduce", 0.8),
            ("infer", 0.7),
            ("step by step", 0.7),
            ("reasoning", 0.85),
            ("arithmetic", 0.9),
            ("formula", 0.8),
        ];

        self.keyword_score(text, &keywords)
    }

    /// Scores likelihood of creative writing task.
    fn creative_score(&self, text: &str) -> f64 {
        let keywords = [
            ("write a story", 0.95),
            ("creative", 0.8),
            ("poem", 0.9),
            ("fiction", 0.9),
            ("imagine", 0.7),
            ("make up", 0.7),
            ("invent", 0.6),
            ("brainstorm", 0.7),
            ("generate ideas", 0.7),
            ("narrative", 0.8),
            ("dialogue", 0.7),
            ("character", 0.5),
            ("plot", 0.6),
        ];

        self.keyword_score(text, &keywords)
    }

    /// Computes weighted keyword score.
    fn keyword_score(&self, text: &str, keywords: &[(&str, f64)]) -> f64 {
        let mut max_weight = 0.0f64;
        let mut total_weight = 0.0;
        let mut count = 0;

        for (keyword, weight) in keywords {
            if text.contains(keyword) {
                max_weight = max_weight.max(*weight);
                total_weight += weight;
                count += 1;
            }
        }

        if count == 0 {
            0.0
        } else {
            // Use the highest weight found, boosted by number of matches
            let match_bonus = ((count - 1) as f64 * 0.05).min(0.15);
            (max_weight + match_bonus).min(1.0)
        }
    }

    /// Detects if the prompt expects code output.
    pub fn expects_code_output(&self, prompt: &str) -> bool {
        let lower = prompt.to_lowercase();

        let code_output_indicators = [
            "write code",
            "write a function",
            "implement",
            "create a class",
            "code that",
            "script that",
            "program that",
            "in rust",
            "in python",
            "in javascript",
            "give me code",
            "show me code",
            "```",
        ];

        code_output_indicators
            .iter()
            .any(|indicator| lower.contains(indicator))
    }

    /// Detects if the prompt expects structured (JSON/XML) output.
    pub fn expects_structured_output(&self, prompt: &str) -> bool {
        let lower = prompt.to_lowercase();

        let structured_indicators = [
            "json",
            "xml",
            "yaml",
            "structured",
            "format:",
            "schema",
            "output format",
            "return a",
            "return an object",
        ];

        structured_indicators
            .iter()
            .any(|indicator| lower.contains(indicator))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_detection() {
        let analyzer = PromptAnalyzer::new();

        assert_eq!(
            analyzer.analyze("Write code to sort a list"),
            TaskCategory::CodeGeneration
        );
        assert_eq!(
            analyzer.analyze("Implement a binary search function in Rust"),
            TaskCategory::CodeGeneration
        );
        assert_eq!(
            analyzer.analyze("Fix the bug in this Python script"),
            TaskCategory::CodeGeneration
        );
    }

    #[test]
    fn test_summarization_detection() {
        let analyzer = PromptAnalyzer::new();

        assert_eq!(
            analyzer.analyze("Summarize this article"),
            TaskCategory::Summarization
        );
        assert_eq!(
            analyzer.analyze("Give me a brief summary of the key points"),
            TaskCategory::Summarization
        );
        assert_eq!(
            analyzer.analyze("TLDR this document"),
            TaskCategory::Summarization
        );
    }

    #[test]
    fn test_reasoning_detection() {
        let analyzer = PromptAnalyzer::new();

        assert_eq!(
            analyzer.analyze("Calculate 15 + 27"),
            TaskCategory::Reasoning
        );
        assert_eq!(
            analyzer.analyze("Solve this math equation"),
            TaskCategory::Reasoning
        );
        assert_eq!(
            analyzer.analyze("Compute the formula step by step"),
            TaskCategory::Reasoning
        );
    }

    #[test]
    fn test_classification_detection() {
        let analyzer = PromptAnalyzer::new();

        assert_eq!(
            analyzer.analyze("Classify this text as positive or negative sentiment"),
            TaskCategory::Classification
        );
        assert_eq!(
            analyzer.analyze("Categorize these items"),
            TaskCategory::Classification
        );
    }

    #[test]
    fn test_creative_detection() {
        let analyzer = PromptAnalyzer::new();

        assert_eq!(
            analyzer.analyze("Write a story about a dragon"),
            TaskCategory::CreativeWriting
        );
        assert_eq!(
            analyzer.analyze("Create a poem about nature"),
            TaskCategory::CreativeWriting
        );
    }

    #[test]
    fn test_general_fallback() {
        let analyzer = PromptAnalyzer::new();

        // Ambiguous prompts should fall back to General
        assert_eq!(analyzer.analyze("Hello"), TaskCategory::General);
        assert_eq!(analyzer.analyze("xyz"), TaskCategory::General);
    }

    #[test]
    fn test_expects_code_output() {
        let analyzer = PromptAnalyzer::new();

        assert!(analyzer.expects_code_output("Write code to parse JSON"));
        assert!(analyzer.expects_code_output("Implement a function in Python"));
        assert!(!analyzer.expects_code_output("Explain how Python works"));
    }

    #[test]
    fn test_expects_structured_output() {
        let analyzer = PromptAnalyzer::new();

        assert!(analyzer.expects_structured_output("Return the data as JSON"));
        assert!(analyzer.expects_structured_output("Output format: YAML"));
        assert!(!analyzer.expects_structured_output("Tell me about databases"));
    }

    #[test]
    fn test_recommended_preset() {
        let analyzer = PromptAnalyzer::new();

        let preset = analyzer.recommend_preset("Write code to sort numbers");
        assert_eq!(preset, MatcherPreset::CodeGeneration);

        let preset = analyzer.recommend_preset("Summarize this text");
        assert_eq!(preset, MatcherPreset::Summarization);

        let preset = analyzer.recommend_preset("Calculate 2 + 2");
        assert_eq!(preset, MatcherPreset::Reasoning);
    }
}
