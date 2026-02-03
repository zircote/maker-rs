//! LLM-Driven Decomposition Agent ("Insight Agent")
//!
//! Implements the `DecompositionAgent` trait using an LLM to propose task
//! decompositions. This enables automated recursive task discovery as described
//! in the MAKER paper's "Insight Agents" concept.
//!
//! # Architecture
//!
//! ```text
//! Task Description
//!       ↓
//! LlmDecompositionAgent.propose_decomposition()
//!       ↓
//! LLM generates JSON proposal
//!       ↓
//! Parse and validate proposal
//!       ↓
//! DecompositionProposal { subtasks, composition_fn }
//! ```
//!
//! # Key Features
//!
//! - **Structured Output**: Uses JSON schema to ensure valid decomposition proposals
//! - **Atomicity Detection**: LLM judges when tasks are atomic (m=1)
//! - **Domain Hints**: Optional domain-specific guidance for better decompositions
//! - **Temperature Control**: Configurable sampling temperature for diversity

use super::{
    CompositionFunction, DecompositionAgent, DecompositionError, DecompositionProposal,
    DecompositionSubtask, MergeStrategy,
};
use crate::core::executor::LlmClient;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for the LLM decomposition agent.
#[derive(Debug, Clone)]
pub struct LlmAgentConfig {
    /// Maximum subtasks per decomposition
    pub max_subtasks: usize,
    /// Temperature for LLM sampling
    pub temperature: f64,
    /// Domain-specific hints to guide decomposition
    pub domain_hints: Option<String>,
    /// Maximum description length for subtasks
    pub max_description_length: usize,
}

impl Default for LlmAgentConfig {
    fn default() -> Self {
        Self {
            max_subtasks: 10,
            temperature: 0.7,
            domain_hints: None,
            max_description_length: 500,
        }
    }
}

impl LlmAgentConfig {
    /// Set maximum subtasks per decomposition.
    pub fn with_max_subtasks(mut self, max: usize) -> Self {
        self.max_subtasks = max;
        self
    }

    /// Set LLM sampling temperature.
    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    /// Set domain-specific hints.
    pub fn with_domain_hints(mut self, hints: impl Into<String>) -> Self {
        self.domain_hints = Some(hints.into());
        self
    }
}

/// LLM-driven decomposition agent.
///
/// Uses an LLM to propose task decompositions based on natural language
/// descriptions. The LLM is prompted to output structured JSON that is
/// parsed into a `DecompositionProposal`.
pub struct LlmDecompositionAgent {
    /// LLM client for generating proposals
    client: Arc<dyn LlmClient>,
    /// Configuration for the agent
    config: LlmAgentConfig,
    /// Agent name for identification
    name: String,
}

impl std::fmt::Debug for LlmDecompositionAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmDecompositionAgent")
            .field("name", &self.name)
            .field("config", &self.config)
            .finish()
    }
}

impl LlmDecompositionAgent {
    /// Create a new LLM decomposition agent.
    pub fn new(client: Arc<dyn LlmClient>, config: LlmAgentConfig) -> Self {
        Self {
            client,
            config,
            name: "llm_decomposition_agent".to_string(),
        }
    }

    /// Create with a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Build the decomposition prompt for the LLM.
    fn build_decomposition_prompt(
        &self,
        task_id: &str,
        description: &str,
        context: &serde_json::Value,
        depth: usize,
    ) -> String {
        let domain_hint = self
            .config
            .domain_hints
            .as_deref()
            .unwrap_or("general task decomposition");

        format!(
            r#"You are a task decomposition expert specializing in {domain_hint}.

Your task is to decompose a high-level task into smaller, manageable subtasks.

## Task Information
- Task ID: {task_id}
- Description: {description}
- Context: {context}
- Current Depth: {depth}

## Rules
1. Each subtask should be ATOMIC - a single, clear action that cannot be meaningfully subdivided
2. Subtasks should be independent where possible
3. The composition function defines how results combine
4. Maximum {max_subtasks} subtasks per decomposition
5. If the task is already atomic (cannot be meaningfully decomposed), return is_atomic: true

## Output Format
Respond with a valid JSON object (no markdown, no code blocks):

{{
  "is_atomic": false,
  "subtasks": [
    {{
      "description": "Clear, actionable description of subtask 1"
    }},
    {{
      "description": "Clear, actionable description of subtask 2"
    }}
  ],
  "composition": {{
    "type": "sequential"
  }},
  "reasoning": "Brief explanation of why this decomposition makes sense"
}}

For atomic tasks:
{{
  "is_atomic": true,
  "reasoning": "This task cannot be meaningfully decomposed because..."
}}

Composition types:
- "sequential": Execute in order, passing state between steps
- "parallel": Execute concurrently, merge results
- "conditional": Choose based on condition

Generate the decomposition JSON now:"#,
            domain_hint = domain_hint,
            task_id = task_id,
            description = description,
            context = serde_json::to_string_pretty(context).unwrap_or_default(),
            depth = depth,
            max_subtasks = self.config.max_subtasks,
        )
    }

    /// Build the atomicity check prompt.
    fn build_atomicity_prompt(&self, task_id: &str, description: &str) -> String {
        format!(
            r#"Determine if this task is ATOMIC (cannot be meaningfully decomposed further).

Task ID: {task_id}
Description: {description}

An atomic task is one that:
1. Represents a single, indivisible action
2. Cannot be broken into meaningful subtasks
3. Can be executed in one step

Respond with JSON only:
{{"is_atomic": true/false, "reason": "brief explanation"}}"#,
            task_id = task_id,
            description = description,
        )
    }

    /// Parse LLM response into a decomposition proposal.
    fn parse_proposal(
        &self,
        response: &str,
        task_id: &str,
        _depth: usize,
    ) -> Result<DecompositionProposal, DecompositionError> {
        // Extract JSON from response (handle markdown code blocks)
        let json_str = extract_json(response);

        // Parse the JSON response
        let parsed: LlmProposalResponse =
            serde_json::from_str(json_str).map_err(|e| DecompositionError::ValidationError {
                message: format!("Failed to parse LLM response as JSON: {}", e),
            })?;

        // Generate proposal ID using hash of task_id + timestamp
        let proposal_id = format!(
            "{}_{:x}",
            task_id,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );

        // If atomic, return empty proposal
        if parsed.is_atomic {
            let mut metadata = HashMap::new();
            metadata.insert("is_atomic".to_string(), serde_json::json!(true));
            if let Some(reasoning) = &parsed.reasoning {
                metadata.insert("reasoning".to_string(), serde_json::json!(reasoning));
            }

            return Ok(DecompositionProposal {
                proposal_id,
                source_task_id: task_id.to_string(),
                subtasks: vec![],
                composition_fn: CompositionFunction::Sequential,
                confidence: 1.0,
                rationale: parsed.reasoning,
                metadata,
            });
        }

        // Convert subtasks
        let subtasks: Vec<DecompositionSubtask> = parsed
            .subtasks
            .into_iter()
            .enumerate()
            .map(|(i, st)| {
                DecompositionSubtask::leaf(
                    format!("{}_sub_{}", task_id, i),
                    truncate(&st.description, self.config.max_description_length),
                )
                .with_parent(task_id)
                .with_order(i)
            })
            .collect();

        // Convert composition function
        let composition_fn = match parsed.composition.composition_type.as_str() {
            "sequential" => CompositionFunction::Sequential,
            "parallel" => CompositionFunction::Parallel {
                merge_strategy: MergeStrategy::Concatenate,
            },
            "conditional" => CompositionFunction::Conditional {
                condition: parsed
                    .composition
                    .condition
                    .unwrap_or_else(|| "true".to_string()),
            },
            other => CompositionFunction::Custom {
                name: other.to_string(),
                params: Default::default(),
            },
        };

        let mut metadata = HashMap::new();
        metadata.insert("agent".to_string(), serde_json::json!(self.name));

        Ok(DecompositionProposal {
            proposal_id,
            source_task_id: task_id.to_string(),
            subtasks,
            composition_fn,
            confidence: 0.8, // Default confidence for LLM-generated proposals
            rationale: parsed.reasoning,
            metadata,
        })
    }
}

impl DecompositionAgent for LlmDecompositionAgent {
    fn propose_decomposition(
        &self,
        task_id: &str,
        description: &str,
        context: &serde_json::Value,
        depth: usize,
    ) -> Result<DecompositionProposal, DecompositionError> {
        let prompt = self.build_decomposition_prompt(task_id, description, context, depth);

        let response = self
            .client
            .generate(&prompt, self.config.temperature)
            .map_err(|e| DecompositionError::AgentError {
                message: format!("LLM generation failed: {:?}", e),
            })?;

        self.parse_proposal(&response.content, task_id, depth)
    }

    fn is_atomic(&self, task_id: &str, description: &str) -> bool {
        let prompt = self.build_atomicity_prompt(task_id, description);

        match self.client.generate(&prompt, 0.0) {
            Ok(response) => {
                let json_str = extract_json(&response.content);
                if let Ok(parsed) = serde_json::from_str::<AtomicityResponse>(json_str) {
                    parsed.is_atomic
                } else {
                    false // Default to non-atomic if parsing fails
                }
            }
            Err(_) => false, // Default to non-atomic if LLM fails
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Internal structure for parsing LLM proposal responses.
#[derive(Debug, Deserialize)]
struct LlmProposalResponse {
    is_atomic: bool,
    #[serde(default)]
    subtasks: Vec<LlmSubtask>,
    #[serde(default)]
    composition: LlmComposition,
    #[serde(default)]
    reasoning: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LlmSubtask {
    description: String,
}

#[derive(Debug, Deserialize, Default)]
struct LlmComposition {
    #[serde(rename = "type", default = "default_sequential")]
    composition_type: String,
    #[serde(default)]
    condition: Option<String>,
}

fn default_sequential() -> String {
    "sequential".to_string()
}

#[derive(Debug, Deserialize)]
struct AtomicityResponse {
    is_atomic: bool,
    #[allow(dead_code)]
    #[serde(default)]
    reason: Option<String>,
}

/// Extract JSON from a response that might contain markdown code blocks.
fn extract_json(response: &str) -> &str {
    let trimmed = response.trim();

    // Check for markdown JSON code block
    if let Some(start) = trimmed.find("```json") {
        if let Some(end) = trimmed[start + 7..].find("```") {
            return trimmed[start + 7..start + 7 + end].trim();
        }
    }

    // Check for generic code block
    if let Some(start) = trimmed.find("```") {
        if let Some(end) = trimmed[start + 3..].find("```") {
            let inner = trimmed[start + 3..start + 3 + end].trim();
            // Skip language identifier if present
            if let Some(newline) = inner.find('\n') {
                return inner[newline + 1..].trim();
            }
            return inner;
        }
    }

    // Assume raw JSON
    trimmed
}

/// Truncate a string to max length, adding ellipsis if needed.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::executor::MockLlmClient;

    #[test]
    fn test_extract_json_raw() {
        let response = r#"{"is_atomic": true}"#;
        assert_eq!(extract_json(response), r#"{"is_atomic": true}"#);
    }

    #[test]
    fn test_extract_json_markdown_code_block() {
        let response = r#"Here's the decomposition:

```json
{"is_atomic": false, "subtasks": []}
```

Done!"#;
        assert_eq!(
            extract_json(response),
            r#"{"is_atomic": false, "subtasks": []}"#
        );
    }

    #[test]
    fn test_extract_json_generic_code_block() {
        let response = r#"```
{"is_atomic": true}
```"#;
        assert_eq!(extract_json(response), r#"{"is_atomic": true}"#);
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 8), "hello...");
    }

    #[test]
    fn test_parse_atomic_proposal() {
        let client = Arc::new(MockLlmClient::constant(
            r#"{"is_atomic": true, "reasoning": "Task is already atomic"}"#,
        ));
        let agent = LlmDecompositionAgent::new(client, LlmAgentConfig::default());

        let proposal = agent
            .propose_decomposition("test_1", "Simple task", &serde_json::Value::Null, 0)
            .unwrap();

        assert!(proposal.subtasks.is_empty());
        assert_eq!(
            proposal.metadata.get("is_atomic").and_then(|v| v.as_bool()),
            Some(true)
        );
    }

    #[test]
    fn test_parse_decomposed_proposal() {
        let client = Arc::new(MockLlmClient::constant(
            r#"{
                "is_atomic": false,
                "subtasks": [
                    {"description": "Step 1"},
                    {"description": "Step 2"}
                ],
                "composition": {"type": "sequential"},
                "reasoning": "Task needs two steps"
            }"#,
        ));
        let agent = LlmDecompositionAgent::new(client, LlmAgentConfig::default());

        let proposal = agent
            .propose_decomposition("task_1", "Complex task", &serde_json::Value::Null, 0)
            .unwrap();

        assert_eq!(proposal.subtasks.len(), 2);
        assert_eq!(proposal.subtasks[0].description, "Step 1");
        assert_eq!(proposal.subtasks[1].description, "Step 2");
        assert_eq!(proposal.composition_fn, CompositionFunction::Sequential);
    }

    #[test]
    fn test_is_atomic() {
        let client = Arc::new(MockLlmClient::constant(
            r#"{"is_atomic": true, "reason": "Single action"}"#,
        ));
        let agent = LlmDecompositionAgent::new(client, LlmAgentConfig::default());

        assert!(agent.is_atomic("task_1", "Print hello world"));
    }

    #[test]
    fn test_agent_name() {
        let client = Arc::new(MockLlmClient::constant("{}"));
        let agent =
            LlmDecompositionAgent::new(client, LlmAgentConfig::default()).with_name("my_agent");

        assert_eq!(agent.name(), "my_agent");
    }

    #[test]
    fn test_config_builder() {
        let config = LlmAgentConfig::default()
            .with_max_subtasks(5)
            .with_temperature(0.5)
            .with_domain_hints("software development");

        assert_eq!(config.max_subtasks, 5);
        assert!((config.temperature - 0.5).abs() < f64::EPSILON);
        assert_eq!(
            config.domain_hints,
            Some("software development".to_string())
        );
    }
}
