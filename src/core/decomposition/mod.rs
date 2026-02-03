//! Recursive Decomposition Framework for MAKER
//!
//! Implements the recursive architecture from Section 7 of the System Design Specification:
//! - Decomposition Agents: Split tasks into subtasks with composition functions
//! - Decomposition Discriminators: Vote on proposed decomposition strategies
//! - Problem Solver Agents: Execute atomic (m=1) leaf nodes
//! - Solution Discriminators: Aggregate results using voted composition functions
//!
//! # Architecture
//!
//! ```text
//! High-level Task
//!       ↓
//! DecompositionAgent.propose_decomposition()
//!       ↓
//! DecompositionProposal { subtasks, composition_fn }
//!       ↓
//! DecompositionDiscriminator votes on proposal
//!       ↓
//! Recurse until all leaf nodes have m=1
//!       ↓
//! LeafNodeExecutor solves atomic tasks with voting
//!       ↓
//! SolutionDiscriminator aggregates results
//! ```
//!
//! # Key Concepts
//!
//! - **m_value**: Steps per agent. Leaf nodes MUST have m_value == 1 (enforced at runtime)
//! - **CompositionFunction**: How subtask results combine (Sequential, Parallel, Conditional, Custom)
//! - **Depth Limit**: Maximum recursion depth (default 10) to prevent infinite loops

pub mod aggregator;
pub mod discriminator;
pub mod domains;
pub mod filesystem;
pub mod llm_agent;
pub mod orchestrator;
pub mod solver;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// Re-export aggregator types
pub use aggregator::{
    compose_recursive, AggregatedResult, AggregationError, AggregationMetrics, AggregatorConfig,
    SolutionDiscriminator,
};

// Re-export orchestrator types
pub use orchestrator::{
    CancellationToken, OrchestrationError, OrchestrationMetrics, OrchestrationResult,
    OrchestratorConfig, RecursiveOrchestrator,
};

// Re-export discriminator types
pub use discriminator::{
    vote_on_decomposition, DecompositionDiscriminator, DecompositionVoteResult, ProposalMatcher,
};

// Re-export solver types
pub use solver::{
    ExecutionError, ExecutionMetrics, ExecutorConfig, LeafNodeExecutor, ProposalResult,
    SubtaskResult,
};

// Re-export LLM decomposition agent types
pub use llm_agent::{LlmAgentConfig, LlmDecompositionAgent};

/// Unique identifier for a subtask within a decomposition tree
pub type TaskId = String;

/// How subtask results should be combined
///
/// Each composition function defines a different aggregation strategy
/// for combining the results of subtasks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "type", content = "params")]
pub enum CompositionFunction {
    /// Execute subtasks in order, passing state from one to the next
    ///
    /// Result = f(f(...f(initial, result_1), result_2), ..., result_n)
    #[default]
    Sequential,

    /// Execute subtasks concurrently, merge results
    ///
    /// Result = merge(result_1, result_2, ..., result_n)
    Parallel {
        /// Strategy for merging parallel results
        #[serde(default)]
        merge_strategy: MergeStrategy,
    },

    /// Execute subtasks based on a condition
    ///
    /// Result = if condition then result_true else result_false
    Conditional {
        /// The condition expression to evaluate
        condition: String,
    },

    /// Custom composition with user-defined logic
    ///
    /// Result = custom_fn(results)
    Custom {
        /// Name of the custom composition function
        name: String,
        /// Parameters for the custom function
        #[serde(default)]
        params: HashMap<String, serde_json::Value>,
    },
}

/// Strategy for merging results in parallel composition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MergeStrategy {
    /// Concatenate all results
    #[default]
    Concatenate,
    /// Use the first successful result
    FirstSuccess,
    /// Use the last successful result
    LastSuccess,
    /// Combine as a JSON array
    CollectArray,
    /// Deep merge JSON objects
    DeepMerge,
}

/// A subtask within a decomposition
///
/// Represents a single node in the decomposition tree. Leaf nodes
/// (those with no further decomposition) MUST have `m_value == 1`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecompositionSubtask {
    /// Unique identifier for this subtask
    pub task_id: TaskId,

    /// Parent task ID (None for root tasks)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<TaskId>,

    /// Steps per agent (m value)
    ///
    /// For leaf nodes, this MUST be 1 (enforced by validation).
    /// For intermediate nodes, this represents the total steps of all children.
    pub m_value: usize,

    /// Human-readable description of this subtask
    pub description: String,

    /// Context/state passed to the agent executing this subtask
    #[serde(default)]
    pub context: serde_json::Value,

    /// Whether this is a leaf node (no further decomposition needed)
    #[serde(default)]
    pub is_leaf: bool,

    /// Ordering hint for execution (lower = earlier)
    #[serde(default)]
    pub order: usize,

    /// Optional metadata for domain-specific extensions
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl DecompositionSubtask {
    /// Create a new subtask
    pub fn new(task_id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            task_id: task_id.into(),
            parent_id: None,
            m_value: 1,
            description: description.into(),
            context: serde_json::Value::Null,
            is_leaf: false,
            order: 0,
            metadata: HashMap::new(),
        }
    }

    /// Create a leaf node (atomic subtask with m=1)
    pub fn leaf(task_id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            task_id: task_id.into(),
            parent_id: None,
            m_value: 1,
            description: description.into(),
            context: serde_json::Value::Null,
            is_leaf: true,
            order: 0,
            metadata: HashMap::new(),
        }
    }

    /// Set the parent task ID
    pub fn with_parent(mut self, parent_id: impl Into<String>) -> Self {
        self.parent_id = Some(parent_id.into());
        self
    }

    /// Set the m value (steps per agent)
    pub fn with_m_value(mut self, m_value: usize) -> Self {
        self.m_value = m_value;
        self
    }

    /// Set the context
    pub fn with_context(mut self, context: serde_json::Value) -> Self {
        self.context = context;
        self
    }

    /// Set the execution order
    pub fn with_order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Mark as a leaf node
    pub fn as_leaf(mut self) -> Self {
        self.is_leaf = true;
        self.m_value = 1;
        self
    }

    /// Validate that leaf nodes have m=1
    pub fn validate(&self) -> Result<(), DecompositionError> {
        if self.is_leaf && self.m_value != 1 {
            return Err(DecompositionError::InvalidMValue {
                task_id: self.task_id.clone(),
                expected: 1,
                actual: self.m_value,
            });
        }
        Ok(())
    }
}

/// A proposed decomposition of a task
///
/// Contains the list of subtasks and the function to compose their results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecompositionProposal {
    /// Unique identifier for this proposal
    pub proposal_id: String,

    /// The task being decomposed
    pub source_task_id: TaskId,

    /// The resulting subtasks
    pub subtasks: Vec<DecompositionSubtask>,

    /// How to compose subtask results
    #[serde(default)]
    pub composition_fn: CompositionFunction,

    /// Confidence score from the decomposition agent (0.0-1.0)
    #[serde(default)]
    pub confidence: f64,

    /// Rationale for this decomposition
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,

    /// Optional metadata
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl DecompositionProposal {
    /// Create a new proposal
    pub fn new(
        proposal_id: impl Into<String>,
        source_task_id: impl Into<String>,
        subtasks: Vec<DecompositionSubtask>,
        composition_fn: CompositionFunction,
    ) -> Self {
        Self {
            proposal_id: proposal_id.into(),
            source_task_id: source_task_id.into(),
            subtasks,
            composition_fn,
            confidence: 0.0,
            rationale: None,
            metadata: HashMap::new(),
        }
    }

    /// Set confidence score
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set rationale
    pub fn with_rationale(mut self, rationale: impl Into<String>) -> Self {
        self.rationale = Some(rationale.into());
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Validate the proposal
    ///
    /// Checks:
    /// - All leaf nodes have m_value == 1
    /// - No empty subtask lists
    /// - All task IDs are unique
    pub fn validate(&self) -> Result<(), DecompositionError> {
        if self.subtasks.is_empty() {
            return Err(DecompositionError::EmptyDecomposition {
                task_id: self.source_task_id.clone(),
            });
        }

        // Check for duplicate task IDs
        let mut seen_ids = std::collections::HashSet::new();
        for subtask in &self.subtasks {
            if !seen_ids.insert(&subtask.task_id) {
                return Err(DecompositionError::DuplicateTaskId {
                    task_id: subtask.task_id.clone(),
                });
            }
            subtask.validate()?;
        }

        Ok(())
    }

    /// Count the total number of leaf nodes
    pub fn leaf_count(&self) -> usize {
        self.subtasks.iter().filter(|s| s.is_leaf).count()
    }

    /// Get all leaf nodes
    pub fn leaves(&self) -> Vec<&DecompositionSubtask> {
        self.subtasks.iter().filter(|s| s.is_leaf).collect()
    }
}

/// Errors that can occur during decomposition
#[derive(Debug, Clone, PartialEq)]
pub enum DecompositionError {
    /// Leaf node has invalid m value (must be 1)
    InvalidMValue {
        task_id: TaskId,
        expected: usize,
        actual: usize,
    },
    /// Decomposition resulted in no subtasks
    EmptyDecomposition { task_id: TaskId },
    /// Duplicate task ID in decomposition
    DuplicateTaskId { task_id: TaskId },
    /// Maximum recursion depth exceeded
    DepthLimitExceeded { depth: usize, limit: usize },
    /// Cycle detected in decomposition tree
    CycleDetected { task_id: TaskId },
    /// Decomposition timed out
    Timeout { elapsed_ms: u64, limit_ms: u64 },
    /// Agent failed to produce a valid proposal
    AgentError { message: String },
    /// Validation failed
    ValidationError { message: String },
}

impl std::fmt::Display for DecompositionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMValue {
                task_id,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Leaf node '{}' has m_value={}, expected {}",
                    task_id, actual, expected
                )
            }
            Self::EmptyDecomposition { task_id } => {
                write!(f, "Decomposition of '{}' resulted in no subtasks", task_id)
            }
            Self::DuplicateTaskId { task_id } => {
                write!(f, "Duplicate task ID in decomposition: '{}'", task_id)
            }
            Self::DepthLimitExceeded { depth, limit } => {
                write!(f, "Decomposition depth {} exceeds limit {}", depth, limit)
            }
            Self::CycleDetected { task_id } => {
                write!(f, "Cycle detected in decomposition at task '{}'", task_id)
            }
            Self::Timeout {
                elapsed_ms,
                limit_ms,
            } => {
                write!(
                    f,
                    "Decomposition timed out after {}ms (limit: {}ms)",
                    elapsed_ms, limit_ms
                )
            }
            Self::AgentError { message } => {
                write!(f, "Decomposition agent error: {}", message)
            }
            Self::ValidationError { message } => {
                write!(f, "Validation error: {}", message)
            }
        }
    }
}

impl std::error::Error for DecompositionError {}

/// Configuration for decomposition operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionConfig {
    /// Maximum recursion depth (default: 10)
    #[serde(default = "default_depth_limit")]
    pub depth_limit: usize,

    /// Timeout in milliseconds (default: 60000 = 60s)
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,

    /// k-margin for decomposition voting
    #[serde(default = "default_k_margin")]
    pub k_margin: usize,

    /// Whether to scale k with depth
    #[serde(default = "default_depth_scaling")]
    pub depth_scaling: bool,

    /// Minimum confidence threshold for accepting proposals
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f64,
}

fn default_depth_limit() -> usize {
    10
}
fn default_timeout_ms() -> u64 {
    60_000
}
fn default_k_margin() -> usize {
    3
}
fn default_depth_scaling() -> bool {
    true
}
fn default_min_confidence() -> f64 {
    0.0
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            depth_limit: default_depth_limit(),
            timeout_ms: default_timeout_ms(),
            k_margin: default_k_margin(),
            depth_scaling: default_depth_scaling(),
            min_confidence: default_min_confidence(),
        }
    }
}

impl DecompositionConfig {
    /// Get k-margin adjusted for current depth
    pub fn k_for_depth(&self, depth: usize) -> usize {
        if self.depth_scaling {
            // Scale k logarithmically with depth: k' = k * (1 + ln(depth + 1))
            let scale = 1.0 + (depth as f64 + 1.0).ln();
            ((self.k_margin as f64) * scale).ceil() as usize
        } else {
            self.k_margin
        }
    }
}

/// Trait for decomposition agents
///
/// Implement this trait to create domain-specific decomposition strategies.
/// The agent proposes how to split a complex task into simpler subtasks.
pub trait DecompositionAgent: Send + Sync {
    /// Propose a decomposition for the given task
    ///
    /// # Arguments
    ///
    /// * `task_id` - Identifier for the task to decompose
    /// * `description` - Human-readable description of the task
    /// * `context` - Current context/state for the task
    /// * `depth` - Current recursion depth (0 for root)
    ///
    /// # Returns
    ///
    /// A `DecompositionProposal` describing how to split the task,
    /// or an error if decomposition fails.
    fn propose_decomposition(
        &self,
        task_id: &str,
        description: &str,
        context: &serde_json::Value,
        depth: usize,
    ) -> Result<DecompositionProposal, DecompositionError>;

    /// Check if a task should be treated as a leaf node
    ///
    /// Returns `true` if the task is atomic and should not be further decomposed.
    /// Default implementation returns `false` (always attempt decomposition).
    fn is_atomic(&self, _task_id: &str, _description: &str) -> bool {
        false
    }

    /// Get the name of this decomposition agent
    fn name(&self) -> &str;
}

/// A simple pass-through decomposition agent that treats all tasks as atomic
///
/// Useful for testing or when decomposition is not needed.
#[derive(Debug, Clone, Default)]
pub struct IdentityDecomposer;

impl DecompositionAgent for IdentityDecomposer {
    fn propose_decomposition(
        &self,
        task_id: &str,
        description: &str,
        context: &serde_json::Value,
        _depth: usize,
    ) -> Result<DecompositionProposal, DecompositionError> {
        let subtask =
            DecompositionSubtask::leaf(task_id, description).with_context(context.clone());

        Ok(DecompositionProposal::new(
            format!("identity-{}", task_id),
            task_id,
            vec![subtask],
            CompositionFunction::Sequential,
        )
        .with_confidence(1.0)
        .with_rationale("Identity decomposition: task is already atomic"))
    }

    fn is_atomic(&self, _task_id: &str, _description: &str) -> bool {
        true
    }

    fn name(&self) -> &str {
        "identity"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ==========================================
    // CompositionFunction Tests
    // ==========================================

    #[test]
    fn test_composition_function_default() {
        assert_eq!(
            CompositionFunction::default(),
            CompositionFunction::Sequential
        );
    }

    #[test]
    fn test_composition_function_serialization() {
        let funcs = vec![
            CompositionFunction::Sequential,
            CompositionFunction::Parallel {
                merge_strategy: MergeStrategy::Concatenate,
            },
            CompositionFunction::Conditional {
                condition: "x > 0".to_string(),
            },
            CompositionFunction::Custom {
                name: "my_fn".to_string(),
                params: HashMap::new(),
            },
        ];

        for func in funcs {
            let json = serde_json::to_string(&func).unwrap();
            let parsed: CompositionFunction = serde_json::from_str(&json).unwrap();
            assert_eq!(func, parsed);
        }
    }

    #[test]
    fn test_merge_strategy_default() {
        assert_eq!(MergeStrategy::default(), MergeStrategy::Concatenate);
    }

    // ==========================================
    // DecompositionSubtask Tests
    // ==========================================

    #[test]
    fn test_subtask_new() {
        let subtask = DecompositionSubtask::new("task-1", "Do something");
        assert_eq!(subtask.task_id, "task-1");
        assert_eq!(subtask.description, "Do something");
        assert_eq!(subtask.m_value, 1);
        assert!(!subtask.is_leaf);
    }

    #[test]
    fn test_subtask_leaf() {
        let subtask = DecompositionSubtask::leaf("leaf-1", "Atomic task");
        assert!(subtask.is_leaf);
        assert_eq!(subtask.m_value, 1);
    }

    #[test]
    fn test_subtask_builder_methods() {
        let subtask = DecompositionSubtask::new("task-1", "Test")
            .with_parent("parent-1")
            .with_m_value(5)
            .with_context(json!({"key": "value"}))
            .with_order(3)
            .with_metadata("domain", json!("coding"));

        assert_eq!(subtask.parent_id, Some("parent-1".to_string()));
        assert_eq!(subtask.m_value, 5);
        assert_eq!(subtask.context, json!({"key": "value"}));
        assert_eq!(subtask.order, 3);
        assert_eq!(subtask.metadata.get("domain"), Some(&json!("coding")));
    }

    #[test]
    fn test_subtask_as_leaf() {
        let subtask = DecompositionSubtask::new("task-1", "Test")
            .with_m_value(5)
            .as_leaf();

        assert!(subtask.is_leaf);
        assert_eq!(subtask.m_value, 1); // as_leaf forces m=1
    }

    #[test]
    fn test_subtask_validate_leaf_m1() {
        let valid_leaf = DecompositionSubtask::leaf("leaf-1", "Valid");
        assert!(valid_leaf.validate().is_ok());

        let mut invalid_leaf = DecompositionSubtask::leaf("leaf-2", "Invalid");
        invalid_leaf.m_value = 2; // Manually set invalid m_value
        assert!(matches!(
            invalid_leaf.validate(),
            Err(DecompositionError::InvalidMValue { .. })
        ));
    }

    #[test]
    fn test_subtask_serialization_roundtrip() {
        let subtask = DecompositionSubtask::new("task-1", "Test task")
            .with_parent("parent-1")
            .with_context(json!({"state": [1, 2, 3]}))
            .with_metadata("domain", json!("test"));

        let json = serde_json::to_string(&subtask).unwrap();
        let parsed: DecompositionSubtask = serde_json::from_str(&json).unwrap();

        assert_eq!(subtask.task_id, parsed.task_id);
        assert_eq!(subtask.parent_id, parsed.parent_id);
        assert_eq!(subtask.m_value, parsed.m_value);
        assert_eq!(subtask.description, parsed.description);
        assert_eq!(subtask.context, parsed.context);
    }

    // ==========================================
    // DecompositionProposal Tests
    // ==========================================

    #[test]
    fn test_proposal_new() {
        let proposal = DecompositionProposal::new(
            "proposal-1",
            "source-task",
            vec![DecompositionSubtask::leaf("sub-1", "Subtask 1")],
            CompositionFunction::Sequential,
        );

        assert_eq!(proposal.proposal_id, "proposal-1");
        assert_eq!(proposal.source_task_id, "source-task");
        assert_eq!(proposal.subtasks.len(), 1);
    }

    #[test]
    fn test_proposal_builder_methods() {
        let proposal = DecompositionProposal::new(
            "p-1",
            "src",
            vec![DecompositionSubtask::leaf("s-1", "Sub")],
            CompositionFunction::Sequential,
        )
        .with_confidence(0.95)
        .with_rationale("Good decomposition")
        .with_metadata("version", json!("1.0"));

        assert!((proposal.confidence - 0.95).abs() < f64::EPSILON);
        assert_eq!(proposal.rationale, Some("Good decomposition".to_string()));
        assert_eq!(proposal.metadata.get("version"), Some(&json!("1.0")));
    }

    #[test]
    fn test_proposal_confidence_clamped() {
        let proposal = DecompositionProposal::new(
            "p-1",
            "src",
            vec![DecompositionSubtask::leaf("s-1", "Sub")],
            CompositionFunction::Sequential,
        )
        .with_confidence(1.5); // Should be clamped to 1.0

        assert!((proposal.confidence - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_proposal_validate_empty() {
        let proposal = DecompositionProposal::new(
            "p-1",
            "src",
            vec![], // Empty subtasks
            CompositionFunction::Sequential,
        );

        assert!(matches!(
            proposal.validate(),
            Err(DecompositionError::EmptyDecomposition { .. })
        ));
    }

    #[test]
    fn test_proposal_validate_duplicate_ids() {
        let proposal = DecompositionProposal::new(
            "p-1",
            "src",
            vec![
                DecompositionSubtask::leaf("dup-id", "First"),
                DecompositionSubtask::leaf("dup-id", "Second"),
            ],
            CompositionFunction::Sequential,
        );

        assert!(matches!(
            proposal.validate(),
            Err(DecompositionError::DuplicateTaskId { .. })
        ));
    }

    #[test]
    fn test_proposal_validate_invalid_leaf() {
        let mut subtask = DecompositionSubtask::leaf("leaf-1", "Invalid");
        subtask.m_value = 2;

        let proposal = DecompositionProposal::new(
            "p-1",
            "src",
            vec![subtask],
            CompositionFunction::Sequential,
        );

        assert!(matches!(
            proposal.validate(),
            Err(DecompositionError::InvalidMValue { .. })
        ));
    }

    #[test]
    fn test_proposal_validate_success() {
        let proposal = DecompositionProposal::new(
            "p-1",
            "src",
            vec![
                DecompositionSubtask::leaf("leaf-1", "First"),
                DecompositionSubtask::leaf("leaf-2", "Second"),
            ],
            CompositionFunction::Sequential,
        );

        assert!(proposal.validate().is_ok());
    }

    #[test]
    fn test_proposal_leaf_count() {
        let proposal = DecompositionProposal::new(
            "p-1",
            "src",
            vec![
                DecompositionSubtask::leaf("leaf-1", "Leaf"),
                DecompositionSubtask::new("non-leaf", "Not a leaf"),
                DecompositionSubtask::leaf("leaf-2", "Another leaf"),
            ],
            CompositionFunction::Sequential,
        );

        assert_eq!(proposal.leaf_count(), 2);
    }

    #[test]
    fn test_proposal_leaves() {
        let proposal = DecompositionProposal::new(
            "p-1",
            "src",
            vec![
                DecompositionSubtask::leaf("leaf-1", "Leaf"),
                DecompositionSubtask::new("non-leaf", "Not a leaf"),
            ],
            CompositionFunction::Sequential,
        );

        let leaves = proposal.leaves();
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].task_id, "leaf-1");
    }

    #[test]
    fn test_proposal_serialization_roundtrip() {
        let proposal = DecompositionProposal::new(
            "p-1",
            "source",
            vec![
                DecompositionSubtask::leaf("s-1", "Sub 1").with_order(0),
                DecompositionSubtask::leaf("s-2", "Sub 2").with_order(1),
            ],
            CompositionFunction::Parallel {
                merge_strategy: MergeStrategy::DeepMerge,
            },
        )
        .with_confidence(0.85)
        .with_rationale("Test rationale");

        let json = serde_json::to_string(&proposal).unwrap();
        let parsed: DecompositionProposal = serde_json::from_str(&json).unwrap();

        assert_eq!(proposal.proposal_id, parsed.proposal_id);
        assert_eq!(proposal.source_task_id, parsed.source_task_id);
        assert_eq!(proposal.subtasks.len(), parsed.subtasks.len());
        assert_eq!(proposal.composition_fn, parsed.composition_fn);
        assert!((proposal.confidence - parsed.confidence).abs() < f64::EPSILON);
        assert_eq!(proposal.rationale, parsed.rationale);
    }

    // ==========================================
    // DecompositionError Tests
    // ==========================================

    #[test]
    fn test_error_display() {
        let errors = vec![
            DecompositionError::InvalidMValue {
                task_id: "t1".to_string(),
                expected: 1,
                actual: 2,
            },
            DecompositionError::EmptyDecomposition {
                task_id: "t2".to_string(),
            },
            DecompositionError::DuplicateTaskId {
                task_id: "t3".to_string(),
            },
            DecompositionError::DepthLimitExceeded {
                depth: 11,
                limit: 10,
            },
            DecompositionError::CycleDetected {
                task_id: "t4".to_string(),
            },
            DecompositionError::Timeout {
                elapsed_ms: 65000,
                limit_ms: 60000,
            },
            DecompositionError::AgentError {
                message: "test error".to_string(),
            },
            DecompositionError::ValidationError {
                message: "invalid".to_string(),
            },
        ];

        for error in errors {
            let display = format!("{}", error);
            assert!(!display.is_empty());
        }
    }

    // ==========================================
    // DecompositionConfig Tests
    // ==========================================

    #[test]
    fn test_config_default() {
        let config = DecompositionConfig::default();
        assert_eq!(config.depth_limit, 10);
        assert_eq!(config.timeout_ms, 60_000);
        assert_eq!(config.k_margin, 3);
        assert!(config.depth_scaling);
    }

    #[test]
    fn test_config_k_for_depth_no_scaling() {
        let config = DecompositionConfig {
            depth_scaling: false,
            k_margin: 5,
            ..Default::default()
        };

        assert_eq!(config.k_for_depth(0), 5);
        assert_eq!(config.k_for_depth(5), 5);
        assert_eq!(config.k_for_depth(10), 5);
    }

    #[test]
    fn test_config_k_for_depth_with_scaling() {
        let config = DecompositionConfig {
            depth_scaling: true,
            k_margin: 3,
            ..Default::default()
        };

        // k' = k * (1 + ln(depth + 1))
        // depth=0: k' = 3 * (1 + ln(1)) = 3 * 1 = 3
        // depth=1: k' = 3 * (1 + ln(2)) ≈ 3 * 1.69 ≈ 6
        // depth=4: k' = 3 * (1 + ln(5)) ≈ 3 * 2.61 ≈ 8

        assert_eq!(config.k_for_depth(0), 3);
        assert!(config.k_for_depth(1) > 3);
        assert!(config.k_for_depth(4) > config.k_for_depth(1));
    }

    #[test]
    fn test_config_serialization() {
        let config = DecompositionConfig {
            depth_limit: 5,
            timeout_ms: 30_000,
            k_margin: 2,
            depth_scaling: false,
            min_confidence: 0.5,
        };

        let json = serde_json::to_string(&config).unwrap();
        let parsed: DecompositionConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.depth_limit, parsed.depth_limit);
        assert_eq!(config.timeout_ms, parsed.timeout_ms);
        assert_eq!(config.k_margin, parsed.k_margin);
        assert_eq!(config.depth_scaling, parsed.depth_scaling);
    }

    // ==========================================
    // DecompositionAgent Trait Tests
    // ==========================================

    #[test]
    fn test_identity_decomposer() {
        let decomposer = IdentityDecomposer;

        assert_eq!(decomposer.name(), "identity");
        assert!(decomposer.is_atomic("any-task", "any description"));

        let result = decomposer.propose_decomposition(
            "task-1",
            "Test task",
            &json!({"state": "initial"}),
            0,
        );

        assert!(result.is_ok());
        let proposal = result.unwrap();

        assert_eq!(proposal.source_task_id, "task-1");
        assert_eq!(proposal.subtasks.len(), 1);
        assert!(proposal.subtasks[0].is_leaf);
        assert_eq!(proposal.subtasks[0].m_value, 1);
        assert!((proposal.confidence - 1.0).abs() < f64::EPSILON);
    }

    // ==========================================
    // m=1 Enforcement Property Tests
    // ==========================================

    #[test]
    fn test_m1_enforcement_on_leaf_creation() {
        // leaf() constructor always creates m=1
        for _ in 0..100 {
            let leaf = DecompositionSubtask::leaf("test", "desc");
            assert_eq!(leaf.m_value, 1);
            assert!(leaf.is_leaf);
            assert!(leaf.validate().is_ok());
        }
    }

    #[test]
    fn test_m1_enforcement_on_as_leaf() {
        // as_leaf() forces m=1 regardless of previous value
        for m in 0..10 {
            let subtask = DecompositionSubtask::new("test", "desc")
                .with_m_value(m)
                .as_leaf();
            assert_eq!(subtask.m_value, 1);
            assert!(subtask.is_leaf);
            assert!(subtask.validate().is_ok());
        }
    }

    #[test]
    fn test_m1_validation_rejects_invalid() {
        // Validation catches manual m_value tampering on leaves
        for m in [0, 2, 3, 5, 10, 100] {
            let mut leaf = DecompositionSubtask::leaf("test", "desc");
            leaf.m_value = m;
            assert!(leaf.validate().is_err());
        }
    }
}
