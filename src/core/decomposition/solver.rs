//! Problem Solver Agent Interface for MAKER Framework
//!
//! Implements the execution layer for atomic (m=1) leaf nodes in the decomposition
//! tree. The `LeafNodeExecutor` uses `vote_with_margin()` to execute each subtask
//! with error correction.
//!
//! # Architecture
//!
//! ```text
//! DecompositionProposal
//!       ↓
//! Extract leaf nodes (is_leaf == true, m_value == 1)
//!       ↓
//! LeafNodeExecutor.execute_leaf()
//!       ↓
//! vote_with_margin() for each leaf
//!       ↓
//! SubtaskResult { output, state, metrics }
//! ```
//!
//! # Key Principles
//!
//! 1. **m=1 Enforcement**: Only leaf nodes with m_value == 1 can be executed
//! 2. **State Passing**: Parent context flows to child, child result flows back
//! 3. **Partial Failure**: Retry mechanism with configurable backoff
//! 4. **Event Emission**: All execution steps emit observable events

use super::{DecompositionError, DecompositionSubtask, TaskId};
use crate::core::executor::{LlmClient, VoteConfig, VoteError};
use crate::events::MakerEvent;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Result of executing a single subtask
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtaskResult {
    /// The task ID that was executed
    pub task_id: TaskId,

    /// The winning response from voting
    pub output: String,

    /// The resulting state after execution
    pub state: serde_json::Value,

    /// Whether execution succeeded
    pub success: bool,

    /// Error message if execution failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,

    /// Execution metrics
    pub metrics: ExecutionMetrics,
}

impl SubtaskResult {
    /// Create a successful result
    pub fn success(task_id: TaskId, output: String, state: serde_json::Value) -> Self {
        Self {
            task_id,
            output,
            state,
            success: true,
            error: None,
            metrics: ExecutionMetrics::default(),
        }
    }

    /// Create a failed result
    pub fn failure(task_id: TaskId, error: String) -> Self {
        Self {
            task_id,
            output: String::new(),
            state: serde_json::Value::Null,
            success: false,
            error: Some(error),
            metrics: ExecutionMetrics::default(),
        }
    }

    /// Set metrics
    pub fn with_metrics(mut self, metrics: ExecutionMetrics) -> Self {
        self.metrics = metrics;
        self
    }
}

/// Metrics from subtask execution
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionMetrics {
    /// Total samples collected during voting
    pub samples_collected: usize,

    /// Number of red-flagged samples
    pub red_flagged: usize,

    /// k-margin used for voting
    pub k_used: usize,

    /// Time taken for execution
    pub elapsed_ms: u64,

    /// Number of retry attempts (0 = succeeded first try)
    pub retry_count: usize,

    /// Input tokens consumed
    pub input_tokens: usize,

    /// Output tokens consumed
    pub output_tokens: usize,
}

/// Configuration for leaf node execution
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Maximum retry attempts for failed executions
    pub max_retries: usize,

    /// Base delay between retries (exponential backoff)
    pub retry_base_delay: Duration,

    /// Maximum delay between retries
    pub retry_max_delay: Duration,

    /// Voting configuration
    pub vote_config: VoteConfig,

    /// k-margin for voting (if not using adaptive)
    pub k_margin: usize,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_base_delay: Duration::from_millis(100),
            retry_max_delay: Duration::from_secs(5),
            vote_config: VoteConfig::default(),
            k_margin: 3,
        }
    }
}

impl ExecutorConfig {
    /// Create config with custom k-margin
    pub fn with_k_margin(mut self, k: usize) -> Self {
        self.k_margin = k;
        self
    }

    /// Create config with custom max retries
    pub fn with_max_retries(mut self, retries: usize) -> Self {
        self.max_retries = retries;
        self
    }

    /// Create config with custom vote config
    pub fn with_vote_config(mut self, config: VoteConfig) -> Self {
        self.vote_config = config;
        self
    }

    /// Calculate delay for a given retry attempt (exponential backoff)
    pub fn retry_delay(&self, attempt: usize) -> Duration {
        let delay = self.retry_base_delay.mul_f64(2.0_f64.powi(attempt as i32));
        delay.min(self.retry_max_delay)
    }
}

/// Errors during leaf node execution
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionError {
    /// Attempted to execute a non-leaf node
    NotLeafNode { task_id: TaskId },

    /// Leaf node has invalid m_value (must be 1)
    InvalidMValue { task_id: TaskId, m_value: usize },

    /// Voting failed after all retries
    VotingFailed {
        task_id: TaskId,
        attempts: usize,
        last_error: String,
    },

    /// Execution timed out
    Timeout { task_id: TaskId, elapsed_ms: u64 },

    /// State validation failed
    StateValidationFailed { task_id: TaskId, message: String },
}

impl std::fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotLeafNode { task_id } => {
                write!(f, "Task '{}' is not a leaf node", task_id)
            }
            Self::InvalidMValue { task_id, m_value } => {
                write!(
                    f,
                    "Leaf node '{}' has invalid m_value {}, expected 1",
                    task_id, m_value
                )
            }
            Self::VotingFailed {
                task_id,
                attempts,
                last_error,
            } => {
                write!(
                    f,
                    "Voting failed for '{}' after {} attempts: {}",
                    task_id, attempts, last_error
                )
            }
            Self::Timeout {
                task_id,
                elapsed_ms,
            } => {
                write!(
                    f,
                    "Execution of '{}' timed out after {}ms",
                    task_id, elapsed_ms
                )
            }
            Self::StateValidationFailed { task_id, message } => {
                write!(f, "State validation failed for '{}': {}", task_id, message)
            }
        }
    }
}

impl std::error::Error for ExecutionError {}

impl From<ExecutionError> for DecompositionError {
    fn from(e: ExecutionError) -> Self {
        DecompositionError::AgentError {
            message: e.to_string(),
        }
    }
}

/// Executor for atomic (m=1) leaf nodes
///
/// Uses `vote_with_margin()` to execute each subtask with error correction.
/// Enforces the m=1 constraint: only leaf nodes with m_value == 1 can be executed.
pub struct LeafNodeExecutor {
    /// Execution configuration
    config: ExecutorConfig,

    /// Optional event emitter for observability
    event_emitter: Option<Arc<dyn Fn(MakerEvent) + Send + Sync>>,

    /// Results from executed subtasks
    results: HashMap<TaskId, SubtaskResult>,
}

impl LeafNodeExecutor {
    /// Create a new executor with default configuration
    pub fn new() -> Self {
        Self {
            config: ExecutorConfig::default(),
            event_emitter: None,
            results: HashMap::new(),
        }
    }

    /// Create a new executor with custom configuration
    pub fn with_config(config: ExecutorConfig) -> Self {
        Self {
            config,
            event_emitter: None,
            results: HashMap::new(),
        }
    }

    /// Set an event emitter for observability
    pub fn with_event_emitter(mut self, emitter: Arc<dyn Fn(MakerEvent) + Send + Sync>) -> Self {
        self.event_emitter = Some(emitter);
        self
    }

    /// Get the configuration
    pub fn config(&self) -> &ExecutorConfig {
        &self.config
    }

    /// Get results from executed subtasks
    pub fn results(&self) -> &HashMap<TaskId, SubtaskResult> {
        &self.results
    }

    /// Get a specific result
    pub fn get_result(&self, task_id: &str) -> Option<&SubtaskResult> {
        self.results.get(task_id)
    }

    /// Validate that a subtask can be executed
    fn validate_leaf(&self, subtask: &DecompositionSubtask) -> Result<(), ExecutionError> {
        if !subtask.is_leaf {
            return Err(ExecutionError::NotLeafNode {
                task_id: subtask.task_id.clone(),
            });
        }

        if subtask.m_value != 1 {
            return Err(ExecutionError::InvalidMValue {
                task_id: subtask.task_id.clone(),
                m_value: subtask.m_value,
            });
        }

        Ok(())
    }

    /// Build a prompt for the subtask
    fn build_prompt(
        &self,
        subtask: &DecompositionSubtask,
        parent_state: &serde_json::Value,
    ) -> String {
        // Combine subtask description with context and parent state
        let mut prompt = subtask.description.clone();

        if !subtask.context.is_null() {
            prompt.push_str("\n\nContext:\n");
            prompt.push_str(&serde_json::to_string_pretty(&subtask.context).unwrap_or_default());
        }

        if !parent_state.is_null() {
            prompt.push_str("\n\nCurrent State:\n");
            prompt.push_str(&serde_json::to_string_pretty(parent_state).unwrap_or_default());
        }

        prompt
    }

    /// Emit a subtask started event
    fn emit_subtask_started(&self, task_id: &str, parent_id: Option<&str>) {
        if let Some(ref emitter) = self.event_emitter {
            emitter(MakerEvent::subtask_started(task_id, parent_id));
        }
    }

    /// Emit a subtask completed event
    fn emit_subtask_completed(&self, task_id: &str, success: bool, elapsed_ms: u64) {
        if let Some(ref emitter) = self.event_emitter {
            emitter(MakerEvent::subtask_completed(task_id, success, elapsed_ms));
        }
    }

    /// Execute a single leaf node with voting
    ///
    /// # Arguments
    ///
    /// * `subtask` - The leaf subtask to execute (must have is_leaf == true, m_value == 1)
    /// * `parent_state` - State from the parent task to pass to this subtask
    /// * `client` - The LLM client to use for generation
    ///
    /// # Returns
    ///
    /// * `Ok(SubtaskResult)` - The result of execution
    /// * `Err(ExecutionError)` - If execution fails
    pub fn execute_leaf(
        &mut self,
        subtask: &DecompositionSubtask,
        parent_state: &serde_json::Value,
        client: &dyn LlmClient,
    ) -> Result<SubtaskResult, ExecutionError> {
        // Validate m=1 constraint
        self.validate_leaf(subtask)?;

        let start = Instant::now();
        self.emit_subtask_started(&subtask.task_id, subtask.parent_id.as_deref());

        // Build prompt with parent state
        let prompt = self.build_prompt(subtask, parent_state);

        // Execute with retries
        let mut last_error = String::new();
        let mut total_attempts = 0;

        for attempt in 0..=self.config.max_retries {
            total_attempts = attempt + 1;

            // Execute voting
            let vote_result = crate::core::executor::vote_with_margin(
                &prompt,
                self.config.k_margin,
                client,
                self.config.vote_config.clone(),
            );

            match vote_result {
                Ok(result) => {
                    let elapsed = start.elapsed();
                    let metrics = ExecutionMetrics {
                        samples_collected: result.total_samples,
                        red_flagged: result.red_flagged,
                        k_used: result.k_used,
                        elapsed_ms: elapsed.as_millis() as u64,
                        retry_count: attempt,
                        input_tokens: result.cost.input_tokens,
                        output_tokens: result.cost.output_tokens,
                    };

                    // Parse output to extract next state if possible
                    let next_state = self.extract_state(&result.winner, parent_state);

                    let subtask_result =
                        SubtaskResult::success(subtask.task_id.clone(), result.winner, next_state)
                            .with_metrics(metrics);

                    // Store result
                    self.results
                        .insert(subtask.task_id.clone(), subtask_result.clone());

                    self.emit_subtask_completed(&subtask.task_id, true, elapsed.as_millis() as u64);

                    return Ok(subtask_result);
                }
                Err(e) => {
                    last_error = e.to_string();

                    // Don't retry on certain errors
                    if matches!(e, VoteError::InvalidConfig { .. }) {
                        break;
                    }

                    // Wait before retry (if not last attempt)
                    if attempt < self.config.max_retries {
                        let delay = self.config.retry_delay(attempt);
                        std::thread::sleep(delay);
                    }
                }
            }
        }

        // All retries exhausted
        let elapsed = start.elapsed();
        let metrics = ExecutionMetrics {
            elapsed_ms: elapsed.as_millis() as u64,
            retry_count: total_attempts - 1,
            ..Default::default()
        };

        let result = SubtaskResult::failure(subtask.task_id.clone(), last_error.clone())
            .with_metrics(metrics);

        self.results.insert(subtask.task_id.clone(), result);
        self.emit_subtask_completed(&subtask.task_id, false, elapsed.as_millis() as u64);

        Err(ExecutionError::VotingFailed {
            task_id: subtask.task_id.clone(),
            attempts: total_attempts,
            last_error,
        })
    }

    /// Execute multiple leaf nodes in sequence
    ///
    /// State flows from parent to first child, and from each child to the next.
    ///
    /// # Arguments
    ///
    /// * `subtasks` - The leaf subtasks to execute (must be sorted by order)
    /// * `initial_state` - Initial state for the first subtask
    /// * `client` - The LLM client to use for generation
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<SubtaskResult>)` - Results in execution order
    /// * `Err(ExecutionError)` - If any execution fails (partial results available via `results()`)
    pub fn execute_sequential(
        &mut self,
        subtasks: &[DecompositionSubtask],
        initial_state: &serde_json::Value,
        client: &dyn LlmClient,
    ) -> Result<Vec<SubtaskResult>, ExecutionError> {
        let mut results = Vec::with_capacity(subtasks.len());
        let mut current_state = initial_state.clone();

        for subtask in subtasks {
            let result = self.execute_leaf(subtask, &current_state, client)?;

            // State flows to next subtask
            current_state = result.state.clone();
            results.push(result);
        }

        Ok(results)
    }

    /// Extract state from output (best-effort parsing)
    ///
    /// Tries to parse the output as JSON to extract a "state" or "next_state" field.
    /// Falls back to wrapping the output in a state object.
    fn extract_state(&self, output: &str, fallback: &serde_json::Value) -> serde_json::Value {
        // Try to parse as JSON
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(output) {
            // Look for explicit state fields
            if let Some(state) = parsed.get("next_state") {
                return state.clone();
            }
            if let Some(state) = parsed.get("state") {
                return state.clone();
            }
            // Return the whole parsed object
            return parsed;
        }

        // Fallback: wrap output as a "result" field in the previous state
        if let serde_json::Value::Object(mut map) = fallback.clone() {
            map.insert(
                "result".to_string(),
                serde_json::Value::String(output.to_string()),
            );
            return serde_json::Value::Object(map);
        }

        // Ultimate fallback: just the output
        serde_json::json!({ "result": output })
    }

    /// Clear all stored results
    pub fn clear_results(&mut self) {
        self.results.clear();
    }

    /// Execute a DecompositionProposal based on its composition function
    ///
    /// Routes to the appropriate execution strategy based on the composition function:
    /// - Sequential: Execute in order, passing state between tasks
    /// - Parallel: Execute all tasks concurrently, merge results
    /// - Conditional: Evaluate condition and execute appropriate branch
    /// - Custom: Delegate to custom handler
    ///
    /// # Arguments
    ///
    /// * `proposal` - The decomposition proposal to execute
    /// * `initial_state` - Initial state for execution
    /// * `client` - The LLM client to use for generation
    ///
    /// # Returns
    ///
    /// * `Ok(ProposalResult)` - Combined result from all subtasks
    /// * `Err(ExecutionError)` - If execution fails
    pub fn execute_proposal(
        &mut self,
        proposal: &super::DecompositionProposal,
        initial_state: &serde_json::Value,
        client: &dyn LlmClient,
    ) -> Result<ProposalResult, ExecutionError> {
        use super::CompositionFunction;

        // Validate the proposal first
        proposal
            .validate()
            .map_err(|e| ExecutionError::StateValidationFailed {
                task_id: proposal.source_task_id.clone(),
                message: e.to_string(),
            })?;

        // Get leaf nodes sorted by order
        let mut leaves: Vec<_> = proposal.leaves().into_iter().cloned().collect();
        leaves.sort_by_key(|s| s.order);

        if leaves.is_empty() {
            return Err(ExecutionError::StateValidationFailed {
                task_id: proposal.source_task_id.clone(),
                message: "No leaf nodes to execute".to_string(),
            });
        }

        match &proposal.composition_fn {
            CompositionFunction::Sequential => {
                let results = self.execute_sequential(&leaves, initial_state, client)?;
                let final_state = results
                    .last()
                    .map(|r| r.state.clone())
                    .unwrap_or_else(|| initial_state.clone());

                Ok(ProposalResult {
                    proposal_id: proposal.proposal_id.clone(),
                    subtask_results: results,
                    final_state,
                    composition_fn: "Sequential".to_string(),
                })
            }
            CompositionFunction::Parallel { merge_strategy } => {
                // Execute all leaves with initial state (no state passing between them)
                let mut results = Vec::with_capacity(leaves.len());
                for leaf in &leaves {
                    match self.execute_leaf(leaf, initial_state, client) {
                        Ok(result) => results.push(result),
                        Err(e) => {
                            // On failure, return partial results
                            return Err(e);
                        }
                    }
                }

                // Merge results based on strategy
                let final_state =
                    Self::merge_parallel_results(&results, merge_strategy, initial_state);

                Ok(ProposalResult {
                    proposal_id: proposal.proposal_id.clone(),
                    subtask_results: results,
                    final_state,
                    composition_fn: format!("Parallel({:?})", merge_strategy),
                })
            }
            CompositionFunction::Conditional { condition } => {
                // For conditional, we evaluate the condition against the initial state
                // and execute only the appropriate branch
                // Simple implementation: first subtask if condition truthy, second if falsy
                let branch_idx = if Self::evaluate_condition(condition, initial_state) {
                    0
                } else {
                    1.min(leaves.len().saturating_sub(1))
                };

                if branch_idx < leaves.len() {
                    let result = self.execute_leaf(&leaves[branch_idx], initial_state, client)?;
                    Ok(ProposalResult {
                        proposal_id: proposal.proposal_id.clone(),
                        subtask_results: vec![result.clone()],
                        final_state: result.state,
                        composition_fn: format!("Conditional(branch={})", branch_idx),
                    })
                } else {
                    Err(ExecutionError::StateValidationFailed {
                        task_id: proposal.source_task_id.clone(),
                        message: "No branches available for conditional".to_string(),
                    })
                }
            }
            CompositionFunction::Custom { name, .. } => {
                // Custom composition defaults to sequential for now
                // Real implementation would look up custom handlers
                let results = self.execute_sequential(&leaves, initial_state, client)?;
                let final_state = results
                    .last()
                    .map(|r| r.state.clone())
                    .unwrap_or_else(|| initial_state.clone());

                Ok(ProposalResult {
                    proposal_id: proposal.proposal_id.clone(),
                    subtask_results: results,
                    final_state,
                    composition_fn: format!("Custom({})", name),
                })
            }
        }
    }

    /// Merge results from parallel execution
    fn merge_parallel_results(
        results: &[SubtaskResult],
        strategy: &super::MergeStrategy,
        fallback: &serde_json::Value,
    ) -> serde_json::Value {
        use super::MergeStrategy;

        match strategy {
            MergeStrategy::Concatenate => {
                // Concatenate all outputs
                let outputs: Vec<String> = results.iter().map(|r| r.output.clone()).collect();
                serde_json::json!({
                    "results": outputs,
                    "count": results.len()
                })
            }
            MergeStrategy::FirstSuccess => results
                .iter()
                .find(|r| r.success)
                .map(|r| r.state.clone())
                .unwrap_or_else(|| fallback.clone()),
            MergeStrategy::LastSuccess => results
                .iter()
                .rev()
                .find(|r| r.success)
                .map(|r| r.state.clone())
                .unwrap_or_else(|| fallback.clone()),
            MergeStrategy::CollectArray => {
                let states: Vec<_> = results.iter().map(|r| r.state.clone()).collect();
                serde_json::json!(states)
            }
            MergeStrategy::DeepMerge => {
                // Start with fallback and merge each result's state
                let mut merged = fallback.clone();
                for result in results {
                    if let (
                        serde_json::Value::Object(ref mut base),
                        serde_json::Value::Object(ref update),
                    ) = (&mut merged, &result.state)
                    {
                        for (key, value) in update {
                            base.insert(key.clone(), value.clone());
                        }
                    }
                }
                merged
            }
        }
    }

    /// Simple condition evaluation (checks if condition key exists and is truthy in state)
    fn evaluate_condition(condition: &str, state: &serde_json::Value) -> bool {
        // Simple implementation: check if the condition as a key exists in state and is truthy
        if let serde_json::Value::Object(map) = state {
            if let Some(value) = map.get(condition) {
                return match value {
                    serde_json::Value::Bool(b) => *b,
                    serde_json::Value::Number(n) => n.as_f64().unwrap_or(0.0) != 0.0,
                    serde_json::Value::String(s) => !s.is_empty(),
                    serde_json::Value::Array(a) => !a.is_empty(),
                    serde_json::Value::Object(o) => !o.is_empty(),
                    serde_json::Value::Null => false,
                };
            }
        }
        // Default: condition string is non-empty
        !condition.is_empty()
    }
}

/// Result of executing a full decomposition proposal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalResult {
    /// The proposal that was executed
    pub proposal_id: String,

    /// Results from all executed subtasks
    pub subtask_results: Vec<SubtaskResult>,

    /// The final merged state after composition
    pub final_state: serde_json::Value,

    /// The composition function that was used
    pub composition_fn: String,
}

impl ProposalResult {
    /// Check if all subtasks succeeded
    pub fn all_succeeded(&self) -> bool {
        self.subtask_results.iter().all(|r| r.success)
    }

    /// Count of successful subtasks
    pub fn success_count(&self) -> usize {
        self.subtask_results.iter().filter(|r| r.success).count()
    }

    /// Count of failed subtasks
    pub fn failure_count(&self) -> usize {
        self.subtask_results.iter().filter(|r| !r.success).count()
    }

    /// Total execution time across all subtasks
    pub fn total_elapsed_ms(&self) -> u64 {
        self.subtask_results
            .iter()
            .map(|r| r.metrics.elapsed_ms)
            .sum()
    }
}

impl Default for LeafNodeExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for LeafNodeExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeafNodeExecutor")
            .field("config", &self.config)
            .field("results_count", &self.results.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::executor::MockLlmClient;

    // ==========================================
    // SubtaskResult Tests
    // ==========================================

    #[test]
    fn test_subtask_result_success() {
        let result = SubtaskResult::success(
            "task-1".to_string(),
            "output".to_string(),
            serde_json::json!({"key": "value"}),
        );

        assert!(result.success);
        assert!(result.error.is_none());
        assert_eq!(result.task_id, "task-1");
        assert_eq!(result.output, "output");
    }

    #[test]
    fn test_subtask_result_failure() {
        let result = SubtaskResult::failure("task-1".to_string(), "error message".to_string());

        assert!(!result.success);
        assert_eq!(result.error, Some("error message".to_string()));
        assert!(result.output.is_empty());
    }

    #[test]
    fn test_subtask_result_with_metrics() {
        let metrics = ExecutionMetrics {
            samples_collected: 5,
            red_flagged: 1,
            k_used: 3,
            elapsed_ms: 100,
            ..Default::default()
        };

        let result =
            SubtaskResult::success("t".to_string(), "o".to_string(), serde_json::Value::Null)
                .with_metrics(metrics);

        assert_eq!(result.metrics.samples_collected, 5);
        assert_eq!(result.metrics.red_flagged, 1);
    }

    // ==========================================
    // ExecutorConfig Tests
    // ==========================================

    #[test]
    fn test_executor_config_default() {
        let config = ExecutorConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.k_margin, 3);
    }

    #[test]
    fn test_executor_config_builder() {
        let config = ExecutorConfig::default()
            .with_k_margin(5)
            .with_max_retries(2);

        assert_eq!(config.k_margin, 5);
        assert_eq!(config.max_retries, 2);
    }

    #[test]
    fn test_executor_config_retry_delay() {
        let config = ExecutorConfig {
            retry_base_delay: Duration::from_millis(100),
            retry_max_delay: Duration::from_secs(1),
            ..Default::default()
        };

        assert_eq!(config.retry_delay(0), Duration::from_millis(100));
        assert_eq!(config.retry_delay(1), Duration::from_millis(200));
        assert_eq!(config.retry_delay(2), Duration::from_millis(400));
        assert_eq!(config.retry_delay(10), Duration::from_secs(1)); // Capped at max
    }

    // ==========================================
    // ExecutionError Tests
    // ==========================================

    #[test]
    fn test_execution_error_display() {
        let errors = vec![
            ExecutionError::NotLeafNode {
                task_id: "t1".to_string(),
            },
            ExecutionError::InvalidMValue {
                task_id: "t2".to_string(),
                m_value: 5,
            },
            ExecutionError::VotingFailed {
                task_id: "t3".to_string(),
                attempts: 3,
                last_error: "timeout".to_string(),
            },
            ExecutionError::Timeout {
                task_id: "t4".to_string(),
                elapsed_ms: 5000,
            },
            ExecutionError::StateValidationFailed {
                task_id: "t5".to_string(),
                message: "invalid".to_string(),
            },
        ];

        for error in errors {
            let display = format!("{}", error);
            assert!(!display.is_empty());
        }
    }

    #[test]
    fn test_execution_error_into_decomposition_error() {
        let exec_error = ExecutionError::NotLeafNode {
            task_id: "t1".to_string(),
        };
        let decomp_error: DecompositionError = exec_error.into();

        assert!(matches!(
            decomp_error,
            DecompositionError::AgentError { .. }
        ));
    }

    // ==========================================
    // LeafNodeExecutor Tests
    // ==========================================

    #[test]
    fn test_executor_new() {
        let executor = LeafNodeExecutor::new();
        assert_eq!(executor.config().k_margin, 3);
        assert!(executor.results().is_empty());
    }

    #[test]
    fn test_executor_with_config() {
        let config = ExecutorConfig::default().with_k_margin(5);
        let executor = LeafNodeExecutor::with_config(config);
        assert_eq!(executor.config().k_margin, 5);
    }

    #[test]
    fn test_validate_leaf_rejects_non_leaf() {
        let executor = LeafNodeExecutor::new();
        let subtask = DecompositionSubtask::new("task-1", "Not a leaf");

        let result = executor.validate_leaf(&subtask);
        assert!(matches!(result, Err(ExecutionError::NotLeafNode { .. })));
    }

    #[test]
    fn test_validate_leaf_rejects_invalid_m_value() {
        let executor = LeafNodeExecutor::new();
        let mut subtask = DecompositionSubtask::leaf("task-1", "Leaf");
        subtask.m_value = 5; // Invalid for leaf

        let result = executor.validate_leaf(&subtask);
        assert!(matches!(result, Err(ExecutionError::InvalidMValue { .. })));
    }

    #[test]
    fn test_validate_leaf_accepts_valid_leaf() {
        let executor = LeafNodeExecutor::new();
        let subtask = DecompositionSubtask::leaf("task-1", "Valid leaf");

        let result = executor.validate_leaf(&subtask);
        assert!(result.is_ok());
    }

    #[test]
    fn test_execute_leaf_success() {
        let mut executor = LeafNodeExecutor::new();
        let client = MockLlmClient::constant("correct answer");
        let subtask = DecompositionSubtask::leaf("task-1", "What is 2+2?");
        let parent_state = serde_json::json!({"step": 0});

        let result = executor.execute_leaf(&subtask, &parent_state, &client);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.success);
        assert_eq!(result.output, "correct answer");
        assert_eq!(result.metrics.k_used, 3);
    }

    #[test]
    fn test_execute_leaf_stores_result() {
        let mut executor = LeafNodeExecutor::new();
        let client = MockLlmClient::constant("answer");
        let subtask = DecompositionSubtask::leaf("task-1", "Test");

        let _ = executor.execute_leaf(&subtask, &serde_json::Value::Null, &client);

        assert!(executor.get_result("task-1").is_some());
        assert_eq!(executor.results().len(), 1);
    }

    #[test]
    fn test_execute_leaf_rejects_non_leaf() {
        let mut executor = LeafNodeExecutor::new();
        let client = MockLlmClient::constant("answer");
        let subtask = DecompositionSubtask::new("task-1", "Not a leaf");

        let result = executor.execute_leaf(&subtask, &serde_json::Value::Null, &client);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ExecutionError::NotLeafNode { .. }
        ));
    }

    #[test]
    fn test_execute_sequential_state_passing() {
        let mut executor = LeafNodeExecutor::new();
        let client = MockLlmClient::constant(r#"{"result": "done", "state": {"step": 1}}"#);

        let subtasks = vec![
            DecompositionSubtask::leaf("task-1", "Step 1").with_order(0),
            DecompositionSubtask::leaf("task-2", "Step 2").with_order(1),
        ];

        let initial_state = serde_json::json!({"step": 0});
        let results = executor.execute_sequential(&subtasks, &initial_state, &client);

        assert!(results.is_ok());
        let results = results.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_build_prompt_includes_context() {
        let executor = LeafNodeExecutor::new();
        let subtask = DecompositionSubtask::leaf("task-1", "Do something")
            .with_context(serde_json::json!({"hint": "use this"}));
        let parent_state = serde_json::json!({"current": "state"});

        let prompt = executor.build_prompt(&subtask, &parent_state);

        assert!(prompt.contains("Do something"));
        assert!(prompt.contains("hint"));
        assert!(prompt.contains("current"));
    }

    #[test]
    fn test_extract_state_from_json() {
        let executor = LeafNodeExecutor::new();
        let output = r#"{"next_state": {"step": 2}, "result": "ok"}"#;
        let fallback = serde_json::json!({"step": 1});

        let state = executor.extract_state(output, &fallback);

        assert_eq!(state, serde_json::json!({"step": 2}));
    }

    #[test]
    fn test_extract_state_fallback() {
        let executor = LeafNodeExecutor::new();
        let output = "plain text output";
        let fallback = serde_json::json!({"step": 1});

        let state = executor.extract_state(output, &fallback);

        assert!(state.get("result").is_some());
        assert_eq!(state["result"], "plain text output");
    }

    #[test]
    fn test_clear_results() {
        let mut executor = LeafNodeExecutor::new();
        let client = MockLlmClient::constant("answer");
        let subtask = DecompositionSubtask::leaf("task-1", "Test");

        let _ = executor.execute_leaf(&subtask, &serde_json::Value::Null, &client);
        assert_eq!(executor.results().len(), 1);

        executor.clear_results();
        assert!(executor.results().is_empty());
    }

    // ==========================================
    // m=1 Enforcement Tests
    // ==========================================

    #[test]
    fn test_m1_strictly_enforced() {
        let executor = LeafNodeExecutor::new();

        // m=0 is invalid
        let mut subtask = DecompositionSubtask::leaf("t", "d");
        subtask.m_value = 0;
        assert!(executor.validate_leaf(&subtask).is_err());

        // m=2 is invalid
        subtask.m_value = 2;
        assert!(executor.validate_leaf(&subtask).is_err());

        // Only m=1 is valid for leaf nodes
        subtask.m_value = 1;
        assert!(executor.validate_leaf(&subtask).is_ok());
    }

    #[test]
    fn test_m1_and_is_leaf_both_required() {
        let executor = LeafNodeExecutor::new();

        // m=1 but not leaf -> error
        let subtask1 = DecompositionSubtask::new("t", "d").with_m_value(1);
        assert!(!subtask1.is_leaf);
        assert!(executor.validate_leaf(&subtask1).is_err());

        // leaf but m!=1 -> error
        let mut subtask2 = DecompositionSubtask::leaf("t", "d");
        subtask2.m_value = 2;
        assert!(executor.validate_leaf(&subtask2).is_err());

        // leaf and m=1 -> ok
        let subtask3 = DecompositionSubtask::leaf("t", "d");
        assert!(subtask3.is_leaf);
        assert_eq!(subtask3.m_value, 1);
        assert!(executor.validate_leaf(&subtask3).is_ok());
    }
}
