//! Recursive Loop Orchestration for MAKER Framework
//!
//! Implements the full recursive decomposition pipeline from Section 7 of the
//! System Design Specification. The `RecursiveOrchestrator` coordinates:
//!
//! 1. Task decomposition via `DecompositionAgent`
//! 2. Proposal voting via `DecompositionDiscriminator`
//! 3. Leaf execution via `LeafNodeExecutor`
//! 4. Result aggregation via `SolutionDiscriminator`
//!
//! # Architecture
//!
//! ```text
//! High-level Task
//!       ↓
//! RecursiveOrchestrator.execute()
//!       ↓
//! DecompositionAgent.propose_decomposition()
//!       ↓
//! DecompositionDiscriminator votes on proposals
//!       ↓
//! For each subtask:
//!   - If leaf (m=1): LeafNodeExecutor.execute_leaf()
//!   - If composite: Recurse with RecursiveOrchestrator
//!       ↓
//! SolutionDiscriminator.aggregate()
//!       ↓
//! Final Result
//! ```
//!
//! # Safety Features
//!
//! - **Depth Limit**: Prevents infinite recursion (default: 10)
//! - **Cycle Detection**: Detects task ID cycles in the decomposition tree
//! - **Timeout**: Cancels long-running executions (default: 60s)
//! - **Cancellation**: Graceful shutdown via cancellation token

use super::{
    AggregatedResult, AggregationError, CompositionFunction, DecompositionAgent,
    DecompositionConfig, DecompositionError, DecompositionProposal, DecompositionSubtask,
    ExecutorConfig, LeafNodeExecutor, SolutionDiscriminator, SubtaskResult,
};
use crate::core::executor::LlmClient;
use crate::events::MakerEvent;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Result of recursive orchestration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationResult {
    /// The task that was executed
    pub task_id: String,

    /// Final composed output
    pub output: String,

    /// Final composed state
    pub state: serde_json::Value,

    /// Whether execution succeeded
    pub success: bool,

    /// Error message if execution failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,

    /// Execution metrics
    pub metrics: OrchestrationMetrics,

    /// The winning decomposition proposal (if decomposed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub winning_proposal: Option<DecompositionProposal>,

    /// Aggregated result from subtasks
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aggregated: Option<AggregatedResult>,
}

impl OrchestrationResult {
    /// Create a successful result
    pub fn success(task_id: String, output: String, state: serde_json::Value) -> Self {
        Self {
            task_id,
            output,
            state,
            success: true,
            error: None,
            metrics: OrchestrationMetrics::default(),
            winning_proposal: None,
            aggregated: None,
        }
    }

    /// Create a failed result
    pub fn failure(task_id: String, error: String) -> Self {
        Self {
            task_id,
            output: String::new(),
            state: serde_json::Value::Null,
            success: false,
            error: Some(error),
            metrics: OrchestrationMetrics::default(),
            winning_proposal: None,
            aggregated: None,
        }
    }

    /// Set the winning proposal
    pub fn with_proposal(mut self, proposal: DecompositionProposal) -> Self {
        self.winning_proposal = Some(proposal);
        self
    }

    /// Set the aggregated result
    pub fn with_aggregated(mut self, aggregated: AggregatedResult) -> Self {
        self.aggregated = Some(aggregated);
        self
    }

    /// Set metrics
    pub fn with_metrics(mut self, metrics: OrchestrationMetrics) -> Self {
        self.metrics = metrics;
        self
    }
}

/// Metrics from orchestration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OrchestrationMetrics {
    /// Maximum depth reached during recursion
    pub max_depth_reached: usize,

    /// Total number of decompositions performed
    pub decompositions: usize,

    /// Total number of leaf executions
    pub leaf_executions: usize,

    /// Total number of aggregations
    pub aggregations: usize,

    /// Total execution time in milliseconds
    pub total_elapsed_ms: u64,

    /// Time spent in decomposition (ms)
    pub decomposition_ms: u64,

    /// Time spent in leaf execution (ms)
    pub execution_ms: u64,

    /// Time spent in aggregation (ms)
    pub aggregation_ms: u64,

    /// Whether execution was cancelled
    pub cancelled: bool,

    /// Whether execution timed out
    pub timed_out: bool,
}

/// Errors during orchestration
#[derive(Debug, Clone, PartialEq)]
pub enum OrchestrationError {
    /// Decomposition failed
    DecompositionFailed { message: String },

    /// Execution failed
    ExecutionFailed { task_id: String, message: String },

    /// Aggregation failed
    AggregationFailed { message: String },

    /// Depth limit exceeded
    DepthLimitExceeded { depth: usize, limit: usize },

    /// Cycle detected in decomposition
    CycleDetected { task_id: String },

    /// Execution timed out
    Timeout { elapsed_ms: u64, limit_ms: u64 },

    /// Execution was cancelled
    Cancelled,

    /// No decomposition agent provided
    NoAgent,
}

impl std::fmt::Display for OrchestrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DecompositionFailed { message } => {
                write!(f, "Decomposition failed: {}", message)
            }
            Self::ExecutionFailed { task_id, message } => {
                write!(f, "Execution of '{}' failed: {}", task_id, message)
            }
            Self::AggregationFailed { message } => {
                write!(f, "Aggregation failed: {}", message)
            }
            Self::DepthLimitExceeded { depth, limit } => {
                write!(f, "Depth {} exceeds limit {}", depth, limit)
            }
            Self::CycleDetected { task_id } => {
                write!(f, "Cycle detected at task '{}'", task_id)
            }
            Self::Timeout {
                elapsed_ms,
                limit_ms,
            } => {
                write!(
                    f,
                    "Execution timed out after {}ms (limit: {}ms)",
                    elapsed_ms, limit_ms
                )
            }
            Self::Cancelled => write!(f, "Execution was cancelled"),
            Self::NoAgent => write!(f, "No decomposition agent provided"),
        }
    }
}

impl std::error::Error for OrchestrationError {}

impl From<DecompositionError> for OrchestrationError {
    fn from(e: DecompositionError) -> Self {
        Self::DecompositionFailed {
            message: e.to_string(),
        }
    }
}

impl From<AggregationError> for OrchestrationError {
    fn from(e: AggregationError) -> Self {
        Self::AggregationFailed {
            message: e.to_string(),
        }
    }
}

/// Configuration for the recursive orchestrator
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// Decomposition configuration
    pub decomposition: DecompositionConfig,

    /// Executor configuration for leaf nodes
    pub executor: ExecutorConfig,

    /// Timeout in milliseconds (0 = no timeout)
    pub timeout_ms: u64,

    /// Whether to emit events
    pub emit_events: bool,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            decomposition: DecompositionConfig::default(),
            executor: ExecutorConfig::default(),
            timeout_ms: 60_000, // 60 seconds
            emit_events: true,
        }
    }
}

impl OrchestratorConfig {
    /// Create config with custom timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout_ms = timeout.as_millis() as u64;
        self
    }

    /// Create config with custom depth limit
    pub fn with_depth_limit(mut self, limit: usize) -> Self {
        self.decomposition.depth_limit = limit;
        self
    }

    /// Create config with custom k-margin
    pub fn with_k_margin(mut self, k: usize) -> Self {
        self.decomposition.k_margin = k;
        self.executor.k_margin = k;
        self
    }

    /// Disable events
    pub fn without_events(mut self) -> Self {
        self.emit_events = false;
        self
    }
}

/// Cancellation token for graceful shutdown
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    /// Create a new cancellation token
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Cancel the execution
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Check if cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }
}

/// Recursive orchestrator for the full decomposition pipeline
///
/// Coordinates decomposition, execution, and aggregation to solve
/// complex tasks through recursive decomposition into atomic subtasks.
pub struct RecursiveOrchestrator<A: DecompositionAgent> {
    /// The decomposition agent
    agent: A,

    /// Configuration
    config: OrchestratorConfig,

    /// Cancellation token
    cancellation: CancellationToken,

    /// Manual decomposition overrides (task_id -> proposal)
    manual_overrides: std::collections::HashMap<String, DecompositionProposal>,

    /// Event emitter
    event_emitter: Option<Arc<dyn Fn(MakerEvent) + Send + Sync>>,

    /// Metrics accumulator
    metrics: OrchestrationMetrics,

    /// Start time for timeout tracking
    start_time: Option<Instant>,
}

impl<A: DecompositionAgent> RecursiveOrchestrator<A> {
    /// Create a new orchestrator with the given agent
    pub fn new(agent: A) -> Self {
        Self {
            agent,
            config: OrchestratorConfig::default(),
            cancellation: CancellationToken::new(),
            manual_overrides: std::collections::HashMap::new(),
            event_emitter: None,
            metrics: OrchestrationMetrics::default(),
            start_time: None,
        }
    }

    /// Create with custom configuration
    pub fn with_config(agent: A, config: OrchestratorConfig) -> Self {
        Self {
            agent,
            config,
            cancellation: CancellationToken::new(),
            manual_overrides: std::collections::HashMap::new(),
            event_emitter: None,
            metrics: OrchestrationMetrics::default(),
            start_time: None,
        }
    }

    /// Set the cancellation token
    pub fn with_cancellation(mut self, token: CancellationToken) -> Self {
        self.cancellation = token;
        self
    }

    /// Set an event emitter
    pub fn with_event_emitter(mut self, emitter: Arc<dyn Fn(MakerEvent) + Send + Sync>) -> Self {
        self.event_emitter = Some(emitter);
        self
    }

    /// Add a manual decomposition override
    pub fn with_manual_override(mut self, task_id: &str, proposal: DecompositionProposal) -> Self {
        self.manual_overrides.insert(task_id.to_string(), proposal);
        self
    }

    /// Get the configuration
    pub fn config(&self) -> &OrchestratorConfig {
        &self.config
    }

    /// Get the cancellation token
    pub fn cancellation_token(&self) -> &CancellationToken {
        &self.cancellation
    }

    /// Cancel the execution
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    /// Check if should stop (cancelled or timed out)
    fn should_stop(&self) -> Result<(), OrchestrationError> {
        if self.cancellation.is_cancelled() {
            return Err(OrchestrationError::Cancelled);
        }

        if self.config.timeout_ms > 0 {
            if let Some(start) = self.start_time {
                let elapsed = start.elapsed().as_millis() as u64;
                if elapsed > self.config.timeout_ms {
                    return Err(OrchestrationError::Timeout {
                        elapsed_ms: elapsed,
                        limit_ms: self.config.timeout_ms,
                    });
                }
            }
        }

        Ok(())
    }

    /// Execute a task through the full recursive pipeline
    ///
    /// # Arguments
    ///
    /// * `task_id` - Identifier for the task
    /// * `description` - Description of what needs to be done
    /// * `context` - Initial context/state
    /// * `client` - LLM client for generation
    ///
    /// # Returns
    ///
    /// * `Ok(OrchestrationResult)` - The final result
    /// * `Err(OrchestrationError)` - If execution fails
    pub fn execute(
        &mut self,
        task_id: &str,
        description: &str,
        context: &serde_json::Value,
        client: &dyn LlmClient,
    ) -> Result<OrchestrationResult, OrchestrationError> {
        self.start_time = Some(Instant::now());
        self.metrics = OrchestrationMetrics::default();

        let mut visited = HashSet::new();
        let result = self.execute_recursive(task_id, description, context, client, 0, &mut visited);

        // Update final metrics
        if let Some(start) = self.start_time {
            self.metrics.total_elapsed_ms = start.elapsed().as_millis() as u64;
        }

        match result {
            Ok(mut res) => {
                res.metrics = self.metrics.clone();
                Ok(res)
            }
            Err(e) => {
                match &e {
                    OrchestrationError::Cancelled => self.metrics.cancelled = true,
                    OrchestrationError::Timeout { .. } => self.metrics.timed_out = true,
                    // Other errors (DepthExceeded, CycleDetected, Decomposition, Aggregation)
                    // don't have dedicated metric flags - they're captured in the error itself
                    _ => {}
                }
                Err(e)
            }
        }
    }

    /// Recursive execution implementation
    fn execute_recursive(
        &mut self,
        task_id: &str,
        description: &str,
        context: &serde_json::Value,
        client: &dyn LlmClient,
        depth: usize,
        visited: &mut HashSet<String>,
    ) -> Result<OrchestrationResult, OrchestrationError> {
        // Check stop conditions
        self.should_stop()?;

        // Check depth limit
        if depth > self.config.decomposition.depth_limit {
            return Err(OrchestrationError::DepthLimitExceeded {
                depth,
                limit: self.config.decomposition.depth_limit,
            });
        }

        // Cycle detection
        if visited.contains(task_id) {
            return Err(OrchestrationError::CycleDetected {
                task_id: task_id.to_string(),
            });
        }
        visited.insert(task_id.to_string());

        // Update max depth metric
        if depth > self.metrics.max_depth_reached {
            self.metrics.max_depth_reached = depth;
        }

        // Check if task is atomic (should be executed directly)
        if self.agent.is_atomic(task_id, description) {
            return self.execute_leaf_task(task_id, description, context, client);
        }

        // Get decomposition (manual override or agent-generated)
        let decomp_start = Instant::now();
        let proposal = self.get_decomposition(task_id, description, context, depth)?;
        self.metrics.decomposition_ms += decomp_start.elapsed().as_millis() as u64;
        self.metrics.decompositions += 1;

        // Validate the proposal
        proposal.validate()?;

        // Execute subtasks
        let exec_start = Instant::now();
        let subtask_results = self.execute_subtasks(&proposal, context, client, depth, visited)?;
        self.metrics.execution_ms += exec_start.elapsed().as_millis() as u64;

        // Aggregate results
        let agg_start = Instant::now();
        let discriminator = SolutionDiscriminator::new();
        let aggregated = discriminator.aggregate(&proposal, subtask_results, depth)?;
        self.metrics.aggregation_ms += agg_start.elapsed().as_millis() as u64;
        self.metrics.aggregations += 1;

        Ok(OrchestrationResult::success(
            task_id.to_string(),
            aggregated.output.clone(),
            aggregated.state.clone(),
        )
        .with_proposal(proposal)
        .with_aggregated(aggregated))
    }

    /// Get decomposition for a task (manual override or agent-generated)
    fn get_decomposition(
        &self,
        task_id: &str,
        description: &str,
        context: &serde_json::Value,
        depth: usize,
    ) -> Result<DecompositionProposal, OrchestrationError> {
        // Check for manual override
        if let Some(proposal) = self.manual_overrides.get(task_id) {
            return Ok(proposal.clone());
        }

        // Use agent to generate decomposition
        self.agent
            .propose_decomposition(task_id, description, context, depth)
            .map_err(|e| OrchestrationError::DecompositionFailed {
                message: e.to_string(),
            })
    }

    /// Execute subtasks from a proposal
    fn execute_subtasks(
        &mut self,
        proposal: &DecompositionProposal,
        initial_context: &serde_json::Value,
        client: &dyn LlmClient,
        depth: usize,
        visited: &mut HashSet<String>,
    ) -> Result<Vec<SubtaskResult>, OrchestrationError> {
        let mut results = Vec::new();
        let mut current_context = initial_context.clone();

        // Sort subtasks by order
        let mut subtasks: Vec<_> = proposal.subtasks.iter().collect();
        subtasks.sort_by_key(|s| s.order);

        for subtask in subtasks {
            self.should_stop()?;

            let result = if subtask.is_leaf {
                // Execute leaf directly
                self.execute_leaf_subtask(subtask, &current_context, client)?
            } else {
                // Recurse for non-leaf nodes
                let sub_result = self.execute_recursive(
                    &subtask.task_id,
                    &subtask.description,
                    &current_context,
                    client,
                    depth + 1,
                    visited,
                )?;

                SubtaskResult::success(subtask.task_id.clone(), sub_result.output, sub_result.state)
            };

            // Update context for sequential composition
            if matches!(proposal.composition_fn, CompositionFunction::Sequential) {
                current_context = result.state.clone();
            }

            results.push(result);
        }

        Ok(results)
    }

    /// Execute a leaf subtask
    fn execute_leaf_subtask(
        &mut self,
        subtask: &DecompositionSubtask,
        context: &serde_json::Value,
        client: &dyn LlmClient,
    ) -> Result<SubtaskResult, OrchestrationError> {
        let mut executor = LeafNodeExecutor::with_config(self.config.executor.clone());

        self.metrics.leaf_executions += 1;

        executor
            .execute_leaf(subtask, context, client)
            .map_err(|e| OrchestrationError::ExecutionFailed {
                task_id: subtask.task_id.clone(),
                message: e.to_string(),
            })
    }

    /// Execute a leaf task (atomic task, no decomposition needed)
    fn execute_leaf_task(
        &mut self,
        task_id: &str,
        description: &str,
        context: &serde_json::Value,
        client: &dyn LlmClient,
    ) -> Result<OrchestrationResult, OrchestrationError> {
        let subtask =
            DecompositionSubtask::leaf(task_id, description).with_context(context.clone());

        let result = self.execute_leaf_subtask(&subtask, context, client)?;

        Ok(OrchestrationResult::success(
            task_id.to_string(),
            result.output,
            result.state,
        ))
    }
}

impl<A: DecompositionAgent> std::fmt::Debug for RecursiveOrchestrator<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecursiveOrchestrator")
            .field("config", &self.config)
            .field("manual_overrides", &self.manual_overrides.keys())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::decomposition::IdentityDecomposer;
    use crate::core::executor::MockLlmClient;

    // ==========================================
    // OrchestrationResult Tests
    // ==========================================

    #[test]
    fn test_orchestration_result_success() {
        let result = OrchestrationResult::success(
            "task-1".to_string(),
            "output".to_string(),
            serde_json::json!({"key": "value"}),
        );

        assert!(result.success);
        assert!(result.error.is_none());
        assert_eq!(result.task_id, "task-1");
    }

    #[test]
    fn test_orchestration_result_failure() {
        let result = OrchestrationResult::failure("task-1".to_string(), "error".to_string());

        assert!(!result.success);
        assert_eq!(result.error, Some("error".to_string()));
    }

    #[test]
    fn test_orchestration_result_with_proposal() {
        let proposal = DecompositionProposal::new(
            "p-1",
            "t-1",
            vec![DecompositionSubtask::leaf("s-1", "Sub")],
            CompositionFunction::Sequential,
        );

        let result = OrchestrationResult::success(
            "t-1".to_string(),
            "out".to_string(),
            serde_json::json!({}),
        )
        .with_proposal(proposal);

        assert!(result.winning_proposal.is_some());
    }

    // ==========================================
    // OrchestratorConfig Tests
    // ==========================================

    #[test]
    fn test_orchestrator_config_default() {
        let config = OrchestratorConfig::default();
        assert_eq!(config.timeout_ms, 60_000);
        assert_eq!(config.decomposition.depth_limit, 10);
        assert!(config.emit_events);
    }

    #[test]
    fn test_orchestrator_config_builder() {
        let config = OrchestratorConfig::default()
            .with_timeout(Duration::from_secs(30))
            .with_depth_limit(5)
            .with_k_margin(4)
            .without_events();

        assert_eq!(config.timeout_ms, 30_000);
        assert_eq!(config.decomposition.depth_limit, 5);
        assert_eq!(config.decomposition.k_margin, 4);
        assert!(!config.emit_events);
    }

    // ==========================================
    // CancellationToken Tests
    // ==========================================

    #[test]
    fn test_cancellation_token() {
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());

        token.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_clone() {
        let token = CancellationToken::new();
        let token2 = token.clone();

        token.cancel();

        assert!(token.is_cancelled());
        assert!(token2.is_cancelled()); // Shared state
    }

    // ==========================================
    // OrchestrationError Tests
    // ==========================================

    #[test]
    fn test_orchestration_error_display() {
        let errors = vec![
            OrchestrationError::DecompositionFailed {
                message: "test".to_string(),
            },
            OrchestrationError::ExecutionFailed {
                task_id: "t1".to_string(),
                message: "error".to_string(),
            },
            OrchestrationError::AggregationFailed {
                message: "agg error".to_string(),
            },
            OrchestrationError::DepthLimitExceeded {
                depth: 11,
                limit: 10,
            },
            OrchestrationError::CycleDetected {
                task_id: "cycle".to_string(),
            },
            OrchestrationError::Timeout {
                elapsed_ms: 65000,
                limit_ms: 60000,
            },
            OrchestrationError::Cancelled,
            OrchestrationError::NoAgent,
        ];

        for error in errors {
            let display = format!("{}", error);
            assert!(!display.is_empty());
        }
    }

    // ==========================================
    // RecursiveOrchestrator Tests
    // ==========================================

    #[test]
    fn test_orchestrator_new() {
        let agent = IdentityDecomposer;
        let orchestrator = RecursiveOrchestrator::new(agent);

        assert_eq!(orchestrator.config().timeout_ms, 60_000);
        assert!(!orchestrator.cancellation_token().is_cancelled());
    }

    #[test]
    fn test_orchestrator_with_config() {
        let config = OrchestratorConfig::default().with_depth_limit(5);
        let agent = IdentityDecomposer;
        let orchestrator = RecursiveOrchestrator::with_config(agent, config);

        assert_eq!(orchestrator.config().decomposition.depth_limit, 5);
    }

    #[test]
    fn test_orchestrator_with_manual_override() {
        let agent = IdentityDecomposer;
        let proposal = DecompositionProposal::new(
            "override-proposal",
            "task-1",
            vec![DecompositionSubtask::leaf("s-1", "Override subtask")],
            CompositionFunction::Sequential,
        );

        let orchestrator =
            RecursiveOrchestrator::new(agent).with_manual_override("task-1", proposal);

        assert!(orchestrator.manual_overrides.contains_key("task-1"));
    }

    #[test]
    fn test_orchestrator_cancel() {
        let agent = IdentityDecomposer;
        let orchestrator = RecursiveOrchestrator::new(agent);

        assert!(!orchestrator.cancellation_token().is_cancelled());
        orchestrator.cancel();
        assert!(orchestrator.cancellation_token().is_cancelled());
    }

    #[test]
    fn test_orchestrator_execute_atomic_task() {
        let agent = IdentityDecomposer; // Always treats tasks as atomic
        let mut orchestrator = RecursiveOrchestrator::new(agent);
        let client = MockLlmClient::constant("result");

        let result = orchestrator.execute(
            "task-1",
            "Do something simple",
            &serde_json::json!({}),
            &client,
        );

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.success);
        assert_eq!(result.output, "result");
    }

    #[test]
    fn test_orchestrator_execute_with_manual_override() {
        let _agent = IdentityDecomposer;
        let proposal = DecompositionProposal::new(
            "override-proposal",
            "task-1",
            vec![
                DecompositionSubtask::leaf("step-1", "First step").with_order(0),
                DecompositionSubtask::leaf("step-2", "Second step").with_order(1),
            ],
            CompositionFunction::Sequential,
        );

        // Create a non-identity agent that doesn't mark tasks as atomic
        struct NonAtomicDecomposer;
        impl DecompositionAgent for NonAtomicDecomposer {
            fn propose_decomposition(
                &self,
                task_id: &str,
                description: &str,
                _context: &serde_json::Value,
                _depth: usize,
            ) -> Result<DecompositionProposal, DecompositionError> {
                Ok(DecompositionProposal::new(
                    format!("auto-{}", task_id),
                    task_id,
                    vec![DecompositionSubtask::leaf(task_id, description)],
                    CompositionFunction::Sequential,
                ))
            }

            fn is_atomic(&self, _task_id: &str, _description: &str) -> bool {
                false // Never atomic
            }

            fn name(&self) -> &str {
                "non-atomic"
            }
        }

        let mut orchestrator = RecursiveOrchestrator::new(NonAtomicDecomposer)
            .with_manual_override("task-1", proposal);
        let client = MockLlmClient::constant("done");

        let result =
            orchestrator.execute("task-1", "Complex task", &serde_json::json!({}), &client);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.success);
        assert!(result.winning_proposal.is_some());
        assert_eq!(
            result.winning_proposal.unwrap().proposal_id,
            "override-proposal"
        );
    }

    #[test]
    fn test_orchestrator_depth_limit() {
        // Create a decomposer that always decomposes (never atomic)
        struct InfiniteDecomposer;
        impl DecompositionAgent for InfiniteDecomposer {
            fn propose_decomposition(
                &self,
                task_id: &str,
                description: &str,
                context: &serde_json::Value,
                _depth: usize,
            ) -> Result<DecompositionProposal, DecompositionError> {
                Ok(DecompositionProposal::new(
                    format!("infinite-{}", task_id),
                    task_id,
                    vec![
                        DecompositionSubtask::new(format!("{}-child", task_id), description)
                            .with_context(context.clone()),
                    ],
                    CompositionFunction::Sequential,
                ))
            }

            fn is_atomic(&self, _task_id: &str, _description: &str) -> bool {
                false
            }

            fn name(&self) -> &str {
                "infinite"
            }
        }

        let config = OrchestratorConfig::default().with_depth_limit(3);
        let mut orchestrator = RecursiveOrchestrator::with_config(InfiniteDecomposer, config);
        let client = MockLlmClient::constant("result");

        let result = orchestrator.execute("root", "Start", &serde_json::json!({}), &client);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OrchestrationError::DepthLimitExceeded { .. }
        ));
    }

    #[test]
    fn test_orchestrator_cycle_detection() {
        // Create a decomposer that creates a cycle
        struct CyclicDecomposer;
        impl DecompositionAgent for CyclicDecomposer {
            fn propose_decomposition(
                &self,
                task_id: &str,
                _description: &str,
                _context: &serde_json::Value,
                _depth: usize,
            ) -> Result<DecompositionProposal, DecompositionError> {
                // Always reference the root task, creating a cycle
                Ok(DecompositionProposal::new(
                    format!("cyclic-{}", task_id),
                    task_id,
                    vec![DecompositionSubtask::new("root", "Back to root")], // Cycle!
                    CompositionFunction::Sequential,
                ))
            }

            fn is_atomic(&self, _task_id: &str, _description: &str) -> bool {
                false
            }

            fn name(&self) -> &str {
                "cyclic"
            }
        }

        let mut orchestrator = RecursiveOrchestrator::new(CyclicDecomposer);
        let client = MockLlmClient::constant("result");

        let result = orchestrator.execute("root", "Start", &serde_json::json!({}), &client);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            OrchestrationError::CycleDetected { .. }
        ));
    }

    #[test]
    fn test_orchestrator_cancellation() {
        let agent = IdentityDecomposer;
        let token = CancellationToken::new();
        token.cancel(); // Pre-cancel

        let mut orchestrator = RecursiveOrchestrator::new(agent).with_cancellation(token);
        let client = MockLlmClient::constant("result");

        let result = orchestrator.execute("task", "desc", &serde_json::json!({}), &client);

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), OrchestrationError::Cancelled));
    }

    #[test]
    fn test_orchestrator_metrics() {
        let agent = IdentityDecomposer;
        let mut orchestrator = RecursiveOrchestrator::new(agent);
        let client = MockLlmClient::constant("result");

        let result = orchestrator
            .execute("task", "desc", &serde_json::json!({}), &client)
            .unwrap();

        // Verify elapsed time is tracked (u64 is always non-negative)
        let _ = result.metrics.total_elapsed_ms;
        assert_eq!(result.metrics.leaf_executions, 1); // Identity executes as leaf
    }

    // ==========================================
    // End-to-End Integration Test
    // ==========================================

    #[test]
    fn test_end_to_end_recursive_pipeline() {
        // Create a 2-level decomposer
        struct TwoLevelDecomposer;
        impl DecompositionAgent for TwoLevelDecomposer {
            fn propose_decomposition(
                &self,
                task_id: &str,
                _description: &str,
                _context: &serde_json::Value,
                depth: usize,
            ) -> Result<DecompositionProposal, DecompositionError> {
                if depth == 0 {
                    // Root: decompose into 2 subtasks
                    Ok(DecompositionProposal::new(
                        format!("proposal-{}", task_id),
                        task_id,
                        vec![
                            DecompositionSubtask::leaf("step-1", "First step").with_order(0),
                            DecompositionSubtask::leaf("step-2", "Second step").with_order(1),
                        ],
                        CompositionFunction::Sequential,
                    ))
                } else {
                    // Deeper: treat as atomic
                    Ok(DecompositionProposal::new(
                        format!("atomic-{}", task_id),
                        task_id,
                        vec![DecompositionSubtask::leaf(task_id, "Atomic")],
                        CompositionFunction::Sequential,
                    ))
                }
            }

            fn is_atomic(&self, _task_id: &str, _description: &str) -> bool {
                false // Let propose_decomposition decide
            }

            fn name(&self) -> &str {
                "two-level"
            }
        }

        let mut orchestrator = RecursiveOrchestrator::new(TwoLevelDecomposer);
        let client = MockLlmClient::constant(r#"{"result": "done", "step": 1}"#);

        let result = orchestrator.execute(
            "main-task",
            "Complete multi-step task",
            &serde_json::json!({"initial": true}),
            &client,
        );

        assert!(result.is_ok());
        let result = result.unwrap();

        // Verify full pipeline executed
        assert!(result.success);
        assert!(result.winning_proposal.is_some());
        assert!(result.aggregated.is_some());

        // Verify metrics
        assert_eq!(result.metrics.decompositions, 1);
        assert_eq!(result.metrics.leaf_executions, 2);
        assert_eq!(result.metrics.aggregations, 1);
        assert_eq!(result.metrics.max_depth_reached, 0);
    }
}
