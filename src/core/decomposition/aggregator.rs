//! Solution Discriminator & Aggregation for MAKER Framework
//!
//! Implements the composition layer for aggregating subtask results according to
//! the decomposition proposal's composition function. The `SolutionDiscriminator`
//! votes on competing result aggregations to select the most reliable composition.
//!
//! # Architecture
//!
//! ```text
//! SubtaskResults from LeafNodeExecutor
//!       ↓
//! SolutionDiscriminator.aggregate()
//!       ↓
//! Vote on composition candidates (if multiple)
//!       ↓
//! compose_results() applies winning CompositionFunction
//!       ↓
//! Emit SolutionComposed event
//!       ↓
//! AggregatedResult { output, state, metrics }
//! ```
//!
//! # Key Principles
//!
//! 1. **Composition Respects Strategy**: Results are combined according to the
//!    winning decomposition's composition function
//! 2. **Recursive Composition**: Nested trees are composed depth-first
//! 3. **Audit Trail**: All composition steps emit events for observability
//! 4. **Schema Validation**: Composed results are validated for consistency

use super::{CompositionFunction, DecompositionProposal, MergeStrategy, SubtaskResult};
use crate::core::voting::VoteRace;
use crate::events::MakerEvent;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Result of aggregating subtask results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedResult {
    /// The proposal that was executed
    pub proposal_id: String,

    /// The source task that was decomposed
    pub source_task_id: String,

    /// The final composed output
    pub output: String,

    /// The final composed state
    pub state: serde_json::Value,

    /// Composition function that was used
    pub composition_fn: String,

    /// Aggregation metrics
    pub metrics: AggregationMetrics,

    /// Whether all subtasks succeeded
    pub all_succeeded: bool,

    /// Individual subtask results (for audit trail)
    pub subtask_results: Vec<SubtaskResult>,
}

impl AggregatedResult {
    /// Create a new aggregated result
    pub fn new(
        proposal_id: String,
        source_task_id: String,
        output: String,
        state: serde_json::Value,
        composition_fn: String,
        subtask_results: Vec<SubtaskResult>,
    ) -> Self {
        let all_succeeded = subtask_results.iter().all(|r| r.success);
        Self {
            proposal_id,
            source_task_id,
            output,
            state,
            composition_fn,
            metrics: AggregationMetrics::default(),
            all_succeeded,
            subtask_results,
        }
    }

    /// Set metrics
    pub fn with_metrics(mut self, metrics: AggregationMetrics) -> Self {
        self.metrics = metrics;
        self
    }

    /// Get the count of successful subtasks
    pub fn success_count(&self) -> usize {
        self.subtask_results.iter().filter(|r| r.success).count()
    }

    /// Get the count of failed subtasks
    pub fn failure_count(&self) -> usize {
        self.subtask_results.iter().filter(|r| !r.success).count()
    }
}

/// Metrics from the aggregation process
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregationMetrics {
    /// Total subtasks aggregated
    pub subtask_count: usize,

    /// Number of successful subtasks
    pub success_count: usize,

    /// Total execution time across all subtasks (ms)
    pub total_execution_ms: u64,

    /// Time spent in aggregation logic (ms)
    pub aggregation_ms: u64,

    /// Current recursion depth
    pub depth: usize,

    /// Number of composition candidates voted on
    pub candidates_voted: usize,

    /// k-margin used for voting (if applicable)
    pub k_margin: usize,
}

/// Errors during aggregation
#[derive(Debug, Clone, PartialEq)]
pub enum AggregationError {
    /// No results to aggregate
    EmptyResults { proposal_id: String },

    /// All subtasks failed
    AllFailed {
        proposal_id: String,
        failure_count: usize,
    },

    /// Schema validation failed
    SchemaValidationFailed { message: String },

    /// Recursive composition failed
    RecursiveCompositionFailed { depth: usize, message: String },

    /// Voting on candidates failed
    VotingFailed { message: String },

    /// Custom composition function not found
    UnknownCompositionFunction { name: String },
}

impl std::fmt::Display for AggregationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyResults { proposal_id } => {
                write!(f, "No results to aggregate for proposal '{}'", proposal_id)
            }
            Self::AllFailed {
                proposal_id,
                failure_count,
            } => {
                write!(
                    f,
                    "All {} subtasks failed for proposal '{}'",
                    failure_count, proposal_id
                )
            }
            Self::SchemaValidationFailed { message } => {
                write!(f, "Schema validation failed: {}", message)
            }
            Self::RecursiveCompositionFailed { depth, message } => {
                write!(
                    f,
                    "Recursive composition failed at depth {}: {}",
                    depth, message
                )
            }
            Self::VotingFailed { message } => {
                write!(f, "Voting on composition candidates failed: {}", message)
            }
            Self::UnknownCompositionFunction { name } => {
                write!(f, "Unknown composition function: '{}'", name)
            }
        }
    }
}

impl std::error::Error for AggregationError {}

/// Configuration for the solution discriminator
#[derive(Debug, Clone)]
pub struct AggregatorConfig {
    /// k-margin for voting on composition candidates
    pub k_margin: usize,

    /// Whether to validate output schema
    pub validate_schema: bool,

    /// Maximum recursion depth for nested composition
    pub max_depth: usize,

    /// Whether to emit events
    pub emit_events: bool,
}

impl Default for AggregatorConfig {
    fn default() -> Self {
        Self {
            k_margin: 3,
            validate_schema: true,
            max_depth: 10,
            emit_events: true,
        }
    }
}

impl AggregatorConfig {
    /// Create config with custom k-margin
    pub fn with_k_margin(mut self, k: usize) -> Self {
        self.k_margin = k;
        self
    }

    /// Create config with schema validation disabled
    pub fn without_schema_validation(mut self) -> Self {
        self.validate_schema = false;
        self
    }

    /// Create config with custom max depth
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }
}

/// Solution Discriminator for voting on and aggregating subtask results
///
/// Uses voting to select the best composition when multiple candidates exist,
/// and applies the composition function to produce the final result.
pub struct SolutionDiscriminator {
    /// Configuration
    config: AggregatorConfig,

    /// Optional event emitter
    event_emitter: Option<Arc<dyn Fn(MakerEvent) + Send + Sync>>,
}

impl SolutionDiscriminator {
    /// Create a new solution discriminator with default configuration
    pub fn new() -> Self {
        Self {
            config: AggregatorConfig::default(),
            event_emitter: None,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: AggregatorConfig) -> Self {
        Self {
            config,
            event_emitter: None,
        }
    }

    /// Set an event emitter
    pub fn with_event_emitter(mut self, emitter: Arc<dyn Fn(MakerEvent) + Send + Sync>) -> Self {
        self.event_emitter = Some(emitter);
        self
    }

    /// Get the configuration
    pub fn config(&self) -> &AggregatorConfig {
        &self.config
    }

    /// Emit a SolutionComposed event
    fn emit_solution_composed(
        &self,
        proposal_id: &str,
        composition_fn: &str,
        success_count: usize,
        total_subtasks: usize,
        total_elapsed_ms: u64,
        depth: usize,
    ) {
        if self.config.emit_events {
            if let Some(ref emitter) = self.event_emitter {
                emitter(MakerEvent::solution_composed(
                    proposal_id,
                    composition_fn,
                    success_count,
                    total_subtasks,
                    total_elapsed_ms,
                    depth,
                ));
            }
        }
    }

    /// Aggregate results from a decomposition proposal
    ///
    /// Applies the proposal's composition function to combine subtask results
    /// into a final aggregated result.
    ///
    /// # Arguments
    ///
    /// * `proposal` - The decomposition proposal that was executed
    /// * `results` - Results from executing the proposal's subtasks
    /// * `depth` - Current recursion depth (for nested compositions)
    ///
    /// # Returns
    ///
    /// * `Ok(AggregatedResult)` - The composed result
    /// * `Err(AggregationError)` - If aggregation fails
    pub fn aggregate(
        &self,
        proposal: &DecompositionProposal,
        results: Vec<SubtaskResult>,
        depth: usize,
    ) -> Result<AggregatedResult, AggregationError> {
        let start = Instant::now();

        // Validate inputs
        if results.is_empty() {
            return Err(AggregationError::EmptyResults {
                proposal_id: proposal.proposal_id.clone(),
            });
        }

        if depth > self.config.max_depth {
            return Err(AggregationError::RecursiveCompositionFailed {
                depth,
                message: format!("Exceeded maximum depth of {}", self.config.max_depth),
            });
        }

        // Check if all failed
        let success_count = results.iter().filter(|r| r.success).count();
        if success_count == 0 {
            return Err(AggregationError::AllFailed {
                proposal_id: proposal.proposal_id.clone(),
                failure_count: results.len(),
            });
        }

        // Apply composition function
        let (output, state) = self.compose_results(&proposal.composition_fn, &results)?;

        // Validate schema if enabled
        if self.config.validate_schema {
            self.validate_composed_schema(&state)?;
        }

        let elapsed = start.elapsed();
        let total_execution_ms: u64 = results.iter().map(|r| r.metrics.elapsed_ms).sum();

        let metrics = AggregationMetrics {
            subtask_count: results.len(),
            success_count,
            total_execution_ms,
            aggregation_ms: elapsed.as_millis() as u64,
            depth,
            candidates_voted: 0, // Single composition path
            k_margin: self.config.k_margin,
        };

        let composition_fn_name = composition_fn_to_string(&proposal.composition_fn);

        // Emit event
        self.emit_solution_composed(
            &proposal.proposal_id,
            &composition_fn_name,
            success_count,
            results.len(),
            total_execution_ms,
            depth,
        );

        Ok(AggregatedResult::new(
            proposal.proposal_id.clone(),
            proposal.source_task_id.clone(),
            output,
            state,
            composition_fn_name,
            results,
        )
        .with_metrics(metrics))
    }

    /// Compose results according to the composition function
    fn compose_results(
        &self,
        composition_fn: &CompositionFunction,
        results: &[SubtaskResult],
    ) -> Result<(String, serde_json::Value), AggregationError> {
        match composition_fn {
            CompositionFunction::Sequential => self.compose_sequential(results),
            CompositionFunction::Parallel { merge_strategy } => {
                self.compose_parallel(results, merge_strategy)
            }
            CompositionFunction::Conditional { condition } => {
                self.compose_conditional(results, condition)
            }
            CompositionFunction::Custom { name, params } => {
                self.compose_custom(results, name, params)
            }
        }
    }

    /// Compose results sequentially (last result wins)
    fn compose_sequential(
        &self,
        results: &[SubtaskResult],
    ) -> Result<(String, serde_json::Value), AggregationError> {
        // Find the last successful result
        let last_success = results.iter().rev().find(|r| r.success);

        match last_success {
            Some(result) => Ok((result.output.clone(), result.state.clone())),
            None => {
                // Fall back to last result even if failed
                let last = results.last().unwrap();
                Ok((last.output.clone(), last.state.clone()))
            }
        }
    }

    /// Compose results in parallel (merge according to strategy)
    fn compose_parallel(
        &self,
        results: &[SubtaskResult],
        strategy: &MergeStrategy,
    ) -> Result<(String, serde_json::Value), AggregationError> {
        let (output, state) = match strategy {
            MergeStrategy::Concatenate => {
                let outputs: Vec<&str> = results.iter().map(|r| r.output.as_str()).collect();
                let combined_output = outputs.join("\n---\n");
                let state = serde_json::json!({
                    "results": outputs,
                    "count": results.len()
                });
                (combined_output, state)
            }
            MergeStrategy::FirstSuccess => {
                let first_success = results.iter().find(|r| r.success);
                match first_success {
                    Some(result) => (result.output.clone(), result.state.clone()),
                    None => {
                        let first = results.first().unwrap();
                        (first.output.clone(), first.state.clone())
                    }
                }
            }
            MergeStrategy::LastSuccess => {
                let last_success = results.iter().rev().find(|r| r.success);
                match last_success {
                    Some(result) => (result.output.clone(), result.state.clone()),
                    None => {
                        let last = results.last().unwrap();
                        (last.output.clone(), last.state.clone())
                    }
                }
            }
            MergeStrategy::CollectArray => {
                let outputs: Vec<&str> = results.iter().map(|r| r.output.as_str()).collect();
                let states: Vec<_> = results.iter().map(|r| r.state.clone()).collect();
                let combined_output = serde_json::to_string(&outputs).unwrap_or_default();
                (combined_output, serde_json::json!(states))
            }
            MergeStrategy::DeepMerge => {
                let mut merged = serde_json::Map::new();
                let mut combined_output = String::new();

                for result in results {
                    if !combined_output.is_empty() {
                        combined_output.push('\n');
                    }
                    combined_output.push_str(&result.output);

                    if let serde_json::Value::Object(map) = &result.state {
                        for (key, value) in map {
                            merged.insert(key.clone(), value.clone());
                        }
                    }
                }

                (combined_output, serde_json::Value::Object(merged))
            }
        };

        Ok((output, state))
    }

    /// Compose results conditionally (select branch based on condition)
    fn compose_conditional(
        &self,
        results: &[SubtaskResult],
        condition: &str,
    ) -> Result<(String, serde_json::Value), AggregationError> {
        // Evaluate condition against the first result's state
        let condition_met = if let Some(first) = results.first() {
            self.evaluate_condition(condition, &first.state)
        } else {
            false
        };

        // Select appropriate branch
        let branch_idx = if condition_met {
            0
        } else {
            1.min(results.len() - 1)
        };

        if let Some(result) = results.get(branch_idx) {
            Ok((result.output.clone(), result.state.clone()))
        } else {
            let last = results.last().unwrap();
            Ok((last.output.clone(), last.state.clone()))
        }
    }

    /// Compose results using a custom function
    fn compose_custom(
        &self,
        results: &[SubtaskResult],
        name: &str,
        _params: &HashMap<String, serde_json::Value>,
    ) -> Result<(String, serde_json::Value), AggregationError> {
        // For now, custom functions fall back to sequential
        // Future: implement plugin system for custom composition functions
        match name {
            "sequential" => self.compose_sequential(results),
            "concatenate" => self.compose_parallel(results, &MergeStrategy::Concatenate),
            "first_success" => self.compose_parallel(results, &MergeStrategy::FirstSuccess),
            "last_success" => self.compose_parallel(results, &MergeStrategy::LastSuccess),
            "collect" => self.compose_parallel(results, &MergeStrategy::CollectArray),
            "merge" => self.compose_parallel(results, &MergeStrategy::DeepMerge),
            _ => Err(AggregationError::UnknownCompositionFunction {
                name: name.to_string(),
            }),
        }
    }

    /// Evaluate a condition against state
    fn evaluate_condition(&self, condition: &str, state: &serde_json::Value) -> bool {
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

    /// Validate the composed schema
    fn validate_composed_schema(&self, _state: &serde_json::Value) -> Result<(), AggregationError> {
        // Basic validation: state must be a valid JSON value (always true if we get here)
        // Future: implement JSON Schema validation
        Ok(())
    }

    /// Aggregate results with voting when multiple composition candidates exist
    ///
    /// This method is used when multiple decomposition proposals were executed
    /// and we need to vote on the best composed result.
    pub fn aggregate_with_voting(
        &self,
        candidates: Vec<(DecompositionProposal, Vec<SubtaskResult>)>,
        depth: usize,
    ) -> Result<AggregatedResult, AggregationError> {
        if candidates.is_empty() {
            return Err(AggregationError::VotingFailed {
                message: "No candidates to vote on".to_string(),
            });
        }

        if candidates.len() == 1 {
            let (proposal, results) = candidates.into_iter().next().unwrap();
            return self.aggregate(&proposal, results, depth);
        }

        // Aggregate each candidate
        let mut aggregated_candidates: Vec<(AggregatedResult, usize)> = Vec::new();
        for (idx, (proposal, results)) in candidates.into_iter().enumerate() {
            if let Ok(result) = self.aggregate(&proposal, results, depth) {
                aggregated_candidates.push((result, idx));
            }
        }

        if aggregated_candidates.is_empty() {
            return Err(AggregationError::VotingFailed {
                message: "All composition candidates failed".to_string(),
            });
        }

        if aggregated_candidates.len() == 1 {
            return Ok(aggregated_candidates.into_iter().next().unwrap().0);
        }

        // Vote on candidates based on success count and metrics
        let race =
            VoteRace::new(self.config.k_margin).map_err(|e| AggregationError::VotingFailed {
                message: e.to_string(),
            })?;

        // Cast votes based on success rate
        for (result, idx) in &aggregated_candidates {
            let score = result.success_count() * 100 / result.subtask_results.len().max(1);
            for _ in 0..score {
                race.cast_vote(idx.to_string().into());
            }
        }

        // Find winner
        let winner_idx = match race.check_winner() {
            crate::core::voting::VoteCheckResult::Winner { candidate, .. } => {
                candidate.0.parse::<usize>().unwrap_or(0)
            }
            crate::core::voting::VoteCheckResult::Ongoing { leader, .. } => leader
                .map(|l| l.0.parse::<usize>().unwrap_or(0))
                .unwrap_or(0),
        };

        let winner = aggregated_candidates
            .into_iter()
            .find(|(_, idx)| *idx == winner_idx)
            .map(|(result, _)| result)
            .unwrap_or_else(|| panic!("Winner index not found"));

        Ok(winner)
    }
}

impl Default for SolutionDiscriminator {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SolutionDiscriminator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SolutionDiscriminator")
            .field("config", &self.config)
            .finish()
    }
}

/// Convert CompositionFunction to a string representation
fn composition_fn_to_string(composition_fn: &CompositionFunction) -> String {
    match composition_fn {
        CompositionFunction::Sequential => "Sequential".to_string(),
        CompositionFunction::Parallel { merge_strategy } => {
            format!("Parallel({:?})", merge_strategy)
        }
        CompositionFunction::Conditional { condition } => {
            format!("Conditional({})", condition)
        }
        CompositionFunction::Custom { name, .. } => format!("Custom({})", name),
    }
}

/// Recursively compose nested decomposition results
///
/// This function handles trees of decomposition proposals where subtasks
/// may themselves be decomposed.
pub fn compose_recursive(
    discriminator: &SolutionDiscriminator,
    proposal: &DecompositionProposal,
    results: Vec<SubtaskResult>,
    nested_results: HashMap<String, AggregatedResult>,
    depth: usize,
) -> Result<AggregatedResult, AggregationError> {
    // Replace results with nested compositions where available
    let composed_results: Vec<SubtaskResult> = results
        .into_iter()
        .map(|result| {
            if let Some(nested) = nested_results.get(&result.task_id) {
                // Convert nested AggregatedResult to SubtaskResult
                SubtaskResult::success(
                    result.task_id.clone(),
                    nested.output.clone(),
                    nested.state.clone(),
                )
                .with_metrics(crate::core::decomposition::ExecutionMetrics {
                    elapsed_ms: nested.metrics.total_execution_ms,
                    ..Default::default()
                })
            } else {
                result
            }
        })
        .collect();

    discriminator.aggregate(proposal, composed_results, depth)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::decomposition::DecompositionSubtask;

    fn create_test_proposal(composition_fn: CompositionFunction) -> DecompositionProposal {
        DecompositionProposal::new(
            "test-proposal",
            "test-task",
            vec![
                DecompositionSubtask::leaf("step-1", "First step").with_order(0),
                DecompositionSubtask::leaf("step-2", "Second step").with_order(1),
            ],
            composition_fn,
        )
    }

    fn create_test_results() -> Vec<SubtaskResult> {
        vec![
            SubtaskResult::success(
                "step-1".to_string(),
                "result-1".to_string(),
                serde_json::json!({"step": 1}),
            ),
            SubtaskResult::success(
                "step-2".to_string(),
                "result-2".to_string(),
                serde_json::json!({"step": 2}),
            ),
        ]
    }

    // ==========================================
    // AggregatedResult Tests
    // ==========================================

    #[test]
    fn test_aggregated_result_new() {
        let results = create_test_results();
        let result = AggregatedResult::new(
            "proposal-1".to_string(),
            "task-1".to_string(),
            "output".to_string(),
            serde_json::json!({}),
            "Sequential".to_string(),
            results,
        );

        assert_eq!(result.proposal_id, "proposal-1");
        assert!(result.all_succeeded);
        assert_eq!(result.success_count(), 2);
        assert_eq!(result.failure_count(), 0);
    }

    #[test]
    fn test_aggregated_result_with_failures() {
        let results = vec![
            SubtaskResult::success("s1".to_string(), "ok".to_string(), serde_json::Value::Null),
            SubtaskResult::failure("s2".to_string(), "error".to_string()),
        ];

        let result = AggregatedResult::new(
            "p1".to_string(),
            "t1".to_string(),
            "out".to_string(),
            serde_json::Value::Null,
            "Sequential".to_string(),
            results,
        );

        assert!(!result.all_succeeded);
        assert_eq!(result.success_count(), 1);
        assert_eq!(result.failure_count(), 1);
    }

    // ==========================================
    // AggregatorConfig Tests
    // ==========================================

    #[test]
    fn test_aggregator_config_default() {
        let config = AggregatorConfig::default();
        assert_eq!(config.k_margin, 3);
        assert!(config.validate_schema);
        assert_eq!(config.max_depth, 10);
    }

    #[test]
    fn test_aggregator_config_builder() {
        let config = AggregatorConfig::default()
            .with_k_margin(5)
            .with_max_depth(20)
            .without_schema_validation();

        assert_eq!(config.k_margin, 5);
        assert_eq!(config.max_depth, 20);
        assert!(!config.validate_schema);
    }

    // ==========================================
    // AggregationError Tests
    // ==========================================

    #[test]
    fn test_aggregation_error_display() {
        let errors = vec![
            AggregationError::EmptyResults {
                proposal_id: "p1".to_string(),
            },
            AggregationError::AllFailed {
                proposal_id: "p2".to_string(),
                failure_count: 3,
            },
            AggregationError::SchemaValidationFailed {
                message: "invalid".to_string(),
            },
            AggregationError::RecursiveCompositionFailed {
                depth: 5,
                message: "too deep".to_string(),
            },
            AggregationError::VotingFailed {
                message: "no candidates".to_string(),
            },
            AggregationError::UnknownCompositionFunction {
                name: "unknown".to_string(),
            },
        ];

        for error in errors {
            let display = format!("{}", error);
            assert!(!display.is_empty());
        }
    }

    // ==========================================
    // SolutionDiscriminator Tests
    // ==========================================

    #[test]
    fn test_solution_discriminator_new() {
        let discriminator = SolutionDiscriminator::new();
        assert_eq!(discriminator.config().k_margin, 3);
    }

    #[test]
    fn test_solution_discriminator_with_config() {
        let config = AggregatorConfig::default().with_k_margin(5);
        let discriminator = SolutionDiscriminator::with_config(config);
        assert_eq!(discriminator.config().k_margin, 5);
    }

    #[test]
    fn test_aggregate_sequential() {
        let discriminator = SolutionDiscriminator::new();
        let proposal = create_test_proposal(CompositionFunction::Sequential);
        let results = create_test_results();

        let aggregated = discriminator.aggregate(&proposal, results, 0);

        assert!(aggregated.is_ok());
        let result = aggregated.unwrap();
        assert_eq!(result.output, "result-2"); // Last result wins
        assert!(result.all_succeeded);
    }

    #[test]
    fn test_aggregate_parallel_concatenate() {
        let discriminator = SolutionDiscriminator::new();
        let proposal = create_test_proposal(CompositionFunction::Parallel {
            merge_strategy: MergeStrategy::Concatenate,
        });
        let results = create_test_results();

        let aggregated = discriminator.aggregate(&proposal, results, 0);

        assert!(aggregated.is_ok());
        let result = aggregated.unwrap();
        assert!(result.output.contains("result-1"));
        assert!(result.output.contains("result-2"));
    }

    #[test]
    fn test_aggregate_parallel_first_success() {
        let discriminator = SolutionDiscriminator::new();
        let proposal = create_test_proposal(CompositionFunction::Parallel {
            merge_strategy: MergeStrategy::FirstSuccess,
        });
        let results = create_test_results();

        let aggregated = discriminator.aggregate(&proposal, results, 0);

        assert!(aggregated.is_ok());
        let result = aggregated.unwrap();
        assert_eq!(result.output, "result-1"); // First success
    }

    #[test]
    fn test_aggregate_parallel_deep_merge() {
        let discriminator = SolutionDiscriminator::new();
        let proposal = create_test_proposal(CompositionFunction::Parallel {
            merge_strategy: MergeStrategy::DeepMerge,
        });
        let results = vec![
            SubtaskResult::success(
                "s1".to_string(),
                "out1".to_string(),
                serde_json::json!({"key1": "value1"}),
            ),
            SubtaskResult::success(
                "s2".to_string(),
                "out2".to_string(),
                serde_json::json!({"key2": "value2"}),
            ),
        ];

        let aggregated = discriminator.aggregate(&proposal, results, 0);

        assert!(aggregated.is_ok());
        let result = aggregated.unwrap();
        assert!(result.state.get("key1").is_some());
        assert!(result.state.get("key2").is_some());
    }

    #[test]
    fn test_aggregate_conditional() {
        let discriminator = SolutionDiscriminator::new();
        let proposal = create_test_proposal(CompositionFunction::Conditional {
            condition: "should_branch".to_string(),
        });
        let results = vec![
            SubtaskResult::success(
                "s1".to_string(),
                "branch-true".to_string(),
                serde_json::json!({"should_branch": true}),
            ),
            SubtaskResult::success(
                "s2".to_string(),
                "branch-false".to_string(),
                serde_json::json!({"should_branch": false}),
            ),
        ];

        let aggregated = discriminator.aggregate(&proposal, results, 0);

        assert!(aggregated.is_ok());
        let result = aggregated.unwrap();
        // First result has should_branch=true, so we take branch 0
        assert_eq!(result.output, "branch-true");
    }

    #[test]
    fn test_aggregate_custom_fallback() {
        let discriminator = SolutionDiscriminator::new();
        let proposal = create_test_proposal(CompositionFunction::Custom {
            name: "sequential".to_string(),
            params: HashMap::new(),
        });
        let results = create_test_results();

        let aggregated = discriminator.aggregate(&proposal, results, 0);

        assert!(aggregated.is_ok());
    }

    #[test]
    fn test_aggregate_unknown_custom() {
        let discriminator = SolutionDiscriminator::new();
        let proposal = create_test_proposal(CompositionFunction::Custom {
            name: "unknown_fn".to_string(),
            params: HashMap::new(),
        });
        let results = create_test_results();

        let aggregated = discriminator.aggregate(&proposal, results, 0);

        assert!(aggregated.is_err());
        assert!(matches!(
            aggregated.unwrap_err(),
            AggregationError::UnknownCompositionFunction { .. }
        ));
    }

    #[test]
    fn test_aggregate_empty_results() {
        let discriminator = SolutionDiscriminator::new();
        let proposal = create_test_proposal(CompositionFunction::Sequential);

        let aggregated = discriminator.aggregate(&proposal, vec![], 0);

        assert!(aggregated.is_err());
        assert!(matches!(
            aggregated.unwrap_err(),
            AggregationError::EmptyResults { .. }
        ));
    }

    #[test]
    fn test_aggregate_all_failed() {
        let discriminator = SolutionDiscriminator::new();
        let proposal = create_test_proposal(CompositionFunction::Sequential);
        let results = vec![
            SubtaskResult::failure("s1".to_string(), "error1".to_string()),
            SubtaskResult::failure("s2".to_string(), "error2".to_string()),
        ];

        let aggregated = discriminator.aggregate(&proposal, results, 0);

        assert!(aggregated.is_err());
        assert!(matches!(
            aggregated.unwrap_err(),
            AggregationError::AllFailed { .. }
        ));
    }

    #[test]
    fn test_aggregate_exceeds_depth() {
        let config = AggregatorConfig::default().with_max_depth(5);
        let discriminator = SolutionDiscriminator::with_config(config);
        let proposal = create_test_proposal(CompositionFunction::Sequential);
        let results = create_test_results();

        let aggregated = discriminator.aggregate(&proposal, results, 10);

        assert!(aggregated.is_err());
        assert!(matches!(
            aggregated.unwrap_err(),
            AggregationError::RecursiveCompositionFailed { .. }
        ));
    }

    #[test]
    fn test_aggregate_metrics() {
        let discriminator = SolutionDiscriminator::new();
        let proposal = create_test_proposal(CompositionFunction::Sequential);
        let results = create_test_results();

        let aggregated = discriminator.aggregate(&proposal, results, 2);

        assert!(aggregated.is_ok());
        let result = aggregated.unwrap();
        assert_eq!(result.metrics.subtask_count, 2);
        assert_eq!(result.metrics.success_count, 2);
        assert_eq!(result.metrics.depth, 2);
    }

    #[test]
    fn test_aggregate_with_voting_single_candidate() {
        let discriminator = SolutionDiscriminator::new();
        let proposal = create_test_proposal(CompositionFunction::Sequential);
        let results = create_test_results();

        let aggregated = discriminator.aggregate_with_voting(vec![(proposal, results)], 0);

        assert!(aggregated.is_ok());
    }

    #[test]
    fn test_aggregate_with_voting_no_candidates() {
        let discriminator = SolutionDiscriminator::new();

        let aggregated = discriminator.aggregate_with_voting(vec![], 0);

        assert!(aggregated.is_err());
        assert!(matches!(
            aggregated.unwrap_err(),
            AggregationError::VotingFailed { .. }
        ));
    }

    // ==========================================
    // Composition Function String Tests
    // ==========================================

    #[test]
    fn test_composition_fn_to_string() {
        assert_eq!(
            composition_fn_to_string(&CompositionFunction::Sequential),
            "Sequential"
        );
        assert!(composition_fn_to_string(&CompositionFunction::Parallel {
            merge_strategy: MergeStrategy::DeepMerge
        })
        .contains("Parallel"));
        assert!(composition_fn_to_string(&CompositionFunction::Conditional {
            condition: "test".to_string()
        })
        .contains("Conditional"));
        assert!(composition_fn_to_string(&CompositionFunction::Custom {
            name: "custom".to_string(),
            params: HashMap::new()
        })
        .contains("Custom"));
    }

    // ==========================================
    // Recursive Composition Tests
    // ==========================================

    #[test]
    fn test_compose_recursive_replaces_nested() {
        let discriminator = SolutionDiscriminator::new();
        let proposal = create_test_proposal(CompositionFunction::Sequential);

        let results = vec![SubtaskResult::success(
            "step-1".to_string(),
            "original".to_string(),
            serde_json::json!({}),
        )];

        let mut nested = HashMap::new();
        nested.insert(
            "step-1".to_string(),
            AggregatedResult::new(
                "nested-proposal".to_string(),
                "nested-task".to_string(),
                "nested-output".to_string(),
                serde_json::json!({"nested": true}),
                "Sequential".to_string(),
                vec![],
            ),
        );

        let aggregated = compose_recursive(&discriminator, &proposal, results, nested, 0);

        assert!(aggregated.is_ok());
        let result = aggregated.unwrap();
        assert_eq!(result.output, "nested-output"); // Replaced with nested
    }

    // ==========================================
    // Integration Test: 3-Level Deep Decomposition
    // ==========================================

    #[test]
    fn test_three_level_deep_decomposition() {
        // Simulates a 3-level deep decomposition:
        // Level 0 (root): "main-task" -> 2 subtasks
        // Level 1: "sub-1" -> 2 subtasks, "sub-2" -> 2 subtasks
        // Level 2: 4 leaf nodes executed

        let discriminator = SolutionDiscriminator::new();

        // Level 2: Leaf node results (4 leaves)
        let level2_results = vec![
            SubtaskResult::success(
                "leaf-1-1".to_string(),
                "result-1-1".to_string(),
                serde_json::json!({"leaf": "1-1", "value": 1}),
            ),
            SubtaskResult::success(
                "leaf-1-2".to_string(),
                "result-1-2".to_string(),
                serde_json::json!({"leaf": "1-2", "value": 2}),
            ),
            SubtaskResult::success(
                "leaf-2-1".to_string(),
                "result-2-1".to_string(),
                serde_json::json!({"leaf": "2-1", "value": 3}),
            ),
            SubtaskResult::success(
                "leaf-2-2".to_string(),
                "result-2-2".to_string(),
                serde_json::json!({"leaf": "2-2", "value": 4}),
            ),
        ];

        // Level 1: Compose sub-1 (leaves 1-1, 1-2) and sub-2 (leaves 2-1, 2-2)
        let sub1_proposal = DecompositionProposal::new(
            "sub1-proposal",
            "sub-1",
            vec![
                DecompositionSubtask::leaf("leaf-1-1", "Leaf 1-1").with_order(0),
                DecompositionSubtask::leaf("leaf-1-2", "Leaf 1-2").with_order(1),
            ],
            CompositionFunction::Parallel {
                merge_strategy: MergeStrategy::DeepMerge,
            },
        );

        let sub2_proposal = DecompositionProposal::new(
            "sub2-proposal",
            "sub-2",
            vec![
                DecompositionSubtask::leaf("leaf-2-1", "Leaf 2-1").with_order(0),
                DecompositionSubtask::leaf("leaf-2-2", "Leaf 2-2").with_order(1),
            ],
            CompositionFunction::Sequential,
        );

        // Aggregate level 1 results
        let sub1_aggregated = discriminator
            .aggregate(
                &sub1_proposal,
                vec![level2_results[0].clone(), level2_results[1].clone()],
                1,
            )
            .expect("sub1 aggregation should succeed");

        let sub2_aggregated = discriminator
            .aggregate(
                &sub2_proposal,
                vec![level2_results[2].clone(), level2_results[3].clone()],
                1,
            )
            .expect("sub2 aggregation should succeed");

        // Level 0: Compose the root task from sub-1 and sub-2
        let root_proposal = DecompositionProposal::new(
            "root-proposal",
            "main-task",
            vec![
                DecompositionSubtask::leaf("sub-1", "Subtask 1").with_order(0),
                DecompositionSubtask::leaf("sub-2", "Subtask 2").with_order(1),
            ],
            CompositionFunction::Sequential,
        );

        // Create intermediate results from level 1 aggregations
        let level1_results = vec![
            SubtaskResult::success(
                "sub-1".to_string(),
                sub1_aggregated.output.clone(),
                sub1_aggregated.state.clone(),
            ),
            SubtaskResult::success(
                "sub-2".to_string(),
                sub2_aggregated.output.clone(),
                sub2_aggregated.state.clone(),
            ),
        ];

        // Final aggregation at level 0
        let root_aggregated = discriminator
            .aggregate(&root_proposal, level1_results, 0)
            .expect("root aggregation should succeed");

        // Verify the 3-level composition worked
        assert!(root_aggregated.all_succeeded);
        assert_eq!(root_aggregated.metrics.depth, 0);
        assert_eq!(root_aggregated.subtask_results.len(), 2);

        // The final output should be from the last subtask (Sequential composition)
        assert_eq!(root_aggregated.output, sub2_aggregated.output);

        // Verify sub-aggregations maintained correct depth
        assert_eq!(sub1_aggregated.metrics.depth, 1);
        assert_eq!(sub2_aggregated.metrics.depth, 1);
    }

    #[test]
    fn test_five_level_deep_decomposition() {
        // Verify composition works to depth 5+
        let discriminator = SolutionDiscriminator::new();

        // Create a chain of 5 nested aggregations
        let mut current_output = "leaf-output".to_string();
        let mut current_state = serde_json::json!({"depth": 5});

        for depth in (0..5).rev() {
            let proposal = DecompositionProposal::new(
                format!("proposal-depth-{}", depth),
                format!("task-depth-{}", depth),
                vec![DecompositionSubtask::leaf(
                    format!("subtask-{}", depth),
                    format!("Subtask at depth {}", depth),
                )],
                CompositionFunction::Sequential,
            );

            let results = vec![SubtaskResult::success(
                format!("subtask-{}", depth),
                current_output.clone(),
                current_state.clone(),
            )];

            let aggregated = discriminator
                .aggregate(&proposal, results, depth)
                .expect(&format!("aggregation at depth {} should succeed", depth));

            current_output = aggregated.output;
            current_state = aggregated.state;

            assert_eq!(aggregated.metrics.depth, depth);
        }

        // Final state should have original leaf value
        assert_eq!(current_state["depth"], 5);
    }

    #[test]
    fn test_nested_parallel_and_sequential() {
        // Test mixing Parallel and Sequential at different levels
        let discriminator = SolutionDiscriminator::new();

        // Level 1: Parallel composition of 3 results
        let parallel_proposal = DecompositionProposal::new(
            "parallel-proposal",
            "parallel-task",
            vec![
                DecompositionSubtask::leaf("p1", "Parallel 1").with_order(0),
                DecompositionSubtask::leaf("p2", "Parallel 2").with_order(1),
                DecompositionSubtask::leaf("p3", "Parallel 3").with_order(2),
            ],
            CompositionFunction::Parallel {
                merge_strategy: MergeStrategy::CollectArray,
            },
        );

        let parallel_results = vec![
            SubtaskResult::success(
                "p1".to_string(),
                "A".to_string(),
                serde_json::json!({"v": 1}),
            ),
            SubtaskResult::success(
                "p2".to_string(),
                "B".to_string(),
                serde_json::json!({"v": 2}),
            ),
            SubtaskResult::success(
                "p3".to_string(),
                "C".to_string(),
                serde_json::json!({"v": 3}),
            ),
        ];

        let parallel_aggregated = discriminator
            .aggregate(&parallel_proposal, parallel_results, 1)
            .expect("parallel aggregation should succeed");

        // Level 0: Sequential composition using parallel result
        let sequential_proposal = DecompositionProposal::new(
            "sequential-proposal",
            "root-task",
            vec![
                DecompositionSubtask::leaf("setup", "Setup").with_order(0),
                DecompositionSubtask::leaf("parallel-result", "Parallel").with_order(1),
            ],
            CompositionFunction::Sequential,
        );

        let sequential_results = vec![
            SubtaskResult::success(
                "setup".to_string(),
                "setup-done".to_string(),
                serde_json::json!({"setup": true}),
            ),
            SubtaskResult::success(
                "parallel-result".to_string(),
                parallel_aggregated.output.clone(),
                parallel_aggregated.state.clone(),
            ),
        ];

        let final_aggregated = discriminator
            .aggregate(&sequential_proposal, sequential_results, 0)
            .expect("final aggregation should succeed");

        // Verify mixed composition
        assert!(final_aggregated.all_succeeded);
        // Sequential takes last result, which is the parallel aggregation
        assert!(final_aggregated.state.is_array()); // CollectArray produces array
    }
}
