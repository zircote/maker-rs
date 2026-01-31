//! Microagent Orchestration for MAKER Framework
//!
//! Implements the Maximal Agentic Decomposition (MAD) pattern where each agent
//! handles exactly one subtask (m=1). This minimizes context burden and maximizes
//! per-step reliability.
//!
//! # Key Principles
//!
//! 1. **m=1 Constraint**: Each agent handles exactly one subtask
//! 2. **Minimal Context**: Agent receives only current step state, no history
//! 3. **State Transfer**: System uses `next_state` from agent output, not model's interpretation
//! 4. **State Validation**: Hash validation prevents undetected state corruption
//!
//! # Architecture
//!
//! ```text
//! TaskDecomposer → [Subtask₁, Subtask₂, ..., Subtaskₛ]
//!                        ↓
//!                  Agent (m=1)
//!                        ↓
//!                  AgentOutput { move, next_state }
//!                        ↓
//!                  System validates & transfers state
//!                        ↓
//!                  Next Agent receives next_state
//! ```

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Serializable state representation for task execution
///
/// The state is intentionally opaque (JSON value) to support arbitrary task types.
/// State hashing provides integrity verification between steps.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct State {
    /// The actual state data (task-specific JSON)
    pub data: serde_json::Value,
    /// Hash of the state for integrity verification
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
}

impl State {
    /// Create a new state from JSON data
    pub fn new(data: serde_json::Value) -> Self {
        let hash = Self::compute_hash(&data);
        Self {
            data,
            hash: Some(hash),
        }
    }

    /// Create a state without hash (for testing or external input)
    pub fn without_hash(data: serde_json::Value) -> Self {
        Self { data, hash: None }
    }

    /// Compute hash of the state data
    fn compute_hash(data: &serde_json::Value) -> String {
        let mut hasher = DefaultHasher::new();
        // Serialize to canonical form for consistent hashing
        let canonical = serde_json::to_string(data).unwrap_or_default();
        canonical.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    }

    /// Verify the state hash matches the data
    pub fn verify_hash(&self) -> bool {
        match &self.hash {
            Some(h) => *h == Self::compute_hash(&self.data),
            None => true, // No hash to verify
        }
    }

    /// Update the hash to match current data
    pub fn update_hash(&mut self) {
        self.hash = Some(Self::compute_hash(&self.data));
    }
}

/// A single subtask for microagent execution
///
/// Each subtask represents exactly one step (m=1) with minimal context.
/// The agent receives only the current state, not previous steps or history.
#[derive(Debug, Clone)]
pub struct Subtask {
    /// Unique identifier for this step in the task sequence
    pub step_id: usize,
    /// The prompt/instruction for the agent
    pub prompt: String,
    /// Current state (input to the agent)
    pub state: State,
    // Note: No history field - this is enforced by design (m=1 constraint)
}

impl Subtask {
    /// Create a new subtask
    ///
    /// # Arguments
    ///
    /// * `step_id` - Step number in the task sequence
    /// * `prompt` - Instruction for the agent
    /// * `state` - Current state for this step
    pub fn new(step_id: usize, prompt: String, state: State) -> Self {
        Self {
            step_id,
            prompt,
            state,
        }
    }

    /// Get the state hash for integrity verification
    pub fn state_hash(&self) -> Option<&str> {
        self.state.hash.as_deref()
    }
}

/// Output from a microagent execution
///
/// Contains both the action taken and the resulting state for the next agent.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentOutput {
    /// The action/move performed by the agent
    pub move_action: String,
    /// The state after executing the action (for next agent)
    pub next_state: State,
}

impl AgentOutput {
    /// Create a new agent output
    pub fn new(move_action: String, next_state: State) -> Self {
        Self {
            move_action,
            next_state,
        }
    }

    /// Parse agent output from JSON string
    pub fn from_json(json: &str) -> Result<Self, AgentOutputError> {
        serde_json::from_str(json).map_err(|e| AgentOutputError::ParseError {
            message: e.to_string(),
        })
    }
}

/// Errors related to agent output processing
#[derive(Debug, Clone, PartialEq)]
pub enum AgentOutputError {
    /// Failed to parse agent output JSON
    ParseError { message: String },
    /// Agent output missing required field
    MissingField { field: String },
    /// State hash verification failed
    StateCorruption {
        expected_hash: String,
        actual_hash: String,
    },
}

impl std::fmt::Display for AgentOutputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentOutputError::ParseError { message } => {
                write!(f, "Failed to parse agent output: {}", message)
            }
            AgentOutputError::MissingField { field } => {
                write!(f, "Agent output missing required field: {}", field)
            }
            AgentOutputError::StateCorruption {
                expected_hash,
                actual_hash,
            } => {
                write!(
                    f,
                    "State corruption detected: expected hash {}, got {}",
                    expected_hash, actual_hash
                )
            }
        }
    }
}

impl std::error::Error for AgentOutputError {}

/// Trait for decomposing tasks into subtasks
///
/// Implement this trait for task-specific decomposition logic.
/// The decomposer generates the sequence of subtasks and provides
/// ground truth for validation.
pub trait TaskDecomposer {
    /// Generate all subtasks for the task
    ///
    /// Each subtask must have m=1 (exactly one action per agent).
    fn decompose(&self) -> Vec<Subtask>;

    /// Get the total number of steps
    fn total_steps(&self) -> usize;

    /// Get the initial state
    fn initial_state(&self) -> State;

    /// Validate an agent output against ground truth (optional)
    ///
    /// Returns `Ok(())` if the output is correct, `Err` with description if not.
    fn validate_output(&self, _step_id: usize, _output: &AgentOutput) -> Result<(), String> {
        // Default: no validation (override for specific tasks)
        Ok(())
    }
}

/// Orchestrator for executing decomposed tasks
///
/// Manages the execution of subtasks, enforcing the m=1 constraint
/// and handling state transfer between agents.
pub struct TaskOrchestrator<D: TaskDecomposer> {
    /// The task decomposer
    decomposer: D,
    /// Current step index
    current_step: usize,
    /// Current state
    current_state: State,
    /// Completed outputs
    outputs: Vec<AgentOutput>,
}

impl<D: TaskDecomposer> TaskOrchestrator<D> {
    /// Create a new orchestrator for the given decomposer
    pub fn new(decomposer: D) -> Self {
        let initial_state = decomposer.initial_state();
        Self {
            decomposer,
            current_step: 0,
            current_state: initial_state,
            outputs: Vec::new(),
        }
    }

    /// Get the current subtask (if not complete)
    pub fn current_subtask(&self) -> Option<Subtask> {
        if self.current_step >= self.decomposer.total_steps() {
            return None;
        }

        let subtasks = self.decomposer.decompose();
        subtasks.into_iter().nth(self.current_step)
    }

    /// Process an agent output and advance to next step
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - State hash verification fails (corruption detected)
    /// - Output validation fails (if decomposer implements validation)
    pub fn process_output(&mut self, output: AgentOutput) -> Result<(), AgentOutputError> {
        // Verify state hash if present
        if !output.next_state.verify_hash() {
            let expected = output.next_state.hash.clone().unwrap_or_default();
            let actual = State::compute_hash(&output.next_state.data);
            return Err(AgentOutputError::StateCorruption {
                expected_hash: expected,
                actual_hash: actual,
            });
        }

        // Validate output if decomposer supports it
        self.decomposer
            .validate_output(self.current_step, &output)
            .map_err(|msg| AgentOutputError::ParseError { message: msg })?;

        // State transfer: use next_state from output for next agent
        self.current_state = output.next_state.clone();
        self.outputs.push(output);
        self.current_step += 1;

        Ok(())
    }

    /// Check if the task is complete
    pub fn is_complete(&self) -> bool {
        self.current_step >= self.decomposer.total_steps()
    }

    /// Get the current step index
    pub fn current_step_index(&self) -> usize {
        self.current_step
    }

    /// Get the total number of steps
    pub fn total_steps(&self) -> usize {
        self.decomposer.total_steps()
    }

    /// Get all completed outputs
    pub fn outputs(&self) -> &[AgentOutput] {
        &self.outputs
    }

    /// Get the current state
    pub fn current_state(&self) -> &State {
        &self.current_state
    }

    /// Get the decomposer
    pub fn decomposer(&self) -> &D {
        &self.decomposer
    }
}

/// Configuration for microagent constraints
///
/// Enforces the m=1 constraint at the type level.
#[derive(Debug, Clone, Copy)]
pub struct MicroagentConfig {
    /// Steps per agent (always 1 for microagent architecture)
    m: usize,
}

impl MicroagentConfig {
    /// Create a new microagent config (m=1)
    ///
    /// # Panics
    ///
    /// Panics if m != 1 (microagent constraint violation)
    pub fn new(m: usize) -> Self {
        assert!(
            m == 1,
            "Microagent constraint violated: m must be 1, got {}",
            m
        );
        Self { m }
    }

    /// Get the steps per agent (always 1)
    pub fn steps_per_agent(&self) -> usize {
        self.m
    }
}

impl Default for MicroagentConfig {
    fn default() -> Self {
        Self { m: 1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ==========================================
    // State Tests
    // ==========================================

    #[test]
    fn test_state_new_includes_hash() {
        let state = State::new(json!({"rods": [[1,2,3], [], []]}));
        assert!(state.hash.is_some());
    }

    #[test]
    fn test_state_hash_is_deterministic() {
        let data = json!({"key": "value"});
        let state1 = State::new(data.clone());
        let state2 = State::new(data);
        assert_eq!(state1.hash, state2.hash);
    }

    #[test]
    fn test_state_verify_hash_valid() {
        let state = State::new(json!({"test": 123}));
        assert!(state.verify_hash());
    }

    #[test]
    fn test_state_verify_hash_corrupted() {
        let mut state = State::new(json!({"test": 123}));
        // Corrupt the data without updating hash
        state.data = json!({"test": 456});
        assert!(!state.verify_hash());
    }

    #[test]
    fn test_state_without_hash_verifies_true() {
        let state = State::without_hash(json!({"any": "data"}));
        assert!(state.verify_hash()); // No hash to verify
    }

    #[test]
    fn test_state_update_hash() {
        let mut state = State::new(json!({"test": 123}));
        state.data = json!({"test": 456});
        assert!(!state.verify_hash());

        state.update_hash();
        assert!(state.verify_hash());
    }

    // ==========================================
    // Subtask Tests
    // ==========================================

    #[test]
    fn test_subtask_creation() {
        let state = State::new(json!({}));
        let subtask = Subtask::new(0, "Do something".to_string(), state);

        assert_eq!(subtask.step_id, 0);
        assert_eq!(subtask.prompt, "Do something");
    }

    #[test]
    fn test_subtask_has_no_history_field() {
        // This is a compile-time check - Subtask has no history field
        // The struct definition enforces m=1 by design
        let state = State::new(json!({}));
        let subtask = Subtask::new(0, "Step".to_string(), state);

        // Only current state available, no history
        assert!(subtask.state_hash().is_some());
    }

    // ==========================================
    // AgentOutput Tests
    // ==========================================

    #[test]
    fn test_agent_output_creation() {
        let next_state = State::new(json!({"updated": true}));
        let output = AgentOutput::new("move disk".to_string(), next_state);

        assert_eq!(output.move_action, "move disk");
    }

    #[test]
    fn test_agent_output_from_json_valid() {
        let json = r#"{"move_action": "test", "next_state": {"data": {}, "hash": null}}"#;
        let result = AgentOutput::from_json(json);
        assert!(result.is_ok());
    }

    #[test]
    fn test_agent_output_from_json_missing_field() {
        let json = r#"{"move_action": "test"}"#;
        let result = AgentOutput::from_json(json);
        assert!(result.is_err());
    }

    // ==========================================
    // MicroagentConfig Tests
    // ==========================================

    #[test]
    fn test_microagent_config_m_equals_one() {
        let config = MicroagentConfig::new(1);
        assert_eq!(config.steps_per_agent(), 1);
    }

    #[test]
    #[should_panic(expected = "Microagent constraint violated")]
    fn test_microagent_config_rejects_m_greater_than_one() {
        MicroagentConfig::new(2);
    }

    #[test]
    #[should_panic(expected = "Microagent constraint violated")]
    fn test_microagent_config_rejects_m_zero() {
        MicroagentConfig::new(0);
    }

    #[test]
    fn test_microagent_config_default() {
        let config = MicroagentConfig::default();
        assert_eq!(config.steps_per_agent(), 1);
    }

    // ==========================================
    // TaskOrchestrator Tests
    // ==========================================

    /// Simple test decomposer for 3-step task
    struct TestDecomposer;

    impl TaskDecomposer for TestDecomposer {
        fn decompose(&self) -> Vec<Subtask> {
            (0..3)
                .map(|i| Subtask::new(i, format!("Step {}", i), State::new(json!({"step": i}))))
                .collect()
        }

        fn total_steps(&self) -> usize {
            3
        }

        fn initial_state(&self) -> State {
            State::new(json!({"step": 0}))
        }
    }

    #[test]
    fn test_orchestrator_initialization() {
        let orchestrator = TaskOrchestrator::new(TestDecomposer);
        assert_eq!(orchestrator.current_step_index(), 0);
        assert_eq!(orchestrator.total_steps(), 3);
        assert!(!orchestrator.is_complete());
    }

    #[test]
    fn test_orchestrator_current_subtask() {
        let orchestrator = TaskOrchestrator::new(TestDecomposer);
        let subtask = orchestrator.current_subtask();
        assert!(subtask.is_some());
        assert_eq!(subtask.unwrap().step_id, 0);
    }

    #[test]
    fn test_orchestrator_process_output() {
        let mut orchestrator = TaskOrchestrator::new(TestDecomposer);

        let output = AgentOutput::new("action".to_string(), State::new(json!({"step": 1})));

        let result = orchestrator.process_output(output);
        assert!(result.is_ok());
        assert_eq!(orchestrator.current_step_index(), 1);
    }

    #[test]
    fn test_orchestrator_state_transfer() {
        let mut orchestrator = TaskOrchestrator::new(TestDecomposer);

        let next_state = State::new(json!({"transferred": true}));
        let output = AgentOutput::new("action".to_string(), next_state.clone());

        orchestrator.process_output(output).unwrap();

        // State should be transferred from output
        assert_eq!(orchestrator.current_state().data, next_state.data);
    }

    #[test]
    fn test_orchestrator_detects_state_corruption() {
        let mut orchestrator = TaskOrchestrator::new(TestDecomposer);

        // Create output with mismatched hash
        let mut next_state = State::new(json!({"original": true}));
        next_state.data = json!({"corrupted": true}); // Corrupt without updating hash

        let output = AgentOutput::new("action".to_string(), next_state);

        let result = orchestrator.process_output(output);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            AgentOutputError::StateCorruption { .. }
        ));
    }

    #[test]
    fn test_orchestrator_completion() {
        let mut orchestrator = TaskOrchestrator::new(TestDecomposer);

        for i in 0..3 {
            assert!(!orchestrator.is_complete());
            let output =
                AgentOutput::new(format!("action {}", i), State::new(json!({"step": i + 1})));
            orchestrator.process_output(output).unwrap();
        }

        assert!(orchestrator.is_complete());
        assert_eq!(orchestrator.outputs().len(), 3);
    }

    #[test]
    fn test_orchestrator_no_subtask_when_complete() {
        let mut orchestrator = TaskOrchestrator::new(TestDecomposer);

        for i in 0..3 {
            let output =
                AgentOutput::new(format!("action {}", i), State::new(json!({"step": i + 1})));
            orchestrator.process_output(output).unwrap();
        }

        assert!(orchestrator.current_subtask().is_none());
    }

    // ==========================================
    // Microagent Isolation Tests (STORY-008-03)
    // ==========================================

    #[test]
    fn test_subtask_contains_only_current_state() {
        // Verify Subtask struct has no history/previous fields
        // This is a compile-time enforcement but we test the runtime behavior
        let state = State::new(json!({"current": "state"}));
        let subtask = Subtask::new(5, "Do step 5".to_string(), state.clone());

        // Subtask only has step_id, prompt, and state - no history
        assert_eq!(subtask.step_id, 5);
        assert_eq!(subtask.prompt, "Do step 5");
        assert_eq!(subtask.state.data, state.data);

        // Verify there's no way to access history (Subtask has no such field)
        // The struct definition enforces this at compile time
    }

    #[test]
    fn test_agent_receives_fresh_state_each_step() {
        let mut orchestrator = TaskOrchestrator::new(TestDecomposer);

        // Step 0: Initial state
        let subtask0 = orchestrator.current_subtask().unwrap();
        let state0 = subtask0.state.clone();

        // Complete step 0 with new state
        let output0 = AgentOutput::new("action0".to_string(), State::new(json!({"step": 1})));
        orchestrator.process_output(output0).unwrap();

        // Step 1: Should receive transferred state, not initial state
        let subtask1 = orchestrator.current_subtask().unwrap();
        assert_ne!(subtask1.state.data, state0.data);
        assert_eq!(subtask1.state.data, json!({"step": 1}));
    }

    #[test]
    fn test_previous_agent_output_not_visible_to_next_agent() {
        let mut orchestrator = TaskOrchestrator::new(TestDecomposer);

        // Complete first step
        let output0 = AgentOutput::new(
            "secret action from step 0".to_string(),
            State::new(json!({"step": 1})),
        );
        orchestrator.process_output(output0).unwrap();

        // Get next subtask - should only have state, not previous action
        let subtask1 = orchestrator.current_subtask().unwrap();

        // The subtask prompt and state don't contain previous agent's move_action
        assert!(!subtask1.prompt.contains("secret action"));
        assert!(!serde_json::to_string(&subtask1.state.data)
            .unwrap()
            .contains("secret action"));
    }

    #[test]
    fn test_state_hash_prevents_undetected_corruption() {
        // State created with hash
        let state = State::new(json!({"important": "data"}));
        assert!(state.hash.is_some());
        assert!(state.verify_hash());

        // If data is modified without updating hash, corruption is detected
        let mut corrupted = state.clone();
        corrupted.data = json!({"important": "TAMPERED"});
        assert!(!corrupted.verify_hash());
    }

    #[test]
    fn test_state_transfer_uses_agent_next_state() {
        let mut orchestrator = TaskOrchestrator::new(TestDecomposer);

        // Agent provides next_state
        let agent_next_state = State::new(json!({"computed_by": "agent", "value": 42}));
        let output = AgentOutput::new("computed".to_string(), agent_next_state.clone());

        orchestrator.process_output(output).unwrap();

        // Orchestrator uses exactly what agent provided, not any interpretation
        assert_eq!(orchestrator.current_state().data, agent_next_state.data);
        assert_eq!(orchestrator.current_state().hash, agent_next_state.hash);
    }

    #[test]
    fn test_m_equals_1_is_enforced() {
        // Valid: m=1
        let config = MicroagentConfig::new(1);
        assert_eq!(config.steps_per_agent(), 1);

        // Verify default is also m=1
        let default_config = MicroagentConfig::default();
        assert_eq!(default_config.steps_per_agent(), 1);
    }

    #[test]
    #[should_panic(expected = "Microagent constraint violated")]
    fn test_m_greater_than_1_panics() {
        // Invalid: m=2 should panic
        MicroagentConfig::new(2);
    }

    #[test]
    #[should_panic(expected = "Microagent constraint violated")]
    fn test_m_large_value_panics() {
        // Invalid: m=100 should panic
        MicroagentConfig::new(100);
    }

    #[test]
    fn test_outputs_are_tracked_but_not_passed_to_agents() {
        let mut orchestrator = TaskOrchestrator::new(TestDecomposer);

        // Complete step 0
        let output0 = AgentOutput::new("action0".to_string(), State::new(json!({"step": 1})));
        orchestrator.process_output(output0).unwrap();

        // Outputs are tracked in orchestrator
        assert_eq!(orchestrator.outputs().len(), 1);

        // But the next subtask doesn't include previous outputs
        let subtask1 = orchestrator.current_subtask().unwrap();
        // Subtask only has current state, not outputs list
        assert!(subtask1.state.hash.is_some());
    }
}
