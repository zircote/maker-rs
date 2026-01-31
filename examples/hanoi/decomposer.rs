//! Task Decomposer for Towers of Hanoi
//!
//! Implements the TaskDecomposer trait to generate subtasks for each move.

use crate::state::{compute_solution, HanoiMove, HanoiState, ROD_NAMES};
use maker::core::orchestration::{AgentOutput, State, Subtask, TaskDecomposer};

/// Task decomposer for Towers of Hanoi
///
/// Generates 2^n - 1 subtasks, one for each optimal move.
/// Provides ground truth validation for each step.
pub struct HanoiDecomposer {
    /// Number of disks
    n_disks: u8,
    /// Precomputed optimal solution
    solution: Vec<HanoiMove>,
    /// Precomputed states after each move
    states: Vec<HanoiState>,
}

impl HanoiDecomposer {
    /// Create a new decomposer for n-disk Hanoi
    pub fn new(n_disks: u8) -> Self {
        assert!(n_disks > 0 && n_disks <= 20, "n_disks must be 1-20");

        let solution = compute_solution(n_disks);

        // Precompute all intermediate states
        let mut states = Vec::with_capacity(solution.len() + 1);
        let mut current = HanoiState::new(n_disks);
        states.push(current.clone());

        for mv in &solution {
            current = current.apply_move(mv.from, mv.to);
            states.push(current.clone());
        }

        Self {
            n_disks,
            solution,
            states,
        }
    }

    /// Get the expected move for a given step
    pub fn expected_move(&self, step: usize) -> Option<&HanoiMove> {
        self.solution.get(step)
    }

    /// Get the expected state after a given step
    pub fn expected_state_after(&self, step: usize) -> Option<&HanoiState> {
        self.states.get(step + 1)
    }

    /// Get the state before a given step
    pub fn state_before(&self, step: usize) -> Option<&HanoiState> {
        self.states.get(step)
    }

    /// Get total number of disks
    pub fn n_disks(&self) -> u8 {
        self.n_disks
    }

    /// Get all solution moves
    pub fn solution(&self) -> &[HanoiMove] {
        &self.solution
    }
}

impl TaskDecomposer for HanoiDecomposer {
    fn decompose(&self) -> Vec<Subtask> {
        self.solution
            .iter()
            .enumerate()
            .map(|(step_id, _mv)| {
                let state = &self.states[step_id];
                let prompt = format!(
                    "You are solving Towers of Hanoi with {} disks.\n\
                     Current state:\n\
                     - Rod A: {:?}\n\
                     - Rod B: {:?}\n\
                     - Rod C: {:?}\n\
                     \n\
                     Goal: Move all disks to Rod C following the rules:\n\
                     1. Only one disk can be moved at a time\n\
                     2. A larger disk cannot be placed on a smaller disk\n\
                     \n\
                     What is the next optimal move? Respond with JSON:\n\
                     {{\"move\": \"Move disk X from Y to Z\", \"next_state\": {{...}}}}",
                    self.n_disks, state.rods[0], state.rods[1], state.rods[2],
                );

                Subtask::new(step_id, prompt, state.to_state())
            })
            .collect()
    }

    fn total_steps(&self) -> usize {
        self.solution.len()
    }

    fn initial_state(&self) -> State {
        self.states[0].to_state()
    }

    fn validate_output(&self, step_id: usize, output: &AgentOutput) -> Result<(), String> {
        let expected_move = self
            .expected_move(step_id)
            .ok_or_else(|| format!("Invalid step_id: {}", step_id))?;

        let expected_state = self
            .expected_state_after(step_id)
            .ok_or_else(|| format!("No expected state for step: {}", step_id))?;

        // Parse the move from the output
        let move_str = &output.move_action;
        let expected_move_str = expected_move.to_string();

        // Validate the move matches expected
        if !move_str.contains(&format!("disk {}", expected_move.disk))
            || !move_str.contains(ROD_NAMES[expected_move.from])
            || !move_str.contains(ROD_NAMES[expected_move.to])
        {
            return Err(format!(
                "Step {}: Expected '{}', got '{}'",
                step_id, expected_move_str, move_str
            ));
        }

        // Validate the next_state matches expected
        let actual_state = HanoiState::from_state(&output.next_state)
            .map_err(|e| format!("Failed to parse next_state: {}", e))?;

        if actual_state.rods != expected_state.rods {
            return Err(format!(
                "Step {}: State mismatch. Expected rods {:?}, got {:?}",
                step_id, expected_state.rods, actual_state.rods
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decomposer_3_disk_total_steps() {
        let decomposer = HanoiDecomposer::new(3);
        assert_eq!(decomposer.total_steps(), 7);
    }

    #[test]
    fn test_decomposer_10_disk_total_steps() {
        let decomposer = HanoiDecomposer::new(10);
        assert_eq!(decomposer.total_steps(), 1023);
    }

    #[test]
    fn test_decomposer_generates_subtasks() {
        let decomposer = HanoiDecomposer::new(3);
        let subtasks = decomposer.decompose();

        assert_eq!(subtasks.len(), 7);

        for (i, subtask) in subtasks.iter().enumerate() {
            assert_eq!(subtask.step_id, i);
        }
    }

    #[test]
    fn test_decomposer_initial_state() {
        let decomposer = HanoiDecomposer::new(3);
        let state = decomposer.initial_state();
        let hanoi = HanoiState::from_state(&state).unwrap();

        assert_eq!(hanoi.rods[0], vec![3, 2, 1]);
    }

    #[test]
    fn test_decomposer_expected_move() {
        let decomposer = HanoiDecomposer::new(3);

        // First move of 3-disk: disk 1 from A to C
        let first_move = decomposer.expected_move(0).unwrap();
        assert_eq!(first_move.disk, 1);
        assert_eq!(first_move.from, 0);
        assert_eq!(first_move.to, 2);
    }

    #[test]
    fn test_decomposer_validate_output_correct() {
        let decomposer = HanoiDecomposer::new(3);

        let expected_state = decomposer.expected_state_after(0).unwrap();
        let output = AgentOutput::new(
            "Move disk 1 from A to C".to_string(),
            expected_state.to_state(),
        );

        let result = decomposer.validate_output(0, &output);
        assert!(result.is_ok());
    }

    #[test]
    fn test_decomposer_validate_output_wrong_move() {
        let decomposer = HanoiDecomposer::new(3);

        let expected_state = decomposer.expected_state_after(0).unwrap();
        let output = AgentOutput::new(
            "Move disk 1 from A to B".to_string(), // Wrong: should be A to C
            expected_state.to_state(),
        );

        let result = decomposer.validate_output(0, &output);
        assert!(result.is_err());
    }

    #[test]
    fn test_decomposer_validate_output_wrong_state() {
        let decomposer = HanoiDecomposer::new(3);

        let wrong_state = HanoiState::new(3).to_state();
        let output = AgentOutput::new("Move disk 1 from A to C".to_string(), wrong_state);

        let result = decomposer.validate_output(0, &output);
        assert!(result.is_err());
    }

    #[test]
    #[should_panic(expected = "n_disks must be 1-20")]
    fn test_decomposer_rejects_zero_disks() {
        HanoiDecomposer::new(0);
    }

    #[test]
    #[should_panic(expected = "n_disks must be 1-20")]
    fn test_decomposer_rejects_too_many_disks() {
        HanoiDecomposer::new(21);
    }
}
