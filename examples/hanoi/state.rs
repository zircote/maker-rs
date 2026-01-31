//! Towers of Hanoi State and Move Representations
//!
//! Implements the state tracking and move validation for the puzzle.

use maker::core::orchestration::State;
use serde::{Deserialize, Serialize};

/// Rod names for human-readable output
pub const ROD_NAMES: [char; 3] = ['A', 'B', 'C'];

/// State of the Towers of Hanoi puzzle
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HanoiState {
    /// Three rods, each containing a stack of disks
    /// Disk values: 1 = smallest, n = largest
    /// Stack order: bottom to top (last element is top disk)
    pub rods: [Vec<u8>; 3],
    /// Current step number
    pub step: usize,
    /// Total number of disks
    pub n_disks: u8,
}

impl HanoiState {
    /// Create initial state with all disks on rod A (index 0)
    pub fn new(n_disks: u8) -> Self {
        // Largest at bottom: [n, n-1, ..., 1]
        let rod_a: Vec<u8> = (1..=n_disks).rev().collect();

        Self {
            rods: [rod_a, Vec::new(), Vec::new()],
            step: 0,
            n_disks,
        }
    }

    /// Create final/goal state with all disks on rod C (index 2)
    pub fn goal(n_disks: u8) -> Self {
        let rod_c: Vec<u8> = (1..=n_disks).rev().collect();
        Self {
            rods: [Vec::new(), Vec::new(), rod_c],
            step: (1 << n_disks) - 1, // 2^n - 1
            n_disks,
        }
    }

    /// Check if a move is legal
    ///
    /// Rules:
    /// 1. Source rod must have a disk
    /// 2. Cannot place larger disk on smaller disk
    pub fn is_legal_move(&self, from: usize, to: usize) -> bool {
        if from >= 3 || to >= 3 || from == to {
            return false;
        }

        let source = &self.rods[from];
        if source.is_empty() {
            return false;
        }

        let moving_disk = *source.last().unwrap();
        let target = &self.rods[to];

        // Target empty or moving disk is smaller than top of target
        target.is_empty() || moving_disk < *target.last().unwrap()
    }

    /// Get the disk being moved (top of source rod)
    pub fn top_disk(&self, rod: usize) -> Option<u8> {
        self.rods.get(rod).and_then(|r| r.last().copied())
    }

    /// Apply a move, returning the new state
    ///
    /// # Panics
    ///
    /// Panics if the move is illegal
    pub fn apply_move(&self, from: usize, to: usize) -> Self {
        assert!(
            self.is_legal_move(from, to),
            "Illegal move: {} -> {}",
            ROD_NAMES[from],
            ROD_NAMES[to]
        );

        let mut new_state = self.clone();
        let disk = new_state.rods[from].pop().unwrap();
        new_state.rods[to].push(disk);
        new_state.step += 1;
        new_state
    }

    /// Check if puzzle is solved (all disks on rod C)
    pub fn is_solved(&self) -> bool {
        self.rods[0].is_empty()
            && self.rods[1].is_empty()
            && self.rods[2].len() == self.n_disks as usize
    }

    /// Format a move as human-readable string
    pub fn format_move(from: usize, to: usize, disk: u8) -> String {
        format!(
            "Move disk {} from {} to {}",
            disk, ROD_NAMES[from], ROD_NAMES[to]
        )
    }

    /// Convert to JSON State for orchestration
    pub fn to_state(&self) -> State {
        State::new(serde_json::to_value(self).unwrap())
    }

    /// Parse from JSON State
    pub fn from_state(state: &State) -> Result<Self, serde_json::Error> {
        serde_json::from_value(state.data.clone())
    }
}

/// A single move in the solution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HanoiMove {
    /// Source rod (0-2)
    pub from: usize,
    /// Destination rod (0-2)
    pub to: usize,
    /// Disk being moved
    pub disk: u8,
}

impl std::fmt::Display for HanoiMove {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            HanoiState::format_move(self.from, self.to, self.disk)
        )
    }
}

/// Compute the optimal solution for Towers of Hanoi
///
/// Uses the classic recursive algorithm to generate the minimum
/// sequence of moves to transfer n disks from rod A to rod C.
pub fn compute_solution(n_disks: u8) -> Vec<HanoiMove> {
    let mut moves = Vec::with_capacity((1 << n_disks) - 1);
    solve_recursive(n_disks, 0, 2, 1, &mut moves);
    moves
}

/// Recursive helper for computing solution
fn solve_recursive(n: u8, from: usize, to: usize, aux: usize, moves: &mut Vec<HanoiMove>) {
    if n == 0 {
        return;
    }

    // Move n-1 disks from source to auxiliary
    solve_recursive(n - 1, from, aux, to, moves);

    // Move largest disk from source to destination
    moves.push(HanoiMove { from, to, disk: n });

    // Move n-1 disks from auxiliary to destination
    solve_recursive(n - 1, aux, to, from, moves);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state_3_disks() {
        let state = HanoiState::new(3);
        assert_eq!(state.rods[0], vec![3, 2, 1]);
        assert!(state.rods[1].is_empty());
        assert!(state.rods[2].is_empty());
        assert_eq!(state.n_disks, 3);
        assert_eq!(state.step, 0);
    }

    #[test]
    fn test_goal_state_3_disks() {
        let state = HanoiState::goal(3);
        assert!(state.rods[0].is_empty());
        assert!(state.rods[1].is_empty());
        assert_eq!(state.rods[2], vec![3, 2, 1]);
        assert_eq!(state.step, 7);
    }

    #[test]
    fn test_is_legal_move_valid() {
        let state = HanoiState::new(3);
        assert!(state.is_legal_move(0, 1));
        assert!(state.is_legal_move(0, 2));
    }

    #[test]
    fn test_is_legal_move_empty_source() {
        let state = HanoiState::new(3);
        assert!(!state.is_legal_move(1, 0));
    }

    #[test]
    fn test_is_legal_move_larger_on_smaller() {
        let mut state = HanoiState::new(3);
        state = state.apply_move(0, 1);
        assert!(!state.is_legal_move(0, 1));
    }

    #[test]
    fn test_apply_move() {
        let state = HanoiState::new(3);
        let new_state = state.apply_move(0, 2);

        assert_eq!(new_state.rods[0], vec![3, 2]);
        assert!(new_state.rods[1].is_empty());
        assert_eq!(new_state.rods[2], vec![1]);
        assert_eq!(new_state.step, 1);
    }

    #[test]
    fn test_is_solved() {
        let goal = HanoiState::goal(3);
        assert!(goal.is_solved());

        let initial = HanoiState::new(3);
        assert!(!initial.is_solved());
    }

    #[test]
    fn test_compute_solution_1_disk() {
        let solution = compute_solution(1);
        assert_eq!(solution.len(), 1);
        assert_eq!(
            solution[0],
            HanoiMove {
                from: 0,
                to: 2,
                disk: 1
            }
        );
    }

    #[test]
    fn test_compute_solution_3_disks() {
        let solution = compute_solution(3);
        assert_eq!(solution.len(), 7);
    }

    #[test]
    fn test_compute_solution_10_disks() {
        let solution = compute_solution(10);
        assert_eq!(solution.len(), 1023);
    }

    #[test]
    fn test_solution_leads_to_goal() {
        let solution = compute_solution(3);
        let mut state = HanoiState::new(3);

        for mv in &solution {
            assert!(state.is_legal_move(mv.from, mv.to));
            state = state.apply_move(mv.from, mv.to);
        }

        assert!(state.is_solved());
    }
}
