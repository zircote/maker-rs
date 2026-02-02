//! Vote Race State Tracking for MAKER Framework
//!
//! Implements the first-to-ahead-by-k voting protocol using Gambler's Ruin
//! dynamics. Tracks vote counts for each candidate and detects when a winner
//! has achieved the required k-margin lead.
//!
//! # Protocol
//!
//! The voting race continues until one candidate leads the nearest competitor
//! by exactly k votes. This is mathematically optimal (SPRT-based) for the
//! binary hypothesis testing underlying MAKER's error correction.
//!
//! # Thread Safety
//!
//! `VoteRace` is designed for concurrent vote casting using interior mutability.
//! Multiple sampling tasks can cast votes simultaneously.
//!
//! # Panic Behavior
//!
//! Methods that acquire the internal mutex will panic if the mutex is poisoned
//! (i.e., another thread panicked while holding the lock). This is intentional:
//! mutex poisoning indicates a serious bug, and continuing with potentially
//! corrupted vote state could lead to incorrect consensus decisions.

use crate::core::matcher::{default_matcher, CandidateMatcher};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Unique identifier for a voting candidate (typically a hash of the response content)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CandidateId(pub String);

impl CandidateId {
    /// Create a new candidate ID from a string
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the inner string value
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<String> for CandidateId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for CandidateId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Result of checking for a winner in the vote race
#[derive(Debug, Clone, PartialEq)]
pub enum VoteCheckResult {
    /// A winner has been determined with the required k-margin
    Winner {
        /// The winning candidate identifier
        candidate: CandidateId,
        /// Total votes received by the winner
        votes: usize,
        /// Margin of victory (>= k_margin)
        margin: usize,
    },
    /// No winner yet - race continues
    Ongoing {
        /// Current leading candidate, if any
        leader: Option<CandidateId>,
        /// Vote count of the leader
        leader_votes: usize,
        /// Vote count of the runner-up
        runner_up_votes: usize,
        /// Current margin between leader and runner-up
        current_margin: usize,
    },
}

/// Error type for voting operations
#[derive(Debug, Clone, PartialEq)]
pub enum VoteError {
    /// k_margin must be at least 1
    InvalidKMargin {
        /// The invalid k value provided
        k: usize,
    },
}

impl std::fmt::Display for VoteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VoteError::InvalidKMargin { k } => {
                write!(f, "k_margin must be at least 1, got {}", k)
            }
        }
    }
}

impl std::error::Error for VoteError {}

/// Callback for vote events (used for observability integration)
pub type VoteEventCallback = Box<dyn Fn(VoteEvent) + Send + Sync>;

/// Events emitted during voting
#[derive(Debug, Clone)]
pub enum VoteEvent {
    /// A vote was cast for a candidate
    VoteCast {
        /// The candidate that received the vote
        candidate_id: String,
        /// Updated vote count for this candidate
        vote_count: usize,
        /// Current margin between this candidate and nearest rival
        current_margin: i32,
    },
    /// A winner was declared
    VoteDecided {
        /// The winning candidate identifier
        winner_id: String,
        /// Total votes cast across all candidates
        total_votes: usize,
        /// The k-margin threshold that was met
        k_margin: usize,
    },
}

/// Thread-safe vote race tracker
///
/// Tracks vote counts for multiple candidates and determines when a winner
/// has achieved the required k-margin lead over all competitors.
///
/// When a `CandidateMatcher` is provided, responses are canonicalized before
/// being used as candidate keys. This allows semantically equivalent responses
/// to be grouped together during voting.
pub struct VoteRace {
    /// Vote counts per candidate
    votes: Arc<Mutex<HashMap<CandidateId, usize>>>,
    /// Required margin for winning
    k_margin: usize,
    /// Optional event callback for observability
    event_callback: Option<Arc<VoteEventCallback>>,
    /// Candidate matcher for response grouping
    matcher: Arc<dyn CandidateMatcher>,
}

impl VoteRace {
    /// Create a new vote race with the specified k-margin
    ///
    /// # Arguments
    ///
    /// * `k_margin` - Required vote lead to declare a winner (must be >= 1)
    ///
    /// # Errors
    ///
    /// Returns `VoteError::InvalidKMargin` if k_margin is 0
    pub fn new(k_margin: usize) -> Result<Self, VoteError> {
        if k_margin == 0 {
            return Err(VoteError::InvalidKMargin { k: k_margin });
        }
        Ok(Self {
            votes: Arc::new(Mutex::new(HashMap::new())),
            k_margin,
            event_callback: None,
            matcher: default_matcher(),
        })
    }

    /// Set a custom candidate matcher for response grouping.
    ///
    /// When set, responses are canonicalized via the matcher before being
    /// used as candidate keys, allowing semantically equivalent responses
    /// to share votes.
    pub fn with_matcher(mut self, matcher: Arc<dyn CandidateMatcher>) -> Self {
        self.matcher = matcher;
        self
    }

    /// Set an event callback for observability
    pub fn with_event_callback(mut self, callback: VoteEventCallback) -> Self {
        self.event_callback = Some(Arc::new(callback));
        self
    }

    /// Cast a vote for a candidate
    ///
    /// The candidate ID is canonicalized via the configured matcher before
    /// being used as the vote key. This means two responses that the matcher
    /// considers equivalent will accumulate votes under the same candidate.
    ///
    /// This method is thread-safe and can be called concurrently from multiple tasks.
    ///
    /// # Arguments
    ///
    /// * `candidate` - The candidate ID to vote for
    ///
    /// # Returns
    ///
    /// The new vote count for this candidate (after canonicalization)
    pub fn cast_vote(&self, candidate: CandidateId) -> usize {
        let canonical = CandidateId::new(self.matcher.canonicalize(candidate.as_str()));
        let mut votes = self.votes.lock().expect("vote state mutex poisoned");
        let count = votes.entry(canonical.clone()).or_insert(0);
        *count += 1;
        let new_count = *count;

        // Calculate current margin for event
        if let Some(ref callback) = self.event_callback {
            let margin = self.calculate_margin_internal(&votes, &canonical);
            callback(VoteEvent::VoteCast {
                candidate_id: canonical.0,
                vote_count: new_count,
                current_margin: margin,
            });
        }

        new_count
    }

    /// Check if there is a winner
    ///
    /// A winner is declared when one candidate leads all others by at least k_margin votes.
    ///
    /// # Returns
    ///
    /// - `VoteCheckResult::Winner` if a candidate has achieved the required margin
    /// - `VoteCheckResult::Ongoing` if no winner yet
    pub fn check_winner(&self) -> VoteCheckResult {
        let votes = self.votes.lock().expect("vote state mutex poisoned");

        if votes.is_empty() {
            return VoteCheckResult::Ongoing {
                leader: None,
                leader_votes: 0,
                runner_up_votes: 0,
                current_margin: 0,
            };
        }

        // Find leader and runner-up
        let mut sorted: Vec<_> = votes.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1)); // Sort by votes descending

        let (leader_id, &leader_votes) = sorted[0];
        let runner_up_votes = if sorted.len() > 1 { *sorted[1].1 } else { 0 };

        let margin = leader_votes.saturating_sub(runner_up_votes);

        if margin >= self.k_margin {
            // Emit decided event
            let total_votes: usize = votes.values().sum();
            if let Some(ref callback) = self.event_callback {
                callback(VoteEvent::VoteDecided {
                    winner_id: leader_id.0.clone(),
                    total_votes,
                    k_margin: self.k_margin,
                });
            }

            VoteCheckResult::Winner {
                candidate: leader_id.clone(),
                votes: leader_votes,
                margin,
            }
        } else {
            VoteCheckResult::Ongoing {
                leader: Some(leader_id.clone()),
                leader_votes,
                runner_up_votes,
                current_margin: margin,
            }
        }
    }

    /// Get the current vote counts (snapshot)
    pub fn get_votes(&self) -> HashMap<CandidateId, usize> {
        self.votes
            .lock()
            .expect("vote state mutex poisoned")
            .clone()
    }

    /// Get the total number of votes cast
    pub fn total_votes(&self) -> usize {
        self.votes
            .lock()
            .expect("vote state mutex poisoned")
            .values()
            .sum()
    }

    /// Get the configured k-margin
    pub fn k_margin(&self) -> usize {
        self.k_margin
    }

    /// Internal helper to calculate margin for a candidate
    fn calculate_margin_internal(
        &self,
        votes: &HashMap<CandidateId, usize>,
        candidate: &CandidateId,
    ) -> i32 {
        let candidate_votes = *votes.get(candidate).unwrap_or(&0) as i32;
        let max_other = votes
            .iter()
            .filter(|(id, _)| *id != candidate)
            .map(|(_, &v)| v as i32)
            .max()
            .unwrap_or(0);
        candidate_votes - max_other
    }
}

impl Clone for VoteRace {
    fn clone(&self) -> Self {
        Self {
            votes: Arc::new(Mutex::new(
                self.votes
                    .lock()
                    .expect("vote state mutex poisoned")
                    .clone(),
            )),
            k_margin: self.k_margin,
            event_callback: self.event_callback.clone(),
            matcher: self.matcher.clone(),
        }
    }
}

impl std::fmt::Debug for VoteRace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let votes = self.votes.lock().expect("vote state mutex poisoned");
        f.debug_struct("VoteRace")
            .field("votes", &*votes)
            .field("k_margin", &self.k_margin)
            .field("has_callback", &self.event_callback.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;

    // ==========================================
    // Construction Tests
    // ==========================================

    #[test]
    fn test_new_with_valid_k_margin() {
        let race = VoteRace::new(3);
        assert!(race.is_ok());
        assert_eq!(race.unwrap().k_margin(), 3);
    }

    #[test]
    fn test_rejects_zero_k_margin() {
        let race = VoteRace::new(0);
        assert!(race.is_err());
        assert_eq!(race.unwrap_err(), VoteError::InvalidKMargin { k: 0 });
    }

    // ==========================================
    // Basic Voting Tests
    // ==========================================

    #[test]
    fn test_cast_vote_returns_count() {
        let race = VoteRace::new(3).unwrap();
        let candidate = CandidateId::new("A");

        assert_eq!(race.cast_vote(candidate.clone()), 1);
        assert_eq!(race.cast_vote(candidate.clone()), 2);
        assert_eq!(race.cast_vote(candidate.clone()), 3);
    }

    #[test]
    fn test_multiple_candidates() {
        let race = VoteRace::new(3).unwrap();

        race.cast_vote(CandidateId::new("A"));
        race.cast_vote(CandidateId::new("B"));
        race.cast_vote(CandidateId::new("A"));

        let votes = race.get_votes();
        assert_eq!(votes.get(&CandidateId::new("A")), Some(&2));
        assert_eq!(votes.get(&CandidateId::new("B")), Some(&1));
    }

    #[test]
    fn test_total_votes() {
        let race = VoteRace::new(3).unwrap();

        race.cast_vote(CandidateId::new("A"));
        race.cast_vote(CandidateId::new("B"));
        race.cast_vote(CandidateId::new("A"));

        assert_eq!(race.total_votes(), 3);
    }

    // ==========================================
    // Winner Detection Tests
    // ==========================================

    #[test]
    fn test_no_winner_when_empty() {
        let race = VoteRace::new(3).unwrap();
        match race.check_winner() {
            VoteCheckResult::Ongoing { leader, .. } => assert!(leader.is_none()),
            _ => panic!("Expected Ongoing with no leader"),
        }
    }

    #[test]
    fn test_no_winner_when_margin_not_reached() {
        let race = VoteRace::new(3).unwrap();

        // A: 2, B: 1 -> margin is 1, need 3
        race.cast_vote(CandidateId::new("A"));
        race.cast_vote(CandidateId::new("A"));
        race.cast_vote(CandidateId::new("B"));

        match race.check_winner() {
            VoteCheckResult::Ongoing {
                leader,
                leader_votes,
                runner_up_votes,
                current_margin,
            } => {
                assert_eq!(leader, Some(CandidateId::new("A")));
                assert_eq!(leader_votes, 2);
                assert_eq!(runner_up_votes, 1);
                assert_eq!(current_margin, 1);
            }
            _ => panic!("Expected Ongoing"),
        }
    }

    #[test]
    fn test_winner_when_margin_exactly_reached() {
        let race = VoteRace::new(3).unwrap();

        // A: 4, B: 1 -> margin is 3, exactly k_margin
        race.cast_vote(CandidateId::new("A"));
        race.cast_vote(CandidateId::new("A"));
        race.cast_vote(CandidateId::new("A"));
        race.cast_vote(CandidateId::new("A"));
        race.cast_vote(CandidateId::new("B"));

        match race.check_winner() {
            VoteCheckResult::Winner {
                candidate,
                votes,
                margin,
            } => {
                assert_eq!(candidate, CandidateId::new("A"));
                assert_eq!(votes, 4);
                assert_eq!(margin, 3);
            }
            _ => panic!("Expected Winner"),
        }
    }

    #[test]
    fn test_winner_when_margin_exceeded() {
        let race = VoteRace::new(2).unwrap();

        // A: 5, B: 1 -> margin is 4, exceeds k_margin of 2
        for _ in 0..5 {
            race.cast_vote(CandidateId::new("A"));
        }
        race.cast_vote(CandidateId::new("B"));

        match race.check_winner() {
            VoteCheckResult::Winner { margin, .. } => {
                assert!(margin >= 2, "Margin {} should be >= k_margin 2", margin);
            }
            _ => panic!("Expected Winner"),
        }
    }

    #[test]
    fn test_single_candidate_needs_k_margin_votes() {
        let race = VoteRace::new(3).unwrap();

        // Single candidate with 2 votes -> margin is 2 (vs 0), need 3
        race.cast_vote(CandidateId::new("A"));
        race.cast_vote(CandidateId::new("A"));

        match race.check_winner() {
            VoteCheckResult::Ongoing { current_margin, .. } => {
                assert_eq!(current_margin, 2);
            }
            _ => panic!("Expected Ongoing"),
        }

        // Add one more vote -> margin is 3
        race.cast_vote(CandidateId::new("A"));

        match race.check_winner() {
            VoteCheckResult::Winner { margin, .. } => {
                assert_eq!(margin, 3);
            }
            _ => panic!("Expected Winner"),
        }
    }

    #[test]
    fn test_no_false_positive_winner() {
        let race = VoteRace::new(5).unwrap();

        // Simulate close race: A and B alternate
        for _ in 0..10 {
            race.cast_vote(CandidateId::new("A"));
            race.cast_vote(CandidateId::new("B"));
        }

        // A: 10, B: 10 -> margin is 0
        match race.check_winner() {
            VoteCheckResult::Ongoing { current_margin, .. } => {
                assert_eq!(current_margin, 0);
            }
            _ => panic!("Expected Ongoing - tied race should not have winner"),
        }
    }

    // ==========================================
    // Thread Safety Tests
    // ==========================================

    #[test]
    fn test_concurrent_voting() {
        let race = Arc::new(VoteRace::new(100).unwrap());

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let r = Arc::clone(&race);
                thread::spawn(move || {
                    for _ in 0..100 {
                        r.cast_vote(CandidateId::new(format!("candidate_{}", i % 3)));
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Total should be 10 threads * 100 votes = 1000
        assert_eq!(race.total_votes(), 1000);
    }

    // ==========================================
    // Event Callback Tests
    // ==========================================

    #[test]
    fn test_event_callback_on_vote() {
        let vote_count = Arc::new(AtomicUsize::new(0));
        let vote_count_clone = vote_count.clone();

        let race = VoteRace::new(3)
            .unwrap()
            .with_event_callback(Box::new(move |event| {
                if matches!(event, VoteEvent::VoteCast { .. }) {
                    vote_count_clone.fetch_add(1, Ordering::SeqCst);
                }
            }));

        race.cast_vote(CandidateId::new("A"));
        race.cast_vote(CandidateId::new("B"));
        race.cast_vote(CandidateId::new("A"));

        assert_eq!(vote_count.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_event_callback_on_winner() {
        let decided_count = Arc::new(AtomicUsize::new(0));
        let decided_clone = decided_count.clone();

        let race = VoteRace::new(2)
            .unwrap()
            .with_event_callback(Box::new(move |event| {
                if matches!(event, VoteEvent::VoteDecided { .. }) {
                    decided_clone.fetch_add(1, Ordering::SeqCst);
                }
            }));

        race.cast_vote(CandidateId::new("A"));
        race.cast_vote(CandidateId::new("A"));

        // Check winner triggers event
        let _ = race.check_winner();

        assert_eq!(decided_count.load(Ordering::SeqCst), 1);
    }

    // ==========================================
    // CandidateId Tests
    // ==========================================

    #[test]
    fn test_candidate_id_from_string() {
        let id: CandidateId = "test".into();
        assert_eq!(id.as_str(), "test");
    }

    #[test]
    fn test_candidate_id_equality() {
        let id1 = CandidateId::new("test");
        let id2 = CandidateId::new("test");
        let id3 = CandidateId::new("other");

        assert_eq!(id1, id2);
        assert_ne!(id1, id3);
    }
}
