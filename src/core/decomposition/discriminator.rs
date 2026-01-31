//! Decomposition Discriminator for MAKER Framework
//!
//! Implements voting-based selection of decomposition proposals. When a task
//! is decomposed, multiple proposal strategies may be generated. The discriminator
//! uses the same first-to-ahead-by-k voting algorithm as execution voting to
//! select the most reliable decomposition.
//!
//! # Architecture
//!
//! ```text
//! Multiple Proposals → DecompositionDiscriminator → Winning Proposal
//!                            ↓
//!                      ProposalMatcher (structural comparison)
//!                            ↓
//!                      VoteRace (first-to-ahead-by-k)
//! ```
//!
//! # Depth-Based K Scaling
//!
//! As decomposition depth increases, the discriminator increases k-margin to
//! maintain reliability. This follows the principle that deeper decompositions
//! require more confidence in each decision.

use super::{CompositionFunction, DecompositionConfig, DecompositionError, DecompositionProposal};
use crate::core::matcher::CandidateMatcher;
use crate::core::voting::{CandidateId, VoteCheckResult, VoteRace};
use crate::events::MakerEvent;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Result of a decomposition voting session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionVoteResult {
    /// The winning proposal
    pub winner: DecompositionProposal,
    /// Total votes cast
    pub total_votes: usize,
    /// k-margin used for this vote
    pub k_margin: usize,
    /// Number of distinct proposals voted on
    pub proposal_count: usize,
    /// Depth at which this vote occurred
    pub depth: usize,
}

/// Structural matcher for decomposition proposals
///
/// Compares proposals based on:
/// 1. Number and IDs of subtasks
/// 2. Composition function type
/// 3. Subtask ordering (when relevant)
///
/// Subtask order matters for Sequential composition but not for Parallel.
#[derive(Debug, Clone)]
pub struct ProposalMatcher {
    /// Whether to consider subtask order in comparison
    order_sensitive: bool,
}

impl ProposalMatcher {
    /// Create a new proposal matcher
    pub fn new() -> Self {
        Self {
            order_sensitive: true,
        }
    }

    /// Create an order-insensitive matcher (for Parallel compositions)
    pub fn order_insensitive() -> Self {
        Self {
            order_sensitive: false,
        }
    }

    /// Serialize a proposal to a canonical string for comparison
    fn proposal_to_canonical(&self, proposal: &DecompositionProposal) -> String {
        let comp_fn = match &proposal.composition_fn {
            CompositionFunction::Sequential => "seq".to_string(),
            CompositionFunction::Parallel { merge_strategy } => {
                format!("par:{:?}", merge_strategy)
            }
            CompositionFunction::Conditional { condition } => format!("cond:{}", condition),
            CompositionFunction::Custom { name, .. } => format!("custom:{}", name),
        };

        let mut subtask_ids: Vec<String> = proposal
            .subtasks
            .iter()
            .map(|s| format!("{}:{}:{}", s.task_id, s.m_value, s.is_leaf))
            .collect();

        if !self.order_sensitive {
            subtask_ids.sort();
        }

        format!("{}|{}", comp_fn, subtask_ids.join(","))
    }
}

impl Default for ProposalMatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl CandidateMatcher for ProposalMatcher {
    fn canonicalize(&self, response: &str) -> String {
        // For proposals, the response is expected to be JSON
        // If it parses as a DecompositionProposal, canonicalize structurally
        // Otherwise, fall back to whitespace normalization
        if let Ok(proposal) = serde_json::from_str::<DecompositionProposal>(response) {
            self.proposal_to_canonical(&proposal)
        } else {
            // Fallback: normalize whitespace
            response.split_whitespace().collect::<Vec<_>>().join(" ")
        }
    }

    fn are_equivalent(&self, a: &str, b: &str) -> bool {
        self.canonicalize(a) == self.canonicalize(b)
    }

    fn similarity_score(&self, a: &str, b: &str) -> f64 {
        // Try to parse both as proposals
        let proposal_a = serde_json::from_str::<DecompositionProposal>(a);
        let proposal_b = serde_json::from_str::<DecompositionProposal>(b);

        match (proposal_a, proposal_b) {
            (Ok(pa), Ok(pb)) => {
                // Compare structurally
                let same_comp_fn = std::mem::discriminant(&pa.composition_fn)
                    == std::mem::discriminant(&pb.composition_fn);
                let same_subtask_count = pa.subtasks.len() == pb.subtasks.len();

                // Count matching subtask IDs
                let ids_a: std::collections::HashSet<_> =
                    pa.subtasks.iter().map(|s| &s.task_id).collect();
                let ids_b: std::collections::HashSet<_> =
                    pb.subtasks.iter().map(|s| &s.task_id).collect();
                let common = ids_a.intersection(&ids_b).count();
                let total = ids_a.len().max(ids_b.len());

                let id_similarity = if total > 0 {
                    common as f64 / total as f64
                } else {
                    1.0
                };

                // Weight the components
                let comp_fn_weight = if same_comp_fn { 0.3 } else { 0.0 };
                let count_weight = if same_subtask_count { 0.2 } else { 0.0 };
                let id_weight = id_similarity * 0.5;

                comp_fn_weight + count_weight + id_weight
            }
            _ => {
                // Fallback: exact match only
                if self.canonicalize(a) == self.canonicalize(b) {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }

    fn matcher_type(&self) -> &str {
        "proposal"
    }
}

/// Discriminator for voting on decomposition proposals
///
/// Wraps a `VoteRace` with proposal-specific matching and depth-aware k-scaling.
pub struct DecompositionDiscriminator {
    /// Configuration for decomposition
    config: DecompositionConfig,
    /// Current depth in the decomposition tree
    depth: usize,
    /// Proposal cache for lookup after voting
    proposals: HashMap<String, DecompositionProposal>,
    /// The proposal matcher
    matcher: ProposalMatcher,
    /// Optional event emitter
    event_emitter: Option<Arc<dyn Fn(MakerEvent) + Send + Sync>>,
}

impl DecompositionDiscriminator {
    /// Create a new discriminator
    pub fn new(config: DecompositionConfig, depth: usize) -> Self {
        Self {
            config,
            depth,
            proposals: HashMap::new(),
            matcher: ProposalMatcher::new(),
            event_emitter: None,
        }
    }

    /// Set an event emitter for observability
    pub fn with_event_emitter(mut self, emitter: Arc<dyn Fn(MakerEvent) + Send + Sync>) -> Self {
        self.event_emitter = Some(emitter);
        self
    }

    /// Get the k-margin for the current depth
    pub fn k_margin(&self) -> usize {
        self.config.k_for_depth(self.depth)
    }

    /// Register a proposal for voting
    ///
    /// Returns the canonical ID used for this proposal in the vote race.
    pub fn register_proposal(&mut self, proposal: DecompositionProposal) -> String {
        let json = serde_json::to_string(&proposal).unwrap_or_default();
        let canonical_id = self.matcher.canonicalize(&json);
        self.proposals.insert(canonical_id.clone(), proposal);
        canonical_id
    }

    /// Vote on registered proposals
    ///
    /// Takes a list of proposal IDs (as returned by `register_proposal`) and
    /// runs a vote race until a winner is determined.
    ///
    /// # Arguments
    ///
    /// * `votes` - Iterator of (proposal_id, weight) pairs. Weight is typically 1.
    ///
    /// # Returns
    ///
    /// The winning proposal after achieving k-margin lead.
    pub fn vote_on_proposals<I>(
        &self,
        votes: I,
    ) -> Result<DecompositionVoteResult, DecompositionError>
    where
        I: IntoIterator<Item = (String, usize)>,
    {
        let k = self.k_margin();
        let race = VoteRace::new(k)
            .map_err(|e| DecompositionError::ValidationError {
                message: format!("Invalid k-margin: {}", e),
            })?
            .with_matcher(Arc::new(self.matcher.clone()));

        // Cast votes
        for (proposal_id, weight) in votes {
            for _ in 0..weight {
                race.cast_vote(CandidateId::new(&proposal_id));
            }
        }

        // Check for winner
        match race.check_winner() {
            VoteCheckResult::Winner {
                candidate,
                votes: _,
                margin: _,
            } => {
                let winner = self
                    .proposals
                    .get(candidate.as_str())
                    .ok_or_else(|| DecompositionError::ValidationError {
                        message: format!("Winner '{}' not found in proposals", candidate.as_str()),
                    })?
                    .clone();

                // Emit accepted event
                if let Some(ref emitter) = self.event_emitter {
                    emitter(MakerEvent::decomposition_accepted(
                        &winner.proposal_id,
                        &winner.source_task_id,
                        race.total_votes(),
                        k,
                    ));
                }

                Ok(DecompositionVoteResult {
                    winner,
                    total_votes: race.total_votes(),
                    k_margin: k,
                    proposal_count: self.proposals.len(),
                    depth: self.depth,
                })
            }
            VoteCheckResult::Ongoing {
                leader,
                current_margin,
                ..
            } => {
                // No winner yet - need more votes
                let leader_id = leader
                    .map(|l| l.as_str().to_string())
                    .unwrap_or_else(|| "none".to_string());

                // Emit rejected event for the session
                if let Some(ref emitter) = self.event_emitter {
                    emitter(MakerEvent::decomposition_rejected(
                        &leader_id,
                        &format!("No winner: margin {} < k {}", current_margin, k),
                    ));
                }

                Err(DecompositionError::ValidationError {
                    message: format!(
                        "No winner achieved k-margin: current margin {}, need {}",
                        current_margin, k
                    ),
                })
            }
        }
    }

    /// Get the current depth
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Get a registered proposal by its canonical ID
    pub fn get_proposal(&self, canonical_id: &str) -> Option<&DecompositionProposal> {
        self.proposals.get(canonical_id)
    }

    /// Get all registered proposals
    pub fn proposals(&self) -> &HashMap<String, DecompositionProposal> {
        &self.proposals
    }
}

impl std::fmt::Debug for DecompositionDiscriminator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecompositionDiscriminator")
            .field("depth", &self.depth)
            .field("k_margin", &self.k_margin())
            .field("proposal_count", &self.proposals.len())
            .finish()
    }
}

/// Convenience function to vote on decomposition proposals
///
/// This is the main entry point for decomposition voting. It:
/// 1. Creates a discriminator at the given depth
/// 2. Registers all proposals
/// 3. Casts votes for each proposal
/// 4. Returns the winner
///
/// # Arguments
///
/// * `proposals` - The proposals to vote on
/// * `config` - Decomposition configuration
/// * `depth` - Current recursion depth
///
/// # Returns
///
/// The winning proposal, or an error if no winner is determined.
pub fn vote_on_decomposition(
    proposals: Vec<DecompositionProposal>,
    config: &DecompositionConfig,
    depth: usize,
) -> Result<DecompositionVoteResult, DecompositionError> {
    if proposals.is_empty() {
        return Err(DecompositionError::EmptyDecomposition {
            task_id: "vote_on_decomposition".to_string(),
        });
    }

    // Validate all proposals first
    for proposal in &proposals {
        proposal.validate()?;
    }

    let mut discriminator = DecompositionDiscriminator::new(config.clone(), depth);

    // Register proposals and collect their canonical IDs
    let mut votes: Vec<(String, usize)> = Vec::new();
    for proposal in proposals {
        let canonical_id = discriminator.register_proposal(proposal);
        // Each proposal gets one initial vote
        votes.push((canonical_id, 1));
    }

    // If all proposals map to the same canonical ID, we have consensus
    let unique_ids: std::collections::HashSet<_> = votes.iter().map(|(id, _)| id).collect();
    if unique_ids.len() == 1 {
        // All proposals are equivalent - the first one wins by default
        let winner_id = votes[0].0.clone();
        let winner = discriminator
            .get_proposal(&winner_id)
            .cloned()
            .ok_or_else(|| DecompositionError::ValidationError {
                message: "Failed to retrieve consensus winner".to_string(),
            })?;

        return Ok(DecompositionVoteResult {
            winner,
            total_votes: votes.len(),
            k_margin: discriminator.k_margin(),
            proposal_count: 1,
            depth,
        });
    }

    // Run the vote
    discriminator.vote_on_proposals(votes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::decomposition::DecompositionSubtask;

    // ==========================================
    // ProposalMatcher Tests
    // ==========================================

    #[test]
    fn test_proposal_matcher_canonicalize_sequential() {
        let matcher = ProposalMatcher::new();
        let proposal = DecompositionProposal::new(
            "p1",
            "source",
            vec![
                DecompositionSubtask::leaf("a", "Task A"),
                DecompositionSubtask::leaf("b", "Task B"),
            ],
            CompositionFunction::Sequential,
        );

        let json = serde_json::to_string(&proposal).unwrap();
        let canonical = matcher.canonicalize(&json);

        assert!(canonical.starts_with("seq|"));
        assert!(canonical.contains("a:1:true"));
        assert!(canonical.contains("b:1:true"));
    }

    #[test]
    fn test_proposal_matcher_canonicalize_parallel() {
        let matcher = ProposalMatcher::new();
        let proposal = DecompositionProposal::new(
            "p1",
            "source",
            vec![DecompositionSubtask::leaf("a", "Task A")],
            CompositionFunction::Parallel {
                merge_strategy: crate::core::decomposition::MergeStrategy::Concatenate,
            },
        );

        let json = serde_json::to_string(&proposal).unwrap();
        let canonical = matcher.canonicalize(&json);

        assert!(canonical.starts_with("par:"));
    }

    #[test]
    fn test_proposal_matcher_order_sensitive() {
        let matcher = ProposalMatcher::new();

        let proposal1 = DecompositionProposal::new(
            "p1",
            "source",
            vec![
                DecompositionSubtask::leaf("a", "A"),
                DecompositionSubtask::leaf("b", "B"),
            ],
            CompositionFunction::Sequential,
        );

        let proposal2 = DecompositionProposal::new(
            "p2",
            "source",
            vec![
                DecompositionSubtask::leaf("b", "B"),
                DecompositionSubtask::leaf("a", "A"),
            ],
            CompositionFunction::Sequential,
        );

        let json1 = serde_json::to_string(&proposal1).unwrap();
        let json2 = serde_json::to_string(&proposal2).unwrap();

        // Order-sensitive matcher should see these as different
        assert_ne!(matcher.canonicalize(&json1), matcher.canonicalize(&json2));
    }

    #[test]
    fn test_proposal_matcher_order_insensitive() {
        let matcher = ProposalMatcher::order_insensitive();

        let proposal1 = DecompositionProposal::new(
            "p1",
            "source",
            vec![
                DecompositionSubtask::leaf("a", "A"),
                DecompositionSubtask::leaf("b", "B"),
            ],
            CompositionFunction::Sequential,
        );

        let proposal2 = DecompositionProposal::new(
            "p2",
            "source",
            vec![
                DecompositionSubtask::leaf("b", "B"),
                DecompositionSubtask::leaf("a", "A"),
            ],
            CompositionFunction::Sequential,
        );

        let json1 = serde_json::to_string(&proposal1).unwrap();
        let json2 = serde_json::to_string(&proposal2).unwrap();

        // Order-insensitive matcher should see these as the same
        assert_eq!(matcher.canonicalize(&json1), matcher.canonicalize(&json2));
    }

    #[test]
    fn test_proposal_matcher_similarity_score() {
        let matcher = ProposalMatcher::new();

        let proposal1 = DecompositionProposal::new(
            "p1",
            "source",
            vec![DecompositionSubtask::leaf("a", "A")],
            CompositionFunction::Sequential,
        );

        let proposal2 = DecompositionProposal::new(
            "p2",
            "source",
            vec![DecompositionSubtask::leaf("a", "A")],
            CompositionFunction::Sequential,
        );

        let json1 = serde_json::to_string(&proposal1).unwrap();
        let json2 = serde_json::to_string(&proposal2).unwrap();

        // Same structure should have high similarity
        let score = matcher.similarity_score(&json1, &json2);
        assert!(score > 0.9, "Expected high similarity, got {}", score);
    }

    #[test]
    fn test_proposal_matcher_type() {
        let matcher = ProposalMatcher::new();
        assert_eq!(matcher.matcher_type(), "proposal");
    }

    // ==========================================
    // DecompositionDiscriminator Tests
    // ==========================================

    #[test]
    fn test_discriminator_new() {
        let config = DecompositionConfig::default();
        let discriminator = DecompositionDiscriminator::new(config, 0);

        assert_eq!(discriminator.depth(), 0);
        assert!(discriminator.k_margin() >= 1);
    }

    #[test]
    fn test_discriminator_k_scales_with_depth() {
        let config = DecompositionConfig {
            depth_scaling: true,
            k_margin: 3,
            ..Default::default()
        };

        let d0 = DecompositionDiscriminator::new(config.clone(), 0);
        let d5 = DecompositionDiscriminator::new(config.clone(), 5);

        assert!(
            d5.k_margin() > d0.k_margin(),
            "k should increase with depth: d0={}, d5={}",
            d0.k_margin(),
            d5.k_margin()
        );
    }

    #[test]
    fn test_discriminator_register_proposal() {
        let config = DecompositionConfig::default();
        let mut discriminator = DecompositionDiscriminator::new(config, 0);

        let proposal = DecompositionProposal::new(
            "p1",
            "source",
            vec![DecompositionSubtask::leaf("a", "Task A")],
            CompositionFunction::Sequential,
        );

        let id = discriminator.register_proposal(proposal.clone());
        assert!(!id.is_empty());

        let retrieved = discriminator.get_proposal(&id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().proposal_id, "p1");
    }

    #[test]
    fn test_discriminator_vote_single_proposal() {
        let config = DecompositionConfig {
            k_margin: 2,
            depth_scaling: false,
            ..Default::default()
        };
        let mut discriminator = DecompositionDiscriminator::new(config, 0);

        let proposal = DecompositionProposal::new(
            "p1",
            "source",
            vec![DecompositionSubtask::leaf("a", "Task A")],
            CompositionFunction::Sequential,
        );

        let id = discriminator.register_proposal(proposal);

        // Vote k times for the single proposal
        let votes = vec![(id.clone(), 2)]; // 2 votes = k_margin
        let result = discriminator.vote_on_proposals(votes);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.winner.proposal_id, "p1");
        assert_eq!(result.k_margin, 2);
    }

    #[test]
    fn test_discriminator_vote_multiple_proposals_winner() {
        let config = DecompositionConfig {
            k_margin: 2,
            depth_scaling: false,
            ..Default::default()
        };
        let mut discriminator = DecompositionDiscriminator::new(config, 0);

        let proposal1 = DecompositionProposal::new(
            "p1",
            "source",
            vec![DecompositionSubtask::leaf("a", "Task A")],
            CompositionFunction::Sequential,
        );
        let proposal2 = DecompositionProposal::new(
            "p2",
            "source",
            vec![DecompositionSubtask::leaf("b", "Task B")],
            CompositionFunction::Sequential,
        );

        let id1 = discriminator.register_proposal(proposal1);
        let id2 = discriminator.register_proposal(proposal2);

        // p1 gets 3 votes, p2 gets 1 -> margin = 2 = k
        let votes = vec![(id1.clone(), 3), (id2.clone(), 1)];
        let result = discriminator.vote_on_proposals(votes);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.winner.proposal_id, "p1");
    }

    #[test]
    fn test_discriminator_vote_no_winner() {
        let config = DecompositionConfig {
            k_margin: 3,
            depth_scaling: false,
            ..Default::default()
        };
        let mut discriminator = DecompositionDiscriminator::new(config, 0);

        let proposal1 = DecompositionProposal::new(
            "p1",
            "source",
            vec![DecompositionSubtask::leaf("a", "Task A")],
            CompositionFunction::Sequential,
        );
        let proposal2 = DecompositionProposal::new(
            "p2",
            "source",
            vec![DecompositionSubtask::leaf("b", "Task B")],
            CompositionFunction::Sequential,
        );

        let id1 = discriminator.register_proposal(proposal1);
        let id2 = discriminator.register_proposal(proposal2);

        // p1 gets 2 votes, p2 gets 1 -> margin = 1 < k = 3
        let votes = vec![(id1.clone(), 2), (id2.clone(), 1)];
        let result = discriminator.vote_on_proposals(votes);

        assert!(result.is_err());
    }

    // ==========================================
    // vote_on_decomposition Tests
    // ==========================================

    #[test]
    fn test_vote_on_decomposition_empty_proposals() {
        let config = DecompositionConfig::default();
        let result = vote_on_decomposition(vec![], &config, 0);

        assert!(result.is_err());
    }

    #[test]
    fn test_vote_on_decomposition_single_proposal() {
        let config = DecompositionConfig::default();
        let proposal = DecompositionProposal::new(
            "p1",
            "source",
            vec![DecompositionSubtask::leaf("a", "Task A")],
            CompositionFunction::Sequential,
        );

        let result = vote_on_decomposition(vec![proposal], &config, 0);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.winner.proposal_id, "p1");
        assert_eq!(result.proposal_count, 1);
    }

    #[test]
    fn test_vote_on_decomposition_equivalent_proposals() {
        let config = DecompositionConfig::default();

        // Two proposals with same structure but different IDs
        let proposal1 = DecompositionProposal::new(
            "p1",
            "source",
            vec![DecompositionSubtask::leaf("a", "Task A")],
            CompositionFunction::Sequential,
        );
        let proposal2 = DecompositionProposal::new(
            "p2",
            "source",
            vec![DecompositionSubtask::leaf("a", "Task A")],
            CompositionFunction::Sequential,
        );

        // These are structurally equivalent, should reach consensus
        let result = vote_on_decomposition(vec![proposal1, proposal2], &config, 0);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.proposal_count, 1); // Merged as equivalent
    }

    #[test]
    fn test_vote_on_decomposition_validates_proposals() {
        let config = DecompositionConfig::default();

        // Proposal with invalid leaf (m != 1)
        let mut invalid_subtask = DecompositionSubtask::leaf("a", "Task A");
        invalid_subtask.m_value = 5; // Invalid for leaf

        let invalid_proposal = DecompositionProposal::new(
            "p1",
            "source",
            vec![invalid_subtask],
            CompositionFunction::Sequential,
        );

        let result = vote_on_decomposition(vec![invalid_proposal], &config, 0);

        assert!(result.is_err());
    }

    // ==========================================
    // Integration Test: 3 Proposals → Single Winner
    // ==========================================

    #[test]
    fn test_three_proposals_single_winner() {
        // Create 3 distinct proposals
        let proposal1 = DecompositionProposal::new(
            "strategy-a",
            "complex-task",
            vec![
                DecompositionSubtask::leaf("a1", "Step A1"),
                DecompositionSubtask::leaf("a2", "Step A2"),
            ],
            CompositionFunction::Sequential,
        );

        let proposal2 = DecompositionProposal::new(
            "strategy-b",
            "complex-task",
            vec![
                DecompositionSubtask::leaf("b1", "Step B1"),
                DecompositionSubtask::leaf("b2", "Step B2"),
                DecompositionSubtask::leaf("b3", "Step B3"),
            ],
            CompositionFunction::Sequential,
        );

        let proposal3 = DecompositionProposal::new(
            "strategy-c",
            "complex-task",
            vec![DecompositionSubtask::leaf("c1", "Single step C")],
            CompositionFunction::Parallel {
                merge_strategy: crate::core::decomposition::MergeStrategy::Concatenate,
            },
        );

        let config = DecompositionConfig {
            k_margin: 2,
            depth_scaling: false,
            ..Default::default()
        };

        let mut discriminator = DecompositionDiscriminator::new(config, 0);

        let id1 = discriminator.register_proposal(proposal1);
        let id2 = discriminator.register_proposal(proposal2);
        let id3 = discriminator.register_proposal(proposal3);

        // Simulate voting where strategy-a wins
        // a: 5 votes, b: 2 votes, c: 1 vote -> a wins by margin 3 >= k=2
        let votes = vec![(id1.clone(), 5), (id2.clone(), 2), (id3.clone(), 1)];

        let result = discriminator.vote_on_proposals(votes);

        assert!(result.is_ok(), "Expected winner, got {:?}", result);
        let result = result.unwrap();
        assert_eq!(result.winner.proposal_id, "strategy-a");
        assert_eq!(result.total_votes, 8);
        assert_eq!(result.proposal_count, 3);
    }
}
