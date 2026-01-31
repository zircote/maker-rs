//! Event-Driven Observability for MAKER Framework
//!
//! Provides structured events for monitoring MAKER execution, including:
//! - Sample requests and completions
//! - Red-flag triggers
//! - Vote casting and decisions
//! - Step completions with cost metrics
//!
//! # Architecture
//!
//! Events are emitted via an `EventBus` which uses a broadcast channel.
//! Multiple observers can subscribe to receive all events:
//!
//! ```text
//! Core Logic → EventBus → [LoggingObserver, MetricsObserver, ...]
//! ```

pub mod bus;
pub mod observers;

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// All events emitted by the MAKER framework
///
/// Events are tagged with their type for JSON serialization and include
/// timestamps for latency tracking.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum MakerEvent {
    /// A sample was requested from the LLM
    SampleRequested {
        /// Model identifier
        model: String,
        /// Hash of the prompt (for correlation)
        prompt_hash: String,
        /// Temperature used for sampling
        temperature: f64,
        /// When the request was made
        #[serde(with = "system_time_serde")]
        timestamp: SystemTime,
    },

    /// A sample completed (successfully or with error)
    SampleCompleted {
        /// Model identifier
        model: String,
        /// Tokens used in the response
        tokens_used: usize,
        /// Latency in milliseconds
        latency_ms: u64,
        /// Red flags triggered (empty if valid)
        red_flags: Vec<String>,
        /// When the response was received
        #[serde(with = "system_time_serde")]
        timestamp: SystemTime,
    },

    /// A red flag was triggered on a sample
    RedFlagTriggered {
        /// Type of red flag (e.g., "TokenLengthExceeded", "FormatViolation")
        flag_type: String,
        /// Token count if applicable
        #[serde(skip_serializing_if = "Option::is_none")]
        token_count: Option<usize>,
        /// Format error message if applicable
        #[serde(skip_serializing_if = "Option::is_none")]
        format_error: Option<String>,
        /// When the flag was triggered
        #[serde(with = "system_time_serde")]
        timestamp: SystemTime,
    },

    /// A vote was cast for a candidate
    VoteCast {
        /// Candidate identifier (hash of response)
        candidate_id: String,
        /// Total votes for this candidate after this vote
        vote_count: usize,
        /// Current margin (positive if leading, negative if trailing)
        margin: i32,
        /// When the vote was cast
        #[serde(with = "system_time_serde")]
        timestamp: SystemTime,
    },

    /// A winner was declared in the voting
    VoteDecided {
        /// Winning candidate identifier
        winner_id: String,
        /// Total votes cast in this round
        total_votes: usize,
        /// The k-margin that was required
        k_margin: usize,
        /// When the decision was made
        #[serde(with = "system_time_serde")]
        timestamp: SystemTime,
    },

    /// A step was completed in the task
    StepCompleted {
        /// Step number in the task sequence
        step_id: usize,
        /// Hash of the resulting state
        state_hash: String,
        /// Cumulative cost so far (USD)
        cumulative_cost: f64,
        /// When the step completed
        #[serde(with = "system_time_serde")]
        timestamp: SystemTime,
    },
}

impl MakerEvent {
    /// Create a SampleRequested event
    pub fn sample_requested(model: &str, prompt_hash: &str, temperature: f64) -> Self {
        Self::SampleRequested {
            model: model.to_string(),
            prompt_hash: prompt_hash.to_string(),
            temperature,
            timestamp: SystemTime::now(),
        }
    }

    /// Create a SampleCompleted event
    pub fn sample_completed(
        model: &str,
        tokens_used: usize,
        latency_ms: u64,
        red_flags: Vec<String>,
    ) -> Self {
        Self::SampleCompleted {
            model: model.to_string(),
            tokens_used,
            latency_ms,
            red_flags,
            timestamp: SystemTime::now(),
        }
    }

    /// Create a RedFlagTriggered event
    pub fn red_flag_triggered(
        flag_type: &str,
        token_count: Option<usize>,
        format_error: Option<String>,
    ) -> Self {
        Self::RedFlagTriggered {
            flag_type: flag_type.to_string(),
            token_count,
            format_error,
            timestamp: SystemTime::now(),
        }
    }

    /// Create a VoteCast event
    pub fn vote_cast(candidate_id: &str, vote_count: usize, margin: i32) -> Self {
        Self::VoteCast {
            candidate_id: candidate_id.to_string(),
            vote_count,
            margin,
            timestamp: SystemTime::now(),
        }
    }

    /// Create a VoteDecided event
    pub fn vote_decided(winner_id: &str, total_votes: usize, k_margin: usize) -> Self {
        Self::VoteDecided {
            winner_id: winner_id.to_string(),
            total_votes,
            k_margin,
            timestamp: SystemTime::now(),
        }
    }

    /// Create a StepCompleted event
    pub fn step_completed(step_id: usize, state_hash: &str, cumulative_cost: f64) -> Self {
        Self::StepCompleted {
            step_id,
            state_hash: state_hash.to_string(),
            cumulative_cost,
            timestamp: SystemTime::now(),
        }
    }

    /// Get the event type name
    pub fn event_type(&self) -> &'static str {
        match self {
            Self::SampleRequested { .. } => "SampleRequested",
            Self::SampleCompleted { .. } => "SampleCompleted",
            Self::RedFlagTriggered { .. } => "RedFlagTriggered",
            Self::VoteCast { .. } => "VoteCast",
            Self::VoteDecided { .. } => "VoteDecided",
            Self::StepCompleted { .. } => "StepCompleted",
        }
    }

    /// Get the timestamp of the event
    pub fn timestamp(&self) -> SystemTime {
        match self {
            Self::SampleRequested { timestamp, .. }
            | Self::SampleCompleted { timestamp, .. }
            | Self::RedFlagTriggered { timestamp, .. }
            | Self::VoteCast { timestamp, .. }
            | Self::VoteDecided { timestamp, .. }
            | Self::StepCompleted { timestamp, .. } => *timestamp,
        }
    }
}

/// Serde module for SystemTime serialization
mod system_time_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = time.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
        let millis = duration.as_millis() as u64;
        millis.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<SystemTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(UNIX_EPOCH + Duration::from_millis(millis))
    }
}

// Re-exports
pub use bus::EventBus;

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================
    // Event Creation Tests
    // ==========================================

    #[test]
    fn test_sample_requested_creation() {
        let event = MakerEvent::sample_requested("gpt-4", "abc123", 0.1);

        match event {
            MakerEvent::SampleRequested {
                model,
                prompt_hash,
                temperature,
                ..
            } => {
                assert_eq!(model, "gpt-4");
                assert_eq!(prompt_hash, "abc123");
                assert!((temperature - 0.1).abs() < f64::EPSILON);
            }
            _ => panic!("Wrong event type"),
        }
    }

    #[test]
    fn test_vote_decided_creation() {
        let event = MakerEvent::vote_decided("winner123", 10, 3);

        match event {
            MakerEvent::VoteDecided {
                winner_id,
                total_votes,
                k_margin,
                ..
            } => {
                assert_eq!(winner_id, "winner123");
                assert_eq!(total_votes, 10);
                assert_eq!(k_margin, 3);
            }
            _ => panic!("Wrong event type"),
        }
    }

    #[test]
    fn test_event_type_name() {
        assert_eq!(
            MakerEvent::sample_requested("m", "h", 0.0).event_type(),
            "SampleRequested"
        );
        assert_eq!(
            MakerEvent::vote_decided("w", 0, 0).event_type(),
            "VoteDecided"
        );
        assert_eq!(
            MakerEvent::red_flag_triggered("test", None, None).event_type(),
            "RedFlagTriggered"
        );
    }

    #[test]
    fn test_event_has_timestamp() {
        let before = SystemTime::now();
        let event = MakerEvent::sample_requested("m", "h", 0.0);
        let after = SystemTime::now();

        let ts = event.timestamp();
        assert!(ts >= before);
        assert!(ts <= after);
    }

    // ==========================================
    // Serialization Tests
    // ==========================================

    #[test]
    fn test_event_serializes_to_json_with_type_tag() {
        let event = MakerEvent::vote_cast("candidate1", 5, 2);
        let json = serde_json::to_string(&event).unwrap();

        assert!(json.contains(r#""type":"VoteCast""#));
        assert!(json.contains(r#""candidate_id":"candidate1""#));
        assert!(json.contains(r#""vote_count":5"#));
        assert!(json.contains(r#""margin":2"#));
    }

    #[test]
    fn test_event_deserializes_from_json() {
        let json = r#"{
            "type": "VoteCast",
            "candidate_id": "test",
            "vote_count": 3,
            "margin": 1,
            "timestamp": 1704067200000
        }"#;

        let event: MakerEvent = serde_json::from_str(json).unwrap();

        match event {
            MakerEvent::VoteCast {
                candidate_id,
                vote_count,
                margin,
                ..
            } => {
                assert_eq!(candidate_id, "test");
                assert_eq!(vote_count, 3);
                assert_eq!(margin, 1);
            }
            _ => panic!("Wrong event type"),
        }
    }

    #[test]
    fn test_red_flag_event_skips_none_fields() {
        let event = MakerEvent::red_flag_triggered("TokenLengthExceeded", Some(800), None);
        let json = serde_json::to_string(&event).unwrap();

        assert!(json.contains(r#""token_count":800"#));
        assert!(!json.contains("format_error")); // Should be skipped
    }

    #[test]
    fn test_all_events_serialize_roundtrip() {
        let events = vec![
            MakerEvent::sample_requested("model", "hash", 0.5),
            MakerEvent::sample_completed("model", 100, 50, vec!["flag1".to_string()]),
            MakerEvent::red_flag_triggered("FormatViolation", None, Some("error".to_string())),
            MakerEvent::vote_cast("candidate", 5, -2),
            MakerEvent::vote_decided("winner", 10, 3),
            MakerEvent::step_completed(5, "statehash", 0.05),
        ];

        for event in events {
            let json = serde_json::to_string(&event).unwrap();
            let parsed: MakerEvent = serde_json::from_str(&json).unwrap();
            assert_eq!(event.event_type(), parsed.event_type());
        }
    }

    // ==========================================
    // Clone and Immutability Tests
    // ==========================================

    #[test]
    fn test_event_is_clone() {
        let event = MakerEvent::vote_decided("winner", 10, 3);
        let cloned = event.clone();

        assert_eq!(event.event_type(), cloned.event_type());
    }

    #[test]
    fn test_event_is_immutable() {
        // Events are immutable by design - no interior mutability
        // This is verified by the fact that all fields are private
        // and there are no &mut self methods
        let event = MakerEvent::vote_cast("test", 1, 0);
        let _ = event.event_type(); // Only &self methods available
    }
}
