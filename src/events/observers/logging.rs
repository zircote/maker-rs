//! Logging Observer for MAKER Framework
//!
//! Provides structured logging for all MAKER events using the `tracing` crate.
//! Events are logged at appropriate levels:
//! - INFO: VoteDecided, StepCompleted
//! - WARN: RedFlagTriggered
//! - DEBUG: SampleRequested, SampleCompleted, VoteCast

use crate::events::{EventBus, MakerEvent};
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

/// Observer that logs MAKER events using tracing
///
/// Maps events to appropriate log levels:
/// - `VoteDecided` → INFO (significant decisions)
/// - `StepCompleted` → INFO (progress tracking)
/// - `RedFlagTriggered` → WARN (potential issues)
/// - `SampleRequested/Completed` → DEBUG (high-volume)
/// - `VoteCast` → DEBUG (high-volume)
pub struct LoggingObserver {
    receiver: broadcast::Receiver<MakerEvent>,
}

impl LoggingObserver {
    /// Create a new logging observer subscribed to the event bus
    pub fn new(bus: &EventBus) -> Self {
        Self {
            receiver: bus.subscribe(),
        }
    }

    /// Run the observer, logging events until the channel closes
    ///
    /// This should be spawned as a tokio task:
    /// ```rust,ignore
    /// tokio::spawn(observer.run());
    /// ```
    pub async fn run(mut self) {
        loop {
            match self.receiver.recv().await {
                Ok(event) => Self::log_event(&event),
                Err(broadcast::error::RecvError::Closed) => {
                    debug!("EventBus closed, logging observer stopping");
                    break;
                }
                Err(broadcast::error::RecvError::Lagged(count)) => {
                    warn!(
                        skipped = count,
                        "Logging observer lagged, skipped {} events", count
                    );
                }
            }
        }
    }

    /// Log a single event at the appropriate level
    pub fn log_event(event: &MakerEvent) {
        match event {
            MakerEvent::SampleRequested {
                model,
                prompt_hash,
                temperature,
                ..
            } => {
                debug!(
                    model = %model,
                    prompt_hash = %prompt_hash,
                    temperature = %temperature,
                    "Sample requested"
                );
            }

            MakerEvent::SampleCompleted {
                model,
                tokens_used,
                latency_ms,
                red_flags,
                ..
            } => {
                debug!(
                    model = %model,
                    tokens = tokens_used,
                    latency_ms = latency_ms,
                    red_flags = ?red_flags,
                    "Sample completed"
                );
            }

            MakerEvent::RedFlagTriggered {
                flag_type,
                token_count,
                format_error,
                ..
            } => {
                warn!(
                    flag_type = %flag_type,
                    token_count = ?token_count,
                    format_error = ?format_error,
                    "Red flag triggered"
                );
            }

            MakerEvent::VoteCast {
                candidate_id,
                vote_count,
                margin,
                ..
            } => {
                debug!(
                    candidate = %candidate_id,
                    votes = vote_count,
                    margin = margin,
                    "Vote cast"
                );
            }

            MakerEvent::VoteDecided {
                winner_id,
                total_votes,
                k_margin,
                ..
            } => {
                info!(
                    winner = %winner_id,
                    total_votes = total_votes,
                    k_margin = k_margin,
                    "Vote decided"
                );
            }

            MakerEvent::EscalationTriggered {
                from_model,
                to_model,
                reason,
                ..
            } => {
                info!(
                    from = %from_model,
                    to = %to_model,
                    reason = %reason,
                    "Ensemble escalation triggered"
                );
            }

            MakerEvent::StepCompleted {
                step_id,
                state_hash,
                cumulative_cost,
                ..
            } => {
                info!(
                    step = step_id,
                    state_hash = %state_hash,
                    cost_usd = %cumulative_cost,
                    "Step completed"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================
    // Event Logging Tests
    // ==========================================

    #[test]
    fn test_log_vote_decided_at_info() {
        // This test verifies the function doesn't panic
        // Actual log output verification requires tracing-test or similar
        let event = MakerEvent::vote_decided("winner", 10, 3);
        LoggingObserver::log_event(&event);
    }

    #[test]
    fn test_log_red_flag_at_warn() {
        let event = MakerEvent::red_flag_triggered("TokenLengthExceeded", Some(800), None);
        LoggingObserver::log_event(&event);
    }

    #[test]
    fn test_log_sample_at_debug() {
        let event = MakerEvent::sample_requested("gpt-4", "hash123", 0.1);
        LoggingObserver::log_event(&event);

        let event = MakerEvent::sample_completed("gpt-4", 100, 50, vec![]);
        LoggingObserver::log_event(&event);
    }

    #[test]
    fn test_log_step_completed_at_info() {
        let event = MakerEvent::step_completed(5, "statehash", 0.05);
        LoggingObserver::log_event(&event);
    }

    #[test]
    fn test_log_all_event_types() {
        // Verify all event types can be logged without panic
        let events = vec![
            MakerEvent::sample_requested("m", "h", 0.0),
            MakerEvent::sample_completed("m", 100, 50, vec!["flag".to_string()]),
            MakerEvent::red_flag_triggered("test", None, Some("err".to_string())),
            MakerEvent::vote_cast("c", 1, 0),
            MakerEvent::vote_decided("w", 5, 2),
            MakerEvent::step_completed(1, "hash", 0.01),
        ];

        for event in events {
            LoggingObserver::log_event(&event);
        }
    }

    // ==========================================
    // Observer Integration Tests
    // ==========================================

    #[tokio::test]
    async fn test_observer_receives_events() {
        let bus = EventBus::new(100);
        let observer = LoggingObserver::new(&bus);

        // Emit some events
        bus.emit(MakerEvent::vote_decided("test", 1, 1));

        // Observer should be able to receive (we just verify it doesn't panic)
        // Full integration testing would require tracing-test
        drop(observer);
    }

    #[tokio::test]
    async fn test_observer_run_stops_on_bus_drop() {
        let bus = EventBus::new(100);
        let observer = LoggingObserver::new(&bus);

        // Emit events before running
        bus.emit(MakerEvent::vote_decided("test", 5, 2));
        bus.emit(MakerEvent::red_flag_triggered(
            "TokenLength",
            Some(800),
            None,
        ));
        bus.emit(MakerEvent::sample_requested("model", "hash", 0.1));
        bus.emit(MakerEvent::vote_cast("candidate", 3, 1));
        bus.emit(MakerEvent::step_completed(1, "hash", 0.01));

        // Drop the bus to close the channel
        drop(bus);

        // Observer run should complete when channel closes
        tokio::time::timeout(std::time::Duration::from_secs(2), observer.run())
            .await
            .expect("Observer should stop when bus is dropped");
    }

    #[tokio::test]
    async fn test_observer_run_processes_events() {
        let bus = EventBus::new(100);
        let observer = LoggingObserver::new(&bus);

        let bus_clone = bus.clone();
        let handle = tokio::spawn(async move {
            observer.run().await;
        });

        // Emit events
        bus_clone.emit(MakerEvent::vote_decided("winner", 10, 3));
        bus_clone.emit(MakerEvent::sample_completed("m", 100, 50, vec![]));

        // Small delay to let observer process
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Drop the bus to stop the observer
        drop(bus);
        drop(bus_clone);

        tokio::time::timeout(std::time::Duration::from_secs(2), handle)
            .await
            .expect("Timeout")
            .expect("Observer task should complete");
    }
}
