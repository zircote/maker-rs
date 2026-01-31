//! EventBus - Central event distribution for MAKER Framework
//!
//! Provides a broadcast-based event bus for decoupled observability.
//! Core logic emits events, and multiple observers can subscribe to
//! receive all events without blocking the emitter.
//!
//! # Design
//!
//! - Uses `tokio::sync::broadcast` for multi-producer, multi-consumer
//! - Non-blocking emit (fire-and-forget)
//! - Lagging receivers drop old events (no backpressure)
//! - Thread-safe via Clone (Arc internally)

use super::MakerEvent;
use std::sync::Arc;
use tokio::sync::broadcast;

/// Default capacity for the event bus channel
pub const DEFAULT_CAPACITY: usize = 1024;

/// Central event bus for MAKER observability
///
/// The EventBus uses a broadcast channel to distribute events to all
/// subscribers. Emitting is non-blocking and fire-and-forget.
///
/// # Example
///
/// ```rust,ignore
/// use maker::events::{EventBus, MakerEvent};
///
/// let bus = EventBus::new(1024);
/// let mut rx = bus.subscribe();
///
/// // Emit events from anywhere
/// bus.emit(MakerEvent::vote_decided("winner", 10, 3));
///
/// // Receive in observers
/// while let Ok(event) = rx.recv().await {
///     println!("Event: {:?}", event);
/// }
/// ```
#[derive(Clone)]
pub struct EventBus {
    sender: Arc<broadcast::Sender<MakerEvent>>,
}

impl EventBus {
    /// Create a new EventBus with the specified capacity
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of events to buffer before old events
    ///   are dropped for lagging receivers
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self {
            sender: Arc::new(sender),
        }
    }

    /// Create an EventBus with default capacity (1024)
    pub fn with_default_capacity() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }

    /// Emit an event to all subscribers
    ///
    /// This is non-blocking and fire-and-forget. If there are no
    /// subscribers, the event is silently dropped.
    ///
    /// # Arguments
    ///
    /// * `event` - The event to emit
    pub fn emit(&self, event: MakerEvent) {
        // Ignore send errors (no receivers)
        let _ = self.sender.send(event);
    }

    /// Subscribe to receive events
    ///
    /// Returns a receiver that will get all events emitted after
    /// subscription. If the receiver falls behind, old events are dropped.
    pub fn subscribe(&self) -> broadcast::Receiver<MakerEvent> {
        self.sender.subscribe()
    }

    /// Get the current number of active subscribers
    pub fn subscriber_count(&self) -> usize {
        self.sender.receiver_count()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::with_default_capacity()
    }
}

impl std::fmt::Debug for EventBus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventBus")
            .field("subscriber_count", &self.subscriber_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================
    // Construction Tests
    // ==========================================

    #[test]
    fn test_new_with_capacity() {
        let bus = EventBus::new(100);
        assert_eq!(bus.subscriber_count(), 0);
    }

    #[test]
    fn test_default_capacity() {
        let bus = EventBus::with_default_capacity();
        assert_eq!(bus.subscriber_count(), 0);
    }

    #[test]
    fn test_clone_shares_channel() {
        let bus1 = EventBus::new(100);
        let _rx1 = bus1.subscribe();

        let bus2 = bus1.clone();
        let _rx2 = bus2.subscribe();

        // Both clones share the same channel
        assert_eq!(bus1.subscriber_count(), 2);
        assert_eq!(bus2.subscriber_count(), 2);
    }

    // ==========================================
    // Emit Tests
    // ==========================================

    #[test]
    fn test_emit_without_subscribers_doesnt_panic() {
        let bus = EventBus::new(100);
        // Should not panic even with no subscribers
        bus.emit(MakerEvent::vote_decided("test", 1, 1));
    }

    #[tokio::test]
    async fn test_emit_reaches_subscriber() {
        let bus = EventBus::new(100);
        let mut rx = bus.subscribe();

        bus.emit(MakerEvent::vote_decided("winner", 10, 3));

        let event = rx.recv().await.unwrap();
        match event {
            MakerEvent::VoteDecided { winner_id, .. } => {
                assert_eq!(winner_id, "winner");
            }
            _ => panic!("Wrong event type"),
        }
    }

    #[tokio::test]
    async fn test_multiple_subscribers_receive_all_events() {
        let bus = EventBus::new(100);
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();
        let mut rx3 = bus.subscribe();

        bus.emit(MakerEvent::vote_cast("A", 1, 1));
        bus.emit(MakerEvent::vote_cast("B", 1, 0));

        // All subscribers should receive both events
        for rx in [&mut rx1, &mut rx2, &mut rx3] {
            let e1 = rx.recv().await.unwrap();
            let e2 = rx.recv().await.unwrap();
            assert_eq!(e1.event_type(), "VoteCast");
            assert_eq!(e2.event_type(), "VoteCast");
        }
    }

    #[tokio::test]
    async fn test_emit_is_non_blocking() {
        let bus = EventBus::new(100);
        let _rx = bus.subscribe();

        // Emit many events - should not block
        for i in 0..1000 {
            bus.emit(MakerEvent::vote_cast(&format!("c{}", i), i, 0));
        }

        // If we get here, emit was non-blocking (test passes by not hanging)
    }

    // ==========================================
    // Subscriber Count Tests
    // ==========================================

    #[test]
    fn test_subscriber_count_increases_on_subscribe() {
        let bus = EventBus::new(100);
        assert_eq!(bus.subscriber_count(), 0);

        let _rx1 = bus.subscribe();
        assert_eq!(bus.subscriber_count(), 1);

        let _rx2 = bus.subscribe();
        assert_eq!(bus.subscriber_count(), 2);
    }

    #[test]
    fn test_subscriber_count_decreases_on_drop() {
        let bus = EventBus::new(100);

        let rx1 = bus.subscribe();
        let rx2 = bus.subscribe();
        assert_eq!(bus.subscriber_count(), 2);

        drop(rx1);
        assert_eq!(bus.subscriber_count(), 1);

        drop(rx2);
        assert_eq!(bus.subscriber_count(), 0);
    }

    // ==========================================
    // Integration Test
    // ==========================================

    #[tokio::test]
    async fn test_emit_1000_events_all_subscribers_receive() {
        let bus = EventBus::new(2000); // Large enough buffer
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();

        // Emit 1000 events
        for i in 0..1000 {
            bus.emit(MakerEvent::step_completed(
                i,
                &format!("hash{}", i),
                i as f64 * 0.01,
            ));
        }

        // Both subscribers should receive all 1000
        for rx in [&mut rx1, &mut rx2] {
            for i in 0..1000 {
                let event = rx.recv().await.unwrap();
                match event {
                    MakerEvent::StepCompleted { step_id, .. } => {
                        assert_eq!(step_id, i);
                    }
                    _ => panic!("Wrong event type"),
                }
            }
        }
    }
}
