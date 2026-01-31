//! Event Observers for MAKER Framework
//!
//! Observers subscribe to the EventBus and process events for different purposes:
//! - `logging`: Structured logging via tracing
//! - `metrics`: Prometheus-compatible metrics tracking

pub mod logging;
pub mod metrics;

pub use logging::LoggingObserver;
pub use metrics::MetricsObserver;
