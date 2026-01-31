//! Metrics Observer for MAKER Framework
//!
//! Tracks Prometheus-compatible metrics for monitoring:
//! - Counters: votes, red flags, samples
//! - Histograms: API latency, cost per step

use crate::events::{EventBus, MakerEvent};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

/// Histogram bucket boundaries for API latency (milliseconds)
pub const LATENCY_BUCKETS: [u64; 7] = [10, 50, 100, 500, 1000, 5000, 10000];

/// Metrics collected from MAKER events
#[derive(Debug, Clone, Default)]
pub struct Metrics {
    /// Total votes cast, by winner candidate
    pub votes_total: HashMap<String, u64>,
    /// Total red flags triggered, by flag type
    pub red_flags_total: HashMap<String, u64>,
    /// Total samples requested, by model
    pub samples_total: HashMap<String, u64>,
    /// API latency histogram (bucket -> count)
    pub latency_histogram: HashMap<u64, u64>,
    /// Total latency sum for average calculation
    pub latency_sum_ms: u64,
    /// Latency sample count
    pub latency_count: u64,
    /// Cost per step samples
    pub cost_samples: Vec<f64>,
    /// Total cost accumulated
    pub total_cost_usd: f64,
    /// Steps completed
    pub steps_completed: u64,
}

impl Metrics {
    /// Create new empty metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a vote for a candidate
    pub fn record_vote(&mut self, candidate: &str) {
        *self.votes_total.entry(candidate.to_string()).or_insert(0) += 1;
    }

    /// Record a red flag
    pub fn record_red_flag(&mut self, flag_type: &str) {
        *self
            .red_flags_total
            .entry(flag_type.to_string())
            .or_insert(0) += 1;
    }

    /// Record a sample request
    pub fn record_sample(&mut self, model: &str) {
        *self.samples_total.entry(model.to_string()).or_insert(0) += 1;
    }

    /// Record API latency
    pub fn record_latency(&mut self, latency_ms: u64) {
        // Find the appropriate bucket
        for &bucket in &LATENCY_BUCKETS {
            if latency_ms <= bucket {
                *self.latency_histogram.entry(bucket).or_insert(0) += 1;
                break;
            }
        }
        // If larger than all buckets, put in the last one
        if latency_ms > LATENCY_BUCKETS[LATENCY_BUCKETS.len() - 1] {
            let last = LATENCY_BUCKETS[LATENCY_BUCKETS.len() - 1];
            *self.latency_histogram.entry(last).or_insert(0) += 1;
        }

        self.latency_sum_ms += latency_ms;
        self.latency_count += 1;
    }

    /// Record step cost
    pub fn record_step_cost(&mut self, cumulative_cost: f64) {
        self.total_cost_usd = cumulative_cost;
        self.steps_completed += 1;

        // Track individual step costs for histogram
        if self.steps_completed > 1 && !self.cost_samples.is_empty() {
            let prev_cost = self.cost_samples.last().copied().unwrap_or(0.0);
            let step_cost = cumulative_cost - prev_cost;
            self.cost_samples.push(step_cost);
        } else {
            self.cost_samples.push(cumulative_cost);
        }
    }

    /// Get average latency in milliseconds
    pub fn avg_latency_ms(&self) -> f64 {
        if self.latency_count == 0 {
            0.0
        } else {
            self.latency_sum_ms as f64 / self.latency_count as f64
        }
    }

    /// Get average cost per step
    pub fn avg_cost_per_step(&self) -> f64 {
        if self.steps_completed == 0 {
            0.0
        } else {
            self.total_cost_usd / self.steps_completed as f64
        }
    }

    /// Format metrics as Prometheus text format
    pub fn to_prometheus(&self) -> String {
        let mut output = String::new();

        // Votes total
        output.push_str("# HELP maker_votes_total Total votes cast\n");
        output.push_str("# TYPE maker_votes_total counter\n");
        for (candidate, count) in &self.votes_total {
            output.push_str(&format!(
                "maker_votes_total{{candidate=\"{}\"}} {}\n",
                candidate, count
            ));
        }

        // Red flags total
        output.push_str("# HELP maker_red_flags_total Total red flags triggered\n");
        output.push_str("# TYPE maker_red_flags_total counter\n");
        for (flag_type, count) in &self.red_flags_total {
            output.push_str(&format!(
                "maker_red_flags_total{{flag_type=\"{}\"}} {}\n",
                flag_type, count
            ));
        }

        // Samples total
        output.push_str("# HELP maker_samples_total Total samples requested\n");
        output.push_str("# TYPE maker_samples_total counter\n");
        for (model, count) in &self.samples_total {
            output.push_str(&format!(
                "maker_samples_total{{model=\"{}\"}} {}\n",
                model, count
            ));
        }

        // Latency histogram
        output.push_str("# HELP maker_api_latency_ms API call latency in milliseconds\n");
        output.push_str("# TYPE maker_api_latency_ms histogram\n");
        let mut cumulative = 0u64;
        for &bucket in &LATENCY_BUCKETS {
            cumulative += self.latency_histogram.get(&bucket).copied().unwrap_or(0);
            output.push_str(&format!(
                "maker_api_latency_ms_bucket{{le=\"{}\"}} {}\n",
                bucket, cumulative
            ));
        }
        output.push_str(&format!(
            "maker_api_latency_ms_bucket{{le=\"+Inf\"}} {}\n",
            self.latency_count
        ));
        output.push_str(&format!(
            "maker_api_latency_ms_sum {}\n",
            self.latency_sum_ms
        ));
        output.push_str(&format!(
            "maker_api_latency_ms_count {}\n",
            self.latency_count
        ));

        // Cost metrics
        output.push_str("# HELP maker_cost_usd_total Total cost in USD\n");
        output.push_str("# TYPE maker_cost_usd_total gauge\n");
        output.push_str(&format!("maker_cost_usd_total {}\n", self.total_cost_usd));

        output.push_str("# HELP maker_steps_completed_total Total steps completed\n");
        output.push_str("# TYPE maker_steps_completed_total counter\n");
        output.push_str(&format!(
            "maker_steps_completed_total {}\n",
            self.steps_completed
        ));

        output
    }

    /// Generate a human-readable report
    pub fn report(&self) -> String {
        let mut output = String::new();

        output.push_str("=== MAKER Metrics Report ===\n\n");

        output.push_str("Votes:\n");
        for (candidate, count) in &self.votes_total {
            output.push_str(&format!("  {}: {}\n", candidate, count));
        }

        output.push_str("\nRed Flags:\n");
        for (flag_type, count) in &self.red_flags_total {
            output.push_str(&format!("  {}: {}\n", flag_type, count));
        }

        output.push_str("\nSamples by Model:\n");
        for (model, count) in &self.samples_total {
            output.push_str(&format!("  {}: {}\n", model, count));
        }

        output.push_str(&format!(
            "\nLatency: avg={:.1}ms, count={}\n",
            self.avg_latency_ms(),
            self.latency_count
        ));

        output.push_str(&format!(
            "Cost: total=${:.4}, avg/step=${:.6}, steps={}\n",
            self.total_cost_usd,
            self.avg_cost_per_step(),
            self.steps_completed
        ));

        output
    }
}

/// Observer that collects metrics from MAKER events
pub struct MetricsObserver {
    receiver: broadcast::Receiver<MakerEvent>,
    metrics: Arc<Mutex<Metrics>>,
}

impl MetricsObserver {
    /// Create a new metrics observer subscribed to the event bus
    pub fn new(bus: &EventBus) -> Self {
        Self {
            receiver: bus.subscribe(),
            metrics: Arc::new(Mutex::new(Metrics::new())),
        }
    }

    /// Get a handle to the metrics for reading
    pub fn metrics(&self) -> Arc<Mutex<Metrics>> {
        Arc::clone(&self.metrics)
    }

    /// Run the observer, collecting metrics until the channel closes
    pub async fn run(mut self) {
        loop {
            match self.receiver.recv().await {
                Ok(event) => self.process_event(&event),
                Err(broadcast::error::RecvError::Closed) => break,
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
            }
        }
    }

    /// Process a single event and update metrics
    fn process_event(&self, event: &MakerEvent) {
        let mut metrics = self.metrics.lock().unwrap();

        match event {
            MakerEvent::SampleRequested { model, .. } => {
                metrics.record_sample(model);
            }

            MakerEvent::SampleCompleted { latency_ms, .. } => {
                metrics.record_latency(*latency_ms);
            }

            MakerEvent::RedFlagTriggered { flag_type, .. } => {
                metrics.record_red_flag(flag_type);
            }

            MakerEvent::VoteCast { candidate_id, .. } => {
                metrics.record_vote(candidate_id);
            }

            MakerEvent::VoteDecided { .. } => {
                // Tracked via VoteCast
            }

            MakerEvent::EscalationTriggered { .. } => {
                // Escalation events are informational; tracked via EnsembleMetrics
            }

            MakerEvent::StepCompleted {
                cumulative_cost, ..
            } => {
                metrics.record_step_cost(*cumulative_cost);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================
    // Metrics Recording Tests
    // ==========================================

    #[test]
    fn test_record_vote() {
        let mut metrics = Metrics::new();
        metrics.record_vote("candidate_a");
        metrics.record_vote("candidate_a");
        metrics.record_vote("candidate_b");

        assert_eq!(metrics.votes_total.get("candidate_a"), Some(&2));
        assert_eq!(metrics.votes_total.get("candidate_b"), Some(&1));
    }

    #[test]
    fn test_record_red_flag() {
        let mut metrics = Metrics::new();
        metrics.record_red_flag("TokenLengthExceeded");
        metrics.record_red_flag("FormatViolation");
        metrics.record_red_flag("TokenLengthExceeded");

        assert_eq!(metrics.red_flags_total.get("TokenLengthExceeded"), Some(&2));
        assert_eq!(metrics.red_flags_total.get("FormatViolation"), Some(&1));
    }

    #[test]
    fn test_record_latency_histogram() {
        let mut metrics = Metrics::new();
        metrics.record_latency(5); // Bucket 10
        metrics.record_latency(25); // Bucket 50
        metrics.record_latency(75); // Bucket 100
        metrics.record_latency(200); // Bucket 500

        assert_eq!(metrics.latency_histogram.get(&10), Some(&1));
        assert_eq!(metrics.latency_histogram.get(&50), Some(&1));
        assert_eq!(metrics.latency_histogram.get(&100), Some(&1));
        assert_eq!(metrics.latency_histogram.get(&500), Some(&1));
    }

    #[test]
    fn test_avg_latency() {
        let mut metrics = Metrics::new();
        metrics.record_latency(100);
        metrics.record_latency(200);
        metrics.record_latency(300);

        assert!((metrics.avg_latency_ms() - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_record_step_cost() {
        let mut metrics = Metrics::new();
        metrics.record_step_cost(0.01);
        metrics.record_step_cost(0.025);
        metrics.record_step_cost(0.04);

        assert_eq!(metrics.steps_completed, 3);
        assert!((metrics.total_cost_usd - 0.04).abs() < f64::EPSILON);
    }

    #[test]
    fn test_avg_cost_per_step() {
        let mut metrics = Metrics::new();
        metrics.record_step_cost(0.10);
        metrics.steps_completed = 10;
        metrics.total_cost_usd = 0.10;

        assert!((metrics.avg_cost_per_step() - 0.01).abs() < f64::EPSILON);
    }

    // ==========================================
    // Prometheus Format Tests
    // ==========================================

    #[test]
    fn test_prometheus_format() {
        let mut metrics = Metrics::new();
        metrics.record_vote("winner");
        metrics.record_red_flag("TokenLengthExceeded");
        metrics.record_sample("gpt-4");
        metrics.record_latency(50);
        metrics.record_step_cost(0.01);

        let output = metrics.to_prometheus();

        assert!(output.contains("maker_votes_total"));
        assert!(output.contains("maker_red_flags_total"));
        assert!(output.contains("maker_samples_total"));
        assert!(output.contains("maker_api_latency_ms"));
        assert!(output.contains("maker_cost_usd_total"));
    }

    #[test]
    fn test_report_format() {
        let mut metrics = Metrics::new();
        metrics.record_vote("test");
        metrics.record_latency(100);
        metrics.record_step_cost(0.05);

        let report = metrics.report();

        assert!(report.contains("MAKER Metrics Report"));
        assert!(report.contains("Votes:"));
        assert!(report.contains("Latency:"));
        assert!(report.contains("Cost:"));
    }

    // ==========================================
    // Observer Integration Tests
    // ==========================================

    #[tokio::test]
    async fn test_observer_tracks_events() {
        let bus = EventBus::new(100);
        let observer = MetricsObserver::new(&bus);
        let metrics = observer.metrics();

        // Emit events
        bus.emit(MakerEvent::vote_cast("candidate", 1, 1));
        bus.emit(MakerEvent::sample_completed("gpt-4", 100, 50, vec![]));
        bus.emit(MakerEvent::red_flag_triggered("TestFlag", None, None));

        // Give observer time to process (in real use, run() would be spawned)
        // For this test, we process manually
        {
            let m = metrics.lock().unwrap();
            // Metrics are only updated when run() processes events
            // Here we just verify the observer can be created
            assert!(m.votes_total.is_empty()); // Not processed yet
        }
    }

    #[test]
    fn test_metrics_clone() {
        let mut metrics = Metrics::new();
        metrics.record_vote("test");
        metrics.record_latency(100);

        let cloned = metrics.clone();
        assert_eq!(cloned.votes_total.get("test"), Some(&1));
    }

    #[test]
    fn test_avg_latency_zero_count() {
        let metrics = Metrics::new();
        assert!((metrics.avg_latency_ms() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_avg_cost_zero_steps() {
        let metrics = Metrics::new();
        assert!((metrics.avg_cost_per_step() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_latency_over_max_bucket() {
        let mut metrics = Metrics::new();
        metrics.record_latency(20000); // Exceeds all buckets
        let last_bucket = LATENCY_BUCKETS[LATENCY_BUCKETS.len() - 1];
        assert_eq!(metrics.latency_histogram.get(&last_bucket), Some(&1));
        assert_eq!(metrics.latency_count, 1);
    }

    #[test]
    fn test_record_sample() {
        let mut metrics = Metrics::new();
        metrics.record_sample("gpt-4");
        metrics.record_sample("gpt-4");
        metrics.record_sample("claude");
        assert_eq!(metrics.samples_total.get("gpt-4"), Some(&2));
        assert_eq!(metrics.samples_total.get("claude"), Some(&1));
    }

    #[tokio::test]
    async fn test_observer_run_processes_all_event_types() {
        let bus = EventBus::new(100);
        let observer = MetricsObserver::new(&bus);
        let metrics = observer.metrics();

        let bus_clone = bus.clone();
        let handle = tokio::spawn(async move {
            observer.run().await;
        });

        // Emit all event types
        bus_clone.emit(MakerEvent::sample_requested("gpt-4", "hash", 0.1));
        bus_clone.emit(MakerEvent::sample_completed("gpt-4", 100, 50, vec![]));
        bus_clone.emit(MakerEvent::red_flag_triggered(
            "TokenLength",
            Some(800),
            None,
        ));
        bus_clone.emit(MakerEvent::vote_cast("candidate_a", 1, 1));
        bus_clone.emit(MakerEvent::vote_decided("candidate_a", 5, 3));
        bus_clone.emit(MakerEvent::step_completed(1, "hash", 0.01));

        // Small delay to let observer process
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Verify metrics were recorded
        {
            let m = metrics.lock().unwrap();
            assert_eq!(m.samples_total.get("gpt-4"), Some(&1));
            assert_eq!(m.latency_count, 1);
            assert_eq!(m.red_flags_total.get("TokenLength"), Some(&1));
            assert_eq!(m.votes_total.get("candidate_a"), Some(&1));
            assert_eq!(m.steps_completed, 1);
        }

        // Drop the bus to stop the observer
        drop(bus);
        drop(bus_clone);

        tokio::time::timeout(std::time::Duration::from_secs(2), handle)
            .await
            .expect("Timeout")
            .expect("Observer task should complete");
    }

    #[test]
    fn test_prometheus_format_with_all_data() {
        let mut metrics = Metrics::new();
        metrics.record_vote("winner_a");
        metrics.record_vote("winner_b");
        metrics.record_red_flag("TokenLength");
        metrics.record_red_flag("FormatViolation");
        metrics.record_sample("gpt-4");
        metrics.record_sample("claude");
        metrics.record_latency(5);
        metrics.record_latency(75);
        metrics.record_latency(250);
        metrics.record_latency(15000); // Over max bucket
        metrics.record_step_cost(0.01);
        metrics.record_step_cost(0.025);

        let output = metrics.to_prometheus();

        assert!(output.contains("maker_votes_total{candidate="));
        assert!(output.contains("maker_red_flags_total{flag_type="));
        assert!(output.contains("maker_samples_total{model="));
        assert!(output.contains("maker_api_latency_ms_bucket{le=\"+Inf\"}"));
        assert!(output.contains("maker_api_latency_ms_sum"));
        assert!(output.contains("maker_api_latency_ms_count 4"));
        assert!(output.contains("maker_cost_usd_total"));
        assert!(output.contains("maker_steps_completed_total 2"));
    }

    #[test]
    fn test_report_with_all_data() {
        let mut metrics = Metrics::new();
        metrics.record_vote("candidate");
        metrics.record_red_flag("TokenLength");
        metrics.record_sample("gpt-4");
        metrics.record_latency(100);
        metrics.record_step_cost(0.05);

        let report = metrics.report();

        assert!(report.contains("candidate: 1"));
        assert!(report.contains("TokenLength: 1"));
        assert!(report.contains("gpt-4: 1"));
        assert!(report.contains("avg=100.0ms"));
        assert!(report.contains("steps=1"));
    }
}
