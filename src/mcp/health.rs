//! Health Check and Metrics for MAKER MCP Server
//!
//! Provides health status endpoint and optional Prometheus metrics.

use rmcp::schemars::{self, JsonSchema};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant, SystemTime};

/// Health status of the MAKER server
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct HealthStatus {
    /// Overall health status
    pub status: HealthState,
    /// Server version
    pub version: String,
    /// Uptime in seconds
    pub uptime_seconds: u64,
    /// Server start time (ISO 8601)
    pub started_at: String,
    /// Component health checks
    pub components: ComponentHealth,
    /// Additional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HealthMetadata>,
}

/// Overall health state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum HealthState {
    /// Server is healthy and operational
    Healthy,
    /// Server is degraded but functional
    Degraded,
    /// Server is unhealthy
    Unhealthy,
}

impl HealthState {
    /// Check if the state is healthy
    pub fn is_healthy(&self) -> bool {
        matches!(self, HealthState::Healthy)
    }
}

/// Health of individual components
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ComponentHealth {
    /// Voting subsystem status
    pub voting: ComponentStatus,
    /// Configuration status
    pub config: ComponentStatus,
    /// LLM provider connectivity (if checked)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_provider: Option<ComponentStatus>,
}

/// Status of a single component
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ComponentStatus {
    /// Component is operational
    pub healthy: bool,
    /// Optional status message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Last check time (ISO 8601)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_check: Option<String>,
}

impl ComponentStatus {
    /// Create a healthy status
    pub fn healthy() -> Self {
        Self {
            healthy: true,
            message: None,
            last_check: Some(iso_now()),
        }
    }

    /// Create an unhealthy status with message
    pub fn unhealthy(message: impl Into<String>) -> Self {
        Self {
            healthy: false,
            message: Some(message.into()),
            last_check: Some(iso_now()),
        }
    }
}

/// Additional health metadata
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct HealthMetadata {
    /// Total votes processed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_votes: Option<u64>,
    /// Active sessions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_sessions: Option<u32>,
    /// Memory usage in bytes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub memory_bytes: Option<u64>,
}

/// Health check service
#[derive(Debug)]
pub struct HealthChecker {
    start_time: Instant,
    start_system_time: SystemTime,
    version: String,
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl HealthChecker {
    /// Create a new health checker
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            start_system_time: SystemTime::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Create with custom version
    pub fn with_version(version: impl Into<String>) -> Self {
        Self {
            start_time: Instant::now(),
            start_system_time: SystemTime::now(),
            version: version.into(),
        }
    }

    /// Get current uptime
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get server version
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Perform health check
    pub fn check(&self) -> HealthStatus {
        let uptime = self.uptime();

        // Check voting subsystem (always healthy if we got here)
        let voting = ComponentStatus::healthy();

        // Check configuration (always healthy for now)
        let config = ComponentStatus::healthy();

        // Determine overall status
        let status = if voting.healthy && config.healthy {
            HealthState::Healthy
        } else if voting.healthy || config.healthy {
            HealthState::Degraded
        } else {
            HealthState::Unhealthy
        };

        HealthStatus {
            status,
            version: self.version.clone(),
            uptime_seconds: uptime.as_secs(),
            started_at: system_time_to_iso(&self.start_system_time),
            components: ComponentHealth {
                voting,
                config,
                llm_provider: None,
            },
            metadata: None,
        }
    }

    /// Perform health check with LLM provider check
    pub fn check_with_provider(&self, provider_healthy: bool) -> HealthStatus {
        let mut status = self.check();

        let provider_status = if provider_healthy {
            ComponentStatus::healthy()
        } else {
            ComponentStatus::unhealthy("Provider connection failed")
        };

        status.components.llm_provider = Some(provider_status);

        // Update overall status if provider is unhealthy
        if !provider_healthy {
            status.status = HealthState::Degraded;
        }

        status
    }
}

/// Validate configuration and return any errors
pub fn validate_config(config: &super::server::ServerConfig) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    if config.k_default == 0 {
        errors.push("k_default must be >= 1".to_string());
    }

    if config.k_bounds.0 > config.k_bounds.1 {
        errors.push(format!(
            "k_bounds min ({}) must be <= max ({})",
            config.k_bounds.0, config.k_bounds.1
        ));
    }

    if config.temperature_diversity < 0.0 || config.temperature_diversity > 2.0 {
        errors.push(format!(
            "temperature_diversity ({}) must be between 0.0 and 2.0",
            config.temperature_diversity
        ));
    }

    if config.ema_alpha <= 0.0 || config.ema_alpha >= 1.0 {
        errors.push(format!(
            "ema_alpha ({}) must be between 0.0 and 1.0 (exclusive)",
            config.ema_alpha
        ));
    }

    if config.max_prompt_length == 0 {
        errors.push("max_prompt_length must be > 0".to_string());
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// Helper functions

fn iso_now() -> String {
    humantime::format_rfc3339(SystemTime::now()).to_string()
}

fn system_time_to_iso(time: &SystemTime) -> String {
    humantime::format_rfc3339(*time).to_string()
}

// ============================================================================
// Prometheus Metrics (behind feature flag)
// ============================================================================

#[cfg(feature = "prometheus")]
pub mod prometheus_metrics {
    //! Prometheus metrics for MAKER server
    //!
    //! Enable with the `prometheus` feature flag.

    use once_cell::sync::Lazy;
    use prometheus::{
        register_counter_vec, register_gauge, register_histogram_vec, CounterVec, Gauge,
        HistogramVec,
    };

    /// Total votes processed
    pub static VOTES_TOTAL: Lazy<CounterVec> = Lazy::new(|| {
        register_counter_vec!(
            "maker_votes_total",
            "Total number of votes processed",
            &["status"]
        )
        .expect("Failed to register votes_total metric")
    });

    /// Vote latency histogram
    pub static VOTE_LATENCY: Lazy<HistogramVec> = Lazy::new(|| {
        register_histogram_vec!(
            "maker_vote_latency_seconds",
            "Vote latency in seconds",
            &["provider"],
            vec![0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        .expect("Failed to register vote_latency metric")
    });

    /// Active voting sessions
    pub static ACTIVE_SESSIONS: Lazy<Gauge> = Lazy::new(|| {
        register_gauge!("maker_active_sessions", "Number of active voting sessions")
            .expect("Failed to register active_sessions metric")
    });

    /// Red-flagged samples counter
    pub static RED_FLAGGED_TOTAL: Lazy<CounterVec> = Lazy::new(|| {
        register_counter_vec!(
            "maker_red_flagged_total",
            "Total number of red-flagged samples",
            &["flag_type"]
        )
        .expect("Failed to register red_flagged_total metric")
    });

    /// Record a successful vote
    pub fn record_vote_success(provider: &str, latency_secs: f64) {
        VOTES_TOTAL.with_label_values(&["success"]).inc();
        VOTE_LATENCY
            .with_label_values(&[provider])
            .observe(latency_secs);
    }

    /// Record a failed vote
    pub fn record_vote_failure() {
        VOTES_TOTAL.with_label_values(&["failure"]).inc();
    }

    /// Record a red-flagged sample
    pub fn record_red_flag(flag_type: &str) {
        RED_FLAGGED_TOTAL.with_label_values(&[flag_type]).inc();
    }

    /// Increment active sessions
    pub fn inc_active_sessions() {
        ACTIVE_SESSIONS.inc();
    }

    /// Decrement active sessions
    pub fn dec_active_sessions() {
        ACTIVE_SESSIONS.dec();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_checker_new() {
        let checker = HealthChecker::new();
        assert!(!checker.version().is_empty());
    }

    #[test]
    fn test_health_checker_uptime() {
        let checker = HealthChecker::new();
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(checker.uptime().as_millis() >= 10);
    }

    #[test]
    fn test_health_check_basic() {
        let checker = HealthChecker::new();
        let status = checker.check();

        assert_eq!(status.status, HealthState::Healthy);
        assert!(status.components.voting.healthy);
        assert!(status.components.config.healthy);
        assert!(status.components.llm_provider.is_none());
    }

    #[test]
    fn test_health_check_with_provider() {
        let checker = HealthChecker::new();

        let healthy_status = checker.check_with_provider(true);
        assert_eq!(healthy_status.status, HealthState::Healthy);
        assert!(healthy_status.components.llm_provider.unwrap().healthy);

        let degraded_status = checker.check_with_provider(false);
        assert_eq!(degraded_status.status, HealthState::Degraded);
        assert!(!degraded_status.components.llm_provider.unwrap().healthy);
    }

    #[test]
    fn test_health_state_is_healthy() {
        assert!(HealthState::Healthy.is_healthy());
        assert!(!HealthState::Degraded.is_healthy());
        assert!(!HealthState::Unhealthy.is_healthy());
    }

    #[test]
    fn test_component_status_healthy() {
        let status = ComponentStatus::healthy();
        assert!(status.healthy);
        assert!(status.message.is_none());
        assert!(status.last_check.is_some());
    }

    #[test]
    fn test_component_status_unhealthy() {
        let status = ComponentStatus::unhealthy("Test error");
        assert!(!status.healthy);
        assert_eq!(status.message.as_deref(), Some("Test error"));
        assert!(status.last_check.is_some());
    }

    #[test]
    fn test_validate_config_valid() {
        let config = super::super::server::ServerConfig::default();
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_validate_config_invalid_k() {
        let mut config = super::super::server::ServerConfig::default();
        config.k_default = 0;
        let result = validate_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err()[0].contains("k_default"));
    }

    #[test]
    fn test_validate_config_invalid_bounds() {
        let mut config = super::super::server::ServerConfig::default();
        config.k_bounds = (10, 5);
        let result = validate_config(&config);
        assert!(result.is_err());
        assert!(result.unwrap_err()[0].contains("k_bounds"));
    }

    #[test]
    fn test_validate_config_invalid_ema() {
        let mut config = super::super::server::ServerConfig::default();
        config.ema_alpha = 0.0;
        let result = validate_config(&config);
        assert!(result.is_err());

        config.ema_alpha = 1.0;
        let result = validate_config(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_health_status_serialization() {
        let checker = HealthChecker::new();
        let status = checker.check();

        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"status\":\"healthy\""));
        assert!(json.contains("\"version\""));
        assert!(json.contains("\"uptime_seconds\""));
    }
}
