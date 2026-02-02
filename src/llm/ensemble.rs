//! Multi-Model Ensemble Configuration for MAKER Framework
//!
//! Enables voting across heterogeneous LLM models to decorrelate errors
//! by model architecture, not just sampling temperature.
//!
//! # Strategies
//!
//! - **RoundRobin**: Distribute samples evenly across models
//! - **CostAware**: Start with cheap models, escalate on disagreement
//! - **ReliabilityWeighted**: More samples from higher-reliability models
//!
//! # Example
//!
//! ```rust,ignore
//! use maker::llm::ensemble::{EnsembleConfig, ModelSlot, EnsembleStrategy, CostTier};
//!
//! let config = EnsembleConfig::new(
//!     vec![
//!         ModelSlot::new(cheap_client, 1.0, CostTier::Cheap),
//!         ModelSlot::new(expensive_client, 1.0, CostTier::Expensive),
//!     ],
//!     EnsembleStrategy::CostAware,
//! ).unwrap();
//!
//! let slot = config.select_model_for_sample(0, None);
//! ```

use crate::llm::LlmClient;
use rmcp::schemars::{self, JsonSchema};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Cost tier for a model in the ensemble
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum CostTier {
    /// Cheapest models (e.g., local Ollama, small cloud models)
    Cheap,
    /// Mid-range models (e.g., Claude Haiku, GPT-4o-mini)
    Medium,
    /// Most expensive models (e.g., Claude Opus, GPT-4)
    Expensive,
}

impl CostTier {
    /// Ordering value for cost comparison (lower = cheaper)
    pub fn cost_order(&self) -> u8 {
        match self {
            CostTier::Cheap => 0,
            CostTier::Medium => 1,
            CostTier::Expensive => 2,
        }
    }
}

/// Strategy for selecting models during ensemble voting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EnsembleStrategy {
    /// Distribute samples evenly across all models
    RoundRobin,
    /// Use cheap models first, escalate to expensive on disagreement
    CostAware,
    /// Weight sampling toward higher-reliability models
    ReliabilityWeighted,
}

/// A slot in the ensemble representing one model
pub struct ModelSlot {
    /// The LLM client for this model
    pub client: Arc<dyn LlmClient>,
    /// Sampling weight (higher = more samples in weighted strategies)
    pub weight: f64,
    /// Cost tier for cost-aware routing
    pub cost_tier: CostTier,
}

impl ModelSlot {
    /// Create a new model slot
    pub fn new(client: Arc<dyn LlmClient>, weight: f64, cost_tier: CostTier) -> Self {
        Self {
            client,
            weight: weight.max(0.0),
            cost_tier,
        }
    }

    /// Get the model name from the underlying client
    pub fn model_name(&self) -> &str {
        self.client.model_name()
    }
}

impl std::fmt::Debug for ModelSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelSlot")
            .field("model", &self.client.model_name())
            .field("weight", &self.weight)
            .field("cost_tier", &self.cost_tier)
            .finish()
    }
}

/// Configuration for multi-model ensemble voting
pub struct EnsembleConfig {
    /// Models in the ensemble (2-5 models)
    models: Vec<ModelSlot>,
    /// Strategy for selecting models
    strategy: EnsembleStrategy,
}

/// Errors when creating an ensemble configuration
#[derive(Debug, Clone, PartialEq)]
pub enum EnsembleError {
    /// Too few models (need at least 2)
    TooFewModels { count: usize },
    /// Too many models (max 5)
    TooManyModels { count: usize },
    /// All weights are zero
    AllZeroWeights,
}

impl std::fmt::Display for EnsembleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnsembleError::TooFewModels { count } => {
                write!(f, "ensemble requires at least 2 models, got {}", count)
            }
            EnsembleError::TooManyModels { count } => {
                write!(f, "ensemble supports at most 5 models, got {}", count)
            }
            EnsembleError::AllZeroWeights => {
                write!(f, "at least one model must have weight > 0")
            }
        }
    }
}

impl std::error::Error for EnsembleError {}

impl EnsembleConfig {
    /// Create a new ensemble configuration
    ///
    /// # Errors
    ///
    /// Returns `EnsembleError` if:
    /// - Fewer than 2 or more than 5 models
    /// - All model weights are zero (for ReliabilityWeighted)
    pub fn new(models: Vec<ModelSlot>, strategy: EnsembleStrategy) -> Result<Self, EnsembleError> {
        if models.len() < 2 {
            return Err(EnsembleError::TooFewModels {
                count: models.len(),
            });
        }
        if models.len() > 5 {
            return Err(EnsembleError::TooManyModels {
                count: models.len(),
            });
        }

        if strategy == EnsembleStrategy::ReliabilityWeighted
            && models.iter().all(|m| m.weight <= 0.0)
        {
            return Err(EnsembleError::AllZeroWeights);
        }

        Ok(Self { models, strategy })
    }

    /// Get the ensemble strategy
    pub fn strategy(&self) -> EnsembleStrategy {
        self.strategy
    }

    /// Get a reference to the models
    pub fn models(&self) -> &[ModelSlot] {
        &self.models
    }

    /// Number of models in the ensemble
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Select which model should handle a given sample
    ///
    /// # Arguments
    /// * `sample_index` - The 0-based index of the sample being collected
    /// * `k_margin` - The k-margin (used by CostAware to determine phase boundaries)
    ///
    /// # Returns
    /// Index into the models vec
    pub fn select_model_for_sample(&self, sample_index: usize, k_margin: Option<usize>) -> usize {
        match self.strategy {
            EnsembleStrategy::RoundRobin => self.select_round_robin(sample_index),
            EnsembleStrategy::CostAware => {
                self.select_cost_aware(sample_index, k_margin.unwrap_or(3))
            }
            EnsembleStrategy::ReliabilityWeighted => self.select_reliability_weighted(sample_index),
        }
    }

    /// Get the client for a given model index
    pub fn client_for_index(&self, index: usize) -> &Arc<dyn LlmClient> {
        &self.models[index].client
    }

    /// Get model name for a given index
    pub fn model_name_for_index(&self, index: usize) -> &str {
        self.models[index].model_name()
    }

    // --- Strategy Implementations ---

    /// Round-robin: distribute samples evenly
    fn select_round_robin(&self, sample_index: usize) -> usize {
        sample_index % self.models.len()
    }

    /// Cost-aware: cheap models first, escalate in phases
    ///
    /// Phase 1 (samples 0..k): cheapest tier only
    /// Phase 2 (samples k..2k): next cheapest tier
    /// Phase 3 (samples 2k..): most expensive tier
    fn select_cost_aware(&self, sample_index: usize, k_margin: usize) -> usize {
        // Sort model indices by cost tier
        let mut indices_by_cost: Vec<usize> = (0..self.models.len()).collect();
        indices_by_cost.sort_by_key(|&i| self.models[i].cost_tier.cost_order());

        // Group by tier
        let tiers = self.group_by_tier(&indices_by_cost);

        // Determine which tier based on sample_index and k_margin
        let phase = if sample_index < k_margin {
            0 // Cheapest tier
        } else if sample_index < k_margin * 2 {
            1 // Next tier (or cheapest if only 1 tier)
        } else {
            2 // Most expensive tier (or latest available)
        };

        let tier_index = phase.min(tiers.len() - 1);
        let tier_models = &tiers[tier_index];

        // Round-robin within the tier
        let within_tier_index = sample_index % tier_models.len();
        tier_models[within_tier_index]
    }

    /// Group model indices by cost tier (ascending)
    fn group_by_tier(&self, sorted_indices: &[usize]) -> Vec<Vec<usize>> {
        let mut tiers: Vec<Vec<usize>> = Vec::new();
        let mut current_tier: Option<CostTier> = None;

        for &idx in sorted_indices {
            let tier = self.models[idx].cost_tier;
            if current_tier != Some(tier) {
                tiers.push(Vec::new());
                current_tier = Some(tier);
            }
            tiers.last_mut().unwrap().push(idx);
        }

        tiers
    }

    /// Reliability-weighted: more samples from higher-weight models
    ///
    /// Uses Bresenham-style deficit tracking: each step assigns the sample
    /// to the model furthest below its target allocation fraction.
    fn select_reliability_weighted(&self, sample_index: usize) -> usize {
        let total_weight: f64 = self.models.iter().map(|m| m.weight).sum();
        if total_weight <= 0.0 {
            return self.select_round_robin(sample_index);
        }

        let fractions: Vec<f64> = self
            .models
            .iter()
            .map(|m| m.weight / total_weight)
            .collect();
        let mut counts = vec![0usize; self.models.len()];
        let mut selected = 0;

        for _ in 0..=sample_index {
            let total_so_far = counts.iter().sum::<usize>() as f64;
            let mut best_idx = 0;
            let mut best_deficit = f64::NEG_INFINITY;

            for (i, &frac) in fractions.iter().enumerate() {
                let target = frac * (total_so_far + 1.0);
                let deficit = target - counts[i] as f64;
                if deficit > best_deficit {
                    best_deficit = deficit;
                    best_idx = i;
                }
            }

            selected = best_idx;
            counts[best_idx] += 1;
        }

        selected
    }
}

impl std::fmt::Debug for EnsembleConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnsembleConfig")
            .field("models", &self.models)
            .field("strategy", &self.strategy)
            .finish()
    }
}

/// Serializable ensemble configuration for MCP requests
///
/// This mirrors `EnsembleConfig` but uses serializable types instead of
/// trait objects, suitable for JSON transport over MCP.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
pub struct EnsembleConfigRequest {
    /// Model definitions
    pub models: Vec<ModelSlotRequest>,
    /// Ensemble strategy
    pub strategy: EnsembleStrategy,
}

/// Serializable model slot for MCP requests
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
pub struct ModelSlotRequest {
    /// LLM provider name (e.g., "ollama", "openai", "anthropic")
    pub provider: String,
    /// Model name (e.g., "llama3", "gpt-4o-mini", "claude-haiku")
    pub model: String,
    /// Cost tier
    pub cost_tier: CostTier,
    /// Sampling weight (optional, default 1.0)
    #[serde(default = "default_weight")]
    pub weight: f64,
}

fn default_weight() -> f64 {
    1.0
}

impl EnsembleConfigRequest {
    /// Validate the ensemble configuration request
    pub fn validate(&self) -> Result<(), String> {
        if self.models.len() < 2 {
            return Err(format!(
                "ensemble requires at least 2 models, got {}",
                self.models.len()
            ));
        }
        if self.models.len() > 5 {
            return Err(format!(
                "ensemble supports at most 5 models, got {}",
                self.models.len()
            ));
        }
        for slot in &self.models {
            if slot.provider.is_empty() {
                return Err("model provider cannot be empty".to_string());
            }
            if slot.model.is_empty() {
                return Err("model name cannot be empty".to_string());
            }
        }
        Ok(())
    }
}

/// Metrics from an ensemble voting session
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, JsonSchema)]
pub struct EnsembleMetrics {
    /// Models that were actually used
    pub models_used: Vec<String>,
    /// Number of samples from each model
    pub samples_per_model: std::collections::HashMap<String, usize>,
    /// Number of cost-tier escalations (CostAware only)
    pub escalations: usize,
    /// Cost breakdown by model (USD)
    pub cost_per_model: std::collections::HashMap<String, f64>,
}

impl EnsembleMetrics {
    /// Record a sample from a model
    pub fn record_sample(&mut self, model_name: &str) {
        if !self.models_used.contains(&model_name.to_string()) {
            self.models_used.push(model_name.to_string());
        }
        *self
            .samples_per_model
            .entry(model_name.to_string())
            .or_insert(0) += 1;
    }

    /// Record cost for a model
    pub fn record_cost(&mut self, model_name: &str, cost_usd: f64) {
        *self
            .cost_per_model
            .entry(model_name.to_string())
            .or_insert(0.0) += cost_usd;
    }

    /// Record an escalation event
    pub fn record_escalation(&mut self) {
        self.escalations += 1;
    }

    /// Total cost across all models
    pub fn total_cost(&self) -> f64 {
        self.cost_per_model.values().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{LlmError, LlmResponse, TokenCost, TokenUsage};
    use std::future::Future;
    use std::pin::Pin;
    use std::time::Duration;

    /// Mock LLM client for ensemble testing
    struct MockEnsembleClient {
        name: String,
        cost: TokenCost,
    }

    impl MockEnsembleClient {
        fn into_client(name: &str, cost_tier: CostTier) -> Arc<dyn LlmClient> {
            let cost = match cost_tier {
                CostTier::Cheap => TokenCost::new(0.0001, 0.0002),
                CostTier::Medium => TokenCost::new(0.001, 0.002),
                CostTier::Expensive => TokenCost::new(0.01, 0.03),
            };
            Arc::new(Self {
                name: name.to_string(),
                cost,
            })
        }
    }

    impl LlmClient for MockEnsembleClient {
        fn generate(
            &self,
            _prompt: &str,
            _temperature: f64,
        ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, LlmError>> + Send + '_>> {
            let name = self.name.clone();
            Box::pin(async move {
                Ok(LlmResponse {
                    content: format!("response from {}", name),
                    tokens: TokenUsage::new(10, 20),
                    latency: Duration::from_millis(50),
                })
            })
        }

        fn model_name(&self) -> &str {
            &self.name
        }

        fn cost_per_1k_tokens(&self) -> TokenCost {
            self.cost
        }
    }

    fn make_2_model_ensemble(strategy: EnsembleStrategy) -> EnsembleConfig {
        EnsembleConfig::new(
            vec![
                ModelSlot::new(
                    MockEnsembleClient::into_client("cheap-model", CostTier::Cheap),
                    1.0,
                    CostTier::Cheap,
                ),
                ModelSlot::new(
                    MockEnsembleClient::into_client("expensive-model", CostTier::Expensive),
                    1.0,
                    CostTier::Expensive,
                ),
            ],
            strategy,
        )
        .unwrap()
    }

    fn make_3_tier_ensemble() -> EnsembleConfig {
        EnsembleConfig::new(
            vec![
                ModelSlot::new(
                    MockEnsembleClient::into_client("cheap", CostTier::Cheap),
                    1.0,
                    CostTier::Cheap,
                ),
                ModelSlot::new(
                    MockEnsembleClient::into_client("medium", CostTier::Medium),
                    2.0,
                    CostTier::Medium,
                ),
                ModelSlot::new(
                    MockEnsembleClient::into_client("expensive", CostTier::Expensive),
                    3.0,
                    CostTier::Expensive,
                ),
            ],
            EnsembleStrategy::CostAware,
        )
        .unwrap()
    }

    // ==========================================
    // EnsembleConfig Creation Tests
    // ==========================================

    #[test]
    fn test_create_2_model_ensemble() {
        let config = make_2_model_ensemble(EnsembleStrategy::RoundRobin);
        assert_eq!(config.model_count(), 2);
        assert_eq!(config.strategy(), EnsembleStrategy::RoundRobin);
    }

    #[test]
    fn test_create_5_model_ensemble() {
        let models: Vec<ModelSlot> = (0..5)
            .map(|i| {
                ModelSlot::new(
                    MockEnsembleClient::into_client(&format!("model-{}", i), CostTier::Cheap),
                    1.0,
                    CostTier::Cheap,
                )
            })
            .collect();

        let config = EnsembleConfig::new(models, EnsembleStrategy::RoundRobin);
        assert!(config.is_ok());
        assert_eq!(config.unwrap().model_count(), 5);
    }

    #[test]
    fn test_reject_single_model() {
        let models = vec![ModelSlot::new(
            MockEnsembleClient::into_client("solo", CostTier::Cheap),
            1.0,
            CostTier::Cheap,
        )];

        let result = EnsembleConfig::new(models, EnsembleStrategy::RoundRobin);
        assert_eq!(
            result.unwrap_err(),
            EnsembleError::TooFewModels { count: 1 }
        );
    }

    #[test]
    fn test_reject_6_models() {
        let models: Vec<ModelSlot> = (0..6)
            .map(|i| {
                ModelSlot::new(
                    MockEnsembleClient::into_client(&format!("model-{}", i), CostTier::Cheap),
                    1.0,
                    CostTier::Cheap,
                )
            })
            .collect();

        let result = EnsembleConfig::new(models, EnsembleStrategy::RoundRobin);
        assert_eq!(
            result.unwrap_err(),
            EnsembleError::TooManyModels { count: 6 }
        );
    }

    #[test]
    fn test_reject_all_zero_weights_reliability() {
        let models = vec![
            ModelSlot::new(
                MockEnsembleClient::into_client("a", CostTier::Cheap),
                0.0,
                CostTier::Cheap,
            ),
            ModelSlot::new(
                MockEnsembleClient::into_client("b", CostTier::Cheap),
                0.0,
                CostTier::Cheap,
            ),
        ];

        let result = EnsembleConfig::new(models, EnsembleStrategy::ReliabilityWeighted);
        assert_eq!(result.unwrap_err(), EnsembleError::AllZeroWeights);
    }

    #[test]
    fn test_zero_weights_ok_for_round_robin() {
        let models = vec![
            ModelSlot::new(
                MockEnsembleClient::into_client("a", CostTier::Cheap),
                0.0,
                CostTier::Cheap,
            ),
            ModelSlot::new(
                MockEnsembleClient::into_client("b", CostTier::Cheap),
                0.0,
                CostTier::Cheap,
            ),
        ];

        let result = EnsembleConfig::new(models, EnsembleStrategy::RoundRobin);
        assert!(result.is_ok());
    }

    // ==========================================
    // RoundRobin Strategy Tests
    // ==========================================

    #[test]
    fn test_round_robin_distributes_evenly() {
        let config = make_2_model_ensemble(EnsembleStrategy::RoundRobin);

        let mut counts = [0usize; 2];
        for i in 0..10 {
            let idx = config.select_model_for_sample(i, None);
            counts[idx] += 1;
        }

        assert_eq!(counts[0], 5);
        assert_eq!(counts[1], 5);
    }

    #[test]
    fn test_round_robin_with_3_models() {
        let models: Vec<ModelSlot> = (0..3)
            .map(|i| {
                ModelSlot::new(
                    MockEnsembleClient::into_client(&format!("m{}", i), CostTier::Cheap),
                    1.0,
                    CostTier::Cheap,
                )
            })
            .collect();

        let config = EnsembleConfig::new(models, EnsembleStrategy::RoundRobin).unwrap();

        assert_eq!(config.select_model_for_sample(0, None), 0);
        assert_eq!(config.select_model_for_sample(1, None), 1);
        assert_eq!(config.select_model_for_sample(2, None), 2);
        assert_eq!(config.select_model_for_sample(3, None), 0);
    }

    // ==========================================
    // CostAware Strategy Tests
    // ==========================================

    #[test]
    fn test_cost_aware_starts_with_cheapest() {
        let config = make_3_tier_ensemble();

        // Phase 1: samples 0..k (k=3) should use cheapest tier
        for i in 0..3 {
            let idx = config.select_model_for_sample(i, Some(3));
            assert_eq!(
                config.models()[idx].cost_tier,
                CostTier::Cheap,
                "Sample {} should use cheap tier",
                i
            );
        }
    }

    #[test]
    fn test_cost_aware_escalates_to_medium() {
        let config = make_3_tier_ensemble();

        // Phase 2: samples k..2k (3..6 with k=3) should use medium tier
        for i in 3..6 {
            let idx = config.select_model_for_sample(i, Some(3));
            assert_eq!(
                config.models()[idx].cost_tier,
                CostTier::Medium,
                "Sample {} should use medium tier",
                i
            );
        }
    }

    #[test]
    fn test_cost_aware_escalates_to_expensive() {
        let config = make_3_tier_ensemble();

        // Phase 3: samples >= 2k (>=6 with k=3) should use expensive tier
        for i in 6..10 {
            let idx = config.select_model_for_sample(i, Some(3));
            assert_eq!(
                config.models()[idx].cost_tier,
                CostTier::Expensive,
                "Sample {} should use expensive tier",
                i
            );
        }
    }

    #[test]
    fn test_cost_aware_2_tiers_only() {
        let config = make_2_model_ensemble(EnsembleStrategy::CostAware);

        // Phase 1: cheap
        assert_eq!(
            config.models()[config.select_model_for_sample(0, Some(3))].cost_tier,
            CostTier::Cheap
        );

        // Phase 2: expensive (no medium tier)
        assert_eq!(
            config.models()[config.select_model_for_sample(3, Some(3))].cost_tier,
            CostTier::Expensive
        );

        // Phase 3: still expensive (only 2 tiers)
        assert_eq!(
            config.models()[config.select_model_for_sample(6, Some(3))].cost_tier,
            CostTier::Expensive
        );
    }

    // ==========================================
    // ReliabilityWeighted Strategy Tests
    // ==========================================

    #[test]
    fn test_reliability_weighted_favors_high_weight() {
        let models = vec![
            ModelSlot::new(
                MockEnsembleClient::into_client("low-weight", CostTier::Cheap),
                1.0,
                CostTier::Cheap,
            ),
            ModelSlot::new(
                MockEnsembleClient::into_client("high-weight", CostTier::Expensive),
                9.0,
                CostTier::Expensive,
            ),
        ];

        let config = EnsembleConfig::new(models, EnsembleStrategy::ReliabilityWeighted).unwrap();

        let mut counts = [0usize; 2];
        for i in 0..100 {
            let idx = config.select_model_for_sample(i, None);
            counts[idx] += 1;
        }

        // High-weight model should get ~90% of samples
        assert!(
            counts[1] > counts[0],
            "High-weight model should get more samples: low={}, high={}",
            counts[0],
            counts[1]
        );
        assert!(
            counts[1] >= 80,
            "High-weight model should get >= 80% of 100 samples, got {}",
            counts[1]
        );
    }

    #[test]
    fn test_reliability_weighted_equal_weights() {
        let models = vec![
            ModelSlot::new(
                MockEnsembleClient::into_client("a", CostTier::Cheap),
                1.0,
                CostTier::Cheap,
            ),
            ModelSlot::new(
                MockEnsembleClient::into_client("b", CostTier::Expensive),
                1.0,
                CostTier::Expensive,
            ),
        ];

        let config = EnsembleConfig::new(models, EnsembleStrategy::ReliabilityWeighted).unwrap();

        let mut counts = [0usize; 2];
        for i in 0..100 {
            let idx = config.select_model_for_sample(i, None);
            counts[idx] += 1;
        }

        // Should be roughly 50/50
        assert!(
            (counts[0] as i32 - counts[1] as i32).unsigned_abs() < 20,
            "Equal weights should distribute roughly evenly: a={}, b={}",
            counts[0],
            counts[1]
        );
    }

    // ==========================================
    // ModelSlot Tests
    // ==========================================

    #[test]
    fn test_model_slot_name() {
        let slot = ModelSlot::new(
            MockEnsembleClient::into_client("test-model", CostTier::Medium),
            1.0,
            CostTier::Medium,
        );
        assert_eq!(slot.model_name(), "test-model");
    }

    #[test]
    fn test_model_slot_negative_weight_clamped() {
        let slot = ModelSlot::new(
            MockEnsembleClient::into_client("test", CostTier::Cheap),
            -5.0,
            CostTier::Cheap,
        );
        assert!((slot.weight - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_model_slot_debug() {
        let slot = ModelSlot::new(
            MockEnsembleClient::into_client("debug-test", CostTier::Cheap),
            2.5,
            CostTier::Cheap,
        );
        let debug = format!("{:?}", slot);
        assert!(debug.contains("debug-test"));
        assert!(debug.contains("2.5"));
    }

    // ==========================================
    // CostTier Tests
    // ==========================================

    #[test]
    fn test_cost_tier_ordering() {
        assert!(CostTier::Cheap.cost_order() < CostTier::Medium.cost_order());
        assert!(CostTier::Medium.cost_order() < CostTier::Expensive.cost_order());
    }

    #[test]
    fn test_cost_tier_serialization() {
        let json = serde_json::to_string(&CostTier::Medium).unwrap();
        assert_eq!(json, "\"medium\"");

        let tier: CostTier = serde_json::from_str("\"cheap\"").unwrap();
        assert_eq!(tier, CostTier::Cheap);
    }

    // ==========================================
    // EnsembleStrategy Tests
    // ==========================================

    #[test]
    fn test_strategy_serialization() {
        let json = serde_json::to_string(&EnsembleStrategy::CostAware).unwrap();
        assert_eq!(json, "\"cost_aware\"");

        let strategy: EnsembleStrategy = serde_json::from_str("\"round_robin\"").unwrap();
        assert_eq!(strategy, EnsembleStrategy::RoundRobin);
    }

    // ==========================================
    // EnsembleConfigRequest Tests
    // ==========================================

    #[test]
    fn test_config_request_serialization() {
        let request = EnsembleConfigRequest {
            models: vec![
                ModelSlotRequest {
                    provider: "ollama".to_string(),
                    model: "llama3".to_string(),
                    cost_tier: CostTier::Cheap,
                    weight: 1.0,
                },
                ModelSlotRequest {
                    provider: "anthropic".to_string(),
                    model: "claude-haiku".to_string(),
                    cost_tier: CostTier::Medium,
                    weight: 2.0,
                },
            ],
            strategy: EnsembleStrategy::CostAware,
        };

        let json = serde_json::to_string_pretty(&request).unwrap();
        assert!(json.contains("ollama"));
        assert!(json.contains("claude-haiku"));
        assert!(json.contains("cost_aware"));

        let parsed: EnsembleConfigRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, request);
    }

    #[test]
    fn test_config_request_validation() {
        let valid = EnsembleConfigRequest {
            models: vec![
                ModelSlotRequest {
                    provider: "ollama".to_string(),
                    model: "llama3".to_string(),
                    cost_tier: CostTier::Cheap,
                    weight: 1.0,
                },
                ModelSlotRequest {
                    provider: "openai".to_string(),
                    model: "gpt-4o-mini".to_string(),
                    cost_tier: CostTier::Medium,
                    weight: 1.0,
                },
            ],
            strategy: EnsembleStrategy::RoundRobin,
        };
        assert!(valid.validate().is_ok());

        let too_few = EnsembleConfigRequest {
            models: vec![ModelSlotRequest {
                provider: "ollama".to_string(),
                model: "llama3".to_string(),
                cost_tier: CostTier::Cheap,
                weight: 1.0,
            }],
            strategy: EnsembleStrategy::RoundRobin,
        };
        assert!(too_few.validate().is_err());
    }

    #[test]
    fn test_config_request_empty_provider_rejected() {
        let request = EnsembleConfigRequest {
            models: vec![
                ModelSlotRequest {
                    provider: "".to_string(),
                    model: "llama3".to_string(),
                    cost_tier: CostTier::Cheap,
                    weight: 1.0,
                },
                ModelSlotRequest {
                    provider: "openai".to_string(),
                    model: "gpt-4".to_string(),
                    cost_tier: CostTier::Expensive,
                    weight: 1.0,
                },
            ],
            strategy: EnsembleStrategy::RoundRobin,
        };
        assert!(request.validate().is_err());
    }

    // ==========================================
    // EnsembleMetrics Tests
    // ==========================================

    #[test]
    fn test_ensemble_metrics_tracking() {
        let mut metrics = EnsembleMetrics::default();

        metrics.record_sample("model-a");
        metrics.record_sample("model-a");
        metrics.record_sample("model-b");
        metrics.record_cost("model-a", 0.001);
        metrics.record_cost("model-b", 0.003);
        metrics.record_escalation();

        assert_eq!(metrics.models_used, vec!["model-a", "model-b"]);
        assert_eq!(metrics.samples_per_model["model-a"], 2);
        assert_eq!(metrics.samples_per_model["model-b"], 1);
        assert_eq!(metrics.escalations, 1);
        assert!((metrics.total_cost() - 0.004).abs() < 1e-10);
    }

    #[test]
    fn test_ensemble_metrics_serialization() {
        let mut metrics = EnsembleMetrics::default();
        metrics.record_sample("test-model");
        metrics.record_cost("test-model", 0.01);

        let json = serde_json::to_string(&metrics).unwrap();
        assert!(json.contains("test-model"));

        let parsed: EnsembleMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.models_used, vec!["test-model"]);
    }

    // ==========================================
    // EnsembleError Tests
    // ==========================================

    #[test]
    fn test_ensemble_error_display() {
        assert!(EnsembleError::TooFewModels { count: 1 }
            .to_string()
            .contains("at least 2"));
        assert!(EnsembleError::TooManyModels { count: 6 }
            .to_string()
            .contains("at most 5"));
        assert!(EnsembleError::AllZeroWeights
            .to_string()
            .contains("weight > 0"));
    }

    #[test]
    fn test_ensemble_error_is_error_trait() {
        let err: Box<dyn std::error::Error> = Box::new(EnsembleError::TooFewModels { count: 0 });
        assert!(err.to_string().contains("at least 2"));
    }

    // ==========================================
    // Single-model mode unchanged
    // ==========================================

    #[test]
    fn test_single_model_rejected_for_ensemble() {
        // Verifies that single-model mode (no ensemble) requires going through
        // the normal vote_with_margin path, not EnsembleConfig
        let result = EnsembleConfig::new(
            vec![ModelSlot::new(
                MockEnsembleClient::into_client("solo", CostTier::Cheap),
                1.0,
                CostTier::Cheap,
            )],
            EnsembleStrategy::RoundRobin,
        );
        assert!(result.is_err());
    }

    // ==========================================
    // Access helpers
    // ==========================================

    #[test]
    fn test_client_for_index() {
        let config = make_2_model_ensemble(EnsembleStrategy::RoundRobin);
        assert_eq!(config.client_for_index(0).model_name(), "cheap-model");
        assert_eq!(config.client_for_index(1).model_name(), "expensive-model");
    }

    #[test]
    fn test_model_name_for_index() {
        let config = make_2_model_ensemble(EnsembleStrategy::RoundRobin);
        assert_eq!(config.model_name_for_index(0), "cheap-model");
        assert_eq!(config.model_name_for_index(1), "expensive-model");
    }

    #[test]
    fn test_ensemble_config_debug() {
        let config = make_2_model_ensemble(EnsembleStrategy::RoundRobin);
        let debug = format!("{:?}", config);
        assert!(debug.contains("EnsembleConfig"));
        assert!(debug.contains("RoundRobin"));
    }
}
