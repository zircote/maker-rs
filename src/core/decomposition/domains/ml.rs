//! ML Pipeline Decomposer
//!
//! Decomposes machine learning tasks following standard pipeline patterns:
//! DataPrep → Config → Training → Evaluation
//!
//! # Features
//!
//! - **Pipeline stages**: Standard ML workflow decomposition
//! - **Hyperparameter search**: Parallel composition for grid/random search
//! - **Validation red-flags**: NaN/infinity detection, metric range validation
//! - **Cross-validation support**: K-fold decomposition strategies
//!
//! # Example
//!
//! ```ignore
//! use maker::core::decomposition::domains::MLPipelineDecomposer;
//!
//! let decomposer = MLPipelineDecomposer::new()
//!     .with_cross_validation(5);
//!
//! let proposal = decomposer.propose_decomposition(
//!     "train-model",
//!     "Train a classification model",
//!     &context,
//!     0,
//! )?;
//! ```

use crate::core::decomposition::{
    CompositionFunction, DecompositionAgent, DecompositionError, DecompositionProposal,
    DecompositionSubtask,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

/// Stages in an ML pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "snake_case")]
pub enum PipelineStage {
    /// Data loading and preparation
    DataPrep,
    /// Feature engineering and selection
    FeatureEngineering,
    /// Model configuration and hyperparameters
    ModelConfig,
    /// Model training
    Training,
    /// Model evaluation and metrics
    Evaluation,
    /// Model validation (cross-validation folds)
    Validation,
    /// Hyperparameter search iteration
    HyperparameterSearch,
    /// Model deployment preparation
    Deployment,
}

impl PipelineStage {
    /// Get the typical order of this stage in a pipeline
    pub fn order(&self) -> usize {
        match self {
            PipelineStage::DataPrep => 0,
            PipelineStage::FeatureEngineering => 1,
            PipelineStage::ModelConfig => 2,
            PipelineStage::Training => 3,
            PipelineStage::Evaluation => 4,
            PipelineStage::Validation => 5,
            PipelineStage::HyperparameterSearch => 3, // Parallel with training
            PipelineStage::Deployment => 6,
        }
    }

    /// Get a human-readable description of the stage
    pub fn description(&self) -> &str {
        match self {
            PipelineStage::DataPrep => "Load, clean, and prepare training data",
            PipelineStage::FeatureEngineering => "Engineer and select features",
            PipelineStage::ModelConfig => "Configure model architecture and hyperparameters",
            PipelineStage::Training => "Train the model on prepared data",
            PipelineStage::Evaluation => "Evaluate model performance with metrics",
            PipelineStage::Validation => "Validate model with cross-validation",
            PipelineStage::HyperparameterSearch => "Search for optimal hyperparameters",
            PipelineStage::Deployment => "Prepare model for deployment",
        }
    }
}

impl std::fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineStage::DataPrep => write!(f, "data_prep"),
            PipelineStage::FeatureEngineering => write!(f, "feature_engineering"),
            PipelineStage::ModelConfig => write!(f, "model_config"),
            PipelineStage::Training => write!(f, "training"),
            PipelineStage::Evaluation => write!(f, "evaluation"),
            PipelineStage::Validation => write!(f, "validation"),
            PipelineStage::HyperparameterSearch => write!(f, "hyperparam_search"),
            PipelineStage::Deployment => write!(f, "deployment"),
        }
    }
}

/// Configuration for hyperparameter search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterSearchConfig {
    /// Number of search iterations
    pub iterations: usize,

    /// Search strategy
    pub strategy: SearchStrategy,

    /// Parameters to search
    pub parameters: HashMap<String, ParameterRange>,
}

impl Default for HyperparameterSearchConfig {
    fn default() -> Self {
        Self {
            iterations: 10,
            strategy: SearchStrategy::Random,
            parameters: HashMap::new(),
        }
    }
}

/// Search strategy for hyperparameters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SearchStrategy {
    /// Grid search over all combinations
    Grid,
    /// Random sampling from parameter space
    #[default]
    Random,
    /// Bayesian optimization
    Bayesian,
}

/// Range specification for a hyperparameter
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ParameterRange {
    /// Continuous range
    Continuous { min: f64, max: f64 },
    /// Discrete integer range
    Discrete { min: i64, max: i64 },
    /// Categorical choices
    Categorical { choices: Vec<String> },
    /// Log-scale continuous range
    LogScale { min: f64, max: f64 },
}

/// Validation result for ML metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricValidation {
    /// Whether all metrics are valid
    pub is_valid: bool,

    /// List of validation issues
    pub issues: Vec<MetricIssue>,
}

/// An issue found during metric validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricIssue {
    /// Name of the metric
    pub metric_name: String,

    /// Issue type
    pub issue_type: MetricIssueType,

    /// Value that caused the issue
    pub value: Option<f64>,

    /// Human-readable message
    pub message: String,
}

/// Types of metric validation issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricIssueType {
    /// Value is NaN
    NaN,
    /// Value is infinite
    Infinite,
    /// Value is out of expected range
    OutOfRange,
    /// Value is missing
    Missing,
    /// Value decreased when it should increase (or vice versa)
    WrongDirection,
}

/// ML Pipeline Decomposer
///
/// Decomposes machine learning tasks into standard pipeline stages,
/// with support for cross-validation and hyperparameter search.
#[derive(Debug, Clone)]
pub struct MLPipelineDecomposer {
    /// Stages to include in decomposition
    stages: Vec<PipelineStage>,

    /// Number of cross-validation folds (0 = no CV)
    cv_folds: usize,

    /// Hyperparameter search configuration
    hyperparam_config: Option<HyperparameterSearchConfig>,

    /// Whether to include deployment stage
    include_deployment: bool,

    /// Expected metric ranges for validation
    metric_ranges: HashMap<String, (f64, f64)>,
}

impl Default for MLPipelineDecomposer {
    fn default() -> Self {
        Self::new()
    }
}

impl MLPipelineDecomposer {
    /// Create a new MLPipelineDecomposer with standard stages
    pub fn new() -> Self {
        Self {
            stages: vec![
                PipelineStage::DataPrep,
                PipelineStage::FeatureEngineering,
                PipelineStage::ModelConfig,
                PipelineStage::Training,
                PipelineStage::Evaluation,
            ],
            cv_folds: 0,
            hyperparam_config: None,
            include_deployment: false,
            metric_ranges: Self::default_metric_ranges(),
        }
    }

    /// Create a minimal decomposer (DataPrep → Training → Evaluation)
    pub fn minimal() -> Self {
        Self {
            stages: vec![
                PipelineStage::DataPrep,
                PipelineStage::Training,
                PipelineStage::Evaluation,
            ],
            cv_folds: 0,
            hyperparam_config: None,
            include_deployment: false,
            metric_ranges: Self::default_metric_ranges(),
        }
    }

    /// Set cross-validation folds
    pub fn with_cross_validation(mut self, folds: usize) -> Self {
        self.cv_folds = folds;
        if folds > 0 && !self.stages.contains(&PipelineStage::Validation) {
            self.stages.push(PipelineStage::Validation);
        }
        self
    }

    /// Set hyperparameter search configuration
    pub fn with_hyperparameter_search(mut self, config: HyperparameterSearchConfig) -> Self {
        self.hyperparam_config = Some(config);
        if !self.stages.contains(&PipelineStage::HyperparameterSearch) {
            // Insert before training
            if let Some(pos) = self
                .stages
                .iter()
                .position(|s| *s == PipelineStage::Training)
            {
                self.stages.insert(pos, PipelineStage::HyperparameterSearch);
            } else {
                self.stages.push(PipelineStage::HyperparameterSearch);
            }
        }
        self
    }

    /// Include deployment stage
    pub fn with_deployment(mut self) -> Self {
        self.include_deployment = true;
        if !self.stages.contains(&PipelineStage::Deployment) {
            self.stages.push(PipelineStage::Deployment);
        }
        self
    }

    /// Set custom stages
    pub fn with_stages(mut self, stages: Vec<PipelineStage>) -> Self {
        self.stages = stages;
        self
    }

    /// Add expected metric range for validation
    pub fn with_metric_range(mut self, metric: impl Into<String>, min: f64, max: f64) -> Self {
        self.metric_ranges.insert(metric.into(), (min, max));
        self
    }

    /// Default metric ranges for common ML metrics
    fn default_metric_ranges() -> HashMap<String, (f64, f64)> {
        let mut ranges = HashMap::new();
        ranges.insert("accuracy".to_string(), (0.0, 1.0));
        ranges.insert("precision".to_string(), (0.0, 1.0));
        ranges.insert("recall".to_string(), (0.0, 1.0));
        ranges.insert("f1".to_string(), (0.0, 1.0));
        ranges.insert("f1_score".to_string(), (0.0, 1.0));
        ranges.insert("auc".to_string(), (0.0, 1.0));
        ranges.insert("roc_auc".to_string(), (0.0, 1.0));
        ranges.insert("loss".to_string(), (0.0, f64::INFINITY));
        ranges.insert("mse".to_string(), (0.0, f64::INFINITY));
        ranges.insert("mae".to_string(), (0.0, f64::INFINITY));
        ranges.insert("rmse".to_string(), (0.0, f64::INFINITY));
        ranges.insert("r2".to_string(), (f64::NEG_INFINITY, 1.0));
        ranges
    }

    /// Validate metrics for NaN, infinity, and range issues
    pub fn validate_metrics(&self, metrics: &HashMap<String, f64>) -> MetricValidation {
        let mut issues = Vec::new();

        for (name, value) in metrics {
            // Check for NaN
            if value.is_nan() {
                issues.push(MetricIssue {
                    metric_name: name.clone(),
                    issue_type: MetricIssueType::NaN,
                    value: None,
                    message: format!("Metric '{}' is NaN", name),
                });
                continue;
            }

            // Check for infinity
            if value.is_infinite() {
                issues.push(MetricIssue {
                    metric_name: name.clone(),
                    issue_type: MetricIssueType::Infinite,
                    value: Some(*value),
                    message: format!("Metric '{}' is infinite", name),
                });
                continue;
            }

            // Check range
            if let Some((min, max)) = self.metric_ranges.get(name) {
                if *value < *min || *value > *max {
                    issues.push(MetricIssue {
                        metric_name: name.clone(),
                        issue_type: MetricIssueType::OutOfRange,
                        value: Some(*value),
                        message: format!(
                            "Metric '{}' value {} is outside expected range [{}, {}]",
                            name, value, min, max
                        ),
                    });
                }
            }
        }

        MetricValidation {
            is_valid: issues.is_empty(),
            issues,
        }
    }

    /// Create subtask for a pipeline stage
    fn create_stage_subtask(
        &self,
        parent_id: &str,
        stage: PipelineStage,
        order: usize,
        context: &serde_json::Value,
    ) -> DecompositionSubtask {
        let task_id = format!("{}-{}", parent_id, stage);

        DecompositionSubtask::leaf(&task_id, stage.description())
            .with_parent(parent_id)
            .with_order(order)
            .with_context(json!({
                "stage": stage.to_string(),
                "stage_description": stage.description(),
                "input_context": context,
            }))
            .with_metadata("domain", json!("ml"))
            .with_metadata("stage", json!(stage.to_string()))
    }

    /// Create subtasks for hyperparameter search iterations
    fn create_hyperparam_subtasks(
        &self,
        parent_id: &str,
        config: &HyperparameterSearchConfig,
        base_order: usize,
    ) -> Vec<DecompositionSubtask> {
        (0..config.iterations)
            .map(|i| {
                let task_id = format!("{}-hyperparam-iter-{}", parent_id, i);

                DecompositionSubtask::leaf(
                    &task_id,
                    format!("Hyperparameter search iteration {}", i + 1),
                )
                .with_parent(parent_id)
                .with_order(base_order + i)
                .with_context(json!({
                    "iteration": i,
                    "total_iterations": config.iterations,
                    "strategy": format!("{:?}", config.strategy),
                    "parameters": config.parameters,
                }))
                .with_metadata("domain", json!("ml"))
                .with_metadata("stage", json!("hyperparam_search"))
            })
            .collect()
    }

    /// Create subtasks for cross-validation folds
    fn create_cv_subtasks(&self, parent_id: &str, base_order: usize) -> Vec<DecompositionSubtask> {
        (0..self.cv_folds)
            .map(|fold| {
                let task_id = format!("{}-cv-fold-{}", parent_id, fold);

                DecompositionSubtask::leaf(
                    &task_id,
                    format!("Cross-validation fold {}/{}", fold + 1, self.cv_folds),
                )
                .with_parent(parent_id)
                .with_order(base_order + fold)
                .with_context(json!({
                    "fold": fold,
                    "total_folds": self.cv_folds,
                }))
                .with_metadata("domain", json!("ml"))
                .with_metadata("stage", json!("validation"))
            })
            .collect()
    }
}

impl DecompositionAgent for MLPipelineDecomposer {
    fn propose_decomposition(
        &self,
        task_id: &str,
        description: &str,
        context: &serde_json::Value,
        _depth: usize,
    ) -> Result<DecompositionProposal, DecompositionError> {
        let mut subtasks = Vec::new();
        let mut order = 0;

        // Check for metrics in context and validate them
        if let Some(metrics_val) = context.get("metrics") {
            if let Ok(metrics) = serde_json::from_value::<HashMap<String, f64>>(metrics_val.clone())
            {
                let validation = self.validate_metrics(&metrics);
                if !validation.is_valid {
                    let issues: Vec<String> = validation
                        .issues
                        .iter()
                        .map(|i| i.message.clone())
                        .collect();
                    return Err(DecompositionError::ValidationError {
                        message: format!("Metric validation failed: {}", issues.join("; ")),
                    });
                }
            }
        }

        // Create subtasks for each stage
        for stage in &self.stages {
            match stage {
                PipelineStage::HyperparameterSearch => {
                    // Create parallel hyperparameter search subtasks
                    if let Some(config) = &self.hyperparam_config {
                        let hp_subtasks = self.create_hyperparam_subtasks(task_id, config, order);
                        order += hp_subtasks.len();
                        subtasks.extend(hp_subtasks);
                    }
                }
                PipelineStage::Validation => {
                    // Create parallel CV fold subtasks
                    if self.cv_folds > 0 {
                        let cv_subtasks = self.create_cv_subtasks(task_id, order);
                        order += cv_subtasks.len();
                        subtasks.extend(cv_subtasks);
                    } else {
                        // Single validation subtask
                        subtasks.push(self.create_stage_subtask(task_id, *stage, order, context));
                        order += 1;
                    }
                }
                _ => {
                    subtasks.push(self.create_stage_subtask(task_id, *stage, order, context));
                    order += 1;
                }
            }
        }

        // Determine composition function
        // Use Parallel for hyperparameter search and CV, Sequential otherwise
        let has_parallel = self.hyperparam_config.is_some() || self.cv_folds > 1;

        let composition_fn = if has_parallel {
            // Mixed: some stages are parallel
            CompositionFunction::Custom {
                name: "ml_pipeline".to_string(),
                params: {
                    let mut p = HashMap::new();
                    p.insert("cv_folds".to_string(), json!(self.cv_folds));
                    p.insert(
                        "has_hyperparam_search".to_string(),
                        json!(self.hyperparam_config.is_some()),
                    );
                    p
                },
            }
        } else {
            CompositionFunction::Sequential
        };

        let mut metadata = HashMap::new();
        metadata.insert("domain".to_string(), json!("ml"));
        metadata.insert(
            "stages".to_string(),
            json!(self
                .stages
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()),
        );
        metadata.insert("cv_folds".to_string(), json!(self.cv_folds));
        metadata.insert(
            "has_hyperparam_search".to_string(),
            json!(self.hyperparam_config.is_some()),
        );

        Ok(DecompositionProposal {
            proposal_id: format!("ml-pipeline-{}", task_id),
            source_task_id: task_id.to_string(),
            subtasks,
            composition_fn,
            confidence: 0.85,
            rationale: Some(format!(
                "ML pipeline decomposition: {} with {} stages",
                description,
                self.stages.len()
            )),
            metadata,
        })
    }

    fn is_atomic(&self, _task_id: &str, description: &str) -> bool {
        // Consider atomic if it's a single-stage operation
        let atomic_hints = [
            "load data",
            "save model",
            "compute metric",
            "set parameter",
            "initialize",
        ];

        let desc_lower = description.to_lowercase();
        atomic_hints.iter().any(|h| desc_lower.contains(h))
    }

    fn name(&self) -> &str {
        "ml_pipeline"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================
    // PipelineStage Tests
    // ==========================================

    #[test]
    fn test_stage_order() {
        assert!(PipelineStage::DataPrep.order() < PipelineStage::Training.order());
        assert!(PipelineStage::Training.order() < PipelineStage::Evaluation.order());
    }

    #[test]
    fn test_stage_description() {
        assert!(!PipelineStage::DataPrep.description().is_empty());
        assert!(!PipelineStage::Training.description().is_empty());
    }

    #[test]
    fn test_stage_display() {
        assert_eq!(PipelineStage::DataPrep.to_string(), "data_prep");
        assert_eq!(PipelineStage::Training.to_string(), "training");
    }

    #[test]
    fn test_stage_serialization() {
        let stages = vec![
            PipelineStage::DataPrep,
            PipelineStage::Training,
            PipelineStage::Evaluation,
        ];

        for stage in stages {
            let json = serde_json::to_string(&stage).unwrap();
            let parsed: PipelineStage = serde_json::from_str(&json).unwrap();
            assert_eq!(stage, parsed);
        }
    }

    // ==========================================
    // MLPipelineDecomposer Construction Tests
    // ==========================================

    #[test]
    fn test_new_decomposer() {
        let decomposer = MLPipelineDecomposer::new();
        assert_eq!(decomposer.stages.len(), 5); // Standard stages
        assert_eq!(decomposer.cv_folds, 0);
    }

    #[test]
    fn test_minimal_decomposer() {
        let decomposer = MLPipelineDecomposer::minimal();
        assert_eq!(decomposer.stages.len(), 3);
        assert!(decomposer.stages.contains(&PipelineStage::DataPrep));
        assert!(decomposer.stages.contains(&PipelineStage::Training));
        assert!(decomposer.stages.contains(&PipelineStage::Evaluation));
    }

    #[test]
    fn test_with_cross_validation() {
        let decomposer = MLPipelineDecomposer::new().with_cross_validation(5);
        assert_eq!(decomposer.cv_folds, 5);
        assert!(decomposer.stages.contains(&PipelineStage::Validation));
    }

    #[test]
    fn test_with_deployment() {
        let decomposer = MLPipelineDecomposer::new().with_deployment();
        assert!(decomposer.include_deployment);
        assert!(decomposer.stages.contains(&PipelineStage::Deployment));
    }

    // ==========================================
    // Metric Validation Tests
    // ==========================================

    #[test]
    fn test_validate_valid_metrics() {
        let decomposer = MLPipelineDecomposer::new();
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.95);
        metrics.insert("f1".to_string(), 0.92);

        let result = decomposer.validate_metrics(&metrics);
        assert!(result.is_valid);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn test_validate_nan_metric() {
        let decomposer = MLPipelineDecomposer::new();
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), f64::NAN);

        let result = decomposer.validate_metrics(&metrics);
        assert!(!result.is_valid);
        assert_eq!(result.issues.len(), 1);
        assert_eq!(result.issues[0].issue_type, MetricIssueType::NaN);
    }

    #[test]
    fn test_validate_infinite_metric() {
        let decomposer = MLPipelineDecomposer::new();
        let mut metrics = HashMap::new();
        metrics.insert("loss".to_string(), f64::INFINITY);

        let _result = decomposer.validate_metrics(&metrics);
        // Loss can be infinity in our default ranges
        // But let's test with accuracy which has [0, 1] range
        let mut metrics2 = HashMap::new();
        metrics2.insert("accuracy".to_string(), f64::INFINITY);

        let result2 = decomposer.validate_metrics(&metrics2);
        assert!(!result2.is_valid);
        assert_eq!(result2.issues[0].issue_type, MetricIssueType::Infinite);
    }

    #[test]
    fn test_validate_out_of_range_metric() {
        let decomposer = MLPipelineDecomposer::new();
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 1.5); // Should be [0, 1]

        let result = decomposer.validate_metrics(&metrics);
        assert!(!result.is_valid);
        assert_eq!(result.issues[0].issue_type, MetricIssueType::OutOfRange);
    }

    // ==========================================
    // Decomposition Tests
    // ==========================================

    #[test]
    fn test_basic_decomposition() {
        let decomposer = MLPipelineDecomposer::new();

        let result = decomposer.propose_decomposition(
            "train-model",
            "Train a classification model",
            &json!({"data_path": "/data/train.csv"}),
            0,
        );

        assert!(result.is_ok());
        let proposal = result.unwrap();

        // Should have subtasks for each stage
        assert_eq!(proposal.subtasks.len(), 5);

        // All should be leaves with m=1
        for subtask in &proposal.subtasks {
            assert!(subtask.is_leaf);
            assert_eq!(subtask.m_value, 1);
        }

        // Should use Sequential composition
        assert!(matches!(
            proposal.composition_fn,
            CompositionFunction::Sequential
        ));
    }

    #[test]
    fn test_cv_decomposition() {
        let decomposer = MLPipelineDecomposer::new().with_cross_validation(5);

        let result = decomposer.propose_decomposition(
            "train-cv",
            "Train with cross-validation",
            &json!({}),
            0,
        );

        assert!(result.is_ok());
        let proposal = result.unwrap();

        // Should have 5 CV fold subtasks
        let cv_subtasks: Vec<_> = proposal
            .subtasks
            .iter()
            .filter(|s| s.task_id.contains("cv-fold"))
            .collect();
        assert_eq!(cv_subtasks.len(), 5);
    }

    #[test]
    fn test_hyperparam_search_decomposition() {
        let config = HyperparameterSearchConfig {
            iterations: 10,
            strategy: SearchStrategy::Random,
            parameters: HashMap::new(),
        };

        let decomposer = MLPipelineDecomposer::new().with_hyperparameter_search(config);

        let result = decomposer.propose_decomposition(
            "train-hp",
            "Train with hyperparameter search",
            &json!({}),
            0,
        );

        assert!(result.is_ok());
        let proposal = result.unwrap();

        // Should have 10 hyperparameter search subtasks
        let hp_subtasks: Vec<_> = proposal
            .subtasks
            .iter()
            .filter(|s| s.task_id.contains("hyperparam"))
            .collect();
        assert_eq!(hp_subtasks.len(), 10);
    }

    #[test]
    fn test_metric_validation_in_decomposition() {
        let decomposer = MLPipelineDecomposer::new();

        // Out-of-range metrics should cause decomposition to fail
        // (Note: NaN doesn't serialize to JSON, so we use out-of-range instead)
        let result = decomposer.propose_decomposition(
            "train-invalid",
            "Train with invalid metrics",
            &json!({
                "metrics": {
                    "accuracy": 1.5
                }
            }),
            0,
        );

        assert!(result.is_err());
        assert!(matches!(
            result,
            Err(DecompositionError::ValidationError { .. })
        ));
    }

    // ==========================================
    // DecompositionAgent Trait Tests
    // ==========================================

    #[test]
    fn test_agent_name() {
        let decomposer = MLPipelineDecomposer::new();
        assert_eq!(decomposer.name(), "ml_pipeline");
    }

    #[test]
    fn test_is_atomic() {
        let decomposer = MLPipelineDecomposer::new();

        assert!(decomposer.is_atomic("task-1", "Load data from CSV"));
        assert!(decomposer.is_atomic("task-2", "Save model to disk"));
        assert!(decomposer.is_atomic("task-3", "Compute metric accuracy"));
        assert!(!decomposer.is_atomic("task-4", "Train the full pipeline"));
    }

    // ==========================================
    // m=1 Enforcement Tests
    // ==========================================

    #[test]
    fn test_all_subtasks_have_m1() {
        let decomposer = MLPipelineDecomposer::new()
            .with_cross_validation(3)
            .with_deployment();

        let result = decomposer.propose_decomposition(
            "full-pipeline",
            "Run full ML pipeline",
            &json!({}),
            0,
        );

        assert!(result.is_ok());
        let proposal = result.unwrap();

        for subtask in &proposal.subtasks {
            assert!(
                subtask.is_leaf,
                "Subtask {} should be a leaf",
                subtask.task_id
            );
            assert_eq!(
                subtask.m_value, 1,
                "Subtask {} should have m_value=1",
                subtask.task_id
            );
            assert!(subtask.validate().is_ok());
        }

        assert!(proposal.validate().is_ok());
    }

    // ==========================================
    // Serialization Tests
    // ==========================================

    #[test]
    fn test_search_strategy_serialization() {
        let strategies = vec![
            SearchStrategy::Grid,
            SearchStrategy::Random,
            SearchStrategy::Bayesian,
        ];

        for strategy in strategies {
            let json = serde_json::to_string(&strategy).unwrap();
            let parsed: SearchStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(strategy, parsed);
        }
    }

    #[test]
    fn test_parameter_range_serialization() {
        let ranges = vec![
            ParameterRange::Continuous { min: 0.0, max: 1.0 },
            ParameterRange::Discrete { min: 1, max: 100 },
            ParameterRange::Categorical {
                choices: vec!["a".to_string(), "b".to_string()],
            },
            ParameterRange::LogScale {
                min: 0.001,
                max: 1.0,
            },
        ];

        for range in ranges {
            let json = serde_json::to_string(&range).unwrap();
            let _parsed: ParameterRange = serde_json::from_str(&json).unwrap();
            // ParameterRange doesn't derive PartialEq, so just check it parses
        }
    }

    #[test]
    fn test_metric_issue_serialization() {
        let issue = MetricIssue {
            metric_name: "accuracy".to_string(),
            issue_type: MetricIssueType::NaN,
            value: None,
            message: "Metric is NaN".to_string(),
        };

        let json = serde_json::to_string(&issue).unwrap();
        let parsed: MetricIssue = serde_json::from_str(&json).unwrap();

        assert_eq!(issue.metric_name, parsed.metric_name);
        assert_eq!(issue.issue_type, parsed.issue_type);
    }
}
