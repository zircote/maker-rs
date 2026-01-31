//! Data Analysis Decomposer
//!
//! Decomposes data analysis tasks following ETL patterns:
//! Extract → Transform → Load → Validate
//!
//! # Features
//!
//! - **ETL stages**: Standard data pipeline decomposition
//! - **Schema inference**: Automatic schema detection and validation
//! - **Type coercion red-flags**: Detect unsafe type conversions
//! - **Null handling**: Explicit strategies for missing data
//!
//! # Example
//!
//! ```ignore
//! use maker::core::decomposition::domains::DataAnalysisDecomposer;
//!
//! let decomposer = DataAnalysisDecomposer::new()
//!     .with_schema_validation(true);
//!
//! let proposal = decomposer.propose_decomposition(
//!     "etl-pipeline",
//!     "Process customer data",
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

/// Stages in an ETL pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "snake_case")]
pub enum EtlStage {
    /// Data extraction from sources
    Extract,
    /// Schema inference and validation
    SchemaInference,
    /// Data cleaning and preprocessing
    Clean,
    /// Data transformation
    Transform,
    /// Data aggregation
    Aggregate,
    /// Data enrichment (joins, lookups)
    Enrich,
    /// Data validation
    Validate,
    /// Data loading to destination
    Load,
    /// Quality checks
    QualityCheck,
}

impl EtlStage {
    /// Get the typical order of this stage
    pub fn order(&self) -> usize {
        match self {
            EtlStage::Extract => 0,
            EtlStage::SchemaInference => 1,
            EtlStage::Clean => 2,
            EtlStage::Transform => 3,
            EtlStage::Aggregate => 4,
            EtlStage::Enrich => 5,
            EtlStage::Validate => 6,
            EtlStage::Load => 7,
            EtlStage::QualityCheck => 8,
        }
    }

    /// Get a human-readable description
    pub fn description(&self) -> &str {
        match self {
            EtlStage::Extract => "Extract data from source systems",
            EtlStage::SchemaInference => "Infer and validate data schema",
            EtlStage::Clean => "Clean and preprocess data",
            EtlStage::Transform => "Apply transformations to data",
            EtlStage::Aggregate => "Aggregate and summarize data",
            EtlStage::Enrich => "Enrich data with joins and lookups",
            EtlStage::Validate => "Validate transformed data",
            EtlStage::Load => "Load data to destination",
            EtlStage::QualityCheck => "Run data quality checks",
        }
    }
}

impl std::fmt::Display for EtlStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EtlStage::Extract => write!(f, "extract"),
            EtlStage::SchemaInference => write!(f, "schema_inference"),
            EtlStage::Clean => write!(f, "clean"),
            EtlStage::Transform => write!(f, "transform"),
            EtlStage::Aggregate => write!(f, "aggregate"),
            EtlStage::Enrich => write!(f, "enrich"),
            EtlStage::Validate => write!(f, "validate"),
            EtlStage::Load => write!(f, "load"),
            EtlStage::QualityCheck => write!(f, "quality_check"),
        }
    }
}

/// Data types for schema inference
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataType {
    String,
    Integer,
    Float,
    Boolean,
    Date,
    DateTime,
    Null,
    Array(Box<DataType>),
    Object,
    Unknown,
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataType::String => write!(f, "string"),
            DataType::Integer => write!(f, "integer"),
            DataType::Float => write!(f, "float"),
            DataType::Boolean => write!(f, "boolean"),
            DataType::Date => write!(f, "date"),
            DataType::DateTime => write!(f, "datetime"),
            DataType::Null => write!(f, "null"),
            DataType::Array(inner) => write!(f, "array<{}>", inner),
            DataType::Object => write!(f, "object"),
            DataType::Unknown => write!(f, "unknown"),
        }
    }
}

/// Schema definition for a column
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnSchema {
    /// Column name
    pub name: String,

    /// Inferred data type
    pub data_type: DataType,

    /// Whether nulls are allowed
    pub nullable: bool,

    /// Sample values (for inference)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sample_values: Vec<String>,

    /// Percentage of null values
    #[serde(default)]
    pub null_percentage: f64,
}

/// Strategy for handling null values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum NullHandling {
    /// Keep nulls as-is
    #[default]
    Keep,
    /// Drop rows with nulls
    DropRow,
    /// Fill with default value
    FillDefault,
    /// Fill with mean (numeric only)
    FillMean,
    /// Fill with median (numeric only)
    FillMedian,
    /// Fill with mode
    FillMode,
    /// Forward fill
    ForwardFill,
    /// Backward fill
    BackwardFill,
    /// Interpolate
    Interpolate,
}

/// Type coercion validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoercionValidation {
    /// Whether all coercions are safe
    pub is_safe: bool,

    /// List of potentially unsafe coercions
    pub warnings: Vec<CoercionWarning>,
}

/// Warning about a potentially unsafe type coercion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoercionWarning {
    /// Column name
    pub column: String,

    /// Source type
    pub from_type: DataType,

    /// Target type
    pub to_type: DataType,

    /// Risk level
    pub risk: CoercionRisk,

    /// Warning message
    pub message: String,
}

/// Risk level for type coercions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoercionRisk {
    /// Safe coercion (no data loss)
    Safe,
    /// May lose precision
    PrecisionLoss,
    /// May truncate data
    Truncation,
    /// May fail for some values
    PartialFailure,
    /// Likely to lose information
    DataLoss,
}

/// Data Analysis Decomposer
///
/// Decomposes data analysis tasks into ETL stages with schema
/// validation and type safety checks.
#[derive(Debug, Clone)]
pub struct DataAnalysisDecomposer {
    /// Stages to include
    stages: Vec<EtlStage>,

    /// Whether to include schema inference
    schema_validation: bool,

    /// Null handling strategy
    null_handling: NullHandling,

    /// Expected schema (for validation)
    expected_schema: Option<Vec<ColumnSchema>>,

    /// Allowed coercion risk level
    max_coercion_risk: CoercionRisk,

    /// Whether to include quality checks
    quality_checks: bool,
}

impl Default for DataAnalysisDecomposer {
    fn default() -> Self {
        Self::new()
    }
}

impl DataAnalysisDecomposer {
    /// Create a new DataAnalysisDecomposer with standard ETL stages
    pub fn new() -> Self {
        Self {
            stages: vec![
                EtlStage::Extract,
                EtlStage::Clean,
                EtlStage::Transform,
                EtlStage::Validate,
                EtlStage::Load,
            ],
            schema_validation: true,
            null_handling: NullHandling::default(),
            expected_schema: None,
            max_coercion_risk: CoercionRisk::PrecisionLoss,
            quality_checks: false,
        }
    }

    /// Create a minimal ETL decomposer
    pub fn minimal() -> Self {
        Self {
            stages: vec![EtlStage::Extract, EtlStage::Transform, EtlStage::Load],
            schema_validation: false,
            null_handling: NullHandling::Keep,
            expected_schema: None,
            max_coercion_risk: CoercionRisk::DataLoss,
            quality_checks: false,
        }
    }

    /// Enable or disable schema validation
    pub fn with_schema_validation(mut self, enabled: bool) -> Self {
        self.schema_validation = enabled;
        if enabled && !self.stages.contains(&EtlStage::SchemaInference) {
            self.stages.insert(1, EtlStage::SchemaInference);
        }
        self
    }

    /// Set null handling strategy
    pub fn with_null_handling(mut self, strategy: NullHandling) -> Self {
        self.null_handling = strategy;
        self
    }

    /// Set expected schema for validation
    pub fn with_expected_schema(mut self, schema: Vec<ColumnSchema>) -> Self {
        self.expected_schema = Some(schema);
        self
    }

    /// Set maximum allowed coercion risk
    pub fn with_max_coercion_risk(mut self, risk: CoercionRisk) -> Self {
        self.max_coercion_risk = risk;
        self
    }

    /// Enable quality checks
    pub fn with_quality_checks(mut self) -> Self {
        self.quality_checks = true;
        if !self.stages.contains(&EtlStage::QualityCheck) {
            self.stages.push(EtlStage::QualityCheck);
        }
        self
    }

    /// Set custom stages
    pub fn with_stages(mut self, stages: Vec<EtlStage>) -> Self {
        self.stages = stages;
        self
    }

    /// Add aggregation stage
    pub fn with_aggregation(mut self) -> Self {
        if !self.stages.contains(&EtlStage::Aggregate) {
            // Insert after transform
            if let Some(pos) = self.stages.iter().position(|s| *s == EtlStage::Transform) {
                self.stages.insert(pos + 1, EtlStage::Aggregate);
            } else {
                self.stages.push(EtlStage::Aggregate);
            }
        }
        self
    }

    /// Add enrichment stage
    pub fn with_enrichment(mut self) -> Self {
        if !self.stages.contains(&EtlStage::Enrich) {
            // Insert after aggregate or transform
            let pos = self
                .stages
                .iter()
                .position(|s| *s == EtlStage::Aggregate)
                .or_else(|| self.stages.iter().position(|s| *s == EtlStage::Transform));

            if let Some(p) = pos {
                self.stages.insert(p + 1, EtlStage::Enrich);
            } else {
                self.stages.push(EtlStage::Enrich);
            }
        }
        self
    }

    /// Validate type coercions
    pub fn validate_coercions(
        &self,
        coercions: &[(String, DataType, DataType)],
    ) -> CoercionValidation {
        let mut warnings = Vec::new();

        for (column, from_type, to_type) in coercions {
            let risk = self.assess_coercion_risk(from_type, to_type);

            if risk != CoercionRisk::Safe {
                warnings.push(CoercionWarning {
                    column: column.clone(),
                    from_type: from_type.clone(),
                    to_type: to_type.clone(),
                    risk,
                    message: self.coercion_message(from_type, to_type, risk),
                });
            }
        }

        let is_safe = warnings
            .iter()
            .all(|w| self.risk_level(w.risk) <= self.risk_level(self.max_coercion_risk));

        CoercionValidation { is_safe, warnings }
    }

    /// Assess risk level for a type coercion
    fn assess_coercion_risk(&self, from: &DataType, to: &DataType) -> CoercionRisk {
        match (from, to) {
            // Same type - safe
            (a, b) if a == b => CoercionRisk::Safe,

            // Integer to Float - safe
            (DataType::Integer, DataType::Float) => CoercionRisk::Safe,

            // Float to Integer - precision loss
            (DataType::Float, DataType::Integer) => CoercionRisk::PrecisionLoss,

            // Any to String - safe
            (DataType::Integer | DataType::Float, DataType::String) => CoercionRisk::Safe,

            // Date/DateTime conversions
            (DataType::Date, DataType::DateTime) => CoercionRisk::Safe,
            (DataType::DateTime, DataType::Date) => CoercionRisk::PrecisionLoss,

            // String to any - may fail (must come after more specific patterns)
            (DataType::String, _) => CoercionRisk::PartialFailure,

            // Null handling
            (DataType::Null, _) => CoercionRisk::Safe,
            (_, DataType::Null) => CoercionRisk::DataLoss,

            // Boolean conversions
            (DataType::Boolean, DataType::Integer) => CoercionRisk::Safe,
            (DataType::Integer, DataType::Boolean) => CoercionRisk::DataLoss,

            // Unknown types
            (DataType::Unknown, _) | (_, DataType::Unknown) => CoercionRisk::PartialFailure,

            // Everything else
            _ => CoercionRisk::DataLoss,
        }
    }

    /// Get numeric risk level for comparison
    fn risk_level(&self, risk: CoercionRisk) -> u8 {
        match risk {
            CoercionRisk::Safe => 0,
            CoercionRisk::PrecisionLoss => 1,
            CoercionRisk::Truncation => 2,
            CoercionRisk::PartialFailure => 3,
            CoercionRisk::DataLoss => 4,
        }
    }

    /// Generate warning message for coercion
    fn coercion_message(&self, from: &DataType, to: &DataType, risk: CoercionRisk) -> String {
        match risk {
            CoercionRisk::Safe => format!("Safe conversion from {} to {}", from, to),
            CoercionRisk::PrecisionLoss => {
                format!("Conversion from {} to {} may lose precision", from, to)
            }
            CoercionRisk::Truncation => {
                format!("Conversion from {} to {} may truncate data", from, to)
            }
            CoercionRisk::PartialFailure => {
                format!(
                    "Conversion from {} to {} may fail for some values",
                    from, to
                )
            }
            CoercionRisk::DataLoss => {
                format!("Conversion from {} to {} is likely to lose data", from, to)
            }
        }
    }

    /// Create subtask for an ETL stage
    fn create_stage_subtask(
        &self,
        parent_id: &str,
        stage: EtlStage,
        order: usize,
        context: &serde_json::Value,
    ) -> DecompositionSubtask {
        let task_id = format!("{}-{}", parent_id, stage);

        let mut subtask_context = json!({
            "stage": stage.to_string(),
            "stage_description": stage.description(),
            "null_handling": format!("{:?}", self.null_handling),
            "input_context": context,
        });

        // Add expected schema if available
        if let Some(schema) = &self.expected_schema {
            if let Some(obj) = subtask_context.as_object_mut() {
                obj.insert("expected_schema".to_string(), json!(schema));
            }
        }

        DecompositionSubtask::leaf(&task_id, stage.description())
            .with_parent(parent_id)
            .with_order(order)
            .with_context(subtask_context)
            .with_metadata("domain", json!("data"))
            .with_metadata("stage", json!(stage.to_string()))
    }

    /// Infer schema from sample data
    pub fn infer_schema(&self, sample: &serde_json::Value) -> Vec<ColumnSchema> {
        let mut schemas = Vec::new();

        if let Some(arr) = sample.as_array() {
            // Array of objects - infer from first object
            if let Some(first) = arr.first() {
                if let Some(obj) = first.as_object() {
                    for (key, value) in obj {
                        schemas.push(ColumnSchema {
                            name: key.clone(),
                            data_type: self.infer_type(value),
                            nullable: false,
                            sample_values: vec![value.to_string()],
                            null_percentage: 0.0,
                        });
                    }
                }
            }
        } else if let Some(obj) = sample.as_object() {
            // Single object
            for (key, value) in obj {
                schemas.push(ColumnSchema {
                    name: key.clone(),
                    data_type: self.infer_type(value),
                    nullable: false,
                    sample_values: vec![value.to_string()],
                    null_percentage: 0.0,
                });
            }
        }

        schemas
    }

    /// Infer type from a JSON value
    fn infer_type(&self, value: &serde_json::Value) -> DataType {
        match value {
            serde_json::Value::Null => DataType::Null,
            serde_json::Value::Bool(_) => DataType::Boolean,
            serde_json::Value::Number(n) => {
                if n.is_i64() || n.is_u64() {
                    DataType::Integer
                } else {
                    DataType::Float
                }
            }
            serde_json::Value::String(s) => {
                // Try to detect date/datetime
                if s.len() == 10 && s.chars().filter(|c| *c == '-').count() == 2 {
                    DataType::Date
                } else if s.contains('T') && s.contains(':') {
                    DataType::DateTime
                } else {
                    DataType::String
                }
            }
            serde_json::Value::Array(arr) => {
                if let Some(first) = arr.first() {
                    DataType::Array(Box::new(self.infer_type(first)))
                } else {
                    DataType::Array(Box::new(DataType::Unknown))
                }
            }
            serde_json::Value::Object(_) => DataType::Object,
        }
    }
}

impl DecompositionAgent for DataAnalysisDecomposer {
    fn propose_decomposition(
        &self,
        task_id: &str,
        description: &str,
        context: &serde_json::Value,
        _depth: usize,
    ) -> Result<DecompositionProposal, DecompositionError> {
        // Check for coercions in context and validate them
        if let Some(coercions_val) = context.get("coercions") {
            if let Ok(coercions) =
                serde_json::from_value::<Vec<(String, DataType, DataType)>>(coercions_val.clone())
            {
                let validation = self.validate_coercions(&coercions);
                if !validation.is_safe {
                    let warnings: Vec<String> = validation
                        .warnings
                        .iter()
                        .map(|w| w.message.clone())
                        .collect();
                    return Err(DecompositionError::ValidationError {
                        message: format!("Unsafe type coercions: {}", warnings.join("; ")),
                    });
                }
            }
        }

        let mut subtasks = Vec::new();

        // Create subtasks for each stage
        for (order, stage) in self.stages.iter().enumerate() {
            subtasks.push(self.create_stage_subtask(task_id, *stage, order, context));
        }

        // Determine composition function
        let composition_fn = CompositionFunction::Sequential;

        let mut metadata = HashMap::new();
        metadata.insert("domain".to_string(), json!("data"));
        metadata.insert(
            "stages".to_string(),
            json!(self
                .stages
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()),
        );
        metadata.insert(
            "schema_validation".to_string(),
            json!(self.schema_validation),
        );
        metadata.insert(
            "null_handling".to_string(),
            json!(format!("{:?}", self.null_handling)),
        );

        Ok(DecompositionProposal {
            proposal_id: format!("data-etl-{}", task_id),
            source_task_id: task_id.to_string(),
            subtasks,
            composition_fn,
            confidence: 0.85,
            rationale: Some(format!(
                "Data analysis decomposition: {} with {} stages",
                description,
                self.stages.len()
            )),
            metadata,
        })
    }

    fn is_atomic(&self, _task_id: &str, description: &str) -> bool {
        let atomic_hints = [
            "read file",
            "write file",
            "single column",
            "drop column",
            "rename column",
            "filter rows",
            "sort by",
        ];

        let desc_lower = description.to_lowercase();
        atomic_hints.iter().any(|h| desc_lower.contains(h))
    }

    fn name(&self) -> &str {
        "data_analysis"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================
    // EtlStage Tests
    // ==========================================

    #[test]
    fn test_stage_order() {
        assert!(EtlStage::Extract.order() < EtlStage::Transform.order());
        assert!(EtlStage::Transform.order() < EtlStage::Load.order());
    }

    #[test]
    fn test_stage_description() {
        assert!(!EtlStage::Extract.description().is_empty());
        assert!(!EtlStage::Load.description().is_empty());
    }

    #[test]
    fn test_stage_display() {
        assert_eq!(EtlStage::Extract.to_string(), "extract");
        assert_eq!(EtlStage::Transform.to_string(), "transform");
        assert_eq!(EtlStage::Load.to_string(), "load");
    }

    #[test]
    fn test_stage_serialization() {
        let stages = vec![EtlStage::Extract, EtlStage::Transform, EtlStage::Load];

        for stage in stages {
            let json = serde_json::to_string(&stage).unwrap();
            let parsed: EtlStage = serde_json::from_str(&json).unwrap();
            assert_eq!(stage, parsed);
        }
    }

    // ==========================================
    // DataType Tests
    // ==========================================

    #[test]
    fn test_data_type_display() {
        assert_eq!(DataType::String.to_string(), "string");
        assert_eq!(DataType::Integer.to_string(), "integer");
        assert_eq!(
            DataType::Array(Box::new(DataType::Integer)).to_string(),
            "array<integer>"
        );
    }

    #[test]
    fn test_data_type_serialization() {
        let types = vec![
            DataType::String,
            DataType::Integer,
            DataType::Float,
            DataType::Boolean,
            DataType::Array(Box::new(DataType::String)),
        ];

        for dtype in types {
            let json = serde_json::to_string(&dtype).unwrap();
            let parsed: DataType = serde_json::from_str(&json).unwrap();
            assert_eq!(dtype, parsed);
        }
    }

    // ==========================================
    // DataAnalysisDecomposer Construction Tests
    // ==========================================

    #[test]
    fn test_new_decomposer() {
        let decomposer = DataAnalysisDecomposer::new();
        assert_eq!(decomposer.stages.len(), 5);
        assert!(decomposer.schema_validation);
    }

    #[test]
    fn test_minimal_decomposer() {
        let decomposer = DataAnalysisDecomposer::minimal();
        assert_eq!(decomposer.stages.len(), 3);
        assert!(!decomposer.schema_validation);
    }

    #[test]
    fn test_with_schema_validation() {
        let decomposer = DataAnalysisDecomposer::new().with_schema_validation(true);
        assert!(decomposer.stages.contains(&EtlStage::SchemaInference));
    }

    #[test]
    fn test_with_quality_checks() {
        let decomposer = DataAnalysisDecomposer::new().with_quality_checks();
        assert!(decomposer.stages.contains(&EtlStage::QualityCheck));
    }

    #[test]
    fn test_with_aggregation() {
        let decomposer = DataAnalysisDecomposer::new().with_aggregation();
        assert!(decomposer.stages.contains(&EtlStage::Aggregate));
    }

    #[test]
    fn test_with_enrichment() {
        let decomposer = DataAnalysisDecomposer::new().with_enrichment();
        assert!(decomposer.stages.contains(&EtlStage::Enrich));
    }

    // ==========================================
    // Coercion Validation Tests
    // ==========================================

    #[test]
    fn test_safe_coercion() {
        let decomposer = DataAnalysisDecomposer::new();
        let coercions = vec![("col1".to_string(), DataType::Integer, DataType::Float)];

        let result = decomposer.validate_coercions(&coercions);
        assert!(result.is_safe);
    }

    #[test]
    fn test_precision_loss_coercion() {
        let decomposer = DataAnalysisDecomposer::new();
        let coercions = vec![("col1".to_string(), DataType::Float, DataType::Integer)];

        let result = decomposer.validate_coercions(&coercions);
        assert!(result.is_safe); // Precision loss is within default tolerance
        assert_eq!(result.warnings.len(), 1);
        assert_eq!(result.warnings[0].risk, CoercionRisk::PrecisionLoss);
    }

    #[test]
    fn test_data_loss_coercion() {
        let decomposer =
            DataAnalysisDecomposer::new().with_max_coercion_risk(CoercionRisk::PrecisionLoss);
        let coercions = vec![("col1".to_string(), DataType::String, DataType::Null)];

        let result = decomposer.validate_coercions(&coercions);
        assert!(!result.is_safe);
    }

    #[test]
    fn test_string_to_date_coercion() {
        let decomposer = DataAnalysisDecomposer::new();
        let coercions = vec![("col1".to_string(), DataType::String, DataType::Date)];

        let result = decomposer.validate_coercions(&coercions);
        assert_eq!(result.warnings[0].risk, CoercionRisk::PartialFailure);
    }

    // ==========================================
    // Schema Inference Tests
    // ==========================================

    #[test]
    fn test_infer_schema_from_object() {
        let decomposer = DataAnalysisDecomposer::new();
        let sample = json!({
            "name": "test",
            "age": 25,
            "score": 95.5,
            "active": true
        });

        let schema = decomposer.infer_schema(&sample);
        assert_eq!(schema.len(), 4);

        let name_col = schema.iter().find(|c| c.name == "name").unwrap();
        assert_eq!(name_col.data_type, DataType::String);

        let age_col = schema.iter().find(|c| c.name == "age").unwrap();
        assert_eq!(age_col.data_type, DataType::Integer);
    }

    #[test]
    fn test_infer_schema_from_array() {
        let decomposer = DataAnalysisDecomposer::new();
        let sample = json!([
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"}
        ]);

        let schema = decomposer.infer_schema(&sample);
        assert_eq!(schema.len(), 2);
    }

    #[test]
    fn test_infer_date_type() {
        let decomposer = DataAnalysisDecomposer::new();
        let sample = json!({"date": "2024-01-15"});

        let schema = decomposer.infer_schema(&sample);
        let date_col = schema.iter().find(|c| c.name == "date").unwrap();
        assert_eq!(date_col.data_type, DataType::Date);
    }

    #[test]
    fn test_infer_datetime_type() {
        let decomposer = DataAnalysisDecomposer::new();
        let sample = json!({"timestamp": "2024-01-15T10:30:00Z"});

        let schema = decomposer.infer_schema(&sample);
        let ts_col = schema.iter().find(|c| c.name == "timestamp").unwrap();
        assert_eq!(ts_col.data_type, DataType::DateTime);
    }

    // ==========================================
    // Decomposition Tests
    // ==========================================

    #[test]
    fn test_basic_decomposition() {
        let decomposer = DataAnalysisDecomposer::new();

        let result = decomposer.propose_decomposition(
            "etl-job",
            "Process customer data",
            &json!({"source": "customers.csv"}),
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
    fn test_coercion_validation_in_decomposition() {
        let decomposer = DataAnalysisDecomposer::new().with_max_coercion_risk(CoercionRisk::Safe);

        let result = decomposer.propose_decomposition(
            "etl-unsafe",
            "Process with unsafe coercions",
            &json!({
                "coercions": [
                    ["col1", "float", "integer"]
                ]
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
        let decomposer = DataAnalysisDecomposer::new();
        assert_eq!(decomposer.name(), "data_analysis");
    }

    #[test]
    fn test_is_atomic() {
        let decomposer = DataAnalysisDecomposer::new();

        assert!(decomposer.is_atomic("task-1", "Read file from CSV"));
        assert!(decomposer.is_atomic("task-2", "Drop column id"));
        assert!(decomposer.is_atomic("task-3", "Filter rows where x > 0"));
        assert!(!decomposer.is_atomic("task-4", "Run full ETL pipeline"));
    }

    // ==========================================
    // m=1 Enforcement Tests
    // ==========================================

    #[test]
    fn test_all_subtasks_have_m1() {
        let decomposer = DataAnalysisDecomposer::new()
            .with_schema_validation(true)
            .with_aggregation()
            .with_enrichment()
            .with_quality_checks();

        let result =
            decomposer.propose_decomposition("full-etl", "Run full ETL pipeline", &json!({}), 0);

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
    // Null Handling Tests
    // ==========================================

    #[test]
    fn test_null_handling_serialization() {
        let strategies = vec![
            NullHandling::Keep,
            NullHandling::DropRow,
            NullHandling::FillDefault,
            NullHandling::FillMean,
        ];

        for strategy in strategies {
            let json = serde_json::to_string(&strategy).unwrap();
            let parsed: NullHandling = serde_json::from_str(&json).unwrap();
            assert_eq!(strategy, parsed);
        }
    }

    // ==========================================
    // Column Schema Tests
    // ==========================================

    #[test]
    fn test_column_schema_serialization() {
        let schema = ColumnSchema {
            name: "test_col".to_string(),
            data_type: DataType::Integer,
            nullable: true,
            sample_values: vec!["1".to_string(), "2".to_string()],
            null_percentage: 5.0,
        };

        let json = serde_json::to_string(&schema).unwrap();
        let parsed: ColumnSchema = serde_json::from_str(&json).unwrap();

        assert_eq!(schema.name, parsed.name);
        assert_eq!(schema.data_type, parsed.data_type);
        assert_eq!(schema.nullable, parsed.nullable);
    }
}
