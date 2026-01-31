//! Domain-Specific Decomposers
//!
//! Provides specialized decomposition strategies for different problem domains:
//!
//! - **Coding**: AST-based code decomposition (function/block/line level)
//! - **ML Pipeline**: Machine learning workflow decomposition (DataPrep/Train/Eval)
//! - **Data Analysis**: ETL pattern decomposition (Extract/Transform/Load)
//!
//! Each decomposer implements the `DecompositionAgent` trait and produces
//! domain-appropriate subtask structures with m=1 leaf nodes.

#[cfg(feature = "code-matcher")]
pub mod coding;

pub mod data;
pub mod ml;

// Re-export domain decomposers
#[cfg(feature = "code-matcher")]
pub use coding::{CodeDecompositionStrategy, CodingDecomposer, SyntaxValidationResult};

pub use data::{DataAnalysisDecomposer, EtlStage};
pub use ml::{MLPipelineDecomposer, PipelineStage};
