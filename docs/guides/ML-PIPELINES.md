# Using MAKER for ML Pipelines

A practical guide to reliable machine learning workflows with MAKER's error-correction framework.

## Why MAKER for ML?

Machine learning pipelines have unique challenges:
- Hyperparameter choices are often arbitrary
- Metric calculations can be wrong (NaN, infinity)
- Pipeline stages have dependencies
- Reproducibility is hard to maintain

MAKER addresses these through:
- **Pipeline decomposition**: Structured DataPrep → Config → Train → Evaluate pattern
- **Metric validation**: Automatic detection of NaN, infinity, out-of-range values
- **Hyperparameter search**: Parallel composition for grid/random/Bayesian search
- **Cross-validation**: Built-in CV fold management with proper state handling

## Setup

### Configure MAKER for ML Tasks

```json
{
  "k_margin": 4,
  "matcher": "exact",  // Metrics should match exactly
  "adaptive_k": true,
  "ensemble": {
    "strategy": "round_robin"  // Different perspectives on ML decisions
  }
}
```

### Recommended Prompting

```
Configure MAKER for machine learning:
- Use exact matching (numeric results)
- k_margin: 4 (ML errors are hard to detect)
- Enable metric validation (NaN, infinity checks)
- Validate all model outputs
```

## Core Patterns

### Pattern 1: Complete ML Pipeline

Use MAKER's MLPipelineDecomposer for structured pipelines:

```
Execute this ML pipeline with MAKER decomposition:

Task: "Build a customer churn prediction model"
Data: customer_features.csv (10,000 rows, 50 features)

Pipeline Stages:
1. DataPrep (maker/vote): Load data, handle missing values, encode categoricals
2. FeatureEngineering (maker/vote): Create derived features, normalize
3. ModelConfig (maker/vote): Select algorithm, set hyperparameters
4. Training (maker/vote): Fit model with cross-validation
5. Evaluation (maker/validate): Check metrics (accuracy, precision, recall, AUC)
6. Validation (maker/validate): Validate on holdout set

For each stage, validate before proceeding.
```

**MAKER internally uses:**
- `PipelineStage::DataPrep` - Data loading and cleaning
- `PipelineStage::FeatureEngineering` - Feature creation
- `PipelineStage::ModelConfig` - Algorithm selection
- `PipelineStage::Training` - Model fitting
- `PipelineStage::Evaluation` - Metric computation
- `PipelineStage::Validation` - Final validation

### Pattern 2: Hyperparameter Tuning

Use parallel composition for hyperparameter search:

```
Perform hyperparameter search with MAKER:

Model: Random Forest Classifier
Search Space:
- n_estimators: [100, 200, 500]
- max_depth: [10, 20, None]
- min_samples_split: [2, 5, 10]

Use MAKER with:
- HyperparameterSearch stage
- SearchStrategy: Grid (exhaustive)
- Composition: Parallel (run all configs)
- Metric: AUC-ROC

For each configuration:
1. Use maker/vote to train and evaluate
2. Validate metrics are in valid range (0-1 for AUC)
3. Collect results

After all complete, vote on best configuration.
```

### Pattern 3: Cross-Validation

Proper CV with MAKER state management:

```
Perform 5-fold cross-validation with MAKER:

For each fold:
1. Use maker/decompose to separate train/val split
2. Use maker/vote for training on train split
3. Use maker/vote for evaluation on val split
4. Validate metrics (no NaN, no infinity)

After all folds:
1. Aggregate metrics across folds (mean, std)
2. Validate aggregated metrics
3. Report confidence intervals
```

## Metric Validation

MAKER's MLPipelineDecomposer includes automatic metric validation:

### Automatic Checks

| Check | Detection | Action |
|-------|-----------|--------|
| NaN values | `value.is_nan()` | Red flag, reject sample |
| Infinity | `value.is_infinite()` | Red flag, reject sample |
| Out of range | `value < 0.0 \|\| value > 1.0` (for probabilities) | Red flag, reject sample |
| Zero variance | `std == 0` on predictions | Warning |

### Example: Metric Validation

```
After training, validate metrics with MAKER:

Metrics to validate:
- accuracy: expect 0.0 - 1.0
- precision: expect 0.0 - 1.0
- recall: expect 0.0 - 1.0
- f1_score: expect 0.0 - 1.0
- auc_roc: expect 0.0 - 1.0
- log_loss: expect > 0.0

Use maker/validate with schema:
{
  "accuracy": {"min": 0.0, "max": 1.0, "not_nan": true},
  "precision": {"min": 0.0, "max": 1.0, "not_nan": true},
  "recall": {"min": 0.0, "max": 1.0, "not_nan": true},
  "f1_score": {"min": 0.0, "max": 1.0, "not_nan": true},
  "auc_roc": {"min": 0.0, "max": 1.0, "not_nan": true},
  "log_loss": {"min": 0.0, "not_nan": true}
}

Flag any metrics outside expected ranges.
```

## Pipeline Stage Details

### DataPrep Stage

```
Use maker/vote for data preparation:

Tasks:
1. Load dataset, infer schema
2. Handle missing values:
   - Numeric: median imputation
   - Categorical: mode imputation
   - Drop if > 50% missing
3. Encode categoricals:
   - Binary: 0/1 encoding
   - Multi-class: one-hot (if < 10 categories)
   - High cardinality: target encoding
4. Split into train/test (80/20, stratified)

Validate:
- No missing values in output
- All columns numeric
- Train/test have same columns
- Target distribution preserved
```

### FeatureEngineering Stage

```
Use maker/vote for feature engineering:

Tasks:
1. Create interaction features (top 5 pairs by correlation)
2. Polynomial features (degree 2 for top 3 predictors)
3. Date features (day of week, month, is_weekend)
4. Normalize numeric features (z-score)
5. Handle outliers (clip at 3 std)

Validate:
- No infinity values after transformations
- Feature count matches expected
- No high correlation (> 0.95) between features
```

### ModelConfig Stage

```
Use maker/vote to select model configuration:

Given:
- Problem type: Binary classification
- Data size: 10,000 rows, 50 features
- Priority: Interpretability over performance

Vote on:
1. Algorithm choice
2. Key hyperparameters
3. Regularization strength
4. Class balancing strategy

Expected output:
{
  "algorithm": "LogisticRegression",
  "hyperparameters": {
    "C": 1.0,
    "penalty": "l2",
    "solver": "lbfgs",
    "class_weight": "balanced"
  }
}
```

### Training Stage

```
Use maker/vote for model training:

Configuration from previous stage: {...}
Training data: X_train, y_train

Tasks:
1. Initialize model with hyperparameters
2. Fit on training data
3. Generate training metrics
4. Save model checkpoint

Validate:
- Model converged (no warnings)
- Training metrics reasonable
- No NaN in model weights
```

### Evaluation Stage

```
Use maker/vote for evaluation:

Tasks:
1. Generate predictions on test set
2. Calculate classification metrics:
   - Accuracy, Precision, Recall, F1
   - AUC-ROC, AUC-PR
   - Confusion matrix
3. Calculate calibration metrics
4. Feature importance

Validate with maker/validate:
- All metrics in valid range
- No NaN/infinity
- Predictions match expected shape
```

## Hyperparameter Search Strategies

### Grid Search (Exhaustive)

```
Use MAKER with Grid search strategy:

Search space:
- learning_rate: [0.001, 0.01, 0.1]
- batch_size: [32, 64, 128]
- epochs: [10, 20]

Total configurations: 3 × 3 × 2 = 18

MAKER executes with Parallel composition:
- All 18 configs run in parallel
- Each uses maker/vote for training
- Results aggregated
- Winner selected by best validation metric
```

### Random Search

```
Use MAKER with Random search strategy:

Search space:
- learning_rate: log-uniform(0.0001, 0.1)
- batch_size: uniform(16, 256)
- dropout: uniform(0.0, 0.5)
- n_layers: int(1, 5)

Samples: 20 random configurations

MAKER executes:
- Sample 20 configurations
- Parallel execution
- Early stopping if time budget exceeded
- Return best and top-5
```

### Bayesian Search

```
Use MAKER with Bayesian search strategy:

Search space:
- Same as above

Configuration:
- Initial random samples: 5
- Acquisition function: Expected Improvement
- Total evaluations: 30

MAKER executes:
- Sequential with state passing
- Each iteration uses maker/vote
- Surrogate model updated after each eval
- Returns best configuration and convergence history
```

## Cross-Validation Patterns

### K-Fold CV

```
Execute 5-fold CV with MAKER:

For fold in [1, 2, 3, 4, 5]:
  1. Split data (fold as validation, rest as train)
  2. Use maker/vote: Train model
  3. Use maker/vote: Evaluate on validation fold
  4. Use maker/validate: Check metrics valid
  5. Store fold results

Aggregate:
- Mean metrics across folds
- Standard deviation
- 95% confidence interval

Final output:
{
  "accuracy": {"mean": 0.85, "std": 0.02, "ci_95": [0.82, 0.88]},
  "auc_roc": {"mean": 0.91, "std": 0.01, "ci_95": [0.89, 0.93]}
}
```

### Stratified CV

```
Execute stratified 5-fold CV with MAKER:

Stratification: Preserve class distribution in each fold

Additional validations:
- Check class proportions match original
- Flag if any fold has < 10 samples of minority class
```

### Nested CV

```
Execute nested CV with MAKER:

Outer loop (5 folds): Model assessment
Inner loop (3 folds): Hyperparameter selection

For each outer fold:
  1. Hold out test set
  2. Run inner CV for hyperparameter search
  3. Select best hyperparameters
  4. Train on full training set
  5. Evaluate on test set

This provides unbiased performance estimate.
```

## Workflow: Complete ML Project

```
# End-to-end ML project with MAKER

## Phase 1: Problem Definition
Use maker/vote to formalize:
- Problem type (classification, regression, etc.)
- Success metrics
- Baseline to beat
- Constraints (latency, interpretability)

## Phase 2: Data Understanding
Use MAKER with DataAnalysisDecomposer:
- Exploratory analysis
- Data quality assessment
- Feature profiling

## Phase 3: Feature Engineering
Use MAKER with MLPipelineDecomposer:
- Feature creation
- Feature selection
- Encoding strategies

## Phase 4: Model Selection
Use maker/vote with ensemble:
- Try multiple algorithms
- Vote on best approach
- Document rationale

## Phase 5: Hyperparameter Tuning
Use MAKER with parallel hyperparameter search:
- Grid/Random/Bayesian search
- Cross-validation for each config
- Metric validation throughout

## Phase 6: Final Evaluation
Use maker/validate for:
- Holdout test performance
- Metric validity (no NaN, in range)
- Comparison to baseline

## Phase 7: Documentation
Use maker/vote with embedding matcher:
- Generate model card
- Document limitations
- Create deployment checklist
```

## Troubleshooting

### NaN in Metrics

**Problem:** Training produces NaN metrics

**Solutions:**
1. Check for NaN in input data
2. Reduce learning rate
3. Add gradient clipping
4. Normalize input features

### Overfitting Detection

**Problem:** Large gap between train and validation metrics

**Solutions:**
1. Add regularization (increase via maker/vote)
2. Reduce model complexity
3. Increase training data
4. Apply early stopping

### Metric Validation Failures

**Problem:** Metrics outside expected range

**Solutions:**
1. Check class imbalance
2. Verify metric calculation
3. Look for data leakage
4. Validate feature engineering

## See Also

- [../PROMPT-ENGINEERING.md](../PROMPT-ENGINEERING.md) - General prompting strategies
- [CODING.md](./CODING.md) - MAKER for programming tasks
- [DATA-ANALYSIS.md](./DATA-ANALYSIS.md) - MAKER for data tasks
