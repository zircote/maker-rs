# Using MAKER for Data Analysis

A practical guide to reliable data transformations, ETL pipelines, and analytical queries with MAKER.

## Why MAKER for Data Analysis?

Data analysis tasks are error-prone:
- Type coercions can silently lose data
- Null handling varies by context
- Aggregations can produce wrong results
- Schema changes break pipelines

MAKER addresses these through:
- **ETL decomposition**: Structured Extract-Transform-Load-Validate patterns
- **Schema inference**: Automatic type detection with risk assessment
- **Coercion validation**: Flags precision loss, truncation, data loss
- **Quality checks**: Built-in validation at each stage

## Setup

### Configure MAKER for Data Tasks

```json
{
  "k_margin": 4,
  "matcher": "exact",  // Data outputs should be deterministic
  "adaptive_k": true
}
```

### Recommended Prompting

```
Configure MAKER for data analysis:
- Use exact matching (data results should match exactly)
- k_margin: 4 (data errors are costly)
- Validate all outputs against schema
```

## Core Patterns

### Pattern 1: ETL Pipeline

Use MAKER's DataAnalysisDecomposer for structured ETL:

```
Execute this ETL pipeline with MAKER decomposition:

Source: customer_orders.csv
Target: analytics_warehouse

Stages:
1. Extract (maker/vote): Parse CSV with correct types
2. SchemaInference (maker/vote): Detect column types
3. Clean (maker/vote): Handle nulls, fix formats
4. Transform (maker/vote): Calculate derived fields
5. Validate (maker/validate): Check referential integrity
6. Load (maker/vote): Generate insert statements

For each stage, validate before proceeding.
```

**MAKER internally uses:**
- `EtlStage::Extract` - Parse source data
- `EtlStage::SchemaInference` - Detect types
- `EtlStage::Clean` - Null handling, normalization
- `EtlStage::Transform` - Apply business logic
- `EtlStage::Validate` - Quality checks
- `EtlStage::Load` - Write to target

### Pattern 2: SQL Query Generation

Generate reliable SQL with voting:

```
Use maker/vote with k_margin=4 to generate SQL for:
"Find customers who made their first purchase in 2024 and have since made at least 3 more purchases"

Tables:
- customers (id, name, email, created_at)
- orders (id, customer_id, total, created_at)

Validate the result with maker/validate:
- Valid SQL syntax
- No destructive operations (DELETE, DROP, UPDATE)
- Uses proper JOINs (no cartesian products)
```

### Pattern 3: Data Transformation

For complex transformations, use voting at each step:

```
Transform this dataset with MAKER reliability:

Input: JSON array of user events
Output: Aggregated metrics by user and day

Steps (use maker/vote for each):
1. Parse JSON, validate structure
2. Extract date from timestamp, handle timezones
3. Group by user_id and date
4. Calculate: event_count, unique_actions, session_duration
5. Handle edge cases: null users, invalid dates, outliers

Final validation:
- No null user_ids in output
- All dates in expected range
- Metrics are non-negative
```

## Schema Inference

MAKER's DataAnalysisDecomposer infers types with risk assessment:

### Type Detection

| Detected Type | Example Values | Confidence |
|---------------|----------------|------------|
| Integer | 1, 42, -7 | High |
| Float | 3.14, 2.0, -1.5 | High |
| Boolean | true, false, 1, 0 | Medium |
| Timestamp | 2024-01-15, 2024-01-15T10:30:00Z | Medium |
| String | Anything else | Default |
| Json | `{"key": "value"}` | Medium |

### Coercion Risk Assessment

When converting types, MAKER flags risks:

```
Use maker/vote to convert these columns:

- price (string "19.99") -> Float
  Risk: Safe (no precision loss)

- quantity (string "1,000") -> Integer
  Risk: Safe (remove comma, parse)

- rating (float 4.7) -> Integer
  Risk: PrecisionLoss (4.7 -> 4)

- id (string "12345678901234567890") -> Integer
  Risk: Truncation (exceeds i64)

- date (string "invalid") -> Timestamp
  Risk: PartialFailure (some rows fail)
```

### Example: Schema-Aware Transformation

```
Analyze this CSV and propose schema:

```csv
id,name,price,quantity,date
1,Widget,19.99,100,2024-01-15
2,Gadget,29.50,50,2024-01-16
```

Use maker/vote to:
1. Infer column types with confidence
2. Identify coercion risks
3. Generate typed schema (SQL CREATE TABLE or JSON Schema)
4. Flag any ambiguous columns

Expected output:
- id: Integer (high confidence)
- name: String (default)
- price: Float (high confidence)
- quantity: Integer (high confidence)
- date: Timestamp (medium confidence, format: YYYY-MM-DD)
```

## Null Handling Strategies

MAKER supports explicit null handling:

### Configuration

```
Configure MAKER for null handling:
- NullHandling::Preserve (keep nulls)
- NullHandling::Default(value) (replace with default)
- NullHandling::Exclude (filter out null rows)
- NullHandling::Fail (error on null)
```

### Example: Explicit Null Strategy

```
Transform this data with explicit null handling:

Input:
| user_id | score |
|---------|-------|
| 1       | 85    |
| 2       | null  |
| 3       | 92    |

Use maker/vote with these null rules:
- user_id: Fail on null (required)
- score: Default to 0 (optional metric)

Expected output after transformation:
| user_id | score |
|---------|-------|
| 1       | 85    |
| 2       | 0     |
| 3       | 92    |
```

## Quality Checks

### Built-in Validation

After each ETL stage, MAKER can validate:

```
After the Transform stage, use maker/validate to check:

1. Row count matches expected (within 5% tolerance)
2. No duplicate primary keys
3. All foreign keys exist in reference table
4. Numeric fields within expected ranges
5. Date fields within expected period
6. Required fields are non-null
```

### Custom Quality Rules

```
Define quality rules for this pipeline:

{
  "rules": [
    {"field": "revenue", "check": "non_negative"},
    {"field": "email", "check": "format_email"},
    {"field": "country_code", "check": "in_list", "values": ["US", "CA", "UK"]},
    {"field": "created_at", "check": "not_future"}
  ]
}

Use maker/validate with these rules on the transformation output.
```

## Common Data Tasks

### Task: Aggregate Metrics

```
Calculate daily revenue metrics with MAKER:

Input: transactions table
Output: daily_revenue table

Use maker/vote for each step:
1. Filter to valid transactions (status = 'completed')
2. Group by date (truncate timestamp to date)
3. Calculate: total_revenue, transaction_count, avg_transaction
4. Handle currency (convert to USD)

Validate:
- All dates represented (no gaps)
- total_revenue = sum(transaction amounts)
- avg_transaction = total_revenue / transaction_count
```

### Task: Data Cleaning

```
Clean this customer dataset with MAKER:

Issues to handle:
- Duplicate emails (keep most recent)
- Invalid phone formats (normalize to E.164)
- Missing zip codes (infer from city/state if possible)
- Inconsistent country names (standardize to ISO codes)

Use maker/vote for each cleaning rule, then validate:
- No duplicate emails remain
- All phones match E.164 pattern
- Zip codes are 5 or 9 digits (US)
- Country codes are valid ISO-3166
```

### Task: Join Multiple Sources

```
Join these three data sources with MAKER:

Sources:
1. orders.csv (order_id, customer_id, total, date)
2. customers.json (id, name, email, tier)
3. products_api response (order_id, product_list)

Use maker/vote to:
1. Parse each source with correct types
2. Identify join keys
3. Generate join logic (handle nulls correctly)
4. Produce denormalized output

Validate:
- No cartesian products
- Row count <= orders count
- All required fields present
```

## Workflow: Complete Data Pipeline

```
# End-to-end data pipeline with MAKER

## Phase 1: Discovery
Use maker/vote to analyze source data:
"Describe the structure, types, and quality issues in this dataset"

## Phase 2: Schema Design
Use maker/vote to design target schema:
"Create optimized schema for analytical queries"

## Phase 3: ETL Implementation
Use maker/decompose with DataAnalysisDecomposer:
"Transform source to target with these business rules..."

For each stage:
- maker/vote generates the transformation
- maker/validate checks quality
- Proceed only on validation pass

## Phase 4: Quality Assurance
Use maker/vote to generate quality checks:
- Row counts match expected
- Primary keys unique
- Foreign keys valid
- Business rules satisfied

## Phase 5: Documentation
Use maker/vote with embedding matcher:
- Generate column descriptions
- Document transformation logic
- Create data dictionary
```

## Troubleshooting

### Inconsistent Aggregation Results

**Problem:** Voting not converging on aggregations

**Solutions:**
1. Specify exact aggregation function (SUM vs COUNT vs AVG)
2. Clarify grouping columns
3. Define null handling for aggregated fields

### Schema Inference Failures

**Problem:** Wrong types inferred

**Solutions:**
1. Provide sample values in prompt
2. Specify expected types explicitly
3. Use lower confidence threshold for ambiguous columns

### ETL Stage Failures

**Problem:** Transformation fails validation

**Solutions:**
1. Check upstream stages for data quality
2. Add intermediate validation steps
3. Use MAKER decomposition for finer granularity

## See Also

- [../PROMPT-ENGINEERING.md](../PROMPT-ENGINEERING.md) - General prompting strategies
- [CODING.md](./CODING.md) - MAKER for programming tasks
- [ML-PIPELINES.md](./ML-PIPELINES.md) - MAKER for ML workflows
