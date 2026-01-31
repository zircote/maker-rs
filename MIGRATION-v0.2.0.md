# Migration Guide: v0.1.0 to v0.2.0

v0.2.0 is **fully backward compatible** with v0.1.0. All new features are opt-in and all existing APIs continue to work without changes.

## No Breaking Changes

v0.1.0 code compiles and runs against v0.2.0 without modification. All new fields on request/response structs use `Option<T>` with `None` defaults.

## New Features (Opt-In)

### Adaptive K-Margin

Dynamically adjust the k-margin based on observed accuracy to reduce API calls:

```rust
use maker::core::{KEstimator, vote_with_margin_adaptive, MockLlmClient, VoteConfig};

let client = MockLlmClient::constant("answer");
let config = VoteConfig::default();
let mut estimator = KEstimator::new(0.85, 0.95, 100);

let result = vote_with_margin_adaptive("prompt", &client, config, &mut estimator).unwrap();
println!("Used k={}, estimated p={:.2}", result.k_used, estimator.p_hat());
```

Via MCP:

```json
{
  "adaptive_k": true,
  "ema_alpha": 0.1,
  "k_bounds": [2, 10]
}
```

### Semantic Matching

Use pluggable matchers for non-deterministic tasks:

```rust
use maker::core::matcher::{CandidateMatcher, ExactMatcher};
use maker::core::VoteConfig;
use std::sync::Arc;

// Default: ExactMatcher (unchanged from v0.1.0)
let config = VoteConfig::default();

// Opt-in: custom matcher
let matcher: Arc<dyn CandidateMatcher> = Arc::new(ExactMatcher);
let config = VoteConfig::default().with_matcher(matcher);
```

Available matchers:

| Matcher | Import | Feature Flag |
|---------|--------|-------------|
| `ExactMatcher` | `maker::core::matcher::ExactMatcher` | None (default) |
| `EmbeddingMatcher` | `maker::core::matchers::embedding::EmbeddingMatcher` | None |
| `CodeMatcher` | `maker::core::matchers::code::CodeMatcher` | `code-matcher` |

Via MCP:

```json
{
  "matcher": { "type": "embedding", "threshold": 0.92, "provider": "ollama" }
}
```

### Multi-Model Ensemble

Vote across multiple LLM models for error decorrelation:

```rust
use maker::llm::ensemble::{EnsembleConfig, ModelSlot, EnsembleStrategy, CostTier};

let config = EnsembleConfig::new(
    vec![
        ModelSlot::new(cheap_client, 1.0, CostTier::Cheap),
        ModelSlot::new(expensive_client, 1.0, CostTier::Expensive),
    ],
    EnsembleStrategy::CostAware,
).unwrap();
```

Via MCP:

```json
{
  "ensemble": {
    "models": [
      { "provider": "ollama", "model": "llama3", "cost_tier": "cheap" },
      { "provider": "anthropic", "model": "claude-haiku", "cost_tier": "medium" }
    ],
    "strategy": "cost_aware"
  }
}
```

## New Response Fields

The following fields are added to `VoteResponse` (all `Option`, default `None`):

| Field | Type | When Present |
|-------|------|-------------|
| `k_used` | `usize` | Always (actual k-margin used) |
| `p_hat` | `Option<f64>` | When adaptive_k is enabled |
| `matcher_type` | `String` | Always (matcher used: "exact", "embedding", "code") |
| `candidate_groups` | `usize` | Always (number of distinct groups after matching) |
| `ensemble_metrics` | `Option<EnsembleMetrics>` | When ensemble is enabled |

## New Configuration Fields

The following fields are added to `ConfigRequest` (all `Option`, default `None`):

| Field | Type | Description |
|-------|------|-------------|
| `adaptive_k` | `Option<bool>` | Enable adaptive k-margin |
| `ema_alpha` | `Option<f64>` | EMA smoothing factor (default: 0.1) |
| `k_bounds` | `Option<(usize, usize)>` | Min/max k bounds (default: [2, 10]) |
| `matcher` | `Option<MatcherConfig>` | Matcher configuration |
| `ensemble` | `Option<EnsembleConfigRequest>` | Ensemble configuration |

## New Dependencies

| Dependency | Version | Feature | Purpose |
|-----------|---------|---------|---------|
| `tree-sitter` | 0.24 | `code-matcher` | AST parsing for CodeMatcher |
| `tree-sitter-rust` | 0.23 | `code-matcher` | Rust grammar |
| `tree-sitter-python` | 0.23 | `code-matcher` | Python grammar |
| `tree-sitter-javascript` | 0.23 | `code-matcher` | JavaScript grammar |

The `code-matcher` feature is optional and not included in the default feature set.

## New Examples

```bash
cargo run --example coding_task    # Semantic matching on coding task
cargo run --example ensemble_demo  # Multi-model ensemble comparison
```

## New Benchmarks

```bash
cargo bench --bench coding_tasks        # Coding task benchmarks
cargo bench --bench math_logic          # Math & logic benchmarks
cargo bench --bench data_analysis       # Data analysis benchmarks
cargo bench --bench ensemble_comparison # Ensemble comparison
```

See [BENCHMARKS.md](./BENCHMARKS.md) for results.
