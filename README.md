# MAKER Framework

**Zero-Error Long-Horizon LLM Execution via SPRT Voting, Red-Flagging, and Microagent Orchestration**

[![Build Status](https://img.shields.io/github/actions/workflow/status/zircote/maker-rs/ci.yml?branch=main)](https://github.com/zircote/maker-rs/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

> ## ⚠️ EXPERIMENTAL - NOT FOR PRODUCTION USE
>
> **This is a research experiment, not production software.**
>
> - **Purpose**: Validate and explore claims from the research paper [Solving a Million-Step LLM Task with Zero Errors](https://arxiv.org/abs/2511.09030) (Meyerson et al., 2025)
> - **Goals**: Implement the MAKER algorithm, discover alternative approaches, test boundaries of zero-error LLM execution
> - **Status**: Active research - may never reach production readiness
> - **Use at your own risk**: APIs will change, results are experimental, not suitable for critical applications

MAKER (Massively decomposed Agentic processes with K-margin Error Reduction) is a Rust implementation exploring mathematically-grounded error correction in LLM agents. It tests zero-error execution through SPRT-based voting, red-flag validation, and m=1 microagent decomposition.

Based on: [Solving a Million-Step LLM Task with Zero Errors](https://arxiv.org/abs/2511.09030) (Meyerson et al., 2025)

## The Problem

Even with 99% per-step accuracy, a 1,000-step task has only a 0.004% success rate. The MAKER algorithm aims to transform this into 95%+ reliability with logarithmic cost scaling.

**This implementation explores whether that claim holds in practice.**

| Task Length | Naive Success Rate | MAKER Success Rate | MAKER Cost Scaling |
|------------|-------------------|-------------------|-------------------|
| 7 steps    | 93%               | 99%+              | 21 samples        |
| 1,023 steps| 0%                | 95%+              | ~6,138 samples    |
| 1M steps   | 0%                | 95%+              | Θ(s ln s)         |

## Quick Start

### As a Rust Library

```rust
use maker::core::{calculate_kmin, MockLlmClient, VoteConfig, vote_with_margin};

// Calculate required k-margin for your task
let k = calculate_kmin(
    0.85,   // p: per-step success probability
    0.95,   // t: target task reliability
    1_023,  // s: total steps (10-disk Hanoi)
    1,      // m: steps per agent (must be 1)
).unwrap();

// Run error-corrected voting
let client = MockLlmClient::constant("correct_answer");
let config = VoteConfig::default();
let result = vote_with_margin("What is 2+2?", k, &client, config).unwrap();
println!("Winner: {} ({} samples)", result.winner, result.total_samples);
```

### As an MCP Server

```bash
# Build and run
cargo build --release
cargo run --bin maker-mcp
```

Add to your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "maker": {
      "command": "/path/to/maker-mcp",
      "args": [],
      "env": {
        "OPENAI_API_KEY": "your-key",
        "ANTHROPIC_API_KEY": "your-key"
      }
    }
  }
}
```

### Run the Demos

See [Validation Demos](#validation-demos) below for detailed documentation and results.

## Validation Demos

These demos validate MAKER's error correction capabilities against different task types.

### Hanoi Demo (`examples/hanoi_demo.rs`)

**Purpose**: Validates MAKER on multi-step reasoning tasks requiring algorithmic thinking.

```bash
# 3-disk Hanoi (7 steps) with OpenAI
cargo run --example hanoi_demo -- --disks 3 --provider openai

# 5-disk Hanoi (31 steps) with strict mode (halt on first error)
cargo run --example hanoi_demo -- --disks 5 --provider openai --strict

# With ensemble (multiple providers)
cargo run --example hanoi_demo -- --disks 3 --provider openai --ensemble

# Mock mode for CI/testing
MAKER_USE_MOCK=1 cargo run --example hanoi_demo -- --disks 10 --accuracy 0.85
```

**Key flags**:
| Flag | Description |
|------|-------------|
| `--disks N` | Number of disks (1-20) |
| `--provider` | LLM provider: `ollama`, `openai`, `anthropic` |
| `--model` | Model name override |
| `--strict` | Halt on first error (true zero-error mode) |
| `--ensemble` | Enable multi-provider ensemble |

**Results** (gpt-5-mini):
- 31/31 steps (5-disk) with **0 errors** using few-shot + chain-of-thought prompt
- ~2.7 samples per step average
- p_hat converges to 0.950 (target reliability)

**Key Finding**: Raw prompts fail on step 2 (systematic reasoning errors). Few-shot examples + chain-of-thought prompting converts systematic errors into random errors that voting can correct.

### Arithmetic Demo (`examples/arithmetic_demo.rs`)

**Purpose**: Validates MAKER on tasks with truly random errors (calculation mistakes).

```bash
# 20 problems with OpenAI
cargo run --example arithmetic_demo -- --problems 20 --provider openai

# 50 problems with higher difficulty
cargo run --example arithmetic_demo -- --problems 50 --provider openai --difficulty 3

# Strict mode with reproducible seed
cargo run --example arithmetic_demo -- --problems 50 --provider openai --strict --seed 42

# Mock mode for CI/testing
MAKER_USE_MOCK=1 cargo run --example arithmetic_demo -- --problems 100 --accuracy 0.90
```

**Key flags**:
| Flag | Description |
|------|-------------|
| `--problems N` | Number of problems to solve |
| `--difficulty 1-5` | Number magnitude (10^difficulty) |
| `--provider` | LLM provider |
| `--strict` | Halt on first error |
| `--seed N` | Reproducible problem generation |

**Results** (gpt-5-mini):
- 50/50 problems correct with **0 errors**
- 2.7 samples per problem average
- Handles addition, subtraction, multiplication

**Key Finding**: Random calculation errors are effectively corrected by voting. This validates MAKER's core premise - voting corrects random errors.

---

## Research Findings

Our validation experiments reveal important insights about MAKER's applicability:

### What MAKER Can Correct

| Error Type | Example | Correctable? | Why |
|------------|---------|--------------|-----|
| **Random errors** | LLM occasionally miscalculates 73-38 | ✅ Yes | Independent errors cancel out through voting |
| **Prompted reasoning** | Hanoi with few-shot examples | ✅ Yes | Converts systematic to random errors |

### What MAKER Cannot Correct

| Error Type | Example | Correctable? | Why |
|------------|---------|--------------|-----|
| **Systematic errors** | LLM can't reason about Hanoi | ❌ No | All samples fail the same way |
| **Knowledge gaps** | LLM doesn't know an algorithm | ❌ No | Voting achieves consensus on wrong answer |

### Key Insight

> **MAKER corrects random errors, not systematic reasoning failures.**
>
> If an LLM consistently fails at a task, voting will achieve consensus on the wrong answer. Prompt engineering (few-shot examples, chain-of-thought) is critical to convert systematic failures into random errors that voting can correct.

### Observed Accuracy

| Task | Model | Per-Step Accuracy | Samples/Step |
|------|-------|-------------------|--------------|
| Arithmetic (difficulty 3) | gpt-5-mini | ~95% | 2.7 |
| Hanoi (few-shot+CoT) | gpt-5-mini | ~95% | 2.7 |
| Hanoi (raw prompt) | gpt-5-mini | <50% | N/A (fails) |

---

## Research Validation

`maker-rs` explores the boundary between academic research and practical implementation. Here's how this experiment compares to the [original paper](https://arxiv.org/abs/2511.09030):

### Scorecard (Self-Assessment)

| Criterion | Score | Assessment |
|-----------|-------|------------|
| **Algorithm Fidelity** | A+ | Exact k_min formula, SPRT voting, strict m=1 enforcement |
| **Experimental Coverage** | A | Tokio concurrency, event sourcing, MCP integration |
| **Completeness** | B | Lacks automated decomposition (future exploration) |
| **Validation Status** | B+ | Demos validate core claims; edge cases need more testing |

### What's Implemented Well

- **Strict m=1 Decomposition**: Micro-agents execute exactly one step with minimal context, matching the paper's core assertion that "smallest possible subtasks" enable scaling
- **SPRT-Based Voting**: Uses actual Sequential Probability Ratio Test and Gambler's Ruin logic—not heuristic "best of 3"
- **Dynamic k_min Calculation**: Computes margin from the paper's logarithmic formula based on target reliability (t) and task length (s)
- **Red-Flagging as Primitive**: Treats validation as statistical necessity to decorrelate errors, discarding (never repairing) malformed outputs

### Implementation Details

Beyond the paper's theoretical model (experimental additions):

| Enhancement | Purpose |
|-------------|---------|
| **Tokio Runtime** | Massive I/O concurrency for parallel vote sampling across thousands of steps |
| **Event Sourcing** | Real-time observability bridging theoretical probability with practical debugging |
| **MCP Integration** | Transforms abstract state machine into consumable tools for Claude Code and other clients |
| **Exponential Backoff** | Handles API rate limits (429s) with jitter and circuit breakers |

### Known Limitations

Deferred to post-MVP (acknowledged gaps vs. paper's full vision):

- **Automated Decomposition**: The paper's "Insight Agents" for recursive task discovery are not yet implemented—decomposition is currently manual/deterministic
- **Semantic Matching**: MVP defaults to exact string matching; embedding-based and AST-based matchers are available but less battle-tested

### Practical Compromises

| Paper Assumption | Implementation Reality |
|------------------|----------------------|
| Idealized sampling | Backoff, circuit breakers, retry budgets for real API constraints |
| Red-flagging for decorrelation | Dual-purposed as security guardrail against prompt injection |
| Unlimited parallelism | Configurable concurrency limits to respect provider quotas |

Run `cargo test --test monte_carlo` to validate the statistical guarantees empirically.

## Architecture

```
src/
├── core/               # Core MAKER algorithms
│   ├── kmin.rs         # k_min = ⌈ln(1 - t^(m/s)) / ln((1-p)/p)⌉
│   ├── voting.rs       # VoteRace: first-to-ahead-by-k (thread-safe)
│   ├── redflag.rs      # RedFlagValidator: discard-don't-repair
│   ├── executor.rs     # vote_with_margin() + vote_with_margin_adaptive()
│   ├── adaptive.rs     # KEstimator: EMA-based dynamic k adjustment
│   ├── matcher.rs      # CandidateMatcher trait + ExactMatcher
│   ├── matchers/       # Pluggable matcher implementations
│   │   ├── embedding.rs        # EmbeddingMatcher (cosine similarity)
│   │   ├── ollama_embedding.rs # Ollama embedding client
│   │   ├── openai_embedding.rs # OpenAI embedding client
│   │   └── code.rs             # CodeMatcher (tree-sitter AST, optional)
│   └── orchestration.rs# TaskOrchestrator with m=1 constraint
├── llm/                # Multi-provider LLM abstraction
│   ├── ollama.rs       # Local inference
│   ├── openai.rs       # OpenAI API
│   ├── anthropic.rs    # Anthropic API
│   ├── ensemble.rs     # Multi-model ensemble configuration
│   ├── retry.rs        # Exponential backoff with jitter
│   └── sampler.rs      # Temperature-diverse parallel sampling
├── mcp/                # MCP server (rmcp v0.13)
│   ├── server.rs       # MakerServer with #[tool_router]
│   └── tools/          # maker/vote, validate, calibrate, configure
└── events/             # Event-driven observability
    ├── bus.rs           # Tokio broadcast channel
    └── observers/       # Logging (tracing) + Metrics
```

## MCP Tools

### `maker/vote` - Error-Corrected Voting

Request:
```json
{ "prompt": "...", "k_margin": 3, "max_samples": 20, "matcher": "embedding" }
```
Response:
```json
{
  "winner": "answer", "vote_counts": {"answer": 5}, "total_samples": 7,
  "k_used": 3, "p_hat": 0.87, "matcher_type": "exact", "candidate_groups": 2
}
```

### `maker/validate` - Red-Flag Checking

Request:
```json
{ "response": "...", "token_limit": 700, "schema": {"required": ["move"]} }
```
Response:
```json
{ "valid": false, "red_flags": [{"flag_type": "TokenLengthExceeded", "details": "..."}] }
```

### `maker/calibrate` - Success Rate Estimation

Request:
```json
{ "samples": [{"prompt": "...", "ground_truth": "..."}] }
```
Response:
```json
{ "p_estimate": 0.87, "confidence_interval": [0.82, 0.92], "recommended_k": 4 }
```

### `maker/configure` - Runtime Configuration

Request:
```json
{
  "k_default": 3, "temperature_diversity": 0.1, "token_limit": 700,
  "adaptive_k": true, "ema_alpha": 0.1, "k_bounds": [2, 10],
  "matcher": { "type": "embedding", "threshold": 0.92, "provider": "ollama" }
}
```
Response:
```json
{ "applied": true, "current_config": {} }
```

## Cost Scaling

MAKER's cost scales as Θ(s ln s) vs. exponential for naive retry:

| Approach | 7 steps | 1,023 steps | 1M steps |
|----------|---------|-------------|----------|
| **MAKER** | 21 samples | ~6K samples | ~20M samples |
| **Naive retry** | 7 attempts | Infeasible | Impossible |

### MAKER vs Naive Retry (Monte Carlo validated, p=0.85, t=0.95)

| Steps (s) | MAKER Cost | Naive Retry Cost | Savings |
|-----------|-----------|-----------------|---------|
| 20        | 80        | 1,520           | 94.7%   |
| 50        | 200       | 506,400         | 99.96%  |
| 100       | 500       | 3.4 billion     | ~100%   |

Naive retry must rerun the entire task on any step failure. With p=0.85, the probability of completing 100 steps without error is 0.85^100 ≈ 7×10⁻⁸, requiring billions of retries. MAKER's per-step voting keeps costs logarithmic.

Run `cargo test --test monte_carlo -- --nocapture` to reproduce these results.

## Adaptive K-Margin

MAKER dynamically adjusts the k-margin based on observed accuracy, reducing API calls when the model is performing well:

```rust
use maker::core::{KEstimator, vote_with_margin_adaptive, MockLlmClient, VoteConfig};

let client = MockLlmClient::constant("answer");
let config = VoteConfig::default();
let mut estimator = KEstimator::new(0.85, 0.95, 100);

// k starts high, decreases as accuracy is confirmed
let result = vote_with_margin_adaptive("prompt", &client, config, &mut estimator).unwrap();
println!("Used k={}, estimated p={:.2}", result.k_used, estimator.p_hat());
```

Configure via MCP: `{"adaptive_k": true, "ema_alpha": 0.1, "k_bounds": [2, 10]}`

## Semantic Matching

For non-deterministic tasks (code generation, natural language), MAKER supports pluggable matchers that group equivalent responses:

| Matcher | Use Case | Method |
|---------|----------|--------|
| `ExactMatcher` (default) | Deterministic tasks | Whitespace-normalized string equality |
| `EmbeddingMatcher` | Natural language | Cosine similarity of embeddings (threshold: 0.92) |
| `CodeMatcher` | Code generation | Tree-sitter AST comparison with alpha-renaming |

```rust
use maker::core::matcher::ExactMatcher;
use maker::core::matchers::embedding::{EmbeddingMatcher, MockEmbeddingClient};

// Embedding matcher groups semantically similar responses
let matcher = EmbeddingMatcher::new(Box::new(MockEmbeddingClient::default()), 0.92);
assert!(matcher.are_equivalent("The answer is 42", "The answer is 42."));
```

The `CodeMatcher` requires the `code-matcher` feature flag:

```bash
cargo build --features code-matcher
cargo test --features code-matcher
```

## Multi-Model Ensemble

MAKER supports voting across heterogeneous LLM models to decorrelate errors by model architecture, not just sampling temperature:

| Strategy | Behavior | Use Case |
|----------|----------|----------|
| `RoundRobin` | Distribute samples evenly across models | Maximize diversity |
| `CostAware` | Cheap models first, escalate on disagreement | Minimize cost |
| `ReliabilityWeighted` | More samples from higher-reliability models | Optimize accuracy |

Configure via MCP:

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

Cost-aware ensemble saves 87.5%+ vs single expensive model. See [BENCHMARKS.md](./docs/project/BENCHMARKS.md) for full results.

## Benchmarks

MAKER includes domain-specific benchmarks covering coding tasks, math/logic, and data analysis:

```bash
cargo bench --bench coding_tasks        # 10 coding tasks (trivial to complex)
cargo bench --bench math_logic          # Arithmetic, symbolic, logic, Hanoi
cargo bench --bench data_analysis       # CSV, statistics, SQL, data cleaning
cargo bench --bench cost_scaling        # Θ(s ln s) cost validation
cargo bench --bench ensemble_comparison # Single-model vs ensemble comparison
```

See [BENCHMARKS.md](./docs/project/BENCHMARKS.md) for detailed results and acceptance criteria.

## Development

```bash
cargo build                          # Build
cargo test                           # All tests (unit + integration + property)
cargo test --features code-matcher   # Include tree-sitter code matcher tests
cargo test --example hanoi           # Hanoi example tests (21 tests)
cargo test --test properties         # Property-based tests (proptest, 21 tests)
cargo test --test mcp_integration    # MCP integration tests (35 tests)
cargo test --test semantic_matching  # Semantic matching tests (16/25 tests)
cargo test --test monte_carlo        # Monte Carlo cost validation
cargo bench --bench cost_scaling     # Cost scaling benchmark
cargo bench --bench coding_tasks    # Coding task benchmark
cargo bench --bench math_logic      # Math & logic benchmark
cargo bench --bench data_analysis   # Data analysis benchmark
cargo bench --bench ensemble_comparison # Ensemble comparison
cargo clippy                         # Lint
cargo fmt --check                    # Format check
cargo doc --no-deps --open           # API documentation
```

## Security

MAKER implements defense-in-depth for MCP tool security:

- **Schema enforcement**: `#[serde(deny_unknown_fields)]` on all inputs
- **Red-flag filtering**: Malformed LLM outputs discarded, never repaired
- **Prompt limits**: 10,000 character maximum
- **Microagent isolation**: No history leakage between steps (m=1)
- **State hash validation**: Corruption detected before state transfer

See [SECURITY.md](./SECURITY.md) for vulnerability reporting.

## References

1. Meyerson, E., et al. (2025). *Solving a Million-Step LLM Task with Zero Errors*. [arXiv:2511.09030](https://arxiv.org/abs/2511.09030)
2. Anthropic. (2024). *Introducing the Model Context Protocol*. [anthropic.com](https://www.anthropic.com/news/model-context-protocol)
3. Wald, A. (1945). *Sequential Analysis*. (SPRT foundational work)

## License

MIT - see [LICENSE](./LICENSE)
