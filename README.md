# MAKER Framework

**Zero-Error Long-Horizon LLM Execution via SPRT Voting, Red-Flagging, and Microagent Orchestration**

[![Build Status](https://img.shields.io/github/actions/workflow/status/zircote/maker-rs/ci.yml?branch=main)](https://github.com/zircote/maker-rs/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

MAKER (Massively decomposed Agentic processes with K-margin Error Reduction) is a Rust library and MCP server for mathematically-grounded error correction in LLM agents. It achieves zero-error execution on 1,000+ step tasks through SPRT-based voting, red-flag validation, and m=1 microagent decomposition.

Based on: [Solving a Million-Step LLM Task with Zero Errors](https://arxiv.org/abs/2511.09030) (Meyerson et al., 2025)

## The Problem

Even with 99% per-step accuracy, a 1,000-step task has only a 0.004% success rate. MAKER transforms this into 95%+ reliability with logarithmic cost scaling.

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

### Run the Demo

```bash
# 3-disk Hanoi (7 steps)
cargo run --example hanoi -- --disks 3

# 10-disk Hanoi with voting (1,023 steps, zero errors)
cargo run --example hanoi_demo -- --disks 10 --accuracy 0.85

# Custom task template
cargo run --example custom_task
```

## Architecture

```
src/
├── core/               # Core MAKER algorithms
│   ├── kmin.rs         # k_min = ⌈ln(1-t^(1/s)) / ln((1-p)/p)⌉
│   ├── voting.rs       # VoteRace: first-to-ahead-by-k (thread-safe)
│   ├── redflag.rs      # RedFlagValidator: discard-don't-repair
│   ├── executor.rs     # vote_with_margin(): the main integration point
│   └── orchestration.rs# TaskOrchestrator with m=1 constraint
├── llm/                # Multi-provider LLM abstraction
│   ├── ollama.rs       # Local inference
│   ├── openai.rs       # OpenAI API
│   ├── anthropic.rs    # Anthropic API
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
{ "prompt": "...", "k_margin": 3, "max_samples": 20 }
```
Response:
```json
{ "winner": "answer", "vote_counts": {"answer": 5}, "total_samples": 7 }
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
{ "k_default": 3, "temperature_diversity": 0.1, "token_limit": 700 }
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

## Development

```bash
cargo build                      # Build
cargo test                       # All tests (unit + integration + property)
cargo test --example hanoi       # Hanoi example tests (21 tests)
cargo test --test properties     # Property-based tests (proptest)
cargo test --test mcp_integration # MCP integration tests
cargo test --test monte_carlo    # Monte Carlo cost validation
cargo bench --bench cost_scaling # Cost scaling benchmark
cargo clippy                     # Lint
cargo fmt --check                # Format check
cargo doc --no-deps --open       # API documentation
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
