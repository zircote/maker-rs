# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MAKER (Massively decomposed Agentic processes with K-margin Error Reduction) is a Rust library and MCP server for zero-error long-horizon LLM agent execution. It implements SPRT-based voting, red-flag validation, and microagent orchestration to achieve mathematically-grounded error correction for LLM agents.

The crate exposes its capabilities as:
- A Rust library (`maker`)
- An MCP server binary (`maker-mcp`) using the `rmcp` SDK with stdio transport
- A standalone CLI (`maker-cli`) with feature parity to MCP tools

Based on: [Solving a Million-Step LLM Task with Zero Errors](https://arxiv.org/abs/2511.09030) (Meyerson et al., 2025)

## Build & Test Commands

```bash
cargo build                          # Build library and binaries
cargo test                           # Run all tests (unit + integration)
cargo test --lib                     # Unit tests only
cargo test --test properties         # Property-based tests (proptest)
cargo test --test mcp_integration    # MCP integration tests
cargo test --test semantic_matching  # Semantic matching tests
cargo test --test cli_functional     # CLI functional tests
cargo test --test error_correction_validation  # Error correction validation
cargo test --features code-matcher   # Include tree-sitter code matcher tests
cargo test <test_name>               # Run a single test by name
cargo clippy                         # Lint
cargo fmt --check                    # Check formatting
cargo run --bin maker-mcp            # Run the MCP server
cargo run --bin maker-cli -- --help  # CLI help
```

## Environment Variables

| Variable | Required For | Description |
|----------|-------------|-------------|
| `OPENAI_API_KEY` | `openai` provider | OpenAI API key |
| `ANTHROPIC_API_KEY` | `anthropic` provider | Anthropic API key |
| Ollama | None | Uses `http://localhost:11434` by default |

## Architecture

### Core Algorithms (`src/core/`)

- **`kmin`** - Calculates minimum k-margin needed for target reliability given per-step success probability (p), target reliability (t), total steps (s), and steps-per-agent (m=1)
- **`voting`** - `VoteRace`: thread-safe first-to-ahead-by-k vote tracker using `Arc<Mutex<HashMap>>`. Accepts `Arc<dyn CandidateMatcher>` for pluggable candidate grouping. A winner is declared when one candidate leads all others by >= k_margin votes
- **`redflag`** - `RedFlagValidator`: discard-don't-repair validation. Checks token length, JSON schema conformance, and format rules. Returns `Vec<RedFlag>` (empty = valid)
- **`executor`** - `vote_with_margin()`: the main integration point. Orchestrates the sample-validate-vote loop. `vote_with_margin_adaptive()` adds dynamic k adjustment via `KEstimator`
- **`adaptive`** - `KEstimator`: EMA-based p-hat estimation with `recommended_k()`. Configurable bounds (floor/ceiling) and smoothing alpha
- **`matcher`** - `CandidateMatcher` trait with `canonicalize()`, `are_equivalent()`, `similarity_score()`. `ExactMatcher` is the default
- **`orchestration`** - `TaskOrchestrator` struct and `TaskDecomposer` trait for microagent (m=1) task decomposition and state transfer

### Matchers (`src/core/matchers/`)

Pluggable candidate grouping strategies:
- **`EmbeddingMatcher`** - Cosine similarity with caching
- **`OllamaEmbeddingClient`** / **`OpenAiEmbeddingClient`** - Embedding providers
- **`CodeMatcher`** - Tree-sitter AST comparison (behind `code-matcher` feature flag)
- **`MatcherPreset`** - Domain presets: `code_generation`, `summarization`, `chat`, `extraction`, `classification`, `reasoning`, `creative`
- **`MatcherFactory`** / **`PromptAnalyzer`** - Auto-detection of appropriate matcher from prompt content

### Decomposition (`src/core/decomposition/`)

Recursive task decomposition framework implementing Section 7 of the design spec:
- **`LlmDecompositionAgent`** - LLM-driven task decomposition
- **`DecompositionDiscriminator`** - Votes on proposed decomposition strategies
- **`LeafNodeExecutor`** - Executes atomic (m=1) leaf nodes with voting
- **`SolutionDiscriminator`** / **`aggregator`** - Aggregates results using voted composition functions
- **`RecursiveOrchestrator`** - Coordinates the full decompose-solve-aggregate cycle
- **`CompositionFunction`** - How subtask results combine (Sequential, Parallel, Conditional, Custom)

### LLM Providers (`src/llm/`)

- **`LlmClient` trait** (in `mod.rs`) - async trait using `Pin<Box<dyn Future>>` for object safety. Defines `generate(prompt, temperature) -> Result<LlmResponse, LlmError>`
- **`adapter`** - `BlockingLlmAdapter` bridges async providers to sync voting engine; `setup_provider_client()` factory for provider instantiation
- **Provider implementations** - `ollama`, `openai`, `anthropic` modules
- **`retry`** - Retry logic with backoff for transient LLM errors
- **`sampler`** - Temperature-diverse sampling (T=0 for first sample, T=diversity for rest)

Note: There are two `LlmClient` traits. The async one in `src/llm/mod.rs` is for real provider integration. The synchronous one in `src/core/executor.rs` is a simpler interface used by the voting engine directly.

### MCP Server (`src/mcp/`)

- **`server`** - `MakerServer` implements rmcp's `ServerHandler` using `#[tool_router]` and `#[tool_handler]` macros. Shared state via `Arc<ServerState>` with `RwLock<ServerConfig>`
- **`tools/`** - Six MCP tools:
  - `maker/vote` - SPRT voting on prompts
  - `maker/validate` - Red-flag validation
  - `maker/calibrate` - k-margin calibration from samples
  - `maker/configure` - Runtime configuration
  - `maker/decompose` - Recursive task decomposition
  - `maker/health` - Server health checks

### CLI (`src/bin/maker-cli.rs`)

Standalone CLI with feature parity to MCP tools:
```bash
maker-cli vote --prompt "..." --k-margin 3 --provider openai
maker-cli validate --response "..." --token-limit 1000
maker-cli calibrate --file samples.json --target-reliability 0.95
maker-cli config --show
maker-cli decompose --task "..." --provider anthropic
maker-cli health
maker-cli completions bash  # Shell completions
```

### Event System (`src/events/`)

- **`MakerEvent`** enum (serde-tagged with `#[serde(tag = "type")]`) for observability: SampleRequested, SampleCompleted, RedFlagTriggered, VoteCast, VoteDecided, StepCompleted
- **`EventBus`** - tokio broadcast channel for fan-out to observers
- **`observers/`** - `LoggingObserver` (tracing) and `MetricsObserver` (counters/gauges)

### Feature Flags

- `code-matcher` - Enables `CodeMatcher` with tree-sitter for AST-based code comparison
- `prometheus` - Enables Prometheus metrics export

### Key Dependencies

- `rmcp` (0.13) - MCP SDK with `server`, `transport-io`, `macros` features
- `tokio` (1) - Async runtime (full features)
- `reqwest` (0.12) - HTTP client for LLM provider API calls (includes `blocking` feature)
- `clap` (4) - CLI argument parsing with derive macros
- `serde` / `serde_json` - Serialization for requests, responses, and events
- `tracing` - Structured logging and observability
- `tree-sitter` + language grammars - AST parsing for CodeMatcher (optional)
- `proptest` - Property-based testing (dev-dependency)
- `wiremock` (0.6) - HTTP mock server for provider tests (dev-dependency)

### Examples

```bash
cargo run --example hanoi -- --disks 3        # 3-disk Hanoi (7 steps)
cargo run --example hanoi_demo -- --disks 10  # 10-disk with voting (1,023 steps)
cargo run --example custom_task               # Custom task template
cargo run --example coding_task               # Coding task demo
cargo run --example ensemble_demo             # Ensemble voting demo
```
