# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MAKER (Massively decomposed Agentic processes with K-margin Error Reduction) is a Rust library and MCP server for zero-error long-horizon LLM agent execution. It implements SPRT-based voting, red-flag validation, and microagent orchestration to achieve mathematically-grounded error correction for LLM agents.

The crate exposes its capabilities both as a Rust library (`maker`) and as an MCP server binary (`maker-mcp`) using the `rmcp` SDK with stdio transport.

## Build & Test Commands

```bash
cargo build                          # Build library and binary
cargo test                           # Run all tests (unit + integration)
cargo test --lib                     # Unit tests only
cargo test --test properties         # Property-based tests (proptest)
cargo test --test mcp_integration    # MCP integration tests
cargo test <test_name>               # Run a single test by name
cargo clippy                         # Lint
cargo fmt --check                    # Check formatting
cargo run --bin maker-mcp            # Run the MCP server
```

## Architecture

### Core Algorithms (`src/core/`)

- **`kmin`** - Calculates minimum k-margin needed for target reliability given per-step success probability (p), target reliability (t), total steps (s), and steps-per-agent (m=1)
- **`voting`** - `VoteRace`: thread-safe first-to-ahead-by-k vote tracker using `Arc<Mutex<HashMap>>`. A winner is declared when one candidate leads all others by >= k_margin votes
- **`redflag`** - `RedFlagValidator`: discard-don't-repair validation. Checks token length, JSON schema conformance, and format rules. Returns `Vec<RedFlag>` (empty = valid)
- **`executor`** - `vote_with_margin()`: the main integration point. Orchestrates the sample-validate-vote loop: generate LLM sample -> red-flag check -> cast vote -> check winner. Synchronous loop (not async) despite async LLM clients
- **`orchestration`** - `TaskOrchestrator` and `TaskDecomposer` traits for microagent (m=1) task decomposition and state transfer between steps

### LLM Providers (`src/llm/`)

- **`LlmClient` trait** (in `mod.rs`) - async trait using `Pin<Box<dyn Future>>` for object safety. Defines `generate(prompt, temperature) -> Result<LlmResponse, LlmError>`
- **Provider implementations** - `ollama`, `openai`, `anthropic` modules
- **`retry`** - Retry logic with backoff for transient LLM errors
- **`sampler`** - Temperature-diverse sampling (T=0 for first sample, T=diversity for rest)

Note: There are two `LlmClient` traits. The async one in `src/llm/mod.rs` is for real provider integration. The synchronous one in `src/core/executor.rs` is a simpler interface used by the voting engine directly.

### MCP Server (`src/mcp/`)

- **`server`** - `MakerServer` implements rmcp's `ServerHandler` using `#[tool_router]` and `#[tool_handler]` macros. Shared state via `Arc<ServerState>` with `RwLock<ServerConfig>`
- **`tools/`** - Four MCP tools: `maker/vote`, `maker/validate`, `maker/calibrate`, `maker/configure`. Each has Request/Response structs with serde

### Event System (`src/events/`)

- **`MakerEvent`** enum (serde-tagged with `#[serde(tag = "type")]`) for observability: SampleRequested, SampleCompleted, RedFlagTriggered, VoteCast, VoteDecided, StepCompleted
- **`EventBus`** - tokio broadcast channel for fan-out to observers
- **`observers/`** - `LoggingObserver` (tracing) and `MetricsObserver` (counters/gauges)

### Key Dependencies

- `rmcp` (0.13) - MCP SDK with `server`, `transport-io`, `macros` features
- `tokio` - Async runtime (full features)
- `proptest` - Property-based testing (dev-dependency)

### Examples

- `examples/hanoi/` - Tower of Hanoi demonstration with task decomposition, showing the m=1 microagent pattern in practice
