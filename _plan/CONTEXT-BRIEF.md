# Context Brief: MAKER Framework

> Read this file first to understand the project before executing the plan.

## What This Project Is

MAKER (Maximal Agentic decomposition, first-to-ahead-by-K Error correction, Red-flagging) is a Rust implementation of a framework for zero-error long-horizon LLM task execution. It solves the fundamental problem that LLM agents fail >50% on tasks with 100+ steps and 100% on million-step tasks. MAKER exposes SPRT-based voting and red-flagging protocols as an MCP server that augments coding assistants like Claude Code with mathematically-grounded error correction.

## Current Status

**v0.1.0 MVP: RELEASED** (Sprints 1-3 complete, 132 story points delivered)

All core functionality is implemented and working:
- Core algorithms (k_min, voting, red-flagging, orchestration)
- 4 MCP tools (vote, validate, calibrate, configure)
- 3 LLM providers (Ollama, OpenAI, Anthropic)
- Event-driven observability (logging, metrics)
- 10-disk Towers of Hanoi demo (1,023 steps, zero errors)
- 94.22% line coverage / 95.45% function coverage

**v0.2.0 Post-MVP: PLANNED** (Sprints 4-7, 93 story points estimated)

## Key Decisions Already Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Language** | Rust 2021 edition | Zero-cost abstractions, fearless concurrency for parallel sampling |
| **MCP SDK** | rmcp v0.13.0+ | Official Rust MCP implementation |
| **Async Runtime** | Tokio | rmcp dependency, mature ecosystem |
| **LLM Providers** | Ollama, OpenAI, Anthropic | Local + cloud hybrid support |
| **Architecture** | Event-driven | Observability, extensibility, testability |
| **Voting Strategy** | First-to-ahead-by-k (SPRT) | Mathematically optimal stopping |
| **Error Handling** | Red-flag (discard) over repair | Maintains error decorrelation |
| **Decomposition** | m=1 (microagent) | Minimizes context, maximizes reliability |
| **Test Coverage** | 95% mandatory | CI/CD enforced |
| **License** | MIT | Maximum community adoption |

## Terminology

| Term | Definition |
|------|------------|
| **MAKER** | Maximal Agentic decomposition, first-to-ahead-by-K Error correction, Red-flagging |
| **SPRT** | Sequential Probability Ratio Test — optimal statistical stopping rule |
| **k-margin** | Required vote lead for declaring winner (k votes ahead) |
| **k_min** | Minimum k to achieve target reliability: ⌈ln(t^(-m/s)-1) / ln((1-p)/p)⌉ |
| **p** | Per-step success probability (model's base accuracy) |
| **t** | Target task reliability (e.g., 0.95 for 95%) |
| **s** | Total steps in task |
| **m** | Subtasks per agent (MAKER enforces m=1) |
| **Red-flagging** | Discarding malformed responses without repair |
| **Microagent** | Agent handling exactly one subtask (m=1) |
| **State transfer** | System uses next_state from agent, not model's interpretation |
| **MCP** | Model Context Protocol — standard for AI tool integration |
| **Θ(s ln s)** | MAKER's cost complexity (log-linear, not exponential) |
| **Adaptive K** | Dynamic k-margin adjustment based on observed error rates (v0.2.0) |
| **Semantic Matching** | Grouping equivalent but textually different responses for voting (v0.2.0) |
| **Ensemble Voting** | Voting across heterogeneous LLM models for error decorrelation (v0.2.0) |

## File Map

| File | Contains | Read When |
|------|----------|-----------|
| `PROJECT-CONTEXT.md` | Original problem statement, stakeholders, constraints | Understanding scope |
| `PROJECT-PLAN.md` | Phase structure, ROI, success criteria | Planning work order |
| `JIRA-STRUCTURE.md` | Epic/story details, acceptance criteria | Detailed task reference |
| `_research/DOMAIN-RESEARCH.md` | Frameworks, terminology, citations | Making design decisions |
| `_research/DEPENDENCY-ANALYSIS.md` | Critical path, bottlenecks | Understanding blockers |
| `RACI-CHART.md` | Role assignments | Knowing who owns what |
| `RISK-REGISTER.md` | Risk mitigations | Encountering issues |
| `SEVERITY-CLASSIFICATION.md` | Priority framework | Triaging problems |
| `BEST-PRACTICES.md` | Industry standards, Rust patterns | Making design decisions |
| `SUCCESS-METRICS.md` | KPIs and closure criteria | Verifying completion |
| `docs/technical-implementation-manual.txt` | Paper-based implementation guidance | Algorithm reference |

## Codebase Map (v0.1.0)

| Module | Path | Purpose |
|--------|------|---------|
| **k_min** | `src/core/kmin.rs` | SPRT-based k-margin calculation |
| **Voting** | `src/core/voting.rs` | VoteRace with first-to-ahead-by-k |
| **Red-flagging** | `src/core/redflag.rs` | Discard-don't-repair validation |
| **Orchestration** | `src/core/orchestration.rs` | TaskDecomposer, m=1 enforcement |
| **Executor** | `src/core/executor.rs` | vote_with_margin integration loop |
| **Events** | `src/events/mod.rs` | MakerEvent enum, EventBus |
| **Logging** | `src/events/observers/logging.rs` | Tracing-based structured logging |
| **Metrics** | `src/events/observers/metrics.rs` | Counters and histograms |
| **LLM trait** | `src/llm/mod.rs` | LlmClient trait (async, object-safe) |
| **Ollama** | `src/llm/ollama.rs` | Local Ollama HTTP client |
| **OpenAI** | `src/llm/openai.rs` | OpenAI chat completions client |
| **Anthropic** | `src/llm/anthropic.rs` | Anthropic messages API client |
| **Retry** | `src/llm/retry.rs` | Exponential backoff with jitter |
| **Sampler** | `src/llm/sampler.rs` | Parallel sampling with JoinSet |
| **MCP Server** | `src/mcp/server.rs` | rmcp ServerHandler, shared state |
| **Vote Tool** | `src/mcp/tools/vote.rs` | maker/vote MCP tool |
| **Validate Tool** | `src/mcp/tools/validate.rs` | maker/validate MCP tool |
| **Calibrate Tool** | `src/mcp/tools/calibrate.rs` | maker/calibrate MCP tool |
| **Configure Tool** | `src/mcp/tools/configure.rs` | maker/configure MCP tool |
| **Hanoi Example** | `examples/hanoi/` | Towers of Hanoi task decomposition |

## Constraints and Guardrails

- **Test Coverage**: 95% minimum mandatory — CI blocks PRs below this
- **Budget**: Open source / community-funded
- **Technology**: Rust 2021, Tokio async, rmcp SDK
- **LLM APIs**: Must support Ollama (local), OpenAI, Anthropic
- **Compliance**: MIT License
- **Quality**: All algorithms must match arxiv paper 2511.09030 specifications
- **Backward Compatibility**: v0.2.0 must not break v0.1.0 public API

## Post-MVP Risks to Watch

| Risk | Mitigation |
|------|------------|
| **Semantic matching accuracy** | AST-based matching, embedding thresholds, domain test suites |
| **Adaptive k instability** | EMA smoothing, bounded k range, simulation validation |
| **Multi-model cost explosion** | Cost-aware routing: cheap first, expensive on disagreement |
| **Embedding latency overhead** | Cache embeddings, async comparison |
| **Backward compatibility breaks** | Trait-based extensibility, default to exact match, feature flags |

## How to Start

### For MVP work (already complete):
v0.1.0 is released. All Sprint 1-3 items are done.

### For post-MVP work (Sprints 4-7):
1. Read `_plan/EXECUTION-PLAN.md` — start at "Phase 4: Adaptive K-Margin"
2. Or read `_plan/RUNSHEET.md` — start at "Sprint 4"
3. Use `_plan/TASK-MANIFEST.json` for programmatic tracking (filter by `"status": "pending"`)
4. Sprint order: Adaptive K (4) → Semantic Matching (5) → Ensemble (6) → Benchmarks & Release (7)

## Key Commands

```bash
# Build
cargo build

# Test with coverage
cargo llvm-cov --html

# Run all tests
cargo test

# Check formatting
cargo fmt --check && cargo clippy

# Run MCP server
cargo run --bin maker-mcp

# Run Towers of Hanoi demo
cargo run --example hanoi_demo -- --disks 10
```

---

**Project Status:** v0.1.0 Released | v0.2.0 Planned (Sprints 4-7)
**v0.1.0 Released:** 2026-01-30
**Target v0.2.0:** Week 10
