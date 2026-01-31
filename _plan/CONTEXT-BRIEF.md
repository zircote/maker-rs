# Context Brief: MAKER Framework v0.3.0

> Read this file first to understand the project before executing the plan.

## What This Project Is

MAKER (Maximal Agentic decomposition, first-to-ahead-by-K Error correction, Red-flagging) is a Rust implementation of a framework for zero-error long-horizon LLM task execution. It solves the fundamental problem that LLM agents fail >50% on tasks with 100+ steps and 100% on million-step tasks. MAKER exposes SPRT-based voting and red-flagging protocols as an MCP server and standalone CLI that augments coding assistants like Claude Code with mathematically-grounded error correction.

## Current Status

**v0.1.0 + v0.2.0: COMPLETE**

All core functionality is implemented and working:
- Core algorithms (k_min, voting, red-flagging, orchestration)
- 4 MCP tools (vote, validate, calibrate, configure)
- 3 LLM providers (Ollama, OpenAI, Anthropic)
- Adaptive k-margin with EMA-based estimation
- Semantic matching (Exact, Embedding, Code AST)
- Multi-model ensemble (RoundRobin, CostAware, ReliabilityWeighted)
- Event-driven observability (logging, metrics)
- 456+ tests, 95%+ coverage

**v0.3.0: IN PROGRESS**

Focus areas:
1. **Recursive Decomposition** — Full automation of insight/execution agent separation
2. **Production Hardening** — Standalone CLI, async executor, operational tooling
3. **Extended Domain Support** — Domain-specific decomposers for coding, ML, data

## Key Decisions Already Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Language** | Rust 2021 edition | Zero-cost abstractions, fearless concurrency |
| **MCP SDK** | rmcp v0.13.0+ | Official Rust MCP implementation |
| **Async Runtime** | Tokio | rmcp dependency, mature ecosystem |
| **LLM Providers** | Ollama, OpenAI, Anthropic | Local + cloud hybrid support |
| **Architecture** | Event-driven | Observability, extensibility, testability |
| **Voting Strategy** | First-to-ahead-by-k (SPRT) | Mathematically optimal stopping |
| **Error Handling** | Red-flag (discard) over repair | Maintains error decorrelation |
| **Decomposition** | m=1 (microagent) | Minimizes context, maximizes reliability |
| **Test Coverage** | 95% mandatory | CI/CD enforced |
| **License** | MIT | Maximum community adoption |

## v0.3.0 Terminology

| Term | Definition |
|------|------------|
| **DecompositionAgent** | Splits tasks into subtasks with composition functions |
| **DecompositionDiscriminator** | Votes on proposed decomposition strategies |
| **ProblemSolverAgent** | Executes atomic (m=1) leaf nodes |
| **SolutionDiscriminator** | Aggregates results using voted composition function |
| **CompositionFunction** | How subtasks combine: Sequential, Parallel, Conditional, Custom |
| **Subtask** | A decomposed piece with task_id, parent_id, m_value, description |
| **DecompositionProposal** | Proposed task split with subtasks and composition_fn |
| **RecursiveOrchestrator** | Manages full decomposition→execution→composition pipeline |
| **LeafNodeExecutor** | Executes m=1 atomic subtasks with voting |
| **Depth Limit** | Maximum recursion depth (default 10) to prevent infinite loops |

## File Map

| File | Contains | Read When |
|------|----------|-----------|
| `PROJECT-CONTEXT.md` | Problem statement, stakeholders, v0.3.0 scope | Understanding scope |
| `_plan/TASK-PLAN-v0.3.0.md` | Epic/story details, acceptance criteria | Detailed task reference |
| `_plan/EXECUTION-PLAN.md` | Full work breakdown with context files | Executing work |
| `_plan/TASK-MANIFEST.json` | Machine-readable story list | Programmatic tracking |
| `_plan/RUNSHEET.md` | Milestone-by-milestone checklist | Progress tracking |
| `docs/SystemDesignSpecification.txt` | Architectural philosophy, Section 7 (Recursive) | Algorithm reference |
| `docs/project/technical-implementation-manual.txt` | Paper-based implementation guidance | Algorithm reference |

## Codebase Map (v0.2.0 → v0.3.0)

### Existing Modules (v0.2.0)

| Module | Path | Purpose |
|--------|------|---------|
| **k_min** | `src/core/kmin.rs` | SPRT-based k-margin calculation |
| **Voting** | `src/core/voting.rs` | VoteRace with first-to-ahead-by-k |
| **Red-flagging** | `src/core/redflag.rs` | Discard-don't-repair validation |
| **Adaptive** | `src/core/adaptive.rs` | KEstimator with EMA-based p-hat |
| **Matcher** | `src/core/matcher.rs` | CandidateMatcher trait |
| **Matchers** | `src/core/matchers/` | Embedding, Code (tree-sitter) |
| **Orchestration** | `src/core/orchestration.rs` | TaskDecomposer, m=1 enforcement |
| **Executor** | `src/core/executor.rs` | vote_with_margin integration loop |
| **Events** | `src/events/` | MakerEvent, EventBus, observers |
| **LLM** | `src/llm/` | Ollama, OpenAI, Anthropic, Ensemble |
| **MCP** | `src/mcp/` | Server, 4 tools (vote, validate, calibrate, configure) |

### New Modules (v0.3.0)

| Module | Path | Purpose |
|--------|------|---------|
| **Decomposition** | `src/core/decomposition/mod.rs` | DecompositionAgent trait, Subtask, CompositionFunction |
| **Discriminator** | `src/core/decomposition/discriminator.rs` | Vote on decomposition proposals |
| **Solver** | `src/core/decomposition/solver.rs` | LeafNodeExecutor for m=1 tasks |
| **Aggregator** | `src/core/decomposition/aggregator.rs` | Compose results by strategy |
| **Orchestrator** | `src/core/decomposition/orchestrator.rs` | RecursiveOrchestrator full pipeline |
| **Domains** | `src/core/decomposition/domains/` | Coding, ML, Data decomposers |
| **Filesystem** | `src/core/decomposition/filesystem.rs` | Multi-file state management |
| **Async Executor** | `src/core/async_executor.rs` | Async voting with cancellation |
| **CLI** | `src/bin/maker-cli.rs` | Standalone command-line interface |
| **Health** | `src/mcp/health.rs` | Health checks and Prometheus metrics |

## Constraints and Guardrails

- **Test Coverage**: 95% minimum mandatory — CI blocks PRs below this
- **Decomposition Depth**: Default limit 10, configurable
- **Timeout**: Default 60s for recursive operations
- **m=1 Enforcement**: Leaf nodes must have m_value == 1 (validated at runtime)
- **Backward Compatibility**: v0.3.0 must not break v0.2.0 public API
- **Quality**: All algorithms must match SystemDesignSpecification.txt Section 7

## v0.3.0 Risks to Watch

| Risk | Mitigation |
|------|------------|
| **Decomposition explosion** | Depth limits (10), cycle detection, timeout (60s) |
| **Voting on decomposition too expensive** | Cache proposals, reuse winning strategies |
| **Domain decomposers too task-specific** | Extensible `DecompositionAgent` trait |
| **Async integration breaks correctness** | Property tests comparing sync/async parity |
| **Claude Code API changes** | Version-pin MCP protocol, integration tests in CI |

## How to Start

### For v0.3.0 work:

1. Read `_plan/EXECUTION-PLAN.md` — start at "Milestone 1: Decomposition MVP"
2. Or read `_plan/RUNSHEET.md` — start at "Milestone 1"
3. Use `_plan/TASK-MANIFEST.json` for programmatic tracking (filter by `"status": "pending"`)

### Parallelism:
- **EPIC-011** (Recursive Decomposition) and **EPIC-012** (Production Hardening) can be worked in parallel
- **EPIC-013** (Domain Support) requires EPIC-011 completion first

### Dependency Order within EPIC-011:
```
011-01 (Framework) → 011-02 (Discriminator) ─┐
                  → 011-03 (Solver) ─────────┴→ 011-04 (Aggregation) → 011-05 (Orchestration)
```

## Key Commands

```bash
# Build
cargo build

# Test with coverage
cargo llvm-cov --html

# Run all tests
cargo test

# Test decomposition module (once created)
cargo test decomposition

# Check formatting
cargo fmt --check && cargo clippy

# Run MCP server
cargo run --bin maker-mcp

# Run CLI (once created)
cargo run --bin maker-cli -- --help

# Run Towers of Hanoi demo
cargo run --example hanoi_demo -- --disks 10
```

## Success Criteria (v0.3.0)

| Criterion | Target | Verification |
|-----------|--------|--------------|
| Test coverage | ≥ 95% | `cargo llvm-cov --html` |
| Recursive depth | ≥ 5 levels | Integration test with nested decomposition |
| CLI feature parity | 100% MCP tools | CLI integration tests |
| Domain decomposers | 3 (coding, ML, data) | Domain-specific integration tests |
| Async correctness | Parity with sync | Property tests, benchmark comparison |
| All existing tests | Pass | `cargo test` |

---

**Project Status:** v0.2.0 Complete | v0.3.0 In Progress
**Previous Releases:** v0.1.0 (2026-01-30), v0.2.0 (2026-01-31)
**Target:** v0.3.0 — Recursive Decomposition
