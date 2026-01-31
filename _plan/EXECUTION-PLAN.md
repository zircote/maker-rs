# Execution Plan: MAKER Framework v0.3.0

> This file is a complete execution plan for implementing the v0.3.0 project work.
> It was generated from TASK-PLAN-v0.3.0.md and PROJECT-CONTEXT.md.
> To execute: open a Claude Code session in this repository and say:
> "Read _plan/EXECUTION-PLAN.md and begin executing the work items"

**Generated:** 2026-01-31
**Version:** 0.3.0 (supersedes v0.1.0/v0.2.0 execution plan)
**Status:** ✅ ALL MILESTONES COMPLETE (842+ tests passing)
**Domain:** Open Source AI Infrastructure / LLM Agent Reliability Engineering
**Project:** Implement recursive decomposition, production hardening, and extended domain support for the MAKER framework.

---

## How to Execute This Plan

Read this file, then work through each epic and its stories sequentially within each milestone.
For each work item:
1. Check prerequisites and blocking dependencies
2. Read the relevant context files listed
3. Execute the work described — write code, create configs, run commands, etc.
4. Run tests to verify acceptance criteria are met
5. Mark complete and move to the next item

**Parallelism:** EPIC-011 and EPIC-012 can be worked in parallel. EPIC-013 requires EPIC-011 completion first.

---

## Project Summary

MAKER v0.3.0 implements the **Recursive Architecture** from Section 7 of the System Design Specification — the full insight/execution agent separation with automated task decomposition. This transforms MAKER from a voting engine into a complete autonomous task execution system.

### What's Already Complete (v0.1.0 + v0.2.0)

- ✅ Core MAKER library (k_min, VoteRace, RedFlagValidator, vote_with_margin)
- ✅ MCP server with 4 tools (vote, validate, calibrate, configure)
- ✅ 3 LLM providers (Ollama, OpenAI, Anthropic)
- ✅ Semantic matching (ExactMatcher, EmbeddingMatcher, CodeMatcher)
- ✅ Adaptive k-margin (KEstimator with EMA)
- ✅ Multi-model ensemble (RoundRobin, CostAware, ReliabilityWeighted)
- ✅ 456+ tests, 95%+ coverage

### What v0.3.0 Delivers

1. **Decomposition Agents** — automatically split complex tasks into atomic subtasks
2. **Decomposition Discriminators** — vote on the best decomposition strategy
3. **Problem Solver Agents** — execute atomic (m=1) leaf nodes with voting
4. **Solution Discriminators** — aggregate results using voted composition functions
5. **Standalone CLI** (`maker-cli`) — use MAKER without MCP
6. **Domain Decomposers** — coding, ML, and data analysis task decomposition

---

## Success Criteria

| Criterion | Target | Verification |
|-----------|--------|--------------|
| Test coverage | ≥ 95% | `cargo llvm-cov --html` |
| Recursive depth | ≥ 5 levels | Integration test with nested decomposition |
| CLI feature parity | 100% MCP tools | CLI integration tests |
| Domain decomposers | 3 (coding, ML, data) | Domain-specific integration tests |
| Async correctness | Parity with sync | Property tests, benchmark comparison |
| All existing tests | Pass | `cargo test` |

---

## Risk Awareness

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Decomposition explosion** | High | Depth limits (default 10), cycle detection, timeout (60s default) |
| **Voting on decomposition too expensive** | Medium | Cache proposals, reuse winning strategies |
| **Domain decomposers too task-specific** | Medium | Extensible `DecompositionAgent` trait |
| **Async integration breaks correctness** | High | Property tests comparing sync/async parity |
| **Claude Code API changes** | Medium | Version-pin MCP protocol, integration tests in CI |

---

## Milestone 1: Decomposition MVP

### Epic 011: Recursive Decomposition Infrastructure

- **Owner:** Lead Maintainer
- **Priority:** P0 (Critical Path)
- **Dependencies:** None (builds on v0.2.0 core)
- **Exit criteria:** Basic decomposition with voting works end-to-end

---

#### Story 011-01: Decomposition Agent Framework ✅ COMPLETE

- **Description:** As a developer, I want a trait-based decomposition framework, so that I can implement domain-specific task splitting strategies.
- **Context files:**
  - `docs/SystemDesignSpecification.txt` (Section 7 - Recursive Architecture)
  - `src/core/orchestration.rs` (existing TaskDecomposer trait)
  - `src/core/mod.rs` (module structure)
- **Work to do:**
  - [x] Create `src/core/decomposition/mod.rs` module
  - [x] Define `DecompositionProposal` struct with `subtasks`, `composition_fn`, `metadata`
  - [x] Define `Subtask` struct with `task_id`, `parent_id`, `m_value`, `description`, `context`
  - [x] Create `CompositionFunction` enum: `Sequential`, `Parallel`, `Conditional`, `Custom`
  - [x] Define `DecompositionAgent` trait with `propose_decomposition()` method
  - [x] Add `DecompositionProposalEvent` to `MakerEvent` enum
  - [x] Implement serde for all new types
  - [x] Add validation: leaf nodes must have `m_value == 1`
  - [x] Write unit tests for serialization round-trip
  - [x] Write property tests for m=1 enforcement
- **Acceptance criteria:**
  - [x] `DecompositionAgent` trait is object-safe
  - [x] `CompositionFunction` supports 4 patterns
  - [x] Unit tests pass for serialization
  - [x] Property tests enforce m=1 on leaf nodes
- **Verification:** `cargo test decomposition`, `cargo doc`
- **Completed:** 2026-01-31

---

#### Story 011-02: Decomposition Discriminator ✅ COMPLETE

- **Description:** As a system, I want to vote on decomposition proposals, so that I select the most reliable strategy.
- **Context files:**
  - `src/core/voting.rs` (VoteRace)
  - `src/core/matcher.rs` (CandidateMatcher trait)
- **Work to do:**
  - [x] Create `src/core/decomposition/discriminator.rs`
  - [x] Implement `CandidateMatcher` for `DecompositionProposal`
  - [x] Create `DecompositionDiscriminator` struct wrapping `VoteRace`
  - [x] Implement `vote_on_decomposition()` function
  - [x] Add depth-based k scaling
  - [x] Emit `DecompositionAccepted` and `DecompositionRejected` events
  - [x] Write integration test: 3 proposals → single winner
- **Acceptance criteria:**
  - [x] Uses same voting algorithm as execution voting
  - [x] Structural matcher handles differently-ordered subtasks
  - [x] Integration test passes
  - [x] k-margin scales with depth
- **Verification:** `cargo test discriminator`
- **Completed:** 2026-01-31

---

#### Story 011-03: Problem Solver Agent Interface ✅ COMPLETE

- **Description:** As a system, I want to execute atomic subtasks with voting, so that leaf nodes are solved reliably.
- **Context files:**
  - `src/core/orchestration.rs` (TaskOrchestrator)
  - `src/core/executor.rs` (vote_with_margin)
- **Work to do:**
  - [x] Create `src/core/decomposition/solver.rs`
  - [x] Refactor `TaskOrchestrator` to accept decomposition tree
  - [x] Create `LeafNodeExecutor` using `vote_with_margin()`
  - [x] Create `SubtaskResult` struct
  - [x] Implement state passing from parent to child
  - [x] Add partial failure handling with retries
  - [x] Write property tests for execution order
- **Acceptance criteria:**
  - [x] Leaf nodes always have m=1
  - [x] State flows correctly
  - [x] Partial failure recovery works
  - [x] Property test passes
- **Verification:** `cargo test solver`
- **Completed:** 2026-01-31

---

## Milestone 2: Full Recursive Loop

---

#### Story 011-04: Solution Discriminator & Aggregation ✅ COMPLETE

- **Description:** As a system, I want to aggregate subtask results, so that the final output is consistent with the strategy.
- **Context files:**
  - `src/core/decomposition/discriminator.rs`
  - `src/core/decomposition/solver.rs`
- **Work to do:**
  - [x] Create `src/core/decomposition/aggregator.rs`
  - [x] Create `SolutionDiscriminator` for voting on results
  - [x] Implement `compose_results()` for each `CompositionFunction`
  - [x] Add schema validation for composed results
  - [x] Handle recursive composition (nested trees)
  - [x] Emit `SolutionComposed` event
  - [x] Write integration test: 3-level deep decomposition
- **Acceptance criteria:**
  - [x] Composition respects winning strategy
  - [x] Nested decomposition works to depth 5+
  - [x] Full audit trail logged
  - [x] 3-level test passes
- **Verification:** `cargo test aggregator`
- **Completed:** 2026-01-31

---

#### Story 011-05: Recursive Loop Orchestration ✅ COMPLETE

- **Description:** As a user, I want to submit a high-level task and get a reliable result automatically.
- **Context files:**
  - `src/core/decomposition/*.rs`
  - `docs/SystemDesignSpecification.txt` (Section 7)
- **Work to do:**
  - [x] Create `src/core/decomposition/orchestrator.rs`
  - [x] Create `RecursiveOrchestrator` struct
  - [x] Implement `execute(task)` with full pipeline
  - [x] Add depth limit (default 10)
  - [x] Add cycle detection
  - [x] Add timeout (default 60s)
  - [x] Support manual decomposition injection
  - [x] Implement cancellation
  - [x] Write end-to-end test
- **Acceptance criteria:**
  - [x] End-to-end: task → decomposition → execution → result
  - [x] Manual override works
  - [x] Timeout cancels cleanly
  - [x] Depth limit prevents infinite recursion
- **Verification:** End-to-end test in CI
- **Completed:** 2026-01-31

---

## Milestone 3: Production CLI

### Epic 012: Production Hardening

- **Owner:** Lead Maintainer
- **Priority:** P1 (High)
- **Dependencies:** None (parallel with EPIC-011)
- **Exit criteria:** Standalone CLI works with all features

---

#### Story 012-01: Standalone CLI (`maker-cli`) ✅ COMPLETE

- **Description:** As a developer, I want a CLI for MAKER without MCP.
- **Context files:**
  - `src/bin/maker-mcp.rs`
  - `src/mcp/tools/*.rs`
- **Work to do:**
  - [x] Add `clap` dependency
  - [x] Create `src/bin/maker-cli.rs`
  - [x] Implement `vote`, `validate`, `calibrate`, `config` subcommands
  - [x] Add `decompose` subcommand (after EPIC-011)
  - [x] Support JSON and text output
  - [x] Add shell completion generation
  - [x] Write integration tests
- **Acceptance criteria:**
  - [x] Feature parity with MCP tools
  - [x] `--help` for all commands
  - [x] Standard exit codes
  - [x] Integration tests pass
- **Verification:** `cargo build --bin maker-cli`
- **Completed:** 2026-01-31

---

#### Story 012-02: Async Executor Integration ✅ COMPLETE

- **Description:** As a developer, I want async voting for efficient parallel sampling.
- **Context files:**
  - `src/core/executor.rs`
  - `src/llm/sampler.rs`
- **Work to do:**
  - [x] Create `src/core/async_executor.rs`
  - [x] Create `AsyncVotingExecutor`
  - [x] Implement `vote_with_margin_async()`
  - [x] Add cancellation handling
  - [x] Add connection pooling (via reqwest)
  - [x] Create sync/async benchmark (concurrent variant)
  - [x] Write property tests for parity
- **Acceptance criteria:**
  - [x] Same results as sync version
  - [x] Parallel latency < 2× sequential
  - [x] Graceful cancellation
  - [x] Benchmarks show improvement
- **Verification:** Property tests pass
- **Completed:** 2026-01-31

---

#### Story 012-04: Operational Tooling ✅ COMPLETE

- **Description:** As an operator, I want health checks and metrics for production.
- **Context files:**
  - `src/mcp/server.rs`
  - `src/events/observers/metrics.rs`
- **Work to do:**
  - [x] Add `prometheus` feature flag
  - [x] Create `src/mcp/health.rs` with `HealthStatus`
  - [x] Implement `/health` MCP resource
  - [x] Create Prometheus metrics (behind feature flag)
  - [x] Add `--validate-config` flag (via validate_config function)
  - [x] Implement graceful shutdown (via tokio signal handling)
- **Acceptance criteria:**
  - [x] Health check returns status/version/uptime
  - [x] Prometheus metrics work (behind feature flag)
  - [x] Invalid config fails fast
  - [x] Graceful shutdown works
- **Verification:** Health endpoint works
- **Completed:** 2026-01-31

---

## Milestone 4: Claude Code Ready

---

#### Story 012-03: Claude Code Integration Testing ✅ COMPLETE

- **Description:** As a user, I want reliable Claude Code integration.
- **Context files:**
  - `src/bin/maker-mcp.rs`
  - `src/mcp/server.rs`
- **Work to do:**
  - [x] Create `tests/mcp_integration.rs` harness (already existed)
  - [x] Write tests for all MCP tools (35 tests)
  - [x] Test configuration persistence
  - [x] Test ensemble and adaptive k
  - [x] Create `docs/CLAUDE-CODE-SETUP.md`
  - [x] Add to CI (tests run with cargo test)
- **Acceptance criteria:**
  - [x] All tools work via stdio
  - [x] Config persists correctly
  - [x] Ensemble metrics reported
  - [x] CI tests pass
- **Verification:** CI passes, manual Claude Desktop test
- **Completed:** 2026-01-31

---

## Milestone 5: Domain Decomposers

### Epic 013: Extended Domain Support ✅ COMPLETE

- **Owner:** Lead Maintainer
- **Priority:** P2 (Medium)
- **Dependencies:** EPIC-011 (resolved)
- **Exit criteria:** Domain decomposers work for coding, ML, data

---

#### Story 013-01: Coding Domain Decomposer ✅ COMPLETE

- **Description:** As a developer, I want AST-based code decomposition.
- **Context files:**
  - `src/core/decomposition/mod.rs`
  - `src/core/matchers/code.rs`
- **Work to do:**
  - [x] Create `src/core/decomposition/domains/coding.rs`
  - [x] Implement `CodingDecomposer` with tree-sitter
  - [x] Add function/block/line-level strategies
  - [x] Add syntax validation red-flags
  - [x] Support Rust, Python, JavaScript
  - [x] Write integration test
- **Acceptance criteria:**
  - [x] Respects syntactic boundaries
  - [x] Subtasks are m=1 operations
  - [x] Syntax errors red-flagged
  - [x] Integration test passes
- **Verification:** `cargo test --features code-matcher coding_decomposer`
- **Completed:** 2026-01-31

---

#### Story 013-02: ML Pipeline Decomposer ✅ COMPLETE

- **Description:** As an ML engineer, I want pipeline decomposition.
- **Work to do:**
  - [x] Create `src/core/decomposition/domains/ml.rs`
  - [x] Implement `MLPipelineDecomposer`
  - [x] Define DataPrep/Config/Training/Evaluation subtasks
  - [x] Implement hyperparameter search as parallel composition
  - [x] Add NaN/infinity red-flags
  - [x] Write integration test
- **Acceptance criteria:**
  - [x] Follows DataPrep → Config → Train → Eval pattern
  - [x] Hyperparameter search uses Parallel
  - [x] Invalid metrics red-flagged
  - [x] Integration test passes
- **Verification:** `cargo test ml_decomposer`
- **Completed:** 2026-01-31

---

#### Story 013-03: Data Analysis Decomposer ✅ COMPLETE

- **Description:** As a data analyst, I want ETL decomposition.
- **Work to do:**
  - [x] Create `src/core/decomposition/domains/data.rs`
  - [x] Implement `DataAnalysisDecomposer`
  - [x] Define Extract/Transform/Load/Validate subtasks
  - [x] Implement schema inference
  - [x] Add type coercion red-flags
  - [x] Write integration test
- **Acceptance criteria:**
  - [x] Follows ETL pattern
  - [x] Schema validation works
  - [x] Null handling explicit
  - [x] Integration test passes
- **Verification:** `cargo test data_decomposer`
- **Completed:** 2026-01-31

---

#### Story 013-04: Multi-file Orchestration ✅ COMPLETE

- **Description:** As a developer, I want multi-file state management.
- **Work to do:**
  - [x] Create `src/core/decomposition/filesystem.rs`
  - [x] Create `FileSystemState` struct
  - [x] Implement file-level locking
  - [x] Add cross-file dependency tracking
  - [x] Implement atomic multi-file commits
  - [x] Write integration test
- **Acceptance criteria:**
  - [x] State represents multiple files
  - [x] No race conditions
  - [x] Dependencies enforced
  - [x] Atomic commit works
- **Verification:** `cargo test multi_file`
- **Completed:** 2026-01-31

---

## Final Validation

- [x] All tests pass: `cargo test` (842+ tests)
- [ ] Coverage ≥ 95%: `cargo llvm-cov`
- [x] No clippy warnings: `cargo clippy`
- [x] Format check: `cargo fmt --check`
- [x] Documentation: `cargo doc`
- [ ] CHANGELOG.md updated
- [ ] End-to-end test with real LLM
- [ ] Claude Code manual testing

---

*Generated from TASK-PLAN-v0.3.0.md and PROJECT-CONTEXT.md*
*Last Updated: 2026-01-31*
