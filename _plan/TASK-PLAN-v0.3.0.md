# MAKER v0.3.0 Task Plan

## Overview

**Version:** 0.3.0
**Focus Areas:** Recursive Decomposition, Production Hardening, Extended Domain Support
**Timeline:** Milestone-based (flexible)
**Quality Gate:** 95% test coverage maintained

---

## Epic Structure

### EPIC-011: Recursive Decomposition Infrastructure

**Priority:** P0 (Critical Path)
**Dependency:** None (builds on v0.2.0 core)

#### STORY-011-01: Decomposition Agent Framework ✅ COMPLETE

**Description:** Create the base infrastructure for decomposition agents that split tasks into subtasks with composition functions.

**Tasks:**
- [x] Define `DecompositionProposal` struct: `{ subtasks: Vec<Subtask>, composition_fn: CompositionFunction }`
- [x] Create `DecompositionAgent` trait with `propose_decomposition(task: &Task) -> DecompositionProposal`
- [x] Implement `CompositionFunction` enum: `Sequential`, `Parallel`, `Conditional`, `Custom(fn)`
- [x] Add `Subtask` struct with `task_id`, `parent_id`, `m_value` (must be 1 for leaf nodes)
- [x] Create `DecompositionProposalEvent` for observability

**Acceptance Criteria:**
- [x] DecompositionAgent trait is object-safe
- [x] CompositionFunction supports the 4 core patterns from the paper
- [x] Unit tests for proposal serialization/deserialization
- [x] Property tests for m=1 enforcement on leaf nodes

---

#### STORY-011-02: Decomposition Discriminator ✅ COMPLETE

**Description:** Implement voting on proposed decomposition strategies using first-to-ahead-by-k.

**Tasks:**
- [x] Create `DecompositionDiscriminator` struct wrapping `VoteRace<DecompositionProposal>`
- [x] Implement `CandidateMatcher` for `DecompositionProposal` (structural equivalence)
- [x] Add `vote_on_decomposition()` function using existing voting infrastructure
- [x] Emit `DecompositionAccepted` and `DecompositionRejected` events
- [x] Configure k-margin based on decomposition depth (deeper = higher k)

**Acceptance Criteria:**
- [x] Discriminator uses same voting algorithm as execution voting
- [x] Structural matcher handles equivalent but differently-ordered subtasks
- [x] Integration test: multiple decomposition proposals → single winner
- [x] k-margin scales with decomposition complexity

---

#### STORY-011-03: Problem Solver Agent Interface 🔄 IN PROGRESS

**Description:** Extend existing microagent infrastructure for atomic (m=1) execution within decomposition tree.

**Tasks:**
- [ ] Refactor `TaskOrchestrator` to accept decomposition tree as input
- [x] Add `LeafNodeExecutor` that wraps `vote_with_margin()` for atomic tasks
- [x] Implement state passing from parent subtask to child
- [x] Create `SubtaskResult` struct with `action`, `next_state`, `metrics`
- [x] Handle partial failures (some subtasks fail, others succeed)

**Acceptance Criteria:**
- [x] Leaf nodes always have m=1 (enforced at compile time or runtime validation)
- [x] State correctly flows from decomposition to execution to aggregation
- [x] Partial failure recovery: retry failed subtasks, preserve succeeded
- [x] Property test: execution order matches composition function semantics

---

#### STORY-011-04: Solution Discriminator & Aggregation

**Description:** Implement voting on composed results to ensure final output aligns with original strategy.

**Tasks:**
- [ ] Create `SolutionDiscriminator` struct for aggregating subtask results
- [ ] Implement `compose_results()` using the winning composition function
- [ ] Add validation that composed result matches expected schema
- [ ] Emit `SolutionComposed` event with full audit trail
- [ ] Handle recursive composition (nested decomposition trees)

**Acceptance Criteria:**
- [ ] Composition respects the winning decomposition strategy
- [ ] Nested decomposition works to arbitrary depth
- [ ] Full audit trail: every decomposition → execution → composition logged
- [ ] Integration test: 3-level deep decomposition with voting at each level

---

#### STORY-011-05: Recursive Loop Orchestration

**Description:** Implement the complete recursive loop from high-level command to final output.

**Tasks:**
- [ ] Create `RecursiveOrchestrator` that manages the full pipeline
- [ ] Implement recursion until all nodes satisfy m=1 condition
- [ ] Add depth limits and cycle detection
- [ ] Support "pre-loaded insight" mode (manual decomposition injection)
- [ ] Implement timeout and cancellation for long-running decompositions

**Acceptance Criteria:**
- [ ] End-to-end test: high-level task → automatic decomposition → execution → result
- [ ] Manual override works at any decomposition level
- [ ] Timeout properly cancels in-flight API calls
- [ ] Depth limit prevents infinite recursion on malformed tasks

---

### EPIC-012: Production Hardening

**Priority:** P1 (High)
**Dependency:** None (parallel with EPIC-011)

#### STORY-012-01: Standalone CLI (`maker-cli`)

**Description:** Build a command-line interface for MAKER operations without MCP.

**Tasks:**
- [ ] Create `src/bin/maker-cli.rs` with clap argument parsing
- [ ] Implement `maker vote` subcommand with all vote options
- [ ] Implement `maker validate` subcommand for red-flag checking
- [ ] Implement `maker calibrate` subcommand for p estimation
- [ ] Implement `maker config` subcommand for configuration management
- [ ] Add `maker decompose` subcommand (once EPIC-011 complete)
- [ ] Support JSON and human-readable output formats
- [ ] Add shell completions generation

**Acceptance Criteria:**
- [ ] CLI provides feature parity with MCP tools
- [ ] `--help` documentation for all subcommands
- [ ] Exit codes follow standard conventions (0=success, 1=error, 2=invalid args)
- [ ] Integration tests for CLI invocations

---

#### STORY-012-02: Async Executor Integration

**Description:** Connect async LlmClient implementations to the synchronous voting executor.

**Tasks:**
- [ ] Create `AsyncVotingExecutor` that wraps `vote_with_margin()` in tokio runtime
- [ ] Implement proper cancellation handling for in-flight API calls
- [ ] Add connection pooling for HTTP clients
- [ ] Profile and optimize parallel sampling latency
- [ ] Document async vs sync usage patterns

**Acceptance Criteria:**
- [ ] Async executor achieves same correctness as sync version
- [ ] Parallel sampling latency < 2x sequential for 10 samples
- [ ] Graceful cancellation: no orphaned API calls
- [ ] Benchmarks comparing async vs sync performance

---

#### STORY-012-03: Claude Code Integration Testing

**Description:** End-to-end integration tests with real Claude Code environment.

**Tasks:**
- [ ] Create integration test harness that launches maker-mcp in subprocess
- [ ] Write tests for all 4 MCP tools with real stdio transport
- [ ] Test configuration persistence across tool calls
- [ ] Test ensemble and adaptive k via MCP
- [ ] Document Claude Desktop configuration steps

**Acceptance Criteria:**
- [ ] All MCP tools work via real stdio transport
- [ ] Configuration changes persist correctly
- [ ] Ensemble metrics reported in vote responses
- [ ] Integration tests run in CI (with mock LLM responses)

---

#### STORY-012-04: Operational Tooling

**Description:** Add health checks, metrics export, and configuration validation.

**Tasks:**
- [ ] Implement `/health` check for MCP server
- [ ] Add Prometheus metrics endpoint (optional feature flag)
- [ ] Create configuration validation on startup
- [ ] Add `--validate-config` flag to CLI
- [ ] Implement graceful shutdown with in-flight request completion

**Acceptance Criteria:**
- [ ] Health check returns current state and version
- [ ] Prometheus metrics include vote counts, latencies, red-flag rates
- [ ] Invalid configuration fails fast with clear error message
- [ ] Graceful shutdown completes in-flight requests before exit

---

### EPIC-013: Extended Domain Support

**Priority:** P2 (Medium)
**Dependency:** EPIC-011 (requires decomposition infrastructure)

#### STORY-013-01: Coding Domain Decomposer

**Description:** AST-based task decomposition for real code generation tasks.

**Tasks:**
- [ ] Create `CodingDecomposer` implementing `DecompositionAgent`
- [ ] Use tree-sitter for AST analysis of target code structure
- [ ] Implement decomposition strategies: function-level, block-level, line-level
- [ ] Add syntax validation red-flags using tree-sitter parse errors
- [ ] Support Rust, Python, JavaScript (matching CodeMatcher languages)

**Acceptance Criteria:**
- [ ] Decomposition respects syntactic boundaries (no mid-expression splits)
- [ ] Generated subtasks are valid m=1 operations
- [ ] Syntax errors caught before voting (red-flagged)
- [ ] Integration test: decompose and execute real coding task

---

#### STORY-013-02: ML Pipeline Decomposer

**Description:** Task decomposition for ML experiment orchestration and hyperparameter search.

**Tasks:**
- [ ] Create `MLPipelineDecomposer` implementing `DecompositionAgent`
- [ ] Define ML-specific subtask types: data prep, model config, training, evaluation
- [ ] Implement hyperparameter search as parallel decomposition
- [ ] Add validation for ML-specific outputs (loss values, metrics)
- [ ] Support common ML frameworks in prompts (PyTorch, sklearn)

**Acceptance Criteria:**
- [ ] Pipeline decomposition follows standard ML workflow patterns
- [ ] Hyperparameter search uses parallel composition function
- [ ] Metrics validation catches invalid outputs (NaN, infinity)
- [ ] Integration test: decompose simple experiment pipeline

---

#### STORY-013-03: Data Analysis Decomposer

**Description:** ETL pipeline generation with validation steps.

**Tasks:**
- [ ] Create `DataAnalysisDecomposer` implementing `DecompositionAgent`
- [ ] Define data ops subtask types: extract, transform, load, validate
- [ ] Implement schema inference and validation
- [ ] Add type coercion red-flags
- [ ] Support SQL, pandas, and raw CSV operations

**Acceptance Criteria:**
- [ ] ETL pipeline follows extract → transform → load pattern
- [ ] Schema validation catches type mismatches
- [ ] Null handling strategies explicit in decomposition
- [ ] Integration test: decompose multi-stage data pipeline

---

#### STORY-013-04: Multi-file Orchestration

**Description:** State passing across file boundaries for complex tasks.

**Tasks:**
- [ ] Extend state representation to include file system state
- [ ] Implement file-level locking for concurrent subtask execution
- [ ] Add cross-file dependency tracking
- [ ] Create `FileSystemState` struct with content hashes
- [ ] Support atomic multi-file commits

**Acceptance Criteria:**
- [ ] State correctly represents multiple files
- [ ] No race conditions in parallel subtask execution
- [ ] Dependencies prevent out-of-order execution
- [ ] Atomic commit: all files or none

---

## Dependency Graph

```
EPIC-011 (Recursive Decomposition)
├── STORY-011-01 (Decomposition Agent Framework)
├── STORY-011-02 (Decomposition Discriminator) ← depends on 011-01
├── STORY-011-03 (Problem Solver Agent) ← depends on 011-01
├── STORY-011-04 (Solution Discriminator) ← depends on 011-02, 011-03
└── STORY-011-05 (Recursive Loop) ← depends on 011-04

EPIC-012 (Production Hardening) — PARALLEL WITH EPIC-011
├── STORY-012-01 (Standalone CLI)
├── STORY-012-02 (Async Executor)
├── STORY-012-03 (Claude Code Integration) ← depends on 012-01
└── STORY-012-04 (Operational Tooling)

EPIC-013 (Extended Domain Support) — AFTER EPIC-011
├── STORY-013-01 (Coding Decomposer) ← depends on 011-05
├── STORY-013-02 (ML Pipeline Decomposer) ← depends on 011-05
├── STORY-013-03 (Data Analysis Decomposer) ← depends on 011-05
└── STORY-013-04 (Multi-file Orchestration) ← depends on 013-01
```

---

## Milestones

### Milestone 1: Decomposition MVP

**Stories:** STORY-011-01, STORY-011-02, STORY-011-03
**Deliverable:** Basic decomposition with voting, manual composition

### Milestone 2: Full Recursive Loop

**Stories:** STORY-011-04, STORY-011-05
**Deliverable:** End-to-end recursive orchestration

### Milestone 3: Production CLI

**Stories:** STORY-012-01, STORY-012-02, STORY-012-04
**Deliverable:** Standalone CLI with async support and operational tooling

### Milestone 4: Claude Code Ready

**Stories:** STORY-012-03
**Deliverable:** Verified Claude Code integration

### Milestone 5: Domain Decomposers

**Stories:** STORY-013-01, STORY-013-02, STORY-013-03, STORY-013-04
**Deliverable:** Domain-specific decomposition for coding, ML, and data tasks

---

## Quality Criteria

- [ ] 95% test coverage maintained across all new code
- [ ] Property-based tests for all probabilistic behavior
- [ ] Integration tests for end-to-end workflows
- [ ] Documentation for all public APIs
- [ ] CHANGELOG updated for each milestone
- [ ] No new clippy warnings
- [ ] All existing tests pass

---

## Risk Register (v0.3.0 Specific)

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Decomposition explosion (infinite recursion) | High | Medium | Depth limits, cycle detection |
| Voting on decomposition too expensive | Medium | Medium | Cache decomposition proposals, reuse strategies |
| Domain decomposers too task-specific | Medium | High | Provide extensible framework, not fixed implementations |
| Async integration breaks voting correctness | High | Low | Comprehensive property tests, sync/async parity tests |
| Claude Code API changes break integration | Medium | Low | Version-pin MCP protocol, integration tests in CI |

---

*Generated from PROJECT-CONTEXT.md and System Design Specification*
*Last Updated: 2026-01-31*
