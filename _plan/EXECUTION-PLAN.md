# Execution Plan: MAKER Framework

> This file is a complete execution plan for implementing the MAKER Framework in Rust.
> It was generated from JIRA-STRUCTURE.md and the project planning artifacts.
> To execute: open a Claude Code session in this repository and say:
> "Read _plan/EXECUTION-PLAN.md and begin executing the work items"

**Generated:** 2026-01-30
**Updated:** 2026-01-30 (post-MVP sprints added)
**Domain:** Open Source AI Infrastructure / LLM Agent Reliability Engineering
**Project:** Implement MAKER framework (SPRT voting, red-flagging, microagent orchestration) as a Rust MCP server for Claude Code integration, enabling zero-error execution on long-horizon tasks.

---

## How to Execute This Plan

Read this file, then work through each epic and its stories sequentially within each phase.
For each work item:
1. Check prerequisites and blocking dependencies
2. Read the relevant context files listed
3. Execute the work described — write code, create configs, run commands, etc.
4. Verify acceptance criteria are met
5. Mark complete and move to the next item

Epics within the same phase that have no cross-dependencies can be worked in parallel.

---

## Project Summary

The MAKER framework solves a fundamental limitation of LLM agents: catastrophic failure on long-horizon tasks. Current agents fail >50% on 100-step tasks and 100% on million-step tasks (even with 99% per-step accuracy). MAKER applies formal error correction from information theory — treating LLMs as noisy channels and using SPRT-based voting as forward error correction.

This project delivers:
1. **Core Rust library** implementing k_min calculation, first-to-ahead-by-k voting, red-flagging parsers, and microagent orchestration
2. **MCP server** exposing 4 tools (vote, validate, calibrate, configure) via rmcp SDK
3. **Multi-provider LLM abstraction** supporting Ollama, OpenAI, and Anthropic
4. **Event-driven observability** with structured logging and metrics
5. **End-to-end demo** achieving zero errors on 10-disk Towers of Hanoi (1,023 steps)

The MVP (v0.1.0) was delivered across 3 sprints (132 story points). This plan covers **Sprints 4-7** for post-MVP extensions: Adaptive K, Semantic Matching, Multi-Model Ensemble, and Benchmark Suite.

---

## Success Criteria

### MVP (v0.1.0) — COMPLETE
- [x] Zero errors on 10-disk Towers of Hanoi (1,023 steps)
- [x] 95% test coverage on all modules
- [x] Cost scaling Θ(s ln s) validated within 20% tolerance
- [x] All 4 MCP tools functional
- [x] Claude Code integration working
- [x] Security audit passed
- [x] Documentation complete
- [x] v0.1.0 GitHub release tagged

### Post-MVP (v0.2.0) — THIS PLAN
- [ ] **Adaptive K**: Dynamic k-margin adjustment based on observed error rates
- [ ] **Semantic Matching**: Voting on non-deterministic tasks (coding, ML, data analysis)
- [ ] **Multi-Model Ensemble**: Voting across heterogeneous models
- [ ] **Benchmark Suite**: Domain-specific reliability measurement framework
- [ ] **Zero errors on 1M+ steps** with adaptive k on deterministic tasks
- [ ] **>90% success rate** on non-deterministic coding tasks (semantic matching)
- [ ] **Cost scaling ±10%** of Θ(s ln s) (tighter than MVP's ±20%)
- [ ] **95% test coverage** maintained across all new modules

---

## Risk Awareness

From RISK-REGISTER.md — watch for these during post-MVP implementation:

| Risk | Indicator | Mitigation |
|------|-----------|------------|
| **Semantic matching accuracy** | False equivalences in code comparison | AST-based matching, embedding similarity thresholds, domain-specific test suites |
| **Adaptive k instability** | k oscillates or converges to wrong value | Exponential moving average, bounded k range, simulation validation |
| **Multi-model cost explosion** | Ensemble more expensive than single-model | Cost-aware routing: cheap models first, expensive on disagreement |
| **Embedding latency** | Semantic comparison adds per-vote overhead | Cache embeddings, pre-compute for common patterns, async comparison |
| **Backward compatibility** | New features break existing exact-match API | Trait-based extensibility, default to exact match, feature flags |

---

## Sprints 1-3: COMPLETE (v0.1.0 MVP)

All work items from Phase 1 (Core Algorithms), Phase 2 (MCP Integration), and Phase 3 (Validation & Hardening) have been completed. See the v0.1.0 release at https://github.com/zircote/maker-rs/releases/tag/v0.1.0.

**Completed Epics:**
- EPIC-001: Core MAKER Library (24 pts) ✅
- EPIC-002: LLM Provider Abstraction (28 pts) ✅
- EPIC-003: MCP Server Implementation (22 pts) ✅
- EPIC-004: Event-Driven Observability (13 pts) ✅
- EPIC-005: Testing Infrastructure (18 pts) ✅
- EPIC-006: Demo & Benchmarks (18 pts) ✅
- EPIC-007: Documentation (14 pts) ✅
- EPIC-008: Security & Guardrails (8 pts) ✅

---

## Phase 4: Adaptive K-Margin (Sprint 4, Weeks 3-4)

**Objective:** Implement dynamic k-margin adjustment that adapts to observed error rates and task complexity, improving cost efficiency while maintaining reliability guarantees.
**Exit Criteria:** Adaptive k reduces total API calls by 20%+ vs. static k on 10-disk Hanoi, while maintaining zero errors.

---

### EPIC-011: Adaptive K-Margin
- **Owner:** Project Maintainer
- **Duration:** 2 weeks (Sprint 4)
- **Priority:** P1 (High)
- **Dependencies:** EPIC-001 (k_min calculation), EPIC-004 (event metrics)
- **Phase exit criteria:** Adaptive k converges to optimal value, cost savings validated

#### STORY-011-01: K-Margin Estimator
- **Description:** As a MAKER framework, I want to estimate the optimal k-margin dynamically from observed vote convergence data, so that I can minimize API calls without sacrificing reliability.
- **Context files:** `src/core/kmin.rs`, `src/core/voting.rs`, `docs/technical-implementation-manual.txt`
- **Work to do:**
  - [ ] Create `src/core/adaptive.rs`
  - [ ] Define `KEstimator` struct tracking: recent vote margins, convergence rates, red-flag rates
  - [ ] Implement exponential moving average (EMA) for p-estimate: `p_hat = α * p_sample + (1-α) * p_hat_prev` (α = 0.1)
  - [ ] Implement `KEstimator::recommended_k(&self, target_t: f64, remaining_steps: usize) -> usize` using live p_hat in k_min formula
  - [ ] Add bounds: `k_min_floor=2, k_max_ceiling=10` to prevent runaway adjustment
  - [ ] Emit `KAdjusted { old_k, new_k, p_hat, reason }` event when k changes
- **Acceptance criteria:**
  - [ ] p_hat converges to within ±5% of true p after 20 observations
  - [ ] recommended_k decreases when observed p > calibrated p (saves cost)
  - [ ] recommended_k increases when observed p < calibrated p (maintains safety)
  - [ ] k never drops below k_min_floor or exceeds k_max_ceiling
- **Verification:** Run `cargo test adaptive` — convergence tests pass; property tests validate EMA convergence

---

#### STORY-011-02: Adaptive Voting Integration
- **Description:** As a MAKER voting engine, I want to use the adaptive k-estimator during multi-step execution, so that each step uses the optimal k-margin based on accumulated evidence.
- **Context files:** `src/core/executor.rs`, `src/core/adaptive.rs`
- **Work to do:**
  - [ ] Add `AdaptiveVoteConfig` extending `VoteConfig` with: `adaptive: bool`, `initial_k: usize`, `ema_alpha: f64`, `k_bounds: (usize, usize)`
  - [ ] Modify `vote_with_margin` to accept optional `&mut KEstimator` parameter
  - [ ] After each vote decision, feed result back to estimator: `estimator.observe(vote_result)`
  - [ ] Use `estimator.recommended_k()` for subsequent steps (not retroactive)
  - [ ] Preserve backward compatibility: `adaptive: false` uses static k (default)
- **Acceptance criteria:**
  - [ ] Existing tests pass unchanged (backward compatible)
  - [ ] Adaptive mode reduces total API calls by 20%+ on 10-disk Hanoi vs. static k=4
  - [ ] Zero errors maintained with adaptive k on deterministic tasks
  - [ ] k adjustment logged via events for debugging
- **Verification:** Run `cargo test executor` — all pass; run adaptive Hanoi demo, compare costs

---

#### STORY-011-03: Adaptive K MCP Tool Extension
- **Description:** As a Claude Code user, I want to enable adaptive k via maker/configure and see k adjustments in maker/vote responses, so that I benefit from cost optimization automatically.
- **Context files:** `src/mcp/tools/configure.rs`, `src/mcp/tools/vote.rs`
- **Work to do:**
  - [ ] Extend `ConfigRequest` with `adaptive_k: Option<bool>`, `ema_alpha: Option<f64>`, `k_bounds: Option<(usize, usize)>`
  - [ ] Extend `VoteResponse` with `k_used: usize` (actual k for this vote), `p_hat: Option<f64>` (current estimate)
  - [ ] Store `KEstimator` in `ServerState` (reset on configure or new task)
  - [ ] Add `maker/vote` parameter: `adaptive: Option<bool>` (per-call override)
- **Acceptance criteria:**
  - [ ] Configure adaptive mode via MCP tool
  - [ ] VoteResponse includes k_used and p_hat when adaptive
  - [ ] Backward compatible: non-adaptive calls unchanged
- **Verification:** MCP integration tests with adaptive flag

---

#### STORY-011-04: Adaptive K Validation Suite
- **Description:** As a MAKER developer, I want comprehensive tests proving adaptive k maintains reliability while reducing cost, so that I can trust the optimization.
- **Context files:** `tests/properties.rs`, `tests/monte_carlo.rs`
- **Work to do:**
  - [ ] Add property test: adaptive k never violates target reliability t over 10,000 simulated tasks
  - [ ] Add Monte Carlo test: adaptive k reduces mean cost vs. static k by ≥15% (for p=0.85, t=0.95)
  - [ ] Add regression test: adaptive k on 10-disk Hanoi produces zero errors
  - [ ] Add stress test: adaptive k with sudden p-drop (model degrades mid-task) still recovers
  - [ ] Benchmark: compare adaptive vs. static k cost across n ∈ {3, 5, 7, 10, 15} disks
- **Acceptance criteria:**
  - [ ] Property test: 0 reliability violations in 10,000 trials
  - [ ] Monte Carlo: adaptive cost ≤ 85% of static cost (mean)
  - [ ] Regression: zero errors on 10-disk Hanoi
  - [ ] Stress test: k increases when p drops, task still succeeds
- **Verification:** Run `cargo test adaptive` and `cargo bench adaptive_cost`

---

## Phase 5: Semantic Matching (Sprint 5, Weeks 5-6)

**Objective:** Extend voting beyond exact string matching to support non-deterministic tasks where multiple correct answers exist (coding, ML, data analysis).
**Exit Criteria:** Semantic matching achieves >90% agreement rate on code equivalence tasks; backward compatible with exact match.

---

### EPIC-009: Semantic Matching
- **Owner:** Project Maintainer
- **Duration:** 2 weeks (Sprint 5)
- **Priority:** P1 (High)
- **Dependencies:** EPIC-001 (voting), EPIC-011 (adaptive k benefits from better matching)
- **Phase exit criteria:** Matcher trait extensible, code matcher works, embedding matcher works

#### STORY-009-01: Matcher Trait Abstraction
- **Description:** As a MAKER framework, I want a pluggable trait for comparing candidate responses, so that voting can work with exact match, semantic similarity, or domain-specific equivalence.
- **Context files:** `src/core/voting.rs`, `src/core/executor.rs`
- **Work to do:**
  - [ ] Create `src/core/matcher.rs`
  - [ ] Define `CandidateMatcher` trait:
    ```rust
    pub trait CandidateMatcher: Send + Sync {
        fn are_equivalent(&self, a: &str, b: &str) -> bool;
        fn similarity_score(&self, a: &str, b: &str) -> f64;
        fn canonicalize(&self, response: &str) -> String;
    }
    ```
  - [ ] Implement `ExactMatcher` (current behavior): `are_equivalent` via string equality after `canonicalize` (trim whitespace)
  - [ ] Modify `VoteRace` to accept `Arc<dyn CandidateMatcher>` instead of string hashing
  - [ ] Update `vote_with_margin` to use matcher for candidate grouping
  - [ ] Default to `ExactMatcher` for full backward compatibility
- **Acceptance criteria:**
  - [ ] All existing tests pass unchanged with `ExactMatcher`
  - [ ] Custom matchers can be injected via `VoteConfig`
  - [ ] `CandidateMatcher` is object-safe and Send + Sync
  - [ ] Canonicalization strips whitespace differences
- **Verification:** Run `cargo test voting` and `cargo test executor` — all pass

---

#### STORY-009-02: Embedding-Based Similarity Matcher
- **Description:** As a MAKER user working on non-deterministic tasks, I want to group semantically similar responses for voting, so that equivalent but textually different answers are counted as the same candidate.
- **Context files:** `src/core/matcher.rs`, `src/llm/mod.rs`
- **Work to do:**
  - [ ] Create `src/core/matchers/embedding.rs`
  - [ ] Define `EmbeddingMatcher { threshold: f64, client: Arc<dyn EmbeddingClient> }`
  - [ ] Define `EmbeddingClient` trait: `async fn embed(&self, text: &str) -> Result<Vec<f64>, LlmError>`
  - [ ] Implement cosine similarity: `similarity_score(a, b) = dot(embed(a), embed(b)) / (|a| * |b|)`
  - [ ] Implement `are_equivalent(a, b) = similarity_score(a, b) >= threshold` (default threshold: 0.92)
  - [ ] Add embedding cache: `HashMap<String, Vec<f64>>` to avoid re-embedding identical strings
  - [ ] Implement `OllamaEmbeddingClient` (POST /api/embeddings)
  - [ ] Implement `OpenAiEmbeddingClient` (text-embedding-3-small)
- **Acceptance criteria:**
  - [ ] Cosine similarity correct for known embedding pairs
  - [ ] Threshold configurable (0.0 to 1.0)
  - [ ] Cache prevents redundant embedding calls (hit rate >80% in voting scenarios)
  - [ ] Integration test: semantically equivalent code snippets grouped correctly
- **Verification:** Run `cargo test embedding` — similarity and caching tests pass

---

#### STORY-009-03: Code AST Matcher
- **Description:** As a MAKER user running coding tasks, I want to compare code responses by their AST structure, so that formatting differences and variable naming don't split votes.
- **Context files:** `src/core/matcher.rs`
- **Work to do:**
  - [ ] Create `src/core/matchers/code.rs`
  - [ ] Add `tree-sitter` dependency for multi-language parsing
  - [ ] Implement `CodeMatcher { language: Language, threshold: f64 }`
  - [ ] Implement `canonicalize`: parse to AST, normalize variable names (alpha-renaming), strip comments, pretty-print
  - [ ] Implement `similarity_score`: tree edit distance / max tree size
  - [ ] Support languages: Rust, Python, JavaScript (via tree-sitter grammars)
  - [ ] Fallback to `EmbeddingMatcher` if parsing fails (malformed code)
- **Acceptance criteria:**
  - [ ] `def foo(x): return x+1` equivalent to `def bar(y): return y + 1` (alpha-renaming)
  - [ ] Comments and whitespace ignored
  - [ ] Parsing errors fall back to embedding matcher gracefully
  - [ ] Tree-sitter grammars for 3 languages bundled
- **Verification:** Run `cargo test code_matcher` — AST normalization tests pass

---

#### STORY-009-04: Matcher Configuration via MCP
- **Description:** As a Claude Code user, I want to configure which matcher to use via maker/configure, so that I can switch between exact, embedding, and code matching per task.
- **Context files:** `src/mcp/tools/configure.rs`, `src/mcp/tools/vote.rs`
- **Work to do:**
  - [ ] Extend `ConfigRequest` with `matcher: Option<MatcherConfig>` where:
    ```json
    { "type": "exact" }
    { "type": "embedding", "threshold": 0.92, "provider": "ollama" }
    { "type": "code", "language": "python", "threshold": 0.85 }
    ```
  - [ ] Extend `VoteResponse` with `matcher_type: String`, `candidate_groups: usize` (how many distinct groups after matching)
  - [ ] Add `matcher` parameter to `VoteRequest` for per-call override
  - [ ] Store active matcher in `ServerState`
- **Acceptance criteria:**
  - [ ] Default matcher is "exact" (backward compatible)
  - [ ] Switching to embedding matcher works via configure tool
  - [ ] VoteResponse shows matcher_type used
  - [ ] Invalid matcher config returns clear error
- **Verification:** MCP integration tests with different matcher configs

---

#### STORY-009-05: Semantic Matching Test Suite
- **Description:** As a MAKER developer, I want comprehensive tests for semantic matching accuracy, so that I can validate that matchers correctly group equivalent responses.
- **Context files:** `tests/properties.rs`
- **Work to do:**
  - [ ] Create `tests/semantic_matching.rs`
  - [ ] Build test corpus: 50 pairs of equivalent code snippets (Python, Rust, JS)
  - [ ] Build test corpus: 50 pairs of semantically similar natural language responses
  - [ ] Build test corpus: 25 pairs of non-equivalent responses (negative cases)
  - [ ] Test `CodeMatcher` accuracy: >95% on code corpus
  - [ ] Test `EmbeddingMatcher` accuracy: >90% on NL corpus
  - [ ] Test false positive rate: <5% on non-equivalent pairs
  - [ ] Property test: matching is reflexive (a ≡ a) and symmetric (a ≡ b ↔ b ≡ a)
- **Acceptance criteria:**
  - [ ] CodeMatcher: >95% accuracy on code equivalence corpus
  - [ ] EmbeddingMatcher: >90% accuracy on NL corpus
  - [ ] False positive rate <5%
  - [ ] Reflexivity and symmetry properties hold
- **Verification:** Run `cargo test --test semantic_matching`

---

## Phase 6: Multi-Model Ensemble (Sprint 6, Weeks 7-8)

**Objective:** Enable voting across heterogeneous LLM models to decorrelate errors by model architecture, not just sampling temperature.
**Exit Criteria:** Ensemble voting achieves lower error rate than single-model voting; cost-aware routing reduces cost by 30%+.

---

### EPIC-010: Multi-Model Ensemble
- **Owner:** Project Maintainer
- **Duration:** 2 weeks (Sprint 6)
- **Priority:** P2 (Medium)
- **Dependencies:** EPIC-002 (LLM clients), EPIC-009 (semantic matching for heterogeneous outputs)
- **Phase exit criteria:** Ensemble voting works across 2+ models, cost routing functional

#### STORY-010-01: Ensemble Configuration
- **Description:** As a MAKER user, I want to configure multiple LLM models for ensemble voting, so that I can decorrelate errors across model architectures.
- **Context files:** `src/llm/mod.rs`, `src/llm/sampler.rs`
- **Work to do:**
  - [ ] Create `src/llm/ensemble.rs`
  - [ ] Define `EnsembleConfig`:
    ```rust
    pub struct EnsembleConfig {
        pub models: Vec<ModelSlot>,
        pub strategy: EnsembleStrategy,
    }
    pub struct ModelSlot {
        pub client: Arc<dyn LlmClient>,
        pub weight: f64,        // Sampling weight (higher = more samples)
        pub cost_tier: CostTier, // Cheap, Medium, Expensive
    }
    pub enum EnsembleStrategy {
        RoundRobin,           // Distribute samples evenly
        CostAware,            // Cheap models first, expensive on disagreement
        ReliabilityWeighted,  // More samples from higher-p models
    }
    ```
  - [ ] Implement `EnsembleConfig::select_model_for_sample(sample_index: usize) -> &ModelSlot`
  - [ ] Add `EnsembleConfig` to `ServerState` (optional, None = single-model mode)
- **Acceptance criteria:**
  - [ ] Can configure 2-5 models in ensemble
  - [ ] RoundRobin distributes evenly (within 1 sample)
  - [ ] CostAware uses cheap models for first 2k samples, expensive for remaining
  - [ ] Single-model mode (no ensemble) unchanged
- **Verification:** Unit tests for all 3 strategies

---

#### STORY-010-02: Ensemble Sampling Integration
- **Description:** As a MAKER voting engine, I want to collect samples from multiple models in the ensemble, so that votes represent diverse model outputs.
- **Context files:** `src/llm/sampler.rs`, `src/llm/ensemble.rs`
- **Work to do:**
  - [ ] Modify `collect_samples` to accept `EnsembleConfig` (optional)
  - [ ] When ensemble active: select model per `EnsembleStrategy` for each sample
  - [ ] Tag each sample with source model: `SampleResult { content, model_name, tokens, latency }`
  - [ ] Emit `SampleRequested` event with model name field
  - [ ] Handle model-specific failures: if one model fails, increase allocation to others
  - [ ] Temperature strategy per model: respect model-specific optimal temperatures
- **Acceptance criteria:**
  - [ ] Samples come from multiple models (verified by model_name in results)
  - [ ] Model failure doesn't halt voting (graceful degradation)
  - [ ] Events include model attribution for cost tracking
  - [ ] Latency ≈ max(individual model latencies) not sum (parallel across models)
- **Verification:** Integration test with 2 mock models, verify distribution

---

#### STORY-010-03: Cost-Aware Routing
- **Description:** As a MAKER user, I want the ensemble to minimize cost by using cheap models first and only escalating to expensive models when there's disagreement, so that I get reliability benefits without cost explosion.
- **Context files:** `src/llm/ensemble.rs`, `src/core/voting.rs`
- **Work to do:**
  - [ ] Implement `CostAware` strategy:
    1. Phase 1: Collect `k` samples from cheapest model only
    2. If winner by k-margin → done (cheapest path)
    3. Phase 2: If no winner, collect `k` samples from next-cheapest model
    4. Phase 3: If still no winner, collect remaining from most expensive model
  - [ ] Track cost per model in `VoteResult`: `cost_by_model: HashMap<String, CostMetrics>`
  - [ ] Emit `EscalationTriggered { from_model, to_model, reason }` event
  - [ ] Add `escalation_count` to metrics observer
- **Acceptance criteria:**
  - [ ] Easy tasks (high p) resolved by cheap model only (no escalation)
  - [ ] Disagreement triggers escalation to next tier
  - [ ] Total cost < single-expensive-model cost in 80%+ of cases
  - [ ] Cost breakdown by model available in VoteResponse
- **Verification:** Benchmark: ensemble cost vs. single-model cost on Hanoi tasks

---

#### STORY-010-04: Ensemble MCP Tool Extension
- **Description:** As a Claude Code user, I want to configure multi-model ensemble via maker/configure and see per-model metrics in responses, so that I can leverage ensemble voting.
- **Context files:** `src/mcp/tools/configure.rs`, `src/mcp/tools/vote.rs`
- **Work to do:**
  - [ ] Extend `ConfigRequest` with `ensemble: Option<EnsembleConfigRequest>`:
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
  - [ ] Extend `VoteResponse` with `ensemble_metrics: Option<EnsembleMetrics>`:
    ```json
    {
      "models_used": ["ollama/llama3", "anthropic/claude-haiku"],
      "samples_per_model": { "ollama/llama3": 4, "anthropic/claude-haiku": 2 },
      "escalations": 1,
      "cost_per_model": { "ollama/llama3": 0.0, "anthropic/claude-haiku": 0.003 }
    }
    ```
  - [ ] Add `maker/vote` parameter: `ensemble: Option<bool>` (per-call override)
- **Acceptance criteria:**
  - [ ] Ensemble configurable via MCP tool
  - [ ] VoteResponse includes per-model breakdown when ensemble active
  - [ ] Backward compatible: no ensemble = single model
- **Verification:** MCP integration tests with ensemble config

---

#### STORY-010-05: Cross-Model Reliability Benchmarks
- **Description:** As a MAKER developer, I want benchmarks comparing single-model vs. ensemble reliability and cost, so that I can demonstrate the value of multi-model voting.
- **Context files:** `tests/monte_carlo.rs`, `benches/cost_scaling.rs`
- **Work to do:**
  - [ ] Create `benches/ensemble_comparison.rs`
  - [ ] Benchmark configurations:
    - Single model (Ollama llama3, p=0.80)
    - Single model (Claude Haiku, p=0.90)
    - Ensemble (llama3 + Haiku, round-robin)
    - Ensemble (llama3 + Haiku, cost-aware)
  - [ ] Metrics per config: total cost, total latency, error rate, samples per step
  - [ ] Monte Carlo simulation: 1,000 trials per config at s=100, s=1000
  - [ ] Generate comparison table for README
- **Acceptance criteria:**
  - [ ] Ensemble error rate < min(individual model error rates) — diversity benefit
  - [ ] Cost-aware ensemble cost < expensive-model-only cost by 30%+
  - [ ] Benchmark results reproducible and documented
- **Verification:** Run `cargo bench ensemble_comparison`, verify improvement

---

## Phase 7: Benchmark Suite & v0.2.0 Release (Sprint 7, Weeks 9-10)

**Objective:** Build comprehensive domain-specific benchmarks and release v0.2.0 with all post-MVP extensions.
**Exit Criteria:** Benchmark suite covers 3 domains, v0.2.0 released with documentation.

---

### EPIC-012: Comprehensive Benchmark Suite
- **Owner:** Project Maintainer
- **Duration:** 1 week (Sprint 7, first half)
- **Priority:** P1 (High)
- **Dependencies:** EPIC-009 (semantic matching for non-deterministic benchmarks), EPIC-010 (ensemble for multi-model benchmarks)
- **Phase exit criteria:** 3 domain-specific benchmarks pass, results documented

#### STORY-012-01: Coding Task Benchmark
- **Description:** As a MAKER developer, I want benchmarks on real coding tasks, so that I can validate MAKER's reliability on the primary target domain.
- **Context files:** `examples/hanoi/`, `src/core/matchers/code.rs`
- **Work to do:**
  - [ ] Create `benches/coding_tasks/` directory
  - [ ] Implement 10 coding task benchmarks:
    - FizzBuzz generation (trivial, p≈0.95)
    - Binary search implementation (moderate, p≈0.85)
    - Linked list reversal (moderate, p≈0.80)
    - JSON parser (complex, p≈0.70)
    - SQL query generation from spec (complex, p≈0.65)
  - [ ] Each benchmark: define prompt, ground truth validator, decomposition into subtasks
  - [ ] Use `CodeMatcher` for equivalence checking
  - [ ] Collect metrics: accuracy, cost, latency, red-flag rate per task
  - [ ] Generate results summary as JSON and markdown table
- **Acceptance criteria:**
  - [ ] All 10 benchmarks execute without crashes
  - [ ] MAKER achieves >90% accuracy on trivial/moderate tasks with k=3-4
  - [ ] MAKER achieves >80% accuracy on complex tasks with k=5-6
  - [ ] Results documented in benchmark README
- **Verification:** Run `cargo bench --bench coding_tasks`

---

#### STORY-012-02: Math & Logic Benchmark
- **Description:** As a MAKER developer, I want benchmarks on mathematical reasoning tasks, so that I can validate MAKER on tasks with verifiable ground truth.
- **Context files:** `examples/hanoi/`
- **Work to do:**
  - [ ] Create `benches/math_logic/` directory
  - [ ] Implement benchmarks:
    - Multi-step arithmetic (100 sequential operations)
    - Symbolic differentiation (chain rule sequences)
    - Logic puzzle solving (Sudoku validation steps)
    - Tower of Hanoi variants (4-peg optimization)
  - [ ] Use `ExactMatcher` (deterministic answers)
  - [ ] Validate against ground truth programmatically
  - [ ] Collect cost scaling data points for Θ(s ln s) regression
- **Acceptance criteria:**
  - [ ] Zero errors on arithmetic tasks with k=3
  - [ ] Θ(s ln s) cost scaling holds across task lengths
  - [ ] Results include confidence intervals
- **Verification:** Run `cargo bench --bench math_logic`

---

#### STORY-012-03: Data Analysis Benchmark
- **Description:** As a MAKER developer, I want benchmarks on data analysis tasks, so that I can validate MAKER in the ML/data science domain.
- **Context files:** `src/core/matchers/embedding.rs`
- **Work to do:**
  - [ ] Create `benches/data_analysis/` directory
  - [ ] Implement benchmarks:
    - CSV parsing and transformation (deterministic)
    - Statistical summary generation (approximate matching)
    - SQL query equivalence (using CodeMatcher)
    - Data cleaning pipeline steps
  - [ ] Use `EmbeddingMatcher` for approximate equivalence
  - [ ] Define acceptable tolerance ranges for numerical outputs
  - [ ] Collect metrics: accuracy within tolerance, cost, red-flag rate
- **Acceptance criteria:**
  - [ ] >85% accuracy on data analysis tasks
  - [ ] Numerical outputs within 1% tolerance of ground truth
  - [ ] Results documented with methodology
- **Verification:** Run `cargo bench --bench data_analysis`

---

#### STORY-012-04: Benchmark Dashboard & Reporting
- **Description:** As a MAKER maintainer, I want automated benchmark reporting with historical tracking, so that I can detect regressions and demonstrate improvements.
- **Context files:** `benches/`, `.github/workflows/ci.yml`
- **Work to do:**
  - [ ] Create `benches/report.rs` — aggregates all benchmark results
  - [ ] Generate `benchmark_results.json` with structured output
  - [ ] Generate `BENCHMARKS.md` with markdown tables and summary
  - [ ] Add GitHub Actions workflow: run benchmarks weekly, store results as artifacts
  - [ ] Add comparison script: current results vs. previous run (detect regressions)
  - [ ] Link from README: "See BENCHMARKS.md for latest performance data"
- **Acceptance criteria:**
  - [ ] All benchmarks produce structured JSON output
  - [ ] BENCHMARKS.md auto-generated from results
  - [ ] CI runs benchmarks weekly
  - [ ] Regressions detectable via comparison script
- **Verification:** Run benchmark suite, verify report generation

---

### EPIC-013: v0.2.0 Release & Documentation
- **Owner:** Project Maintainer
- **Duration:** 1 week (Sprint 7, second half)
- **Priority:** P0 (Critical)
- **Dependencies:** EPIC-009, EPIC-010, EPIC-011, EPIC-012 (all post-MVP work)
- **Phase exit criteria:** v0.2.0 tagged, documentation updated, CHANGELOG complete

#### STORY-013-01: Documentation Update for v0.2.0
- **Description:** As a MAKER user, I want updated documentation covering adaptive k, semantic matching, and ensemble features, so that I can use the new capabilities.
- **Context files:** `README.md`, `CHANGELOG.md`
- **Work to do:**
  - [ ] Update README.md:
    - Add "Semantic Matching" section with code/embedding/exact matcher comparison
    - Add "Adaptive K" section explaining dynamic optimization
    - Add "Multi-Model Ensemble" section with configuration examples
    - Update architecture diagram with matcher and ensemble components
    - Update benchmark section with domain-specific results
  - [ ] Add rustdoc to all new public APIs (matchers, adaptive, ensemble)
  - [ ] Create `examples/coding_task.rs` — semantic matching on a coding task
  - [ ] Create `examples/ensemble_demo.rs` — multi-model voting demo
  - [ ] Update existing examples to show adaptive k option
- **Acceptance criteria:**
  - [ ] README covers all v0.2.0 features with examples
  - [ ] All new public APIs have doc comments with examples
  - [ ] Doc tests pass: `cargo test --doc`
  - [ ] New examples compile and run
- **Verification:** `cargo doc --open`, review all new sections

---

#### STORY-013-02: CHANGELOG & Release
- **Description:** As a MAKER user, I want a v0.2.0 release with release notes, so that I can upgrade and use new features.
- **Context files:** `CHANGELOG.md`
- **Work to do:**
  - [ ] Update CHANGELOG.md:
    - `[0.2.0] - 2026-MM-DD`
    - Added: Adaptive k-margin, Semantic matching (embedding + code AST), Multi-model ensemble, Domain benchmarks
    - Changed: VoteConfig extended with matcher and adaptive options
    - Deprecated: None
  - [ ] Run full test suite: `cargo test --all-features`
  - [ ] Run benchmarks: verify no regressions from v0.1.0
  - [ ] Run `cargo publish --dry-run`
  - [ ] Tag: `git tag -a v0.2.0 -m "MAKER Framework v0.2.0 — Semantic Matching & Ensemble Voting"`
  - [ ] Create GitHub release with release notes
  - [ ] Publish to crates.io: `cargo publish`
- **Acceptance criteria:**
  - [ ] All tests pass (including new modules)
  - [ ] 95% test coverage maintained
  - [ ] CHANGELOG follows Keep a Changelog format
  - [ ] GitHub release published with notes
  - [ ] crates.io published successfully
- **Verification:** `cargo publish --dry-run` succeeds, CI green

---

#### STORY-013-03: Migration Guide
- **Description:** As a v0.1.0 user, I want a migration guide for upgrading to v0.2.0, so that I can adopt new features without breaking existing integrations.
- **Context files:** `README.md`
- **Work to do:**
  - [ ] Create `MIGRATION-v0.2.0.md`:
    - Breaking changes (if any): list exact changes to public API
    - New optional fields in ConfigRequest/VoteRequest (backward compatible)
    - How to enable adaptive k (opt-in)
    - How to configure matchers (opt-in, default is exact)
    - How to configure ensemble (opt-in, default is single model)
  - [ ] Add deprecation notices to any changed APIs
  - [ ] Link migration guide from CHANGELOG and README
- **Acceptance criteria:**
  - [ ] Migration guide covers all API changes
  - [ ] v0.1.0 code compiles against v0.2.0 without changes (backward compat)
  - [ ] Guide includes code snippets for each new feature
- **Verification:** Review guide, verify backward compatibility with v0.1.0 test suite

---

## Release Checklist (v0.2.0)

Before tagging v0.2.0:

- [ ] All acceptance criteria met across EPIC-009, EPIC-010, EPIC-011, EPIC-012, EPIC-013
- [ ] Adaptive k reduces cost by 20%+ while maintaining zero errors (deterministic tasks)
- [ ] Semantic matching >90% accuracy on coding tasks
- [ ] Ensemble voting improves reliability over single-model
- [ ] Cost-aware routing saves 30%+ vs. single expensive model
- [ ] 95% test coverage maintained
- [ ] All benchmarks pass and results documented
- [ ] README updated with all v0.2.0 features
- [ ] API docs complete for new modules
- [ ] CHANGELOG updated
- [ ] Migration guide written
- [ ] CI/CD pipeline green
- [ ] `cargo publish --dry-run` succeeds
- [ ] Git tag: `git tag -a v0.2.0`
- [ ] GitHub release created

---

**End of Execution Plan**
