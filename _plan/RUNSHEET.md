# Runsheet: MAKER Framework

> Work through each sprint's items in order.
> Check off acceptance criteria as you complete each story.
> Items marked with ⊘ are blocked until dependencies complete.

**Generated:** 2026-01-30
**Updated:** 2026-01-30 (post-MVP sprints added)
**Total Story Points:** 132 (MVP) + 93 (post-MVP) = 225
**Timeline:** 10 weeks (3 sprints MVP + 4 sprints post-MVP)

---

## Sprint 1 — Core MAKER Algorithms — Days 1-5 ✅ COMPLETE

**Story Points:** 42 | **Status:** All items complete

- [x] STORY-001-01 through STORY-001-05 (EPIC-001: Core Library) ✅
- [x] STORY-004-01 through STORY-004-04 (EPIC-004: Observability) ✅
- [x] STORY-005-01 (EPIC-005: Property Tests) ✅

---

## Sprint 2 — MCP Server Integration — Days 6-10 ✅ COMPLETE

**Story Points:** 58 | **Status:** All items complete

- [x] STORY-002-01 through STORY-002-06 (EPIC-002: LLM Providers) ✅
- [x] STORY-003-01 through STORY-003-06 (EPIC-003: MCP Server) ✅
- [x] STORY-008-01 through STORY-008-03 (EPIC-008: Security) ✅
- [x] STORY-005-03 (EPIC-005: MCP Compliance Tests) ✅

---

## Sprint 3 — Validation & Hardening — Days 11-14 ✅ COMPLETE

**Story Points:** 32 | **Status:** All items complete, v0.1.0 released

- [x] STORY-006-01 through STORY-006-04 (EPIC-006: Demo & Benchmarks) ✅
- [x] STORY-007-01 through STORY-007-05 (EPIC-007: Documentation) ✅
- [x] STORY-005-02, STORY-005-04 (EPIC-005: Monte Carlo, CI/CD) ✅

**v0.1.0 Released:** https://github.com/zircote/maker-rs/releases/tag/v0.1.0

---

## Sprint 4 — Adaptive K-Margin — Weeks 3-4 ✅ COMPLETE

**Objective:** Implement dynamic k-margin adjustment for cost optimization.
**Story Points:** 23 | **Status:** All items complete

### EPIC-011: Adaptive K-Margin (23 pts)

- [x] **STORY-011-01** — K-Margin Estimator (5 pts) ✅
  - [x] Create `src/core/adaptive.rs` with `KEstimator` struct
  - [x] Implement EMA-based p-hat estimation (α=0.1)
  - [x] Implement `recommended_k()` using live p-hat in k_min formula
  - [x] Add k bounds: floor=2, ceiling=10
  - [x] ✓ p_hat converges to ±5% of true p after 20 observations
  - [x] ✓ k increases when observed p drops, decreases when p rises
  - [x] ✓ k never violates bounds

- [x] **STORY-011-02** — Adaptive Voting Integration (8 pts) ✅
  - [x] Add `vote_with_margin_adaptive()` wrapping `vote_with_margin` with `KEstimator`
  - [x] Feed vote results back to estimator after each decision
  - [x] Preserve backward compatibility (adaptive: false = static k)
  - [x] ✓ Existing tests pass unchanged (357 unit + 35 integration)
  - [x] ✓ Adaptive reduces API calls vs. static k
  - [x] ✓ Zero errors maintained on deterministic tasks

- [x] **STORY-011-03** — Adaptive K MCP Extension (5 pts) ✅
  - [x] Extend `ConfigRequest` with adaptive_k, ema_alpha, k_bounds
  - [x] Extend `VoteResponse` with k_used, p_hat
  - [x] Extend `ServerConfig` with adaptive_k, ema_alpha, k_bounds
  - [x] ✓ Configure adaptive mode via MCP tool
  - [x] ✓ VoteResponse includes k_used and p_hat
  - [x] ✓ Backward compatible

- [x] **STORY-011-04** — Adaptive K Validation Suite (5 pts) ✅
  - [x] Property tests: adaptive k respects bounds, p_hat stays valid, high p gives lower k
  - [x] Monte Carlo: adaptive cost comparison vs static
  - [x] Regression: zero errors on deterministic task with adaptive k
  - [x] Stress test: k recovers when p suddenly drops
  - [x] ✓ All 21 property tests pass
  - [x] ✓ Cost comparison validates adaptive approach
  - [x] ✓ Stress test passes (k increases on p drop)

**Sprint 4 Gate:**
- [x] Adaptive k implemented with configurable bounds and EMA estimation
- [x] All existing tests pass (backward compatible) — 423 total tests, 0 failures
- [x] KEstimator converges within 20 observations
- [x] Clippy clean, no warnings

---

## Sprint 5 — Semantic Matching — Weeks 5-6 ✅ COMPLETE

**Objective:** Extend voting to non-deterministic tasks via pluggable matchers.
**Story Points:** 28 | **Status:** All items complete

### EPIC-009: Semantic Matching (28 pts)

- [x] **STORY-009-01** — Matcher Trait Abstraction (5 pts) ✅
  - [x] Create `src/core/matcher.rs` with `CandidateMatcher` trait
  - [x] Implement `ExactMatcher` (current behavior)
  - [x] Modify `VoteRace` to accept `Arc<dyn CandidateMatcher>`
  - [x] Update `vote_with_margin` to use matcher for candidate grouping
  - [x] ✓ All existing tests pass with ExactMatcher (backward compat)
  - [x] ✓ CandidateMatcher is object-safe, Send + Sync

- [x] **STORY-009-02** — Embedding-Based Similarity Matcher (8 pts) ✅
  - [x] Create `src/core/matchers/embedding.rs`
  - [x] Define `EmbeddingClient` trait, implement for Ollama and OpenAI
  - [x] Implement cosine similarity with configurable threshold (default 0.92)
  - [x] Add embedding cache to avoid redundant API calls
  - [x] ✓ Cosine similarity correct for known pairs
  - [x] ✓ Cache hit rate >80% in voting scenarios
  - [x] ✓ Semantically equivalent code snippets grouped correctly

- [x] **STORY-009-03** — Code AST Matcher (8 pts) ✅
  - [x] Create `src/core/matchers/code.rs`
  - [x] Add tree-sitter dependency for multi-language parsing (behind `code-matcher` feature)
  - [x] Implement AST normalization: alpha-renaming, comment stripping
  - [x] Implement LCS-based token similarity scoring
  - [x] Support Rust, Python, JavaScript grammars
  - [x] ✓ `def foo(x): return x+1` ≡ `def bar(y): return y + 1`
  - [x] ✓ Parsing errors fall back to whitespace-normalized comparison

- [x] **STORY-009-04** — Matcher MCP Configuration (3 pts) ✅
  - [x] Extend `ConfigRequest` with matcher config (exact/embedding/code)
  - [x] Extend `VoteResponse` with matcher_type, candidate_groups
  - [x] Add matcher parameter to VoteRequest for per-call override
  - [x] ✓ Default matcher is "exact" (backward compatible)
  - [x] ✓ Invalid matcher config returns clear error via `MatcherConfig::validate()`

- [x] **STORY-009-05** — Semantic Matching Test Suite (4 pts) ✅
  - [x] Build test corpus: 47 equivalent code pairs, 50 NL pairs, 25 negative code pairs
  - [x] Test CodeMatcher accuracy: >95% on code corpus (achieved 100% on curated corpus)
  - [x] Test EmbeddingMatcher mechanism with mock client + high-overlap NL pairs
  - [x] Test false positive rate <5% on structurally different code pairs
  - [x] ✓ CodeMatcher >95% accuracy (per-language and combined)
  - [x] ✓ EmbeddingMatcher mechanism validated (real accuracy requires live providers)
  - [x] ✓ Code false positive rate <5%
  - [x] ✓ Reflexivity and symmetry properties verified for all matchers

**Sprint 5 Gate:**
- [x] Matcher trait extensible (custom matchers via `CandidateMatcher` trait)
- [x] Code AST matcher handles Python, Rust, JS (behind `code-matcher` feature)
- [x] Embedding matcher works with Ollama and OpenAI clients
- [x] All existing exact-match tests still pass — 419 unit + 35 integration + 21 property
- [x] 25 semantic matching tests (16 without code-matcher, 25 with)
- [x] Clippy clean, zero warnings
- [x] Total tests: 501 (with code-matcher feature), 485 (without)

---

## Sprint 6 — Multi-Model Ensemble — Weeks 7-8

**Objective:** Enable voting across heterogeneous LLM models.
**Story Points:** 25

### EPIC-010: Multi-Model Ensemble (25 pts)

- [ ] **STORY-010-01** — Ensemble Configuration (5 pts)
  - [ ] Create `src/llm/ensemble.rs` with EnsembleConfig, ModelSlot, EnsembleStrategy
  - [ ] Implement RoundRobin, CostAware, ReliabilityWeighted strategies
  - [ ] Implement model selection for each sample index
  - [ ] ✓ 2-5 models configurable
  - [ ] ✓ RoundRobin distributes evenly
  - [ ] ✓ Single-model mode unchanged

- [ ] **STORY-010-02** — Ensemble Sampling Integration (5 pts) ⊘ STORY-010-01
  - [ ] Modify `collect_samples` for ensemble model selection
  - [ ] Tag samples with source model name
  - [ ] Handle per-model failures with graceful fallback
  - [ ] ✓ Samples from multiple models (verified by model_name)
  - [ ] ✓ Model failure doesn't halt voting
  - [ ] ✓ Latency ≈ max(individual) not sum

- [ ] **STORY-010-03** — Cost-Aware Routing (5 pts) ⊘ STORY-010-01, STORY-010-02
  - [ ] Implement 3-phase CostAware strategy (cheap → medium → expensive)
  - [ ] Track cost per model in VoteResult
  - [ ] Emit EscalationTriggered event on tier change
  - [ ] ✓ Easy tasks resolved by cheap model only
  - [ ] ✓ Total cost < single-expensive-model by 30%+
  - [ ] ✓ Cost breakdown by model in VoteResponse

- [ ] **STORY-010-04** — Ensemble MCP Extension (5 pts) ⊘ STORY-010-02
  - [ ] Extend ConfigRequest with ensemble config
  - [ ] Extend VoteResponse with ensemble_metrics
  - [ ] Add per-call ensemble override
  - [ ] ✓ Ensemble configurable via MCP
  - [ ] ✓ Per-model breakdown in response
  - [ ] ✓ Backward compatible

- [ ] **STORY-010-05** — Cross-Model Benchmarks (5 pts) ⊘ STORY-010-03
  - [ ] Create `benches/ensemble_comparison.rs`
  - [ ] Benchmark: single-model vs. round-robin vs. cost-aware
  - [ ] Monte Carlo: 1,000 trials at s=100, s=1000
  - [ ] ✓ Ensemble error rate < individual models
  - [ ] ✓ Cost-aware 30%+ cheaper than expensive-only
  - [ ] ✓ Results documented

**Sprint 6 Gate:**
- [ ] Ensemble voting works across 2+ models
- [ ] Cost-aware routing reduces cost by 30%+
- [ ] Ensemble improves reliability over any single model
- [ ] Backward compatible (single-model default unchanged)
- [ ] 95% test coverage maintained

---

## Sprint 7 — Benchmark Suite & v0.2.0 Release — Weeks 9-10

**Objective:** Comprehensive benchmarks and v0.2.0 release.
**Story Points:** 17

### EPIC-012: Benchmark Suite (10 pts)

- [ ] **STORY-012-01** — Coding Task Benchmark (3 pts)
  - [ ] Create `benches/coding_tasks/` with 10 coding benchmarks
  - [ ] Use CodeMatcher for equivalence
  - [ ] ✓ >90% accuracy on trivial/moderate tasks
  - [ ] ✓ >80% accuracy on complex tasks

- [ ] **STORY-012-02** — Math & Logic Benchmark (3 pts)
  - [ ] Create `benches/math_logic/` with arithmetic/logic benchmarks
  - [ ] Use ExactMatcher (deterministic)
  - [ ] ✓ Zero errors on arithmetic tasks with k=3
  - [ ] ✓ Θ(s ln s) scaling holds

- [ ] **STORY-012-03** — Data Analysis Benchmark (2 pts)
  - [ ] Create `benches/data_analysis/` with CSV/SQL benchmarks
  - [ ] Use EmbeddingMatcher for approximate matching
  - [ ] ✓ >85% accuracy on data analysis tasks
  - [ ] ✓ Numerical outputs within 1% tolerance

- [ ] **STORY-012-04** — Benchmark Dashboard & Reporting (2 pts)
  - [ ] Create benchmark report generator (JSON + markdown)
  - [ ] Add weekly CI benchmark run
  - [ ] ✓ All benchmarks produce structured JSON
  - [ ] ✓ BENCHMARKS.md auto-generated

### EPIC-013: v0.2.0 Release (7 pts)

- [ ] **STORY-013-01** — Documentation Update (3 pts) ⊘ EPIC-009, EPIC-010, EPIC-011
  - [ ] Update README with adaptive k, semantic matching, ensemble sections
  - [ ] Add rustdoc to all new public APIs
  - [ ] Create coding_task.rs and ensemble_demo.rs examples
  - [ ] ✓ README covers all v0.2.0 features
  - [ ] ✓ Doc tests pass

- [ ] **STORY-013-02** — CHANGELOG & Release (2 pts) ⊘ STORY-013-01
  - [ ] Update CHANGELOG.md for v0.2.0
  - [ ] Full test suite pass, 95% coverage
  - [ ] Tag v0.2.0, create GitHub release, publish to crates.io
  - [ ] ✓ All tests pass
  - [ ] ✓ GitHub release published

- [ ] **STORY-013-03** — Migration Guide (2 pts) ⊘ STORY-013-01
  - [ ] Create MIGRATION-v0.2.0.md
  - [ ] Document all API changes (all backward compatible)
  - [ ] ✓ v0.1.0 code compiles against v0.2.0 without changes
  - [ ] ✓ Guide includes code snippets for new features

**Sprint 7 Gate:**
- [ ] Domain benchmarks pass for coding, math, data analysis
- [ ] v0.2.0 tagged and released
- [ ] Documentation complete for all new features
- [ ] Migration guide validates backward compatibility
- [ ] 95% test coverage on entire codebase

---

## Final Validation (v0.2.0)

- [ ] Adaptive k reduces cost by 20%+ while maintaining zero errors (deterministic tasks)
- [ ] Semantic matching >90% accuracy on coding tasks
- [ ] Ensemble voting improves reliability over single-model
- [ ] Cost-aware ensemble 30%+ cheaper than single expensive model
- [ ] Domain benchmarks pass (coding, math, data analysis)
- [ ] 95% test coverage across all modules
- [ ] Zero errors on 10-disk Hanoi with all feature combinations
- [ ] All 4 MCP tools work with new features (backward compatible)
- [ ] README, API docs, examples updated
- [ ] CHANGELOG, migration guide complete
- [ ] CI/CD pipeline green
- [ ] v0.2.0 published to crates.io

---

## Release Commands (v0.2.0)

```bash
# Full test suite
cargo test --all-features
cargo llvm-cov --fail-under-lines 90 --ignore-filename-regex '(main\.rs|maker-mcp\.rs)'
cargo fmt --check
cargo clippy -- -D warnings
cargo doc --no-deps

# Benchmarks
cargo bench --bench coding_tasks
cargo bench --bench math_logic
cargo bench --bench ensemble_comparison

# Dry run publish
cargo publish --dry-run

# Tag and release
git tag -a v0.2.0 -m "MAKER Framework v0.2.0 — Semantic Matching & Ensemble Voting"
git push origin v0.2.0

# Publish to crates.io
cargo publish
```

---

**Runsheet Status:** Sprints 1-5 ✅ COMPLETE | Sprints 6-7 ⏳ PENDING
**v0.1.0 Released:** 2026-01-30
**Target v0.2.0:** Week 10
