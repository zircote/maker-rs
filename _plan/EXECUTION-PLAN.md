# Execution Plan: MAKER Framework

> This file is a complete execution plan for implementing the MAKER Framework in Rust.
> It was generated from JIRA-STRUCTURE.md and the project planning artifacts.
> To execute: open a Claude Code session in this repository and say:
> "Read _plan/EXECUTION-PLAN.md and begin executing the work items"

**Generated:** 2026-01-30
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

The MVP timeline is 14 days across 3 sprints (127 story points total).

---

## Success Criteria

From SUCCESS-METRICS.md and PROJECT-PLAN.md:

- [ ] **Zero errors** on 10-disk Towers of Hanoi (1,023 steps)
- [ ] **95% test coverage** on all modules (enforced by CI)
- [ ] **Cost scaling Θ(s ln s)** validated within 20% tolerance
- [ ] **All 4 MCP tools functional** (vote, validate, calibrate, configure)
- [ ] **Claude Code integration** working (manual test)
- [ ] **API retry success rate** ≥99%
- [ ] **Security audit** passed (prompt injection mitigation, schema validation)
- [ ] **Documentation** complete (README, API docs, examples)
- [ ] **v0.1.0 GitHub release** tagged

---

## Risk Awareness

From RISK-REGISTER.md — watch for these during implementation:

| Risk | Indicator | Mitigation |
|------|-----------|------------|
| **R-001: Mathematical errors** | Property tests failing, wrong k_min values | Reference arxiv paper, Monte Carlo validation with 10K trials |
| **R-002: API reliability** | 429 errors, timeouts | Exponential backoff with jitter, Ollama fallback |
| **R-003: MCP security** | Schema bypass, injection | Red-flag all outputs, strict schema validation |
| **R-005: Coverage gaps** | Coverage < 95% | Add tests immediately, don't defer |
| **R-006: Cost deviation** | Cost ≠ Θ(s ln s) | Profile, benchmark, check algorithm correctness |

---

## Phase 1: Core MAKER Algorithms (Days 1-5)

**Objective:** Implement mathematically correct MAKER protocols with comprehensive testing.
**Exit Criteria:** Zero errors on 3-disk Hanoi, 95% coverage, events emit correctly.

---

### EPIC-001: Core MAKER Library
- **Owner:** Project Maintainer
- **Duration:** 4 days (Days 1-4)
- **Priority:** P0 (Critical)
- **Dependencies:** None
- **Phase exit criteria:** k_min correct, voting converges, red-flags work, 3-disk Hanoi passes

#### STORY-001-01: k_min Calculation
- **Description:** As a MAKER framework user, I want to calculate the minimum k-margin required for target reliability, so that I can configure voting to achieve mathematically-grounded error correction.
- **Context files:** `docs/technical-implementation-manual.txt` (Section 2.2), arxiv paper 2511.09030
- **Work to do:**
  - [ ] Create `src/lib.rs` and `src/core/mod.rs` module structure
  - [ ] Create `src/core/kmin.rs` with `calculate_kmin(p: f64, t: f64, s: usize, m: usize) -> usize`
  - [ ] Implement formula: `k_min = ⌈ln(t^(-m/s)-1) / ln((1-p)/p)⌉`
  - [ ] Add input validation: p ∈ (0.5, 1.0), t ∈ (0, 1), m=1, s > 0
  - [ ] Handle edge cases: p → 1 (k_min → 1), s → ∞ (k_min grows logarithmically)
  - [ ] Write doc comments with mathematical derivation
  - [ ] Add unit tests for paper's test cases (20-disk Hanoi: k_min=3-4 for p=0.85, t=0.95)
- **Acceptance criteria:**
  - [ ] Function returns correct k_min for test cases from paper
  - [ ] Property test: k_min increases logarithmically with s (proptest)
  - [ ] Property test: k_min decreases as p approaches 1.0
  - [ ] Property test: k_min increases as t approaches 1.0
- **Verification:** Run `cargo test kmin` — all tests pass

---

#### STORY-001-02: Vote Race State Tracking
- **Description:** As a MAKER voting engine, I want to track vote counts for each candidate and detect k-margin leaders, so that I can terminate voting at the optimal stopping point.
- **Context files:** `docs/technical-implementation-manual.txt` (Section 2)
- **Work to do:**
  - [ ] Create `src/core/voting.rs`
  - [ ] Define `CandidateId` as newtype over String or hash
  - [ ] Implement `VoteRace` struct with `HashMap<CandidateId, usize>` for vote counts
  - [ ] Implement `VoteRace::cast_vote(&mut self, candidate: CandidateId)`
  - [ ] Implement `VoteRace::check_winner(&self, k_margin: usize) -> Option<CandidateId>` using Gambler's Ruin boundary
  - [ ] Handle edge case: single candidate (return None, request more samples)
  - [ ] Make thread-safe with `Arc<Mutex<VoteRace>>` or interior mutability
  - [ ] Emit `VoteCast` event on each vote (integrate with EventBus in EPIC-004)
  - [ ] Emit `VoteDecided` event when winner declared
- **Acceptance criteria:**
  - [ ] Winner correctly identified when lead = k_margin
  - [ ] No false positives (winner declared before k-margin reached)
  - [ ] Thread-safe for concurrent vote casting
  - [ ] Events emitted to EventBus
- **Verification:** Run `cargo test voting` — convergence tests pass

---

#### STORY-001-03: Red-Flagging Parsers
- **Description:** As a MAKER framework, I want to discard malformed LLM outputs without attempting repair, so that I maintain error decorrelation for effective voting.
- **Context files:** `docs/technical-implementation-manual.txt` (Section 3)
- **Work to do:**
  - [ ] Create `src/core/redflag.rs`
  - [ ] Define `RedFlag` enum: `TokenLengthExceeded { actual: usize, limit: usize }`, `FormatViolation { message: String }`, `LogicLoop`
  - [ ] Implement `validate_token_length(content: &str, limit: usize) -> Result<(), RedFlag>`
  - [ ] Implement `validate_json_schema<T: DeserializeOwned>(content: &str) -> Result<T, RedFlag>`
  - [ ] Add logic loop detection stub (future: semantic analysis)
  - [ ] Return `Result<ValidatedResponse, RedFlag>` — never repair, only accept or reject
  - [ ] Emit `RedFlagTriggered` event on validation failure
- **Acceptance criteria:**
  - [ ] Rejects response with 701 tokens when limit is 700
  - [ ] Rejects response missing required JSON fields
  - [ ] Accepts valid responses without false positives
  - [ ] Red-flag rate < 10% on well-calibrated models (empirical test)
- **Verification:** Run `cargo test redflag` — all validation tests pass

---

#### STORY-001-04: Microagent Orchestration
- **Description:** As a MAKER task executor, I want to decompose tasks into m=1 subtasks per agent, so that I minimize context burden and maximize per-step reliability.
- **Context files:** `docs/technical-implementation-manual.txt` (Section 1.1, 5.3)
- **Work to do:**
  - [ ] Create `src/core/orchestration.rs`
  - [ ] Define `Subtask` struct: `{ step_id: usize, prompt: String, state: State }`
  - [ ] Define `AgentOutput` struct: `{ move_action: String, next_state: State }`
  - [ ] Define `State` as serializable state representation (generic or task-specific)
  - [ ] Implement `TaskDecomposer` trait for task-specific decomposition
  - [ ] Enforce m=1 constraint: agent receives only current step state, no history
  - [ ] Implement state transfer: system uses `next_state` from output for next agent
  - [ ] Add state hash validation before next agent invocation
  - [ ] Emit `StepCompleted` event with step_id and state_hash
- **Acceptance criteria:**
  - [ ] Cannot create agent with m > 1 (compile-time via type system or runtime panic)
  - [ ] Agent output includes both move and next_state
  - [ ] System uses next_state for subsequent agent, not model's interpretation
  - [ ] State hash prevents undetected state corruption
- **Verification:** Run `cargo test orchestration` — state transfer tests pass

---

#### STORY-001-05: Parallel Voting Integration
- **Description:** As a MAKER voting engine, I want to integrate parallel sampling, red-flagging, and voting, so that I can execute the complete first-to-ahead-by-k protocol.
- **Context files:** `docs/technical-implementation-manual.txt` (Section 5.2)
- **Work to do:**
  - [ ] Create `src/core/executor.rs`
  - [ ] Implement `vote_with_margin(prompt: &str, k: usize, client: &dyn LlmClient, config: VoteConfig) -> Result<VoteResult, VoteError>`
  - [ ] Orchestrate: collect samples → filter via red-flagging → cast votes → check winner
  - [ ] Handle timeout: max_samples limit, return error if no convergence
  - [ ] Define `VoteResult { winner: CandidateId, vote_counts: HashMap, total_samples: usize, cost: CostMetrics }`
  - [ ] Write integration test with mock LLM returning deterministic samples
  - [ ] Write integration test for 3-disk Towers of Hanoi (7 steps)
- **Acceptance criteria:**
  - [ ] Zero errors on 3-disk Towers of Hanoi (7 steps) with mock LLM
  - [ ] Voting converges within expected sample count (Θ(k))
  - [ ] Red-flagged samples excluded from vote pool
  - [ ] Timeout returns error when max_samples exceeded
- **Verification:** Run `cargo test executor` and `cargo test --test hanoi_3disk`

---

### EPIC-004: Event-Driven Observability
- **Owner:** Project Maintainer
- **Duration:** 3 days (Days 3-5, parallel with EPIC-001)
- **Priority:** P1 (High)
- **Dependencies:** EPIC-001 (partial - needs event definitions after Day 2)
- **Phase exit criteria:** Events emit, logging works, metrics tracked

#### STORY-004-01: Event Definitions
- **Description:** As a MAKER framework, I want to define all event types as an enum, so that I can emit structured, immutable events for observability.
- **Context files:** `PROJECT-CONTEXT.md` (Logging Strategy section)
- **Work to do:**
  - [ ] Create `src/events/mod.rs`
  - [ ] Define `MakerEvent` enum with `#[serde(tag = "type")]`:
    - `SampleRequested { model: String, prompt_hash: String, temperature: f64, timestamp: SystemTime }`
    - `SampleCompleted { model: String, tokens_used: usize, latency_ms: u64, red_flags: Vec<RedFlag>, timestamp: SystemTime }`
    - `RedFlagTriggered { flag_type: String, token_count: Option<usize>, format_error: Option<String>, timestamp: SystemTime }`
    - `VoteCast { candidate_id: String, vote_count: usize, margin: i32, timestamp: SystemTime }`
    - `VoteDecided { winner_id: String, total_votes: usize, k_margin: usize, timestamp: SystemTime }`
    - `StepCompleted { step_id: usize, state_hash: String, cumulative_cost: f64, timestamp: SystemTime }`
  - [ ] Derive `Clone`, `Debug`, `Serialize`, `Deserialize` for all events
  - [ ] Add timestamp to all events using `std::time::SystemTime`
- **Acceptance criteria:**
  - [ ] All events serialize to JSON with type tag
  - [ ] Events are immutable (no interior mutability)
  - [ ] Clone implementation is efficient
  - [ ] Timestamp precision sufficient for latency tracking (ms)
- **Verification:** Run `cargo test events` — serialization tests pass

---

#### STORY-004-02: EventBus Implementation
- **Description:** As a MAKER framework, I want a central EventBus for publishing and subscribing to events, so that core logic is decoupled from observability implementations.
- **Context files:** None (standard pattern)
- **Work to do:**
  - [ ] Create `src/events/bus.rs`
  - [ ] Implement `EventBus` struct with `tokio::sync::broadcast::Sender<MakerEvent>`
  - [ ] Implement `EventBus::new(capacity: usize) -> Self` (default 1024)
  - [ ] Implement `EventBus::emit(&self, event: MakerEvent)` (fire-and-forget, non-blocking)
  - [ ] Implement `EventBus::subscribe(&self) -> broadcast::Receiver<MakerEvent>`
  - [ ] Handle lagging receivers: they drop old events (broadcast default behavior)
  - [ ] Make EventBus `Clone` for sharing across tasks
- **Acceptance criteria:**
  - [ ] Multiple subscribers receive all events
  - [ ] Emit is non-blocking (doesn't wait for receivers)
  - [ ] Lagging receivers don't block emitters
  - [ ] Integration test: emit 1000 events, all subscribers receive them
- **Verification:** Run `cargo test bus` — broadcast tests pass

---

#### STORY-004-03: Logging Observer
- **Description:** As a MAKER operator, I want structured logs for all events at appropriate log levels, so that I can debug issues and track task progress.
- **Context files:** `PROJECT-CONTEXT.md` (Log Levels section)
- **Work to do:**
  - [ ] Create `src/events/observers/logging.rs`
  - [ ] Add `tracing` and `tracing-subscriber` dependencies to Cargo.toml
  - [ ] Implement `LoggingObserver` that subscribes to EventBus
  - [ ] Map events to log levels:
    - `VoteDecided` → INFO with step_id, winner, votes
    - `RedFlagTriggered` → WARN with flag_type
    - `SampleRequested/Completed` → DEBUG
    - `StepCompleted` → INFO
  - [ ] Use tracing macros with structured fields: `tracing::info!(step_id = %step, winner = %winner, "Vote decided")`
  - [ ] Spawn observer task with `tokio::spawn`
  - [ ] Support RUST_LOG env var for filtering
- **Acceptance criteria:**
  - [ ] VoteDecided logged at INFO with step_id, winner, votes
  - [ ] RedFlagTriggered logged at WARN with flag_type
  - [ ] SampleRequested/Completed logged at DEBUG
  - [ ] Logs are machine-parseable (JSON format with tracing-subscriber fmt)
- **Verification:** Run with `RUST_LOG=debug cargo test` — logs appear correctly

---

#### STORY-004-04: Metrics Observer
- **Description:** As a MAKER operator, I want Prometheus-compatible metrics for votes, red-flags, latency, and cost, so that I can monitor performance and cost in production.
- **Context files:** `SUCCESS-METRICS.md`
- **Work to do:**
  - [ ] Create `src/events/observers/metrics.rs`
  - [ ] Add `metrics` crate dependency (or `prometheus` crate)
  - [ ] Implement `MetricsObserver` that subscribes to EventBus
  - [ ] Track counters:
    - `maker_votes_total` (labels: winner)
    - `maker_red_flags_total` (labels: flag_type)
    - `maker_samples_total` (labels: model)
  - [ ] Track histograms:
    - `maker_api_latency_ms` (buckets: 10, 50, 100, 500, 1000, 5000, 10000)
    - `maker_cost_per_step_usd`
  - [ ] Implement `MetricsObserver::report() -> String` for stdout export (MVP)
  - [ ] Optional: expose /metrics HTTP endpoint (post-MVP)
- **Acceptance criteria:**
  - [ ] Metrics increment on corresponding events
  - [ ] Histogram buckets appropriate for API latency (10ms-10s)
  - [ ] Cost metric includes model-specific pricing
  - [ ] Metrics queryable via stdout report
- **Verification:** Run integration test, check metrics report output

---

### EPIC-005: Testing Infrastructure (Phase 1 portion)
- **Owner:** Project Maintainer
- **Duration:** Ongoing (starts Day 1)
- **Priority:** P0 (Critical)
- **Dependencies:** None
- **Phase exit criteria:** 95% coverage on Phase 1 code, property tests pass

#### STORY-005-01: Property-Based Testing Framework
- **Description:** As a MAKER developer, I want property-based tests for probabilistic guarantees, so that I can validate voting convergence and k_min correctness across wide input ranges.
- **Context files:** `BEST-PRACTICES.md` (Property-Based Testing section)
- **Work to do:**
  - [ ] Add `proptest = "1"` dependency to Cargo.toml
  - [ ] Create `tests/properties.rs` for property-based tests
  - [ ] Write property: `k_min` increases logarithmically with `s` (for fixed p, t)
  - [ ] Write property: voting converges for any p > 0.5 with sufficient samples
  - [ ] Write property: red-flag rate < threshold for randomly generated valid inputs
  - [ ] Configure test iterations: 1000 per property (configurable via PROPTEST_CASES)
  - [ ] Ensure shrinking produces minimal failing test cases
- **Acceptance criteria:**
  - [ ] Properties pass with 1000+ random inputs each
  - [ ] Shrinking produces minimal failing test cases
  - [ ] Tests complete in <60s on CI
  - [ ] Property failures include reproducible seeds
- **Verification:** Run `cargo test --test properties` — all properties pass

---

## Phase 2: MCP Server Integration (Days 6-10)

**Objective:** Expose MAKER as an MCP server with multi-provider LLM support.
**Exit Criteria:** All 4 MCP tools functional, Claude Code integration working, parallel sampling operational.

---

### EPIC-002: LLM Provider Abstraction
- **Owner:** Project Maintainer
- **Duration:** 4 days (Days 6-9)
- **Priority:** P0 (Critical)
- **Dependencies:** EPIC-001 (voting needs LLM client)
- **Phase exit criteria:** 3 providers work, retry logic handles 429s, parallel sampling operational

#### STORY-002-01: LlmClient Trait
- **Description:** As a MAKER framework, I want a unified trait for LLM API calls, so that voting logic is provider-agnostic.
- **Context files:** None (standard pattern)
- **Work to do:**
  - [ ] Create `src/llm/mod.rs`
  - [ ] Add `async-trait = "0.1"` dependency
  - [ ] Define `LlmResponse { content: String, tokens_used: TokenUsage, latency: Duration }`
  - [ ] Define `TokenUsage { input: usize, output: usize }`
  - [ ] Define `LlmError` enum: `RateLimited { retry_after: Option<Duration> }`, `Timeout`, `NetworkError(String)`, `ApiError { status: u16, message: String }`, `InvalidResponse(String)`
  - [ ] Define `#[async_trait] trait LlmClient: Send + Sync`:
    - `async fn generate(&self, prompt: &str, temperature: f64) -> Result<LlmResponse, LlmError>`
    - `fn model_name(&self) -> &str`
    - `fn cost_per_1k_tokens(&self) -> (f64, f64)` (input, output pricing)
  - [ ] Make trait object-safe (use async_trait macro)
- **Acceptance criteria:**
  - [ ] Trait is async and object-safe
  - [ ] Response includes all fields needed for event emission
  - [ ] LlmError variants cover all failure modes
  - [ ] Doc comments explain retry strategy expectations
- **Verification:** Trait compiles, unit tests for error conversions pass

---

#### STORY-002-02: Ollama Client Implementation
- **Description:** As a MAKER user, I want to use local Ollama models for cost-free inference, so that I can develop and test without cloud API costs.
- **Context files:** Ollama API docs (https://ollama.ai/docs/api)
- **Work to do:**
  - [ ] Create `src/llm/ollama.rs`
  - [ ] Add `reqwest = { version = "0.11", features = ["json"] }` dependency
  - [ ] Implement `OllamaClient { base_url: String, model: String, timeout: Duration }`
  - [ ] Implement `LlmClient` for `OllamaClient`:
    - POST to `{base_url}/api/generate` with `{ model, prompt, options: { temperature } }`
    - Parse response JSON for `response` (content) and token counts
  - [ ] Handle connection refused (Ollama not running) → `NetworkError`
  - [ ] Implement timeout with `tokio::time::timeout`
  - [ ] Write integration test (skip if Ollama not available)
- **Acceptance criteria:**
  - [ ] Successful generation returns Response with content
  - [ ] Token counts parsed from Ollama response
  - [ ] Connection failure returns NetworkError
  - [ ] Timeout returns Timeout error after configured duration
- **Verification:** Run `cargo test ollama` (requires Ollama running or skips)

---

#### STORY-002-03: OpenAI Client Implementation
- **Description:** As a MAKER user, I want to use OpenAI GPT-5.X-nano for cost-effective cloud inference, so that I can balance performance and cost.
- **Context files:** OpenAI API docs
- **Work to do:**
  - [ ] Create `src/llm/openai.rs`
  - [ ] Add `async-openai = "0.20"` dependency (or use reqwest directly)
  - [ ] Implement `OpenAiClient { api_key: String, model: String, timeout: Duration }`
  - [ ] Load API key from `OPENAI_API_KEY` env var
  - [ ] Implement `LlmClient`:
    - Call chat.completions.create with model, messages=[{role: user, content: prompt}], temperature
    - Extract content and usage.{prompt_tokens, completion_tokens}
  - [ ] Handle 429 rate limit: parse `Retry-After` header → `RateLimited { retry_after }`
  - [ ] Set cost: GPT-5.X-nano pricing (check current rates)
- **Acceptance criteria:**
  - [ ] API key loaded from OPENAI_API_KEY env var
  - [ ] Successful generation returns correct token counts
  - [ ] 429 errors return RateLimited with retry_after
  - [ ] Integration test with mock or real API
- **Verification:** Run `cargo test openai` (requires API key or mocks)

---

#### STORY-002-04: Anthropic Client Implementation
- **Description:** As a MAKER user, I want to use Anthropic Claude Haiku for lowest-cost cloud inference, so that I can minimize API costs for large-scale tasks.
- **Context files:** Anthropic API docs
- **Work to do:**
  - [ ] Create `src/llm/anthropic.rs`
  - [ ] Use reqwest with Anthropic messages API (or official SDK when available)
  - [ ] Implement `AnthropicClient { api_key: String, model: String, timeout: Duration }`
  - [ ] Load API key from `ANTHROPIC_API_KEY` env var
  - [ ] Implement `LlmClient`:
    - POST to messages API with model=claude-3-haiku, messages, temperature
    - Extract content and usage.{input_tokens, output_tokens}
  - [ ] Handle rate limits with Retry-After header
  - [ ] Set cost: Haiku pricing ($0.25/M input, $1.25/M output as of 2026)
- **Acceptance criteria:**
  - [ ] API key loaded from ANTHROPIC_API_KEY env var
  - [ ] Token usage includes separate input/output counts
  - [ ] Cost calculation matches current pricing
  - [ ] Rate limit handling consistent with OpenAI client
- **Verification:** Run `cargo test anthropic` (requires API key or mocks)

---

#### STORY-002-05: Exponential Backoff Retry Strategy
- **Description:** As a MAKER framework, I want automatic retry with exponential backoff for transient failures, so that I handle rate limits and network issues gracefully.
- **Context files:** `BEST-PRACTICES.md` (Retry section)
- **Work to do:**
  - [ ] Create `src/llm/retry.rs`
  - [ ] Define `RetryConfig { max_retries: usize, base_delay: Duration, max_delay: Duration, jitter_factor: f64 }`
  - [ ] Implement `async fn call_with_retry<F, T, E>(operation: F, config: RetryConfig) -> Result<T, E>` where F: async FnMut() -> Result<T, E>
  - [ ] Exponential backoff: delay = min(base * 2^attempt, max_delay)
  - [ ] Add jitter: delay * (1.0 + random(0, jitter_factor))
  - [ ] Respect Retry-After header: if `RateLimited { retry_after: Some(d) }`, use d instead of exponential
  - [ ] Classify errors: retryable (429, 500, 503, NetworkError) vs non-retryable (400, 401, 403)
  - [ ] Default config: max_retries=5, base_delay=1s, max_delay=60s, jitter=0.25
- **Acceptance criteria:**
  - [ ] 429 errors retry with exponential + jitter delay
  - [ ] Retry-After header overrides exponential calculation
  - [ ] Non-retryable errors fail immediately
  - [ ] Max retries prevents infinite loops
- **Verification:** Unit test with mock: 3 failures then success

---

#### STORY-002-06: Parallel Sampling with Tokio
- **Description:** As a MAKER voting engine, I want to collect k samples in parallel, so that voting latency is minimized.
- **Context files:** `docs/technical-implementation-manual.txt` (Section 5.2)
- **Work to do:**
  - [ ] Create `src/llm/sampler.rs`
  - [ ] Implement `async fn collect_samples(prompt: &str, num_samples: usize, client: Arc<dyn LlmClient>, config: SampleConfig) -> Vec<Result<LlmResponse, LlmError>>`
  - [ ] Use `tokio::task::JoinSet` for bounded concurrency
  - [ ] Temperature strategy: T=0.0 for first sample, T=config.diversity_temp (0.1) for rest
  - [ ] Emit `SampleRequested` event for each sample
  - [ ] Emit `SampleCompleted` event on each response
  - [ ] Handle timeout: cancel remaining tasks if global timeout exceeded
- **Acceptance criteria:**
  - [ ] Latency ≈ 1x API call time (not num_samples × latency)
  - [ ] First sample deterministic (T=0)
  - [ ] Subsequent samples diverse (T=0.1)
  - [ ] Benchmark: 10 parallel samples complete in ~1.2x single sample time
- **Verification:** Run `cargo bench sampler` (criterion benchmark)

---

### EPIC-003: MCP Server Implementation
- **Owner:** Project Maintainer
- **Duration:** 3 days (Days 8-10)
- **Priority:** P0 (Critical)
- **Dependencies:** EPIC-001 (core lib), EPIC-002 (LLM clients)
- **Phase exit criteria:** 4 tools work, Claude Code integration tested

#### STORY-003-01: rmcp Server Setup
- **Description:** As a MAKER MCP server, I want to initialize rmcp with stdio transport, so that Claude Code can discover and invoke MAKER tools.
- **Context files:** rmcp documentation
- **Work to do:**
  - [ ] Add `rmcp = "0.13"` dependency to Cargo.toml
  - [ ] Create `src/bin/maker-mcp.rs` as server binary
  - [ ] Initialize `rmcp::Server` with stdio transport
  - [ ] Register all 4 tools with server
  - [ ] Handle shutdown signals (SIGINT, SIGTERM) with graceful cleanup
  - [ ] Log server lifecycle events (started, tool registered, shutdown)
  - [ ] Add to Cargo.toml: `[[bin]] name = "maker-mcp" path = "src/bin/maker-mcp.rs"`
- **Acceptance criteria:**
  - [ ] Server starts and listens on stdio
  - [ ] Tools listed in MCP discovery response
  - [ ] Server logs initialization and shutdown
  - [ ] Ctrl+C triggers graceful shutdown
- **Verification:** Run `cargo run --bin maker-mcp` and send initialize request

---

#### STORY-003-02: maker/vote Tool
- **Description:** As a Claude Code user, I want to invoke maker/vote with a prompt and k_margin, so that I get the voted winner with confidence metrics.
- **Context files:** None (new tool design)
- **Work to do:**
  - [ ] Create `src/mcp/tools/vote.rs`
  - [ ] Define `VoteRequest` (serde): `{ prompt: String, k_margin: usize, max_samples: Option<usize>, temperature_diversity: Option<f64>, provider: Option<String> }`
  - [ ] Define `VoteResponse`: `{ winner: String, vote_counts: HashMap<String, usize>, total_samples: usize, red_flags: usize, cost_tokens: usize, cost_usd: f64, latency_ms: u64 }`
  - [ ] Implement `vote_handler(request: VoteRequest, state: ServerState) -> Result<VoteResponse, ToolError>`
  - [ ] Validate: k_margin >= 1, max_samples > k_margin
  - [ ] Call `vote_with_margin` from EPIC-001
  - [ ] Register tool with rmcp server
- **Acceptance criteria:**
  - [ ] Valid request returns VoteResponse with winner
  - [ ] Invalid k_margin (0) returns descriptive error
  - [ ] Cost calculated based on actual LLM provider
  - [ ] Integration test with mock LLM client
- **Verification:** Run MCP server, invoke vote tool via test client

---

#### STORY-003-03: maker/validate Tool
- **Description:** As a Claude Code user, I want to invoke maker/validate to check if a response passes red-flagging, so that I can test red-flag rules before committing to voting.
- **Context files:** None
- **Work to do:**
  - [ ] Create `src/mcp/tools/validate.rs`
  - [ ] Define `ValidateRequest`: `{ response: String, token_limit: Option<usize>, schema: Option<Value> }`
  - [ ] Define `ValidateResponse`: `{ valid: bool, red_flags: Vec<RedFlagInfo> }`
  - [ ] Define `RedFlagInfo`: `{ flag_type: String, details: String }`
  - [ ] Implement `validate_handler` calling red-flag parsers from EPIC-001
  - [ ] Return ALL triggered red-flags (not just first)
  - [ ] Register tool with rmcp server
- **Acceptance criteria:**
  - [ ] Valid response returns `{ valid: true, red_flags: [] }`
  - [ ] Invalid response returns `{ valid: false, red_flags: [TokenLengthExceeded, ...] }`
  - [ ] Multiple red-flags returned if multiple violations
- **Verification:** Run MCP server, invoke validate tool with test inputs

---

#### STORY-003-04: maker/calibrate Tool
- **Description:** As a Claude Code user, I want to invoke maker/calibrate to estimate per-step success rate (p), so that I can calculate optimal k_min for my task.
- **Context files:** None
- **Work to do:**
  - [ ] Create `src/mcp/tools/calibrate.rs`
  - [ ] Define `CalibrateRequest`: `{ samples: Vec<CalibrationSample> }` where `CalibrationSample = { prompt: String, ground_truth: String }`
  - [ ] Define `CalibrateResponse`: `{ p_estimate: f64, confidence_interval: (f64, f64), sample_count: usize, recommended_k: usize }`
  - [ ] Implement `calibrate_handler`:
    - Run each sample through LLM
    - Compare output to ground_truth (exact match for MVP)
    - Calculate p = correct / total
    - Calculate Wilson score confidence interval at 95%
    - Calculate recommended k using k_min formula with default t=0.95, s=1000
  - [ ] Register tool with rmcp server
- **Acceptance criteria:**
  - [ ] p_estimate = correct_samples / total_samples
  - [ ] Confidence interval calculated at 95% confidence level
  - [ ] Recommendation includes suggested k for default reliability target
- **Verification:** Run with known-p synthetic data, verify estimates

---

#### STORY-003-05: maker/configure Tool
- **Description:** As a Claude Code user, I want to invoke maker/configure to set default k, temperature, and token limits, so that I don't need to specify them on every vote call.
- **Context files:** None
- **Work to do:**
  - [ ] Create `src/mcp/tools/configure.rs`
  - [ ] Define `ConfigRequest`: `{ k_default: Option<usize>, temperature_diversity: Option<f64>, token_limit: Option<usize>, provider: Option<String> }`
  - [ ] Define `ConfigResponse`: `{ applied: bool, current_config: Config }`
  - [ ] Define `Config` struct with all configurable fields and defaults
  - [ ] Store config in `ServerState` as `Arc<RwLock<Config>>`
  - [ ] Use configured defaults in vote tool when not overridden
  - [ ] Register tool with rmcp server
- **Acceptance criteria:**
  - [ ] Configuration updated in server state
  - [ ] Subsequent vote calls use configured defaults
  - [ ] Response shows current configuration
- **Verification:** Set config, invoke vote without params, verify defaults used

---

#### STORY-003-06: Schema Validation for Security
- **Description:** As a MAKER MCP server, I want to validate all tool inputs and LLM outputs against schemas, so that I prevent prompt injection and malformed data.
- **Context files:** `RISK-REGISTER.md` (R-003)
- **Work to do:**
  - [ ] Define JSON schemas for all tool request/response types (via serde derive)
  - [ ] Ensure serde deserialization validates required fields (automatic)
  - [ ] Add `#[serde(deny_unknown_fields)]` to all request types
  - [ ] Validate LLM outputs via red-flag parsers before returning in VoteResponse
  - [ ] Log validation failures with security context (WARN level)
  - [ ] Document security model in README
- **Acceptance criteria:**
  - [ ] Invalid JSON tool inputs rejected by rmcp/serde layer
  - [ ] LLM outputs validated before inclusion in VoteResponse
  - [ ] Security audit: no prompt injection bypasses schema validation
- **Verification:** Attempt invalid inputs, verify rejection

---

### EPIC-008: Security & Guardrails
- **Owner:** Project Maintainer
- **Duration:** 2 days (Days 9-10, parallel with EPIC-003)
- **Priority:** P1 (High)
- **Dependencies:** EPIC-001 (red-flagging), EPIC-003 (MCP server)
- **Phase exit criteria:** Security audit passes, no injection vulnerabilities

#### STORY-008-01: Schema Enforcement for Agent Outputs
- **Description:** As a MAKER framework, I want to enforce strict JSON schemas on all agent outputs, so that malicious inputs cannot manipulate state transitions.
- **Context files:** `docs/technical-implementation-manual.txt` (Section 5.3)
- **Work to do:**
  - [ ] Define expected schema for agent outputs: `{ move: String, next_state: Value }`
  - [ ] Add red-flag rule: reject outputs missing required fields → `FormatViolation`
  - [ ] Add red-flag rule: reject outputs with unexpected fields in strict mode
  - [ ] Log schema violations at WARN level with sample hash (not full content)
  - [ ] Document schema requirements in README
- **Acceptance criteria:**
  - [ ] Output missing next_state triggers RedFlag::FormatViolation
  - [ ] Output with unexpected fields rejected in strict mode
  - [ ] Schema violations logged for security monitoring
- **Verification:** Unit tests for all schema violation cases

---

#### STORY-008-02: Prompt Injection Protection
- **Description:** As a MAKER MCP server, I want to sanitize and validate all user-provided prompts, so that I prevent injection attacks via tool inputs.
- **Context files:** `RISK-REGISTER.md` (R-003)
- **Work to do:**
  - [ ] Add prompt length validation: max 10,000 characters for MVP
  - [ ] Add suspicious pattern detection (optional for MVP):
    - "Ignore previous instructions"
    - "System:" prefix in user content
  - [ ] Log rejected prompts with hash only (not full content for privacy)
  - [ ] Document: MAKER operates on user-provided prompts (user responsibility)
  - [ ] Create `SECURITY.md` with vulnerability reporting process
- **Acceptance criteria:**
  - [ ] Prompt > 10K chars rejected with clear error
  - [ ] Suspicious patterns logged (optionally blocked)
  - [ ] SECURITY.md created with responsible disclosure process
- **Verification:** Attempt injection via vote tool, verify protection

---

#### STORY-008-03: Microagent Isolation Enforcement
- **Description:** As a MAKER framework, I want to enforce m=1 microagent constraint, so that a single malicious/erroneous agent cannot compromise entire task.
- **Context files:** `docs/technical-implementation-manual.txt` (Section 1.2)
- **Work to do:**
  - [ ] Enforce at type level: `Subtask` cannot contain history
  - [ ] Validate state transfer: next_state format checked before next agent
  - [ ] Ensure agent only receives current step state, no previous outputs
  - [ ] Document isolation guarantees in README
- **Acceptance criteria:**
  - [ ] Creating agent context with history triggers compile error or runtime panic
  - [ ] Agent only receives current step state
  - [ ] State corruption detected before next agent invocation
- **Verification:** Attempt to violate isolation, verify enforcement

---

### EPIC-005: Testing Infrastructure (Phase 2 portion)

#### STORY-005-03: MCP Protocol Compliance Tests
- **Description:** As a MAKER MCP server, I want integration tests validating MCP protocol compliance, so that I ensure interoperability with Claude Code.
- **Context files:** MCP specification
- **Work to do:**
  - [ ] Create `tests/mcp_integration.rs`
  - [ ] Write integration test for each MCP tool (vote, validate, calibrate, configure)
  - [ ] Test schema validation: invalid inputs rejected with clear errors
  - [ ] Test successful flows: valid inputs return expected outputs
  - [ ] Use mock LLM responses for deterministic testing
  - [ ] Test MCP protocol lifecycle: initialize, tool call, shutdown
- **Acceptance criteria:**
  - [ ] All 4 tools pass integration tests
  - [ ] Invalid JSON rejected with descriptive error messages
  - [ ] Tool outputs match declared JSON schema
  - [ ] Tests run against actual rmcp server instance
- **Verification:** Run `cargo test --test mcp_integration`

---

## Phase 3: Validation & Hardening (Days 11-14)

**Objective:** Demonstrate production readiness through benchmarks, security audit, and documentation.
**Exit Criteria:** Zero errors on 10-disk Hanoi, cost scaling validated, v0.1.0 released.

---

### EPIC-006: Demo & Benchmarks
- **Owner:** Project Maintainer
- **Duration:** 3 days (Days 11-13)
- **Priority:** P0 (Critical)
- **Dependencies:** EPIC-001, EPIC-002, EPIC-003 (complete integration)
- **Phase exit criteria:** Zero errors on 1,023 steps, Θ(s ln s) validated

#### STORY-006-01: Towers of Hanoi Task Decomposition
- **Description:** As a MAKER demo, I want to decompose Towers of Hanoi into microagent steps, so that I can demonstrate MAKER on a canonical long-horizon task.
- **Context files:** `docs/technical-implementation-manual.txt` (Towers of Hanoi example)
- **Work to do:**
  - [ ] Create `examples/hanoi/mod.rs`
  - [ ] Implement `HanoiState { rods: [Vec<u8>; 3] }` (rod 0=A, 1=B, 2=C)
  - [ ] Implement `HanoiState::is_legal_move(disk: u8, from: usize, to: usize) -> bool`
  - [ ] Implement `HanoiState::apply_move(&mut self, disk: u8, from: usize, to: usize)`
  - [ ] Implement `HanoiDecomposer` for `TaskDecomposer` trait:
    - Compute optimal solution (recursive algorithm)
    - Generate 2^n - 1 subtask prompts
    - Each prompt: "Given state {state}, what is the next optimal move?"
  - [ ] Precompute ground truth for validation
- **Acceptance criteria:**
  - [ ] 3-disk Hanoi generates 7 steps (2^3 - 1)
  - [ ] 10-disk Hanoi generates 1,023 steps (2^10 - 1)
  - [ ] Each step's next_state matches precomputed ground truth
  - [ ] Unit test: only legal moves accepted
- **Verification:** Run `cargo test --example hanoi`

---

#### STORY-006-02: End-to-End 10-Disk Hanoi Execution
- **Description:** As a MAKER user, I want to execute 10-disk Hanoi with voting, so that I can validate zero-error execution on 1,023 steps.
- **Context files:** Paper results for reference
- **Work to do:**
  - [ ] Create `examples/hanoi_demo.rs`
  - [ ] Implement `async fn run_hanoi(n_disks: u8, k_margin: usize, client: Arc<dyn LlmClient>) -> DemoResult`
  - [ ] Execute all 2^n - 1 steps with voting (k=3 or k=4)
  - [ ] Validate each step's winner matches ground truth
  - [ ] Collect metrics: total API calls, cost (USD), latency, red-flag rate, convergence rate
  - [ ] Log final summary: "10-disk Hanoi: 1,023 steps, 0 errors, $X cost, Y seconds"
  - [ ] Add CLI: `cargo run --example hanoi_demo -- --disks 10 --provider ollama`
- **Acceptance criteria:**
  - [ ] Zero errors (all 1,023 steps match ground truth)
  - [ ] k=3 or k=4 achieves 95%+ step-wise success
  - [ ] Total cost logged in USD
  - [ ] Execution completes (time depends on provider latency)
- **Verification:** Run demo with Ollama, verify zero errors

---

#### STORY-006-03: Cost Scaling Benchmark Suite
- **Description:** As a MAKER developer, I want to benchmark cost scaling across task lengths, so that I can empirically validate Θ(s ln s) complexity.
- **Context files:** `docs/technical-implementation-manual.txt` (Section 4)
- **Work to do:**
  - [ ] Create `benches/cost_scaling.rs`
  - [ ] Implement benchmark: run Hanoi for n ∈ {3, 5, 7, 10, 15} disks
  - [ ] Collect for each: s (steps), k (margin), total_samples, cost_usd, time_ms
  - [ ] Fit data to y = a * x * ln(x) + b using linear regression on (s * ln(s), total_samples)
  - [ ] Statistical test: R² > 0.95 for Θ(s ln s) fit
  - [ ] Export results to JSON: `benchmark_results.json`
  - [ ] Generate plot if matplotlib available (Python helper script)
- **Acceptance criteria:**
  - [ ] Benchmark completes for all n values
  - [ ] Fit shows Θ(s ln s) relationship (R² > 0.95)
  - [ ] Statistical test passes (cost ratio ≈ s_ratio * ln(s_ratio) within 20%)
  - [ ] Results exported to JSON
- **Verification:** Run `cargo bench cost_scaling`, check results JSON

---

#### STORY-006-04: Comparison to Naive Retry
- **Description:** As a MAKER evangelist, I want to compare MAKER cost to naive retry approaches, so that I can demonstrate cost efficiency.
- **Context files:** None
- **Work to do:**
  - [ ] Implement naive retry baseline: retry each step up to 5 times on error
  - [ ] Run naive retry on 3, 5, 7 disk Hanoi (10-disk infeasible with naive approach)
  - [ ] Compare: MAKER total_cost vs. naive retry total_cost
  - [ ] Calculate savings: (naive - maker) / naive * 100%
  - [ ] Add comparison table to README
- **Acceptance criteria:**
  - [ ] Naive retry cost > MAKER cost by 60%+ on multi-step tasks
  - [ ] Comparison documented in README
  - [ ] Results reproducible with benchmark script
- **Verification:** Run comparison benchmark, verify savings

---

### EPIC-007: Documentation
- **Owner:** Project Maintainer
- **Duration:** 2 days (Days 13-14)
- **Priority:** P1 (High)
- **Dependencies:** All epics (needs complete implementation)
- **Phase exit criteria:** README complete, API docs generated, examples work

#### STORY-007-01: README.md Update
- **Description:** As a prospective MAKER user, I want a comprehensive README, so that I understand the value proposition and can get started quickly.
- **Context files:** Current README.md (planning artifacts)
- **Work to do:**
  - [ ] Update README.md with implementation details:
    - Overview: problem statement, MAKER solution, zero-error claim
    - Quickstart: `cargo install maker-mcp`, configure Claude Code
    - Architecture diagram (Mermaid)
    - MCP tool reference: 4 tools with example requests/responses
    - Benchmarks: link to cost_scaling results
    - Citations: arxiv paper, SPRT references
  - [ ] Update badges: CI status, coverage, crates.io version
  - [ ] Ensure all code examples are tested (doc tests or examples)
- **Acceptance criteria:**
  - [ ] README under 500 lines, scannable structure
  - [ ] Quickstart enables first vote in <5 minutes
  - [ ] All code examples syntax-highlighted and tested
  - [ ] Citations include full URLs
- **Verification:** Follow quickstart on fresh system, verify it works

---

#### STORY-007-02: API Documentation (rustdoc)
- **Description:** As a MAKER library user, I want comprehensive API documentation, so that I can integrate MAKER into my own Rust projects.
- **Context files:** None
- **Work to do:**
  - [ ] Add doc comments to all public structs, traits, functions
  - [ ] Include examples in doc comments (doc tests)
  - [ ] Document error types and when they occur
  - [ ] Run `cargo doc --no-deps --open` to verify
  - [ ] Ensure doc tests pass: `cargo test --doc`
- **Acceptance criteria:**
  - [ ] All public APIs have doc comments with examples
  - [ ] Doc tests compile and pass
  - [ ] `cargo doc` generates complete documentation
  - [ ] Examples demonstrate common use cases
- **Verification:** Run `cargo doc --open`, review all public items

---

#### STORY-007-03: Example Integrations
- **Description:** As a MAKER user, I want example integration code, so that I can see MAKER in action and adapt for my use case.
- **Context files:** None
- **Work to do:**
  - [ ] Create `examples/hanoi.rs` - Towers of Hanoi demo
  - [ ] Create `examples/custom_task.rs` - Custom task integration template
  - [ ] Create `examples/README.md` explaining each example
  - [ ] Ensure all examples compile: `cargo build --examples`
  - [ ] Link examples from main README
- **Acceptance criteria:**
  - [ ] All examples compile and run successfully
  - [ ] Examples demonstrate different MAKER features
  - [ ] Examples include comments explaining key steps
  - [ ] Examples linked from main README
- **Verification:** Run each example, verify they complete

---

#### STORY-007-04: Security Documentation
- **Description:** As a MAKER operator, I want clear security documentation, so that I understand risks and mitigations.
- **Context files:** `RISK-REGISTER.md` (R-003)
- **Work to do:**
  - [ ] Add Security section to README:
    - MCP security risks (prompt injection, tool permissions)
    - MAKER mitigations: schema validation, red-flagging, isolation
    - User responsibility disclaimer
  - [ ] Create `SECURITY.md`:
    - Vulnerability reporting process
    - Security contact (maintainer email or GitHub security advisories)
    - Supported versions for security updates
- **Acceptance criteria:**
  - [ ] Security section in README covers MCP risks
  - [ ] SECURITY.md follows GitHub security advisory format
  - [ ] Responsible disclosure process documented
- **Verification:** Review security docs, verify completeness

---

#### STORY-007-05: CHANGELOG.md for v0.1.0
- **Description:** As a MAKER user, I want a CHANGELOG tracking version history, so that I can see what's new.
- **Context files:** Existing CHANGELOG.md (planning phase)
- **Work to do:**
  - [ ] Update CHANGELOG.md:
    - Move [Unreleased] content to [0.1.0] - 2026-MM-DD
    - Add implementation features under Added section
    - List known limitations under separate section
  - [ ] Follow Keep a Changelog 1.1.0 format
  - [ ] Link from README
- **Acceptance criteria:**
  - [ ] CHANGELOG follows Keep a Changelog format
  - [ ] All v0.1.0 features listed
  - [ ] ISO 8601 date format
- **Verification:** Review CHANGELOG structure

---

### EPIC-005: Testing Infrastructure (Phase 3 portion)

#### STORY-005-02: Monte Carlo Cost Validation
- **Description:** As a MAKER developer, I want Monte Carlo simulations to validate cost scaling, so that I can confirm Θ(s ln s) complexity statistically.
- **Context files:** None
- **Work to do:**
  - [ ] Create `tests/monte_carlo.rs`
  - [ ] Implement `simulate_task_cost(s: usize, p: f64, k: usize, trials: usize) -> CostStats`
  - [ ] Run simulations for s ∈ {100, 1_000, 10_000} (skip 100K+ for test speed)
  - [ ] Calculate mean, std dev, 95% confidence interval for cost
  - [ ] Compare to theoretical Θ(s ln s): verify within 20% tolerance
  - [ ] Compare MAKER vs. naive retry: MAKER should be 60%+ cheaper
- **Acceptance criteria:**
  - [ ] Cost ratio matches s_ratio * ln(s_ratio) within 20%
  - [ ] MAKER cost < naive retry cost by 60%+ for 1000-step tasks
  - [ ] Simulation completes in <5 minutes
  - [ ] Results logged with confidence intervals
- **Verification:** Run `cargo test monte_carlo` — assertions pass

---

#### STORY-005-04: CI/CD Pipeline with Coverage Enforcement
- **Description:** As a MAKER maintainer, I want automated testing and coverage enforcement on every commit, so that quality never regresses.
- **Context files:** None
- **Work to do:**
  - [ ] Create `.github/workflows/ci.yml`:
    ```yaml
    name: CI
    on: [push, pull_request]
    jobs:
      test:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - uses: dtolnay/rust-toolchain@stable
          - run: cargo test --all-features
          - run: cargo fmt --check
          - run: cargo clippy -- -D warnings
      coverage:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - uses: dtolnay/rust-toolchain@stable
          - run: cargo install cargo-llvm-cov
          - run: cargo llvm-cov --fail-under-lines 95
    ```
  - [ ] Configure branch protection: require CI pass before merge
  - [ ] Publish coverage report to GitHub Actions artifacts
- **Acceptance criteria:**
  - [ ] CI runs on every PR and push to main
  - [ ] Tests complete in <5 minutes
  - [ ] Coverage report viewable in GitHub Actions
  - [ ] PRs blocked if coverage < 95%
- **Verification:** Open PR, verify CI runs and blocks on low coverage

---

## Release Checklist (Day 14)

Before tagging v0.1.0:

- [ ] All acceptance criteria met across all epics
- [ ] Zero errors on 10-disk Hanoi (1,023 steps)
- [ ] 95% test coverage verified
- [ ] Cost scaling Θ(s ln s) validated
- [ ] All 4 MCP tools tested with Claude Code
- [ ] Security audit passed
- [ ] README complete with quickstart
- [ ] API docs generated (`cargo doc`)
- [ ] CHANGELOG updated for v0.1.0
- [ ] SECURITY.md created
- [ ] CI/CD pipeline green
- [ ] `cargo publish --dry-run` succeeds
- [ ] Git tag: `git tag -a v0.1.0 -m "MAKER Framework v0.1.0 MVP"`
- [ ] GitHub release created with release notes

---

**End of Execution Plan**
