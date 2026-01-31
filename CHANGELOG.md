# Changelog

All notable changes to this project are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

*No unreleased changes.*

## [0.1.0] - 2026-01-30

### Added

#### Core Algorithms (`src/core/`)
- **k_min calculation**: `calculate_kmin(p, t, s, m)` computes minimum k-margin for target reliability using SPRT/Gambler's Ruin theory
- **Vote race tracking**: `VoteRace` with thread-safe `Arc<Mutex<HashMap>>`, first-to-ahead-by-k winner detection, event callbacks
- **Red-flag validation**: `RedFlagValidator` with token length, JSON schema, and format violation checks (discard-don't-repair)
- **Voting executor**: `vote_with_margin()` orchestrates the sample-validate-vote loop with configurable timeouts and max samples
- **Microagent orchestration**: `TaskOrchestrator` and `TaskDecomposer` traits enforcing m=1 constraint with state hash validation

#### LLM Providers (`src/llm/`)
- **Async `LlmClient` trait**: Object-safe async interface using `Pin<Box<dyn Future>>`
- **Ollama client**: Local inference via `/api/generate` endpoint
- **OpenAI client**: Chat completions API with `OPENAI_API_KEY` support
- **Anthropic client**: Messages API with `ANTHROPIC_API_KEY` support
- **Exponential backoff retry**: Configurable retry with jitter, Retry-After header respect
- **Parallel sampling**: Temperature-diverse sampling (T=0 first, T=0.1 rest) via `tokio::task::JoinSet`

#### MCP Server (`src/mcp/`)
- **rmcp v0.13 server**: Stdio transport with `#[tool_router]` and `#[tool_handler]` macros
- **`maker/vote` tool**: K-margin voting with configurable provider, max samples, temperature
- **`maker/validate` tool**: Red-flag checking with token limits and JSON schema validation
- **`maker/calibrate` tool**: Per-step success rate estimation with Wilson score confidence intervals
- **`maker/configure` tool**: Runtime configuration stored in `Arc<RwLock<ServerConfig>>`
- **Schema validation**: `#[serde(deny_unknown_fields)]` on all request types

#### Event System (`src/events/`)
- **`MakerEvent` enum**: Serde-tagged events (SampleRequested, SampleCompleted, RedFlagTriggered, VoteCast, VoteDecided, StepCompleted)
- **`EventBus`**: Tokio broadcast channel for fan-out to observers
- **`LoggingObserver`**: Structured logging via `tracing` (VoteDecided=INFO, RedFlagTriggered=WARN)
- **`MetricsObserver`**: Counters and histograms for votes, red-flags, latency, cost

#### Security & Guardrails
- Schema enforcement for agent outputs (`StrictAgentOutput`)
- Prompt injection protection (10K character limit)
- Microagent isolation (m=1, no history in subtask context)
- `SECURITY.md` with vulnerability reporting process

#### Testing
- 385+ tests across unit, integration, and property-based suites
- Property-based tests via `proptest` (15 properties, 1000+ iterations each)
- MCP integration tests (35 tests covering all 4 tools)
- Monte Carlo cost validation (Θ(s ln s) scaling verified)
- HTTP mock tests via `wiremock` for all LLM providers (OpenAI, Anthropic, Ollama)
- Example tests (21 tests in hanoi example)

#### Examples & Benchmarks
- `examples/hanoi/`: Tower of Hanoi with task decomposition (3 to 20 disks)
- `examples/hanoi_demo.rs`: End-to-end voting demo achieving zero errors on 1,023 steps
- `examples/custom_task.rs`: Template for custom task integration
- `benches/cost_scaling.rs`: Cost scaling benchmark with JSON export

#### CI/CD
- GitHub Actions workflow: test, lint, coverage, documentation
- Coverage threshold enforcement (90% line coverage, excluding binaries)

### Known Limitations
- Parallel sampling via async LLM clients not yet integrated into the synchronous voting executor
- Semantic matching for non-deterministic tasks (e.g., coding) not yet implemented
- MCP tool names use `/` separator (e.g., `maker/vote`) which triggers rmcp naming warnings

## [0.0.0] - 2026-01-30

### Added
- Complete project planning phase with 17 artifacts
- PROJECT-CONTEXT.md, PROJECT-PLAN.md, JIRA-STRUCTURE.md
- Risk register, RACI chart, severity classification
- README with architecture diagrams and MCP tool reference

---

[Unreleased]: https://github.com/zircote/maker-rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/zircote/maker-rs/compare/v0.0.0...v0.1.0
[0.0.0]: https://github.com/zircote/maker-rs/releases/tag/v0.0.0
