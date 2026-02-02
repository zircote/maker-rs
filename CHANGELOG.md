# Changelog

All notable changes to this project are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

#### Vote Tool Field Remediation (`src/mcp/tools/vote.rs`)
- **`matcher` field wired**: Creates matcher from request (`"exact"`, `"embedding"`, `"code"`), passes to `VoteConfig`, populates `matcher_type` in response
- **`adaptive` field wired**: Branches to `vote_with_margin_adaptive()` when `adaptive=true`, populates `p_hat` in response
- **`ensemble` field placeholder**: Returns clear error when `ensemble=true` explaining server-level config is required
- **`create_matcher()` helper**: Factory function supporting `"exact"`, `"embedding"`, and `"code"` (with `code-matcher` feature flag)
- **`VoteToolError::InvalidMatcher`**: New error variant with `provided` and `valid` fields for invalid matcher types
- **11 unit tests**: Covering matcher creation, error display, invalid matcher handling, ensemble error, and adaptive p_hat

### Fixed

#### Async Runtime Fix (`src/mcp/server.rs`)
- **`vote` handler spawn_blocking**: Wrapped blocking `execute_vote` call in `tokio::task::spawn_blocking` to prevent tokio runtime nesting panic when called from async context

## [0.3.0] - 2026-01-31

### Added

#### Recursive Decomposition Framework (`src/core/decomposition/`)

##### Core Types (`mod.rs`)
- **`DecompositionProposal`**: Represents a proposed task breakdown with subtasks, composition function, and metadata
- **`Subtask`**: Individual decomposed task with `task_id`, `parent_id`, `m_value`, `description`, and `context`
- **`CompositionFunction` enum**: `Sequential`, `Parallel`, `Conditional`, `Custom` for result aggregation patterns
- **`DecompositionAgent` trait**: Object-safe trait with `propose_decomposition()` for domain-specific decomposers
- **`DecompositionProposalEvent`**: New `MakerEvent` variant for decomposition observability
- **m=1 enforcement**: Property-tested validation ensuring leaf nodes always have `m_value == 1`

##### Decomposition Discriminator (`discriminator.rs`)
- **`DecompositionDiscriminator`**: Votes on competing decomposition proposals using `VoteRace`
- **`DecompositionCandidateMatcher`**: `CandidateMatcher` implementation for structural comparison of proposals
- **`vote_on_decomposition()`**: Entry point for decomposition voting with depth-based k scaling
- **`DecompositionAccepted` / `DecompositionRejected` events**: Observability for decomposition decisions

##### Problem Solver Interface (`solver.rs`)
- **`LeafNodeExecutor`**: Executes m=1 leaf subtasks using `vote_with_margin()`
- **`SubtaskResult`**: Result wrapper with task ID, output, and execution metadata
- **State passing**: Parent-to-child context propagation through decomposition tree
- **Partial failure handling**: Configurable retry and error recovery for subtask execution

##### Solution Aggregation (`aggregator.rs`)
- **`SolutionDiscriminator`**: Votes on competing subtask results before composition
- **`compose_results()`**: Aggregates results according to `CompositionFunction` semantics
- **Recursive composition**: Supports nested decomposition trees to arbitrary depth
- **Schema validation**: Validates composed results against expected output schemas
- **`SolutionComposed` event**: Emitted when results are successfully aggregated

##### Recursive Orchestrator (`orchestrator.rs`)
- **`RecursiveOrchestrator`**: Full pipeline coordinator from task to final result
- **`execute(task)`**: Main entry point implementing decomposition → voting → execution → aggregation loop
- **Depth limit**: Configurable maximum recursion depth (default 10) prevents infinite decomposition
- **Cycle detection**: Detects and prevents circular decomposition dependencies
- **Timeout**: Configurable execution timeout (default 60s) with clean cancellation
- **Manual decomposition injection**: Override automatic decomposition with user-provided breakdown
- **Cancellation support**: Graceful shutdown via `CancellationToken`

#### Domain Decomposers (`src/core/decomposition/domains/`)

##### Coding Decomposer (`coding.rs`, behind `code-matcher` feature)
- **`CodingDecomposer`**: Tree-sitter-based decomposer respecting syntactic boundaries
- **`CodeDecompositionStrategy` enum**: `FunctionLevel`, `BlockLevel`, `LineLevel`, `Auto` for granularity control
- **`SyntaxValidationResult`**: Red-flag detection for syntax errors in generated code
- **Multi-language support**: Rust, Python, JavaScript via `CodeLanguage` enum
- **17 unit tests** for decomposition strategies and m=1 enforcement

##### ML Pipeline Decomposer (`ml.rs`)
- **`MLPipelineDecomposer`**: Domain-specific decomposer for machine learning workflows
- **`PipelineStage` enum**: `DataPrep`, `FeatureEngineering`, `ModelConfig`, `Training`, `Evaluation`, `Validation`, `HyperparameterSearch`, `Deployment`
- **`HyperparameterSearchConfig`**: Grid, Random, and Bayesian search strategies with `Parallel` composition
- **`MetricValidation`**: Red-flags for NaN, infinity, and out-of-range metric values
- **Cross-validation support**: Automatic CV fold creation with configurable splits
- **22 unit tests** including metric validation and hyperparameter search

##### Data Analysis Decomposer (`data.rs`)
- **`DataAnalysisDecomposer`**: ETL-pattern decomposer for data processing tasks
- **`EtlStage` enum**: `Extract`, `SchemaInference`, `Clean`, `Transform`, `Aggregate`, `Enrich`, `Validate`, `Load`, `QualityCheck`
- **`DataType` enum**: Schema inference type system (String, Integer, Float, Boolean, Timestamp, Json, Binary, Null)
- **`CoercionValidation`**: Risk assessment for type conversions (Safe, PrecisionLoss, Truncation, PartialFailure, DataLoss)
- **`NullHandling` strategies**: Explicit null value handling configuration
- **27 unit tests** for ETL patterns and coercion validation

#### Multi-file Orchestration (`src/core/decomposition/filesystem.rs`)
- **`FileSystemState`**: Thread-safe state container for multi-file operations
- **`FileContent`**: Tracks original and current content with diff computation
- **`FileLock` enum**: `Unlocked`, `ReadLocked(count)`, `WriteLocked(holder_id)` for concurrency control
- **`FileOperation` enum**: `Create`, `Modify`, `Delete`, `Rename` with rollback support
- **Cross-file dependency tracking**: Ensures operations execute in dependency order
- **Cycle detection**: Prevents circular file dependencies
- **`FileCommit`**: Atomic multi-file commit with all-or-nothing semantics
- **30 unit tests** for locking, dependencies, and atomic commits

#### Production CLI (`src/bin/maker-cli.rs`)
- **`maker-cli` binary**: Standalone command-line interface for MAKER operations
- **`vote` subcommand**: Execute k-margin voting with provider selection
- **`validate` subcommand**: Run red-flag validation on outputs
- **`calibrate` subcommand**: Estimate per-step success probability
- **`config` subcommand**: View and modify runtime configuration
- **`decompose` subcommand**: Execute recursive decomposition pipeline
- **JSON/text output modes**: `--format json|text` for programmatic integration
- **Shell completion**: `--generate-completions bash|zsh|fish`
- **Standard exit codes**: 0 success, 1 error, 2 validation failure

#### Async Executor (`src/core/async_executor.rs`)
- **`AsyncVotingExecutor`**: Tokio-native async voting executor
- **`vote_with_margin_async()`**: Async version of voting with concurrent sampling
- **Cancellation handling**: Respects `CancellationToken` for graceful shutdown
- **Connection pooling**: Leverages reqwest connection pool for parallel requests
- **Property tests**: Verify parity with synchronous executor results

#### Operational Tooling (`src/mcp/health.rs`)
- **`prometheus` feature flag**: Optional Prometheus metrics export
- **`HealthStatus`**: Health check response with status, version, and uptime
- **`/health` MCP resource**: Health check endpoint for monitoring integration
- **Prometheus metrics**: Counters and histograms for votes, samples, latency (behind feature flag)
- **`validate_config()` function**: Fail-fast configuration validation
- **Graceful shutdown**: Tokio signal handling for SIGTERM/SIGINT

#### Documentation
- **`docs/CLAUDE-CODE-SETUP.md`**: Integration guide for Claude Code/Claude Desktop
- **MCP integration tests**: 35 tests covering all tools via stdio transport

#### Testing
- **842+ total tests** across all modules
- **17 coding decomposer tests** (function/block/line strategies, m=1 enforcement)
- **22 ML pipeline tests** (stages, hyperparameter search, metric validation)
- **27 data analysis tests** (ETL patterns, schema inference, coercion)
- **30 filesystem tests** (locking, dependencies, atomic commits)
- **Property tests**: m=1 leaf enforcement, execution order validation

### Changed
- `MakerEvent` enum extended with decomposition-related variants
- `LoggingObserver` and `MetricsObserver` handle new decomposition events
- `src/core/mod.rs` exports new `decomposition` module
- Test count increased from 456 to 842+

### Fixed
- N/A

### Known Limitations
- Real LLM end-to-end testing pending (mock tests complete)
- `prometheus` feature metrics are basic counters; histograms in future release

## [0.2.0] - 2026-01-31

### Added

#### Multi-Model Ensemble (`src/llm/ensemble.rs`, `src/llm/sampler.rs`)
- **`EnsembleConfig`**: Configuration for 2-5 model voting ensembles with pluggable strategies
- **`ModelSlot`**: Per-model wrapper holding `Arc<dyn LlmClient>`, sampling weight, and `CostTier`
- **`EnsembleStrategy` enum**: `RoundRobin` (even distribution), `CostAware` (cheap→medium→expensive phases), `ReliabilityWeighted` (Bresenham-style deficit tracking for proportional allocation)
- **`CostTier` enum**: `Cheap`, `Medium`, `Expensive` for cost-aware routing phases
- **`EnsembleMetrics`**: Tracks `models_used`, `samples_per_model`, `escalations`, `cost_per_model`
- **`TaggedSample`**: Sample result tagged with source model name for attribution
- **`EnsembleSampleResult`**: Collection of tagged samples with `by_model()` grouping
- **`collect_ensemble_samples()`**: Async ensemble-aware sampling using `EnsembleConfig` for model selection
- **`EnsembleConfigRequest`** / **`ModelSlotRequest`**: Serde/JsonSchema request types for MCP configuration
- **`EscalationTriggered` event**: New `MakerEvent` variant emitted on cost-tier escalation during CostAware routing

#### Ensemble MCP Extension
- **`ConfigRequest.ensemble`**: Configure ensemble via `maker/configure` tool
- **`VoteRequest.ensemble`**: Per-call ensemble enable/disable override
- **`VoteResponse.ensemble_metrics`**: Per-model sample counts, cost breakdown, and escalation count in vote results
- **`Config.ensemble`**: Ensemble configuration exposed in current config response

#### Domain Benchmarks
- **`benches/ensemble_comparison.rs`**: Monte Carlo comparison of single-model vs. round-robin vs. cost-aware ensemble (1,000 trials at s=100, s=1,000)
- **`benches/coding_tasks.rs`**: 10 coding task benchmarks across trivial, moderate, and complex difficulty (FizzBuzz to regex engine)
- **`benches/math_logic.rs`**: 10 math/logic benchmarks including arithmetic chains, symbolic differentiation, logic puzzles, and Hanoi variants
- **`benches/data_analysis.rs`**: 10 data analysis benchmarks covering CSV processing, statistics, SQL generation, and data cleaning
- **`BENCHMARKS.md`**: Aggregated benchmark results with acceptance criteria summary
- Weekly CI benchmark workflow (`.github/workflows/benchmarks.yml`)

#### Examples
- **`examples/coding_task.rs`**: Semantic matching on a multi-step coding task
- **`examples/ensemble_demo.rs`**: Multi-model ensemble voting cost comparison

#### Adaptive K-Margin (`src/core/adaptive.rs`)
- **`KEstimator`**: EMA-based p-hat estimation (α=0.1) with live `recommended_k()` using observed success rate
- **`vote_with_margin_adaptive()`**: Wraps `vote_with_margin` with dynamic k adjustment based on running accuracy estimates
- **Configurable bounds**: k floor (default 2) and ceiling (default 10) prevent runaway margins
- **MCP extension**: `maker/configure` accepts `adaptive_k`, `ema_alpha`, `k_bounds`; `maker/vote` response includes `k_used` and `p_hat`

#### Semantic Matching (`src/core/matcher.rs`, `src/core/matchers/`)
- **`CandidateMatcher` trait**: Pluggable trait for response comparison with `canonicalize()`, `are_equivalent()`, and `similarity_score()` methods
- **`ExactMatcher`**: Default whitespace-normalized string equality (backward compatible with v0.1.0 behavior)
- **`EmbeddingMatcher`**: Cosine similarity of embedding vectors with configurable threshold (default 0.92) and internal cache
- **`EmbeddingClient` trait**: Synchronous interface for embedding providers
- **`OllamaEmbeddingClient`**: Ollama `/api/embeddings` endpoint integration
- **`OpenAiEmbeddingClient`**: OpenAI `text-embedding-3-small` integration
- **`CodeMatcher`** (behind `code-matcher` feature): AST-based code comparison using tree-sitter with alpha-renaming, comment stripping, and LCS-based token similarity; supports Rust, Python, JavaScript
- **`MatcherConfig` enum**: Serde-tagged configuration for Exact, Embedding, and Code matchers via MCP `maker/configure` tool
- **`VoteRequest.matcher`**: Per-call matcher override
- **`VoteResponse.matcher_type`** and **`candidate_groups`**: New response fields showing matcher used and number of distinct candidate groups

#### Testing
- 31 unit tests for ensemble configuration (strategy selection, validation, metrics)
- 7 ensemble sampler tests (tagged samples, multi-model, iterators)
- 21 property-based tests for adaptive k behavior (bounds, convergence, recovery)
- 25 semantic matching tests covering code corpus accuracy (>95%), NL pairs, false positive rate, reflexivity, and symmetry
- Total test count: 456 unit + 35 integration (without `code-matcher` feature)

### Changed
- `VoteResult` extended with `cost_by_model` and `ensemble_metrics` fields for per-model cost tracking
- `ServerConfig` extended with `ensemble` field (`Option<EnsembleConfigRequest>`)
- `VoteRace` now accepts `Arc<dyn CandidateMatcher>` for pluggable candidate grouping
- `vote_with_margin()` uses matcher for candidate grouping instead of raw string equality
- `reqwest` dependency now includes `blocking` feature for synchronous embedding HTTP clients
- `ServerConfig` extended with `adaptive_k`, `ema_alpha`, `k_bounds`, and `matcher` fields
- `LoggingObserver` and `MetricsObserver` handle `EscalationTriggered` events

### Fixed
- N/A

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
- ~~Semantic matching for non-deterministic tasks (e.g., coding) not yet implemented~~ *(resolved in [0.2.0])*
- ~~Multi-model ensemble voting not yet implemented~~ *(resolved in [0.2.0])*
- MCP tool names use `/` separator (e.g., `maker/vote`) which triggers rmcp naming warnings

## [0.0.0] - 2026-01-30

### Added
- Complete project planning phase with 17 artifacts
- PROJECT-CONTEXT.md, PROJECT-PLAN.md, JIRA-STRUCTURE.md
- Risk register, RACI chart, severity classification
- README with architecture diagrams and MCP tool reference

---

[Unreleased]: https://github.com/zircote/maker-rs/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/zircote/maker-rs/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/zircote/maker-rs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/zircote/maker-rs/compare/v0.0.0...v0.1.0
[0.0.0]: https://github.com/zircote/maker-rs/releases/tag/v0.0.0
