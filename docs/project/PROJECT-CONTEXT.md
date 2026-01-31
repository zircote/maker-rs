# Project Context

## Problem Statement

Existing LLM-based agents fail catastrophically on long-horizon tasks. Empirical observation shows >50% failure rates on tasks requiring 100+ sequential steps, with reliability degrading exponentially as task length increases. For million-step tasks, even a 1% per-step error rate guarantees 100% task failure.

This project implements the **MAKER framework** (Maximal Agentic decomposition, first-to-ahead-by-K Error correction, and Red-flagging) in Rust, exposing its protocols as an **MCP (Model Context Protocol) server** to augment existing coding assistants like Claude Code with mathematically-grounded error correction capabilities.

**Reference:** [MAKER: Turning Language Models Into Reliable Reasoners](https://arxiv.org/abs/2511.09030)

## Domain / Industry

**Open Source / Research — LLM Agent Reliability & MCP Tooling**

Sub-domain: Distributed AI systems for error-corrected long-horizon task execution, specifically targeting integration with AI coding assistants via the Model Context Protocol.

**Target Problem Domains:**
- **Coding Tasks**: Multi-step code generation, refactoring, migration, and automated fixes
- **ML Problem Space**: Model training pipelines, hyperparameter search, experiment orchestration
- **Data Analysis**: ETL pipelines, data validation, multi-stage analytical workflows

## Stakeholders

- **Open Source Community**: Primary developers and maintainers; community-driven contributions
- **AI/LLM Developers**: Building LLM-powered applications requiring reliable multi-step execution
- **Claude Code Users**: End users seeking enhanced reliability for complex coding tasks
- **Researchers**: Academic and industry researchers studying agent reliability, voting protocols, and error correction in language models

## Constraints

- **Budget**: Open source / community-funded (no direct budget allocation)
- **Timeline**: 1-2 weeks for initial MVP
- **Team size**: Community-driven; initial implementation by project maintainer(s)
- **Compliance**: MIT License for maximum adoption
- **Technology**:
  - **Language**: Rust (2021 edition)
  - **MCP SDK**: `rmcp` (Rust MCP implementation)
  - **LLM APIs**: Provider-agnostic design with primary targets:
    - Ollama (local inference)
    - LM Studio (local inference)
    - OpenAI GPT-5.X-nano (cost-effective cloud)
    - Anthropic Haiku (cost-effective cloud)
  - **Async Runtime**: Tokio (via rmcp dependencies)
  - **Architecture**: Event-driven for observability and extensibility
- **Quality Gates**:
  - **Test Coverage**: 95% minimum mandatory for all code
  - **Logging**: Structured logging strategy with configurable verbosity

## Available Data

- **Primary Source**: Arxiv paper [2511.09030](https://arxiv.org/abs/2511.09030) — full algorithmic specification
- **Technical Manual**: `docs/technical-implementation-manual.txt` — implementation guidance for MAKER framework
- **No existing reference implementation** — Rust implementation built from paper specifications

## Desired Outcomes

### MVP (1-2 weeks) — ✅ Complete (v0.1.0)
- [x] **Core Rust library** implementing MAKER protocols:
  - Maximal Agentic Decomposition (MAD) task orchestration ✅
  - First-to-Ahead-by-K voting with configurable margins ✅
  - Red-flagging parsers (token length, format validation) ✅
  - k_min calculation based on target reliability ✅
- [x] **MCP server** exposing MAKER tools for Claude Code integration:
  - `maker/vote` — parallel sampling with K-margin voting ✅
  - `maker/validate` — red-flag validation for responses ✅
  - `maker/calibrate` — estimate per-step success rate (p) ✅
  - `maker/configure` — set k, temperature, token limits ✅
- [x] **End-to-end demo** — Towers of Hanoi (3-20 disks) in `examples/hanoi/`
- [x] **Functional correctness** — core algorithms validated via 456+ tests
- [x] **Production hardening** — exponential backoff, retry strategies, circuit breakers

### Quality Targets
- **Test Coverage**: 95% minimum mandatory — no exceptions
- **Reliability**: Algorithms match paper's mathematical specifications
- **Extensibility**: Easy to add new voting strategies and red-flag rules
- **Observability**: Event-driven architecture with structured logging and token economics tracking
- **Security**: Prompt injection protections and guardrails

## Additional Context

### Key Architectural Decisions

1. **Microagent Design (m=1)**: Each agent handles exactly one subtask to minimize context burden and maximize per-step reliability
2. **First-to-Ahead-by-K Voting**: Sequential probability ratio test providing optimal decision-making with logarithmic cost scaling (Θ(s ln s))
3. **Red-Flagging over Repairing**: Discard malformed responses rather than attempting to repair them — maintains error decorrelation
4. **State Transfer**: Agents output both `move` and `next_state`; system (not model) uses `next_state` to seed subsequent agents
5. **Event-Driven Architecture**: All operations emit structured events for observability, extensibility, and debugging

### Event-Driven Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Sampler   │────▶│   Events    │────▶│  Observers  │
│  (LLM API)  │     │   (Channel) │     │  (Logging,  │
└─────────────┘     └─────────────┘     │   Metrics,  │
                           │            │   Tracing)  │
┌─────────────┐            │            └─────────────┘
│   Voter     │────────────┤
│ (K-Margin)  │            │            ┌─────────────┐
└─────────────┘            └───────────▶│  Exporters  │
                                        │  (OTLP,     │
┌─────────────┐                         │   JSON,     │
│  Red-Flag   │─────────────────────────│   Metrics)  │
│  Validator  │                         └─────────────┘
└─────────────┘
```

**Benefits:**
- **Decoupled**: Components communicate via events, not direct calls
- **Extensible**: Add new observers without modifying core logic
- **Testable**: Events can be captured and asserted in tests
- **Debuggable**: Complete audit trail of every decision

### Performance Considerations

- **Parallel Sampling**: Critical for latency — samples collected concurrently
- **Temperature Strategy**: T=0 for first sample, T=0.1 for subsequent votes (diversity without chaos)
- **Token Economics**: Track cost per step, votes per decision, red-flag rates

### Security Requirements

- **Prompt Injection Protection**: Validate agent outputs before state transitions
- **Guardrails**: Enforce format schemas, reject unexpected output patterns
- **Isolation**: Microagent context isolation prevents cross-contamination

### Research Extensions — ✅ Implemented in v0.2.0

These features were architected from the start and are now fully implemented:

- **Semantic Matching** ✅: `CandidateMatcher` trait with `ExactMatcher`, `EmbeddingMatcher`, `CodeMatcher` (tree-sitter AST)
- **Adaptive K** ✅: `KEstimator` with EMA-based p-hat estimation and configurable bounds (`k_bounds`)
- **Multi-Model Ensemble** ✅: `EnsembleConfig` with `RoundRobin`, `CostAware`, `ReliabilityWeighted` strategies
- **Benchmark Suite** ✅: Domain benchmarks in `benches/` — coding tasks, math/logic, data analysis, ensemble comparison

### Logging Strategy

Event-driven architecture enables comprehensive observability:

| Event Type | Data Captured | Use Case |
|------------|---------------|----------|
| `sample_request` | model, prompt_hash, temperature, timestamp | Audit trail |
| `sample_response` | tokens_used, latency_ms, red_flags | Cost tracking |
| `vote_cast` | candidate_id, vote_count, margin | Convergence analysis |
| `vote_decided` | winner_id, total_votes, k_margin | Decision audit |
| `red_flag_triggered` | flag_type, token_count, format_error | Quality metrics |
| `step_complete` | step_id, state_hash, cumulative_cost | Progress tracking |

**Log Levels:**
- `TRACE`: Individual API calls and parsing details
- `DEBUG`: Vote tallies, state transitions
- `INFO`: Step completions, decisions, red-flags
- `WARN`: Retries, degraded performance
- `ERROR`: API failures, unrecoverable states
