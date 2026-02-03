# Project Context

## Problem Statement

Existing LLM-based agents fail catastrophically on long-horizon tasks. Empirical observation shows >50% failure rates on tasks requiring 100+ sequential steps, with reliability degrading exponentially as task length increases. For million-step tasks, even a 1% per-step error rate guarantees 100% task failure.

This project implements the **MAKER framework** (Maximal Agentic decomposition, first-to-ahead-by-K Error correction, and Red-flagging) in Rust, exposing its protocols as an **MCP (Model Context Protocol) server** and **standalone CLI** to augment existing coding assistants with mathematically-grounded error correction capabilities.

**Quantified Impact:**
- Current LLM agents: ~50% success on 100-step tasks, 0% on 1M-step tasks
- MAKER framework: 100% success demonstrated on 1,048,575-step tasks (Towers of Hanoi)
- Cost scaling: Θ(s ln s) vs exponential — enables economically feasible million-step execution

**Reference:** [MAKER: Turning Language Models Into Reliable Reasoners](https://arxiv.org/abs/2511.09030)

## Domain / Industry

**Open Source / Research — LLM Agent Reliability Engineering & MCP Tooling**

Sub-domain: Massively Decomposed Agentic Processes (MDAPs) for zero-error long-horizon task execution, targeting integration with AI coding assistants via MCP and standalone CLI.

**Target Problem Domains:**
- **Coding Tasks**: Multi-step code generation, refactoring, migration, automated fixes (AST-based decomposition)
- **ML Problem Space**: Model training pipelines, hyperparameter search, experiment orchestration
- **Data Analysis**: ETL pipelines, data validation, multi-stage analytical workflows

## Stakeholders

- **Open Source Community**: Primary developers and maintainers; community-driven contributions
- **AI/LLM Developers**: Building LLM-powered applications requiring reliable multi-step execution
- **Claude Code Users**: End users seeking enhanced reliability for complex coding tasks
- **Standalone CLI Users**: Developers wanting MAKER capabilities without MCP integration
- **Researchers**: Academic and industry researchers studying agent reliability, voting protocols, and error correction

## Constraints

- **Budget**: Open source / community-funded (no direct budget allocation)
- **Timeline**: Flexible/milestone-based — features released as they mature
- **Team size**: Community-driven; initial implementation by project maintainer(s)
- **Compliance**: MIT License for maximum adoption
- **Technology**:
  - **Language**: Rust (2021 edition)
  - **MCP SDK**: `rmcp` v0.13+ (Rust MCP implementation)
  - **LLM APIs**: Provider-agnostic design:
    - Ollama (local inference)
    - OpenAI GPT-4.1-mini / GPT-5.X-nano (cost-effective cloud)
    - Anthropic Haiku (cost-effective cloud)
  - **Async Runtime**: Tokio (via rmcp dependencies)
  - **Architecture**: Event-driven for observability and extensibility
- **Quality Gates**:
  - **Test Coverage**: 95% minimum mandatory for all code
  - **Structured Logging**: Configurable verbosity via `tracing`

## Available Data

- **Primary Source**: Arxiv paper [2511.09030](https://arxiv.org/abs/2511.09030) — full algorithmic specification
- **System Design Specification**: `docs/SystemDesignSpecification.txt` — architectural philosophy and implementation standards
- **Technical Manual**: `docs/project/technical-implementation-manual.txt` — implementation guidance
- **Existing Implementation**: v0.3.0 codebase with core voting, semantic matching, adaptive k, ensemble support, recursive decomposition, domain decomposers, and standalone CLI

## Desired Outcomes

### Completed (v0.1.0 + v0.2.0 + v0.3.0)

- [x] **Core MAKER Library**: k_min calculation, VoteRace, RedFlagValidator, vote_with_margin()
- [x] **MCP Server**: 5 tools (vote, validate, calibrate, configure, decompose) via rmcp
- [x] **Semantic Matching**: ExactMatcher, EmbeddingMatcher, CodeMatcher (tree-sitter)
- [x] **Adaptive K-Margin**: KEstimator with EMA-based p-hat estimation
- [x] **Multi-Model Ensemble**: RoundRobin, CostAware, ReliabilityWeighted strategies
- [x] **Benchmark Suite**: Coding, math/logic, data analysis, ensemble comparison
- [x] **768+ Tests**: Unit, integration, property-based (proptest)
- [x] **Recursive Decomposition Framework**: Full Section 7 implementation
  - [x] Decomposition Agents (LlmDecompositionAgent)
  - [x] Decomposition Discriminators (vote on proposals)
  - [x] Problem Solver Agents (LeafNodeExecutor)
  - [x] Solution Discriminators (aggregator)
  - [x] Recursive Orchestrator (full pipeline)
- [x] **Domain Decomposers**: Coding (AST-based), ML Pipeline, Data Analysis (ETL)
- [x] **Standalone CLI** (`maker-cli`): Full feature parity with MCP tools
- [x] **Multi-file Orchestration**: FileSystemState with locking and atomic commits
- [x] **Operational Tooling**: Health checks, metrics, configuration validation
- [x] **Validation Demos**: Hanoi (few-shot+CoT), Arithmetic (random error correction)

### Phase 4: Future Work — Production Validation & Real-World Testing

#### Remaining Production Hardening

- [ ] **Async Executor Integration**: Connect async LlmClient to synchronous voting executor
- [ ] **Real-world Claude Code Testing**: Comprehensive end-to-end integration tests
- [ ] **Performance Optimization**: Parallel sampling latency, memory efficiency
- [ ] **Decomposition Real-World Validation**: Test LLM-driven decomposition across diverse task types

#### Research Validation

- [ ] **Extended Hanoi Testing**: Test beyond 5 disks with live LLMs
- [ ] **Systematic vs Random Error Analysis**: More validation of MAKER boundaries
- [ ] **Cost Model Validation**: Verify Θ(s ln s) scaling with real providers

### Quality Targets

- **Test Coverage**: 95% minimum mandatory — no exceptions
- **Reliability**: Algorithms match paper's mathematical specifications
- **Extensibility**: Easy to add new decomposition strategies and domain matchers
- **Observability**: Complete audit trail via event-driven architecture
- **Security**: Prompt injection protection, schema enforcement, microagent isolation

## Additional Context

### Key Architectural Decisions

1. **Microagent Design (m=1)**: Each agent handles exactly one subtask to minimize context burden
2. **First-to-Ahead-by-K Voting**: SPRT-based optimal decision-making with Θ(s ln s) cost scaling
3. **Red-Flagging over Repairing**: Discard malformed responses to maintain error decorrelation
4. **State Transfer**: Agents output both `action` and `next_state`; system controls state flow
5. **Event-Driven Architecture**: All operations emit structured events for observability
6. **Separation of Insight and Execution**: Decomposition Agents plan; Solver Agents execute

### System Design Specification Alignment

All major sections of the System Design Specification are now implemented:

| Spec Section | Status | Implementation |
|--------------|--------|----------------|
| 1. Architectural Philosophy (MDAPs) | ✅ Complete | Core architecture established |
| 2. Micro-Role Protocol (m=1) | ✅ Complete | TaskOrchestrator enforces m=1 |
| 3. Context Shielding (φ function) | ✅ Complete | Minimal context prompts, domain-specific decomposers |
| 4. I/O Contracts (State Passing) | ✅ Complete | StrictAgentOutput, FileSystemState |
| 5. First-to-Ahead-by-k Voting | ✅ Complete | VoteRace, vote_with_margin, adaptive k |
| 6. Red-Flagging Protocols | ✅ Complete | RedFlagValidator with domain-specific extensions |
| 7. Recursive Architecture | ✅ Complete | LlmDecompositionAgent, RecursiveOrchestrator, domain decomposers |
| 8. Economic Analysis | ✅ Complete | Θ(s ln s) validated in benchmarks |

### Economic Model

The MAKER cost formula from the specification:

```
Expected Cost = Θ(p^{-m} × s × ln(s))
```

Where:
- `p` = per-step success probability
- `m` = steps per subtask (must be 1 for optimal cost)
- `s` = total task steps

With m=1 enforcement, cost scales as Θ(s ln s) — verified in benchmarks with R² > 0.99.

### Logging Strategy

Event-driven architecture enables comprehensive observability:

| Event Type | Data Captured | Use Case |
|------------|---------------|----------|
| `sample_request` | model, prompt_hash, temperature | Audit trail |
| `sample_response` | tokens_used, latency_ms, red_flags | Cost tracking |
| `vote_cast` | candidate_id, vote_count, margin | Convergence analysis |
| `vote_decided` | winner_id, total_votes, k_margin | Decision audit |
| `red_flag_triggered` | flag_type, token_count, format_error | Quality metrics |
| `step_complete` | step_id, state_hash, cumulative_cost | Progress tracking |
| `decomposition_proposed` | subtasks, composition_fn | Insight agent audit |
| `decomposition_accepted` | winning_strategy, vote_count | Discriminator decision |

**Log Levels:**
- `TRACE`: Individual API calls and parsing details
- `DEBUG`: Vote tallies, state transitions, decomposition proposals
- `INFO`: Step completions, decisions, red-flags
- `WARN`: Retries, degraded performance
- `ERROR`: API failures, unrecoverable states
