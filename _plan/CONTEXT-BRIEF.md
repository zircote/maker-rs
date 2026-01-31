# Context Brief: MAKER Framework

> Read this file first to understand the project before executing the plan.

## What This Project Is

MAKER (Maximal Agentic decomposition, first-to-ahead-by-K Error correction, Red-flagging) is a Rust implementation of a framework for zero-error long-horizon LLM task execution. It solves the fundamental problem that LLM agents fail >50% on tasks with 100+ steps and 100% on million-step tasks. MAKER exposes SPRT-based voting and red-flagging protocols as an MCP server that augments coding assistants like Claude Code with mathematically-grounded error correction.

## Key Decisions Already Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Language** | Rust 2021 edition | Zero-cost abstractions, fearless concurrency for parallel sampling |
| **MCP SDK** | rmcp v0.13.0+ | Official Rust MCP implementation |
| **Async Runtime** | Tokio | rmcp dependency, mature ecosystem |
| **LLM Providers** | Ollama, OpenAI, Anthropic | Local + cloud hybrid support |
| **Architecture** | Event-driven | Observability, extensibility, testability |
| **Voting Strategy** | First-to-ahead-by-k (SPRT) | Mathematically optimal stopping |
| **Error Handling** | Red-flag (discard) over repair | Maintains error decorrelation |
| **Decomposition** | m=1 (microagent) | Minimizes context, maximizes reliability |
| **Test Coverage** | 95% mandatory | CI/CD enforced |
| **License** | MIT | Maximum community adoption |

## Terminology

| Term | Definition |
|------|------------|
| **MAKER** | Maximal Agentic decomposition, first-to-ahead-by-K Error correction, Red-flagging |
| **SPRT** | Sequential Probability Ratio Test - optimal statistical stopping rule |
| **k-margin** | Required vote lead for declaring winner (k votes ahead) |
| **k_min** | Minimum k to achieve target reliability: ⌈ln(t^(-m/s)-1) / ln((1-p)/p)⌉ |
| **p** | Per-step success probability (model's base accuracy) |
| **t** | Target task reliability (e.g., 0.95 for 95%) |
| **s** | Total steps in task |
| **m** | Subtasks per agent (MAKER enforces m=1) |
| **Red-flagging** | Discarding malformed responses without repair |
| **Microagent** | Agent handling exactly one subtask (m=1) |
| **State transfer** | System uses next_state from agent, not model's interpretation |
| **MCP** | Model Context Protocol - standard for AI tool integration |
| **Θ(s ln s)** | MAKER's cost complexity (log-linear, not exponential) |

## File Map

| File | Contains | Read When |
|------|----------|-----------|
| `PROJECT-CONTEXT.md` | Original problem statement, stakeholders, constraints | Understanding scope |
| `PROJECT-PLAN.md` | Phase structure, ROI, success criteria | Planning work order |
| `JIRA-STRUCTURE.md` | Epic/story details, acceptance criteria | Detailed task reference |
| `_research/DOMAIN-RESEARCH.md` | Frameworks, terminology, citations | Making design decisions |
| `_research/DEPENDENCY-ANALYSIS.md` | Critical path, bottlenecks | Understanding blockers |
| `RACI-CHART.md` | Role assignments | Knowing who owns what |
| `RISK-REGISTER.md` | Risk mitigations | Encountering issues |
| `SEVERITY-CLASSIFICATION.md` | Priority framework | Triaging problems |
| `BEST-PRACTICES.md` | Industry standards, Rust patterns | Making design decisions |
| `SUCCESS-METRICS.md` | KPIs and closure criteria | Verifying completion |
| `docs/technical-implementation-manual.txt` | Paper-based implementation guidance | Algorithm reference |

## Constraints and Guardrails

- **Timeline**: 14 days (2 weeks) to MVP
- **Test Coverage**: 95% minimum mandatory - CI blocks PRs below this
- **Budget**: Open source / community-funded
- **Technology**: Rust 2021, Tokio async, rmcp SDK
- **LLM APIs**: Must support Ollama (local), OpenAI, Anthropic
- **Compliance**: MIT License
- **Quality**: All algorithms must match arxiv paper 2511.09030 specifications

## Critical Path

```
EPIC-001 (Days 1-4) → EPIC-002 (Days 6-9) → EPIC-003 (Days 8-10) → EPIC-006 (Days 11-13) → Release (Day 14)
```

Any delay on critical path items directly extends the timeline.

## Top 5 Risks to Watch

| Risk | Mitigation |
|------|------------|
| **R-001: Mathematical errors in SPRT/k_min** | Property-based tests, Monte Carlo validation, paper reference |
| **R-002: API reliability at scale** | Exponential backoff, Ollama fallback, rate limit tracking |
| **R-003: MCP security (prompt injection)** | Schema validation, red-flag guardrails, security audit |
| **R-005: Test coverage below 95%** | CI/CD enforcement, property-based tests from Day 1 |
| **R-006: Cost scaling deviation** | Empirical benchmarks, profiling, optimization |

## How to Start

1. Read `_plan/EXECUTION-PLAN.md` for the full work breakdown
2. Or read `_plan/RUNSHEET.md` for a sprint-by-sprint checklist
3. Begin with Phase 1, Sprint 1 items (EPIC-001 first)
4. Use `_plan/TASK-MANIFEST.json` for programmatic task tracking

## Key Commands

```bash
# Build
cargo build

# Test with coverage
cargo llvm-cov --html

# Run all tests
cargo test

# Check formatting
cargo fmt --check && cargo clippy

# Run MCP server
cargo run --bin maker-mcp

# Run Towers of Hanoi demo
cargo run --example hanoi -- --disks 10
```

---

**Project Status:** Planning Complete - Ready for Implementation
**Start Date:** 2026-01-30
**Target Release:** v0.1.0 MVP by Day 14
