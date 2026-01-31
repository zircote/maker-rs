# Gantt Chart: MAKER Framework Implementation

**Project:** Rust Implementation of MAKER Framework with MCP Integration
**Timeline:** 14 days (January 30 - February 12, 2026)
**Sprints:** 3 sprints aligned with project phases
**Total Epics:** 10 (8 MVP + 2 Post-MVP)

---

## Project Timeline Overview

**Start Date:** January 30, 2026 (Thursday)
**End Date:** February 12, 2026 (Wednesday)
**Duration:** 14 calendar days (10 working days assuming 5-day work week)
**Phases:** 3 phases (Core Algorithms → MCP Integration → Validation & Hardening)

---

## Visual Timeline

![Gantt Chart](.github/gantt-chart.svg)

---

## Critical Path

The critical path represents the minimum time required to complete the project. Any delay on these epics extends the overall timeline.

```mermaid
graph LR
    Start([Project Start<br/>Jan 30]) --> E001[EPIC-001<br/>Core MAKER Library<br/>Days 1-4]

    E001 --> E002[EPIC-002<br/>LLM Provider Abstraction<br/>Days 6-9]

    E002 --> E003[EPIC-003<br/>MCP Server<br/>Days 8-10]

    E003 --> E006[EPIC-006<br/>Demo & Benchmarks<br/>Days 11-13]

    E006 --> E007[EPIC-007<br/>Documentation<br/>Days 13-14]

    E007 --> Release([v0.1.0 Release<br/>Feb 12])

    style E001 fill:#ef4444,color:#fff,stroke:#991b1b,stroke-width:3px
    style E002 fill:#ef4444,color:#fff,stroke:#991b1b,stroke-width:3px
    style E003 fill:#ef4444,color:#fff,stroke:#991b1b,stroke-width:3px
    style E006 fill:#ef4444,color:#fff,stroke:#991b1b,stroke-width:3px
    style E007 fill:#ef4444,color:#fff,stroke:#991b1b,stroke-width:3px
    style Start fill:#6366f1,color:#fff,stroke:#4f46e5,stroke-width:2px
    style Release fill:#10b981,color:#fff,stroke:#059669,stroke-width:2px
```

**Critical Path Duration:** 13 days (1-day buffer on Day 14)

**Critical Epics:**
1. EPIC-001: Core MAKER Library (4 days) — Blocks all integration work
2. EPIC-002: LLM Provider Abstraction (4 days) — Required for MCP tools
3. EPIC-003: MCP Server Implementation (3 days) — Required for demo
4. EPIC-006: Demo & Benchmarks (3 days) — Validates complete integration
5. EPIC-007: Documentation (2 days) — Release readiness

---

## Dependency Matrix

| Epic | Duration | Start | End | Depends On | Blocks | Phase | Float |
|------|----------|-------|-----|------------|--------|-------|-------|
| **EPIC-001** | 4 days | Day 1 | Day 4 | None | EPIC-002, EPIC-003, EPIC-004, EPIC-006 | Phase 1 | 0 days (Critical) |
| **EPIC-002** | 4 days | Day 6 | Day 9 | EPIC-001 | EPIC-003, EPIC-006 | Phase 2 | 0 days (Critical) |
| **EPIC-003** | 3 days | Day 8 | Day 10 | EPIC-001, EPIC-002 | EPIC-006 | Phase 2 | 0 days (Critical) |
| **EPIC-004** | 3 days | Day 3 | Day 5 | EPIC-001 (partial) | EPIC-006 (metrics) | Phase 1 | 2 days |
| **EPIC-005** | 14 days | Day 1 | Day 14 | All epics (cross-cutting) | None | All Phases | 0 days (Parallel) |
| **EPIC-006** | 3 days | Day 11 | Day 13 | EPIC-001, EPIC-002, EPIC-003 | EPIC-007 | Phase 3 | 0 days (Critical) |
| **EPIC-007** | 2 days | Day 13 | Day 14 | All MVP epics | Release | Phase 3 | 0 days (Critical) |
| **EPIC-008** | 2 days | Day 9 | Day 10 | EPIC-001, EPIC-003 | EPIC-006 | Phase 2-3 | 1 day |
| **EPIC-009** | TBD | Post-MVP | Post-MVP | EPIC-001 | None | Post-MVP | N/A |
| **EPIC-010** | TBD | Post-MVP | Post-MVP | EPIC-002 | None | Post-MVP | N/A |

---

## Milestones

| Milestone | Week | Date | Success Criteria | Owner | Dependencies |
|-----------|------|------|------------------|-------|--------------|
| **M1: Core Algorithms Complete** | Week 1 | Feb 2 (Day 4) | - Zero errors on 3-disk Towers of Hanoi<br>- 95% test coverage on core modules<br>- Property-based tests pass (1000+ iterations)<br>- Events emit correctly to observers | Maintainer | EPIC-001, EPIC-004 (partial), EPIC-005 (partial) |
| **M2: MCP Server Operational** | Week 2 | Feb 7 (Day 9) | - All 4 MCP tools functional (vote, validate, calibrate, configure)<br>- Claude Code integration working (manual test)<br>- Parallel sampling 10x faster than sequential<br>- Schema validation prevents prompt injection | Maintainer | EPIC-001, EPIC-002, EPIC-003, EPIC-008 |
| **M3: End-to-End Demo Validated** | Week 2 | Feb 10 (Day 12) | - Zero errors on 10-disk Hanoi (1,023 steps)<br>- Cost scaling Θ(s ln s) validated empirically<br>- Security audit passes (no vulnerabilities)<br>- Benchmarks complete for 3, 5, 7, 10, 15 disks | Maintainer | EPIC-006, EPIC-008 |
| **M4: v0.1.0 Release Ready** | Week 2 | Feb 12 (Day 14) | - README complete with quickstart and examples<br>- API docs published (rustdoc)<br>- CHANGELOG.md v0.1.0 documented<br>- GitHub release tagged and published<br>- Community announcement prepared | Maintainer | EPIC-007, All MVP epics |

---

## Phase Breakdown

### Phase 1: Core Algorithms (Days 1-5)

**Duration:** 5 days (Jan 30 - Feb 3)
**Objective:** Implement mathematically correct MAKER protocols with comprehensive test coverage

| Epic | Stories | Story Points | Days | Key Deliverables |
|------|---------|--------------|------|------------------|
| **EPIC-001** | 5 stories | 24 points | Days 1-4 | k_min calculation, first-to-ahead-by-k voting, red-flagging parsers, microagent orchestration, state transfer |
| **EPIC-004** | 4 stories | 13 points | Days 3-5 | Event definitions, EventBus, logging observer, metrics observer |
| **EPIC-005** | 2 stories | 10 points | Days 1-5 | Property-based testing framework, CI/CD pipeline setup |

**Total Story Points:** 47

**Exit Criteria:**
- Zero errors on 3-disk Towers of Hanoi (7 steps)
- 95% test coverage on core modules
- All events emit correctly to EventBus
- Property-based tests validate voting convergence

---

### Phase 2: MCP Integration (Days 6-10)

**Duration:** 5 days (Feb 4 - Feb 8)
**Objective:** Expose MAKER protocols as MCP tools for Claude Code integration

| Epic | Stories | Story Points | Days | Key Deliverables |
|------|---------|--------------|------|------------------|
| **EPIC-002** | 6 stories | 28 points | Days 6-9 | LlmClient trait, Ollama/OpenAI/Anthropic clients, exponential backoff, parallel sampling |
| **EPIC-003** | 6 stories | 22 points | Days 8-10 | rmcp server, 4 MCP tools (vote/validate/calibrate/configure), schema validation |
| **EPIC-008** | 3 stories | 8 points | Days 9-10 | Schema enforcement, prompt injection protection, microagent isolation |
| **EPIC-005** | 1 story | 5 points | Days 8-10 | MCP protocol compliance tests |

**Total Story Points:** 63

**Exit Criteria:**
- All 4 MCP tools functional
- Claude Code integration working (manual test)
- Parallel sampling latency < 1.5x single API call time
- Schema validation rejects malformed inputs
- Security audit passes (no prompt injection)

---

### Phase 3: Validation & Hardening (Days 11-14)

**Duration:** 4 days (Feb 9 - Feb 12)
**Objective:** Demonstrate production readiness through benchmarks, security audit, and documentation

| Epic | Stories | Story Points | Days | Key Deliverables |
|------|---------|--------------|------|------------------|
| **EPIC-006** | 4 stories | 18 points | Days 11-13 | Towers of Hanoi task decomposition, 10-disk execution, cost scaling benchmarks, comparison to naive retry |
| **EPIC-007** | 5 stories | 14 points | Days 13-14 | README, API docs (rustdoc), examples, security documentation, CHANGELOG |
| **EPIC-005** | 2 stories | 8 points | Days 11-14 | Monte Carlo cost validation, CI/CD coverage enforcement |

**Total Story Points:** 40

**Exit Criteria:**
- Zero errors on 10-disk Hanoi (1,023 steps)
- Cost scaling Θ(s ln s) validated within 20% tolerance
- README complete with quickstart
- API docs published
- v0.1.0 GitHub release tagged

---

## Sprint Allocation

### Sprint 1: Foundation (Days 1-5)

**Goal:** Implement mathematically correct core algorithms with comprehensive testing

**Epics:**
- EPIC-001: Core MAKER Library (4 days)
- EPIC-004: Event-Driven Observability (3 days, parallel)
- EPIC-005: Testing Infrastructure (5 days, parallel)

**Story Points:** 47

**Daily Breakdown:**
- **Day 1 (Thu Jan 30):** Project setup, k_min calculation, test framework setup
- **Day 2 (Fri Jan 31):** Vote race state tracking, event definitions
- **Day 3 (Sat Feb 1):** Red-flagging parsers, EventBus implementation
- **Day 4 (Sun Feb 2):** Microagent orchestration, logging observer
- **Day 5 (Mon Feb 3):** Parallel voting integration, metrics observer, **Milestone M1**

**Risks:**
- Mathematical correctness errors in SPRT/k_min
- Property-based tests slow convergence
- Red-flagging too aggressive or lenient

**Mitigation:**
- Monte Carlo validation (10,000 simulated races)
- Independent code review of statistical logic
- Baseline test: 3-disk Hanoi continuous validation

---

### Sprint 2: Integration (Days 6-10)

**Goal:** Expose MAKER as MCP tools with multi-provider LLM support

**Epics:**
- EPIC-002: LLM Provider Abstraction (4 days)
- EPIC-003: MCP Server Implementation (3 days, overlap with EPIC-002)
- EPIC-008: Security & Guardrails (2 days, parallel)
- EPIC-005: MCP Protocol Compliance Tests (1 day)

**Story Points:** 63

**Daily Breakdown:**
- **Day 6 (Tue Feb 4):** LlmClient trait, Ollama client
- **Day 7 (Wed Feb 5):** OpenAI client, Anthropic client
- **Day 8 (Thu Feb 6):** Exponential backoff, rmcp server setup, maker/vote tool
- **Day 9 (Fri Feb 7):** Parallel sampling, maker/validate + calibrate tools, schema enforcement, **Milestone M2**
- **Day 10 (Sat Feb 8):** maker/configure tool, prompt injection protection, MCP compliance tests

**Risks:**
- API rate limits during testing
- rmcp SDK breaking changes
- Claude Code integration failures

**Mitigation:**
- Ollama fallback (no rate limits)
- Pin rmcp to v0.13.0+
- Manual integration test Day 9 (1-day buffer)

---

### Sprint 3: Validation (Days 11-14)

**Goal:** Demonstrate production readiness through benchmarks and documentation

**Epics:**
- EPIC-006: Demo & Benchmarks (3 days)
- EPIC-007: Documentation (2 days)
- EPIC-005: Final testing and coverage (2 days, parallel)

**Story Points:** 40

**Daily Breakdown:**
- **Day 11 (Sun Feb 9):** Hanoi task decomposition, 10-disk execution start
- **Day 12 (Mon Feb 10):** Cost scaling benchmarks, naive retry comparison, **Milestone M3**
- **Day 13 (Tue Feb 11):** README, API docs, examples, Monte Carlo cost validation
- **Day 14 (Wed Feb 12):** Security documentation, CHANGELOG, CI/CD final validation, **Milestone M4**, **Release**

**Risks:**
- Benchmark execution time exceeds estimates
- Documentation incomplete at deadline
- Last-minute test failures

**Mitigation:**
- Run benchmarks overnight/async (Days 11-12)
- Draft documentation incrementally during implementation
- Day 14 buffer for release polish

---

## Epic Timeline Details

### EPIC-001: Core MAKER Library (Days 1-4, Phase 1)

**Duration:** 4 days
**Owner:** Maintainer
**Priority:** P0 (Critical)
**Story Points:** 24

| Story | Description | Days | Dependencies |
|-------|-------------|------|--------------|
| STORY-001-01 | k_min Calculation | Day 1 | None |
| STORY-001-02 | Vote Race State Tracking | Days 2-3 | STORY-001-01 |
| STORY-001-03 | Red-Flagging Parsers | Days 3-4 | None |
| STORY-001-04 | Microagent Orchestration | Day 4 | STORY-001-01 |
| STORY-001-05 | Parallel Voting Integration | Day 4-5 | All above |

**Key Milestones:**
- Day 2: k_min calculation validated
- Day 3: Voting converges on test cases
- Day 4: Red-flagging rejects malformed outputs
- Day 5: Zero errors on 3-disk Hanoi

---

### EPIC-002: LLM Provider Abstraction (Days 6-9, Phase 2)

**Duration:** 4 days
**Owner:** Maintainer
**Priority:** P0 (Critical)
**Story Points:** 28

| Story | Description | Days | Dependencies |
|-------|-------------|------|--------------|
| STORY-002-01 | LlmClient Trait | Day 6 | EPIC-001 |
| STORY-002-02 | Ollama Client | Day 6 | STORY-002-01 |
| STORY-002-03 | OpenAI Client | Day 7 | STORY-002-01 |
| STORY-002-04 | Anthropic Client | Day 7-8 | STORY-002-01 |
| STORY-002-05 | Exponential Backoff Retry | Day 8 | Provider clients |
| STORY-002-06 | Parallel Sampling with Tokio | Day 9 | All above |

**Key Milestones:**
- Day 6: Ollama client functional
- Day 7: Multi-provider support
- Day 8: Retry logic handles 429 errors
- Day 9: Parallel sampling 10x faster than sequential

---

### EPIC-003: MCP Server Implementation (Days 8-10, Phase 2)

**Duration:** 3 days
**Owner:** Maintainer
**Priority:** P0 (Critical)
**Story Points:** 22

| Story | Description | Days | Dependencies |
|-------|-------------|------|--------------|
| STORY-003-01 | rmcp Server Setup | Day 8 | EPIC-001, EPIC-002 |
| STORY-003-02 | maker/vote Tool | Day 8-9 | STORY-003-01, EPIC-002 |
| STORY-003-03 | maker/validate Tool | Day 9 | STORY-003-01, EPIC-001 |
| STORY-003-04 | maker/calibrate Tool | Day 9 | STORY-003-01, EPIC-002 |
| STORY-003-05 | maker/configure Tool | Day 9-10 | STORY-003-01 |
| STORY-003-06 | Schema Validation for Security | Day 10 | All tools |

**Key Milestones:**
- Day 8: rmcp server running with stdio transport
- Day 9: maker/vote and maker/validate functional
- Day 10: All 4 tools operational, schema validation enforced

---

### EPIC-004: Event-Driven Observability (Days 3-5, Phase 1)

**Duration:** 3 days (parallel with EPIC-001)
**Owner:** Maintainer
**Priority:** P1 (High)
**Story Points:** 13

| Story | Description | Days | Dependencies |
|-------|-------------|------|--------------|
| STORY-004-01 | Event Definitions | Day 3 | EPIC-001 (partial) |
| STORY-004-02 | EventBus Implementation | Day 3-4 | STORY-004-01 |
| STORY-004-03 | Logging Observer | Day 4 | STORY-004-02 |
| STORY-004-04 | Metrics Observer | Day 5 | STORY-004-02 |

**Key Milestones:**
- Day 3: All MakerEvent variants defined
- Day 4: EventBus broadcasting to observers
- Day 5: Logging and metrics operational

---

### EPIC-005: Testing Infrastructure (Days 1-14, All Phases)

**Duration:** 14 days (ongoing, parallel)
**Owner:** Maintainer
**Priority:** P0 (Critical)
**Story Points:** 18

| Story | Description | Days | Dependencies |
|-------|-------------|------|--------------|
| STORY-005-01 | Property-Based Testing Framework | Days 1-5 | EPIC-001 |
| STORY-005-02 | Monte Carlo Cost Validation | Days 11-13 | EPIC-006 |
| STORY-005-03 | MCP Protocol Compliance Tests | Days 8-10 | EPIC-003 |
| STORY-005-04 | CI/CD Pipeline with Coverage | Days 1-14 | All epics |

**Key Milestones:**
- Day 1: CI/CD pipeline enforcing 95% coverage
- Day 5: Property tests validate voting convergence
- Day 10: MCP compliance tests pass
- Day 14: Full coverage achieved

---

### EPIC-006: Demo & Benchmarks (Days 11-13, Phase 3)

**Duration:** 3 days
**Owner:** Maintainer
**Priority:** P0 (Critical)
**Story Points:** 18

| Story | Description | Days | Dependencies |
|-------|-------------|------|--------------|
| STORY-006-01 | Towers of Hanoi Task Decomposition | Day 11 | EPIC-001 |
| STORY-006-02 | End-to-End 10-Disk Execution | Days 11-12 | All EPIC-001/002/003 |
| STORY-006-03 | Cost Scaling Benchmark Suite | Day 12-13 | STORY-006-02 |
| STORY-006-04 | Comparison to Naive Retry | Day 13 | STORY-006-03 |

**Key Milestones:**
- Day 11: Hanoi decomposition complete
- Day 12: Zero errors on 10-disk Hanoi (1,023 steps)
- Day 13: Θ(s ln s) cost scaling validated

---

### EPIC-007: Documentation (Days 13-14, Phase 3)

**Duration:** 2 days
**Owner:** Maintainer
**Priority:** P1 (High)
**Story Points:** 14

| Story | Description | Days | Dependencies |
|-------|-------------|------|--------------|
| STORY-007-01 | README.md | Day 13 | All MVP epics |
| STORY-007-02 | API Documentation (rustdoc) | Day 13 | All MVP epics |
| STORY-007-03 | Example Integrations | Day 13-14 | EPIC-006 |
| STORY-007-04 | Security Documentation | Day 14 | EPIC-008 |
| STORY-007-05 | CHANGELOG.md | Day 14 | All MVP epics |

**Key Milestones:**
- Day 13: README complete with quickstart
- Day 14: All docs published, v0.1.0 release ready

---

### EPIC-008: Security & Guardrails (Days 9-10, Phase 2-3)

**Duration:** 2 days
**Owner:** Maintainer
**Priority:** P1 (High)
**Story Points:** 8

| Story | Description | Days | Dependencies |
|-------|-------------|------|--------------|
| STORY-008-01 | Schema Enforcement for Agent Outputs | Day 9 | EPIC-001, EPIC-003 |
| STORY-008-02 | Prompt Injection Protection | Day 9-10 | EPIC-003 |
| STORY-008-03 | Microagent Isolation Enforcement | Day 10 | EPIC-001 |

**Key Milestones:**
- Day 9: Schema validation enforced
- Day 10: Security audit passes, penetration test complete

---

### EPIC-009: Semantic Matching (Post-MVP)

**Duration:** TBD
**Owner:** Community
**Priority:** P2 (Medium)
**Story Points:** TBD (estimated 20-30)

**Planned Start:** After v0.1.0 release
**Dependencies:** EPIC-001 (extends voting beyond exact match)

**Features:**
- Semantic similarity scoring (embeddings, AST comparison)
- Configurable equivalence thresholds
- Domain-specific matchers (code, SQL, natural language)

---

### EPIC-010: Multi-Model Ensemble (Post-MVP)

**Duration:** TBD
**Owner:** Community
**Priority:** P2 (Medium)
**Story Points:** TBD (estimated 15-25)

**Planned Start:** After v0.1.0 release
**Dependencies:** EPIC-002 (multiple LLM clients)

**Features:**
- Ensemble configuration: multiple LlmClients per vote
- Cost optimization: cheap models first, expensive on disagreement
- Model-specific red-flag tuning

---

## Resource Allocation

**Team:** Single maintainer (open source model)
**Concurrent Work:** Limited to context-switching between parallel epics

| Week | Days | Allocation | Notes |
|------|------|------------|-------|
| **Week 1** | Days 1-5 | 80% EPIC-001, 20% EPIC-004/005 | Front-load critical path |
| **Week 2 (Part 1)** | Days 6-10 | 90% EPIC-002/003, 10% EPIC-005/008 | Integration sprint |
| **Week 2 (Part 2)** | Days 11-14 | 70% EPIC-006, 30% EPIC-007/005 | Validation and docs |

---

## Risk Mitigation Timeline

| Risk | Impact Window | Mitigation Timeline |
|------|---------------|---------------------|
| **EPIC-001 mathematical errors** | Days 1-4 | Property tests Day 2, Monte Carlo validation Day 3, 3-disk Hanoi baseline Day 4 |
| **API rate limits** | Days 6-13 | Ollama fallback Day 6, exponential backoff Day 8, batch API usage Day 11 |
| **MCP protocol mismatches** | Days 8-10 | Manual Claude Code test Day 9, compliance tests Day 10 |
| **Benchmark execution time** | Days 11-13 | Async overnight runs Days 11-12, results analysis Day 13 |
| **Documentation incomplete** | Days 13-14 | Incremental drafting Days 1-12, finalization Day 13-14 |

---

## Success Metrics

| Metric | Baseline | Target | Measurement Point |
|--------|----------|--------|-------------------|
| **Test Coverage** | 0% | 95%+ | Continuous (CI/CD) |
| **3-Disk Hanoi Success** | 0% | 100% (0 errors) | Day 5 (Milestone M1) |
| **10-Disk Hanoi Success** | 0% | 100% (0 errors) | Day 12 (Milestone M3) |
| **MCP Tools Functional** | 0/4 | 4/4 | Day 10 (Milestone M2) |
| **Cost Scaling Validation** | TBD | Θ(s ln s) within 20% | Day 13 (Benchmark suite) |
| **GitHub Stars** | 0 | 100+ (6 months) | Post-release tracking |
| **Community PRs** | 0 | 10+ (6 months) | Post-release tracking |

---

## Next Steps

1. Create GitHub Project board with Sprint 1/2/3 columns
2. Generate GitHub Issues for all stories using JIRA-STRUCTURE.md
3. Begin Day 1 implementation:
   - Initialize Rust workspace
   - Configure CI/CD (GitHub Actions)
   - Implement k_min calculation (STORY-001-01)
4. Daily standup (async via GitHub comments):
   - Progress: stories completed today
   - Blockers: issues preventing progress
   - Plan: stories for tomorrow

---

**Gantt Chart Status:** ✅ Complete
**Critical Path:** 13 days (1-day buffer)
**Ready for Implementation:** Yes
**Next Milestone:** M1 - Core Algorithms Complete (Day 5, Feb 3)
