---
description: Generate all project planning artifacts from PROJECT-CONTEXT.md using the full orchestrated workflow
allowed-tools: Read, Write, Edit, Grep, Glob, Bash, WebSearch, WebFetch
---

# Generate All Project Planning Artifacts

You are orchestrating the complete project planning artifact generation pipeline. Follow the phased workflow below exactly, using the specialist agents to produce a complete, internally consistent artifact set.

## Prerequisites Check

1. Read `PROJECT-CONTEXT.md` at the repository root
2. If it does not exist or contains only placeholder text, stop and ask the user to fill it in first
3. Confirm the project context with the user before proceeding

## Execution Phases

### Phase 1: Research

Use the `research-specialist` agent to:
1. Run `/domain-research` with the problem statement from PROJECT-CONTEXT.md
2. Run `/best-practices` using the domain research output
3. **Gate**: Verify `_research/DOMAIN-RESEARCH.md` exists with 8+ framework citations and a terminology dictionary

### Phase 2: Core Planning

1. Run `/project-plan` using domain research, problem data, and constraints
2. Run `/jira-structure` using the project plan phases
3. Run `/dependency-analyzer` using JIRA structure output
4. **Gate**: Verify `PROJECT-PLAN.md` and `JIRA-STRUCTURE.md` exist with phases defined and 10+ epics

### Phase 3: Detailed Artifacts (Parallel)

Use the specialist agents:

**timeline-architect**:
- Run `/gantt-chart` with JIRA and dependency data
- Run `/svg-gantt-generator` for the SVG visualization

**governance-architect**:
- Run `/raci-chart` with project plan and JIRA structure
- Run `/risk-register` with project plan and domain research
- Run `/severity-classification` with domain research

**metrics-architect**:
- Run `/success-metrics` with project plan goals
- Run `/runbook-template` with domain research and severity classification

**Gate**: All Phase 3 artifacts generated

### Phase 4: Documentation & Visuals

Use the `documentation-architect` agent to:
1. Run `/infographic-generator` to create SVG visual summaries
2. Run `/readme-generator` aggregating all artifacts
3. Run `/changelog` documenting all additions

### Phase 5: Executive Deliverables

Use the `metrics-architect` agent to:
1. Run `/executive-briefing` aggregating key data into slides

### Phase 6: Validation

Cross-check all artifacts for:
- [ ] Internal links resolve to actual files
- [ ] Role names consistent across RACI, JIRA, Project Plan
- [ ] Epic IDs match between JIRA and Gantt
- [ ] Metrics in Success Metrics appear in Project Plan
- [ ] Phase names consistent everywhere
- [ ] Terminology matches domain research dictionary

Report any inconsistencies to the user with specific file locations and suggested fixes.

## Output

When complete, list all generated artifacts with their file paths and a one-line summary of each.
