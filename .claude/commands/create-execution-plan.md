---
description: Create a detailed execution plan from JIRA-STRUCTURE.md that a future Claude session can load and execute the actual project work
allowed-tools: Read, Write, Grep, Glob
---

# Create Execution Plan

Read the generated project artifacts and produce a comprehensive, persistent execution plan for the **actual project work** — the epics, stories, and tasks defined in JIRA-STRUCTURE.md. A future Claude session should be able to load this plan and begin implementing the real deliverables.

**This is NOT a plan for generating planning artifacts.** The planning artifacts already exist. This plan captures the work those artifacts describe — the implementation, configuration, migration, integration, or whatever the project entails.

## Context

- Project context: !`cat PROJECT-CONTEXT.md 2>/dev/null || echo "PROJECT-CONTEXT.md not found"`
- JIRA structure: !`cat JIRA-STRUCTURE.md 2>/dev/null || echo "JIRA-STRUCTURE.md not found — run /jira-structure first"`
- Project plan: !`cat PROJECT-PLAN.md 2>/dev/null || echo "PROJECT-PLAN.md not found — run /project-plan first"`
- Dependencies: !`cat _research/DEPENDENCY-ANALYSIS.md 2>/dev/null || echo "No dependency analysis found"`

If `JIRA-STRUCTURE.md` does not exist or contains only placeholder text, stop and tell the user to generate it first with `/jira-structure`.

## Process

1. Read `JIRA-STRUCTURE.md` thoroughly — extract every epic, story, task, acceptance criterion, sprint allocation, and dependency
2. Read `PROJECT-PLAN.md` for phase structure, timeline, and success criteria
3. Read `_research/DEPENDENCY-ANALYSIS.md` for critical path and blocking relationships
4. Read `RACI-CHART.md` for role assignments and decision authority
5. Read `RISK-REGISTER.md` for risks that may affect execution
6. Create the `_plan/` directory
7. Write four files as described below, translating every JIRA story into an actionable work item Claude can execute

## File 1: `_plan/EXECUTION-PLAN.md`

Write a full execution plan as a Claude-readable instruction document. A future Claude session should be able to read this file and know exactly what project work to do. Structure it as follows:

```markdown
# Execution Plan: [Project Name]

> This file is a complete execution plan for implementing the project work.
> It was generated from JIRA-STRUCTURE.md and the project planning artifacts.
> To execute: open a Claude Code session in this repository and say:
> "Read _plan/EXECUTION-PLAN.md and begin executing the work items"

Generated: [ISO 8601 date]
Domain: [detected domain from PROJECT-CONTEXT.md]
Project: [1-2 sentence summary of the actual problem being solved]

## How to Execute This Plan

Read this file, then work through each epic and its stories sequentially within each phase.
For each work item:
1. Check prerequisites and blocking dependencies
2. Read the relevant context files listed
3. Execute the work described — write code, create configs, run commands, etc.
4. Verify acceptance criteria are met
5. Mark complete and move to the next item

Epics within the same phase that have no cross-dependencies can be worked in parallel.

## Project Summary

[2-3 paragraph summary of what the project delivers, extracted from PROJECT-PLAN.md]

## Success Criteria

[Extract from PROJECT-PLAN.md and SUCCESS-METRICS.md — the measurable outcomes]

## Risk Awareness

[Top 5 risks from RISK-REGISTER.md with their mitigations, so the executing session knows what to watch for]

---

## Phase 1: [Phase Name from PROJECT-PLAN.md]

### Epic [ID]: [Title]
- **Owner:** [from JIRA-STRUCTURE.md]
- **Duration:** [from JIRA-STRUCTURE.md]
- **Priority:** [from JIRA-STRUCTURE.md]
- **Dependencies:** [blocking epics]
- **Phase exit criteria:** [from PROJECT-PLAN.md]

#### Story [ID]: [Title]
- **Description:** As a [role], I want [capability], so that [benefit]
- **Context files:** [list any files this story needs to read or modify]
- **Work to do:**
  - [ ] [Concrete task 1 — what to build, configure, write, etc.]
  - [ ] [Concrete task 2]
  - [ ] [Concrete task 3]
- **Acceptance criteria:**
  - [ ] [Criterion 1 — measurable/testable]
  - [ ] [Criterion 2]
- **Verification:** [How to verify this is done — run a test, check output, etc.]

[... repeat for all stories in this epic ...]

[... repeat for all epics in this phase ...]

---

## Phase 2: [Phase Name]
[... same structure ...]

---

[... continue for all phases ...]
```

**Critical rules for the execution plan:**
- Every story from JIRA-STRUCTURE.md MUST appear as a work item
- Tasks must be concrete and actionable — "write code to...", "configure...", "create file...", "run migration..."
- Do NOT include meta-tasks like "generate planning documents" — only real project work
- Include file paths where code/config should be created or modified, when determinable
- Translate abstract stories into specific implementation steps Claude can follow
- Preserve all acceptance criteria exactly as defined in JIRA-STRUCTURE.md
- Add verification steps so the executing session knows how to confirm completion

## File 2: `_plan/TASK-MANIFEST.json`

Write a machine-readable JSON array of all work items extracted from JIRA-STRUCTURE.md:

```json
[
  {
    "id": "E1-S1",
    "epic_id": "E1",
    "epic_title": "...",
    "phase": 1,
    "phase_name": "...",
    "story_title": "...",
    "description": "As a ... I want ... so that ...",
    "owner": "...",
    "priority": "...",
    "sprint": 1,
    "story_points": 5,
    "blocked_by": [],
    "tasks": [
      "Concrete task description 1",
      "Concrete task description 2"
    ],
    "acceptance_criteria": [
      "Criterion 1",
      "Criterion 2"
    ],
    "context_files": ["file1.md", "file2.py"]
  }
]
```

Include ALL stories from JIRA-STRUCTURE.md. Preserve the epic/story ID scheme used in the JIRA structure.

## File 3: `_plan/RUNSHEET.md`

Write a concise operator checklist organized by sprint:

```markdown
# Runsheet: [Project Name]

> Work through each sprint's items in order.
> Check off acceptance criteria as you complete each story.
> Items marked with ⊘ are blocked until dependencies complete.

## Sprint 1 — [Phase Name] — [Date Range]

### Epic [ID]: [Title]

- [ ] **[Story ID]** — [Story Title]
  - [ ] [Task 1]
  - [ ] [Task 2]
  - [ ] ✓ [Acceptance criterion 1]
  - [ ] ✓ [Acceptance criterion 2]

- [ ] **[Story ID]** — [Story Title] ⊘ [Blocking Story ID]
  [...]

**Sprint Gate:** [What must be true before moving to Sprint 2]

## Sprint 2 — [Phase Name] — [Date Range]
[... continue ...]

## Final Validation
- [ ] All acceptance criteria met across all epics
- [ ] Success metrics verified against targets
- [ ] Risk mitigations confirmed effective
- [ ] Project closure criteria from SUCCESS-METRICS.md satisfied
```

## File 4: `_plan/CONTEXT-BRIEF.md`

Write a concise briefing document that gives a new Claude session all the context it needs without reading every artifact:

```markdown
# Context Brief: [Project Name]

> Read this file first to understand the project before executing the plan.

## What This Project Is
[3-4 sentences: problem, domain, approach, expected outcome]

## Key Decisions Already Made
[List architectural, technology, and methodology decisions from the planning artifacts]

## Terminology
[10-15 key domain terms from DOMAIN-RESEARCH.md that the executing session needs to know]

## File Map
| File | Contains | Read When |
|------|----------|-----------|
| PROJECT-CONTEXT.md | Original problem statement | Understanding scope |
| PROJECT-PLAN.md | Phase structure, ROI, success criteria | Planning work order |
| JIRA-STRUCTURE.md | Epic/story details | Detailed task reference |
| RACI-CHART.md | Role assignments | Knowing who owns what |
| RISK-REGISTER.md | Risk mitigations | Encountering issues |
| SEVERITY-CLASSIFICATION.md | Priority framework | Triaging problems |
| BEST-PRACTICES.md | Industry standards | Making design decisions |
| SUCCESS-METRICS.md | KPIs and closure criteria | Verifying completion |

## Constraints and Guardrails
[Budget, timeline, compliance, technology constraints from PROJECT-CONTEXT.md]

## How to Start
1. Read `_plan/EXECUTION-PLAN.md` for the full work breakdown
2. Or read `_plan/RUNSHEET.md` for a sprint-by-sprint checklist
3. Begin with Phase 1, Sprint 1 items
4. Use `_plan/TASK-MANIFEST.json` for programmatic task tracking
```

## Output

After writing all four files, report:

```
## Execution Plan Created

Source: JIRA-STRUCTURE.md ([N] epics, [M] stories)

Files written:
- _plan/EXECUTION-PLAN.md — Full work breakdown with implementation steps
- _plan/TASK-MANIFEST.json — Machine-readable story manifest
- _plan/RUNSHEET.md — Sprint-by-sprint operator checklist
- _plan/CONTEXT-BRIEF.md — Quick-start briefing for new sessions

To execute:
1. Open a new Claude Code session in this repository
2. Say: "Read _plan/CONTEXT-BRIEF.md then _plan/EXECUTION-PLAN.md and begin the work"
3. Or: "Read _plan/RUNSHEET.md and work through the sprint checklist"
```
