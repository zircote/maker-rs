---
description: Generate the master project plan, JIRA structure, and dependency analysis
argument-hint: "[optional focus area or constraint override]"
allowed-tools: Read, Write, Grep, Glob, WebSearch
---

# Core Planning

Run the core planning phase of the project planning pipeline.

## Prerequisites

- Domain research must exist: !`ls _research/DOMAIN-RESEARCH.md 2>/dev/null || echo "MISSING - run /research first"`
- Best practices must exist: !`ls BEST-PRACTICES.md 2>/dev/null || echo "MISSING - run /research first"`

## Context

- Project context: !`head -30 PROJECT-CONTEXT.md 2>/dev/null || echo "PROJECT-CONTEXT.md not found"`

## Process

$IF($ARGUMENTS,
  Apply constraint override: $ARGUMENTS,
  Use constraints from PROJECT-CONTEXT.md
)

1. If prerequisites are missing, inform the user to run `/research` first and stop
2. Run `/project-plan` using domain research, problem data, and constraints to produce `PROJECT-PLAN.md`
3. Run `/jira-structure` using the project plan phases to produce `JIRA-STRUCTURE.md`
4. Run `/dependency-analyzer` to produce `_research/DEPENDENCY-ANALYSIS.md`
5. Verify outputs meet quality gates:
   - 3-5 phases with clear exit criteria
   - 10+ epics with acceptance criteria
   - Critical path identified
   - ROI calculated with 3-year projection

## Output

Report what was generated: phase count, epic count, critical path summary, and ROI headline figure.
