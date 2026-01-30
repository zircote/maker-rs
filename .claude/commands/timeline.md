---
description: Generate Gantt chart, SVG timeline, and dependency visualization
allowed-tools: Read, Write, Grep, Glob, Bash
---

# Timeline Artifacts

Run the timeline phase of the project planning pipeline.

## Prerequisites

- JIRA structure must exist: !`ls JIRA-STRUCTURE.md 2>/dev/null || echo "MISSING - run /plan first"`
- Dependency analysis must exist: !`ls _research/DEPENDENCY-ANALYSIS.md 2>/dev/null || echo "MISSING - run /plan first"`

## Process

1. If any prerequisites are missing, inform the user which commands to run first and stop
2. Run `/gantt-chart` with JIRA and dependency data to produce `GANTT-CHART.md`
3. Run `/svg-gantt-generator` to produce `.github/gantt-chart.svg`
4. Verify outputs:
   - All epics from JIRA structure represented in Gantt
   - SVG renders without external dependencies
   - Critical path Mermaid diagram present
   - 3-5 milestones with success criteria

## Output

Report what was generated: total timeline duration, milestone count, and critical path length.
