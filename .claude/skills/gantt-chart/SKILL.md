---
name: gantt-chart
description: Create a visual project timeline with SVG Gantt chart, Mermaid critical path diagram, dependency matrix, and milestone tracking. Use after JIRA structure and dependency analysis are complete.
allowed-tools: Read, Grep, Glob, Write, Bash
---

Generate `GANTT-CHART.md` at the repository root and `.github/gantt-chart.svg`.

## Process

1. Read `JIRA-STRUCTURE.md` for epic IDs, durations, and phase assignments
2. Read `_research/DEPENDENCY-ANALYSIS.md` for critical path and dependencies
3. Generate the SVG Gantt chart following the [SVG Gantt Generator specification](../svg-gantt-generator/SKILL.md)
4. Write the Mermaid critical path flowchart
5. Build the dependency matrix table
6. Define milestones with success criteria

## Required Sections in GANTT-CHART.md

1. **Project Timeline header**: Duration and date range
2. **Embedded SVG**: `![Gantt Chart](.github/gantt-chart.svg)`
3. **Critical Path**: Mermaid `graph LR` flowchart showing blocking dependencies, color-coded by phase
4. **Dependency Matrix**: Table (Epic | Depends On | Blocks | Phase)
5. **Milestones**: Table (Milestone | Week | Date | Success Criteria | Owner), 3-5 milestones

## SVG Requirements

See the `svg-gantt-generator` skill for the full SVG specification. Key points:
- 1280px wide, dark theme (#0f172a background)
- Phase-coded gradient bars (blue/green/amber/purple)
- Sub-task bars nested under epics
- Week grid columns
- Milestone diamond markers

## Quality Criteria

- [ ] **SVG validates as well-formed XML** (`cd workspace && node validate-svg.js ../.github/gantt-chart.svg`)
- [ ] SVG renders in GitHub markdown
- [ ] All epics from JIRA structure represented
- [ ] Critical path Mermaid diagram renders
- [ ] Dependencies mapped in matrix table
- [ ] 3-5 milestones with success criteria
