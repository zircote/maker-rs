---
name: dependency-analyzer
description: Analyze the work breakdown structure to identify critical path, parallel work opportunities, dependency chains, resource conflicts, and bottlenecks. Use after JIRA structure defines epics.
user-invocable: false
allowed-tools: Read, Grep, Glob
---

Generate `_research/DEPENDENCY-ANALYSIS.md`. Invoked by `timeline-architect` and `project-architect`.

## Process

1. Read `JIRA-STRUCTURE.md` for epics with durations and blocking relationships
2. Map all dependency chains
3. Calculate critical path (longest dependency chain)
4. Identify parallel work tracks
5. Detect resource conflicts
6. Calculate float for non-critical items

## Required Sections

1. **Critical Path**: Mermaid `graph LR` showing blocking sequence, color-coded
2. **Dependency Matrix**: Table (Epic | Duration | Depends On | Blocks | Earliest Start | Latest Start | Float)
3. **Parallel Tracks**: Table (Track | Epics | Duration | Resource)
4. **Bottlenecks**: Table (Bottleneck | Impact | Mitigation)
5. **Resource Conflicts**: Table (Week | Conflict | Teams Affected | Resolution)
6. **Recommendations**: Numbered list for schedule optimization

## Quality Criteria

- [ ] Critical path identified with Mermaid diagram
- [ ] All dependencies mapped
- [ ] Parallel tracks identified
- [ ] Bottlenecks flagged with mitigations
- [ ] Float calculated for non-critical items
