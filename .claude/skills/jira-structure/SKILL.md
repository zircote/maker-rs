---
name: jira-structure
description: Generate a complete JIRA epic/story hierarchy with acceptance criteria, sprint planning, issue type taxonomy, labels, and team assignments. Use after project plan defines phases.
allowed-tools: Read, Grep, Glob, WebSearch
---

Generate `JIRA-STRUCTURE.md` at the repository root.

## Process

1. Read `PROJECT-PLAN.md` for phases, goals, and timeline
2. Read `_research/DOMAIN-RESEARCH.md` for terminology and team structure
3. Decompose each phase into 2-5 epics (10-20 total)
4. Decompose each epic into 3-8 stories (40-80 total)
5. Define acceptance criteria for every story
6. Plan sprint allocation across timeline

## Required Sections

1. **Epic Hierarchy**: Mermaid `mindmap` showing all epics grouped by phase
2. **Epic Definitions** (per phase):
   - Epic ID, title, owner, duration, priority, dependencies
   - Acceptance criteria (checkbox list)
   - Stories with "As a... I want... So that..." format
   - Story tasks (checkbox list) and acceptance criteria
3. **Issue Type Reference**: Table (Type | When to Use | Example) covering Epic, Story, Task, Bug, Spike
4. **Labels & Components**: Tables defining project labels and component ownership
5. **Sprint Planning**: Cadence, total sprints, ceremony list
6. **Sprint Allocation**: Table (Sprint | Dates | Phase | Epics | Story Points)
7. **Team Assignments**: Table (Area | Owner Team | Epic IDs)

## Scaling Rules

| Project Size | Epics | Stories | Sprints |
|-------------|-------|---------|---------|
| Small (1-3mo) | 5-8 | 20-30 | 3-6 |
| Medium (3-6mo) | 10-15 | 40-60 | 6-12 |
| Large (6+mo) | 15-25 | 60-100 | 12+ |

## Quality Criteria

- [ ] Each epic has clear owner and duration
- [ ] Stories follow "As a... I want... So that..." format
- [ ] Acceptance criteria are measurable and testable
- [ ] Sprint allocation is balanced
- [ ] Mermaid mindmap renders correctly
- [ ] Issue type taxonomy defined
