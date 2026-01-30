---
name: raci-chart
description: Create a responsibility assignment matrix mapping all project activities to roles with R/A/C/I designations including ongoing governance. Use after project plan and JIRA structure define activities and roles.
allowed-tools: Read, Grep, Glob
---

Generate `RACI-CHART.md` at the repository root.

## Process

1. Read `PROJECT-PLAN.md` for phases and activities
2. Read `_research/DOMAIN-RESEARCH.md` for organizational roles
3. Read `JIRA-STRUCTURE.md` for epic/story assignments
4. Define 6-12 project roles with descriptions
5. List all activities from phases and epics
6. Assign R/A/C/I enforcing single-A rule per activity
7. Add ongoing governance section

## Required Sections

1. **Legend**: Table (Code | Meaning | Description) for R, A, C, I
2. **Role Definitions**: Table (# | Role | Description | Typical Title)
3. **Phase RACI Matrices**: One table per phase (Activity | Role1 | Role2 | ...)
4. **Ongoing Governance**: Post-project RACI for maintenance activities
5. **Decision Authority Matrix**: Table (Decision Type | Authority | Escalation Path)

## Rules

- Every activity has exactly **ONE** Accountable (A)
- Every activity has at least one Responsible (R)
- No role should be R or A for more than 60% of activities
- 30-60 activities typical for medium-complexity projects

## Quality Criteria

- [ ] Single-A rule enforced for every activity
- [ ] 6-12 roles with clear descriptions
- [ ] All phases covered plus ongoing governance
- [ ] Decision authority matrix included
- [ ] Balanced workload distribution
