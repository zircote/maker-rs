---
name: risk-register
description: Identify, assess, and plan mitigation for all project risks with likelihood/impact scoring, contingency plans, and monitoring schedules. Use after project plan defines scope and constraints.
allowed-tools: Read, Grep, Glob, WebSearch
---

Generate `RISK-REGISTER.md` at the repository root.

## Process

1. Read `PROJECT-PLAN.md` for scope, timeline, budget, approach
2. Read `_research/DOMAIN-RESEARCH.md` for industry-specific risks and compliance
3. Read `RACI-CHART.md` for role assignments (risk owners)
4. Identify 8-15 risks across categories: Operational, Technical, Organizational, Financial, Compliance
5. Score each with Likelihood (1-4) x Impact (1-4)
6. Define 3+ mitigation strategies per high-score risk
7. Create contingency plans for each risk

## Required Sections

1. **Risk Assessment Overview**: Mermaid `quadrantChart` plotting all risks by likelihood/impact
2. **Risk Scoring Methodology**: Table defining L and I scales (1-4) with descriptions
3. **Risk Summary**: Table (ID | Risk | Category | L | I | Score | Owner | Status)
4. **Detailed Risk Profiles** (per risk):
   - Attributes table (Category, Likelihood with rationale, Impact with rationale, Score, Owner, Detection method)
   - Mitigation Strategies (numbered list, 3+ for high-score)
   - Contingency Plan (numbered steps if risk materializes)
   - Triggers / Early Warning Signs (bullet list)
5. **Risk Categories**: Summary table (Category | Count | Highest Score)
6. **Risk Monitoring Schedule**: Table (Cadence | Activity | Participants | Output)

## Quality Criteria

- [ ] 8-15 risks across multiple categories
- [ ] Mermaid quadrant chart renders correctly
- [ ] Every risk has mitigation AND contingency plan
- [ ] Each risk has a named owner
- [ ] Early warning triggers defined
- [ ] Monitoring schedule specified
