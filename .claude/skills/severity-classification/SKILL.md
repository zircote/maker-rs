---
name: severity-classification
description: Define a severity/priority classification framework with decision criteria, routing rules, escalation procedures, and governance process. Use when the project involves operational processes, incidents, or work item triage.
allowed-tools: Read, Grep, Glob, WebSearch
---

Generate `SEVERITY-CLASSIFICATION.md` at the repository root.

## Process

1. Read `_research/DOMAIN-RESEARCH.md` for industry classification standards
2. Read `PROJECT-PLAN.md` for operational context
3. Define 3-4 severity levels with SLAs
4. Build Mermaid decision tree flowchart
5. Map routing rules to communication channels
6. Define escalation procedures with timelines

## Required Sections

1. **Classification Levels** (P1-P4 or equivalent): Definition, response time SLA, resolution target, notification channel, 2-3 domain-specific examples per level
2. **Decision Tree**: Mermaid `flowchart TD` for classification, color-coded by severity
3. **Routing Rules**: Table (Level | Primary Channel | Secondary | Escalation | Response SLA)
4. **Classification Examples**: Table (Scenario | Classification | Rationale), minimum 6 examples
5. **Escalation Procedures**: Trigger table and Mermaid escalation path diagram
6. **Governance**: Quarterly review process and reclassification guidelines

## Quality Criteria

- [ ] 3-4 severity levels with SLAs
- [ ] Decision tree renders in Mermaid
- [ ] Routing rules cover all levels
- [ ] 6+ domain-specific classification examples
- [ ] Escalation paths with timelines
- [ ] Governance review process included
