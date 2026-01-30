---
name: project-plan
description: Generate a comprehensive master project plan with executive summary, current state analysis, phased approach, financial ROI, and actionable next steps. Use after domain research is complete.
argument-hint: "[optional focus area or constraint override]"
allowed-tools: Read, Grep, Glob, WebSearch, WebFetch
---

Generate `PROJECT-PLAN.md` at the repository root using context from `_research/DOMAIN-RESEARCH.md` and `PROJECT-CONTEXT.md`.

## Process

1. **Analyze current state**: Process available data to establish quantified baselines
2. **Benchmark**: Compare findings against industry benchmarks from domain research
3. **Structure phases**: Create 3-5 phases with clear exit criteria
4. **Calculate ROI**: Build 3-year financial model from cost data
5. **Identify risks**: Flag top 5 for risk register cross-reference
6. **Define next steps**: Create 5+ actionable, owned items

## Required Sections

1. **Executive Summary** (under 200 words): Problem quantified, approach, expected outcome, ROI
2. **Current State Analysis**: Findings table (Category | Finding | Impact | Evidence), root cause analysis, top problem areas ranked
3. **Project Goals**: Primary objective with metrics, success criteria checklist, industry alignment
4. **Project Timeline**: Phase overview table (Phase | Name | Duration | Deliverable | Target), detailed phase descriptions with exit criteria
5. **Jira Project Structure Overview**: Summary pointing to JIRA-STRUCTURE.md
6. **RACI Overview**: Summary pointing to RACI-CHART.md
7. **Financial ROI**: Annual value table, project investment table, 3-year net ROI with ROI multiple
8. **Risk Management Summary**: Top 5 risks pointing to RISK-REGISTER.md
9. **Immediate Next Steps**: Numbered table (Action | Owner | Timeline)
10. **Project Artifacts**: Index table linking to all documents

## Quality Criteria

- [ ] Executive summary under 200 words
- [ ] All metrics quantified with baselines and targets
- [ ] 3-5 phases with clear exit criteria
- [ ] ROI calculated with 3-year projection
- [ ] 5+ immediate next steps with owners
- [ ] Cross-references to all other artifacts
