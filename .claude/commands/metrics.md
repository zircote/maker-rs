---
description: Generate success metrics, KPIs, and executive briefing deck
allowed-tools: Read, Write, Grep, Glob, Bash, WebSearch
---

# Metrics & Executive Deliverables

Run the metrics and executive deliverables phase of the project planning pipeline.

## Prerequisites

- Project plan must exist: !`ls PROJECT-PLAN.md 2>/dev/null || echo "MISSING - run /plan first"`
- Domain research must exist: !`ls _research/DOMAIN-RESEARCH.md 2>/dev/null || echo "MISSING - run /research first"`

## Process

1. If any prerequisites are missing, inform the user which commands to run first and stop
2. Run `/success-metrics` with project plan goals to produce `SUCCESS-METRICS.md`
3. Run `/runbook-template` with domain research and severity classification to produce `RUNBOOK-TEMPLATE.md`
4. Run `/executive-briefing` aggregating all artifact data to produce slides in `workspace/slides/`
5. Verify outputs:
   - 8-15 KPIs with formulas and data sources
   - Dashboard specification with 4 sections
   - 8 HTML slide sources generated
   - Build scripts configured

## Output

Report what was generated: KPI count, closure criteria count, slide count, and ROI headline.
