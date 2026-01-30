---
description: Generate RACI chart, risk register, and severity classification
allowed-tools: Read, Write, Edit, Grep, Glob, WebSearch
---

# Governance Artifacts

Run the governance phase of the project planning pipeline.

## Prerequisites

- Project plan must exist: !`ls PROJECT-PLAN.md 2>/dev/null || echo "MISSING - run /plan first"`
- JIRA structure must exist: !`ls JIRA-STRUCTURE.md 2>/dev/null || echo "MISSING - run /plan first"`
- Domain research must exist: !`ls _research/DOMAIN-RESEARCH.md 2>/dev/null || echo "MISSING - run /research first"`

## Process

1. If any prerequisites are missing, inform the user which commands to run first and stop
2. Run `/raci-chart` with project plan and JIRA structure to produce `RACI-CHART.md`
3. Run `/risk-register` with project plan and domain research to produce `RISK-REGISTER.md`
4. Run `/severity-classification` with domain research to produce `SEVERITY-CLASSIFICATION.md`
5. Cross-validate:
   - Roles in RACI match risk owners in risk register
   - Severity levels align with escalation paths
   - Risk mitigations map to RACI activities

## Output

Report what was generated: role count, risk count with top risk highlighted, severity levels defined.
