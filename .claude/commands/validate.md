---
description: Validate all generated artifacts for internal consistency and completeness
allowed-tools: Read, Grep, Glob, Bash
---

# Validate Artifacts

Run a comprehensive validation of all generated project planning artifacts.

## Inventory

Check which artifacts exist:
- Domain Research: !`ls _research/DOMAIN-RESEARCH.md 2>/dev/null || echo "MISSING"`
- Best Practices: !`ls BEST-PRACTICES.md 2>/dev/null || echo "MISSING"`
- Project Plan: !`ls PROJECT-PLAN.md 2>/dev/null || echo "MISSING"`
- JIRA Structure: !`ls JIRA-STRUCTURE.md 2>/dev/null || echo "MISSING"`
- Dependency Analysis: !`ls _research/DEPENDENCY-ANALYSIS.md 2>/dev/null || echo "MISSING"`
- Gantt Chart: !`ls GANTT-CHART.md 2>/dev/null || echo "MISSING"`
- SVG Gantt: !`ls .github/gantt-chart.svg 2>/dev/null || echo "MISSING"`
- RACI Chart: !`ls RACI-CHART.md 2>/dev/null || echo "MISSING"`
- Risk Register: !`ls RISK-REGISTER.md 2>/dev/null || echo "MISSING"`
- Severity Classification: !`ls SEVERITY-CLASSIFICATION.md 2>/dev/null || echo "MISSING"`
- Success Metrics: !`ls SUCCESS-METRICS.md 2>/dev/null || echo "MISSING"`
- Runbook Template: !`ls RUNBOOK-TEMPLATE.md 2>/dev/null || echo "MISSING"`
- README: !`ls README.md 2>/dev/null || echo "MISSING"`
- Changelog: !`ls CHANGELOG.md 2>/dev/null || echo "MISSING"`
- Infographic: !`ls .github/readme-infographic.svg 2>/dev/null || echo "MISSING"`
- Social Preview: !`ls .github/social-preview.svg 2>/dev/null || echo "MISSING"`

## Validation Checks

For all artifacts that exist, check:

### 1. Completeness
- List which artifacts are present vs missing
- For each present artifact, verify it has content (not just headers)

### 2. Internal Link Integrity
- Grep all markdown files for internal links `[text](file.md)` and verify targets exist
- Check that `.github/*.svg` files referenced in README exist

### 3. Cross-Reference Consistency
- Phase names must be identical across Project Plan, JIRA, Gantt, RACI
- Role names must match across RACI and Risk Register owners
- Epic IDs must match between JIRA Structure and Gantt Chart
- Metrics referenced in Project Plan must appear in Success Metrics

### 4. Terminology Consistency
- Read the terminology dictionary from `_research/DOMAIN-RESEARCH.md`
- Verify key terms are used consistently across all artifacts

### 5. Mermaid Diagram Syntax
- Find all mermaid code blocks and check for basic syntax validity

## Output

Produce a validation report with:
- **Completeness**: X/16 artifacts present
- **Link integrity**: X broken links found (list each)
- **Consistency issues**: List each mismatch with file locations
- **Terminology issues**: List any inconsistent term usage
- **Overall status**: PASS / FAIL with issue count
