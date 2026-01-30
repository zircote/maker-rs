---
description: Generate README, changelog, infographics, and social preview images
allowed-tools: Read, Write, Grep, Glob
---

# Documentation & Visuals

Run the documentation phase of the project planning pipeline. This should be the final step after all other artifacts are complete.

## Prerequisites

Check that core artifacts exist:
- Project plan: !`ls PROJECT-PLAN.md 2>/dev/null || echo "MISSING"`
- JIRA structure: !`ls JIRA-STRUCTURE.md 2>/dev/null || echo "MISSING"`
- Success metrics: !`ls SUCCESS-METRICS.md 2>/dev/null || echo "MISSING"`

## Process

1. If core artifacts are missing, inform the user which commands to run first and stop
2. Run `/infographic-generator` to produce:
   - `.github/readme-infographic.svg`
   - `.github/social-preview.svg`
   - `.github/social-preview-dark.svg`
3. Run `/readme-generator` aggregating all artifacts to produce `README.md`
4. Run `/changelog` documenting all additions to produce `CHANGELOG.md`
5. Verify outputs:
   - All internal links resolve to actual files
   - SVGs render in GitHub markdown (no external deps)
   - Dark mode support via `<picture>` element
   - README under 150 lines

## Output

Report what was generated: artifact count linked in README, SVG files created, and any broken links found.
