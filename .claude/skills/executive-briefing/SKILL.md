---
name: executive-briefing
description: Generate an executive-ready slide deck as HTML slide sources with Node.js build scripts for PowerPoint assembly. Use as the final deliverable after all artifacts are complete.
allowed-tools: Read, Grep, Glob, Write, Bash
---

Generate the executive briefing deck in `workspace/`.

## Process

1. Read all completed artifacts for key data
2. Create 8 HTML slide sources in `workspace/slides/`
3. Ensure `workspace/generate-assets.js` and `workspace/build-deck.js` are configured
4. Update `workspace/package.json` with project name

## Slide Structure (8 slides)

| # | File | Content | Background |
|---|------|---------|------------|
| 1 | slide1-title.html | Project name, key metric, date range | Dark gradient |
| 2 | slide2-problem.html | Current state metrics, pain points | Light |
| 3 | slide3-before-after.html | Side-by-side comparison, improvement arrow | Light |
| 4 | slide4-phases.html | 4 phase cards, timeline bar, milestones | Light |
| 5 | slide5-roi.html | Bar chart placeholder (built programmatically) | Light |
| 6 | slide6-metrics.html | Primary KPIs with targets | Light |
| 7 | slide7-risks.html | Top 5 risks with mitigation | Light |
| 8 | slide8-next-steps.html | Actions, budget ask, contacts | Blue accent |

## HTML Slide Template

Each slide: 720pt x 405pt (16:9), font-family Segoe UI/Arial, with header/content/footer sections.

## Design System

| Element | Value |
|---------|-------|
| Title font | 22-30pt bold |
| Body font | 11-14pt |
| Caption | 7-9pt |
| Phase 1 | #3b82f6 |
| Phase 2 | #10b981 |
| Phase 3 | #f59e0b |
| Phase 4 | #8b5cf6 |

## Quality Criteria

- [ ] 8 slides covering all executive topics
- [ ] Consistent design system
- [ ] ROI chart generated programmatically in build-deck.js
- [ ] Build scripts produce valid PPTX
- [ ] Executive-appropriate language
