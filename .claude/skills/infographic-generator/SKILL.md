---
name: infographic-generator
description: Create visual project summary infographics for the README and GitHub social preview images with before/after comparison, ROI highlights, and phase timeline. Use after core artifacts are complete.
user-invocable: false
allowed-tools: Read, Grep, Glob, Write
---

Generate visual summary SVGs. Invoked by `documentation-architect` and `project-architect`.

## Output Files

1. `.github/readme-infographic.svg` (800x420px, light theme)
2. `.github/social-preview.svg` (1280x640px, light theme)
3. `.github/social-preview-dark.svg` (1280x640px, dark theme)

## Infographic Layout (800x420)

1. **Section 1** (top-left): Before/After comparison boxes
   - "Before" box: red theme (#fef2f2 bg, #fca5a5 border), key metrics in red
   - Arrow with improvement percentage in green
   - "After" box: green theme (#f0fdf4 bg, #86efac border), key metrics in green
2. **Section 2** (top-right): ROI box (#eff6ff bg, #93c5fd border) with headline figures
3. **Section 3** (middle): 4-segment phase timeline bar with milestone markers
4. **Section 4** (bottom-left): 4 phase cards with descriptions
5. **Section 5** (bottom-right): Key data table

## Social Preview (1280x640)

- Project title and subtitle
- Key metrics highlights
- Phase visualization
- Organic illustrated background style (waves, circles, blobs)
- Light and dark theme variants

## Color System

| Element | Light | Dark |
|---------|-------|------|
| Background | #f8fafc → #e2e8f0 | #0f172a → #1e293b |
| Text primary | #0f172a | #f8fafc |
| Card bg | #ffffff | #1e293b |
| Success | #10b981 | #34d399 |
| Danger | #ef4444 | #f87171 |

## Post-Write Validation (MANDATORY)

After writing each SVG, run validation and fix any errors before proceeding. See [SVG Validation](../../docs/SVG-VALIDATION.md) for the full procedure and common pitfalls.

```bash
cd workspace && node validate-svg.js ../.github/readme-infographic.svg ../.github/social-preview.svg ../.github/social-preview-dark.svg
```

## Quality Criteria

- [ ] **XML validates without errors** (`npm run validate-svg` passes)
- [ ] GitHub markdown rendering (no external deps)
- [ ] Dark mode support via `<picture>` element
- [ ] Key metrics visible without zooming
- [ ] Phase colors consistent with Gantt chart
- [ ] WCAG AA contrast ratios
