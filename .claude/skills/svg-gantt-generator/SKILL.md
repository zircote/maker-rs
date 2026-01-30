---
name: svg-gantt-generator
description: Generate a programmatic SVG Gantt chart with phase-coded colors, task bars, sub-tasks, milestones, and dark-theme design. Use when creating or updating the visual timeline.
user-invocable: false
allowed-tools: Read, Grep, Glob, Write
---

Generate `.github/gantt-chart.svg` from structured timeline data.

This skill is invoked by `gantt-chart` and `project-architect`, not directly by users.

## SVG Specification

### Layout
- **Canvas**: 1280px wide, height = 120 + (epic_count x 60) + 80
- **Background**: #0f172a (slate-950)
- **Left margin**: 200px for epic labels
- **Week columns**: (1280 - 200) / total_weeks per column

### Required Elements

1. **Gradient definitions** for each phase and sub-task variant:
   - Phase 1: #3b82f6 → #60a5fa (blue)
   - Phase 2: #10b981 → #34d399 (green)
   - Phase 3: #f59e0b → #fbbf24 (amber)
   - Phase 4: #8b5cf6 → #a78bfa (purple)
   - Sub-task variants at 70% opacity with lighter shades

2. **Background**: Full-width rect with #0f172a fill, 12px border radius

3. **Title**: Centered at y=40, #f8fafc, 20px bold

4. **Week header row**: Week labels (W1, W2, ...) centered in columns

5. **Grid lines**: Vertical lines per week at 10% opacity (#334155)

6. **Phase background bands**: Colored rects at 5% opacity spanning phase weeks

7. **Epic bars**: For each epic:
   - Label at x=-10 (right-aligned), #94a3b8, 11px
   - Main bar: gradient fill, 24px height, 4px border radius
   - Sub-task bars: lighter shade, 16px height, offset below parent
   - Sub-task labels: #cbd5e1, 9px

8. **Milestone diamonds**: `<polygon>` at milestone positions with phase color

9. **Legend**: Phase color key at bottom

### Typography
- Font family: `-apple-system, 'Segoe UI', Arial, sans-serif`
- Title: 20px bold, #f8fafc
- Epic labels: 11px, #94a3b8
- Sub-task labels: 9px, #cbd5e1
- Week headers: 10px, #64748b

## Post-Write Validation (MANDATORY)

After writing the SVG, run validation and fix any errors before proceeding. See [SVG Validation](../../docs/SVG-VALIDATION.md) for the full procedure and common pitfalls.

```bash
cd workspace && node validate-svg.js ../.github/gantt-chart.svg
```

## Quality Criteria

- [ ] **XML validates without errors** (`npm run validate-svg` passes)
- [ ] Renders in GitHub markdown (no external deps)
- [ ] All epics and sub-tasks shown
- [ ] Phase colors consistent
- [ ] Week grid aligned
- [ ] Milestone markers at correct positions
- [ ] Dark theme with accessible contrast
