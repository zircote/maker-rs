---
name: readme-generator
description: Generate a professional GitHub README with status badges, visual overview infographic, Mermaid diagrams, artifact navigation, team information, and timeline summary. Use as the final documentation step after all artifacts are complete.
allowed-tools: Read, Grep, Glob
---

Generate `README.md` at the repository root using data from all completed artifacts.

## Process

1. Read all completed artifacts to extract summary data
2. Create status badges (shields.io format)
3. Embed infographic with dark mode support via `<picture>` element
4. Build Mermaid phase diagram
5. Create complete artifact index with links
6. Summarize team roles and milestones

## Required Sections

1. **Title + Badges**: Project name, status/duration/target/ROI badges
2. **One-line description** with key metric
3. **Infographic**: `<picture>` element with dark mode support referencing `.github/readme-infographic.svg`
4. **Problem**: 2-3 sentences with metrics table (Metric | Current | Target | Improvement)
5. **Approach**: Mermaid `graph LR` showing phases, color-coded
6. **Domain-Specific Key Section**: Top items table relevant to the problem
7. **Financial Impact**: Table (Category | Value) with savings, cost, ROI multiple
8. **Project Artifacts**: Table (Document | Description) linking all .md files
9. **Team**: Table (Role | Responsibility)
10. **Timeline**: Table (Milestone | Week | Deliverable)

## Constraints

- Under 150 lines total
- Scannable without scrolling (key info above fold)
- All internal links must resolve
- No placeholder content

## Quality Criteria

- [ ] Status badges present
- [ ] Infographic with dark mode support
- [ ] Mermaid phase diagram renders
- [ ] Complete artifact index with working links
- [ ] Under 150 lines
