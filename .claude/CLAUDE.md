# Project Planning Template - CLAUDE.md

## Overview

This repository is a **GitHub template** for generating comprehensive, professional project planning artifacts for ANY problem domain. It uses Claude agents and skills to research the domain and produce industry-appropriate documentation.

## How to Use

### Step 1: Create Repository from Template
Click "Use this template" on GitHub to create your project repository.

### Step 2: Define Project Context

**Option A: Interactive Interview (Recommended)**

Run the guided elicitation in Claude Code:
```text
/project-context
```
This walks you through a structured interview covering problem statement, domain, stakeholders, constraints, available data, and desired outcomes.

**Option B: Manual Creation**

Copy the template to the repository root and fill it in:
```bash
cp .claude/templates/PROJECT-CONTEXT.md PROJECT-CONTEXT.md
```

Then edit `PROJECT-CONTEXT.md` with your project details (problem statement, domain, stakeholders, constraints, available data, desired outcomes).

### Step 3: Generate All Artifacts
Invoke the master orchestrator:

```text
/project-architect Generate complete project planning artifacts based on PROJECT-CONTEXT.md
```

Or generate individual artifacts:
```text
/project-context
/domain-research
/project-plan
/gantt-chart
/jira-structure
/raci-chart
/risk-register
/severity-classification
/best-practices
/success-metrics
/runbook-template
/readme-generator
/changelog
/executive-briefing
```

> **Note:** The skills `svg-gantt-generator`, `infographic-generator`, and `dependency-analyzer` are internal — they are invoked by agents and other skills, not directly by users.

## Agent Directory

| Agent | Purpose | Use When |
|-------|---------|----------|
| `project-architect` | Master orchestrator - full artifact set | Starting a new project |
| `context-elicitor` | Interactive context interview | Need help defining project scope |
| `research-specialist` | Domain research and best practices | Need industry context |
| `timeline-architect` | Gantt, JIRA, dependencies | Planning schedule |
| `governance-architect` | RACI, risk, severity | Setting up governance |
| `metrics-architect` | KPIs, dashboards, exec briefing | Defining measurement |
| `documentation-architect` | README, changelog, infographics | Final documentation |

## Generated Artifacts

After execution, you will have:

| Artifact | File | Purpose |
|----------|------|---------|
| Domain Research | `_research/DOMAIN-RESEARCH.md` | Industry analysis, frameworks, terminology |
| Master Plan | `PROJECT-PLAN.md` | Executive summary, phases, ROI |
| Gantt Chart | `GANTT-CHART.md` + `.github/gantt-chart.svg` | Visual timeline |
| Work Breakdown | `JIRA-STRUCTURE.md` | Epics, stories, sprints |
| Dependencies | `_research/DEPENDENCY-ANALYSIS.md` | Critical path analysis |
| RACI | `RACI-CHART.md` | Responsibility matrix |
| Risks | `RISK-REGISTER.md` | Risk management |
| Severity | `SEVERITY-CLASSIFICATION.md` | Priority framework |
| Best Practices | `BEST-PRACTICES.md` | Industry standards |
| Metrics | `SUCCESS-METRICS.md` | KPIs and measurement |
| Runbooks | `RUNBOOK-TEMPLATE.md` | Operational procedures |
| README | `README.md` | GitHub landing page |
| Changelog | `CHANGELOG.md` | Version history |
| Visual Assets | `.github/readme-infographic.svg`, `.github/social-preview.svg` | SVG infographics and social previews |
| Exec Deck | `workspace/*.pptx` | Executive briefing |
| Execution Plan | `_plan/*.md` + `_plan/*.json` | Persistent plan for future sessions |

## Quality Standards

All artifacts MUST:
- Use consistent terminology from the domain research
- Include quantified metrics with baselines and targets
- Reference industry frameworks with citations
- Contain Mermaid diagrams or SVG visualizations
- Cross-reference related artifacts
- Match the exemplar quality standard ([project-sre](https://github.com/hmhco/project-sre) — private repository)

## Build Commands

```bash
cd workspace && npm install
node validate-svg.js      # Validate all .github/*.svg files
node generate-assets.js   # Generate PNG assets for slides
node build-deck.js        # Build executive briefing deck
```

## Customization

- **Industry sources**: Edit `.claude/skills/domain-research/SKILL.md` adaptation rules
- **Visual branding**: Edit `.claude/skills/svg-gantt-generator/SKILL.md` color scheme
- **Artifact structure**: Edit individual skill templates in `.claude/skills/`
- **Agent workflows**: Edit orchestration in `.claude/agents/`
