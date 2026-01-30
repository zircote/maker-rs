# Project Planning Template

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Claude Code](https://img.shields.io/badge/Claude-Code-blueviolet)](https://claude.ai/claude-code)
[![Artifacts](https://img.shields.io/badge/Artifacts-16-orange)](./docs/COMMAND-GUIDE.md)
[![Agents](https://img.shields.io/badge/Agents-7-blue)](./docs/COMMAND-GUIDE.md)
[![Skills](https://img.shields.io/badge/Skills-17-green)](./docs/COMMAND-GUIDE.md)
[![Commands](https://img.shields.io/badge/Commands-10-brightgreen)](./docs/COMMAND-GUIDE.md)

> Generate comprehensive, professional project planning artifacts for any problem domain using Claude agents and skills.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../.github/readme-infographic.svg">
  <source media="(prefers-color-scheme: light)" srcset="../.github/readme-infographic.svg">
  <img alt="Project Planning Template - How It Works" src="../.github/readme-infographic.svg" width="100%">
</picture>

## What This Template Produces

From a single problem statement, this template generates a complete set of project planning artifacts:

| Artifact | Description |
|----------|-------------|
| **Domain Research** | Industry analysis, frameworks, terminology dictionary |
| **Project Plan** | Executive summary, phased approach, ROI analysis |
| **Gantt Chart** | SVG timeline with critical path and dependencies |
| **JIRA Structure** | Epic/story hierarchy with sprint planning |
| **Dependencies** | Critical path analysis and blocking relationships |
| **RACI Chart** | Responsibility assignment matrix |
| **Risk Register** | Risk assessment with mitigation strategies |
| **Severity Classification** | Priority framework with decision tree |
| **Best Practices** | Industry standards catalog with citations |
| **Success Metrics** | KPIs, dashboard specs, closure criteria |
| **Runbook Template** | Operational procedure template with examples |
| **README** | GitHub landing page with badges and infographic |
| **Changelog** | Version history in Keep a Changelog format |
| **Executive Briefing** | HTML slide sources assembled into PowerPoint deck |
| **Visual Assets** | SVG infographics, social previews, Gantt chart |
| **Execution Plan** | Persistent plan files for async session execution |

## Quick Start

### 1. Create a Repository from This Template

Click **"Use this template"** on GitHub.

### 2. Define Your Project Context

**Option A: Interactive Interview (Recommended)**

Run the guided elicitation in Claude Code:
```text
/project-context
```

**Option B: Manual Creation**

Copy the template to the repository root and fill it in:
```bash
cp .claude/templates/PROJECT-CONTEXT.md PROJECT-CONTEXT.md
```

Then edit `PROJECT-CONTEXT.md` with your project details (problem statement, domain, stakeholders, constraints, available data, desired outcomes).

### 3. Generate All Artifacts

Open Claude Code in the repository and run:

```text
/project-architect Generate complete project planning artifacts based on PROJECT-CONTEXT.md
```

Or generate phases individually:
- `/research` - Domain research + best practices
- `/plan` - Project plan + JIRA structure + dependencies
- `/governance` - RACI + risk register + severity classification
- `/timeline` - Gantt chart + SVG visualization
- `/metrics` - KPIs + runbook + executive briefing
- `/docs` - README + changelog + infographics
- `/validate` - Cross-artifact consistency check

Or generate single artifacts:
- `/project-context` - Interactive context elicitation
- `/domain-research` - Industry analysis and terminology
- `/project-plan` - Master project plan
- `/gantt-chart` - Timeline visualization
- `/jira-structure` - Work breakdown structure
- `/raci-chart` - Responsibility matrix
- `/risk-register` - Risk management
- `/severity-classification` - Priority framework
- `/best-practices` - Industry standards catalog
- `/success-metrics` - KPI definitions
- `/runbook-template` - Operational procedures
- `/readme-generator` - GitHub landing page
- `/changelog` - Version history
- `/executive-briefing` - Slide deck generation
- `/create-execution-plan` - Detailed execution plan for async sessions

See [Command Guide](./docs/COMMAND-GUIDE.md) for full reference with examples.

## Architecture

```text
.claude/
├── agents/                              # Orchestrating agents
│   ├── project-architect.md             # Master orchestrator
│   ├── context-elicitor.md              # Project scope interview
│   ├── research-specialist.md           # Domain research
│   ├── timeline-architect.md            # Scheduling
│   ├── governance-architect.md          # RACI, risk, severity
│   ├── metrics-architect.md             # KPIs, exec briefing
│   └── documentation-architect.md       # README, visuals
├── commands/                            # Batch workflow commands
│   ├── generate-all.md                  # Full pipeline
│   ├── create-execution-plan.md         # Async execution plan
│   ├── research.md                      # Domain research phase
│   ├── plan.md                          # Planning phase
│   ├── governance.md                    # Governance phase
│   ├── timeline.md                      # Timeline phase
│   ├── metrics.md                       # Metrics phase
│   ├── docs.md                          # Documentation phase
│   ├── validate.md                      # Artifact validation
│   └── project-context.md               # Context elicitation
├── templates/                           # Source templates
│   └── PROJECT-CONTEXT.md               # Template for project context
├── skills/                              # Individual artifact generators
│   ├── best-practices/SKILL.md          # Industry standards
│   ├── changelog/SKILL.md               # Version history
│   ├── dependency-analyzer/SKILL.md     # Critical path
│   ├── domain-research/SKILL.md         # Industry research
│   ├── executive-briefing/SKILL.md      # Slide deck
│   ├── gantt-chart/SKILL.md             # Timeline
│   ├── infographic-generator/SKILL.md   # Visual summaries
│   ├── jira-structure/SKILL.md          # Work breakdown
│   ├── project-context/SKILL.md         # Context elicitation
│   ├── project-plan/SKILL.md            # Master plan
│   ├── raci-chart/SKILL.md              # Responsibilities
│   ├── readme-generator/SKILL.md        # GitHub landing page
│   ├── risk-register/SKILL.md           # Risk management
│   ├── runbook-template/SKILL.md        # Operational procedures
│   ├── severity-classification/SKILL.md # Priority framework
│   ├── success-metrics/SKILL.md         # KPIs
│   └── svg-gantt-generator/SKILL.md     # SVG visualization
└── CLAUDE.md                            # Project instructions
```

### Execution Flow

```text
Context Elicitation (project-context)
    ↓
Research (domain-research, best-practices)
    ↓
Project Plan → Dependencies
    ↓
┌──────────────┬──────────────┬────────────┬──────────────┐
│ Timeline     │ Governance   │ Metrics    │ Operations   │
│ gantt-chart  │ raci-chart   │ success-   │ runbook-     │
│ jira-struct  │ risk-reg     │ metrics    │ template     │
│ svg-gantt    │ severity     │            │              │
└──────────────┴──────────────┴────────────┴──────────────┘
    ↓
Documentation (readme, changelog, infographic)
    ↓
Executive Briefing
```

## Domain Adaptability

The skills are industry-agnostic and adapt based on domain research:

- **Technology / SaaS**: Google SRE, ITIL, DORA, AWS Well-Architected
- **Healthcare**: HIPAA, HL7 FHIR, HITRUST, Joint Commission
- **Financial Services**: PCI DSS, SOX, Basel III, FFIEC
- **Manufacturing**: Lean Six Sigma, ISO 9001, TPM, Industry 4.0
- **General Business**: PMI PMBOK, PRINCE2, SAFe, OKR frameworks

## Exemplar

This template was derived from [project-sre](https://github.com/hmhco/project-sre) (private repository), a PagerDuty Incident Noise Reduction initiative demonstrating gold-standard project planning with:
- 14 epics, 60+ stories across 4 phases
- 10 industry frameworks cited
- $279K annual ROI with 11.6x return
- Hand-crafted SVG Gantt chart and infographics
- 8-slide executive briefing deck

## License

MIT
