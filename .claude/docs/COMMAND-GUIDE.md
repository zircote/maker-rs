# Command Guide

How to use the project planning commands to generate artifacts for any domain.

## Command Reference

| Command | What It Does | Prerequisites |
|---------|-------------|---------------|
| `/generate-all` | Full pipeline — all phases, all artifacts | `PROJECT-CONTEXT.md` filled in |
| `/create-execution-plan` | Write a detailed execution plan to files for a future session | `/plan` completed (needs `JIRA-STRUCTURE.md`) |
| `/research` | Domain research + industry best practices | `PROJECT-CONTEXT.md` filled in |
| `/plan` | Project plan + JIRA structure + dependencies | `/research` completed |
| `/governance` | RACI chart + risk register + severity levels | `/research` + `/plan` completed |
| `/timeline` | Gantt chart + SVG visualization | `/plan` completed |
| `/metrics` | KPIs + runbook + executive briefing deck | `/research` + `/plan` completed |
| `/docs` | README + changelog + infographics | All above completed |
| `/validate` | Check consistency across all artifacts | Any artifacts present |

## Workflow

```text
/research ──► /plan ──► /governance ──► /docs ──► /validate
                    ├──► /timeline  ──┘
                    └──► /metrics   ──┘
```

Run them individually in order, or use `/generate-all` to execute the full pipeline automatically.

## Quick Start

### One Command — Everything

```text
/generate-all
```

This reads the populated `PROJECT-CONTEXT.md` at the repository root and produces every artifact in dependency order. Use this when you want the complete set without manual steps.

### Plan Now, Execute Later

```text
/create-execution-plan
```

This writes a comprehensive execution plan to persistent files — no artifacts are generated. A future Claude session can load the plan and execute against it.

**Files created:**

| File | Purpose |
|------|---------|
| `_plan/EXECUTION-PLAN.md` | Full task specifications with inputs, outputs, acceptance criteria, domain-tailored guidance |
| `_plan/TASK-MANIFEST.json` | Machine-readable JSON of all stories with dependencies |
| `_plan/RUNSHEET.md` | Operator checklist — work through tasks ticking off criteria |
| `_plan/CONTEXT-BRIEF.md` | Quick-start briefing for new sessions |

**To execute in a future session:**

```text
Read _plan/CONTEXT-BRIEF.md then _plan/EXECUTION-PLAN.md and begin the work
```

Or for manual control:

```text
Read _plan/RUNSHEET.md and work through the checklist
```

The plan is tailored to your project context — it references your actual stakeholders, compliance requirements, technology stack, and metrics from `PROJECT-CONTEXT.md`.

### Step by Step

For more control, run phases individually:

```text
/research                     # Domain analysis + frameworks
/plan                         # Project plan + work breakdown
/governance                   # RACI + risks + severity
/timeline                     # Gantt chart + SVG
/metrics                      # KPIs + exec briefing
/docs                         # README + visuals + changelog
/validate                     # Check everything is consistent
```

This lets you review and adjust artifacts between phases.

### Targeted Regeneration

Already have most artifacts but need to redo one phase?

```text
/governance                   # Regenerate just governance artifacts
/timeline                     # Regenerate just the Gantt chart
```

Each command checks its prerequisites and tells you what's missing.

## Examples by Project Type

### SaaS Platform Migration

**PROJECT-CONTEXT.md:**

```markdown
## Problem Statement
Our monolithic Rails application serves 12,000 daily active users but deployment
frequency has dropped to once per month due to coupled components and a 45-minute
test suite. Customer-facing incidents increased 300% year-over-year.

## Domain / Industry
SaaS Operations - Platform Engineering

## Stakeholders
- VP Engineering: Executive sponsor, budget authority
- Platform Team Lead: Technical owner, architecture decisions
- Product Manager: Feature prioritization, customer impact assessment
- QA Lead: Test strategy, quality gates
- DevOps Engineer: CI/CD pipeline, infrastructure

## Constraints
- Budget: $180K (tooling, infrastructure, contractor support)
- Timeline: 9 months (3 quarters)
- Team size: 6 engineers + 1 contractor
- Compliance: SOC 2 Type II (audit in Q3)
- Technology: Ruby on Rails, PostgreSQL, AWS ECS, GitHub Actions

## Available Data
- 18 months of deployment frequency data from GitHub
- Incident history from PagerDuty (450+ incidents)
- AWS cost reports showing $23K/month infrastructure spend
- Customer churn data correlating with incident frequency

## Desired Outcomes
- Deploy daily (from monthly) — 30x improvement
- Reduce incidents by 70% (from 15/week to <5/week)
- Cut test suite from 45 minutes to under 10 minutes
- Achieve SOC 2 compliance with automated evidence collection
```

**Workflow:**

```text
/research "SaaS platform migration microservices"
# Produces: Domain research citing Strangler Fig Pattern, DORA metrics,
# AWS Well-Architected Framework, 12-Factor App methodology

/plan
# Produces: 4-phase plan (Assess → Extract → Migrate → Optimize),
# 14 epics, 58 stories, $540K projected annual savings

/governance
# Produces: RACI with 8 roles, 12 risks including "data migration
# corruption" and "SOC 2 audit timing conflict"

/timeline
# Produces: 36-week Gantt chart, critical path through auth service extraction

/metrics
# Produces: DORA metrics (deployment frequency, lead time, MTTR, change failure rate),
# 8-slide executive deck with ROI chart

/docs
# Produces: README with architecture diagram, infographic comparing before/after

/validate
# Checks all cross-references, confirms epic IDs match across JIRA and Gantt
```

---

### Healthcare Clinical Workflow

**PROJECT-CONTEXT.md:**

```markdown
## Problem Statement
Nurses spend 2.3 hours per 12-hour shift on manual medication reconciliation
across three disconnected systems (EHR, pharmacy, bedside scanner). This contributes
to a 4.2% medication discrepancy rate — above the 2% Joint Commission threshold.

## Domain / Industry
Healthcare IT - Clinical Informatics

## Stakeholders
- Chief Nursing Officer: Executive sponsor
- Clinical Informatics Director: Technical owner
- Pharmacy Director: Medication safety, formulary management
- IT Security Officer: HIPAA compliance, access controls
- Charge Nurses (3 units): End-user representatives

## Constraints
- Budget: $320K (EHR customization, integration middleware, training)
- Timeline: 12 months (regulatory deadline for Joint Commission survey)
- Team size: 4 clinical informaticists + 2 integration developers
- Compliance: HIPAA, Joint Commission NPSG.03.06.01, FDA 21 CFR Part 11
- Technology: Epic EHR, Omnicell pharmacy system, Zebra barcode scanners

## Available Data
- 6-month audit of medication reconciliation errors (1,200 events)
- Time-motion study of nursing workflow (42 observed shifts)
- Current BCMA (Barcode-Assisted Medication Administration) scan rates
- Joint Commission preliminary survey findings

## Desired Outcomes
- Reduce medication discrepancy rate from 4.2% to <1.5%
- Cut reconciliation time from 2.3 hours to <45 minutes per shift
- Achieve 99.5% BCMA scan compliance
- Pass Joint Commission survey with zero medication safety findings
```

**Workflow:**

```text
/research "healthcare clinical workflow medication safety"
# Produces: Research citing Joint Commission NPSG, ISMP guidelines,
# HIMSS EMRAM model, Leapfrog Group safety standards

/plan
# Produces: 4-phase plan (Assess → Integrate → Pilot → Scale),
# 11 epics including "HL7 FHIR interface development" and
# "clinical decision support rules engine"

/governance
# Produces: RACI with clinical and IT roles, 15 risks including
# "EHR downtime during go-live" and "HIPAA breach during data migration",
# severity classification aligned to patient safety impact

/metrics
# Produces: Clinical KPIs (medication error rate, BCMA compliance,
# nurse satisfaction score), FDA-compliant audit trail metrics
```

---

### Manufacturing Process Optimization

**PROJECT-CONTEXT.md:**

```markdown
## Problem Statement
Our automotive parts production line (Line 7) has an OEE (Overall Equipment
Effectiveness) of 62%, significantly below the 85% world-class benchmark.
Unplanned downtime averages 47 hours/month, costing $28K per hour in lost
production — $1.3M monthly.

## Domain / Industry
Manufacturing - Lean Operations / Industry 4.0

## Stakeholders
- Plant Manager: Executive sponsor, P&L responsibility
- Production Manager: Line 7 operations, shift scheduling
- Maintenance Director: Predictive maintenance, spare parts
- Quality Manager: SPC, defect reduction, ISO 9001
- Controls Engineer: PLC programming, SCADA integration

## Constraints
- Budget: $450K (sensors, edge computing, software licenses)
- Timeline: 6 months (before Q4 production surge)
- Team size: 3 engineers + 2 maintenance technicians + 1 data analyst
- Compliance: ISO 9001:2015, IATF 16949, OSHA machine safety
- Technology: Siemens S7-1500 PLCs, Ignition SCADA, SAP ERP

## Available Data
- 12 months of OEE data by shift (availability, performance, quality)
- CMMS work order history (3,200 maintenance events)
- SPC data for 8 critical-to-quality dimensions
- Energy consumption logs from smart meters

## Desired Outcomes
- Increase OEE from 62% to 82% (world-class threshold)
- Reduce unplanned downtime by 60% (47 hrs → <19 hrs/month)
- Achieve predictive maintenance coverage for top 10 failure modes
- Save $780K/month in recovered production capacity
```

**Workflow:**

```text
/research "manufacturing OEE improvement predictive maintenance"
# Produces: Research citing TPM (Total Productive Maintenance),
# Lean Six Sigma DMAIC, ISA-95 automation standard, MTTR/MTBF benchmarks

/plan "Prioritize quick wins in first 8 weeks"
# Produces: 3-phase plan (Quick Wins → Predictive → Optimize),
# 12 epics, ROI showing $9.4M annual recovery at 82% OEE

/governance
# Produces: RACI mapping maintenance and production roles,
# 10 risks including "sensor installation production disruption"
# and "false positive predictive alerts", severity aligned to
# production impact (units/hour lost)

/timeline
# Produces: 24-week Gantt, critical path through PLC integration
# and ML model training phases

/metrics
# Produces: OEE components (availability, performance, quality rate),
# MTBF/MTTR targets, predictive accuracy thresholds,
# executive deck with monthly savings waterfall chart
```

---

### Financial Services Compliance Program

**PROJECT-CONTEXT.md:**

```markdown
## Problem Statement
Our fintech lending platform processes $2.1B annually but our compliance
monitoring is manual, covering only 12% of transactions. Recent OCC examination
identified 3 MRAs (Matters Requiring Attention) related to BSA/AML monitoring
gaps and fair lending analysis deficiencies.

## Domain / Industry
Financial Services - Regulatory Compliance / RegTech

## Stakeholders
- Chief Compliance Officer: Executive sponsor, regulatory liaison
- BSA Officer: Anti-money laundering program owner
- Fair Lending Officer: ECOA/HMDA compliance
- CTO: Technology platform, data architecture
- Internal Audit Director: Testing, evidence management

## Constraints
- Budget: $600K (compliance technology, model development, staffing)
- Timeline: 8 months (OCC follow-up examination in 9 months)
- Team size: 2 compliance analysts + 3 engineers + 1 data scientist
- Compliance: BSA/AML, ECOA, HMDA, UDAAP, TILA, OCC Heightened Standards
- Technology: Python/Airflow data pipeline, Snowflake, Looker dashboards

## Available Data
- 24 months of loan application data (180K applications)
- Current SAR filing history and investigation logs
- OCC examination report with MRA details
- Existing rule-based alert system generating 4,200 alerts/month (92% false positive)

## Desired Outcomes
- Achieve 100% transaction monitoring coverage (from 12%)
- Reduce false positive rate from 92% to <30%
- Resolve all 3 MRAs before follow-up examination
- Establish automated fair lending regression testing
```

**Workflow:**

```text
/research "financial services BSA AML compliance monitoring"
# Produces: Research citing FFIEC BSA/AML Manual, OCC Heightened Standards,
# FinCEN guidance, ECOA Regulation B, model risk management SR 11-7

/plan
# Produces: 4-phase plan (Remediate MRAs → Enhance Monitoring →
# Fair Lending → Sustain), 16 epics, projected penalty avoidance
# value of $2-5M

/governance
# Produces: RACI with three lines of defense mapped, 18 risks including
# "model bias in fair lending analysis" and "SAR filing deadline breach",
# severity classification: Regulatory (immediate), Compliance (48-hour),
# Operational (5-day), Enhancement (sprint)

/validate
# Critical for this domain — verifies regulatory framework references
# are consistent across all artifacts
```

---

### Open Source Developer Tool

**PROJECT-CONTEXT.md:**

```markdown
## Problem Statement
Developers using our CLI tool spend an average of 23 minutes configuring
new projects because the setup wizard asks 47 questions, most with sensible
defaults. Our GitHub issues show 340 open feature requests and the contributor
drop-off rate is 89% after first PR due to unclear architecture.

## Domain / Industry
Developer Tooling - Open Source Community

## Stakeholders
- Project Maintainer: Architecture decisions, release management
- Core Contributors (5): Feature development, code review
- Community Manager: Issue triage, contributor onboarding
- Documentation Lead: Guides, API reference, tutorials

## Constraints
- Budget: $0 (open source, volunteer time only)
- Timeline: 4 months (target v2.0 release at conference)
- Team size: 1 maintainer + 5 core contributors (part-time, ~10 hrs/week each)
- Compliance: MIT License, Semantic Versioning, REUSE compliance
- Technology: Rust, TOML config, GitHub Actions CI, docs via mdBook

## Available Data
- GitHub analytics (stars, forks, clone traffic, issue velocity)
- CLI telemetry (opt-in): command usage frequency, error rates
- Contributor survey results (42 responses)
- Benchmark data: setup time, binary size, command latency

## Desired Outcomes
- Reduce project setup from 23 minutes to <3 minutes (smart defaults)
- Cut contributor onboarding from "weeks" to "first PR in one session"
- Ship v2.0 with 50% of top-voted feature requests resolved
- Grow monthly active contributors from 5 to 15
```

**Workflow:**

```text
/research "developer tooling CLI open source community"
# Produces: Research citing Rust API Guidelines, Conventional Commits,
# GitHub Community Health metrics, InnerSource patterns

/plan "Focus on contributor experience and smart defaults"
# Produces: 3-phase plan (Quick Wins → v2.0 Core → Community),
# 8 epics focused on DX improvements, architecture docs, plugin system

/governance
# Produces: Lightweight RACI (4 roles), 8 risks including
# "key contributor burnout" and "breaking change migration pain"

/metrics
# Produces: Community health KPIs (time-to-first-PR, issue close rate,
# contributor retention), binary size and latency benchmarks

/docs
# Produces: README with contributor quickstart, architecture overview,
# infographic showing v1 → v2 improvements
```

## Tips

**Iterate, don't restart.** After generating artifacts, edit them directly and re-run individual phases. Changed the project scope? Run `/plan` again — it reads the updated `PROJECT-CONTEXT.md`.

**Provide data.** The more quantitative data in `PROJECT-CONTEXT.md`, the more specific the artifacts. Vague inputs produce generic outputs. The template source is at `.claude/templates/PROJECT-CONTEXT.md`; the populated file lives at the repository root.

**Use arguments.** Some commands accept focus hints:

```text
/research "edge computing IoT"       # Focus the research scope
/plan "Prioritize security first"    # Influence phase structure
```

**Validate often.** Run `/validate` after any manual edits to catch inconsistencies before they compound.

**Phase commands are independent.** Need to redo just governance? Run `/governance`. It checks its prerequisites and only regenerates its own artifacts.
