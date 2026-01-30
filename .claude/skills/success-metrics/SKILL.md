---
name: success-metrics
description: Define comprehensive KPIs with measurement methods, baselines, targets, calculation formulas, dashboard specifications, and project closure criteria. Use after project plan defines goals.
allowed-tools: Read, Grep, Glob, WebSearch
---

Generate `SUCCESS-METRICS.md` at the repository root.

## Process

1. Read `PROJECT-PLAN.md` for goals, targets, ROI data
2. Read `_research/DOMAIN-RESEARCH.md` for industry benchmarks and standard KPIs
3. Define 8-15 KPIs (primary + secondary)
4. Establish baselines from data or industry averages
5. Set phased targets
6. Design dashboard layout with widget specifications
7. Define project closure criteria

## Required Sections

1. **Target Achievement Timeline**: Table (Metric | Baseline | Phase 1-4 targets | Final Target)
2. **Primary Metrics** (6-8, each containing):
   - Attributes table: Definition, Unit, Baseline, Target, Improvement %, Measurement Method, Data Source, Frequency, Formula, Dashboard Widget, Owner
3. **Secondary Metrics** (4-6, same structure)
4. **Financial Metrics**: Cost of current state table and projected savings table
5. **Dashboard Specification**: 4 sections (Executive Summary, Trend Analysis, Detailed Breakdown, Project Progress) with widget types and refresh rates
6. **Data Sources**: Table (Source | Type | Update Frequency | Access Method)
7. **Reporting Cadence**: Table (Report | Audience | Frequency | Format | Owner)
8. **Project Closure Criteria**: Numbered table (Criterion | Measurement | Pass Threshold), 8-12 items

## Quality Criteria

- [ ] 8-15 total KPIs with formulas and data sources
- [ ] Mix of leading and lagging indicators
- [ ] Financial model with 3-year projection
- [ ] Dashboard layout with 4 sections
- [ ] Reporting cadence defined
- [ ] 8-12 project closure criteria
