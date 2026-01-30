---
name: project-context
description: "Conduct a structured interview to elicit project knowledge and produce a complete PROJECT-CONTEXT.md. Use when starting a new project or when PROJECT-CONTEXT.md needs to be created or refined."
argument-hint: "[optional: 'refine' to update existing, or a brief project description to seed the interview]"
allowed-tools: Read, Write, Grep, Glob, AskUserQuestion
---

Guide the user through a structured elicitation interview to produce a complete, high-quality `PROJECT-CONTEXT.md` at the repository root. Use `.claude/templates/PROJECT-CONTEXT.md` as the template structure; the populated result is written to root `PROJECT-CONTEXT.md`.

## Prerequisites

Check if `PROJECT-CONTEXT.md` already exists and contains user content (not just template placeholders). If it contains real content and $ARGUMENTS does not include "refine", ask the user whether they want to:
- **Refine**: Keep existing content and fill gaps
- **Replace**: Start fresh with a new interview

## Interview Protocol

Conduct the interview in **phases**. Each phase targets one section of PROJECT-CONTEXT.md. Use `AskUserQuestion` for structured choices and direct conversation for open-ended elicitation.

**Ground rules for the interview:**
- Ask one phase at a time; do not dump all questions at once
- After each answer, probe for specifics: numbers, names, dates, dollar amounts
- If the user gives a vague answer, rephrase and ask for quantification
- Accept "TBD" or "unknown" gracefully — note it and move on
- Reflect back what you heard before moving to the next phase

### Phase 1: Problem Statement

Elicit the core problem. Ask:
- "What problem are you trying to solve?"
- Probe for **quantified impact**: "How does this problem manifest today? Can you put numbers to it — frequency, cost, hours lost, error rates?"
- Probe for **scope**: "Who is affected? How many users/teams/systems?"

Goal: A 2-4 sentence problem statement with at least one quantified metric.

### Phase 2: Domain / Industry

Use AskUserQuestion to offer common domain categories, then refine:

Suggested options:
- Technology / SaaS
- Healthcare / Life Sciences
- Financial Services / FinTech
- Manufacturing / Supply Chain
- Government / Public Sector
- Education
- Retail / E-Commerce
- Other (free text)

After selection, probe for sub-domain specificity (e.g., "SaaS Operations — Site Reliability Engineering" not just "Technology").

### Phase 3: Stakeholders

Elicit the organizational structure:
- "Who is sponsoring this project? What's their title?"
- "Who will do the hands-on work? How is the team structured?"
- "Who needs to approve deliverables or funding?"
- "Are there external stakeholders — vendors, regulators, customers?"

Goal: A named list of 3-8 roles with their relationship to the project.

### Phase 4: Constraints

Walk through each constraint category explicitly:

1. **Budget**: "Is there an allocated budget? Even a rough range helps calibrate the ROI analysis."
2. **Timeline**: "Is there a hard deadline or target duration? What drives it — a contract, a quarter boundary, a regulatory date?"
3. **Team size**: "How many people are available? Full-time or shared?"
4. **Compliance**: "Are there regulatory requirements — HIPAA, SOX, PCI DSS, GDPR, FedRAMP, SOC 2?"
5. **Technology**: "What platforms, languages, or tools are already in use or mandated?"

Accept "TBD" for any of these but note that more specificity produces better artifacts.

### Phase 5: Available Data

Ask:
- "Do you have existing analysis, reports, metrics, or datasets related to this problem?"
- "Any prior attempts to solve it? What happened?"
- Probe for **data format and access**: "Can you share exports, dashboards, or documentation?"

### Phase 6: Desired Outcomes

Ask:
- "What does success look like when this project is done?"
- Probe for **quantification**: "Can you express that as measurable targets? e.g., reduce X from Y to Z, achieve N% improvement"
- "What's the minimum viable outcome vs. the ideal outcome?"

Goal: 3-5 measurable success criteria with baseline and target values.

### Phase 7: Additional Context

Open-ended wrap-up:
- "Is there anything else I should know — organizational culture, politics, technical debt, dependencies on other projects, lessons from past failures?"
- "Any specific risks or concerns you're already aware of?"

## Synthesis

After all phases, assemble the complete `PROJECT-CONTEXT.md`:

1. Draft the document using the user's exact words where possible
2. Present a summary back to the user: "Here's what I captured — does this accurately reflect your project?"
3. Allow the user to correct or add detail
4. Write the final `PROJECT-CONTEXT.md`

## Output

Write `PROJECT-CONTEXT.md` at the repository root with these sections populated:

```markdown
# Project Context

## Problem Statement
[Synthesized from Phase 1]

## Domain / Industry
[From Phase 2]

## Stakeholders
[From Phase 3 — bulleted list with roles and responsibilities]

## Constraints
- **Budget**: [From Phase 4]
- **Timeline**: [From Phase 4]
- **Team size**: [From Phase 4]
- **Compliance**: [From Phase 4]
- **Technology**: [From Phase 4]

## Available Data
[From Phase 5]

## Desired Outcomes
[From Phase 6 — bulleted list with quantified targets]

## Additional Context
[From Phase 7]
```

## Quality Criteria

- [ ] Problem statement includes at least one quantified metric (or explicit "TBD")
- [ ] Domain is specific to sub-domain level, not just top-level industry
- [ ] 3+ stakeholder roles identified
- [ ] All 5 constraint categories addressed (even if "TBD")
- [ ] Desired outcomes include measurable targets
- [ ] User confirmed accuracy before final write
