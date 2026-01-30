---
description: "Conduct a structured interview to elicit project details and build PROJECT-CONTEXT.md"
argument-hint: "[optional: 'refine' to update existing context, or a brief project description]"
allowed-tools: Read, Write, Grep, Glob, AskUserQuestion
---

# Project Context Elicitation

Build or refine `PROJECT-CONTEXT.md` through a guided interview. The template source lives at `.claude/templates/PROJECT-CONTEXT.md`; the populated output is written to root `PROJECT-CONTEXT.md`.

## Context

- Existing context: !`cat PROJECT-CONTEXT.md 2>/dev/null | head -5 || echo "No PROJECT-CONTEXT.md found"`
- Template source: `.claude/templates/PROJECT-CONTEXT.md`

## Process

$IF($ARGUMENTS,
  $IF($ARGUMENTS == "refine",
    Read existing PROJECT-CONTEXT.md and identify gaps or placeholder sections. Interview the user only for missing or incomplete sections.,
    Use "$ARGUMENTS" as a seed description. Pre-populate what you can infer and then interview the user to fill remaining gaps.
  ),
  Start a fresh interview from scratch.
)

Run `/project-context` to conduct the structured elicitation interview.

Walk the user through each section of PROJECT-CONTEXT.md one phase at a time:
1. Problem Statement (with quantified impact)
2. Domain / Industry (to sub-domain specificity)
3. Stakeholders (named roles and responsibilities)
4. Constraints (budget, timeline, team, compliance, technology)
5. Available Data (existing analysis and datasets)
6. Desired Outcomes (measurable success criteria)
7. Additional Context (culture, risks, history)

After gathering all inputs, synthesize and present a draft for user confirmation, then write the final `PROJECT-CONTEXT.md` at the repository root (using `.claude/templates/PROJECT-CONTEXT.md` as the structural template).

## Output

Report what was captured: section completeness, any "TBD" items that should be revisited, and suggest running `/generate-all` or `/research` as the next step.
