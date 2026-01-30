---
description: Run domain research and best practices analysis for the project
argument-hint: "[optional domain or focus area]"
allowed-tools: Read, Write, Grep, Glob, WebSearch, WebFetch
---

# Domain Research

Run the research phase of the project planning pipeline.

## Context

- Project context: !`cat PROJECT-CONTEXT.md 2>/dev/null || echo "PROJECT-CONTEXT.md not found"`

## Process

$IF($ARGUMENTS,
  Focus research on: $ARGUMENTS,
  Research the full domain from PROJECT-CONTEXT.md
)

1. Read `PROJECT-CONTEXT.md` for the problem statement and domain
2. Run `/domain-research` to produce `_research/DOMAIN-RESEARCH.md`
3. Run `/best-practices` to produce `BEST-PRACTICES.md`
4. Verify outputs meet quality gates:
   - 8+ framework citations with URLs
   - 15+ terminology entries
   - Industry benchmarks from authoritative sources
   - Regulatory requirements identified

## Output

Report what was generated and highlight key findings: primary frameworks identified, terminology count, and any compliance requirements discovered.
