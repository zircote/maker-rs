---
name: best-practices
description: Research and catalog industry standards and frameworks applied to the project with citations, application mapping, and maturity model. Use alongside domain research to ground the project in established practice.
allowed-tools: Read, Grep, Glob, WebSearch, WebFetch
---

Generate `BEST-PRACTICES.md` at the repository root.

## Process

1. Read `_research/DOMAIN-RESEARCH.md` for identified frameworks
2. Read `PROJECT-PLAN.md` for approach and methodology
3. Deep-research 8+ frameworks with full citations
4. Map each framework's principles to project epics
5. Build maturity model progression
6. Create practice application matrix

## Required Sections

1. **Framework Entries** (8+ frameworks, each containing):
   - Framework name, author/org, year
   - Source citation with URL
   - Key Principles (bullet list)
   - Application table (Principle | Applied In | Epic IDs)
   - Rationale for inclusion
2. **Process Flow**: Mermaid `flowchart TD` for a domain-specific process
3. **Maturity Model**: Table (Level | Name | Characteristics | Target Phase), 4-5 levels
4. **Practice Application Matrix**: Table (Practice | Phase 1 | Phase 2 | Phase 3 | Phase 4)
5. **Implementation Priority**: Table (Priority | Practice | Effort | Impact | Phase)

## Quality Criteria

- [ ] 8+ frameworks with full citations and URLs
- [ ] Mix of industry-agnostic and domain-specific frameworks
- [ ] Each framework mapped to specific project activities
- [ ] Maturity model progression defined
- [ ] Practice application matrix complete
- [ ] At least one Mermaid process flow diagram
