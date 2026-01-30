---
name: domain-research
description: Research the problem domain to understand industry standards, terminology, regulatory requirements, and artifact expectations. Use when starting a new project or when industry context is needed for planning artifacts.
argument-hint: "[problem statement or domain description]"
allowed-tools: Read, Grep, Glob, WebSearch, WebFetch
---

Research the problem domain described in $ARGUMENTS (or PROJECT-CONTEXT.md if no arguments).

## Process

1. **Classify the domain**: Map the problem to an industry and sub-domain
2. **Identify frameworks**: Search for 8+ authoritative standards (ISO, NIST, PMI, industry-specific)
3. **Gather benchmarks**: Find quantitative baselines from industry reports
4. **Build terminology**: Create a 15+ term domain vocabulary
5. **Assess compliance**: Identify regulatory requirements (HIPAA, SOX, PCI DSS, etc.)
6. **Determine artifacts**: Confirm which planning artifacts are required vs. optional
7. **Cite sources**: Document all references with full citations and URLs

## Output

Write `_research/DOMAIN-RESEARCH.md` containing:

- **Industry Classification**: Primary industry, sub-domain, project type
- **Terminology Dictionary**: Table with 15+ domain terms and definitions
- **Applicable Frameworks**: Table of 8+ frameworks with relevance and key principles
- **Regulatory & Compliance Requirements**: List with descriptions and project impact
- **Industry Benchmarks**: Table with metric, industry average, top quartile, source
- **Artifact Requirements**: Table mapping each planning artifact to industry-specific considerations
- **Recommended Approach**: Methodology (Agile/Waterfall/Hybrid), phase structure, cadence
- **Domain-Specific Risks**: 3+ risks common to this industry
- **Sources & Citations**: Full citations with URLs

## Industry Adaptation

| Domain | Key Frameworks | Compliance |
|--------|---------------|------------|
| Technology/SaaS | Google SRE, ITIL 4, DORA, AWS Well-Architected | SOC 2, ISO 27001, GDPR |
| Healthcare | HIPAA, HL7 FHIR, HITRUST, Joint Commission | HIPAA Security Rule, HITECH |
| Financial | PCI DSS, SOX, Basel III, FFIEC | SEC, FINRA, CFPB |
| Manufacturing | Lean Six Sigma, ISO 9001, TPM | OSHA, EPA, ISO 14001 |
| General | PMI PMBOK, PRINCE2, SAFe, Scrum Guide | Industry-dependent |

## Quality Criteria

- [ ] 8+ framework citations with URLs
- [ ] 15+ terminology entries
- [ ] Industry benchmarks from authoritative sources
- [ ] Regulatory requirements identified
- [ ] Methodology recommendation with rationale
