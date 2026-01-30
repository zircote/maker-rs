---
name: runbook-template
description: Create a standardized operational runbook/playbook template with triage steps, diagnosis procedures, resolution scenarios, escalation paths, and an automation example. Use when the project involves operational procedures.
allowed-tools: Read, Grep, Glob, WebSearch
---

Generate `RUNBOOK-TEMPLATE.md` at the repository root.

## Process

1. Read `_research/DOMAIN-RESEARCH.md` for operational context and terminology
2. Read `SEVERITY-CLASSIFICATION.md` for priority levels and routing
3. Build the standard template with all operational sections
4. Create a domain-appropriate automation example
5. Define quality checklist and assignment table

## Required Sections

1. **Standard Runbook Template** containing:
   - Overview table (ID, Category, Severity, Owner, Last Updated, Review Cadence)
   - Trigger Conditions
   - Triage steps (first 5 minutes, checkbox list)
   - Diagnosis by symptom (with code blocks for commands/procedures)
   - Resolution Scenarios (2+ scenarios with steps and verification)
   - Escalation table (Condition | Escalate To | Contact | Timeline)
   - Rollback Procedure (with code blocks)
   - Related Items table (alerts, other runbooks, dashboards)
   - Recent Incidents table
   - Automation Status table (Detection/Diagnosis/Resolution/Notification)
2. **Automation Example**: Domain-appropriate (AWS SSM, Ansible, CI/CD, script)
3. **Runbook Library Organization**: Table (Category | Count Target | Topics)
4. **Quality Checklist**: 12+ items every runbook must pass
5. **Assignment Table**: Table (# | Area | Owner Team | Priority | Status)

## Quality Criteria

- [ ] Template covers triage through rollback
- [ ] Domain-appropriate automation example included
- [ ] Quality checklist with 12+ items
- [ ] Library organization plan
- [ ] Assignment table with team ownership
