# Runbook Template: MAKER Framework Operations

**Project:** Rust Implementation of MAKER Framework with MCP Integration
**Version:** 1.0
**Date:** 2026-01-30
**Status:** Active Template

---

## Document Purpose

This document provides the **standard runbook template** for operational procedures related to the MAKER framework. Each runbook instance follows this structure to ensure consistent incident response, troubleshooting, and escalation.

**Target Audience:** Operators, maintainers, and contributors managing MAKER deployments or debugging integration issues.

---

## Standard Runbook Template

### Runbook Metadata

| Field | Value |
|-------|-------|
| **Runbook ID** | RB-XXX (e.g., RB-001, RB-002) |
| **Title** | [Descriptive title of operational scenario] |
| **Category** | Deployment / Performance / Security / Integration / Algorithm |
| **Severity** | P0 (Critical) / P1 (High) / P2 (Medium) / P3 (Low) |
| **Owner** | [Team/Individual responsible] |
| **Last Updated** | [YYYY-MM-DD] |
| **Review Cadence** | Monthly / Quarterly / Annually |

---

### Trigger Conditions

**When to use this runbook:**
- [ ] [Observable symptom 1]
- [ ] [Observable symptom 2]
- [ ] [Observable symptom 3]

**Example:** Alert fires for "API retry success rate < 95%" or "Vote convergence failure > 30%"

---

### Triage (First 5 Minutes)

**Immediate Actions:**
- [ ] **Confirm Scope:** Is this affecting one user, one deployment, or all instances?
- [ ] **Check Dashboard:** Review metrics dashboard for correlated anomalies
- [ ] **Review Recent Changes:** git log, recent deployments, configuration changes
- [ ] **Collect Basic Info:** MAKER version, model provider, task type
- [ ] **Assess Impact:** Production down? Degraded performance? Dev environment only?

**Quick Health Checks:**
```bash
# Check MAKER server is running
ps aux | grep maker-mcp-server

# Review recent error logs
tail -n 100 /var/log/maker/error.log | grep -i "error\|panic"

# Test MCP connectivity
echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | maker-mcp-server

# Check API provider status (OpenAI example)
curl -s https://status.openai.com/api/v2/status.json | jq '.status.description'
```

---

### Diagnosis by Symptom

#### Symptom 1: [Observable Problem A]

**Common Causes:**
1. [Root cause possibility 1]
2. [Root cause possibility 2]
3. [Root cause possibility 3]

**Diagnostic Procedure:**
```bash
# Step 1: Check [specific component]
[command to inspect component]

# Step 2: Verify [configuration/state]
[command to verify state]

# Step 3: Test [isolated functionality]
[command to test in isolation]
```

**Expected Output:**
- **Normal:** [Description of healthy state]
- **Abnormal:** [Description of problematic state]

---

#### Symptom 2: [Observable Problem B]

**Common Causes:**
1. [Root cause possibility 1]
2. [Root cause possibility 2]

**Diagnostic Procedure:**
```bash
# Step 1: Inspect event logs
grep "VoteDecided" /var/log/maker/events.json | jq '.total_votes, .k_margin'

# Step 2: Check red-flag rates
grep "RedFlagTriggered" /var/log/maker/events.json | jq -r '.flag_type' | sort | uniq -c
```

**Expected Output:**
- **Normal:** Red-flag rate <8%, votes converge within k_min samples
- **Abnormal:** Red-flag rate >15%, votes require 2×k_min+ samples

---

### Resolution Scenarios

#### Scenario 1: [Resolution Path A]

**When to use:** [Condition indicating this resolution applies]

**Steps:**
1. **[Action 1]:** [Detailed command or procedure]
   ```bash
   [command]
   ```
   **Verification:** [How to confirm success]

2. **[Action 2]:** [Next step]
   ```bash
   [command]
   ```
   **Verification:** [Expected result]

3. **[Action 3]:** [Final step]
   ```bash
   [command]
   ```
   **Verification:** [Success criteria]

**Post-Resolution Validation:**
- [ ] Metric X returns to normal range
- [ ] No new errors in logs for 10 minutes
- [ ] End-to-end test passes

---

#### Scenario 2: [Resolution Path B]

**When to use:** [Condition indicating this resolution applies]

**Steps:**
1. **Restart Service:**
   ```bash
   systemctl restart maker-mcp-server
   systemctl status maker-mcp-server
   ```
   **Verification:** Service active, no errors in `journalctl -u maker-mcp-server`

2. **Clear Cache/State:**
   ```bash
   rm -rf /tmp/maker-cache/*
   ```
   **Verification:** New events show fresh state hashes

3. **Re-run Calibration:**
   ```bash
   maker-cli calibrate --task hanoi-3disk --samples 100
   ```
   **Verification:** `p_estimate` output matches expected range (0.80-0.95)

---

### Escalation Paths

| Condition | Escalate To | Contact | Timeline |
|-----------|-------------|---------|----------|
| **Triage fails within 15 minutes** | Senior Engineer | @maintainer-handle | Immediate (Slack) |
| **Issue affects production users** | Project Lead | maintainer@example.com | Within 1 hour |
| **Security vulnerability suspected** | Security Team | security@example.com | Immediate (encrypt report) |
| **Requires code change** | Core Maintainers | GitHub Issue (tag: `P1-bug`) | Within 24 hours |
| **Unresolved after 4 hours** | Community (GitHub Discussions) | Public discussion | Transparency update |

**Escalation Template (Slack/Email):**
```
Subject: MAKER Incident - [Brief Description]

Severity: [P0/P1/P2/P3]
Runbook: [RB-XXX]
Affected: [Scope: one user / all deployments / specific model provider]

Symptoms:
- [Observable issue 1]
- [Observable issue 2]

Attempted Resolutions:
- [Action taken 1] - [Result]
- [Action taken 2] - [Result]

Current State: [Degraded / Down / Partially Functional]
Impact: [Description of user impact]

Requesting: [Specific help needed]
```

---

### Rollback Procedure

**When to Rollback:**
- [ ] Resolution attempts fail after 2 hours
- [ ] Issue introduced by recent deployment/config change
- [ ] Production impact is severe (P0 severity)

**Steps:**
1. **Identify Last Known Good Version:**
   ```bash
   git log --oneline -10
   git tag -l "v*"
   ```

2. **Revert Deployment:**
   ```bash
   # For binary deployment
   cp /backups/maker-mcp-server-v0.1.0 /usr/local/bin/maker-mcp-server
   systemctl restart maker-mcp-server

   # For source deployment
   git checkout v0.1.0
   cargo build --release
   systemctl restart maker-mcp-server
   ```

3. **Verify Rollback:**
   ```bash
   maker-mcp-server --version  # Should show v0.1.0
   echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | maker-mcp-server
   ```

4. **Document Rollback:**
   - Update GitHub Issue with rollback decision
   - Note time to resolution (TTR) in incident log

**Post-Rollback:**
- [ ] Production service restored
- [ ] Root cause investigation scheduled (separate from incident response)
- [ ] Fix developed in dev environment before re-deployment

---

### Related Items

| Type | Reference | Description |
|------|-----------|-------------|
| **Alert** | `maker_api_retry_success_rate < 95%` | Triggers when API reliability degrades |
| **Alert** | `maker_vote_convergence_rate < 70%` | Triggers when voting inefficiency detected |
| **Dashboard** | [MAKER Ops Dashboard](http://dashboard.example.com) | Real-time metrics |
| **Runbook** | RB-001: API Rate Limit Recovery | Related to API provider issues |
| **Runbook** | RB-002: Vote Convergence Tuning | Related to k-margin optimization |
| **Log Query** | `grep "StepCompleted" /var/log/maker/events.json` | Extract task completion metrics |

---

### Recent Incidents

| Date | Incident Summary | Root Cause | Resolution | TTR |
|------|------------------|------------|------------|-----|
| 2026-01-25 | API retry failures | OpenAI rate limit | Implemented exponential backoff | 45 min |
| 2026-01-20 | Vote convergence slow | k_min underestimated | Recalibrated p estimate | 2 hours |
| [Add incidents as they occur] | | | | |

**TTR:** Time to Resolution (from alert to service restoration)

---

### Automation Status

| Phase | Detection | Diagnosis | Resolution | Notification |
|-------|-----------|-----------|------------|--------------|
| **Current (MVP)** | Manual / Alert | Manual | Manual | Manual (Slack) |
| **Phase 1 (Month 1)** | Automated (Prometheus) | Manual | Manual | Automated (PagerDuty) |
| **Phase 2 (Month 3)** | Automated | Partial (health checks) | Manual | Automated |
| **Phase 3 (Month 6+)** | Automated | Automated | Partial (auto-restart, fallback) | Automated |

**Automation Goals:**
- **Detection:** Prometheus alerts for all KPIs
- **Diagnosis:** Automated log analysis and correlation
- **Resolution:** Auto-restart on transient failures, auto-fallback to Ollama
- **Notification:** PagerDuty integration with on-call rotation

---

## Example Runbook: RB-001 - API Rate Limit Recovery

### Runbook Metadata

| Field | Value |
|-------|-------|
| **Runbook ID** | RB-001 |
| **Title** | OpenAI/Anthropic API Rate Limit Recovery |
| **Category** | Performance / Integration |
| **Severity** | P1 (High) - Blocks task execution but not system failure |
| **Owner** | Project Maintainer |
| **Last Updated** | 2026-01-30 |
| **Review Cadence** | Monthly |

---

### Trigger Conditions

**When to use this runbook:**
- [ ] Alert: `maker_api_retry_success_rate < 95%` over 1-hour window
- [ ] Logs show repeated 429 errors from LLM provider
- [ ] Users report "voting timeout" errors
- [ ] Metrics show increased API latency (P95 > 3× baseline)

---

### Triage (First 5 Minutes)

**Immediate Actions:**
- [ ] **Check Provider Status:** Visit https://status.openai.com or https://status.anthropic.com
- [ ] **Review Rate Limit Metrics:** Current requests/min vs. quota
- [ ] **Identify Affected Tasks:** Which tasks are failing? Single user or widespread?
- [ ] **Check Retry Logic:** Verify exponential backoff is active (not disabled)
- [ ] **Assess Blast Radius:** Production tasks or dev experiments?

**Quick Health Checks:**
```bash
# Check recent API errors
grep -A 5 "429\|RateLimited" /var/log/maker/error.log | tail -n 50

# Count errors by provider
grep "LlmApi" /var/log/maker/events.json | jq -r '.provider' | sort | uniq -c

# Check current retry delay
grep "retrying after" /var/log/maker/error.log | tail -n 10

# Verify provider status (OpenAI example)
curl -s https://status.openai.com/api/v2/status.json | jq '.status.description'
```

---

### Diagnosis by Symptom

#### Symptom 1: Persistent 429 Errors Despite Backoff

**Common Causes:**
1. Token-based rate limit exceeded (not request-based)
2. Retry logic not respecting `Retry-After` header
3. Concurrent tasks from multiple deployments sharing quota
4. Provider-side incident (degraded service)

**Diagnostic Procedure:**
```bash
# Step 1: Inspect retry logic behavior
grep "Rate limited" /var/log/maker/error.log | jq -r '.retry_after, .attempt'

# Step 2: Check token consumption rate
grep "SampleCompleted" /var/log/maker/events.json | \
  jq -r '.tokens_used' | \
  awk '{sum+=$1; count++} END {print "Avg tokens/sample:", sum/count}'

# Step 3: Verify Retry-After header handling
grep "retry_after" /var/log/maker/error.log | jq '.retry_after'
```

**Expected Output:**
- **Normal:** `retry_after` values increasing exponentially (1s, 2s, 4s, 8s...), retries succeed after backoff
- **Abnormal:** Constant retry delays, missing `Retry-After` parsing, retries fail repeatedly

---

#### Symptom 2: Quota Exhausted (Daily/Monthly Limit)

**Common Causes:**
1. Task volume exceeds purchased quota
2. Misconfigured cost controls (no budget caps)
3. Inefficient k-margin (too many samples per vote)

**Diagnostic Procedure:**
```bash
# Step 1: Calculate daily token usage
grep "SampleCompleted" /var/log/maker/events.json | \
  jq -r '.timestamp, .tokens_used' | \
  awk '{date=substr($1,1,10); sum[date]+=$2} END {for(d in sum) print d, sum[d]}'

# Step 2: Compare to quota
echo "Daily quota: 1M tokens (example)"
# Output from Step 1 should be < quota

# Step 3: Check k-margin efficiency
grep "VoteDecided" /var/log/maker/events.json | \
  jq -r '.total_votes, .k_margin' | \
  awk '{avg_samples+=$1; count++} END {print "Avg samples/vote:", avg_samples/count}'
```

**Expected Output:**
- **Normal:** Daily usage <80% of quota, avg samples ≈ k_min
- **Abnormal:** Usage >90% quota, avg samples >> k_min (inefficient voting)

---

### Resolution Scenarios

#### Scenario 1: Transient Rate Limit (Provider Side)

**When to use:** Provider status page shows degraded performance, retry logic is working.

**Steps:**
1. **Increase Retry Patience:**
   ```bash
   # Edit configuration (if using config file)
   vim /etc/maker/config.toml
   # Set: max_retries = 7, max_delay_ms = 120000
   systemctl restart maker-mcp-server
   ```
   **Verification:** Logs show longer backoff delays, retries eventually succeed.

2. **Monitor Recovery:**
   ```bash
   watch -n 10 'grep "Rate limited" /var/log/maker/error.log | tail -n 5'
   ```
   **Verification:** Error rate decreases, `api_retry_success_rate` metric returns to >99%.

3. **Validate Service Restoration:**
   ```bash
   # Run 3-disk Hanoi test
   maker-cli test hanoi-3disk
   ```
   **Verification:** Task completes with zero errors.

---

#### Scenario 2: Quota Exhausted (Hard Limit)

**When to use:** Provider returns quota error, daily/monthly limit reached.

**Steps:**
1. **Fallback to Local Model (Ollama):**
   ```bash
   # Temporarily switch provider
   export MAKER_LLM_PROVIDER=ollama
   export MAKER_OLLAMA_URL=http://localhost:11434
   systemctl restart maker-mcp-server
   ```
   **Verification:** New samples use Ollama, no API costs incurred.

2. **Notify Users (if multi-user deployment):**
   ```bash
   # Post to status page
   echo "MAKER temporarily using Ollama due to cloud quota. Performance may vary." | \
     tee /var/www/status/current.txt
   ```

3. **Schedule Quota Increase:**
   - Contact provider to increase tier
   - Estimate required quota: `daily_tasks × avg_samples × avg_tokens × 1.5 (buffer)`

4. **Optimize k-Margin (if chronically over-quota):**
   ```bash
   # Recalibrate p estimate (may allow lower k)
   maker-cli calibrate --task hanoi-5disk --samples 200
   # Use new k_min in configuration
   ```

**Post-Resolution Validation:**
- [ ] Cloud quota restored or Ollama fallback stable
- [ ] Task execution resumes
- [ ] Cost tracking updated to reflect provider change

---

#### Scenario 3: Retry Logic Bug

**When to use:** Retries fail even with ample quota, backoff not exponential.

**Steps:**
1. **Check MAKER Version:**
   ```bash
   maker-mcp-server --version
   ```
   **Verification:** If <v0.1.0, upgrade to latest.

2. **Review Code/Logs:**
   ```bash
   # Inspect retry implementation
   grep -A 20 "fn call_llm_with_retry" src/llm_client.rs
   ```
   **Verification:** Code matches best practices (exponential backoff, jitter, Retry-After parsing).

3. **Apply Hotfix (if bug confirmed):**
   ```bash
   git pull origin main
   cargo build --release
   systemctl restart maker-mcp-server
   ```

4. **Report Bug:**
   - GitHub Issue: "Retry logic not respecting Retry-After header"
   - Include: logs, version, provider, expected vs. actual behavior

**Post-Resolution Validation:**
- [ ] Retries succeed after fix
- [ ] Unit tests added for retry logic
- [ ] Regression test in CI

---

### Escalation Paths

| Condition | Escalate To | Contact | Timeline |
|-----------|-------------|---------|----------|
| **Provider incident confirmed (status page red)** | No action (wait for provider) | Monitor https://status.openai.com | Check hourly |
| **Quota issue unresolved after 2 hours** | Project Lead for budget approval | maintainer@example.com | Within 3 hours |
| **Retry logic bug suspected** | Core Maintainers | GitHub Issue (tag: `P1-bug`, `api`) | Within 1 hour |
| **Widespread production impact** | Community (transparency update) | GitHub Discussions | Within 4 hours |

---

### Rollback Procedure

**Not typically applicable for API rate limits.** Fallback to Ollama is the "rollback" equivalent.

**If Recent Deployment Caused Issue:**
```bash
# Revert to last stable version
git checkout v0.1.0
cargo build --release
systemctl restart maker-mcp-server
```

---

### Related Items

| Type | Reference | Description |
|------|-----------|-------------|
| **Alert** | `maker_api_retry_success_rate < 95%` | Prometheus alert triggering this runbook |
| **Dashboard** | Token Usage Panel | Real-time cost and quota tracking |
| **Runbook** | RB-002: Vote Convergence Tuning | May help reduce API calls if k too high |
| **Config** | `/etc/maker/config.toml` | Retry settings: `max_retries`, `max_delay_ms` |
| **External** | [OpenAI Rate Limits Docs](https://platform.openai.com/docs/guides/rate-limits) | Provider-specific guidance |

---

### Recent Incidents

| Date | Provider | Root Cause | Resolution | TTR |
|------|----------|------------|------------|-----|
| 2026-01-25 | OpenAI | GPT-4.1-mini quota exceeded | Fallback to Ollama, quota increased next day | 45 min |

---

### Automation Status

| Phase | Detection | Diagnosis | Resolution | Notification |
|-------|-----------|-----------|------------|--------------|
| **Current (MVP)** | Alert (Prometheus) | Manual (this runbook) | Manual | Manual |
| **Phase 1 (Month 1)** | Automated | Partial (quota check script) | Manual | Automated (PagerDuty) |
| **Phase 2 (Month 3)** | Automated | Automated (log analysis) | Auto-fallback to Ollama | Automated |

**Future Automation:**
```python
# Auto-fallback script (Phase 2)
if api_retry_success_rate < 95% and provider_status == "degraded":
    switch_provider("ollama")
    notify_oncall("Switched to Ollama due to API issues")
```

---

## Runbook Library Organization

| Category | Count Target | Example Topics |
|----------|--------------|----------------|
| **Deployment** | 3-5 runbooks | Fresh install, upgrade, rollback, configuration management |
| **Performance** | 4-6 runbooks | API rate limits, vote convergence tuning, latency optimization, cost reduction |
| **Security** | 2-3 runbooks | Prompt injection response, schema validation failure, unauthorized access |
| **Integration** | 3-5 runbooks | MCP server startup failure, Claude Code connection issues, tool registration errors |
| **Algorithm** | 2-4 runbooks | k_min calculation errors, red-flag tuning, calibration failures, voting race deadlock |

**Total Target:** 15-25 runbooks covering 90%+ of operational scenarios.

---

## Runbook Quality Checklist

Every runbook must pass these criteria before publication:

- [ ] **1. Metadata Complete:** ID, title, category, severity, owner, last updated, review cadence
- [ ] **2. Trigger Conditions Clear:** Observable symptoms listed (alert names, log patterns, user reports)
- [ ] **3. Triage Steps Defined:** 5-minute quick checks with commands
- [ ] **4. Diagnosis by Symptom:** At least 2 symptoms with diagnostic procedures
- [ ] **5. Resolution Scenarios:** At least 2 resolution paths with step-by-step commands
- [ ] **6. Verification Steps:** Each resolution step has "Expected Output" or "Verification" guidance
- [ ] **7. Escalation Paths:** Table with conditions, contacts, and timelines
- [ ] **8. Rollback Procedure:** Defined (or explicitly marked N/A with justification)
- [ ] **9. Related Items:** Links to alerts, dashboards, other runbooks
- [ ] **10. Recent Incidents:** Placeholder table (empty OK for new runbooks)
- [ ] **11. Automation Status:** Current and planned automation levels
- [ ] **12. Tested:** Runbook validated against actual incident or simulated scenario
- [ ] **13. Peer Reviewed:** At least one other maintainer has reviewed
- [ ] **14. Searchable:** Keywords in title and trigger conditions for easy lookup

**Quality Gate:** All 14 items must be checked before runbook is considered production-ready.

---

## Runbook Assignment Table

| # | Operational Area | Owner Team | Priority | Status |
|---|------------------|------------|----------|--------|
| 1 | **API Rate Limit Recovery** | Maintainer | P1 | ✅ Complete (RB-001) |
| 2 | **Vote Convergence Tuning** | Maintainer | P1 | ⏳ Pending |
| 3 | **MCP Server Startup Failure** | Maintainer | P1 | ⏳ Pending |
| 4 | **Red-Flag Threshold Tuning** | Maintainer | P2 | ⏳ Pending |
| 5 | **Claude Code Connection Issues** | Maintainer | P2 | ⏳ Pending |
| 6 | **Cost Spike Investigation** | Maintainer | P2 | ⏳ Pending |
| 7 | **Latency Degradation** | Maintainer | P2 | ⏳ Pending |
| 8 | **Security Incident Response** | Maintainer + Security | P0 | ⏳ Pending |
| 9 | **k_min Calculation Errors** | Maintainer | P1 | ⏳ Pending |
| 10 | **Fresh Deployment** | Maintainer | P2 | ⏳ Pending |
| 11 | **Upgrade from v0.X to v0.Y** | Maintainer | P2 | ⏳ Pending |
| 12 | **Rollback to Previous Version** | Maintainer | P1 | ⏳ Pending |
| 13 | **Calibration Procedure** | Maintainer | P2 | ⏳ Pending |
| 14 | **Prometheus Alert Configuration** | Maintainer | P3 | ⏳ Pending |
| 15 | **Log Analysis for Debugging** | Maintainer | P3 | ⏳ Pending |

**Status Legend:** ✅ Complete | 🚧 In Progress | ⏳ Pending

**Priority Guidance:**
- **P0:** Security or system-wide outage
- **P1:** Blocks critical functionality (voting, API calls)
- **P2:** Degrades performance or affects subset of users
- **P3:** Operational efficiency or nice-to-have

---

## References & Citations

1. **Google SRE Book.** *Effective Troubleshooting*. [https://sre.google/sre-book/effective-troubleshooting/](https://sre.google/sre-book/effective-troubleshooting/)
2. **PagerDuty.** *Incident Response Documentation*. [https://response.pagerduty.com/](https://response.pagerduty.com/)
3. **Atlassian.** *Incident Management Handbook*. [https://www.atlassian.com/incident-management/handbook](https://www.atlassian.com/incident-management/handbook)
4. **OpenAI.** *Rate Limits*. [https://platform.openai.com/docs/guides/rate-limits](https://platform.openai.com/docs/guides/rate-limits)
5. **Anthropic.** *API Error Codes*. [https://docs.anthropic.com/claude/reference/errors](https://docs.anthropic.com/claude/reference/errors)

---

**Document Maintenance:**
- **Update Frequency:** Monthly during MVP; Quarterly post-release
- **Owner:** Project Maintainer
- **Review Process:** Update after each incident; annual audit for all runbooks

**Version History:**
- v1.0 (2026-01-30): Initial runbook template and RB-001

---

**End of Runbook Template Document**
