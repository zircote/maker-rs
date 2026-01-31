# Prompt Engineering for MAKER

How to effectively instruct your AI assistant (Claude, etc.) to use MAKER for reliable, error-corrected outputs.

## Why This Matters

MAKER is a tool that AI assistants can use, but they need to know when and how to use it. This guide shows you how to:

1. Tell Claude when to use MAKER
2. Configure MAKER for different task types
3. Structure multi-step workflows
4. Interpret MAKER results

## Quick Reference: Prompt Templates

### Request Error-Corrected Response

```
Use MAKER to vote on: "<your question>"
```

```
I need a reliable answer to this. Use maker/vote with k_margin=5:
"<your question>"
```

### Validate Generated Output

```
Generate <X> and then validate it with maker/validate to ensure it meets:
- Token limit: 500
- Valid JSON format
```

### Multi-Step Reliable Workflow

```
This is a multi-step task. For each step, use MAKER voting to ensure reliability:
1. <step 1>
2. <step 2>
3. <step 3>

Configure MAKER with adaptive k-margin to optimize sampling.
```

---

## Understanding MAKER Parameters

### When to Adjust K-Margin

| Scenario | Recommended k | Rationale |
|----------|---------------|-----------|
| Factual questions | 3 | Low variance, converges quickly |
| Calculations | 5 | Need certainty on exact values |
| Code generation | 3-5 | Use with code matcher |
| Creative writing | 2-3 | High variance acceptable |
| Multi-step tasks | 4+ | Errors compound |

**Prompt example:**
```
This is a critical calculation for financial reporting. Use maker/vote
with k_margin=5 and max_samples=50 to ensure accuracy:
"Calculate the compound interest on $10,000 at 5% for 10 years"
```

### When to Use Different Matchers

| Matcher | Use Case | Example Prompt |
|---------|----------|----------------|
| `exact` | Deterministic answers | "What is 2+2?" |
| `embedding` | Natural language, explanations | "Explain how TCP works" |
| `code` | Programming, algorithms | "Write a sort function" |

**Prompt example:**
```
Configure MAKER with the code matcher for Rust, then vote on:
"Implement a binary search tree insert function"
```

### When to Use Adaptive K-Margin

Use adaptive k when:
- Running many similar queries
- Accuracy varies throughout a session
- Want to optimize API costs

**Prompt example:**
```
I'm going to ask you 20 math problems. Configure MAKER with:
- adaptive_k: true
- k_bounds: [2, 8]
- ema_alpha: 0.1

This will automatically adjust the voting margin based on observed accuracy.
```

---

## Task-Specific Prompting

### For Coding Tasks

```
# Good: Specific, uses code matching
Configure MAKER for Python code matching with threshold 0.95.
Then use maker/vote with k_margin=3 to implement:
"A function that finds all prime numbers up to n using the Sieve of Eratosthenes"

After voting, validate the result passes these checks:
- Valid Python syntax
- Under 50 lines
- Has docstring

# Bad: Vague, no matcher specified
Use MAKER to write some code for prime numbers
```

### For Data Analysis

```
# Good: Clear steps, validation criteria
This is a 3-step data analysis task. For each step, use MAKER voting:

1. Use maker/vote to determine the correct SQL query for:
   "Find the top 10 customers by total purchase amount"

2. Validate the query with maker/validate ensuring:
   - Valid SQL syntax
   - No destructive operations (DELETE, DROP, TRUNCATE)

3. Use maker/vote to format the results as a markdown table
```

### For Multi-Step Workflows

```
# Good: Explicit decomposition, per-step voting
This is a 5-step migration task. Use MAKER's decomposition for reliability.

MAKER Configuration:
- Provider: ollama
- Adaptive k-margin: enabled
- Matcher: exact

Steps:
1. List all files matching *.config.json
2. For each file, extract the "database" section
3. Transform connection strings from format A to format B
4. Validate each transformed config passes JSON schema
5. Write the updated files

For each step, use maker/vote with k_margin=3 before proceeding.
```

### For Critical Decisions

```
# Good: High k-margin, explicit reliability requirement
This decision affects production systems. Use MAKER with:
- k_margin: 7
- max_samples: 100

Question: "Given these server metrics, should we scale up, scale out, or do nothing?"

Metrics:
- CPU: 85%
- Memory: 60%
- Request latency p99: 450ms
- Error rate: 0.1%

I need 95%+ confidence in this recommendation.
```

---

## Workflow Patterns

### Pattern 1: Generate-Validate-Commit

```
I'm implementing a feature. For each code change:

1. Use maker/vote to generate the code
2. Use maker/validate to check:
   - Token limit: 300 lines
   - No syntax errors
3. Only apply changes that pass validation

Start with: "Add input validation to the user registration endpoint"
```

### Pattern 2: Calibrate-Configure-Execute

```
Before starting the main task, let's calibrate MAKER.

Phase 1 - Calibration:
I'll give you 5 test cases. Use maker/calibrate to estimate accuracy:
- "2+2" -> "4"
- "capital of France" -> "Paris"
- "reverse 'hello'" -> "olleh"
- "10! (factorial)" -> "3628800"
- "HTTP 404 meaning" -> "Not Found"

Phase 2 - Configure:
Based on calibration results, configure MAKER with the recommended k_margin.

Phase 3 - Execute:
Now proceed with the main task using these settings.
```

### Pattern 3: Recursive Decomposition

```
This is a complex task. Use MAKER's decomposition framework:

Task: "Refactor the authentication module to use OAuth2"

Instructions:
1. Use maker/decompose to break this into subtasks
2. Each leaf subtask should be atomic (m=1)
3. Execute subtasks with maker/vote
4. Aggregate results according to the composition function

Report the decomposition tree before executing.
```

---

## Instructing Claude About MAKER

### Add to Your System Prompt / CLAUDE.md

If you want Claude to automatically use MAKER for certain tasks, add instructions:

```markdown
# MAKER Error Correction

When I ask for reliable or critical outputs, use the MAKER MCP tools:

## Automatic MAKER Usage

Use maker/vote automatically when:
- I ask for "reliable", "accurate", or "verified" results
- The task is multi-step (3+ steps)
- I'm asking about calculations or code
- I say "make sure this is correct"

## MAKER Configuration Defaults

- k_margin: 3 (increase to 5+ for critical tasks)
- Matcher: Use 'code' for programming, 'embedding' for explanations
- Adaptive: Enable for long sessions

## Validation

Always use maker/validate before presenting:
- Generated code
- JSON/YAML output
- Configuration files
```

### Example CLAUDE.md Section

```markdown
## MAKER Integration

This project uses MAKER for error-corrected LLM outputs.

### When to Use MAKER

- **Code generation**: Always use `maker/vote` with `matcher: code`
- **Data transformations**: Use `maker/vote` then `maker/validate` with schema
- **Multi-step tasks**: Use `maker/decompose` to break into atomic operations

### Project Settings

```json
{
  "k_default": 4,
  "provider": "ollama",
  "adaptive_k": true,
  "matcher": {"type": "code", "language": "rust"}
}
```

Apply these settings at the start of each session with `maker/configure`.
```

---

## Common Mistakes

### 1. Not Specifying Matcher Type

```
# Bad
Use MAKER to write a Python function

# Good
Configure MAKER with Python code matcher, then vote on the function
```

### 2. Too Low K-Margin for Critical Tasks

```
# Bad (for financial calculations)
Use maker/vote with k_margin=2 for this tax calculation

# Good
Use maker/vote with k_margin=5 and max_samples=50 for this tax calculation
```

### 3. Not Validating Before Use

```
# Bad
Generate the config file with MAKER

# Good
Generate the config file with maker/vote, then validate with maker/validate
using the JSON schema before applying
```

### 4. Ignoring MAKER Results

```
# Bad
Use MAKER for this, then do whatever seems right

# Good
Use maker/vote. If it converges with 5+ samples agreeing, use that result.
If it doesn't converge, increase max_samples or rephrase the question.
```

---

## Debugging MAKER Usage

### Ask Claude to Explain MAKER Decisions

```
After using MAKER for this task, explain:
1. What k_margin was used and why
2. How many samples were needed
3. What matcher was applied
4. Whether the result converged

Then provide the result.
```

### Request MAKER Statistics

```
Use MAKER for these 10 questions. After each, report:
- Total samples
- Winner votes
- p_hat estimate
- Time to convergence

Then summarize MAKER's performance across all questions.
```

---

## Quick Reference Card

| Need | Prompt Pattern |
|------|----------------|
| Reliable answer | "Use maker/vote with k_margin=3: ..." |
| Validated output | "Use maker/validate with token_limit=X and schema=..." |
| Estimate accuracy | "Use maker/calibrate with these samples: ..." |
| Configure session | "Configure MAKER with adaptive_k=true, k_bounds=[2,8]" |
| Code generation | "Configure MAKER with code matcher for [language], then vote" |
| Multi-step task | "Use maker/decompose, then execute each subtask with voting" |
| Critical decision | "Use maker/vote with k_margin=7, max_samples=100" |

---

## See Also

- [MCP-INTEGRATION.md](./MCP-INTEGRATION.md) - Technical MCP setup
- [guides/CODING.md](./guides/CODING.md) - MAKER for programming tasks
- [guides/DATA-ANALYSIS.md](./guides/DATA-ANALYSIS.md) - MAKER for data work
- [guides/ML-PIPELINES.md](./guides/ML-PIPELINES.md) - MAKER for ML workflows
