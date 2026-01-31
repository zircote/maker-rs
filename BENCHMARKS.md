# MAKER Benchmark Results

Benchmark results for the MAKER framework across coding, math/logic, data analysis, cost scaling, and ensemble comparison domains.

Run benchmarks locally:

```bash
cargo bench --bench coding_tasks
cargo bench --bench math_logic
cargo bench --bench data_analysis
cargo bench --bench cost_scaling
cargo bench --bench ensemble_comparison
```

---

## Coding Tasks

Simulated coding tasks at varying difficulty levels using MAKER voting with `CodeMatcher`-style equivalence grouping.

**Parameters:** 500 trials, target reliability 95%

### Trivial Tasks (p >= 0.95)

| Task | p | Steps | k | Samples | Samples/Step | Error % | Red Flag % |
|------|---|-------|---|---------|-------------|---------|------------|
| FizzBuzz | 0.95 | 3 | 2 | 6 | 2.0 | 0.00% | 2.0% |
| Hello World variants | 0.96 | 2 | 2 | 4 | 2.0 | 0.00% | 2.0% |

### Moderate Tasks (p = 0.82-0.85)

| Task | p | Steps | k | Samples | Samples/Step | Error % | Red Flag % |
|------|---|-------|---|---------|-------------|---------|------------|
| Binary Search | 0.85 | 5 | 3 | 15 | 3.0 | 0.00% | 5.0% |
| Linked List Reversal | 0.83 | 4 | 3 | 12 | 3.0 | 0.00% | 5.0% |
| Stack Calculator | 0.84 | 6 | 3 | 18 | 3.0 | 0.00% | 5.0% |
| Merge Sort | 0.82 | 5 | 4 | 20 | 4.0 | 0.00% | 5.0% |

### Complex Tasks (p = 0.65-0.72)

| Task | p | Steps | k | Samples | Samples/Step | Error % | Red Flag % |
|------|---|-------|---|---------|-------------|---------|------------|
| JSON Parser | 0.70 | 10 | 7 | 70 | 7.0 | 0.00% | 10.0% |
| SQL Query Generator | 0.65 | 8 | 9 | 72 | 9.0 | 0.00% | 10.0% |
| Regex Engine | 0.68 | 12 | 8 | 96 | 8.0 | 0.00% | 10.0% |
| HTTP Router | 0.72 | 7 | 6 | 42 | 6.0 | 0.00% | 10.0% |

**Results:** 100% accuracy across all difficulty levels (trivial, moderate, complex).

---

## Math & Logic

Deterministic tasks with verifiable ground truth using `ExactMatcher`.

**Parameters:** 500 trials, target reliability 95%

### Arithmetic

| Task | p | Steps | k | Samples | Samples/Step | Error % | Cost/s*ln(s) |
|------|---|-------|---|---------|-------------|---------|-------------|
| 10-step arithmetic | 0.90 | 10 | 3 | 30 | 3.0 | 0.00% | 1.30 |
| 50-step arithmetic | 0.90 | 50 | 3 | 150 | 3.0 | 0.77 | - |
| 100-step arithmetic | 0.90 | 100 | 4 | 400 | 4.0 | 0.87 | - |

### Symbolic Math

| Task | p | Steps | k | Samples | Samples/Step | Error % |
|------|---|-------|---|---------|-------------|---------|
| Chain rule (5 steps) | 0.85 | 5 | 3 | 15 | 3.0 | 0.00% |
| Chain rule (15 steps) | 0.85 | 15 | 4 | 60 | 4.0 | 0.00% |

### Logic Puzzles

| Task | p | Steps | k | Samples | Samples/Step | Error % |
|------|---|-------|---|---------|-------------|---------|
| Sudoku validation | 0.88 | 9 | 3 | 27 | 3.0 | 0.00% |
| Logic grid | 0.85 | 20 | 4 | 80 | 4.0 | 0.00% |

### Tower of Hanoi

| Task | p | Steps | k | Samples | Samples/Step | Error % |
|------|---|-------|---|---------|-------------|---------|
| 3-disk Hanoi | 0.90 | 7 | 2 | 14 | 2.0 | 0.00% |
| 5-disk Hanoi | 0.90 | 31 | 3 | 93 | 3.0 | 0.00% |
| 7-disk Hanoi | 0.90 | 127 | 4 | 508 | 4.0 | 0.00% |

**Cost Scaling:** R^2 = 0.9956 for cost vs s*ln(s) — confirms Theta(s ln s) scaling.

---

## Data Analysis

Data analysis pipeline tasks with both exact and approximate matching.

**Parameters:** 500 trials, target reliability 95%

| Task | Category | p | Steps | Match | Samples | Error % |
|------|----------|---|-------|-------|---------|---------|
| CSV column select | csv | 0.92 | 3 | exact | 6 | 0.00% |
| CSV filter rows | csv | 0.90 | 4 | exact | 8 | 0.00% |
| CSV join tables | csv | 0.85 | 5 | exact | 15 | 0.00% |
| Mean/median/mode | statistics | 0.88 | 3 | approx | 6 | 0.00% |
| Correlation matrix | statistics | 0.82 | 6 | approx | 18 | 0.00% |
| Regression coefficients | statistics | 0.80 | 5 | approx | 15 | 0.00% |
| SQL SELECT query | sql | 0.88 | 3 | exact | 6 | 0.00% |
| SQL GROUP BY | sql | 0.83 | 4 | exact | 12 | 0.00% |
| Null handling | cleaning | 0.90 | 4 | exact | 8 | 0.00% |
| Type coercion | cleaning | 0.87 | 5 | exact | 15 | 0.00% |

**Results:** 100% accuracy across all data analysis tasks (exact and approximate matching).

---

## Cost Scaling

Validates Theta(s ln s) cost scaling across Towers of Hanoi task sizes.

**Parameters:** p=0.85, t=0.95, 20 trials

| Disks | Steps (s) | k | Mean Samples | s*ln(s) |
|-------|-----------|---|-------------|---------|
| 3 | 7 | 2 | 14 | 13.6 |
| 5 | 31 | 3 | 93 | 106.5 |
| 7 | 127 | 4 | 508 | 615.2 |

R^2 > 0.99 — cost scales as Theta(s ln s).

---

## Ensemble Comparison

Multi-model ensemble vs single-model reliability and cost (1,000 trials).

| Configuration | k | Steps | Mean Samples | Mean Cost ($) | Error % |
|--------------|---|-------|-------------|--------------|---------|
| Single: llama3 (p=0.80) | 5 | 100 | 500 | 0.50 | 0.00% |
| Single: Haiku (p=0.90) | 3 | 100 | 300 | 3.00 | 0.00% |
| Ensemble RR: llama3+Haiku | 3 | 100 | 300 | 1.65 | 0.00% |
| Ensemble CA: llama3+Haiku | 3 | 100 | 300 | 0.38 | 0.00% |

**Cost-aware ensemble saves 87.5%+ vs expensive-only model** (target: >= 30%).

---

## Acceptance Criteria Summary

| Criterion | Status |
|-----------|--------|
| Coding: >90% accuracy on trivial/moderate tasks | PASS (100%) |
| Coding: >80% accuracy on complex tasks | PASS (100%) |
| Math: Zero errors on arithmetic with k>=3 | PASS |
| Math: Theta(s ln s) cost scaling (R^2 > 0.95) | PASS (R^2=0.9956) |
| Data: >85% accuracy on data analysis tasks | PASS (100%) |
| Data: Approximate matching accuracy >85% | PASS (100%) |
| Ensemble: Error rate <= single-model | PASS |
| Ensemble: Cost-aware saves >= 30% vs expensive-only | PASS (87.5%) |
| All benchmarks execute without crashes | PASS |

---

*Generated by MAKER benchmark suite. Run `cargo bench` to reproduce.*
