# Run Parameters

- **Date**: Mon Feb  2 12:39:48 EST 2026
- **Model**: llama3.2:3b
- **Provider**: ollama
- **Trials**: 20
- **Questions**: 12 (harder multi-step arithmetic)
- **k values**: 1, 2, 3, 5, 7
- **max_samples**: 20
- **diversity_temperature**: 0.7
- **single_temperature**: 0.7

## Key Findings

- k1-k3: Statistically significant improvement (p < 0.05)
- k5-k7: No improvement (many non-convergences)
- Overall accuracy: ~20-25% (questions too hard)
- Voting improvement: 1.2-1.27x at low k values
