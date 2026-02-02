# Run Parameters

- **Date**: Mon Feb  2 12:52:13 EST 2026
- **Model**: llama3.2:3b
- **Provider**: ollama
- **Trials**: 20
- **Questions**: 12 (harder multi-step arithmetic)
- **k values**: 1, 2, 3 (validated sweet spot)
- **max_samples**: 20
- **diversity_temperature**: 0.7
- **single_temperature**: 0.7

## Key Findings

- k1: 25.0% vs 21.2%, p=0.0116 (significant)
- k2: 25.0% vs 22.9%, p=0.1647 (not significant)
- k3: 22.9% vs 24.2%, p=0.3659 (not significant)
- k=1 most robust - always converges
