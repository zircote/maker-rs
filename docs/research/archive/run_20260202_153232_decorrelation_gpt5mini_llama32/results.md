# Research Validation Results

Generated: Unix timestamp: 1770064334

## Summary Table

| Config | n | Voting Acc | Single Acc | Improvement | Cohen's d | p-value | Sig |
|--------|---|------------|------------|-------------|-----------|---------|-----|
| decorr_ensemble_k1 | 24 | 58.3% (36.4-80.2) | 20.8% | 2.80x | 0.76 | 0.0002 | *** |
| decorr_ensemble_k2 | 24 | 62.5% (41.0-84.0) | 20.8% | 3.00x | 0.64 | 0.0018 | ** |
| decorr_ensemble_k3 | 24 | 54.2% (32.0-76.3) | 20.8% | 2.60x | 0.52 | 0.0104 | * |
| decorr_ollama_k1 | 24 | 25.0% (5.8-44.2) | 20.8% | 1.20x | 0.20 | 0.3173 |  |
| decorr_ollama_k2 | 24 | 25.0% (5.8-44.2) | 20.8% | 1.20x | 0.20 | 0.3173 |  |
| decorr_ollama_k3 | 24 | 25.0% (5.8-44.2) | 20.8% | 1.20x | 0.12 | 0.5692 |  |
| decorr_openai_k1 | 24 | 100.0% (100.0-100.0) | 100.0% | 1.00x | 0.00 | 1.0000 |  |
| decorr_openai_k2 | 24 | 100.0% (100.0-100.0) | 100.0% | 1.00x | 0.00 | 1.0000 |  |
| decorr_openai_k3 | 24 | 100.0% (100.0-100.0) | 100.0% | 1.00x | 0.00 | 1.0000 |  |

## Statistical Notes

- **Sig column**: *** p<0.001, ** p<0.01, * p<0.05
- **Cohen's d interpretation**: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large
- Confidence intervals are 95% Wilson score intervals for proportions
