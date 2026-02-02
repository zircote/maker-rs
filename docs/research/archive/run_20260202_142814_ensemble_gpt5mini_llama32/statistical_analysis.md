# Statistical Analysis

## Hypothesis Tests

**H0**: Voting accuracy = Single sample accuracy
**H1**: Voting accuracy > Single sample accuracy

### ensemble

- **n**: 12
- **Voting mean**: 50.0%
- **Single mean**: 16.7%
- **t-statistic**: 1.7728
- **p-value**: 0.076260
- **Cohen's d**: 0.5118 (medium)
- **Conclusion**: Fail to reject H0

### ollama_only

- **n**: 12
- **Voting mean**: 25.0%
- **Single mean**: 16.7%
- **t-statistic**: 1.0000
- **p-value**: 0.317311
- **Cohen's d**: 0.2887 (small)
- **Conclusion**: Fail to reject H0

### openai_only

- **n**: 12
- **Voting mean**: 100.0%
- **Single mean**: 100.0%
- **t-statistic**: 0.0000
- **p-value**: 1.000000
- **Cohen's d**: 0.0000 (negligible)
- **Conclusion**: Fail to reject H0

