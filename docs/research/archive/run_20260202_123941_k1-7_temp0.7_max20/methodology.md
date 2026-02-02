# Methodology

## Experimental Design

This validation study evaluates MAKER's error correction through
SPRT-based voting using controlled mock experiments.

### Parameters

- **Trials per configuration**: 20
- **RNG Seed**: 42
- **Mode**: Live (Ollama)

### Experiments

1. **Accuracy Comparison**: k-margin voting vs single sample
   - k values: 1, 2, 3, 5, 7
   - p values: 0.6, 0.7, 0.8, 0.9
2. **Multi-Step Reliability**: Task success over multiple steps
   - Steps: 1, 3, 5, 10
   - k=3, p=0.85
3. **Convergence Analysis**: Samples required for k-margin
   - k values: 1, 2, 3, 5, 7, 10
4. **Cost Scaling**: Total samples vs step count
   - Steps: 7, 31, 127, 511

## Statistical Methods

- **Mean/Std**: Sample mean with Bessel-corrected standard deviation
- **95% CI**: t-distribution confidence intervals
- **Paired t-test**: Two-tailed test comparing voting vs single sample
- **Cohen's d**: Effect size for paired samples
- **Wilson CI**: Confidence intervals for proportions

## Reproducibility

```bash
cargo run --bin research-validation -- --mock --seed 42 --trials 20
```

## Limitations

- Mock experiments use pseudo-random sequences, not true LLM behavior
- Error correlation assumed to be independent (may not hold for real LLMs)
- Ensemble experiments simulate decorrelation effect
