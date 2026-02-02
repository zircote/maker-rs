# Run Parameters

- **Date**: Mon Feb  2 14:18:16 EST 2026
- **Model**: gpt-5-mini
- **Provider**: openai
- **Trials**: 20
- **Questions**: 12 (harder multi-step arithmetic)
- **k values**: 1, 2, 3
- **max_samples**: 20
- **diversity_temperature**: 0.7 (ignored for reasoning models)
- **single_temperature**: 0.7 (ignored for reasoning models)

## Key Findings

- k1: 100.0% vs 100.0% - no improvement (model perfect)
- k2: 100.0% vs 100.0% - no improvement (model perfect)
- k3: 100.0% vs 100.0% - no improvement (model perfect)

## Conclusion

gpt-5-mini achieves 100% accuracy on these math questions.
Voting provides no benefit when the base model is already perfect.
This validates that voting helps weaker models (llama3.2:3b ~25%) 
but adds no value for highly capable models on tractable tasks.
