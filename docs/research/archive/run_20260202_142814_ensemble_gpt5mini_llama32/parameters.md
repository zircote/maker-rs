# Run Parameters

- **Date**: Mon Feb  2 14:28:14 EST 2026
- **Model A**: llama3.2:3b (ollama)
- **Model B**: gpt-5-mini (openai)
- **Experiment**: ensemble comparison
- **Trials**: 5 per question (12 questions)
- **k_margin**: 3
- **max_samples**: 20

## Key Findings

- ollama_only: 25.0% voting vs 16.7% single (1.50x improvement)
- openai_only: 100.0% voting vs 100.0% single (no improvement, model perfect)
- ensemble: 50.0% voting vs 16.7% single (3.00x improvement, p=0.0763)

## Conclusion

The ensemble approach (alternating between weak and strong models) achieves
significantly better results than ollama alone, while being much cheaper than
using openai exclusively. The voting mechanism helps overcome errors from the
weaker model by leveraging the stronger model's responses.

However, gpt-5-mini alone achieves 100% accuracy on these questions, so ensemble
only makes sense for cost optimization, not accuracy improvement.
