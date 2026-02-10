# DSPy Cost Estimator

Estimates Google Gemini API costs for this project by scanning `results/` for experiment outputs and reconstructing token usage.

## Usage

```bash
python dspy-costs/estimate_costs.py
python dspy-costs/estimate_costs.py --retry-multiplier 1.5
python dspy-costs/estimate_costs.py --save dspy-costs/results_myrun.json
```

## How it works

1. Scans `results/{benchmark}/baseline/` and `results/{benchmark}/optimized/` for JSON files
2. Identifies the model and optimizer type from each filename
3. Estimates API calls per experiment:
   - **Baselines**: 1 call per image in the full dataset
   - **MIPROv2**: `trials × dev_set_size + 50` overhead (bootstrapping + instruction generation)
   - **SIMBA/GEPA**: `train_size × 10` (mini-batch reflection rounds)
   - **LOO folds**: per-fold MIPROv2 optimization × number of folds
   - **Refine wrapper**: test calls × 3 (max retries)
   - **Test evaluations**: counted from `per_image` entries in `*_test_scores.json`
4. Estimates tokens per call using benchmark-specific averages (image tokens, prompt size, demo count, output length)
5. Multiplies by a retry factor (default 1.3×) for rate-limit retries, debugging, and failed runs
6. Computes costs under both **AI Studio** and **Vertex AI** pricing

## Token cost sources

Pricing retrieved February 2026:

| Model | AI Studio (in/out per 1M) | Vertex AI (in/out per 1M) |
|---|---|---|
| Gemini 1.5 Pro | $1.25 / $5.00 | $1.25 / $5.00 |
| Gemini 2.0 Flash | $0.10 / $0.40 | $0.15 / $0.60 |
| Gemini 2.5 Flash | $0.15 / $0.60 | $0.15 / $0.60 |
| Gemini 2.5 Pro | $1.25 / $10.00 | $1.25 / $10.00 |
| Gemini 3 Pro Preview | $2.00 / $12.00 | $2.00 / $12.00 |

- AI Studio: https://ai.google.dev/gemini-api/docs/pricing
- Vertex AI: https://cloud.google.com/vertex-ai/generative-ai/pricing

Gemini 2.0 Flash has a **free tier** on AI Studio with rate limits (1,500 RPD, 1M TPM). If all Flash usage fell within the free tier, subtract the Flash line from the AI Studio total.

## Limitations

- Token counts are estimates based on average prompt/output sizes, not actual logged usage
- Image token cost uses ~258 tokens per image (Gemini's standard image tokenization)
- The retry multiplier is a rough heuristic — actual retries depend on rate limit errors encountered
- Does not account for non-Gemini API costs (OpenAI, Anthropic, OpenRouter)
