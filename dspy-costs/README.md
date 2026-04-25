# DSPy Cost Estimator

Estimates API costs for this project by scanning `results/` and computing
prices against current provider rates. Prefers **measured** token usage
logged by `scripts/evaluate_optimized.py` (via `dspy.configure(track_usage=True)`),
falling back to filename-based estimation when a run predates usage logging.

## Usage

```bash
.venv/bin/python dspy-costs/estimate_costs.py
.venv/bin/python dspy-costs/estimate_costs.py --retry-multiplier 1.5
.venv/bin/python dspy-costs/estimate_costs.py --save dspy-costs/results_myrun.json
```

The `M` column in the per-benchmark table flags measured rows (`✓`). Rows
without the flag are estimates from filename heuristics.

## How it works

1. Scans `results/{benchmark}/{baseline,optimized}/` for JSON files
2. For each `*_test_scores.json`:
   - If a top-level `usage` key is present, uses those token counts directly
   - Otherwise estimates calls via the optimizer/module heuristics below
3. For baseline and optimization runs (no `usage` logged yet):
   - **Baseline**: one call per image in the full dataset
   - **MIPROv2**: `trials × dev_set_size + 50` overhead (bootstrap + instruction proposals)
   - **SIMBA/GEPA**: `train_size × 10` reflection rounds
   - **LOO folds**: per-fold MIPROv2 × number of folds
   - **Refine wrapper**: test calls × 3 (max retries)
4. Estimates tokens-per-call from hardcoded per-benchmark averages when measured data is unavailable
5. Multiplies by a retry factor (default 1.3×) for rate-limit retries, debugging, failed runs
6. Emits a report plus a structured JSON (`dspy-costs/results.json` by default)

## Pricing (USD per 1M tokens, retrieved 2026-04-24)

### Google Gemini — AI Studio

| Model | Input | Output |
|---|---|---|
| gemini-3.1-pro-preview | $2.00 | $12.00 |
| gemini-3-pro-preview | $2.00 | $12.00 |
| gemini-3-flash-preview | $0.50 | $3.00 |
| gemini-3.1-flash-lite-preview | $0.25 | $1.50 |
| gemini-2.5-pro | $1.25 | $10.00 |
| gemini-2.5-flash | $0.30 | $2.50 |
| gemini-2.0-flash (deprecated, shuts down 2026-06-01) | $0.10 | $0.40 |

Tiered: Gemini 2.5 Pro and 3.x Pro Preview double above 200k input tokens
(our prompts are well under). Every Gemini 3.x variant carries the `-preview`
suffix as of 2026-04-24 — no GA Flash 3 yet. Gemini 1.5 Pro is deprecated
on the Gemini API.

### Google Gemini — Vertex AI

| Model | Input | Output |
|---|---|---|
| gemini-3-pro-preview | $2.00 | $12.00 |
| gemini-2.5-pro | $1.25 | $10.00 |
| gemini-2.5-flash | $0.30 | $2.50 |
| gemini-2.0-flash | $0.15 | $0.60 |
| gemini-1.5-pro | $1.25 | $5.00 |

For Gemini rows, the report shows AI Studio as the primary and surfaces the
Vertex AI alternative price beneath it (with the delta).

### Anthropic

| Model | Input | Output |
|---|---|---|
| claude-sonnet-4-5 | $3.00 | $15.00 |
| claude-haiku-3-5 | $1.00 | $5.00 |

### OpenAI

| Model | Input | Output |
|---|---|---|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |

### OpenRouter

Pass-through at upstream list prices with no per-token markup (a separate
credit-purchase fee applies). Resolves to the same rates as the underlying
provider (Anthropic / OpenAI / Google).

## Pricing sources

- AI Studio:  https://ai.google.dev/gemini-api/docs/pricing
- Vertex AI:  https://cloud.google.com/vertex-ai/generative-ai/pricing
- Anthropic:  https://claude.com/pricing
- OpenAI:     https://platform.openai.com/docs/pricing
- OpenRouter: https://openrouter.ai/models

## Limitations

- For runs without a `usage` key (everything before this change landed),
  token counts are still averages based on per-benchmark heuristics
- The retry multiplier is a rough heuristic — actual retries depend on
  rate-limit errors encountered
- Image token cost is folded into the prompt token estimate (~258 tok/image)
- Tiered "above 200k tokens" pricing is not modelled — all inputs assumed
  to be in the lower tier (safe for this project's images)
