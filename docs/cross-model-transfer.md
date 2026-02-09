# Cross-Model Transfer: Applying Flash-Optimized Programs to Leaderboard Models

*Date: 2026-02-09*

## Question

Can the MIPROv2-optimized programs (created for Gemini 2.0 Flash) be applied to the current RISE leaderboard-winning models (GPT-5, GPT-4.1, GPT-4o, Gemini 3 Pro, etc.) to improve their scores?

## Short Answer

Technically trivial — a single `--model` flag change. Strategically promising for Personnel Cards and Business Letters. Diminishing returns likely on Library Cards where the leaderboard top is already high.

## Technical Feasibility

### Programs are model-agnostic

Saved optimized programs (JSON) contain only:
- Instructions (text)
- Field descriptions (text)
- Few-shot demonstrations (images + text)
- Reasoning traces (for CoT)

No model identifiers are hardcoded. Transfer is a single flag change:

```bash
uv run python scripts/evaluate_optimized.py \
  --benchmark library_cards \
  --model gpt-4o \
  --module cot \
  --program results/library_cards/optimized/mipro-cot_gemini-2.0-flash_optimized.json
```

### Models already configured

Available presets in `benchmarks/shared/config.py`:
- `gpt-4o` → `openai/gpt-4o`
- `claude-sonnet` → `anthropic/claude-sonnet-4-5-20250929`
- `gemini-3-pro-preview` → `gemini/gemini-3-pro-preview`
- `gemini-2.5-pro` → `gemini/gemini-2.5-pro`
- `gemini-2.5-flash` → `gemini/gemini-2.5-flash`

Missing: GPT-5, GPT-4.1, GPT-4.5, o3 — but any litellm model string works directly.

### Scoring is model-independent

All scoring (fuzzy matching, F1, set matching) depends only on predicted vs ground-truth structure. No model-specific logic.

## Empirical Evidence: Library Cards Pro→Flash Transfer

| Configuration | f1_macro |
|---|---|
| Gemini 2.5 Pro (native optimization) | 0.8912 |
| Gemini 2.0 Flash (using Pro program) | 0.8743 (-1.7 pts) |
| Gemini 2.0 Flash (native optimization) | 0.9017 (+2.7 pts over transfer) |

Key takeaways:
- Transfer works but leaves ~2.7 pts on the table vs native optimization
- This was expensive→cheap direction. Cheap→expensive (Flash→GPT-5) is untested but should work at least as well

## Where Transfer Could Move the Needle

### Current standings vs our best

| Benchmark | Our best (Flash+Refine) | Leaderboard #1 | Gap |
|---|---|---|---|
| Library Cards | 0.9167 | GPT-5: 0.895 | +2.2 pts above |
| Personnel Cards | 0.8894 | ~0.790 (2.5 Pro) | +9.9 pts above |
| Business Letters | 0.7312 | GPT-5: 0.770 | -3.9 pts below |
| Bibliographic Data | 0.7072 | GPT-4o: 0.714 | -0.7 pts below |

### Most promising targets

1. **Personnel Cards** (+9.9 pts over leaderboard): Even lossy transfer to a stronger model would likely beat the leaderboard significantly
2. **Business Letters** (-3.9 pts below GPT-5): A stronger model might close this gap; the Flash-optimized demos teach "First Last" format which benefits any model
3. **Library Cards** (+2.2 pts above): Less room to grow but transfer could confirm ceiling
4. **Bibliographic Data**: Bottleneck is positional scoring, not model capability — transfer unlikely to help

## Important Caveats

### Comparability problem
- Leaderboard evaluates on ALL images with no train/dev split
- Our scores are on 70% held-out test sets
- Fair comparison requires either:
  - Running transferred program on all images (slightly leaky — some images were demos)
  - Leaderboard adopting train/test splits

### Diminishing returns on strong models
- Project's clearest finding: optimization helps cheap models more than expensive ones
- Flash gained +14.3 pts (Library Cards), Pro gained +7.4 pts
- GPT-5 at 89.5 likely has less room than Flash at 81.3

### Refine needs ground truth
- Refine(3) gave biggest boosts (+9.3 pts on Business Letters)
- But it requires GT labels at inference time for quality-aware reward
- Transfer of base MIPROv2 program (without Refine) is the realistic scenario
- Without Refine, our scores are: LC 0.9017, BD 0.7072, PC 0.8858, BL 0.6378

### Why programs don't transfer perfectly
Programs encode two kinds of knowledge:
1. **Task knowledge** — what to extract, format rules, schema conventions → transfers perfectly
2. **Model-specific compensation** — workarounds for Flash's weaknesses (explicit JSON formatting, verbose instructions) → may be unnecessary or slightly counterproductive for stronger models

Native re-optimization finds the optimal specificity level for the target model.

## Cost Estimates

### Evaluation-only transfer (Phase 1)
- ~test_set_size API calls per benchmark
- GPT-4o: ~$25-50 per benchmark
- Gemini 2.5 Pro: ~$5-10 per benchmark

### Re-optimization with MIPROv2 medium (Phase 2)
- ~18 trials × dev_set_size API calls
- GPT-4o: ~$100-300 per benchmark
- Would recover ~2-5 pts over transfer based on Library Cards precedent

### Rate limit considerations
- Evaluation (single pass) is much less demanding than optimization
- GPT-4o had 30K TPM limit issues during optimization but should be fine for evaluation
- Gemini 2.5 Pro has generous limits (1M+ TPM)

## Recommended Experiment Plan

### Phase 1: Quick transfer test
Run Flash-optimized programs on GPT-4o and Gemini 2.5 Pro (both available, reasonable cost).

```bash
# For each benchmark × model combination:
uv run python scripts/evaluate_optimized.py \
  --benchmark {benchmark} \
  --model {model} \
  --module cot \
  --program results/{benchmark}/optimized/mipro-cot_gemini-2.0-flash_optimized.json

# Also run unoptimized baselines for comparison:
uv run python scripts/evaluate_baseline.py \
  --benchmark {benchmark} \
  --model {model} \
  --module cot
```

Priority order: Personnel Cards (biggest gap) → Business Letters → Library Cards → Bibliographic Data

### Phase 2: Conditional re-optimization
If Phase 1 shows:
- Transfer beats baseline but gap persists → Run MIPROv2 medium on target model
- Transfer matches/exceeds leaderboard → No re-optimization needed
- Transfer doesn't improve baseline → Investigate prompt compatibility

### Phase 3: Leaderboard submission
If results are strong, evaluate on full dataset (all images) for leaderboard comparability.
Note: few-shot demos were drawn from the training split, so running on all images is slightly leaky but standard practice for leaderboard submissions.
