# Experiment: Cross-Model Evaluation

**Branch**: `experiment/cross-model-eval`
**Purpose**: Test whether MIPROv2-optimized programs (trained on Gemini 2.0 Flash) transfer to newer models without re-optimizing.

## Results: Gemini 2.5 Flash

| Benchmark | 2.0 Flash Best | 2.5 Flash Best | Delta | vs RISE #1 |
|-----------|---------------|---------------|-------|------------|
| Library Cards | 91.67* | **92.58*** | +0.9 | beats 89.5 |
| Business Letters | 73.12* | **80.87*** | +7.8 | **beats 77.0** |
| Personnel Cards | 88.94* | 88.74* | -0.2 | beats ~79.0 |
| Company Lists | 87.71 | 86.82 | -0.9 | beats 58.4 |
| Blacklist Cards | 97.13* | 94.74* | -2.4 | beats 95.7 (2.0 only) |
| Bibliographic Data | 70.72 | 46.07 | -24.7 | neither beats 71.4 |

\* = with Refine(3)

## Key Findings

- **Business Letters**: biggest winner (+7.8 pts). 2.5 Flash follows "First Last" name format better. Now beats RISE leaderboard #1.
- **Library Cards**: new overall best (92.58). Slight OCR improvement.
- **Bibliographic Data**: collapsed due to page 10 parse failure. With only 2 test images, one failure halves the score.
- **Prompt optimization is partially model-transferable**: 4/6 benchmarks within 3 pts. Instructions+demos carry useful signal, but format compliance can break.

## Score Files

All score files use the `--output-tag gemini-2.5-flash` naming convention:
- Base: `*_optimized_gemini-2.5-flash_test_scores.json`
- Refine: `*_optimized_refine3_gemini-2.5-flash_test_scores.json`

## Evaluation Commands

```bash
# Base evaluation (all 6 benchmarks)
.venv/bin/python scripts/evaluate_optimized.py \
  --benchmark <name> \
  --program results/<name>/optimized/mipro-*_gemini-2.0-flash_optimized.json \
  --model gemini-2.5-flash --module cot --output-tag gemini-2.5-flash

# With Refine (4 benchmarks: library_cards, personnel_cards, business_letters, blacklist_cards)
# Add: --refine 3
```

## Next Steps

- [ ] Test with Gemini 2.5 Pro (use `--model gemini-2.5-pro --output-tag gemini-2.5-pro`)
- [ ] Consider re-optimizing on 2.5 Flash for benchmarks where it improved (Library Cards, Business Letters)
- [ ] Investigate Bibliographic Data parse failure on 2.5 Flash
