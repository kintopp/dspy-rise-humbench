# Changelog

All notable changes to this project are documented here. Since the project
was not versioned, entries are grouped by date.

---

## 2026-02-08

### Added
- DSPy best practices analysis document (`docs/dspy-best-practices.md`).
- Reflection model name now included in GEPA output filenames for traceability
  (e.g., `gepa-cot_gemini-2.0-flash_reflect-gemini-2.5-pro_optimized.json`).

### Changed
- README restructured: all benchmark sections now use consistent Phase 1 / Phase 2
  numbering (optimizer comparison → GEPA with stronger reflection model).
- GEPA results integrated into each benchmark section and the cross-benchmark
  combined results table.
- Future Work: GEPA section rewritten from proposed experiment to completed
  experiment summary with cross-benchmark conclusions.

---

## 2026-02-07

### Library Cards — Experiment Phases 1–3

- **Phase 1 (Gemini 2.5 Pro ceiling):** MIPROv2 light with Predict achieved
  f1_macro = 0.8912, competitive with the RISE leaderboard top scores.
- **Phase 2 (Flash experiment matrix):** Added ChainOfThought module support,
  MIPROv2 medium/heavy search budgets, SIMBA and GEPA optimizers, cross-model
  transfer evaluation, and Refine inference-time wrapper. Fixed Gemini Flash
  JSON code fence wrapping and GEPA metric arithmetic (`FeedbackScore` class).
  Best result: **MIPROv2 medium-CoT = 0.9017 f1_macro** (+14.3 pts over
  baseline). Removed early GPT-4o baseline from comparison (rate-limited run
  was not representative).
- **Phase 3 (GEPA with stronger reflection):** Re-ran GEPA medium-CoT with
  Gemini 2.5 Pro as the reflection model (23 iterations). No candidate
  instruction beat the base CoT program — Library Cards' diverse formats
  require few-shot demonstrations, not instructions alone.

### Multi-Benchmark Architecture

- Refactored from single-benchmark codebase (`src/`) to a plugin architecture
  under `benchmarks/`. Each benchmark is a self-contained package exporting
  `Extractor`, `load_and_split`, `dspy_metric`, `score_single_prediction`,
  `compute_aggregate_scores`, etc.
- Added `benchmarks/shared/` with `config.py` (LM setup, model presets,
  `results_dir()`) and `scoring_helpers.py` (generic fuzzy scoring, key
  traversal, `FeedbackScore`).
- All scripts now accept `--benchmark` flag with dynamic imports via
  `importlib`.
- Deleted `src/` directory.

### Bibliographic Data Benchmark

- New benchmark package (`benchmarks/bibliographic_data/`) with schema,
  signature, data loader, module, and scoring for 5-image bibliography dataset.
- Ground truth normalization: CSL-JSON hyphenated keys converted to
  underscores; type values normalized (`article-journal` → `journal-article`,
  `chapter` → `book`, `review` → `journal-article`).
- Best result: **MIPROv2 heavy-CoT = 0.7072 avg fuzzy** (2/1/2 split).
- Leave-one-out cross-validation with `scripts/loo_mipro.py` (aggregate
  0.6969 — did not improve over standard split).
- Discovered bimodal score distribution: 3 pages at 0.89–0.91, 2 pages at
  ~0.39 due to cascading alignment errors in position-based scoring.
- Filed upstream issues:
  [#91](https://github.com/RISE-UNIBAS/humanities_data_benchmark/issues/91)
  (GT inconsistencies),
  [#92](https://github.com/RISE-UNIBAS/humanities_data_benchmark/issues/92)
  (scoring methodology).

### Personnel Cards Benchmark

- New benchmark package (`benchmarks/personnel_cards/`) for 61 Swiss Federal
  personnel card images with nested row/sub-field schema.
- Best result: **MIPROv2 medium-CoT = 0.8858 f1_macro** (+25.6 pts over
  predict baseline). Exceeds previously reported leaderboard top (~79.0) by
  ~10 points.
- CoT helped the unoptimized baseline (+16.9 pts) — JSON parse failures
  dropped from 8/43 to 3/43 zero-scoring cards.
- GEPA medium-CoT with Gemini 2.5 Pro reflection (84 iterations) achieved
  0.8750 f1_macro — within 1.1 pts of MIPROv2, the closest any optimizer came
  on any benchmark.

### Business Letters Benchmark

- New benchmark package (`benchmarks/business_letters/`) for 57 multi-page
  letters (98 page images) with category-level set matching against a
  `persons.json` alias table.
- Prompt schema changed from "Last, First" to "First Last" name format to
  match alias table conventions (+18 pts baseline lift).
- Best result: **MIPROv2 medium-CoT = 0.6378 f1_macro** (+18.1 pts over
  predict baseline).
- GEPA medium-CoT with Gemini 2.5 Pro reflection (96 iterations) achieved
  0.5472 — severe dev→test overfitting (-34.9 pts gap) with only 8 dev
  letters.

### Documentation and Infrastructure

- Mermaid pipeline architecture diagram (`docs/pipeline.svg`).
- `CLAUDE.md` project instructions (scoring comparability policy, benchmark
  conventions).
- MIT license.
- Gemini API cost estimation script (`scripts/estimate_cost.py`) covering all
  4 benchmarks with AI Studio vs Vertex AI pricing.
- Optimized prompt documentation for all benchmarks under `docs/`.
- RISE leaderboard references updated with fresh dashboard data.
- `offline/` directory added to `.gitignore`.

---

## 2026-02-06

### Added
- Initial commit: DSPy optimization pipeline for RISE Humanities Data
  Benchmarks.
- Library Cards benchmark with schema, signature, data loader, module, and
  F1-based scoring (263 images, field-level fuzzy matching at 0.92 threshold).
- Scripts: `evaluate_baseline.py`, `optimize.py`, `evaluate_optimized.py`,
  `compare_results.py`, `check_rate_limits.py`.
- Multi-provider support via litellm (OpenAI, Google, Anthropic, OpenRouter).
- Rate limit findings: GPT-4o's 30K TPM limit severely degraded MIPROv2
  optimization; Gemini 2.5 Pro ran without issues.
