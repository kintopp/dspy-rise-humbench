# Changelog

All notable changes to this project are documented here. Since the project
is not versioned, entries are grouped by date.

---

## 2026-02-10

### Changed
- **README streamlined**: Individual benchmark sections cut from ~380 to ~225
  lines (~41% reduction). Removed experiment-phase headers and detailed
  subsections for GEPA, KNN, MultiChainComparison, two-stage pipeline, and
  verify-and-correct. Each benchmark now shows a simplified results table
  (baseline → best) and 2 key findings. Cross-benchmark summary table dropped
  GEPA column; "Task structure" paragraph generalized to "instruction-only
  optimisation." Optimizer description section also simplified for readability.
- **Demo visualizations updated**: Bibliographic Data demo now shows all 5
  images (was 3); added Refine comparison columns for Bibliographic Data and
  Company Lists demos.
- **Business Letters demo**: added letter60 (the residual format-mismatch case
  flagged by upstream reviewer on issue #93); fixed note background color in
  Bibliographic Data demo.

### Added
- **Issue #93 response draft** (`offline/issue93-response-draft.md`): analysis
  of upstream reviewer's comments on Business Letters inferred-sender claims.
  Verified all claims correct (14/41 test letters with angle-bracket markers,
  24/57 total). Max's persons.json alias update has zero impact on scores.
  Cross-benchmark spot-check confirmed all 15 factual claims across 6
  benchmarks.

---

## 2026-02-09

### Added
- **Company Lists benchmark** (`benchmarks/company_lists/`): 6th benchmark added
  to the pipeline. 15 images of printed Swiss company trade index pages
  (1925-1958) with 15-31 entries per page. F1 metric with hybrid scoring:
  0.92 fuzzy threshold (Library Cards pattern) + null normalization (Blacklist
  Cards pattern). First multi-input benchmark (`page_image` + `page_id`).
  Best result: **MIPROv2 medium-CoT = 0.8771 f1_macro** (+11.3 pts over
  predict baseline), far exceeding the upstream leaderboard top (GPT-5 at
  58.40). Refine(3) hurt here (-1.1 pts) due to near-threshold page
  regressions on retry.
- **Multi-input support** in `evaluate_baseline.py`, `evaluate_optimized.py`,
  and `generate_demo_data.py`: replaced single `input_field` lookup with
  `input_keys` dict comprehension, enabling benchmarks with multiple input
  fields (backward-compatible with all existing single-input benchmarks).
- **Blacklist Cards benchmark** (`benchmarks/blacklist_cards/`): 5th benchmark
  added to the pipeline. 33 images of 1940s British blacklist index cards with
  average fuzzy metric. Near-ceiling baseline (93.0) pushed to **97.1** with
  MIPROv2 medium-CoT + Refine(3), exceeding the upstream leaderboard top (~95.7).
- **Module-level experiments** across four benchmarks:
  - **KNN dynamic demo selection** (`benchmarks/library_cards/knn_module.py`,
    `scripts/evaluate_knn.py`): two-pass inference with nearest-neighbor demo
    retrieval. Tested with both `sentence-transformers/all-MiniLM-L6-v2`
    (English-only) and `gemini-embedding-001` (multilingual) — identical
    results: 0.9017 f1_macro (equal to static MIPROv2 demos — no improvement).
  - **MultiChainComparison** (`benchmarks/personnel_cards/multi_chain_module.py`,
    `scripts/evaluate_multi_chain.py`): M=3 diverse attempts synthesized by a
    comparator. Includes `FullMultiChainComparison` subclass fixing DSPy's
    truncation bug on multi-line JSON. Result: 0.8763 f1_macro (-0.95 pts below
    MIPROv2 CoT).
  - **Verify-and-correct** (`benchmarks/business_letters/verify_module.py`,
    `benchmarks/business_letters/tools.py`, `scripts/evaluate_verify.py`):
    post-hoc person name lookup against `persons.json` alias table. Result:
    0.6378 f1_macro (neutral — zero corrections with exact-match-only mode).
  - **Two-stage pipeline** (`benchmarks/bibliographic_data/two_stage_module.py`,
    `benchmarks/bibliographic_data/two_stage_signature.py`): split OCR from
    structuring. Result: 0.6265 fuzzy (worse than single-stage 0.7072).
- **New evaluation scripts**: `evaluate_knn.py`, `evaluate_multi_chain.py`,
  `evaluate_verify.py` for the module-level experiments.
- **HumBench export script** (`scripts/export_humbench_results.py`): runs the
  optimized DSPy program on all letters and outputs per-letter JSONs in the
  upstream `humanities_data_benchmark` results format, including `usage`,
  `duration`, `score`, and `scoring.json`. Supports `--refine N` for
  quality-aware retries. Exported results in `results/business_letters/export/`
  (f1_macro = 0.74 with Refine(3)).
- **RISE infrastructure run (T0623)**: ran the MIPROv2-optimized prompt through
  the upstream `humanities_data_benchmark` pipeline for native-format results
  with all fields (`raw_response`, `conversation_id`, etc.). Scored f1_macro =
  0.52 — lower than DSPy pipeline (0.64 single-pass / 0.73 with Refine) due to
  loss of explicit chain-of-thought (structured output forces direct JSON),
  multi-modal few-shot demos, and Refine retries.
- **Company Lists demo visualization** (`results/demo/demo_company_lists.html`):
  added 6th benchmark to the interactive HTML demo pages and updated the index.
- **Company Lists optimized prompt documentation**
  (`docs/optimized-prompt-company-lists.md`): persona-framed instruction with
  2 demos covering alphabetical and thematic page types, dual input fields.
- **Blacklist Cards demo visualization** (`results/demo/demo_blacklist_cards.html`):
  added 5th benchmark to the interactive HTML demo pages and updated the index.
- **Blacklist Cards optimized prompt documentation**
  (`docs/optimized-prompt-blacklist-cards.md`): 9-guideline instruction with
  2 demos, matching the format of the other four benchmark prompt docs.
- **Experiment planning docs** (`docs/experiment-plan.md`,
  `docs/cross-model-transfer.md`): experiment matrix and cross-model transfer
  analysis.

### Changed
- **README — Library Cards key findings**: moved from between Phase 2 and
  Phase 3 to after Phase 4, and tightened from five overlapping bullets to
  three distinct findings (cheap-model uplift, demos vs. instructions, joint
  optimization coupling).

---

## 2026-02-08

### Fixed
- **README directory tree**: added missing `__init__.py` files, `results/demo/`
  directory, and `generate_demo_data.py` / `generate_demo_html.py` scripts.
- **`filter_parent_keys` bracket notation**: now also checks `key + "["` in
  addition to `key + "."`, so parent keys like `"items"` are correctly filtered
  when child keys use bracket notation (`"items[0].name"`). Defensive fix — no
  current benchmark affected.
- **`pyproject.toml` description**: updated stale "Library Cards Benchmark" to
  "HumBench benchmarks".
- **Dead code in `config.py`**: removed env-var re-assignment (lines that read
  `GEMINI_API_KEY`/`OPENROUTER_API_KEY` then wrote them back unchanged).
  `load_dotenv()` already handles `.env` loading.
- **Business Letters `refine_reward_fn`**: used `intersection` (any key present)
  instead of `issubset` (all keys present), inconsistent with other benchmarks.
  Now requires all three required keys (`send_date`, `sender_persons`,
  `receiver_persons`) to accept a prediction during Refine.
- **`EvalReward` fallback path**: unreachable fallback called `refine_reward_fn`
  with wrong argument types (`dict` instead of `dspy.Example`). Replaced with
  explicit `RuntimeError` and removed unused `_fallback_fn` attribute.
- Removed unused `SKIP_SUFFIXES` constant from `personnel_cards/scoring.py`.

### Added
- **Interactive demo visualizations** (`results/demo/`): self-contained HTML pages
  for all 4 benchmarks with embedded base64 images, field-level GT vs prediction
  diff views, and an index page linking them together.
- **`scripts/generate_demo_data.py`**: runs optimizer predictions on selected test
  images (5 per benchmark) and saves raw prediction JSON alongside ground truth
  and scores.
- **`scripts/generate_demo_html.py`**: generates the HTML demo pages from exported
  data.
- **Tracked `results/` and `data/` directories**: baseline scores, optimized
  programs, test score JSONs, and data symlinks to the upstream benchmark repo
  are now committed (previously gitignored).
- **`choices=` on `--benchmark` argparse** in all 5 scripts (`optimize.py`,
  `evaluate_baseline.py`, `evaluate_optimized.py`, `compare_results.py`,
  `loo_mipro.py`). Invalid benchmark names now produce a clean argparse error
  instead of a `ModuleNotFoundError`.
- **`split_data` overflow assertion**: prevents silently empty test splits when
  `n_train + n_dev >= n`.
- **`loo_mipro.py` guards**: `hasattr` check before `load_loo_folds()` call,
  plus `choices=["bibliographic_data"]` since only that benchmark supports LOO.
- **`docs/code-review-triage.md`**: documents triage principles, deferred items
  (D1–D7), and rejected items (R1–R3) from two independent code reviews.
  - **Quality-aware Refine(3)** inference-time refinement: `EvalReward` class in
  `evaluate_optimized.py` uses the actual benchmark metric (F1 or fuzzy) as
  reward instead of binary JSON-valid check. Auto-detects metric key by probing
  `score_single_prediction({}, {})`. Threshold=0.95 enables early stopping.
- Refine(3) experiment results across all 4 benchmarks:
  - Business Letters: **0.7312 f1_macro** (+9.3 pts over MIPROv2 CoT alone)
  - Library Cards: **0.9167 f1_macro** (+1.5 pts)
  - Personnel Cards: **0.8894 f1_macro** (+0.4 pts)
  - Bibliographic Data: 0.7043 fuzzy (-0.3 pts, within noise)
- Output JSON now includes `"refine_reward": "quality"|null` provenance key
  for distinguishing reward function types.
- DSPy best practices analysis document (`docs/dspy-best-practices.md`).
- Reflection model name now included in GEPA output filenames for traceability
  (e.g., `gepa-cot_gemini-2.0-flash_reflect-gemini-2.5-pro_optimized.json`).

### Changed
- **README prose**: trimmed verbose descriptions, removed Gemini JSON quirk
  and GEPA compatibility notes (implementation details better suited to code
  comments), and removed demo screenshot.

### Refactored
- Extracted shared `compute_f1()` and `filter_parent_keys()` helpers into
  `benchmarks/shared/scoring_helpers.py`, replacing 8 inline F1 computations
  and 2 parent-key filtering loops across the three F1-based scoring modules.
  No scoring logic changes — behavior-preserving deduplication only.
- Replaced verbose loops with list/set comprehensions in `library_cards` and
  `business_letters` scoring.
- Removed dead `"binary"` ternary branch in `evaluate_optimized.py` and
  unnecessary `getattr` fallback in `optimize.py`.
- Added `benchmarks/shared/data_helpers.py` with generic `split_data()` and
  `load_and_split()`, replacing duplicated splitting logic in all four
  benchmark `data.py` modules.
- Added shared F1 scoring factories (`f1_refine_reward_fn`,
  `f1_dspy_metric`, `f1_gepa_feedback_metric`, `f1_compute_aggregate_scores`)
  to `scoring_helpers.py`, replacing ~130 lines of identical metric/reward
  code in `library_cards` and `personnel_cards`. `bibliographic_data` shares
  only `refine_reward_fn` (fuzzy metric differs from F1).
- Added `__all__` re-exports to all benchmark `__init__.py` files for a
  clean public API.
- Modernized `List[X]` to `list[X]` in all schema files.
- Extracted helpers in `check_rate_limits.py` (`_get_api_key`,
  `_extract_rate_limit_headers`, `_print_limits`) and `compare_results.py`
  (`_F1_METRICS` constant).

### Changed
- **`dspy-costs/results.json`**: added GEPA and Refine(3) experiment costs
  across all 4 benchmarks; updated model totals.
  - README: added `data_helpers.py` to project structure tree, updated
  `scoring_helpers.py` description to mention shared F1 factories, fixed
  `configure_lm` → `configure_dspy` in code example.
- README: added Refine(3) rows to all benchmark result tables, new
  "Inference-time refinement (Refine)" subsection under Cross-Benchmark
  Findings, updated Combined Results table, revised Future Work.
- README restructured: all benchmark sections now use consistent Phase 1 / Phase 2
  numbering (optimizer comparison → GEPA with stronger reflection model).
- GEPA results integrated into each benchmark section and the cross-benchmark
  combined results table.
- Future Work: GEPA section rewritten from proposed experiment to completed
  experiment summary with cross-benchmark conclusions.
- README: removed duplicate sections on optimized pipelines and
  cost-performance tradeoff.
- README: fixed Library Cards instruction description (5-sentence, not
  2-sentence).
- Optimized prompt docs (`docs/optimized-prompt-*.md`): added resized demo
  images (600px width, ~477KB total in `docs/figures/`) so readers can see the
  source scans alongside extraction output. Images matched to originals via
  MD5 hash of base64-decoded program data.

### Removed
  - **`docs/poss-dspy-improvements.md`**: superseded by the two independent code
  reviews and `docs/code-review-triage.md`.

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
