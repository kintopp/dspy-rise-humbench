# Code Review Triage

Two independent code reviews surfaced 21 combined issues. This document
records the triage principles, the items that were deferred or rejected,
and the rationale for each decision.

## Triage Principles

1. **Scoring is untouchable.** `CLAUDE.md` mandates upstream-identical
   scoring for leaderboard comparability. No change may alter the numeric
   output of any scoring function.
2. **The project is shipped.** Guard rails and clarity improvements only.
   No behavior changes to working pipelines.
3. **Guard rails > logic changes.** Prefer assertions, `choices=`, and
   better error messages over restructuring.
4. **One change, one purpose.** No "while we're at it" scope creep. Each
   fix is independently reviewable.
5. **Dead code needs proof.** Don't remove "possibly dead" defensive code
   unless evidence shows it never triggers.
6. **Cosmetic consistency is low-value.** Renaming constants or adding
   comments to working code is churn, not improvement.

## Accepted Items (implemented)

| # | Issue | Files Changed |
|---|-------|--------------|
| A1 | `split_data` overflow guard — assertion prevents silently empty test splits | `benchmarks/shared/data_helpers.py` |
| A2 | `loo_mipro.py` guard for `load_loo_folds` — `hasattr` check + restricted `choices=` | `scripts/loo_mipro.py` |
| A3 | Add `choices=` to `--benchmark` in all scripts — replaces raw `ModuleNotFoundError` with argparse error | `scripts/optimize.py`, `evaluate_baseline.py`, `evaluate_optimized.py`, `compare_results.py`, `loo_mipro.py` |
| A4 | Fix stale `pyproject.toml` description | `pyproject.toml` |
| A5 | Remove redundant API key re-assignment in `config.py` — `load_dotenv()` already handles `.env` | `benchmarks/shared/config.py` |
| B1 | `filter_parent_keys` bracket notation fix — also check `key + "["` alongside `key + "."` | `benchmarks/shared/scoring_helpers.py` |

## Deferred Items

### D1: EvalReward probe fragility

The `EvalReward.__init__` probes `score_single_prediction({}, {})` to
auto-detect the metric key. Reviewers suggested replacing with a mapping.
**Deferred** because the probe works, avoids config duplication, and if a
5th benchmark crashes on empty input, that's the right time to fix it.

### D2: `compare_results.py` silent fallback

`METRIC_KEYS.get(benchmark, ["fuzzy"])` silently falls back for unknown
benchmarks. **Deferred** because A3 (argparse `choices=`) now prevents
unknown benchmarks from reaching this code at all.

### D3: Personnel Cards inline skip logic

The `_filter_keys` function has inline skip logic for `row_number` instead
of a named constant like Library Cards uses. **Deferred** — it's 2 lines
of clear code. Adding a constant for consistency is churn (Principle 6).

### D4: Business Letters metadata wrapper

The `_unwrap_metadata` function may be dead code (defensive fallback for
models wrapping output in `{"metadata": ...}`). **Deferred** — removing it
risks regression if models occasionally produce that format. Principle 5
applies: dead code needs proof before removal.

### D5: `f1_macro` semantic difference across benchmarks

Library Cards and Personnel Cards compute per-field F1 then average, while
Business Letters computes per-category set-match F1. Both are called
`f1_macro`. **Deferred** — the semantics match upstream exactly. Already
documented in the MEMORY.md key differences table.

### D6: Cost estimator hardcodes N=3

The cost estimation script hardcodes `N=3` for Refine retries. **Deferred**
— it's an approximation in an estimation script, not production code.

### D7: Redundant GT normalization in bibliographic_data

Ground truth normalization is potentially idempotent (applied to
already-clean data). **Deferred** — removing it risks subtle scoring
differences if the normalization path ever diverges. Harmless with 5
images.

## Rejected Items

### R1: FP+FN double-counting in F1 scoring

Both reviewers identified that a prediction field present with low fuzzy
score counts as *both* FP and FN. **Rejected** — this is upstream
benchmark behavior. Changing it would violate Principle 1 (scoring
comparability). Documented in upstream issues
[#91](https://github.com/RISE-UNIBAS/humanities_data_benchmark/issues/91)
and [#92](https://github.com/RISE-UNIBAS/humanities_data_benchmark/issues/92).

### R2: Exception handling asymmetry in parse functions

`parse_prediction_document` tries code-fence stripping as fallback;
`parse_gt_document` does not. Reviewers flagged the inconsistency.
**Rejected** — this is intentional. Ground truth should never have code
fences (it comes from curated JSON files). Adding a fallback to GT parsing
would mask data quality issues rather than surfacing them.

### R3: Schema files unused at runtime

Pydantic schema files exist but aren't imported by scoring code.
**Rejected** — they serve as human-readable documentation of the expected
JSON structure. Removing them loses reference material. Adding runtime
validation would be over-engineering for a pipeline that's already shipped.
