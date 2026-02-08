# Possible DSPy Improvements

*Analysis of DSPy best practices vs. current implementation. February 2026.*

This document evaluates whether the project's DSPy usage aligns with current best practices, based on a review of the [DSPy documentation](https://dspy.ai) and the project's codebase and results.

---

## What the project already does well

The project follows most DSPy best practices for structured extraction tasks:

- **Class-based signatures** with detailed `desc` fields -- exactly what DSPy recommends for complex structured extraction.
- **Dual-mode metrics** (float for eval, bool for bootstrapping) -- the canonical DSPy pattern.
- **MIPROv2 as the primary optimizer** -- the correct choice for these data sizes and task types.
- **Refine wrapper** at inference time for retry logic -- already wired up in `evaluate_optimized.py`.
- **FeedbackScore for GEPA** -- correctly bridges the dict/arithmetic API gap that DSPy's parallelizer requires.
- **`dspy.Image` typed input fields** -- proper multimodal signature usage.

---

## Areas worth evaluating

### A. Refine reward function is too coarse

**Impact: HIGH | Effort: LOW**

The `refine_reward_fn` across all 4 benchmarks is binary: 1.0 if the JSON parses and has required top-level keys, 0.0 otherwise. This means Refine can't distinguish between a nearly-perfect extraction and a badly wrong one -- both get 1.0 as long as the JSON is structurally valid.

The [DSPy Refine documentation](https://dspy.ai/api/modules/Refine/) shows that Refine uses the reward score to select the **best** attempt out of N, and generates feedback based on intermediate scores. A binary reward means Refine has no signal to prefer one valid JSON over another, and it can't generate meaningful feedback about what to improve.

**Recommendation**: Use the actual F1/fuzzy score as the Refine reward, not just a validity check:

```python
def refine_reward_fn(example, prediction, trace=None) -> float:
    pred_dict = parse_prediction_document(prediction)
    gt_dict = parse_gt_document(example)
    if pred_dict is None or gt_dict is None:
        return 0.0
    scores = score_single_prediction(pred_dict, gt_dict)
    return scores["f1_score"]
```

This gives Refine a continuous signal to select the best of N attempts. With N=3, this could help the 3/43 Personnel Cards that still score 0.0 and the sporadic JSON failures on other benchmarks.

**Caveat**: This requires the `example` argument to carry ground truth at inference time. Check whether DSPy passes the full example (with GT) to the reward function -- if it only passes inputs, the binary approach is correct and BestOfN with a validity-only check is the alternative.

**Files**: `benchmarks/library_cards/scoring.py`, `benchmarks/bibliographic_data/scoring.py`, `benchmarks/personnel_cards/scoring.py`, `benchmarks/business_letters/scoring.py`

---

### B. BestOfN as a simpler alternative to Refine

**Impact: MEDIUM | Effort: LOW**

[dspy.BestOfN](https://dspy.ai/api/modules/BestOfN/) runs the module N times at temperature=1.0 and picks the best result by reward, **without** generating feedback between attempts.

For this project's use case, BestOfN might be more appropriate than Refine for production inference:
- Refine's feedback loop adds latency (sequential attempts with inter-step feedback generation)
- BestOfN's attempts could potentially run concurrently
- The task benefits from sampling diversity (different JSON structures from the same image)

This maps directly to the "Ensemble and self-consistency" item in the README's Future Work section.

**Files**: `scripts/evaluate_optimized.py`

---

### C. GEPA reflection model should differ from the target model

**Impact: HIGH | Effort: NONE (already in design)**

The README already notes this: "GEPA's genetic-evolutionary approach was limited by using Flash as its own reflection model — the model struggled to diagnose its own failures." The [GEPA tutorial](https://dspy.ai/tutorials/gepa_facilitysupportanalyzer/) confirms that the reflection model should ideally be a stronger model.

The implementation at `scripts/optimize.py:76` correctly instantiates a separate `reflection_lm` with `temperature=1.0`. The GEPA runs currently underway should use a stronger model (e.g., Gemini 2.5 Pro or Claude Sonnet) for reflection while keeping Flash as the target.

---

### D. Adapter configuration: consider JSONAdapter for Gemini models

**Impact: MEDIUM | Effort: MEDIUM**

`configure_dspy()` at `benchmarks/shared/config.py:68-69` uses the default ChatAdapter. The [DSPy adapter docs](https://dspy.ai/learn/programming/adapters/) describe JSONAdapter as an alternative that uses `response_format` for models supporting native structured output (Gemini 2.0 Flash supports this via `response_mime_type`).

Benefits of JSONAdapter:
- Less boilerplate in the prompt = faster/cheaper responses
- Native JSON mode = fewer parse failures (no markdown code fences)
- The project already has `strip_code_fences()` as a workaround for this exact problem

Configuration:
```python
dspy.configure(lm=lm, adapter=dspy.JSONAdapter())
```

**Risk**: This might change prompt formatting enough to invalidate existing MIPROv2 optimized programs. Worth testing on baseline first to see if it helps or hurts before re-optimizing.

**Files**: `benchmarks/shared/config.py`

---

### E. Bootstrapping threshold in `dspy_metric`

**Impact: LOW-MEDIUM | Effort: LOW**

The `dspy_metric` returns `f1 >= 0.5` in bootstrapping mode (when trace is set). This threshold controls which training examples MIPROv2 accepts as valid demonstrations. At 0.5, fairly poor extractions are allowed as demos.

The [DSPy cheatsheet](https://dspy.ai/cheatsheet/) notes that bootstrapping quality matters. For Library Cards where the baseline is already 0.81+, a threshold of 0.5 means ~30% of the demo could be wrong fields. Consider raising to 0.7 or 0.8 for benchmarks with higher baselines (Library Cards, Personnel Cards) while keeping 0.5 for the harder ones (Business Letters).

This is a small lever -- MIPROv2's Bayesian search already validates the full program against the dev set, so bad demos get filtered indirectly.

**Files**: `benchmarks/*/scoring.py` (the `dspy_metric` function in each)

---

### F. `max_bootstrapped_demos` and `max_labeled_demos` defaults

**Impact: LOW | Effort: LOW**

`optimize.py` defaults are `max_bootstrapped=2, max_labeled=2`. DSPy defaults are `4, 4`. The optimized programs ended up with 1-2 demos per benchmark, so the lower cap seems fine -- MIPROv2 apparently found that fewer demos work better for these tasks. The DSPy docs suggest values up to 4; increasing beyond the current 2 is worth a quick experiment on one benchmark to verify nothing is being left on the table.

**Files**: `scripts/optimize.py`

---

## Things to skip (low expected value in this context)

- **Assertions/Suggestions**: Now deprecated in favor of Refine, which the project already uses.
- **Typed predictors** (Pydantic output): The output is a raw JSON string parsed downstream, which is the right approach for matching upstream scoring exactly. Typed predictors would add a Pydantic layer that might normalize values differently.
- **KNNFewShot**: Requires an embedder, and inputs are images (not text). Semantic similarity would need to operate on image embeddings, which is a significant addition. With 9-39 training examples, the diversity benefit is limited.
- **InferRules**: Generates natural-language rules from demos. The MIPROv2 optimized instructions already contain task-specific rules, and InferRules is reportedly strongest for coding tasks.
- **BetterTogether / BootstrapFinetune**: Requires fine-tuning, which is outside the project's cost/infrastructure scope.
- **ProgramOfThought**: For code-generation reasoning, not document extraction.
- **Multi-step pipelines** (two separate DSPy modules): The README correctly identifies this as future work. It would require a new architecture and re-optimization -- significant effort for uncertain gain.

---

## Summary

| Item | Impact | Effort | Status |
|------|--------|--------|--------|
| Improve `refine_reward_fn` to use F1 score | High | Low | Ready to implement |
| Try BestOfN at inference for zero-score cards | Medium | Low | Ready to implement |
| Ensure GEPA uses stronger reflection model | High | None | Already in design |
| Test JSONAdapter for Gemini Flash | Medium | Medium | Needs baseline test first |
| Experiment with bootstrapping threshold | Low | Low | Quick experiment |
| Try `max_bootstrapped_demos=4` on one benchmark | Low | Low | Quick experiment |

The most impactful change is **improving the Refine reward function** -- a one-line change per benchmark that could meaningfully help with remaining zero-scoring images. The rest are either already underway (GEPA reflection model) or diminishing-returns experiments.

---

## Sources

- [DSPy Homepage](https://dspy.ai)
- [DSPy Optimizers](https://dspy.ai/learn/optimization/optimizers/)
- [DSPy Signatures](https://dspy.ai/learn/programming/signatures/)
- [DSPy Modules](https://dspy.ai/learn/programming/modules/)
- [dspy.Refine API](https://dspy.ai/api/modules/Refine/)
- [dspy.BestOfN API](https://dspy.ai/api/modules/BestOfN/)
- [DSPy Assertions](https://dspy.ai/learn/programming/7-assertions/)
- [DSPy Adapters](https://dspy.ai/learn/programming/adapters/)
- [DSPy Cheatsheet](https://dspy.ai/cheatsheet/)
- [KNNFewShot API](https://dspy.ai/api/optimizers/KNNFewShot/)
- [InferRules API](https://dspy.ai/api/optimizers/InferRules/)
- [BetterTogether API](https://dspy.ai/api/optimizers/BetterTogether/)
- [GEPA Tutorial](https://dspy.ai/tutorials/gepa_facilitysupportanalyzer/)
- [DSPy Entity Extraction Tutorial](https://dspy.ai/tutorials/entity_extraction/)
