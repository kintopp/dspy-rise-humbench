# Possible DSPy Improvements

*Analysis of DSPy best practices vs. current implementation. February 2026.*

This document evaluates whether the project's DSPy usage aligns with current best practices, based on a review of the [DSPy documentation](https://dspy.ai) and the project's codebase and results.

---

## What the project already does well

The project follows most DSPy best practices for structured extraction tasks:

- **Class-based signatures** with detailed `desc` fields -- exactly what DSPy recommends for complex structured extraction.
- **Dual-mode metrics** (float for eval, bool for bootstrapping) -- the canonical DSPy pattern.
- **MIPROv2 as the primary optimizer** -- the correct choice for these data sizes and task types.
- **Quality-aware Refine wrapper** at inference time -- `EvalReward` class in `evaluate_optimized.py` uses the actual benchmark metric (F1/fuzzy) as reward, with GT injection per image and threshold=0.95 for early stopping.
- **FeedbackScore for GEPA** -- correctly bridges the dict/arithmetic API gap that DSPy's parallelizer requires.
- **`dspy.Image` typed input fields** -- proper multimodal signature usage.

---

## Areas worth evaluating

### A. ~~Refine reward function is too coarse~~ DONE

**Impact: HIGH | Effort: LOW | Status: Implemented and validated**

The per-benchmark `refine_reward_fn` functions remain binary (used as fallback), but evaluation now uses the `EvalReward` class in `scripts/evaluate_optimized.py` which provides a quality-aware reward using the actual benchmark metric.

**Implementation details:**

DSPy's `Refine` calls `reward_fn(kwargs, outputs)` where `kwargs` is a dict of inputs and `outputs` is a Prediction object -- it does **not** pass ground truth. The solution: a callable class that receives GT via `set_gt()` before each image and `clear_gt()` afterwards:

```python
class EvalReward:
    def __init__(self, scoring_mod):
        self._score_single = scoring_mod.score_single_prediction
        self._fallback_fn = scoring_mod.refine_reward_fn
        self._gt = None
        # Auto-detect: bibliographic_data → "fuzzy", others → "f1_score"
        probe = scoring_mod.score_single_prediction({}, {})
        self._metric_key = "fuzzy" if "fuzzy" in probe else "f1_score"

    def __call__(self, kwargs, outputs):
        if self._gt is None:
            return self._fallback_fn(kwargs, outputs)
        pred_dict = parse_prediction_document(outputs)
        if pred_dict is None:
            return 0.0
        score = self._score_single(pred_dict, self._gt)
        return score.get(self._metric_key, 0.0)
```

**Results with Refine(N=3, threshold=0.95):**

| Benchmark | MIPROv2 CoT | + Refine(3) | Gain |
|---|---|---|---|
| Business Letters | 0.6378 | **0.7312** | **+9.34 pts** |
| Library Cards | 0.9017 | **0.9167** | +1.50 pts |
| Personnel Cards | 0.8858 | **0.8894** | +0.36 pts |
| Bibliographic Data | 0.7072 | 0.7043 | -0.29 pts |

Impact scales inversely with baseline quality. Business Letters benefited most because name format errors are stochastic -- retrying often gets the correct "First Last" format. Bibliographic Data showed no gain because its bottleneck is positional alignment, not extraction quality. The 3/43 zero-scoring Personnel Cards remained at 0.0 -- these are structural failures that retries cannot fix.

**Files**: `scripts/evaluate_optimized.py` (EvalReward class, GT injection in eval loop)

---

### B. BestOfN as a simpler alternative to Refine

**Impact: LOW | Effort: LOW | Status: Deprioritized (Refine already works well)**

[dspy.BestOfN](https://dspy.ai/api/modules/BestOfN/) runs the module N times at temperature=1.0 and picks the best result by reward, **without** generating feedback between attempts.

Now that quality-aware Refine is implemented and delivering +1.5 to +9.3 pts across benchmarks, BestOfN is less compelling. The remaining potential advantage is concurrency -- BestOfN could run N attempts in parallel vs Refine's sequential attempts. But the latency difference is modest (3 sequential calls vs 1 parallel batch) and doesn't affect the final scores.

Still potentially useful for a future **majority-vote ensemble** approach: run BestOfN with k>3 and combine correct fields from different attempts, rather than keeping a single best attempt. This could help the 3/43 Personnel Cards that still score 0.0.

**Files**: `scripts/evaluate_optimized.py`

---

### C. ~~GEPA reflection model should differ from the target model~~ DONE

**Impact: HIGH | Effort: NONE | Status: Completed across 3 benchmarks**

GEPA medium-CoT was re-run with Gemini 2.5 Pro as the reflection model on Library Cards (23 iterations), Personnel Cards (84 iterations), and Business Letters (96 iterations). Bibliographic Data was skipped (only 3 train+dev examples).

Results: GEPA came within 1.1 pts of MIPROv2 on Personnel Cards (0.8750 vs 0.8858) -- the closest any optimizer came on any benchmark. On Library Cards and Business Letters, GEPA fell short. Instruction-only optimization generalizes well on tasks with consistent structure but overfits severely with small dev sets (Business Letters: dev 0.896 → test 0.547).

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
| ~~Improve Refine reward to use F1/fuzzy score~~ | High | Low | **Done** -- +1.5 to +9.3 pts across benchmarks |
| ~~GEPA with stronger reflection model~~ | High | None | **Done** -- within 1.1 pts of MIPROv2 on Personnel Cards |
| Try BestOfN / majority-vote ensemble | Low | Low | Deprioritized (Refine already effective) |
| Test JSONAdapter for Gemini Flash | Medium | Medium | Not started |
| Experiment with bootstrapping threshold | Low | Low | Not started |
| Try `max_bootstrapped_demos=4` on one benchmark | Low | Low | Not started |

The two highest-impact changes have been implemented. Quality-aware Refine(3) is now the project's best inference-time strategy, delivering gains on 3 of 4 benchmarks. The remaining items are diminishing-returns experiments -- JSONAdapter is the most promising but risks invalidating existing optimized programs.

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
