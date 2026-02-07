# Claude Code Insights

This file contains educational insights extracted from Claude Code sessions.
Generated automatically by the session-end hook.

---

## Session: 2026-02-07 10:41

**Session ID:** `14dcf2a1`  
**End Reason:** clear  
**Insights Captured:** 4

### 1. Insight

**Context:** 

**DSPy's core idea**: Instead of hand-crafting prompts, you declare *what* the LLM should do (via typed signatures), define a metric, and let an optimizer algorithmically search for the best instruction phrasing + few-shot demonstrations. The prompt becomes a *compiled artifact*, not a hand-written one.

---

### 2. Insight

**Context:** 

**Scoring approach**: The metric flattens both predicted and ground-truth JSON to leaf keys (e.g., `author.last_name`, `publication.year`), then does fuzzy string matching (rapidfuzz ratio >= 0.92 = TP). This is a faithful reimplementation of the RISE benchmark's own `scoring_helper.py`. The DSPy metric returns a float in eval mode and a bool (`f1 >= 0.5`) in bootstrapping mode — this dual return type is how DSPy optimizers decide which demos to keep.

---

### 3. Insight

**Context:** 

**Why this matters**: The original phrasing compared f1_micro from the optimized pipeline (0.8965) against a GPT-4o number (0.8961) that didn't match any score on the live leaderboard, and framed GPT-4o as "the benchmark's reference" when it's actually just one of 30+ models. The new phrasing compares f1_macro (which is what the leaderboard heatmap displays), lists the actual top models for context, and notes the subset limitation honestly.

---

### 4. Insight

**Context:** 

**Why leaderboard scores matter for optimization planning**: DSPy optimization works best when there's a gap between current model performance and perfect scores — that gap is where better prompts and few-shot examples can help. Blacklist Cards at 95%+ is near ceiling, so optimization yields diminishing returns. Bibliographic Data at ~67 and Business Letters at ~68 have the most headroom, making them the best targets for the same MIPROv2 approach that worked on Library Cards.

---

## Session: 2026-02-07 11:16

**Session ID:** `c9bbca88`  
**End Reason:** clear  
**Insights Captured:** 3

### 1. Insight

**Context:** 

**DSPy Module Architecture**: `dspy.Predict` directly calls the LM with the signature. `dspy.ChainOfThought` wraps `Predict` internally — it adds a `reasoning` field before the output, encouraging step-by-step thinking. This means when you save/load state, the key paths differ: a Predict module stores state under `"predict"`, while a ChainOfThought module stores it under `"predict.predict"` (because CoT contains a Predict inside it). This is why `--module` must match between optimize and evaluate.

**dspy.Refine**: This is an inference-time wrapper (not an optimizer). It runs the module up to N times, checking each output against a `reward_fn`. If the reward threshold is met, it returns early. This is useful for recovering from parse failures — if the first attempt produces invalid JSON, it retries. Importantly, Refine uses `inspect.getsource()` on the reward function, so it must be defined in a source file, not a lambda or closure.

---

### 2. Insight

**Context:** 

**Filename convention matters for auto-discovery**: The `compare_results.py` script uses `scores*.json` for baselines and `*_test_scores.json` for optimized results. By inserting module/refine tags *between* the fixed prefix/suffix (`scores_cot_...`, `..._refine3_test_scores.json`), the glob patterns automatically discover new experiments without modifying the comparison script.

**Why `self.predict` keeps the same attribute name**: DSPy's `Module.save()` / `Module.load()` serializes the module tree using attribute names as keys. By always naming the inner module `self.predict` (whether it's `Predict` or `ChainOfThought`), the `forward()` method stays unchanged. However, the saved JSON differs internally — CoT stores `"predict.predict"` (the nested Predict inside ChainOfThought) — which is why you can't cross-load between module types.

---

### 3. Insight

**Context:** 

**Code review triage**: Not every finding from a code review is equal. The reviewer flagged 8 issues — some are genuine improvements to make, while others reflect misunderstandings of how DSPy works or are intentional design choices documented in the plan. A good code review response acknowledges valid points, explains the design rationale for intentional choices, and identifies what's worth fixing now vs. later.

---

## Session: 2026-02-07 11:19

**Session ID:** `4dd2cf64`  
**End Reason:** prompt_input_exit  
**Insights Captured:** 2

### 1. Insight

**Context:** 

**GEPA's 5-argument metric signature** is different from the standard DSPy 2-3 arg metric. GEPA passes `(gold, pred, trace, pred_name, pred_trace)` and expects a dict `{"score": float, "feedback": str}` back. The textual feedback is what makes GEPA special — it can tell the reflection LM *why* a prediction scored low (e.g., "author field: predicted 'J. Smith' but expected 'John Smith', fuzzy score 0.72"), enabling targeted instruction improvement.

**SIMBA's mini-batch approach** works differently from MIPROv2's Bayesian optimization. It samples random mini-batches from the trainset, identifies high-variability examples (ones the model sometimes gets right, sometimes wrong), and uses LLM self-reflection to generate improvement rules. This means it doesn't need a separate validation set — it uses the trainset's internal variance as signal.

---

### 2. Insight

**Context:** 

**Design pattern: lazy imports for optimizer-specific dependencies.** Notice how `run_simba()` and `run_gepa()` use local imports (`from dspy.teleprompt import SIMBA`) rather than top-level imports. This is intentional — it means the script doesn't fail at import time if one optimizer's dependencies are missing. A user running `--optimizer mipro` won't hit an import error for GEPA's `MultiModalInstructionProposer`, and vice versa.

**GEPA's `parser.error()` pattern** is cleaner than raising an exception. Argparse's `parser.error()` prints the usage message, the error, and exits with code 2 — giving the user immediate context about valid flags. This is why we kept the `parser` variable accessible and did the validation in `main()` rather than inside `run_gepa()`.

**The save filename pattern** (`{optimizer}{module_tag}_{model_tag}_optimized.json`) automatically handles the new optimizers — e.g., `simba-cot_gemini-2.0-flash_optimized.json` — which means `compare_results.py`'s glob patterns will pick them up without changes.

---

## Session: 2026-02-07 12:16

**Session ID:** `92f41938`  
**End Reason:** prompt_input_exit  
**Insights Captured:** 6

### 1. Insight

**Context:** 

**Experiment dependency chain:** E1 (CoT baseline) is independent and can run immediately. E2 (optimization) is also independent but long-running — MIPROv2 medium searches 12 candidate programs vs 6 for light. E2 eval and E3 both depend on E2's optimized program artifact. The estimated cost is $42-55 across all three experiments, primarily from Gemini 2.5 Pro vision API calls on 263 images across multiple optimization trials.

---

### 2. Insight

**Context:** 

**Why Flash first makes sense:** Gemini 2.0 Flash is significantly cheaper than 2.5 Pro, runs faster, and has generous rate limits. The "uplift" hypothesis is compelling — can optimization close the gap between a cheap model and an expensive one? If SIMBA/GEPA can meaningfully boost Flash performance, that's both a cost-saving finding and validates the optimizers before committing $40+ to Pro experiments.

---

### 3. Insight

**Context:** 

**Parallelizing independent experiments:** Since the baseline evaluation and both optimizations use independent data flows (baseline runs on test set, optimizers run on train/dev sets), they can safely run concurrently without interfering with each other. All three share the same Gemini 2.0 Flash API endpoint, but Flash has generous rate limits (~2000 RPM), so 16 threads x 3 processes should be well within capacity.

---

### 4. Insight

**Context:** 

**Why this didn't affect Gemini 2.5 Pro:** DSPy's JSON adapter first tries "structured output mode" (native JSON mode via the API), which works with 2.5 Pro. For 2.0 Flash, it falls back to "JSON mode" where the model generates text freely. Smaller models like Flash tend to wrap JSON in markdown code fences even when instructed not to — a well-known issue. The fix is to strip fences in the parsing layer, not in the prompt.

**GEPA's metric architecture:** GEPA needs both a numeric score (for optimization) and textual feedback (for reflection). But DSPy's `Evaluate` class assumes metrics return numbers so it can track average score via `sum(vals)/len(vals)`. A dict subclass that supports arithmetic resolves this tension.

---

### 5. Insight

**Context:** 

**SIMBA's optimization strategy:** Notice how SIMBA identifies "buckets" of examples with high variability (max-to-min gap) and applies different strategies per bucket: `append_a_demo_` for examples that need demonstration, and `append_a_rule` for examples where the model needs explicit guidance (like the shelfmark extraction rule). This targeted approach contrasts with MIPROv2's global Bayesian search — SIMBA focuses on fixing specific failure modes.

---

### 6. Insight

**Context:** 

**What the comparison script shows**: The "Deltas vs First Baseline" compares against the GPT-4o predict baseline (0.8172 f1_macro), but for the Flash experiments the proper baseline is `cot_gemini-2.0-flash` (0.7583 f1_macro). So the actual uplift from DSPy optimization on Gemini 2.0 Flash is:

- **SIMBA**: 0.7583 → 0.8481 = **+0.0898 f1_macro** (+11.8% relative)
- **GEPA**: 0.7583 → 0.8148 = **+0.0565 f1_macro** (+7.5% relative)

**Why SIMBA outperformed GEPA here**: SIMBA's mini-batch approach with self-reflection + demo accumulation generated targeted extraction rules (like "pay close attention to author spelling" and "extract shelfmarks even if abbreviated") that directly addressed the model's weak spots. GEPA's genetic-evolutionary approach optimized instructions but didn't accumulate as many task-specific demos — and it used the same model for both extraction and reflection, limiting its ability to self-improve.

**The precision story is striking**: SIMBA achieved 0.9116 precision (highest of all experiments), meaning it learned to be conservative — producing fewer false positives — while GEPA tracked closer to baseline precision. This suggests SIMBA's rules helped Flash avoid hallucinating fields that aren't on the card.

---

## Session: 2026-02-07 13:32

**Session ID:** `93f7e8eb`  
**End Reason:** prompt_input_exit  
**Insights Captured:** 13

### 1. Insight

**Context:** Session start

Reviewing uncommitted changes before committing is a good practice — it helps catch accidental modifications, debug leftovers, or unintended file inclusions before they become part of the project history.

---

### 2. Insight

**Context:** 

Both changes are documentation-only — no code logic is affected. The README changes bring the project documentation in sync with the Phase 1 experiments (SIMBA, GEPA, CoT support) that were already committed in code. The INSIGHTS.md file serves as an auto-generated session log, preserving experiment reasoning for future reference.

---

### 3. Insight

**Context:** 

E3 is the low-hanging fruit here — it's just an evaluation pass on the existing SIMBA-CoT program with a Refine safety net, so it's cheap and fast. E2 is more expensive (MIPROv2 medium = 12 trials vs 6 for light) but still far cheaper than running it on 2.5 Pro.

---

### 4. Insight

**Context:** 

Both scripts were built with these experiments in mind. The key design choices: (1) `optimize.py` constructs the output filename from `{optimizer}{module_tag}_{model}_optimized.json`, so MIPROv2 medium + CoT + Flash will get a distinct filename automatically. (2) `evaluate_optimized.py` appends `_refine{N}` to the score filename, so E3 results won't overwrite E2's scores. (3) Refine uses `threshold=1.0` — meaning it retries unless the reward function returns a perfect score, which is appropriate since only structural parse failures should trigger retries.

---

### 5. Insight

**Context:** 

The plan originally called for MIPROv2 **light** on Flash (F3), but we're running **medium** (12 candidates vs 6). This is a deliberate upgrade — the light search already found a strong program on 2.5 Pro, so a broader search on Flash is worth trying to see if MIPROv2 can match SIMBA's 0.8481 result. Two experiments from the plan remain undone: F1 (Predict baseline without CoT) and F6 (transferring the Pro-optimized program to Flash) — both are cheap ~$0.50 eval-only runs.

---

### 6. Insight

**Context:** 

E2's early results are promising — 86.76% on the dev set after just 4 trials is already higher than SIMBA's test f1_macro of 0.8481. But dev scores tend to be optimistic (the optimizer selects *for* dev performance), so the real test will be the held-out evaluation. Still, MIPROv2 medium's broader search is clearly exploring more of the instruction/demo space than light would.

---

### 7. Insight

**Context:** 

The medium search budget (12 candidates) clearly paid off here. The best configuration wasn't found until trial 18 — with `light` (6 candidates), the search would have stopped at ~90.91 (trials 11-13). That extra exploration discovered a configuration scoring 91.82 on dev, which is remarkable for Flash — it's **higher than the MIPROv2 light + Gemini 2.5 Pro dev score** that produced the 0.8912 test result. Of course dev scores are optimistic, but this is very promising.

---

### 8. Insight

**Context:** 

The restructured plan shifts from a "how do we implement SIMBA/GEPA?" document (which is now done) to a "where are we and what's next?" living experiment tracker. The motivation section foregrounds the cost-performance rationale — this isn't just about chasing higher numbers, it's about making optimization practical for real archival digitisation workflows where budget matters. The decision tree after F3 eval makes next steps concrete rather than open-ended.

---

### 9. Insight

**Context:** 

Refine actually *hurt* slightly. This makes sense: Refine forces `temperature=1.0` for diversity on retries, which introduces variation on the first attempt too. Since SIMBA-CoT's program was already producing parseable JSON consistently (we saw no retry activity in the logs), the temperature increase just added noise without catching any failures. Refine is designed to help with structural parse failures — but SIMBA had already minimised those. The lesson: Refine is a safety net for fragile outputs, not a general booster for already-robust programs.

---

### 10. Insight

**Context:** 

**Optimized Flash now beats optimized Pro.** MIPROv2 medium pushed Gemini 2.0 Flash (0.9017) past MIPROv2 light + Gemini 2.5 Pro (0.8912) — at roughly 1/10th the inference cost. This is exactly the cost-performance story the plan was designed to test. The medium search budget was critical: the winning trial was #18 out of 18, which light's 6-trial budget would never have reached.

The balance of precision (0.9083) and recall (0.9057) is also notable — unlike SIMBA which traded recall for precision, MIPROv2 medium achieved nearly equal precision and recall, producing a well-rounded extractor.

---

### 11. Insight

**Context:** 

The README restructuring makes a deliberate narrative choice: rather than presenting results chronologically (Pro first, then Flash), it frames the Pro result as "establishing a ceiling" and the Flash experiments as "the main event." This mirrors how the RISE benchmarks themselves frame evaluation — practical deployment on large archival collections is the goal, not leaderboard climbing. The methodological insight about optimizer search budget (trial #18 out of 18) is woven into the Flash narrative rather than listed as a standalone bullet, which keeps it connected to the story.

---

### 12. Insight

**Context:** 

Two surprises here:

**F1 (Flash Predict) scores higher than F2 (Flash CoT): 0.8134 vs 0.7583.** ChainOfThought actually *hurts* unoptimized Flash — the reasoning step introduces errors when there's no optimized instruction guiding it. But CoT becomes essential once optimization is applied (0.9017 with CoT vs the Predict ceiling). This suggests CoT creates a wider "optimisation surface" — more room for the optimizer to improve, but a worse starting point.

**F6 (Pro transfer) works remarkably well: 0.8743.** Instructions and demos optimized for Gemini 2.5 Pro transfer to Flash with only a modest drop (0.8912 → 0.8743). But Flash-native optimization (F3: 0.9017) still beats the transfer by +2.7 points, confirming the DSPy community's recommendation to re-optimize per model.

---

### 13. Insight

**Context:** 

The CoT finding is perhaps the most methodologically interesting result from the whole project: ChainOfThought isn't inherently better — it gives the *optimizer* more to work with. This reframes CoT from "always use it for harder tasks" to "use it when you plan to optimize, because it widens the search space." The transfer finding is similarly nuanced: programs transfer well enough to be useful as starting points, but re-optimization on the target model is cheap and worthwhile.

---

## Session: 2026-02-07 13:38

**Session ID:** `ecac94ba`  
**End Reason:** prompt_input_exit  
**Insights Captured:** 2

### 1. Insight

**Context:** 

Node.js `server.listen(port)` without a host argument binds to `0.0.0.0` (IPv4) or `::` (dual-stack), making it network-accessible. This is a common mistake in developer tools — always pass `'127.0.0.1'` or `'localhost'` as the second argument when the server should be local-only.

---

### 2. Insight

**Context:** 

`npx -y` is convenient but dangerous for security-sensitive contexts: (1) It auto-accepts package installation without user consent, (2) without a pinned version (e.g., `@mermaid-js/mermaid-cli@11.12.0`), a supply chain attack could push a malicious version. The `mermaid-cli` dependency brings in a full headless Chromium browser, which dramatically increases the attack surface.

---

## Session: 2026-02-07 14:10

**Session ID:** `e5493995`  
**End Reason:** clear  
**Insights Captured:** 2

### 1. Key structural differences between Library Cards and Bibliographic Data that drive the experiment design:

**Context:** 

1. **5 images vs 263** — this is the single biggest constraint. DSPy optimizers were designed for 100+ examples; with 5, we're in a fundamentally different regime where train/dev/test splits become 2/1/2 at best.
2. **Multi-entry extraction** — each image yields 14-20 bibliographic entries, so the 5 images contain ~82 total entries. The per-image score averages many field comparisons, giving meaningful signal even with few images.
3. **Different metric** — Bibliographic Data uses average fuzzy score (not F1), which skips the TP/FP/FN threshold step. This means our entire `score_single_prediction` function needs replacement, not just tweaking.
4. **Ground truth inconsistency** — page_10.json uses CSL-JSON hyphenated keys (`publisher-place`, `container-title`) while pages 2-5 use underscored keys (`publisher_place`). This is a real data quality issue that will need handling.

---

### 2. Why the metric change (F1 → average fuzzy) matters for optimization:

**Context:** 

- Library Cards' F1 metric has a **hard threshold** (0.92 fuzzy → TP, below → counts as FP+FN). This creates a step function: a field at 0.91 similarity scores the same as 0.0.
- Bibliographic Data's average fuzzy metric is **continuous** — every improvement in field similarity directly increases the score. This means DSPy optimizers get smoother gradient signal, which could make optimization more effective despite the smaller dataset.
- However, the continuous metric also means the optimizer can't "game" thresholds by pushing borderline fields just above 0.92. It needs genuine quality improvements across all fields.

---

## Session: 2026-02-07 16:36

**Session ID:** `44b40fcd`  
**End Reason:** other  
**Insights Captured:** 14

### 1. Insight

**Context:** 

**Multi-benchmark architecture**: The key design pattern here is a plugin-style package structure where each benchmark exports the same interface (`load_and_split`, `Extractor`, `dspy_metric`, etc.) and scripts use `importlib` to dynamically select which benchmark to use at runtime. This is a common pattern in ML experiment frameworks — it decouples the "what to evaluate" from "how to evaluate" so you can add new benchmarks without modifying the orchestration scripts.

**Shared vs benchmark-specific code**: The fuzzy scoring helpers (`get_all_keys`, `get_nested_value`, `calculate_fuzzy_score`) are truly generic, while the scoring *logic* (F1 vs average fuzzy) differs per benchmark. This split keeps shared utilities DRY while allowing metric divergence.

---

### 2. Insight

**Context:** 

**Key normalization for page_10**: The ground truth for page_10.json uses CSL-JSON conventions (hyphenated keys like `publisher-place`) while the Pydantic schema and other pages use underscored keys (`publisher_place`). We handle this with a recursive `_normalize_keys()` function at data load time — this is a common pattern when ingesting data from inconsistent sources. It's better to normalize early (at load) than late (at scoring) because it keeps all downstream code clean.

**Split sizing with 5 images**: With only 5 images, the standard 15/15/70 Library Cards split doesn't work. Instead we use 40/20/40 (2/1/2 images), which gives just enough for MIPROv2's train/val requirement while keeping 2 for testing.

---

### 3. Insight

**Context:** 

**Dynamic import pattern**: Using `importlib.import_module(f"benchmarks.{benchmark}.data")` lets scripts work with any benchmark without hardcoding imports. The key contract is that every benchmark package exports the same set of names (`load_and_split`, `Extractor`, `dspy_metric`, etc.). This is essentially Python's version of the "Strategy pattern" — the scripts are the context, and each benchmark package is a swappable strategy.

**Input field name divergence**: Library Cards uses `card_image` while Bibliographic Data uses `page_image`. The scripts need to discover this dynamically. We handle it by inspecting the module's signature or just looking at the example's input fields — `example.inputs().keys()` gives us the right field name.

---

### 4. Insight

**Context:** 

**Why this architecture scales**: Adding a third benchmark (e.g., newspaper clippings) now requires only creating `benchmarks/newspaper_clippings/` with the same 5-file interface — zero changes to any script. The `--benchmark` flag handles routing automatically.

**DSPy saved program portability**: DSPy's `.save()` serializes instructions and field descriptions as plain JSON strings — no Python import paths are stored. This means all existing Library Cards optimized programs in `results/library_cards/optimized/` continue to work despite the file moves, as long as the `Extractor` class has the same signature structure at load time.

---

### 5. Insight

**Context:** 

**Much lower ceiling than Library Cards.** The top score is only 70.2 (vs 89.1 for Library Cards). This suggests the task is fundamentally harder — extracting 14-20 structured entries per page with nested authors, relations, and optional fields is far more complex than one-card-one-record. This means there's significant room for DSPy optimization to add value, but also a risk that the tiny dataset (5 images) limits what optimizers can learn.

**Gemini 2.0 Flash isn't on the leaderboard** — it's an older model not tested by RISE. Our baselines will establish its unoptimized score for comparison. Given that Gemini 2.5 Flash scores 66.9, Flash 2.0 will likely score somewhat lower, making the optimization uplift story even more compelling if we can close the gap.

---

### 6. Insight

**Context:** 

**Defense in depth for key normalization:** We now normalize at two layers: (1) at data load time in `data.py` so GT is consistent for training/demos, and (2) at scoring time in `scoring.py` so both prediction and GT are normalized before comparison. This means even if the model outputs CSL-JSON conventions (`container-title`), it won't be penalized. Without this, any field with a hyphenated key would silently score 0.0, dragging down the average without any obvious error message — a subtle bug that could have taken a while to diagnose from experiment results alone.

---

### 7. Insight

**Context:** 

**Huge variance between pages.** page_5 scores 0.92 (excellent) while page_10 scores 0.38 (poor). This 53-point gap on just 2 test images means aggregate scores will be very volatile. page_10 is likely harder because it has more entries (the largest page) and uses the CSL-JSON format in the GT — even though we normalized keys, the model may still be struggling with the extraction complexity of that page.

**Flash 2.0 at 0.6494 vs leaderboard.** This is slightly below the leaderboard range for comparable models (GPT-4o: 61.9, Gemini 2.5 Flash: 66.9). Note we're only evaluating on 2 test images (the held-out split), not all 5 — so this isn't directly comparable to leaderboard scores which use all pages. The optimization experiments will tell us how much room there is to improve.

---

### 8. Insight

**Context:** 

**Same pattern as Library Cards:** Unoptimized CoT slightly hurts Flash (-0.56 points), mirroring the Library Cards finding where CoT dropped from 0.8134 to 0.7583. The effect is smaller here, but the direction is consistent — without optimized instructions to guide the reasoning step, CoT adds noise rather than clarity. page_5 is identical across both (the model gets it right regardless), while page_10 is where CoT introduces slightly more errors.

This sets up the optimization hypothesis: CoT should become the better module *after* MIPROv2 optimizes its instructions, just as it did for Library Cards.

---

### 9. Insight

**Context:** 

**MIPROv2 wins again, but gains are modest.** The +4.1 point uplift from MIPROv2 is the largest, but nowhere near the +14.3 points seen on Library Cards. The key bottleneck is page_10 — all optimizers improve page_5 marginally (it's already at ~0.92) but struggle to move page_10 past ~0.46. MIPROv2's best trial (#18 of 18) pushed page_10 from 0.38 to 0.46 — an 8-point gain on that one hard page.

**The tiny dataset is the constraint.** With only 1 validation image for MIPROv2 and 2 training images for all optimizers, there's very little signal for the optimizers to learn from. SIMBA and GEPA barely improve over the baseline because their self-reflection and feedback mechanisms need more diverse examples to identify generalizable patterns. The plan's Phase 3 suggestion of leave-one-out or all-images optimization may be worth exploring.

**page_10 is the unsolved problem.** Scoring 0.38-0.46 on a page with ~20 entries suggests the model is either missing entries entirely or getting the structure wrong systematically. A deeper error analysis of page_10 predictions would reveal whether this is a structural issue (e.g., missing entries) or a field-level issue (e.g., wrong values).

---

### 10. Insight

**Context:** 

**Ground truth quality directly bounds benchmark scores.** The `article-journal` vs `journal-article` mismatch affected *every* article entry on page_10 (roughly 14 of 21 entries). Each got a fuzzy score of ~0.87 instead of 1.0 on the type field, dragging down the per-image average by ~3 points. Combined with the key normalization fix from earlier, we've now addressed two systematic annotation inconsistencies in page_10. This is a general lesson: before optimizing model performance, always audit the ground truth for annotation inconsistencies — they create a false ceiling that no amount of optimization can break through.

---

### 11. Insight

**Context:** 

**Leave-one-out** is ideal for tiny datasets like this (5 images). Instead of the current 2/1/2 split where the optimizer only sees 2 training images, LOO gives each fold 3 train + 1 dev images — 50% more training data. The trade-off is 5x the optimization cost (one MIPROv2 run per fold), but we get a score on every image without data leakage.

---

### 12. Insight

**Context:** 

The LOO design gives every image a turn as both training data and test data. Fold 4 is particularly interesting — it uses page_10 (the hardest image) as dev, meaning the optimizer will explicitly try to improve on difficult cases. The current 2/1/2 split may never expose the optimizer to page_10 during training depending on the seed.

---

### 13. Insight

**Context:** 

**MIPROv2 auto levels** control the search budget: `light` bootstraps fewer examples and proposes fewer instructions, `medium` (used in our F1 experiment) uses 12 bootstrap sets, while `heavy` increases to 18 bootstrap sets and proposes 9 instructions. The extra search budget lets the optimizer explore more of the prompt space, potentially finding better instructions — especially valuable when the training set is tiny.

---

### 14. Insight

**Context:** 

The heavy search budget (27 trials vs 18 for medium) found a configuration with a stellar dev score (87.29), but the test improvement was minimal. With only 1 dev image, the optimizer can overfit to that specific image's characteristics. This is exactly why LOO is valuable — it distributes the evaluation across all images.

---

