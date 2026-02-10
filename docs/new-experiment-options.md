# New Experiment Options for DSPy RISE HumBench

*Report prepared 2026-02-09. Updated 2026-02-10 with experiment outcomes. DSPy version: 3.1.3 (latest). All six benchmarks complete.*

---

## 1. Current State

Six RISE benchmarks have been optimized with a consistent pipeline: `dspy.ChainOfThought` module → MIPROv2 medium optimizer → quality-aware Refine(3) at inference time. All use Gemini 2.0 Flash.

| Benchmark | Images | Metric | Best score | Technique |
|-----------|--------|--------|------------|-----------|
| Library Cards | 263 | F1 macro | **0.9167** | MIPROv2 CoT + Refine(3) |
| Bibliographic Data | 5 | Avg fuzzy | **0.7072** | MIPROv2 heavy CoT |
| Personnel Cards | 61 | F1 macro | **0.8894** | MIPROv2 CoT + Refine(3) |
| Business Letters | 57 (98 pg) | F1 macro | **0.7312** | MIPROv2 CoT + Refine(3) |
| Blacklist Cards | 33 | Avg fuzzy | **0.9713** | MIPROv2 CoT + Refine(3) |
| Company Lists | 15 | F1 macro | **0.8771** | MIPROv2 CoT |

Optimizers tested: MIPROv2 (all), SIMBA (Library Cards), GEPA with Gemini 2.5 Pro reflection (Library Cards, Personnel Cards, Business Letters), BootstrapFewShot (Library Cards). Inference wrappers tested: Refine(3) with quality-aware reward (all). Module-level experiments tested: KNN demos (Library Cards), MultiChainComparison (Personnel Cards), verify-and-correct (Business Letters), two-stage pipeline (Bibliographic Data) — none improved on MIPROv2+Refine (see §4 outcomes).

---

## 2. Available DSPy Functionality Not Yet Used

### 2.1 Modules (how the LM is invoked per image)

We currently use `dspy.Predict` and `dspy.ChainOfThought`. DSPy 3.1.3 offers several more:

**RLM (Recursive Language Model)**

The model receives metadata about the context (type, length, preview) but not the full content. It then writes Python code in a sandboxed REPL to explore the data — searching, filtering, sampling — and calls `llm_query()` for semantic analysis on specific snippets. When done, it calls `SUBMIT()` with the final answer.

- *Parameters*: `max_iterations` (default 20), `max_llm_calls` (default 50), optional `sub_lm` for cheaper sub-queries.
- *Pros*: Decouples exploration from answer generation. The model decides dynamically how to decompose a complex extraction. Could solve alignment cascades by processing entries one at a time via code.
- *Cons*: Dramatically more expensive (20+ REPL turns per image vs 1 call). Unclear how well it handles vision inputs — designed for text contexts. Requires the extracted data to be amenable to programmatic exploration.
- *Best fit*: Benchmarks with list-of-entries structure where position-based scoring is a bottleneck (Bibliographic Data, Company Lists).

**MultiChainComparison**

Runs M reasoning attempts (default 3), formats each as `"I'm trying to [rationale], my prediction is [answer]"`, then passes all M to a comparator module that synthesizes a single refined answer.

- *Parameters*: `M` (number of attempts), `temperature` for sampling diversity.
- *Pros*: More principled than Refine — instead of keeping the best by reward score, the LM explicitly reasons about which attempt is best and can combine correct elements from multiple attempts. Built-in "self-consistency with a judge".
- *Cons*: M+1 LM calls per image (M attempts + 1 comparison). The comparison step adds its own error source. With large JSON outputs, the comparator's context window fills quickly.
- *Best fit*: Benchmarks where stochastic errors produce different failure modes per attempt — the comparator can pick correct fields from each (Business Letters, Personnel Cards).

**ReAct (Reasoning + Acting)**

Agent-style module that iterates between reasoning, tool selection, and observation. The model can call external tools during extraction.

- *Parameters*: `tools` (list of callables), `max_iters` (default 20).
- *Pros*: Enables tool-augmented extraction. For Business Letters, a `lookup_person(name)` tool could check the alias table during extraction, not after. For Bibliographic Data, a `validate_entry_id(id)` tool could catch alignment errors in real time.
- *Cons*: Many more LM calls per image (one per tool invocation). Harder to optimize with MIPROv2 (the agent loop is less predictable than a single forward pass). Tool design is non-trivial.
- *Best fit*: Business Letters (where alias table lookup during extraction could eliminate name format errors at source).

**ProgramOfThought**

Generates and executes Python code to solve the task, with retry on errors.

- *Pros*: Could handle structured extraction as a code-generation problem — write a parser for the image's OCR output.
- *Cons*: Requires Deno runtime. Can't operate directly on images (no vision in the sandbox). Would only work as Stage 2 in a multi-stage pipeline.
- *Best fit*: Stage 2 of a multi-stage pipeline (raw OCR text → Python code → structured JSON).

**CodeAct**

Similar to ProgramOfThought but with explicit tool integration. Generates code that calls predefined tools.

- *Pros*: Combines code generation with tool use.
- *Cons*: Same limitations as ProgramOfThought (no vision, needs sandbox). Functions-only interface (no class instances).
- *Best fit*: Not a natural fit for the current image → JSON pipeline.

### 2.2 Optimizers (how the prompt is tuned)

We've used MIPROv2, SIMBA, GEPA, and BootstrapFewShot. Available but untested:

**KNNFewShot**

At inference time, uses an embedding model to find the k most similar training examples and attaches them as demos. Different demos for every input.

- *Parameters*: `k` (demo count), `trainset`, `vectorizer` (a `dspy.Embedder`).
- *Pros*: Addresses the core problem of static demo selection — Library Cards' diverse formats (typed vs handwritten, German vs French, dissertations vs references) would get format-specific demos. No optimization loop needed (runs BootstrapFewShot internally per query).
- *Cons*: Requires a good embedding space. For vision tasks, you'd need to embed image descriptions or OCR text, not the images directly. Adds an embedding call per inference. Unclear how to combine with MIPROv2's instruction optimization (KNNFewShot replaces demo selection entirely).
- *Best fit*: Library Cards (most diverse image formats), potentially Personnel Cards (varying table layouts).

**BestOfN**

Runs module N times with different rollout IDs at temperature=1.0, returns the best prediction by reward score. No feedback generation between attempts.

- *Parameters*: `N`, `reward_fn`, `threshold` (same interface as Refine).
- *Pros*: Direct ablation of Refine's feedback loop — isolates whether "retry with feedback" matters or if "just retry" is sufficient. Simpler implementation, slightly cheaper (no feedback LM calls).
- *Cons*: Without feedback, each attempt is independent — the model can't learn from why previous attempts failed.
- *Best fit*: All 4 benchmarks as a controlled comparison to Refine. Especially informative for Business Letters where Refine gave +9.3 pts.

**Ensemble**

Combines multiple programs (e.g. MIPROv2 + GEPA + SIMBA) via a configurable reduce function. Supports majority voting via `dspy.majority`.

- *Parameters*: `programs` (list of compiled programs), `reduce_fn`, optional `size` for sampling.
- *Pros*: Leverages complementary strengths of different optimizers. MIPROv2 (demos) + GEPA (instructions) might cover different failure modes. We already have multiple optimized programs per benchmark.
- *Cons*: Multiplies inference cost by number of programs. Majority voting on JSON structures is non-trivial (need field-level voting, not output-level). Reduce function design matters.
- *Best fit*: Personnel Cards (MIPROv2 at 0.8858 and GEPA at 0.8750 are closest, suggesting complementary coverage).

**COPRO (Competitive Prompt Optimizer)**

Instruction-only optimizer using breadth-first generation + iterative depth refinement. Different search strategy from GEPA's genetic evolution.

- *Parameters*: Depth (refinement iterations), breadth (candidates per round), temperature.
- *Pros*: Different instruction search landscape — might find optima that GEPA's genetic search misses. Tracks min/max/avg/stddev statistics.
- *Cons*: Still instruction-only (no demo selection). GEPA already underperformed MIPROv2 on all benchmarks, and COPRO is an older optimizer.
- *Best fit*: Could serve as GEPA ablation on Personnel Cards (where instruction-only came closest to MIPROv2).

**BootstrapFewShotWithRandomSearch**

Generates random candidate programs and evaluates them. Progressive elimination strategy.

- *Parameters*: `num_candidate_programs`, `stop_at_score`, `metric_threshold`.
- *Pros*: Direct ablation of MIPROv2's Bayesian search — answers "is Bayesian search actually better than random for these tasks?" Lower computational overhead per trial.
- *Cons*: No instruction optimization (demo selection only). Needs many trials to match Bayesian efficiency.
- *Best fit*: Scientific ablation on Library Cards (where MIPROv2 medium found its best trial as #18 of 18).

**InferRules**

Extends BootstrapFewShot by generating explicit extraction rules from training examples in natural language.

- *Parameters*: Inherits from BootstrapFewShot, adds rule generation.
- *Pros*: Produces human-readable rules that can be inspected and edited. Could generate rules like "When you see 's.' on a separate line, classify as Reference" or "Always output names as First Last".
- *Cons*: Rules are generated per-predictor, not globally. Rule quality depends on training examples.
- *Best fit*: Business Letters (where explicit name-format rules are critical) and Library Cards (where type classification rules are learnable).

### 2.3 Inference-Time Wrappers

We've used Refine(3) with quality-aware reward. Also available:

**BestOfN** (covered above as an optimizer comparison, but note it's technically a prediction wrapper like Refine, not a teleprompter).

### 2.4 Evaluation Metrics

**SemanticF1**

Built-in DSPy metric that uses a ChainOfThought module internally to compute semantic precision and recall. Returns an F1 score.

- *Pros*: LM-judged semantic similarity instead of character-level fuzzy matching. Could capture cases where the extraction is semantically correct but string-different (e.g. "Prof. Dr." vs "Professor Doctor").
- *Cons*: Adds LM calls to every evaluation step. Our current fuzzy metrics are much cheaper and are aligned with the upstream RISE benchmark. Using SemanticF1 for optimization would diverge from leaderboard comparability.
- *Best fit*: Not recommended for scoring (must match upstream). Could be used as a secondary diagnostic to identify cases where fuzzy F1 undervalues correct extractions.

### 2.5 Pipeline Architectures (not specific to DSPy features)

**Multi-stage pipeline** (2 modules optimized jointly)

Stage 1: `image → raw_text` (OCR/transcription). Stage 2: `raw_text + schema → structured_json`. MIPROv2 optimizes both stages end-to-end via the final metric.

- *Pros*: Decouples vision errors from schema mapping. Stage 1 errors are visible and debuggable. Stage 2 can be cheaper (text-only, no image tokens). Each stage can use a different model.
- *Cons*: Doubles LM calls per image. Stage 1 errors propagate (no image context in Stage 2). Requires careful signature design for the intermediate representation.
- *Best fit*: Business Letters (multi-page documents where OCR text is more manageable than multi-image input) and Bibliographic Data (complex nested schema where structuring is the hard part).

**Self-consistency / majority vote**

Run k=5 independent extractions with the same program, take majority vote per field.

- *Pros*: Reduces stochastic errors without any prompt changes. Could eliminate zero-scoring outliers.
- *Cons*: k× inference cost. Majority voting on structured JSON requires field-level alignment. More expensive than Refine(3) which stops early at threshold.
- *Best fit*: Personnel Cards (3/43 still score 0.0, possibly stochastic).

---

## 3. New Benchmark Candidates

Two additional RISE benchmarks follow the image → JSON pattern:

### 3.1 Blacklist Cards (33 images)

1940s British blacklist index cards of companies. Simple flat schema: company (transcription), location (transcription), b_id (transcription), optional date and information list.

- **Metric**: Average fuzzy score (same as Bibliographic Data). Ranking metric in `meta.json`: `"fuzzy"`.
- **Scoring**: Field-level fuzzy average, skip `metadata.*` fields. Null/"null" normalized to empty string.
- **Leaderboard**: GPT-4.1 Mini 95.65, Gemini 2.5 Flash Preview 95.31, Gemini 2.0 Flash 90.85.
- **Data split**: 33 images → 5 train / 5 dev / 23 test (15/15/70).
- **Integration effort**: Low. Flat schema, familiar scoring pattern, reuses `scoring_helpers.py` fuzzy infrastructure.
- **Research question**: Can DSPy optimization push Gemini 2.0 Flash past 90% on a near-saturated benchmark? The gap between Flash (90.85) and the leader (95.65) is only ~5 pts — can few-shot demos close it?
- **Estimated cost**: ~$1-2 for full MIPROv2 run.

### 3.2 Company Lists (15 images)

Printed book pages listing Swiss companies. List-based schema: `page_id` + `entries[]` with `entry_id`, `company_name`, `location`.

- **Metric**: F1 macro/micro with 0.92 fuzzy threshold (same as Library Cards). Ranking metric: `"f1_macro"`.
- **Scoring**: Field-level fuzzy F1 with parent-key filtering. Position-based entry matching (entries[0] vs entries[0]).
- **Leaderboard**: GPT-5 58.40, o3 55.20, Gemini 2.0 Flash 47.93.
- **Data split**: 15 images → 2 train / 2 dev / 11 test (very small, similar to Bibliographic Data).
- **Integration effort**: Medium. List handling needed, position-based scoring raises alignment cascade risk.
- **Research question**: Are the low scores (~48-58%) due to extraction quality or position-based alignment cascades (same issue as Bibliographic Data)? If the latter, optimization may hit the same ceiling.
- **Risk**: With only 15 images and position-based scoring, this may reproduce the Bibliographic Data situation where the metric is the bottleneck.
- **Estimated cost**: ~$0.50-1 for full run.

### 3.3 Other candidates (lower priority)

- **Medieval Manuscripts** (12 images): Folio text extraction. Medium dataset, would need metric investigation.
- **Fraktur Adverts** (5 images): Date/section/text extraction from historical advertisements. Very small.
- **Book Advert XML** (50 images): XML correction task — fundamentally different from extraction, would need a different pipeline.

---

## 4. Per-Benchmark Recommendations

### 4.1 Library Cards → KNNFewShot

**Current situation**: 0.9167 F1 with MIPROv2 + Refine. The score is already high, but ~15 test images still fall below 0.7 F1, and the main failure mode is diversity — the 263 images span typed vs handwritten cards, German vs French text, dissertations vs references vs monographs. MIPROv2 selected 2 static demos that work well on average but can't cover all card types.

**Why KNNFewShot**: This optimizer directly addresses the diversity problem. Instead of showing the same 2 demos to every card, KNNFewShot finds the k most similar training examples at inference time. A handwritten French dissertation card would get demos of other handwritten French cards; a typed German reference card would get demos of typed German cards. The embedding-based retrieval acts as an implicit card-type classifier.

**Implementation sketch**:
1. Generate text descriptions of each training image (OCR text or a brief model-generated summary).
2. Create a `dspy.Embedder` using a sentence transformer (e.g. `all-MiniLM-L6-v2`).
3. Compile with `KNNFewShot(k=3, trainset=train_set, vectorizer=embedder)`.
4. Combine with MIPROv2's optimized instruction (keep the instruction, replace demo selection).
5. Evaluate on the 185-image test set.

**Expected outcome**: Modest gain (+1-3 pts F1) from better demo coverage on the tail of low-scoring cards. The ceiling is close, so gains will be small but targeted at the hardest images.

**Cost**: ~$2-3 (embedding calls are cheap; main cost is the k=3 demos increasing prompt tokens).

**Risk**: The embedding space may not capture visual card-type similarity well from text descriptions alone. If OCR quality varies, the nearest neighbors may be noisy.

**Outcome**: KNN (k=3) produced **identical results** to static MIPROv2 demos (0.9017 f1_macro, TP/FP/FN unchanged at 1546/156/161). Tested with both `sentence-transformers/all-MiniLM-L6-v2` (English-only) and `gemini-embedding-001` (multilingual) — identical results with both. The bottleneck was not embedding quality but instruction-demo coupling: MIPROv2's instruction and demos were jointly optimized, and swapping demos at inference broke this coupling without adding value. The prediction-JSON embeddings did not capture the visual card-type diversity that determines which demos are most useful.

### 4.2 Bibliographic Data → RLM (Recursive Language Model)

**Current situation**: 0.7072 average fuzzy. The bimodal score distribution (pages 2/4/5 at 0.89-0.91, pages 3/10 at ~0.39) is caused by position-based alignment cascades, not poor extraction. When the model inserts an extra entry or flattens nested entries, every subsequent entry is scored against the wrong ground truth.

**Why RLM**: The fundamental problem is that single-shot extraction of 14-20 entries per page is fragile — one structural error cascades to all downstream entries. RLM inverts this by letting the model programmatically explore the page content. Instead of dumping all entries in one JSON blob, the model could:
1. First scan the page to count entries and identify their IDs.
2. Process each entry individually via `llm_query()`.
3. Handle nested entries (like page_10's entries 146-148 under 145) through explicit code logic.
4. Build the final JSON array entry by entry, with ID-aware ordering.

This decomposes the monolithic extraction into an iterative, controllable process. If entry 145 has nested sub-entries, the model can write code to handle that structure explicitly rather than hoping the single-shot extraction gets the nesting right.

**Implementation sketch**:
1. Create an RLM signature: `page_image: Image, schema: str -> entries_json: str`.
2. Set `max_iterations=25`, `max_llm_calls=30` (enough for 20 entries).
3. Provide the image as context that the model can query via `llm_query()`.
4. The model writes code to iterate over entries, calling `llm_query("Extract entry N from this page image")` per entry.
5. Evaluate against the standard position-based metric.

**Expected outcome**: If the model processes entries sequentially with correct ID tracking, alignment cascades should disappear. Pages 3 and 10 could jump from ~0.39 to 0.70-0.85, lifting the aggregate from 0.70 to potentially 0.80+.

**Cost**: ~$3-5 per full evaluation (many more LM calls per page, but only 5 images). Optimization would be expensive (MIPROv2 on an RLM module means 20+ calls per trial per image).

**Risk**: High. RLM is designed for text contexts, and it's unclear how well it handles vision inputs. The model may struggle to write useful code for image-based extraction. The REPL sandbox may not support the `llm_query()` calls with image inputs. This is the most experimental recommendation — if RLM proves infeasible with vision, fall back to a **multi-stage pipeline** (Stage 1: image → raw text listing all entries; Stage 2: RLM on the text to structure them).

**Outcome**: RLM was not tested (vision feasibility concerns remained unresolved). The **fallback two-stage pipeline** was tested instead — Stage 1: image → structured text listing, Stage 2: text → JSON. Result: **0.6265 fuzzy, -8.1 pts below single-stage MIPROv2** (0.7072). Both pages degraded: page 5 dropped from 0.9126 to 0.8615 (-5.1 pts) and page 10 from 0.5018 to 0.3915 (-11.0 pts). The structuring stage lost access to the original image, missing visual cues for entry boundaries. The alignment cascade errors on page 10 were not reduced — the text listing inherited the same ordering errors from the model's interpretation of the image.

### 4.3 Personnel Cards → MultiChainComparison

**Current situation**: 0.8894 F1 with MIPROv2 + Refine. Very close to ceiling. 3/43 test cards still score 0.0 (down from 8/43 in the predict baseline). Refine added only +0.36 pts — most cards are already well-extracted, and the 3 zero-scorers fail structurally (blank/unreadable cards).

**Why MultiChainComparison**: The remaining improvement opportunity is narrow: reduce false positives (94 remaining) and possibly rescue 1-2 of the 3 zero-scoring cards. MultiChainComparison generates M=3 reasoning attempts and has a comparator module synthesize the best answer. Unlike Refine (which keeps the single best attempt by reward score), the comparator can combine correct fields from different attempts — e.g., take the salary column from attempt 1 but the date column from attempt 2.

For Personnel Cards specifically, the table structure means each row is semi-independent. One attempt might correctly extract rows 1-5 but hallucinate row 6, while another attempt gets row 6 right but misses row 3. The comparator, seeing all three attempts, can select the most consistent row set.

**Implementation sketch**:
1. Wrap the existing MIPROv2-optimized Extractor in `dspy.MultiChainComparison(signature, M=3)`.
2. The 3 attempts run with temperature=1.0 for diversity.
3. The comparator receives all 3 formatted attempts and produces a synthesized JSON.
4. Evaluate on the 43-card test set.

**Expected outcome**: Small gain (+0.5-1.5 pts F1) from better field-level synthesis. The comparator should reduce false positives by filtering out hallucinated fields that appear in only 1 of 3 attempts. Unlikely to rescue the truly blank cards (those are vision-level failures, not extraction-level).

**Cost**: ~$4-6 (4 LM calls per card × 43 test cards, but Flash is cheap).

**Risk**: Low. The technique is straightforward and the fallback is "no worse than current". The main unknown is whether the comparator module can meaningfully reason about JSON table structures. If the comparison step itself introduces errors, the net effect could be neutral.

**Outcome**: MultiChainComparison (M=3) scored **0.8763 f1_macro, -0.95 pts below MIPROv2 CoT** (0.8858). Implementation required subclassing as `FullMultiChainComparison` to fix a DSPy bug: `MultiChainComparison.forward()` truncates each completion to its first line via `.split("\n")[0]`, destroying multi-line JSON output. Even with the fix, the comparator at temperature=1.0 introduced noise — while it produced 3 perfect scores (1.0) on cards where MIPROv2 scored <1.0, it degraded several others. The 3/43 zero-scoring cards remained at zero. At 4× the LM calls per card, worse cost-performance than Refine(3).

### 4.4 Business Letters → ReAct with persons.json Lookup Tool

**Current situation**: 0.7312 F1 with MIPROv2 + Refine. The biggest remaining problem is person name matching — names must exactly match the `persons.json` alias table (119 entries, all "First Last" format). Refine helped hugely (+9.3 pts) because name format errors are stochastic, but 26 false positives remain. The model extracts names like "Vischer" or "M. Vischer" that don't match the alias "Max Vischer".

**Why ReAct with a lookup tool**: The current pipeline extracts names and hopes they match the alias table. ReAct inverts this by giving the model direct access to the alias table during extraction. When the model reads a letter and identifies a person, it can call `lookup_person("Vischer")` and get back `["Max Vischer", "Wilhelm Vischer-Bilfinger"]` — the matching aliases. It then knows to output "Max Vischer" instead of guessing at the first name.

This directly addresses the root cause: the model lacks information about which name variants are valid. Currently, this information is embedded implicitly in few-shot demos (which show "First Last" format), but the model can't generalize to names it hasn't seen in demos. With a lookup tool, every name can be verified against the alias table in real time.

**Implementation sketch**:
1. Load `persons.json` alias table.
2. Define tools:
   - `lookup_person(partial_name: str) -> list[str]`: fuzzy-search the alias table, return top-5 matches.
   - `get_all_persons() -> list[str]`: return the full list of 119 known persons.
   - `validate_date(date_str: str) -> bool`: check date format consistency.
3. Create a `dspy.ReAct(signature, tools=[lookup_person, get_all_persons, validate_date], max_iters=15)`.
4. The model reads the letter pages, identifies candidate persons, calls `lookup_person()` to verify/correct each name, and builds the output JSON.
5. Evaluate on the 41-letter test set.

**Expected outcome**: Significant gain (+3-8 pts F1) from reduced false positives on person matching. The lookup tool should eliminate name format errors entirely for persons in the alias table. Dates and organizations would see smaller gains.

**Cost**: ~$5-10 for evaluation (multiple tool calls per letter × 41 letters). Optimization with MIPROv2 would be expensive (~$15-25) due to the agent loop.

**Risk**: Medium. ReAct's agent loop is less predictable than a single forward pass. The model might over-rely on the tool (calling it for every word), make inefficient tool sequences, or fail to integrate tool results correctly. MIPROv2 optimization of ReAct modules is less well-tested than optimization of simple Predict/CoT modules. If ReAct proves too unstable, fall back to **InferRules** — generate explicit name-format rules from the training set and inject them into the MIPROv2 instruction.

**Outcome**: ReAct was not tested. A simpler **verify-and-correct** approach was tried instead: run the MIPROv2-optimized program, parse output, look up each predicted person name in `persons.json`, and replace with the canonical alias if found. Exact-match mode made **zero corrections** — MIPROv2's few-shot demos already taught the model to output "First Last" format, so there were no format errors left to fix. Substring matching (e.g., matching "Ritter" to "Fritz Ritter-Dreier") was too aggressive, adding +19 false positives and dropping the score to 0.6021. The remaining false negatives are names the model fails to detect entirely, not names it detects in the wrong format. Post-hoc correction is unnecessary when demos already teach the correct format.

### 4.5 New Benchmarks → Blacklist Cards with MIPROv2 CoT + Refine(3)

**Rationale**: Blacklist Cards is the lowest-effort addition (flat schema, fuzzy metric, familiar pattern) and asks a different research question than the other four benchmarks: *can optimization help when the baseline is already strong?* At 90.85 fuzzy for unoptimized Flash, this is the closest to a saturated benchmark in the suite. If MIPROv2 can push Flash past 95 (matching GPT-4.1 Mini's 95.65), it strengthens the paper's central claim that optimization makes cheap models competitive with expensive ones.

**Why not Company Lists**: While Company Lists asks an interesting question (can optimization help at the floor?), its 15-image dataset with position-based F1 scoring is likely to reproduce the Bibliographic Data situation — alignment cascades capping gains regardless of prompt quality. The research payoff is uncertain, and the small dataset makes optimization results unreliable.

**Implementation**: Standard pipeline — create `benchmarks/blacklist_cards/` package, reuse shared infrastructure, run MIPROv2 medium-CoT, then Refine(3). Estimated total cost: ~$2-3.

**Outcome — Blacklist Cards**: Done. MIPROv2 medium-CoT + Refine(3) achieved **0.9713 fuzzy**, surpassing the RISE leaderboard top (GPT-4.1: 95.7) by 1.5 points. Even without Refine, MIPROv2 CoT (0.9599) matched leaderboard leaders. Confirmed: optimization helps even at the ceiling.

**Outcome — Company Lists**: Also completed, despite the recommendation against it. MIPROv2 medium-CoT achieved **0.8771 f1_macro**, far exceeding the leaderboard top (GPT-5: 58.4) by ~29 points. The alignment cascade concern proved less severe than expected — Company Lists' simpler schema (3 flat fields vs Bibliographic Data's 18+ nested fields) makes positional errors less catastrophic. However, the "why not" prediction was partially correct: Refine(3) hurt by -1.1 pts (near-threshold pages regressed on retry), and the dev-test gap (-7.4 pts) confirmed small-sample overfitting risk.

---

## 5. Summary of Recommendations

| Benchmark | Recommended | Technique | Expected | Actual | Verdict |
|-----------|-------------|-----------|----------|--------|---------|
| **Library Cards** | Dynamic demo selection | KNNFewShot (k=3) | +1-3 pts F1 | ±0.0 pts | Neutral — joint optimization resists demo swapping |
| **Bibliographic Data** | Iterative entry extraction | Two-stage pipeline* | +5-15 pts fuzzy | **-8.1 pts** | Worse — lost image context |
| **Personnel Cards** | Multi-attempt synthesis | MultiChainComparison (M=3) | +0.5-1.5 pts F1 | **-0.95 pts** | Worse — comparator introduced noise |
| **Business Letters** | Tool-augmented extraction | Verify-and-correct* | +3-8 pts F1 | ±0.0 pts | Neutral — zero corrections needed |
| **Blacklist Cards** | Standard pipeline | MIPROv2 CoT + Refine(3) | Establish baseline | **0.9713** | Success — beat leaderboard |
| **Company Lists** | *(not recommended)* | MIPROv2 CoT | — | **0.8771** | Success — beat leaderboard by 29 pts |

\* RLM and ReAct were not tested; simpler variants were tried instead (see §4.2 and §4.4 outcomes).

**None of the four module-level experiments improved on MIPROv2+Refine.** The new benchmarks (Blacklist Cards and Company Lists) were successful. The meta-lesson: MIPROv2's jointly optimized instruction+demo programs are tightly coupled — modifying any component (swapping demos, adding a comparator, post-processing outputs, splitting the pipeline) disrupts the balance without adding compensating value. The standard recipe (optimized instruction + optimized demos + quality-aware Refine) remains the best approach.

---

## 6. Decision Factors

**If prioritizing scientific novelty**: RLM on Bibliographic Data and ReAct on Business Letters are the most novel — they test fundamentally different extraction paradigms (iterative vs tool-augmented) against the current single-shot approach.

**If prioritizing reliability**: Blacklist Cards (new benchmark) and MultiChainComparison on Personnel Cards are safest. Both use well-understood techniques with predictable outcomes.

**If prioritizing maximum F1 lift**: ReAct on Business Letters has the highest expected absolute gain (+3-8 pts), followed by RLM on Bibliographic Data (+5-15 pts, but high variance).

**If prioritizing paper narrative**: Running BestOfN as a controlled ablation of Refine across all 4 benchmarks (answering "does the feedback loop in Refine actually matter, or is retry alone sufficient?") would strengthen the Refine section of the write-up with minimal effort (~$3-4 total).
