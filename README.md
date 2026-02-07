# DSPy Optimization for RISE Humanities Data Benchmarks

This project applies [DSPy](https://dspy.ai/) — a framework for programming and optimizing language model pipelines — to the [RISE Humanities Data Benchmark](https://github.com/rise-unibas/humanities_data_benchmark), a suite of structured information extraction tasks over digitised historical documents.

The aim is to explore whether automated prompt optimization and few-shot example selection can improve LLM performance on document understanding tasks that are central to digital humanities research, compared to the hand-crafted prompts used in the benchmark.

## Aims

The RISE Humanities Data Benchmark evaluates LLMs on extracting structured data from historical documents — library catalog cards, personnel records, medieval manuscripts, business letters, and more. Each task involves reading a scanned document image and producing a structured JSON output that matches a ground-truth annotation.

The benchmark establishes baseline scores using carefully engineered prompts. But prompt engineering is manual, task-specific, and hard to iterate on systematically. This project investigates a different approach:

- **Can automated optimization match or surpass hand-crafted prompts?** DSPy optimizers search over instruction phrasings and few-shot example selections to find configurations that maximize a task-specific metric.
- **How do optimized pipelines generalize across benchmark tasks?** Starting with library catalog cards, the pipeline is designed to be adapted to other RISE benchmark tasks with minimal changes — swap the schema, scoring function, and data loader.
- **What is the cost-performance tradeoff?** Vision LLM calls with image inputs are expensive. DSPy's optimization strategies (MIPROv2, BootstrapFewShot) add few-shot demonstrations that increase per-call cost. Understanding where this investment pays off is important for practical adoption.

## DSPy Methodology

### What DSPy does

DSPy treats LLM interactions as modular, optimizable programs rather than static prompt strings. A DSPy pipeline consists of:

1. **Signatures** — typed input/output specifications (e.g., `card_image: Image -> document: str`) that define *what* the LLM should do, not *how*.
2. **Modules** — composable building blocks (like `dspy.Predict`, `dspy.ChainOfThought`) that implement the control flow.
3. **Metrics** — task-specific scoring functions that evaluate output quality.
4. **Optimizers** — algorithms that automatically tune the pipeline by modifying instructions, selecting few-shot demonstrations, or adjusting module composition.

The key insight is that the *prompt* is not the program — it is a compiled artifact. DSPy compiles a declarative specification (signature + module) into an effective prompt by searching over instruction phrasings and demonstration examples that maximize the metric on a training set.

### Why DSPy for RISE benchmarks

The RISE benchmarks are well-suited for DSPy optimization for several reasons:

- **Structured output with clear metrics.** Each benchmark has a well-defined JSON schema and a quantitative scoring function (field-level fuzzy F1). This gives DSPy's optimizers a concrete signal to optimize against.
- **Consistent task structure.** Every benchmark follows the same pattern: read a document image, extract structured data. This means a single DSPy pipeline architecture (image → structured JSON) can be reused across tasks.
- **Room for improvement via demonstrations.** The benchmark's hand-crafted prompts describe the extraction rules in natural language. But some extraction decisions (e.g., distinguishing "Dissertation or thesis" from "Reference" based on a subtle "s." marker) might be better communicated through worked examples than through instructions alone.
- **Cost-constrained optimization.** With ~263 images per task and vision API calls costing $0.01–0.03 each, the dataset is small enough that optimization runs remain affordable while being large enough for meaningful held-out evaluation.

### Optimizers used

- **MIPROv2** (Multiprompt Instruction Proposal Optimizer v2): Jointly optimizes instructions and bootstraps few-shot demonstrations. Uses a Bayesian search over candidate prompts, evaluating each on a validation set. The `auto="light"` setting keeps the search budget small.
- **BootstrapFewShot**: A simpler optimizer that selects demonstrations from the training set by running the model and keeping examples where the metric exceeds a threshold. No instruction optimization — just few-shot selection.

## Technical Approach

### Pipeline architecture

```
┌─────────────┐     ┌──────────────────────┐     ┌──────────────┐
│  Card Image  │ ──> │  LibraryCardExtractor │ ──> │  JSON Output  │
│  (dspy.Image)│     │  (dspy.Predict)       │     │  (document)   │
└─────────────┘     └──────────────────────┘     └──────────────┘
                              │
                    ┌─────────┴─────────┐
                    │  LibraryCard       │
                    │  Extraction        │
                    │  (dspy.Signature)  │
                    │                   │
                    │  Input: card_image │
                    │  Output: document  │
                    └───────────────────┘
```

The module wraps a single `dspy.Predict` call with a signature that specifies the expected JSON schema in its output field description. The LM receives the image and the schema description, and returns a JSON string which is then parsed and scored.

### Scoring

Scoring faithfully reimplements the benchmark's field-level fuzzy F1:

1. Both prediction and ground truth are flattened to their leaf keys (e.g., `author.last_name`, `publication.year`).
2. For each key present in either side:
   - If both have non-null values and their fuzzy similarity (rapidfuzz ratio) is ≥ 0.92 → **true positive**
   - If both have values but similarity < 0.92 → **false positive + false negative**
   - If only the prediction has a value → **false positive**
   - If only the ground truth has a value → **false negative**
   - If both are null/empty → **skip** (not counted)
3. Per-image F1 is computed from TP/FP/FN. Macro F1 averages per-image F1 scores; micro F1 aggregates TP/FP/FN globally.

### Data split

The 263 matched image/ground-truth pairs are split deterministically (seed=42):

| Split | Count | Purpose |
|-------|-------|---------|
| Train | 39 (15%) | Few-shot candidate pool + MIPROv2 training signal |
| Dev   | 39 (15%) | MIPROv2 validation / optimizer evaluation |
| Test  | 185 (70%) | Held-out final evaluation |

### Project structure

```
src/
  config.py       # LM setup, path constants
  schema.py       # Pydantic Document schema (mirrors benchmark)
  data.py         # Data loading, splitting, dspy.Example conversion
  signature.py    # DSPy Signature with schema-guided output description
  module.py       # DSPy Module wrapping Predict
  scoring.py      # Field-level fuzzy F1 metric (benchmark-compatible)

scripts/
  check_rate_limits.py     # Check provider API rate limits
  evaluate_baseline.py     # Run unoptimized module on test set
  optimize.py              # Run MIPROv2 or BootstrapFewShot
  evaluate_optimized.py    # Evaluate saved optimized program
  compare_results.py       # Print side-by-side comparison
```

### Running the pipeline

```bash
# Install dependencies
uv sync

# 0. Check provider rate limits
uv run python scripts/check_rate_limits.py

# 1. Evaluate unoptimized baseline
uv run python scripts/evaluate_baseline.py --model gemini-2.5-pro

# 2. Run optimization (MIPROv2 recommended)
uv run python scripts/optimize.py --optimizer mipro --auto light --model gemini-2.5-pro --num-threads 16
# or: uv run python scripts/optimize.py --optimizer bootstrap --model gemini-2.5-pro

# 3. Evaluate optimized program on test set
uv run python scripts/evaluate_optimized.py --program results/optimized/mipro_gemini-2.5-pro_optimized.json --model gemini-2.5-pro

# 4. Compare all results
uv run python scripts/compare_results.py
```

### Multi-provider support

The pipeline supports multiple LLM providers via [litellm](https://docs.litellm.ai/). Configure API keys in `.env` and select a model with `--model`:

```bash
uv run python scripts/optimize.py --optimizer mipro --model gemini-2.5-pro --num-threads 16
uv run python scripts/evaluate_baseline.py --model gemini-2.5-pro
uv run python scripts/check_rate_limits.py  # check provider rate limits
```

Available presets: `gpt-4o`, `gpt-4o-mini`, `gemini-3-pro-preview`, `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.0-flash`, `claude-sonnet`, `claude-haiku`, and OpenRouter variants (`or-gemini-2.5-pro`, `or-claude-sonnet`, `or-gpt-4o`). Any full litellm model string also works.

### Benchmark reference (RISE leaderboard)

Top scores on the Library Cards benchmark from the [RISE dashboard](https://rise-services.rise.unibas.ch/benchmarks/) (full 263 images, hand-crafted prompts):

| Rank | Model | f1_macro |
|------|-------|----------|
| 1 | Gemini 3 Pro (preview) | 89.1 |
| 2 | GPT-5 | 87.9 |
| 3 | Claude 3.5 Sonnet | 87.6 |
| 4 | Gemini 2.5 Pro | 87.2 |
| 5 | GPT-4.1 | 86.5 |

## Results

### MIPROv2 + Gemini 2.5 Pro (test split, 185 images)

| Configuration | f1_macro | f1_micro | Precision | Recall |
|---|---|---|---|---|
| **MIPROv2 optimized** | **0.8912** | **0.8965** | 0.8852 | 0.9080 |
| DSPy baseline (GPT-4o) | 0.8172 | 0.8237 | 0.8608 | 0.7897 |

The MIPROv2-optimized pipeline with Gemini 2.5 Pro achieves **f1_macro=0.8912** on the held-out test set (185 images), a **+7.4 point improvement** over the unoptimized DSPy baseline with GPT-4o. This result is on par with the benchmark's best hand-crafted prompt scores — and notably exceeds Gemini 2.5 Pro's own benchmark score (87.2) by +1.7 points.

The optimization discovered an effective instruction + 2 few-shot demonstration combination. The optimized instruction emphasizes role-framing and OCR accuracy, while the bootstrapped demonstrations implicitly teach the extraction schema through worked examples rather than explicit field-by-field rules.

### Key findings

- **Automated optimization matches hand-crafted prompts.** MIPROv2's f1_macro (0.8912) is competitive with the benchmark leaderboard's top scores (Gemini 3 Pro preview: 89.1, GPT-5: 87.9, GPT-4o: 85.7), despite evaluating on a held-out 70% test subset rather than the full 263-image dataset.
- **Recall is the main driver of improvement.** The optimized pipeline achieves 0.9080 recall vs the baseline's 0.7897 — the model extracts more fields correctly, largely thanks to few-shot demonstrations that clarify the expected output structure.
- **Few-shot demos matter more than verbose instructions.** MIPROv2's best instruction is concise (2 sentences of role-framing) compared to the benchmark's detailed multi-paragraph prompt. The 2 bootstrapped demonstrations do the heavy lifting.

### Issues encountered

**Rate limiting is the main practical challenge for DSPy optimization with vision models:**

- **OpenAI (GPT-4o):** The initial MIPROv2 run was severely degraded by a 30,000 TPM (tokens per minute) rate limit. With image inputs consuming thousands of tokens per call and 16 concurrent threads, most trials hit rate limit errors, causing JSON parse failures that were scored as 0.0. The best trial scored only 78.3 on the dev set.
- **Gemini 3 Pro Preview:** Attempted optimization hit a 25 RPM (requests per minute) per-model limit — far more restrictive than GA models. Only 2 of 11 trials completed (best: 84.59). The daily quota was also exhausted mid-run.
- **Gemini 2.5 Pro:** No rate limit issues. The GA model has generous limits (1M+ TPM), making it well-suited for optimization workloads with many parallel calls.

**Recommendation:** Use GA (generally available) models with high rate limits for optimization. Preview/experimental models typically have restrictive quotas unsuitable for MIPROv2's parallel evaluation strategy. Use `scripts/check_rate_limits.py` to verify provider limits before running optimization.

## Adapting to other RISE benchmarks

To apply this pipeline to a different RISE benchmark task:

1. Update `src/schema.py` with the task's Pydantic schema
2. Update `src/signature.py` with the appropriate output description
3. Update `src/data.py` to point to the new images/ground_truths directories
4. Update `src/scoring.py` if the task uses a different scoring method
5. Symlink the new benchmark's data directories

The module, optimizer scripts, and comparison tooling remain unchanged.

### Next benchmarks to test

The following RISE benchmarks are strong candidates for DSPy optimization, given their structural similarity to Library Cards (image → structured JSON extraction). Current leaderboard top scores are included to indicate headroom for improvement.

| Benchmark | Description | Top score | Why it's a good fit |
|---|---|---|---|
| **[Bibliographic Data](https://github.com/RISE-UNIBAS/humanities_data_benchmark/tree/main/benchmarks/bibliographic_data)** | Extract publication details, authors, dates, and metadata from digitized historical documents | ~66.9 | Closest analog to Library Cards — same bibliographic domain, same image → JSON pattern, minimal schema changes needed. Moderate leaderboard scores suggest significant room for optimization gains. |
| **[Personnel Cards](https://github.com/RISE-UNIBAS/humanities_data_benchmark/tree/main/benchmarks/personnel_cards)** | Extract structured employment data (position, location, salary, dates) from 20th-century Swiss personnel card tables | ~79.0 | Table-like card images with structured output — similar pipeline to Library Cards, but with tabular rather than bibliographic data. Schema is more complex (each field has diplomatic transcript, interpretation, and is_crossed_out sub-fields). |
| **[Business Letters](https://github.com/RISE-UNIBAS/humanities_data_benchmark/tree/main/benchmarks/business_letters)** | Extract persons, organizations, dates, and locations from 20th-century Swiss historical correspondence | ~67.8 | Same image → JSON pattern with moderate scores. Schema is more complex (multiple dataclass files for persons, organizations, categories). |

Other benchmarks that follow the image → JSON pattern but may be harder to improve with DSPy optimization:

| Benchmark | Top score | Notes |
|---|---|---|
| **[Blacklist Cards](https://github.com/RISE-UNIBAS/humanities_data_benchmark/tree/main/benchmarks/blacklist_cards)** | ~95.5 | Card-based extraction from 1940s British company index cards. Top models already score 95%+, leaving little room for optimization gains. |
| **[Company Lists](https://github.com/RISE-UNIBAS/humanities_data_benchmark/tree/main/benchmarks/company_lists)** | ~49.7 | List-format extraction from company directory pages. Very low top scores suggest fundamental task difficulty beyond what prompt optimization alone may address. |

The remaining RISE benchmarks (Book Advert XML, Fraktur Adverts, Medieval Manuscripts) use different task types (text-to-XML, OCR transcription, page segmentation) that would require a different pipeline architecture rather than the image → structured JSON approach used here.
