#!/usr/bin/env python3
"""Estimate Google Gemini API costs for the DSPy RISE HumBench project.

Scans the results/ directory to identify experiments, estimates API calls
and token usage per experiment, then calculates costs under both
Google AI Studio and Vertex AI pricing.

Usage:
    python dspy-costs/estimate_costs.py [--retry-multiplier 1.3]
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# Pricing tables (per 1M tokens)
# Source: https://ai.google.dev/gemini-api/docs/pricing  (AI Studio, Feb 2026)
# Source: https://cloud.google.com/vertex-ai/generative-ai/pricing (Vertex AI, Feb 2026)
# ---------------------------------------------------------------------------

PRICING = {
    "ai_studio": {
        "gemini-2.0-flash":      {"input": 0.10,  "output": 0.40},
        "gemini-2.5-pro":        {"input": 1.25,  "output": 10.00},
        "gemini-3-pro-preview":  {"input": 2.00,  "output": 12.00},
    },
    "vertex_ai": {
        "gemini-2.0-flash":      {"input": 0.15,  "output": 0.60},
        "gemini-2.5-pro":        {"input": 1.25,  "output": 10.00},
        "gemini-3-pro-preview":  {"input": 2.00,  "output": 12.00},
    },
}

# ---------------------------------------------------------------------------
# Benchmark metadata
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "library_cards": {
        "total_images": 263,
        "train": 15,
        "dev": 15,
        "test": 185,
        "avg_input_tokens_baseline": 500,     # prompt + image (~258 img tokens)
        "avg_input_tokens_optimized": 3200,   # instructions + 2 demos w/ images + query image
        "avg_output_tokens": 800,             # single card JSON
        "image_field": "card_image",
    },
    "bibliographic_data": {
        "total_images": 5,
        "train": 2,
        "dev": 1,
        "test": 2,
        "avg_input_tokens_baseline": 700,     # prompt + larger page image
        "avg_input_tokens_optimized": 3500,   # instructions + demos + query
        "avg_output_tokens": 3000,            # multi-entry JSON (14-20 entries)
        "image_field": "page_image",
    },
    "personnel_cards": {
        "total_images": 61,
        "train": 9,
        "dev": 9,
        "test": 43,
        "avg_input_tokens_baseline": 600,     # prompt + card image (~258 img tokens)
        "avg_input_tokens_optimized": 4000,   # instructions + 2 demos w/ images + query image (larger JSON schema)
        "avg_output_tokens": 2000,            # multi-row table JSON (6 cols × ~5-10 rows × 3 sub-fields)
        "image_field": "card_image",
    },
    "business_letters": {
        "total_images": 57,
        "train": 8,
        "dev": 8,
        "test": 41,
        "avg_input_tokens_baseline": 900,     # prompt + multi-page images (~1.7 pages avg)
        "avg_input_tokens_optimized": 4500,   # instructions + 2 demos w/ multi-page images + query
        "avg_output_tokens": 400,             # flat JSON (persons, orgs, dates — small output)
        "image_field": "page_images",
    },
}


@dataclass
class Experiment:
    """A single experiment's estimated usage."""
    name: str
    benchmark: str
    model: str
    calls: int
    input_tokens: int
    output_tokens: int
    notes: str = ""


def detect_model(filename: str) -> str:
    """Extract the Gemini model from a result filename."""
    if "gemini-3-pro-preview" in filename:
        return "gemini-3-pro-preview"
    if "gemini-2.5-pro" in filename:
        return "gemini-2.5-pro"
    if "gemini-2.0-flash" in filename:
        return "gemini-2.0-flash"
    return "unknown"


def detect_optimizer(filename: str) -> str:
    """Extract the optimizer type from a result filename."""
    base = filename.replace("_test_scores", "").replace("_optimized", "")
    if base.startswith("scores_cot_"):
        return "baseline-cot"
    if base.startswith("scores_"):
        return "baseline-predict"
    if "loo-mipro" in base:
        return "loo-mipro"
    if "mipro-heavy" in base:
        return "mipro-heavy"
    if "mipro-cot" in base or "mipro_" in base:
        return "mipro-medium"
    if "simba" in base:
        return "simba"
    if "gepa" in base:
        return "gepa"
    return "unknown"


def is_refine(filename: str) -> bool:
    return "refine" in filename.lower()


def count_test_samples(filepath: Path) -> int | None:
    """Read a test_scores JSON and count samples."""
    try:
        data = json.loads(filepath.read_text())
        for key in ("per_image", "scores", "results"):
            if key in data and isinstance(data[key], (list, dict)):
                return len(data[key])
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Estimation logic per optimizer type
# ---------------------------------------------------------------------------

# MIPROv2 search budgets (trials) by preset
MIPRO_TRIALS = {"mipro-medium": 18, "mipro-heavy": 27}

# Overhead calls for instruction generation + bootstrapping
MIPRO_OVERHEAD_CALLS = 50


def estimate_optimization_calls(optimizer: str, bm: dict) -> int:
    """Estimate calls used during the optimization phase."""
    if optimizer in ("baseline-predict", "baseline-cot"):
        return bm["total_images"]  # single pass over full dataset

    if optimizer.startswith("mipro"):
        trials = MIPRO_TRIALS.get(optimizer, 12)
        # trials × dev evaluations + bootstrapping + instruction proposals
        return trials * bm["dev"] + MIPRO_OVERHEAD_CALLS

    if optimizer == "loo-mipro":
        # Leave-one-out: 5 folds, each a MIPROv2 medium optimization
        trials = MIPRO_TRIALS["mipro-medium"]
        folds = bm["total_images"]
        per_fold_dev = 1  # LOO uses 1 dev image per fold
        return folds * (trials * per_fold_dev + MIPRO_OVERHEAD_CALLS)

    if optimizer == "simba":
        # SIMBA: mini-batch training with reflection rounds
        return max(bm["train"] * 10, 100)

    if optimizer == "gepa":
        # GEPA: feedback loop optimization
        return max(bm["train"] * 10, 100)

    return 50  # fallback


def estimate_test_calls(filepath: Path, bm: dict, is_refine_run: bool) -> int:
    """Estimate calls for a test evaluation run."""
    n = count_test_samples(filepath)
    if n is None:
        n = bm["test"]
    multiplier = 3 if is_refine_run else 1  # Refine retries up to 3×
    return n * multiplier


def build_experiments() -> list[Experiment]:
    """Scan results/ and build a list of experiments with estimated usage."""
    experiments = []

    for benchmark_name, bm in BENCHMARKS.items():
        benchmark_dir = RESULTS_DIR / benchmark_name
        if not benchmark_dir.exists():
            continue

        # Track which optimizers we've already counted optimization costs for
        counted_optimizations: set[tuple[str, str]] = set()

        # --- Scan baseline results ---
        baseline_dir = benchmark_dir / "baseline"
        if baseline_dir.exists():
            for f in sorted(baseline_dir.glob("*.json")):
                model = detect_model(f.name)
                if model == "unknown":
                    continue
                optimizer = detect_optimizer(f.name)
                calls = bm["total_images"]
                input_tok = calls * bm["avg_input_tokens_baseline"]
                output_tok = calls * bm["avg_output_tokens"]

                experiments.append(Experiment(
                    name=f"Baseline {optimizer.replace('baseline-', '')}",
                    benchmark=benchmark_name,
                    model=model,
                    calls=calls,
                    input_tokens=input_tok,
                    output_tokens=output_tok,
                    notes=f.name,
                ))

        # --- Scan optimized results ---
        opt_dir = benchmark_dir / "optimized"
        if not opt_dir.exists():
            continue

        for f in sorted(opt_dir.glob("*.json")):
            model = detect_model(f.name)
            if model == "unknown":
                continue
            optimizer = detect_optimizer(f.name)
            refine = is_refine(f.name)

            is_test_file = "_test_scores" in f.name

            if is_test_file:
                # Test evaluation
                calls = estimate_test_calls(f, bm, refine)
                input_tok = calls * bm["avg_input_tokens_optimized"]
                output_tok = calls * bm["avg_output_tokens"]
                label = f"Test eval: {optimizer}"
                if refine:
                    label += " +Refine(3)"

                experiments.append(Experiment(
                    name=label,
                    benchmark=benchmark_name,
                    model=model,
                    calls=calls,
                    input_tokens=input_tok,
                    output_tokens=output_tok,
                    notes=f.name,
                ))
            else:
                # Optimization run (only count once per optimizer+model)
                key = (optimizer, model)
                if key not in counted_optimizations:
                    counted_optimizations.add(key)
                    calls = estimate_optimization_calls(optimizer, bm)
                    # Optimization calls use a mix of baseline and optimized
                    # token counts (early trials are baseline-like, later have demos)
                    avg_input = (bm["avg_input_tokens_baseline"] + bm["avg_input_tokens_optimized"]) // 2
                    input_tok = calls * avg_input
                    output_tok = calls * bm["avg_output_tokens"]

                    experiments.append(Experiment(
                        name=f"Optimize: {optimizer}",
                        benchmark=benchmark_name,
                        model=model,
                        calls=calls,
                        input_tokens=input_tok,
                        output_tokens=output_tok,
                        notes=f.name,
                    ))

    return experiments


def compute_cost(input_tokens: int, output_tokens: int,
                 model: str, provider: str) -> float:
    """Compute dollar cost for a given token count and pricing tier."""
    rates = PRICING.get(provider, {}).get(model)
    if not rates:
        return 0.0
    return (input_tokens / 1_000_000) * rates["input"] + \
           (output_tokens / 1_000_000) * rates["output"]


def print_report(experiments: list[Experiment], retry_multiplier: float):
    """Print a formatted cost report."""

    # --- Per-experiment table ---
    print("=" * 90)
    print("DSPy RISE HumBench — Gemini API Cost Estimate")
    print("=" * 90)
    print()

    # Group by benchmark
    for benchmark_name in BENCHMARKS:
        bm_exps = [e for e in experiments if e.benchmark == benchmark_name]
        if not bm_exps:
            continue

        print(f"  {benchmark_name}")
        print(f"  {'─' * 86}")
        print(f"  {'Experiment':<40} {'Model':<22} {'Calls':>6} {'In(K)':>8} {'Out(K)':>8}")
        print(f"  {'─' * 86}")

        for e in bm_exps:
            print(f"  {e.name:<40} {e.model:<22} {e.calls:>6} "
                  f"{e.input_tokens/1000:>7.0f} {e.output_tokens/1000:>7.0f}")

        sub_calls = sum(e.calls for e in bm_exps)
        sub_in = sum(e.input_tokens for e in bm_exps)
        sub_out = sum(e.output_tokens for e in bm_exps)
        print(f"  {'─' * 86}")
        print(f"  {'SUBTOTAL':<40} {'':<22} {sub_calls:>6} "
              f"{sub_in/1000:>7.0f} {sub_out/1000:>7.0f}")
        print()

    # --- Totals by model ---
    totals: dict[str, dict] = {}
    for e in experiments:
        if e.model not in totals:
            totals[e.model] = {"calls": 0, "input": 0, "output": 0}
        totals[e.model]["calls"] += e.calls
        totals[e.model]["input"] += e.input_tokens
        totals[e.model]["output"] += e.output_tokens

    print("=" * 90)
    print("TOTALS BY MODEL")
    print("=" * 90)
    print(f"  {'Model':<24} {'Calls':>8} {'Input Tokens':>14} {'Output Tokens':>14}")
    print(f"  {'─' * 62}")
    for model, t in sorted(totals.items()):
        print(f"  {model:<24} {t['calls']:>8,} {t['input']:>14,} {t['output']:>14,}")

    grand_calls = sum(t["calls"] for t in totals.values())
    grand_in = sum(t["input"] for t in totals.values())
    grand_out = sum(t["output"] for t in totals.values())
    print(f"  {'─' * 62}")
    print(f"  {'TOTAL':<24} {grand_calls:>8,} {grand_in:>14,} {grand_out:>14,}")
    print()

    # --- Cost comparison ---
    print("=" * 90)
    print(f"COST COMPARISON  (retry multiplier: {retry_multiplier}x)")
    print("=" * 90)

    for provider, label in [("ai_studio", "Google AI Studio"), ("vertex_ai", "Vertex AI")]:
        print(f"\n  {label}")
        print(f"  {'─' * 72}")
        print(f"  {'Model':<24} {'Base Cost':>12} {'× {:.1f}'.format(retry_multiplier):>10} {'Notes'}")
        print(f"  {'─' * 72}")

        provider_total = 0.0
        for model, t in sorted(totals.items()):
            base = compute_cost(t["input"], t["output"], model, provider)
            adjusted = base * retry_multiplier
            provider_total += adjusted

            rates = PRICING[provider].get(model, {})
            note = f"${rates.get('input', '?')}/{rates.get('output', '?')} per 1M tok"
            print(f"  {model:<24} ${base:>10.2f} ${adjusted:>8.2f}   {note}")

        print(f"  {'─' * 72}")
        base_total = provider_total / retry_multiplier
        print(f"  {'TOTAL':<24} ${base_total:>10.2f} ${provider_total:>8.2f}")

    # --- Free tier note ---
    print()
    print("─" * 90)
    print("NOTE: Gemini 2.0 Flash has a FREE tier on AI Studio (1,500 RPD, 1M TPM).")
    flash_total = totals.get("gemini-2.0-flash", {})
    if flash_total:
        flash_calls = flash_total["calls"]
        flash_cost_studio = compute_cost(
            flash_total["input"], flash_total["output"],
            "gemini-2.0-flash", "ai_studio"
        ) * retry_multiplier
        print(f"If Flash was free, subtract ${flash_cost_studio:.2f} from AI Studio total.")
    print("─" * 90)


def save_results(experiments: list[Experiment], retry_multiplier: float,
                 output_path: Path):
    """Save structured results to JSON."""
    totals: dict[str, dict] = {}
    for e in experiments:
        if e.model not in totals:
            totals[e.model] = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        totals[e.model]["calls"] += e.calls
        totals[e.model]["input_tokens"] += e.input_tokens
        totals[e.model]["output_tokens"] += e.output_tokens

    costs = {}
    for provider in PRICING:
        costs[provider] = {}
        for model, t in totals.items():
            base = compute_cost(t["input_tokens"], t["output_tokens"], model, provider)
            costs[provider][model] = {
                "base_cost": round(base, 4),
                "adjusted_cost": round(base * retry_multiplier, 4),
            }
        costs[provider]["total"] = round(
            sum(v["adjusted_cost"] for v in costs[provider].values()
                if isinstance(v, dict) and "adjusted_cost" in v), 2
        )

    result = {
        "generated_by": "dspy-costs/estimate_costs.py",
        "retry_multiplier": retry_multiplier,
        "pricing_sources": {
            "ai_studio": "https://ai.google.dev/gemini-api/docs/pricing",
            "vertex_ai": "https://cloud.google.com/vertex-ai/generative-ai/pricing",
        },
        "totals_by_model": totals,
        "experiments": [
            {
                "name": e.name,
                "benchmark": e.benchmark,
                "model": e.model,
                "calls": e.calls,
                "input_tokens": e.input_tokens,
                "output_tokens": e.output_tokens,
                "notes": e.notes,
            }
            for e in experiments
        ],
        "costs": costs,
    }

    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retry-multiplier", type=float, default=1.3,
        help="Multiplier for retries, debugging, rate-limit retries (default: 1.3)",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Path to save JSON results (default: dspy-costs/results.json)",
    )
    args = parser.parse_args()

    save_path = Path(args.save) if args.save else Path(__file__).parent / "results.json"

    experiments = build_experiments()
    if not experiments:
        print("No experiments found in results/ directory.")
        return

    print_report(experiments, args.retry_multiplier)
    save_results(experiments, args.retry_multiplier, save_path)


if __name__ == "__main__":
    main()
