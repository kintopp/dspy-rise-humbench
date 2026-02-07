#!/usr/bin/env python3
"""Compare baseline vs optimized results side-by-side for any benchmark."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.shared.config import results_dir


def load_scores(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("aggregate", data)


# Metric keys to display per benchmark type
METRIC_KEYS = {
    "library_cards": ["f1_macro", "f1_micro", "micro_precision", "micro_recall"],
    "bibliographic_data": ["fuzzy", "total_instances", "total_keys"],
}


def primary_metric_key(benchmark: str) -> str:
    """Return the primary metric name for a benchmark."""
    if benchmark == "bibliographic_data":
        return "fuzzy"
    return "f1_macro"


def print_row(label: str, scores: dict | None, metric_keys: list[str], ref: dict | None = None):
    if scores is None:
        print(f"  {label:35s}  (not available)")
        return
    parts = [f"{label:35s}"]
    for key in metric_keys:
        val = scores.get(key)
        if val is None:
            continue
        if isinstance(val, float):
            delta_str = ""
            if ref and key in ref and isinstance(ref[key], float):
                delta = val - ref[key]
                delta_str = f" ({delta:+.4f})"
            parts.append(f"{key}={val:.4f}{delta_str}")
        else:
            parts.append(f"{key}={val}")
    print("  " + "  ".join(parts))


def main():
    parser = argparse.ArgumentParser(description="Compare results")
    parser.add_argument("--benchmark", default="library_cards",
                        help="Benchmark name (e.g. library_cards, bibliographic_data)")
    args = parser.parse_args()

    benchmark = args.benchmark
    res_dir = results_dir(benchmark)
    metric_keys = METRIC_KEYS.get(benchmark, ["fuzzy"])
    primary_key = primary_metric_key(benchmark)

    title = benchmark.upper().replace("_", " ")
    print("=" * 100)
    print(f"{title} — RESULTS COMPARISON")
    print("=" * 100)

    # All baselines
    baseline_dir = res_dir / "baseline"
    baseline_files = sorted(baseline_dir.glob("scores*.json")) if baseline_dir.exists() else []

    baselines = {}
    if baseline_files:
        print("\n--- Baselines (test split) ---")
        for f in baseline_files:
            scores = load_scores(f)
            label = f.stem.replace("scores_", "baseline/").replace("scores", "baseline/default")
            baselines[label] = scores
            print_row(label, scores, metric_keys)

    # All optimized results
    optimized_dir = res_dir / "optimized"
    opt_files = sorted(optimized_dir.glob("*_test_scores.json")) if optimized_dir.exists() else []

    if opt_files:
        print("\n--- Optimized Results (test split) ---")
        first_baseline = next(iter(baselines.values()), None) if baselines else None
        for f in opt_files:
            scores = load_scores(f)
            label = f.stem.replace("_test_scores", "")
            print_row(label, scores, metric_keys, ref=first_baseline)
    else:
        print("\n(No optimized results found yet)")

    # Delta summary
    first_baseline = next(iter(baselines.values()), None) if baselines else None
    if first_baseline and opt_files:
        print(f"\n--- Deltas vs First Baseline ({primary_key}) ---")
        for f in opt_files:
            scores = load_scores(f)
            if scores is None:
                continue
            label = f.stem.replace("_test_scores", "")
            val = scores.get(primary_key, 0)
            base = first_baseline.get(primary_key, 0)
            if isinstance(val, (int, float)) and isinstance(base, (int, float)):
                delta = val - base
                print(f"  {label:35s}  {primary_key}: {delta:+.4f}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
