#!/usr/bin/env python3
"""Compare baseline vs optimized results side-by-side."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS_DIR


def load_scores(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("aggregate", data)


def print_row(label: str, scores: dict | None, ref: dict | None = None):
    if scores is None:
        print(f"  {label:30s}  (not available)")
        return
    parts = [f"{label:30s}"]
    for key in ["f1_macro", "f1_micro", "micro_precision", "micro_recall"]:
        val = scores.get(key, 0.0)
        delta_str = ""
        if ref and key in ref:
            delta = val - ref[key]
            delta_str = f" ({delta:+.4f})"
        parts.append(f"{key}={val:.4f}{delta_str}")
    print("  " + "  ".join(parts))


def main():
    print("=" * 100)
    print("LIBRARY CARDS — RESULTS COMPARISON")
    print("=" * 100)

    # All baselines (may have model tags: scores_gpt-4o.json, scores_gemini-2.5-pro.json, etc.)
    baseline_dir = RESULTS_DIR / "baseline"
    baseline_files = sorted(baseline_dir.glob("scores*.json")) if baseline_dir.exists() else []

    baselines = {}
    if baseline_files:
        print("\n--- DSPy Baselines (test split) ---")
        for f in baseline_files:
            scores = load_scores(f)
            label = f.stem.replace("scores_", "baseline/").replace("scores", "baseline/default")
            baselines[label] = scores
            print_row(label, scores)

    # All optimized results
    optimized_dir = RESULTS_DIR / "optimized"
    opt_files = sorted(optimized_dir.glob("*_test_scores.json")) if optimized_dir.exists() else []

    if opt_files:
        print("\n--- Optimized Results (test split) ---")
        # Use first baseline as reference for deltas
        first_baseline = next(iter(baselines.values()), None) if baselines else None
        for f in opt_files:
            scores = load_scores(f)
            label = f.stem.replace("_test_scores", "")
            print_row(label, scores, ref=first_baseline)
    else:
        print("\n(No optimized results found yet)")

    # Delta summary
    first_baseline = next(iter(baselines.values()), None) if baselines else None
    if first_baseline and opt_files:
        print("\n--- Deltas vs First Baseline ---")
        for f in opt_files:
            scores = load_scores(f)
            if scores is None:
                continue
            label = f.stem.replace("_test_scores", "")
            macro_delta = scores.get("f1_macro", 0) - first_baseline.get("f1_macro", 0)
            micro_delta = scores.get("f1_micro", 0) - first_baseline.get("f1_micro", 0)
            print(f"  {label:30s}  f1_macro: {macro_delta:+.4f}  f1_micro: {micro_delta:+.4f}")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
