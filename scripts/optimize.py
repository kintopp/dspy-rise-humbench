#!/usr/bin/env python3
"""Run DSPy optimization (MIPROv2 or BootstrapFewShot) on the library card extraction task."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import configure_dspy, resolve_model, RESULTS_DIR
from src.data import load_and_split
from src.module import LibraryCardExtractor
from src.scoring import dspy_metric

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_mipro(trainset, devset, auto="light", max_bootstrapped=2, max_labeled=2, num_threads=8):
    """Run MIPROv2 optimization."""
    import dspy

    optimizer = dspy.MIPROv2(
        metric=dspy_metric,
        auto=auto,
        num_threads=num_threads,
        max_bootstrapped_demos=max_bootstrapped,
        max_labeled_demos=max_labeled,
    )
    extractor = LibraryCardExtractor()
    optimized = optimizer.compile(
        extractor,
        trainset=trainset,
        valset=devset,
    )
    return optimized


def run_bootstrap(trainset, max_bootstrapped=3, max_labeled=3, num_threads=8):
    """Run BootstrapFewShot optimization."""
    import dspy

    optimizer = dspy.BootstrapFewShot(
        metric=dspy_metric,
        max_bootstrapped_demos=max_bootstrapped,
        max_labeled_demos=max_labeled,
        num_threads=num_threads,
    )
    extractor = LibraryCardExtractor()
    optimized = optimizer.compile(extractor, trainset=trainset)
    return optimized


def main():
    parser = argparse.ArgumentParser(description="Optimize library card extraction with DSPy")
    parser.add_argument(
        "--optimizer",
        choices=["mipro", "bootstrap"],
        default="mipro",
        help="Optimizer to use (default: mipro)",
    )
    parser.add_argument("--auto", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--max-bootstrapped", type=int, default=2)
    parser.add_argument("--max-labeled", type=int, default=2)
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model preset or full model string")
    parser.add_argument("--num-threads", type=int, default=8, help="Number of parallel threads for evaluation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_id = resolve_model(args.model)
    logger.info(f"Using model: {model_id}")
    configure_dspy(model=args.model)

    train_ex, dev_ex, test_ex, *_ = load_and_split(seed=args.seed)
    logger.info(f"Data split: train={len(train_ex)}, dev={len(dev_ex)}, test={len(test_ex)}")

    if args.optimizer == "mipro":
        logger.info(f"Running MIPROv2 (auto={args.auto}, bootstrapped={args.max_bootstrapped}, labeled={args.max_labeled}, threads={args.num_threads})")
        optimized = run_mipro(
            train_ex, dev_ex,
            auto=args.auto,
            max_bootstrapped=args.max_bootstrapped,
            max_labeled=args.max_labeled,
            num_threads=args.num_threads,
        )
    else:
        logger.info(f"Running BootstrapFewShot (bootstrapped={args.max_bootstrapped}, labeled={args.max_labeled}, threads={args.num_threads})")
        optimized = run_bootstrap(
            train_ex,
            max_bootstrapped=args.max_bootstrapped,
            max_labeled=args.max_labeled,
            num_threads=args.num_threads,
        )

    # Save optimized program
    out_dir = RESULTS_DIR / "optimized"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = args.model.replace("/", "_")
    save_path = out_dir / f"{args.optimizer}_{model_tag}_optimized.json"
    optimized.save(str(save_path))
    logger.info(f"Optimized program saved to {save_path}")


if __name__ == "__main__":
    main()
