#!/usr/bin/env python3
"""Leave-one-out MIPROv2 optimization for small-dataset benchmarks.

Runs N folds (one per image), each time holding out 1 image for testing,
using 1 as dev, and the rest as training data. Aggregates scores across
all held-out predictions for a fair evaluation.
"""

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.shared.config import configure_dspy, resolve_model, results_dir
from benchmarks.shared.scoring_helpers import parse_prediction_document
from scripts.optimize import run_mipro

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LOO MIPROv2 optimization")
    parser.add_argument("--benchmark", default="bibliographic_data",
                        choices=["bibliographic_data"],
                        help="Benchmark name (only bibliographic_data supports LOO folds)")
    parser.add_argument("--model", default="gemini-2.0-flash")
    parser.add_argument("--module", default="cot")
    parser.add_argument("--auto", default="medium", choices=["light", "medium", "heavy"])
    parser.add_argument("--max-bootstrapped", type=int, default=1)
    parser.add_argument("--max-labeled", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()

    benchmark_pkg = f"benchmarks.{args.benchmark}"
    data_mod = importlib.import_module(f"{benchmark_pkg}.data")
    module_mod = importlib.import_module(f"{benchmark_pkg}.module")
    scoring_mod = importlib.import_module(f"{benchmark_pkg}.scoring")

    configure_dspy(model=args.model)
    model_tag = args.model.replace("/", "_")

    if not hasattr(data_mod, "load_loo_folds"):
        parser.error(f"benchmark '{args.benchmark}' does not support LOO folds (no load_loo_folds function)")
    folds = data_mod.load_loo_folds()
    out_dir = results_dir(args.benchmark) / "optimized"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_scores = []
    per_image_results = []

    for fold_idx, (train_raw, dev_raw, test_raw) in enumerate(folds):
        test_id = test_raw[0]["id"]
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_idx+1}/{len(folds)}: hold-out={test_id}")
        logger.info(f"  train: {[s['id'] for s in train_raw]}")
        logger.info(f"  dev:   {[s['id'] for s in dev_raw]}")
        logger.info(f"{'='*60}")

        train_ex = data_mod.samples_to_examples(train_raw)
        dev_ex = data_mod.samples_to_examples(dev_raw)
        test_ex = data_mod.samples_to_examples(test_raw)

        # Optimize on this fold
        optimized = run_mipro(
            train_ex, dev_ex, scoring_mod.dspy_metric, module_mod.Extractor,
            auto=args.auto,
            max_bootstrapped=args.max_bootstrapped,
            max_labeled=args.max_labeled,
            num_threads=args.num_threads,
            module_type=args.module,
        )

        # Save fold program
        fold_path = out_dir / f"loo-mipro-{args.auto}-{args.module}_{model_tag}_fold{fold_idx}_{test_id}.json"
        optimized.save(str(fold_path))
        logger.info(f"  Saved fold program to {fold_path}")

        # Evaluate on held-out image
        input_field = list(test_ex[0].inputs().keys())[0]
        try:
            prediction = optimized(**{input_field: getattr(test_ex[0], input_field)})
            pred_dict = parse_prediction_document(prediction)
            if pred_dict is None:
                logger.warning(f"  Failed to parse prediction for {test_id}")
                score = scoring_mod.score_single_prediction({}, test_raw[0]["ground_truth"])
            else:
                score = scoring_mod.score_single_prediction(pred_dict, test_raw[0]["ground_truth"])
        except Exception as e:
            logger.error(f"  Error evaluating {test_id}: {e}")
            score = scoring_mod.score_single_prediction({}, test_raw[0]["ground_truth"])

        all_scores.append(score)
        per_image_results.append({"id": test_id, "fold": fold_idx, **score})
        primary = score.get("fuzzy", score.get("f1_score", 0.0))
        logger.info(f"  FOLD {fold_idx+1} result: {test_id} = {primary:.4f}")

    # Aggregate across all folds
    aggregate = scoring_mod.compute_aggregate_scores(all_scores)
    logger.info(f"\n{'='*60}")
    logger.info("LOO AGGREGATE RESULTS")
    for k, v in aggregate.items():
        logger.info(f"  {k}: {v}")

    summary = {
        "benchmark": args.benchmark,
        "method": f"loo-mipro-{args.auto}",
        "module_type": args.module,
        "model": resolve_model(args.model),
        "num_folds": len(folds),
        "aggregate": aggregate,
        "per_image": per_image_results,
    }
    summary_path = out_dir / f"loo-mipro-{args.auto}-{args.module}_{model_tag}_test_scores.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
