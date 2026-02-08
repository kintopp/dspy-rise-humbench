#!/usr/bin/env python3
"""Evaluate the unoptimized (baseline) extractor on the test set."""

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.shared.config import configure_dspy, resolve_model, results_dir
from benchmarks.shared.scoring_helpers import parse_prediction_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline")
    parser.add_argument("--benchmark", default="library_cards",
                        choices=["library_cards", "bibliographic_data", "personnel_cards", "business_letters"],
                        help="Benchmark name")
    parser.add_argument("--model", default="gpt-4o", help="Model preset or full model string")
    parser.add_argument("--module", choices=["predict", "cot"], default="predict", help="Module type: predict or cot (ChainOfThought)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Dynamic benchmark imports
    benchmark_pkg = f"benchmarks.{args.benchmark}"
    data_mod = importlib.import_module(f"{benchmark_pkg}.data")
    module_mod = importlib.import_module(f"{benchmark_pkg}.module")
    scoring_mod = importlib.import_module(f"{benchmark_pkg}.scoring")

    model_id = resolve_model(args.model)
    logger.info(f"Benchmark: {args.benchmark} | Model: {model_id}")
    configure_dspy(model=args.model)
    extractor = module_mod.Extractor(module_type=args.module)

    samples = data_mod.load_matched_samples()
    _, _, test_raw = data_mod.split_data(samples, seed=args.seed)
    test_examples = data_mod.samples_to_examples(test_raw)

    # Determine input field name from examples
    input_field = list(test_examples[0].inputs().keys())[0]

    logger.info(f"Evaluating baseline on {len(test_examples)} test images...")

    all_scores = []
    per_image_results = []

    for i, (example, raw) in enumerate(zip(test_examples, test_raw)):
        image_id = raw["id"]
        logger.info(f"[{i+1}/{len(test_examples)}] Processing {image_id}...")
        try:
            prediction = extractor(**{input_field: getattr(example, input_field)})
            pred_dict = parse_prediction_document(prediction)
            if pred_dict is None:
                logger.warning(f"  Failed to parse prediction for {image_id}")
                score = scoring_mod.score_single_prediction({}, raw["ground_truth"])
            else:
                score = scoring_mod.score_single_prediction(pred_dict, raw["ground_truth"])
        except Exception as e:
            logger.error(f"  Error processing {image_id}: {e}")
            score = scoring_mod.score_single_prediction({}, raw["ground_truth"])

        all_scores.append(score)
        per_image_results.append({"id": image_id, **score})

        # Log the primary metric
        primary = score.get("f1_score", score.get("fuzzy", 0.0))
        logger.info(f"  score={primary:.4f}")

    aggregate = scoring_mod.compute_aggregate_scores(all_scores)
    logger.info(f"\n=== BASELINE RESULTS ({args.benchmark}) ===")
    for key, val in aggregate.items():
        logger.info(f"  {key}: {val}")

    # Save results
    model_tag = args.model.replace("/", "_")
    module_tag = f"_{args.module}" if args.module != "predict" else ""
    out_dir = results_dir(args.benchmark) / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "benchmark": args.benchmark,
        "model": model_id,
        "module_type": args.module,
        "aggregate": aggregate,
        "per_image": [
            {k: v for k, v in r.items() if k != "field_scores"}
            for r in per_image_results
        ],
    }
    out_path = out_dir / f"scores{module_tag}_{model_tag}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
