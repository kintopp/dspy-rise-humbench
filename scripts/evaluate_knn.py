#!/usr/bin/env python3
"""Evaluate KNN dynamic demo selection on a benchmark's test set.

Loads the full training set (for KNN index) and a MIPROv2-optimized
program, wraps with KNNDemoExtractor for two-pass inference with
nearest-neighbor demo retrieval.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate KNN demo selection")
    parser.add_argument("--benchmark", default="library_cards",
                        choices=["library_cards", "bibliographic_data", "personnel_cards", "business_letters", "blacklist_cards"],
                        help="Benchmark name")
    parser.add_argument("--program", type=str, required=True,
                        help="Path to saved MIPROv2-optimized program JSON")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--module", choices=["predict", "cot"], default="cot")
    parser.add_argument("--k", type=int, default=3, help="Number of nearest neighbors")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    benchmark_pkg = f"benchmarks.{args.benchmark}"
    data_mod = importlib.import_module(f"{benchmark_pkg}.data")
    module_mod = importlib.import_module(f"{benchmark_pkg}.module")
    scoring_mod = importlib.import_module(f"{benchmark_pkg}.scoring")
    knn_mod = importlib.import_module(f"{benchmark_pkg}.knn_module")

    model_id = resolve_model(args.model)
    logger.info(f"Benchmark: {args.benchmark} | Model: {model_id} | k={args.k}")
    configure_dspy(model=args.model)

    # Load ALL training examples (for KNN index)
    train_examples, _, test_examples, _, _, test_raw = data_mod.load_and_split(seed=args.seed)
    logger.info(f"Training set: {len(train_examples)} examples for KNN index")

    # Load the optimized base program
    base_extractor = module_mod.Extractor(module_type=args.module)
    base_extractor.load(args.program)
    logger.info(f"Loaded base program from {args.program}")

    # Wrap with KNN demo selection
    extractor = knn_mod.KNNDemoExtractor(
        base_module=base_extractor,
        trainset=train_examples,
        k=args.k,
    )

    input_field = list(test_examples[0].inputs().keys())[0]

    logger.info(f"Evaluating on {len(test_examples)} test images...")

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

        primary = score.get("f1_score", score.get("fuzzy", 0.0))
        logger.info(f"  score={primary:.4f}")

    aggregate = scoring_mod.compute_aggregate_scores(all_scores)
    logger.info(f"\n=== KNN RESULTS ({args.benchmark}, k={args.k}) ===")
    for key, val in aggregate.items():
        logger.info(f"  {key}: {val}")

    # Save results
    program_name = Path(args.program).stem
    out_dir = results_dir(args.benchmark) / "optimized"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "benchmark": args.benchmark,
        "program": args.program,
        "module_type": args.module,
        "knn_k": args.k,
        "train_size": len(train_examples),
        "aggregate": aggregate,
        "per_image": [
            {k: v for k, v in r.items() if k != "field_scores"}
            for r in per_image_results
        ],
    }
    out_path = out_dir / f"{program_name}_knn{args.k}_test_scores.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
