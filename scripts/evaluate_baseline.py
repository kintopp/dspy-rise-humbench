#!/usr/bin/env python3
"""Evaluate the unoptimized (baseline) LibraryCardExtractor on the test set."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import configure_dspy, resolve_model, RESULTS_DIR
from src.data import load_matched_samples, split_data, samples_to_examples
from src.module import LibraryCardExtractor
from src.scoring import score_single_prediction, compute_aggregate_scores, _parse_prediction_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model preset or full model string")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_id = resolve_model(args.model)
    logger.info(f"Using model: {model_id}")
    configure_dspy(model=args.model)
    extractor = LibraryCardExtractor()

    samples = load_matched_samples()
    _, _, test_raw = split_data(samples, seed=args.seed)
    test_examples = samples_to_examples(test_raw)

    logger.info(f"Evaluating baseline on {len(test_examples)} test images...")

    all_scores = []
    per_image_results = []

    for i, (example, raw) in enumerate(zip(test_examples, test_raw)):
        image_id = raw["id"]
        logger.info(f"[{i+1}/{len(test_examples)}] Processing {image_id}...")
        try:
            prediction = extractor(card_image=example.card_image)
            pred_dict = _parse_prediction_document(prediction)
            if pred_dict is None:
                logger.warning(f"  Failed to parse prediction for {image_id}")
                score = {
                    "f1_score": 0.0, "precision": 0.0, "recall": 0.0,
                    "true_positives": 0, "false_positives": 0, "false_negatives": 0,
                    "field_scores": {}, "total_fields": 0,
                }
            else:
                score = score_single_prediction(pred_dict, raw["ground_truth"])
        except Exception as e:
            logger.error(f"  Error processing {image_id}: {e}")
            score = {
                "f1_score": 0.0, "precision": 0.0, "recall": 0.0,
                "true_positives": 0, "false_positives": 0, "false_negatives": 0,
                "field_scores": {}, "total_fields": 0,
            }

        all_scores.append(score)
        per_image_results.append({"id": image_id, **score})
        logger.info(f"  f1={score['f1_score']:.4f}  tp={score['true_positives']} fp={score['false_positives']} fn={score['false_negatives']}")

    aggregate = compute_aggregate_scores(all_scores)
    logger.info(f"\n=== BASELINE RESULTS ===")
    logger.info(f"  f1_macro: {aggregate['f1_macro']:.4f}")
    logger.info(f"  f1_micro: {aggregate['f1_micro']:.4f}")
    logger.info(f"  precision: {aggregate['micro_precision']:.4f}")
    logger.info(f"  recall: {aggregate['micro_recall']:.4f}")
    logger.info(f"  instances: {aggregate['total_instances']}")
    logger.info(f"  TP={aggregate['total_tp']} FP={aggregate['total_fp']} FN={aggregate['total_fn']}")

    # Save results — use model short name in filename
    model_tag = args.model.replace("/", "_")
    out_dir = RESULTS_DIR / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": model_id,
        "aggregate": aggregate,
        "per_image": [
            {k: v for k, v in r.items() if k != "field_scores"}
            for r in per_image_results
        ],
    }
    out_path = out_dir / f"scores_{model_tag}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
