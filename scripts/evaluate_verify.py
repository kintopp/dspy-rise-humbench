#!/usr/bin/env python3
"""Evaluate verify-and-correct on Business Letters test set.

Loads a MIPROv2-optimized program, wraps with VerifyExtractor for
post-hoc person name correction using the alias table, and evaluates.
Supports stacking with Refine.
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
from scripts.evaluate_optimized import EvalReward

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate verify-and-correct")
    parser.add_argument("--benchmark", default="business_letters",
                        choices=["business_letters"],
                        help="Benchmark name (only business_letters supported)")
    parser.add_argument("--program", type=str, required=True,
                        help="Path to saved MIPROv2-optimized program JSON")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--module", choices=["predict", "cot"], default="cot")
    parser.add_argument("--refine", type=int, default=0,
                        help="Refine retries stacked on top of verify (0=disabled)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    benchmark_pkg = f"benchmarks.{args.benchmark}"
    data_mod = importlib.import_module(f"{benchmark_pkg}.data")
    module_mod = importlib.import_module(f"{benchmark_pkg}.module")
    scoring_mod = importlib.import_module(f"{benchmark_pkg}.scoring")
    verify_mod = importlib.import_module(f"{benchmark_pkg}.verify_module")

    model_id = resolve_model(args.model)
    logger.info(f"Benchmark: {args.benchmark} | Model: {model_id}")
    configure_dspy(model=args.model)

    # Load the optimized base program
    base_extractor = module_mod.Extractor(module_type=args.module)
    base_extractor.load(args.program)
    logger.info(f"Loaded base program from {args.program}")

    # Wrap with verify-and-correct
    extractor = verify_mod.VerifyExtractor(base_module=base_extractor)
    logger.info("Wrapped with VerifyExtractor (post-hoc person name correction)")

    # Optionally wrap with Refine on top
    eval_reward = None
    if args.refine > 0:
        import dspy
        eval_reward = EvalReward(scoring_mod)
        extractor = dspy.Refine(
            module=extractor,
            N=args.refine,
            reward_fn=eval_reward,
            threshold=0.95,
        )
        logger.info(f"Stacked Refine(N={args.refine}, threshold=0.95)")

    # Load test split
    samples = data_mod.load_matched_samples()
    _, _, test_raw = data_mod.split_data(samples, seed=args.seed)
    test_examples = data_mod.samples_to_examples(test_raw)

    input_field = list(test_examples[0].inputs().keys())[0]

    logger.info(f"Evaluating on {len(test_examples)} test images...")

    all_scores = []
    per_image_results = []

    for i, (example, raw) in enumerate(zip(test_examples, test_raw)):
        image_id = raw["id"]
        logger.info(f"[{i+1}/{len(test_examples)}] Processing {image_id}...")
        if eval_reward is not None:
            eval_reward.set_gt(raw["ground_truth"])
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
        finally:
            if eval_reward is not None:
                eval_reward.clear_gt()

        all_scores.append(score)
        per_image_results.append({"id": image_id, **score})

        primary = score.get("f1_score", score.get("fuzzy", 0.0))
        logger.info(f"  score={primary:.4f}")

    aggregate = scoring_mod.compute_aggregate_scores(all_scores)
    logger.info(f"\n=== VERIFY RESULTS ({args.benchmark}) ===")
    for key, val in aggregate.items():
        logger.info(f"  {key}: {val}")

    # Save results
    program_name = Path(args.program).stem
    refine_tag = f"_refine{args.refine}" if args.refine > 0 else ""
    out_dir = results_dir(args.benchmark) / "optimized"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "benchmark": args.benchmark,
        "program": args.program,
        "module_type": args.module,
        "verify": True,
        "refine_n": args.refine,
        "aggregate": aggregate,
        "per_image": [
            {k: v for k, v in r.items() if k != "field_scores"}
            for r in per_image_results
        ],
    }
    out_path = out_dir / f"{program_name}_verify{refine_tag}_test_scores.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
