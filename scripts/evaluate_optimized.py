#!/usr/bin/env python3
"""Evaluate a saved optimized program on the test set."""

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


# ---------------------------------------------------------------------------
# Quality-aware reward for dspy.Refine
# ---------------------------------------------------------------------------


class EvalReward:
    """Reward function that uses the actual benchmark metric when GT is available.

    DSPy's Refine calls ``reward_fn(kwargs, outputs)`` where *kwargs* is the
    input dict and *outputs* is a Prediction object.  At evaluation time we
    know the ground truth for each image, so we inject it via ``set_gt()``
    before calling the extractor and ``clear_gt()`` afterwards.

    When GT is not set, falls back to the benchmark's binary
    ``refine_reward_fn`` (valid-JSON check).
    """

    def __init__(self, scoring_mod):
        self._score_single = scoring_mod.score_single_prediction
        self._gt = None
        # Auto-detect primary metric key: bibliographic_data → "fuzzy", others → "f1_score"
        probe = scoring_mod.score_single_prediction({}, {})
        self._metric_key = "fuzzy" if "fuzzy" in probe else "f1_score"

    def set_gt(self, gt_dict):
        self._gt = gt_dict

    def clear_gt(self):
        self._gt = None

    def __call__(self, kwargs, outputs):
        if self._gt is None:
            raise RuntimeError("EvalReward called without GT; call set_gt() first")
        try:
            pred_dict = parse_prediction_document(outputs)
        except Exception:
            return 0.0
        if pred_dict is None:
            return 0.0
        score = self._score_single(pred_dict, self._gt)
        return score.get(self._metric_key, 0.0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate an optimized program")
    parser.add_argument("--benchmark", default="library_cards",
                        choices=["library_cards", "bibliographic_data", "personnel_cards", "business_letters", "blacklist_cards", "company_lists"],
                        help="Benchmark name")
    parser.add_argument(
        "--program",
        type=str,
        required=True,
        help="Path to saved optimized program JSON",
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model preset or full model string")
    parser.add_argument("--module", choices=["predict", "cot"], default="predict", help="Module type: predict or cot (ChainOfThought)")
    parser.add_argument("--refine", type=int, default=0, help="Refine retries (0=disabled, e.g. 3 for N=3)")
    parser.add_argument("--output-tag", type=str, default="", help="Tag appended to output filename (e.g. model name for cross-model eval)")
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

    # Load the optimized program
    extractor = module_mod.Extractor(module_type=args.module)
    extractor.load(args.program)
    logger.info(f"Loaded optimized program from {args.program}")

    # Optionally wrap with Refine for inference-time retries
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
        logger.info(
            f"Wrapped extractor with Refine(N={args.refine}, threshold=0.95, "
            f"metric={eval_reward._metric_key})"
        )

    # Load test split
    samples = data_mod.load_matched_samples()
    _, _, test_raw = data_mod.split_data(samples, seed=args.seed)
    test_examples = data_mod.samples_to_examples(test_raw)

    # Determine input field names from examples
    input_keys = list(test_examples[0].inputs().keys())

    logger.info(f"Evaluating on {len(test_examples)} test images...")

    all_scores = []
    per_image_results = []

    for i, (example, raw) in enumerate(zip(test_examples, test_raw)):
        image_id = raw["id"]
        logger.info(f"[{i+1}/{len(test_examples)}] Processing {image_id}...")
        if eval_reward is not None:
            eval_reward.set_gt(raw["ground_truth"])
        try:
            prediction = extractor(**{k: getattr(example, k) for k in input_keys})
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
    logger.info(f"\n=== OPTIMIZED RESULTS ({args.benchmark}) ===")
    for key, val in aggregate.items():
        logger.info(f"  {key}: {val}")

    # Determine output name from program path
    program_name = Path(args.program).stem
    refine_tag = f"_refine{args.refine}" if args.refine > 0 else ""
    out_dir = results_dir(args.benchmark) / "optimized"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "benchmark": args.benchmark,
        "program": args.program,
        "model": model_id,
        "module_type": args.module,
        "refine_n": args.refine,
        "refine_reward": "quality" if eval_reward is not None else None,
        "aggregate": aggregate,
        "per_image": [
            {k: v for k, v in r.items() if k != "field_scores"}
            for r in per_image_results
        ],
    }
    output_tag = f"_{args.output_tag}" if args.output_tag else ""
    out_path = out_dir / f"{program_name}{refine_tag}{output_tag}_test_scores.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
