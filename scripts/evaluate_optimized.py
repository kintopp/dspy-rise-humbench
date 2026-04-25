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
from benchmarks.shared.refine import EvalReward
from benchmarks.shared.scoring_helpers import parse_prediction_document
from benchmarks.shared.usage import aggregate_usage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate an optimized program")
    parser.add_argument("--benchmark", default="library_cards",
                        choices=["library_cards", "bibliographic_data", "personnel_cards", "business_letters", "blacklist_cards", "company_lists", "fraktur_adverts", "general_meeting_minutes", "medieval_manuscripts", "magazine_pages", "book_advert_xml"],
                        help="Benchmark name")
    parser.add_argument(
        "--program",
        type=str,
        required=True,
        help="Path to saved optimized program JSON",
    )
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Model preset or full model string")
    parser.add_argument("--module", choices=["predict", "cot"], default="predict", help="Module type: predict or cot (ChainOfThought)")
    parser.add_argument("--refine", type=int, default=0, help="Refine retries (0=disabled, e.g. 3 for N=3)")
    parser.add_argument("--output-tag", type=str, default="", help="Tag appended to output filename (e.g. model name for cross-model eval)")
    parser.add_argument("--num-threads", type=int, default=8, help="Parallel threads for non-Refine eval (ignored when --refine > 0)")
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
            f"metric={eval_reward.metric_key})"
        )

    # Load test split
    samples = data_mod.load_matched_samples()
    _, _, test_raw = data_mod.split_data(samples, seed=args.seed)
    test_examples = data_mod.samples_to_examples(test_raw)

    logger.info(f"Evaluating on {len(test_examples)} test images...")

    predictions: list = []
    if eval_reward is not None:
        # Refine needs per-image GT injection via set_gt — must run sequentially.
        for i, (example, raw) in enumerate(zip(test_examples, test_raw)):
            image_id = raw["id"]
            logger.info(f"[{i+1}/{len(test_examples)}] Processing {image_id}...")
            eval_reward.set_gt(raw["ground_truth"])
            try:
                prediction = extractor(**example.inputs())
            except Exception as e:
                logger.error(f"  Error processing {image_id}: {e}")
                prediction = None
            finally:
                eval_reward.clear_gt()
            predictions.append(prediction)
            # Log a preview score without full detail scoring
            if prediction is not None:
                pd = parse_prediction_document(prediction)
                if pd is not None:
                    preview = scoring_mod.score_single_prediction(pd, raw["ground_truth"])
                    logger.info(f"  score={preview.get('f1_score', preview.get('fuzzy', 0.0)):.4f}")
    else:
        # Parallel inference via dspy.Evaluate — order of .results matches devset.
        import dspy
        # Abort if more than 10% of inferences throw — protects against a
        # broken model or API outage silently burning the rest of the run.
        evaluator = dspy.Evaluate(
            devset=test_examples,
            metric=scoring_mod.dspy_metric,
            num_threads=args.num_threads,
            display_progress=True,
            provide_traceback=True,
            max_errors=max(10, len(test_examples) // 10),
            failure_score=0.0,
        )
        eval_result = evaluator(extractor)
        logger.info(f"dspy.Evaluate mean: {eval_result.score:.2f}  (detailed aggregate below)")
        predictions = [pred for _ex, pred, _sc in eval_result.results]

    # Score predictions for per-image detail (scoring is pure-Python, no API calls).
    all_scores = []
    per_image_results = []
    for example, raw, prediction in zip(test_examples, test_raw, predictions):
        image_id = raw["id"]
        pred_dict = parse_prediction_document(prediction) if prediction is not None else None
        if pred_dict is None:
            if prediction is not None:
                logger.warning(f"  Failed to parse prediction for {image_id}")
            score = scoring_mod.score_single_prediction({}, raw["ground_truth"])
        else:
            score = scoring_mod.score_single_prediction(pred_dict, raw["ground_truth"])
        all_scores.append(score)
        per_image_results.append({"id": image_id, **score})

    aggregate = scoring_mod.compute_aggregate_scores(all_scores)
    logger.info(f"\n=== OPTIMIZED RESULTS ({args.benchmark}) ===")
    for key, val in aggregate.items():
        logger.info(f"  {key}: {val}")

    # Measured token usage (requires dspy.configure(track_usage=True), enabled globally).
    usage = aggregate_usage(predictions)
    if usage:
        logger.info("  usage (measured):")
        for model_id, u in usage.items():
            logger.info(f"    {model_id}: calls={u['calls']} in={u['input_tokens']} out={u['output_tokens']}")

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
        "usage": usage,
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
