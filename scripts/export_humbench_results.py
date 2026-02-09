#!/usr/bin/env python3
"""Export per-letter results in the upstream HumBench results format.

Runs the optimized program on all letters and outputs one JSON per letter
plus a scoring.json, matching the format used by:
  https://github.com/RISE-UNIBAS/humanities_data_benchmark/tree/main/results/

Usage:
    python scripts/export_humbench_results.py \
        --program results/business_letters/optimized/mipro-cot_gemini-2.0-flash_optimized.json \
        --module cot --refine 3 \
        --output results/business_letters/export
"""

import argparse
import importlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.shared.config import configure_dspy, resolve_model
from benchmarks.shared.scoring_helpers import parse_prediction_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Quality-aware reward for dspy.Refine (reused from evaluate_optimized.py)
# ---------------------------------------------------------------------------


class EvalReward:
    """Reward function using actual benchmark metric when GT is available."""

    def __init__(self, scoring_mod):
        self._score_single = scoring_mod.score_single_prediction
        self._gt = None
        probe = scoring_mod.score_single_prediction({}, {})
        self._metric_key = "fuzzy" if "fuzzy" in probe else "f1_score"

    def set_gt(self, gt_dict):
        self._gt = gt_dict

    def clear_gt(self):
        self._gt = None

    def __call__(self, kwargs, outputs):
        if self._gt is None:
            raise RuntimeError("EvalReward called without GT")
        try:
            pred_dict = parse_prediction_document(outputs)
        except Exception:
            return 0.0
        if pred_dict is None:
            return 0.0
        score = self._score_single(pred_dict, self._gt)
        return score.get(self._metric_key, 0.0)


def main():
    parser = argparse.ArgumentParser(
        description="Export per-letter results in upstream HumBench format"
    )
    parser.add_argument("--program", type=str, required=True,
                        help="Path to saved optimized program JSON")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash",
                        help="Model preset or full model string")
    parser.add_argument("--module", choices=["predict", "cot"], default="cot",
                        help="Module type")
    parser.add_argument("--refine", type=int, default=0,
                        help="Refine retries (0=disabled)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for per-letter JSON files")
    args = parser.parse_args()

    import dspy

    benchmark = "business_letters"
    benchmark_pkg = f"benchmarks.{benchmark}"
    data_mod = importlib.import_module(f"{benchmark_pkg}.data")
    module_mod = importlib.import_module(f"{benchmark_pkg}.module")
    scoring_mod = importlib.import_module(f"{benchmark_pkg}.scoring")

    model_id = resolve_model(args.model)
    logger.info(f"Model: {model_id}")
    configure_dspy(model=args.model)

    # Load optimized program
    extractor = module_mod.Extractor(module_type=args.module)
    extractor.load(args.program)
    logger.info(f"Loaded program from {args.program}")

    # Optionally wrap with Refine
    eval_reward = None
    if args.refine > 0:
        eval_reward = EvalReward(scoring_mod)
        extractor = dspy.Refine(
            module=extractor,
            N=args.refine,
            reward_fn=eval_reward,
            threshold=0.95,
        )
        logger.info(f"Refine(N={args.refine}, threshold=0.95)")

    # Load ALL letters (not just test split)
    samples = data_mod.load_matched_samples()
    logger.info(f"Processing {len(samples)} letters")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_scores = []

    for i, raw in enumerate(samples):
        letter_id = raw["id"]
        image_paths = raw["image_paths"]
        gt = raw["ground_truth"]

        logger.info(f"[{i+1}/{len(samples)}] {letter_id} ({len(image_paths)} page(s))")

        if eval_reward is not None:
            eval_reward.set_gt(gt)

        t0 = time.time()
        try:
            input_val = [dspy.Image(p) for p in image_paths]
            prediction = extractor(page_images=input_val)
            pred_dict = parse_prediction_document(prediction)
            raw_text = prediction.document if hasattr(prediction, "document") else None
        except Exception as e:
            logger.error(f"  Error: {e}")
            pred_dict = None
            raw_text = None
        finally:
            if eval_reward is not None:
                eval_reward.clear_gt()

        duration = time.time() - t0

        # Score
        if pred_dict is None:
            score = scoring_mod.score_single_prediction({}, gt)
            pred_dict = {}
            if raw_text is None:
                raw_text = ""
        else:
            score = scoring_mod.score_single_prediction(pred_dict, gt)

        # Reconstruct text field: the raw JSON string from the model
        if isinstance(raw_text, str):
            text_field = raw_text
        else:
            text_field = json.dumps({"metadata": {**pred_dict, "document_number": letter_id}},
                                    ensure_ascii=False)

        # Build parsed field in their format: {"metadata": {..., "document_number": ...}}
        parsed_field = {
            "metadata": {
                **pred_dict,
                "document_number": letter_id,
            }
        }

        # Build score field (only TP/FP/FN per category, no f1/precision/recall)
        score_field = {
            k: v for k, v in score.items()
            if k.endswith("_tp") or k.endswith("_fp") or k.endswith("_fn")
        }

        # Build per-letter result in upstream format
        result = {
            "text": text_field,
            "model": model_id,
            "provider": "dspy+litellm",
            "finish_reason": "STOP",
            "duration": round(duration, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "parsed": parsed_field,
            "score": score_field,
        }

        # Save per-letter file
        out_path = out_dir / f"request_T0013_{letter_id}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        primary = score.get("f1_score", 0.0)
        logger.info(f"  f1={primary:.4f} duration={duration:.1f}s → {out_path.name}")

        all_scores.append(score)

    # Aggregate scoring
    aggregate = scoring_mod.compute_aggregate_scores(all_scores)

    scoring_out = {
        "f1_macro": round(aggregate["f1_macro"], 2),
        "f1_micro": round(aggregate["f1_micro"], 2),
    }

    scoring_path = out_dir / "scoring.json"
    with open(scoring_path, "w") as f:
        json.dump(scoring_out, f, indent=4, ensure_ascii=False)

    logger.info(f"\nAggregate: f1_macro={aggregate['f1_macro']:.4f}, f1_micro={aggregate['f1_micro']:.4f}")
    logger.info(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    main()
