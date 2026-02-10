#!/usr/bin/env python3
"""Generate demo data: run optimizer predictions on selected test images.

Requires the dspy environment with all benchmark dependencies.

Usage:
    python scripts/generate_demo_data.py --all
    python scripts/generate_demo_data.py --benchmark library_cards
    python scripts/generate_demo_data.py --benchmark library_cards --model gemini-2.0-flash

Output:
    offline/demo_data/{benchmark}.json  — one file per benchmark with GT,
    predictions, and scores for each selected image × optimizer.
"""

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.shared.config import configure_dspy, results_dir
from benchmarks.shared.scoring_helpers import parse_prediction_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Per-benchmark configuration
# ---------------------------------------------------------------------------

BENCHMARK_CONFIG = {
    "library_cards": {
        "display_name": "Library Cards",
        "metric_key": "f1_score",
        "reference_scores": "results/library_cards/optimized/mipro-cot_gemini-2.0-flash_optimized_test_scores.json",
        "optimizers": [
            {
                "name": "Predict Baseline",
                "module_type": "predict",
                "program": None,
                "refine": 0,
            },
            {
                "name": "MIPROv2 CoT",
                "module_type": "cot",
                "program": "results/library_cards/optimized/mipro-cot_gemini-2.0-flash_optimized.json",
                "refine": 0,
            },
            {
                "name": "MIPROv2 CoT + Refine(3)",
                "module_type": "cot",
                "program": "results/library_cards/optimized/mipro-cot_gemini-2.0-flash_optimized.json",
                "refine": 3,
            },
        ],
    },
    "bibliographic_data": {
        "display_name": "Bibliographic Data",
        "metric_key": "fuzzy",
        "note": "Only 5 images total; 3 of 5 were seen during optimization (train/dev). Scores on those images may be inflated.",
        "reference_scores": "results/bibliographic_data/optimized/mipro-heavy-cot_gemini-2.0-flash_optimized_test_scores.json",
        "optimizers": [
            {
                "name": "Predict Baseline",
                "module_type": "predict",
                "program": None,
                "refine": 0,
            },
            {
                "name": "MIPROv2 Heavy CoT",
                "module_type": "cot",
                "program": "results/bibliographic_data/optimized/mipro-heavy-cot_gemini-2.0-flash_optimized.json",
                "refine": 0,
            },
            {
                "name": "MIPROv2 Heavy CoT + Refine(3)",
                "module_type": "cot",
                "program": "results/bibliographic_data/optimized/mipro-heavy-cot_gemini-2.0-flash_optimized.json",
                "refine": 3,
            },
        ],
    },
    "personnel_cards": {
        "display_name": "Personnel Cards",
        "metric_key": "f1_score",
        "reference_scores": "results/personnel_cards/optimized/mipro-cot_gemini-2.0-flash_optimized_test_scores.json",
        "optimizers": [
            {
                "name": "Predict Baseline",
                "module_type": "predict",
                "program": None,
                "refine": 0,
            },
            {
                "name": "MIPROv2 CoT",
                "module_type": "cot",
                "program": "results/personnel_cards/optimized/mipro-cot_gemini-2.0-flash_optimized.json",
                "refine": 0,
            },
            {
                "name": "MIPROv2 CoT + Refine(3)",
                "module_type": "cot",
                "program": "results/personnel_cards/optimized/mipro-cot_gemini-2.0-flash_optimized.json",
                "refine": 3,
            },
        ],
    },
    "business_letters": {
        "display_name": "Business Letters",
        "metric_key": "f1_score",
        "reference_scores": "results/business_letters/optimized/mipro-cot_gemini-2.0-flash_optimized_test_scores.json",
        "demo_images": ["letter34", "letter53", "letter60", "letter25", "letter15"],
        "optimizers": [
            {
                "name": "Predict Baseline",
                "module_type": "predict",
                "program": None,
                "refine": 0,
            },
            {
                "name": "MIPROv2 CoT",
                "module_type": "cot",
                "program": "results/business_letters/optimized/mipro-cot_gemini-2.0-flash_optimized.json",
                "refine": 0,
            },
            {
                "name": "MIPROv2 CoT + Refine(3)",
                "module_type": "cot",
                "program": "results/business_letters/optimized/mipro-cot_gemini-2.0-flash_optimized.json",
                "refine": 3,
            },
        ],
    },
    "blacklist_cards": {
        "display_name": "Blacklist Cards",
        "metric_key": "fuzzy",
        "reference_scores": "results/blacklist_cards/optimized/mipro-cot_gemini-2.0-flash_optimized_test_scores.json",
        "optimizers": [
            {
                "name": "Predict Baseline",
                "module_type": "predict",
                "program": None,
                "refine": 0,
            },
            {
                "name": "MIPROv2 CoT",
                "module_type": "cot",
                "program": "results/blacklist_cards/optimized/mipro-cot_gemini-2.0-flash_optimized.json",
                "refine": 0,
            },
            {
                "name": "MIPROv2 CoT + Refine(3)",
                "module_type": "cot",
                "program": "results/blacklist_cards/optimized/mipro-cot_gemini-2.0-flash_optimized.json",
                "refine": 3,
            },
        ],
    },
    "company_lists": {
        "display_name": "Company Lists",
        "metric_key": "f1_score",
        "reference_scores": "results/company_lists/optimized/mipro-cot_gemini-2.0-flash_optimized_test_scores.json",
        "optimizers": [
            {
                "name": "Predict Baseline",
                "module_type": "predict",
                "program": None,
                "refine": 0,
            },
            {
                "name": "MIPROv2 CoT",
                "module_type": "cot",
                "program": "results/company_lists/optimized/mipro-cot_gemini-2.0-flash_optimized.json",
                "refine": 0,
            },
            {
                "name": "MIPROv2 CoT + Refine(3)",
                "module_type": "cot",
                "program": "results/company_lists/optimized/mipro-cot_gemini-2.0-flash_optimized.json",
                "refine": 3,
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Image selection
# ---------------------------------------------------------------------------


def select_images(benchmark: str, all_samples: list[dict], n: int = 5) -> list[str]:
    """Select n image IDs spanning the score distribution.

    Uses the reference score file to pick images at percentile intervals
    (min, 25th, 50th, 75th, max). Falls back to all images if n >= total.

    When the reference scores file has fewer images than *n* but the full
    dataset has more (e.g. bibliographic_data with 2 test images but 5 total),
    includes all dataset images sorted by any available scores.
    """
    cfg = BENCHMARK_CONFIG[benchmark]

    # Explicit override takes precedence
    if "demo_images" in cfg:
        return cfg["demo_images"][:n]

    scores_path = PROJECT_ROOT / cfg["reference_scores"]
    metric = cfg["metric_key"]

    with open(scores_path) as f:
        data = json.load(f)

    per_img = sorted(data["per_image"], key=lambda x: x.get(metric, 0))

    # If the reference scores cover enough images, use them directly
    if len(per_img) >= n:
        total = len(per_img)
        indices = [0, total // 4, total // 2, 3 * total // 4, total - 1]
        seen = set()
        selected = []
        for i in indices:
            img_id = per_img[i]["id"]
            if img_id not in seen:
                seen.add(img_id)
                selected.append(img_id)
        return selected[:n]

    # Reference scores have fewer than n images — include all dataset images.
    # Put scored images first (sorted by score), then any unscored ones.
    scored_ids = [img["id"] for img in per_img]
    all_ids = [s["id"] for s in all_samples]
    unscored = [sid for sid in all_ids if sid not in set(scored_ids)]
    return (scored_ids + sorted(unscored))[:n]


# ---------------------------------------------------------------------------
# EvalReward for Refine (reused from evaluate_optimized.py)
# ---------------------------------------------------------------------------


class EvalReward:
    """Quality-aware reward for dspy.Refine using actual benchmark metric."""

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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_benchmark(benchmark: str, model: str, n_images: int = 5):
    """Generate demo data for one benchmark."""
    import dspy

    cfg = BENCHMARK_CONFIG[benchmark]
    logger.info(f"=== {cfg['display_name']} ===")

    # Dynamic imports
    pkg = f"benchmarks.{benchmark}"
    data_mod = importlib.import_module(f"{pkg}.data")
    module_mod = importlib.import_module(f"{pkg}.module")
    scoring_mod = importlib.import_module(f"{pkg}.scoring")

    configure_dspy(model=model)

    # Load and split data
    samples = data_mod.load_matched_samples()
    _, _, test_raw = data_mod.split_data(samples, seed=42)

    # Select images — pass all samples so small benchmarks can include non-test images
    selected_ids = select_images(benchmark, samples, n=n_images)
    logger.info(f"Selected {len(selected_ids)} images: {selected_ids}")

    # Build lookup from all samples (not just test) so small benchmarks can
    # include train/dev images in the demo for visual completeness.
    all_by_id = {s["id"]: s for s in samples}
    test_ids = {s["id"] for s in test_raw}

    # Determine input field name(s) from the module's forward signature
    input_field = "card_image"
    if benchmark == "bibliographic_data":
        input_field = "page_image"
    elif benchmark == "business_letters":
        input_field = "page_images"
    elif benchmark == "company_lists":
        input_field = "page_image"  # also needs page_id, handled below

    # Process each image × optimizer
    image_results = []
    for img_id in selected_ids:
        if img_id not in all_by_id:
            logger.warning(f"  {img_id} not found in dataset, skipping")
            continue

        if img_id not in test_ids:
            logger.info(f"  {img_id} is a train/dev image (included for demo completeness)")

        raw = all_by_id[img_id]
        gt = raw["ground_truth"]

        # Resolve image path(s)
        if "image_paths" in raw:
            image_paths = raw["image_paths"]
        else:
            image_paths = [raw["image_path"]]

        logger.info(f"  Image {img_id} ({len(image_paths)} page(s))")

        optimizer_results = []
        for opt_cfg in cfg["optimizers"]:
            opt_name = opt_cfg["name"]
            logger.info(f"    Optimizer: {opt_name}")

            try:
                # Create fresh extractor
                extractor = module_mod.Extractor(module_type=opt_cfg["module_type"])

                # Load program if specified
                if opt_cfg["program"] is not None:
                    prog_path = str(PROJECT_ROOT / opt_cfg["program"])
                    extractor.load(prog_path)

                # Wrap with Refine if specified
                eval_reward = None
                if opt_cfg["refine"] > 0:
                    eval_reward = EvalReward(scoring_mod)
                    extractor = dspy.Refine(
                        module=extractor,
                        N=opt_cfg["refine"],
                        reward_fn=eval_reward,
                        threshold=0.95,
                    )

                # Build input kwargs
                input_kwargs = {}
                if input_field == "page_images":
                    input_kwargs[input_field] = [dspy.Image(p) for p in image_paths]
                else:
                    input_kwargs[input_field] = dspy.Image(image_paths[0])
                if benchmark == "company_lists":
                    input_kwargs["page_id"] = img_id

                # Set GT for Refine reward
                if eval_reward is not None:
                    eval_reward.set_gt(gt)

                # Run prediction
                prediction = extractor(**input_kwargs)
                pred_dict = parse_prediction_document(prediction)

                if eval_reward is not None:
                    eval_reward.clear_gt()

                # Score
                if pred_dict is None:
                    logger.warning(f"      Failed to parse prediction")
                    scores = scoring_mod.score_single_prediction({}, gt)
                    pred_dict = {}
                else:
                    scores = scoring_mod.score_single_prediction(pred_dict, gt)

                # Strip field_scores from saved scores (they're large and we
                # can reconstruct from pred+gt); keep them in a separate key
                field_scores = scores.pop("field_scores", None)

                primary = scores.get(cfg["metric_key"], 0.0)
                logger.info(f"      {cfg['metric_key']}={primary:.4f}")

            except Exception as e:
                logger.error(f"      Error: {e}")
                pred_dict = {}
                scores = scoring_mod.score_single_prediction({}, gt)
                scores.pop("field_scores", None)
                field_scores = None

            optimizer_results.append({
                "name": opt_name,
                "prediction": pred_dict,
                "scores": scores,
            })

        image_results.append({
            "id": img_id,
            "image_paths": [str(p) for p in image_paths],
            "ground_truth": gt,
            "optimizers": optimizer_results,
        })

    # Save output
    out_dir = PROJECT_ROOT / "offline" / "demo_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{benchmark}.json"

    output = {
        "benchmark": benchmark,
        "display_name": cfg["display_name"],
        "metric_key": cfg["metric_key"],
        "model": model,
        "images": image_results,
    }
    if cfg.get("note"):
        output["note"] = cfg["note"]

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved to {out_path}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Generate demo data for benchmark visualizations")
    parser.add_argument("--benchmark", type=str, default=None,
                        choices=list(BENCHMARK_CONFIG.keys()),
                        help="Single benchmark to process")
    parser.add_argument("--all", action="store_true", help="Process all benchmarks")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash",
                        help="Model preset or full model string")
    parser.add_argument("--n-images", type=int, default=5,
                        help="Number of test images to process per benchmark")
    args = parser.parse_args()

    if not args.all and args.benchmark is None:
        parser.error("Specify --benchmark or --all")

    benchmarks = list(BENCHMARK_CONFIG.keys()) if args.all else [args.benchmark]

    for bench in benchmarks:
        run_benchmark(bench, model=args.model, n_images=args.n_images)


if __name__ == "__main__":
    main()
