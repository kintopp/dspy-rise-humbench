"""Average fuzzy scoring logic for the Bibliographic Data benchmark."""

import logging

from benchmarks.shared.scoring_helpers import (
    get_all_keys,
    get_nested_value,
    calculate_fuzzy_score,
    f1_refine_reward_fn,
    fuzzy_dspy_metric,
    fuzzy_gepa_feedback_metric,
)
from benchmarks.bibliographic_data.data import _normalize_keys, _normalize_type_values

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-image scoring (reimplements BibliographicData.score_request_answer)
# ---------------------------------------------------------------------------


def score_single_prediction(pred_dict: dict, gt_dict: dict) -> dict:
    """Score a single prediction against its ground truth.

    Returns dict with fuzzy (average fuzzy score across non-metadata leaf keys).
    Both dicts are key-normalized (hyphens -> underscores) to handle CSL-JSON
    inconsistency in ground truths and potential model output variation.
    """
    pred_dict = _normalize_type_values(_normalize_keys(pred_dict))
    gt_dict = _normalize_type_values(_normalize_keys(gt_dict))
    gt_keys = get_all_keys(gt_dict)

    total_score = 0
    total_keys = 0
    field_scores = {}

    for k in gt_keys:
        if k.startswith("metadata"):
            continue
        test_value = get_nested_value(pred_dict, k)
        gold_value = get_nested_value(gt_dict, k)
        score = calculate_fuzzy_score(test_value, gold_value)
        field_scores[k] = {
            "response": test_value,
            "ground_truth": gold_value,
            "score": score,
        }
        total_score += score
        total_keys += 1

    avg = total_score / total_keys if total_keys > 0 else 0

    return {
        "fuzzy": round(avg, 4),
        "total_keys": total_keys,
        "field_scores": field_scores,
    }


# ---------------------------------------------------------------------------
# DSPy metric wrappers
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"entries"}
BOOTSTRAP_THRESHOLD = 0.3

refine_reward_fn = f1_refine_reward_fn(REQUIRED_KEYS)
dspy_metric = fuzzy_dspy_metric(score_single_prediction)
gepa_feedback_metric = fuzzy_gepa_feedback_metric(score_single_prediction)


# ---------------------------------------------------------------------------
# Aggregate scoring
# ---------------------------------------------------------------------------


def compute_aggregate_scores(all_scores: list[dict]) -> dict:
    """Compute average fuzzy score across all scored images."""
    if not all_scores:
        return {"fuzzy": 0.0}

    fuzzy_scores = []
    total_keys = 0

    for s in all_scores:
        if isinstance(s, dict) and "fuzzy" in s:
            fuzzy_scores.append(s["fuzzy"])
            total_keys += s.get("total_keys", 0)

    avg_fuzzy = sum(fuzzy_scores) / len(fuzzy_scores) if fuzzy_scores else 0.0

    return {
        "fuzzy": round(avg_fuzzy, 4),
        "total_instances": len(fuzzy_scores),
        "total_keys": total_keys,
    }
