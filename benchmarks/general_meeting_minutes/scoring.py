"""Scoring for the General Meeting Minutes benchmark.

Ported from upstream benchmark.py: per-leaf fuzzy similarity averaged across
all ground-truth leaf keys (same pattern as Personnel Cards + upstream
scoring_helper's get_all_keys / get_nested_value).
"""

import logging

from benchmarks.shared.scoring_helpers import (
    get_all_keys,
    get_nested_value,
    calculate_fuzzy_score,
    parse_prediction_document,
    fuzzy_dspy_metric,
    fuzzy_gepa_feedback_metric,
)

logger = logging.getLogger(__name__)

BOOTSTRAP_THRESHOLD = 0.3
REQUIRED_KEYS = {"document", "page_number", "entries", "total_actions"}


def score_single_prediction(pred_dict: dict, gt_dict: dict) -> dict:
    """Per-leaf fuzzy score, macro-averaged over ground-truth leaf keys.

    Faithful to upstream: iterate all leaf keys of the ground truth, look up
    each in the prediction, fuzzy-compare, and average. Extra keys in the
    prediction are ignored.
    """
    leaf_keys = get_all_keys(gt_dict or {})
    if not leaf_keys:
        return {"fuzzy": 0.0, "field_scores": {}, "total_keys": 0}

    total = 0.0
    field_scores: dict[str, dict] = {}
    for k in leaf_keys:
        test_value = get_nested_value(pred_dict or {}, k)
        gold_value = get_nested_value(gt_dict, k)
        score = calculate_fuzzy_score(test_value, gold_value)
        total += score
        field_scores[k] = {
            "response": test_value,
            "ground_truth": gold_value,
            "score": score,
        }

    avg = total / len(leaf_keys)
    return {
        "fuzzy": round(avg, 4),
        "field_scores": field_scores,
        "total_keys": len(leaf_keys),
    }


def compute_aggregate_scores(all_scores: list[dict]) -> dict:
    """Macro-average fuzzy across all scored pages."""
    if not all_scores:
        return {"fuzzy": 0.0, "total_instances": 0, "total_keys": 0}
    total_fuzzy = sum(s.get("fuzzy", 0.0) for s in all_scores)
    total_keys = sum(s.get("total_keys", 0) for s in all_scores)
    return {
        "fuzzy": round(total_fuzzy / len(all_scores), 4),
        "total_instances": len(all_scores),
        "total_keys": total_keys,
    }


# DSPy metric wrappers built from the shared fuzzy factories.
dspy_metric = fuzzy_dspy_metric(score_single_prediction)
gepa_feedback_metric = fuzzy_gepa_feedback_metric(score_single_prediction)


def refine_reward_fn(example, prediction, trace=None) -> float:
    """Valid-JSON + required-top-level-keys check for dspy.Refine."""
    doc = parse_prediction_document(prediction)
    if doc is None or not isinstance(doc, dict):
        return 0.0
    if not REQUIRED_KEYS.issubset(doc.keys()):
        return 0.0
    return 1.0
