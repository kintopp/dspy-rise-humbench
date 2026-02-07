"""Average fuzzy scoring logic for the Bibliographic Data benchmark."""

import logging

from benchmarks.shared.scoring_helpers import (
    get_all_keys,
    get_nested_value,
    calculate_fuzzy_score,
    parse_prediction_document,
    parse_gt_document,
    FeedbackScore,
)
from benchmarks.bibliographic_data.data import _normalize_keys

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-image scoring (reimplements BibliographicData.score_request_answer)
# ---------------------------------------------------------------------------


def score_single_prediction(pred_dict: dict, gt_dict: dict) -> dict:
    """Score a single prediction against its ground truth.

    Returns dict with fuzzy (average fuzzy score across non-metadata leaf keys).
    Both dicts are key-normalized (hyphens → underscores) to handle CSL-JSON
    inconsistency in ground truths and potential model output variation.
    """
    pred_dict = _normalize_keys(pred_dict)
    gt_dict = _normalize_keys(gt_dict)
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


def refine_reward_fn(example, prediction, trace=None) -> float:
    """Reward function for dspy.Refine: 1.0 if output is valid JSON with entries, else 0.0."""
    doc = parse_prediction_document(prediction)
    if doc is None:
        return 0.0
    if not REQUIRED_KEYS.issubset(doc.keys()):
        return 0.0
    return 1.0


def gepa_feedback_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA-compatible metric returning score + textual feedback."""
    pred_dict = parse_prediction_document(pred)
    gt_dict = parse_gt_document(gold)

    if pred_dict is None or gt_dict is None:
        return FeedbackScore(0.0, "Failed to parse JSON output")

    scores = score_single_prediction(pred_dict, gt_dict)
    fuzzy = scores["fuzzy"]

    if fuzzy >= 1.0:
        return FeedbackScore(fuzzy, "Perfect score")

    low_fields = []
    for key, info in scores["field_scores"].items():
        if info["score"] < 0.8:
            low_fields.append(
                f"  - {key}: predicted={info['response']!r}, expected={info['ground_truth']!r}, fuzzy={info['score']:.2f}"
            )

    feedback = f"fuzzy={fuzzy:.3f}. Low-scoring fields:\n" + "\n".join(low_fields[:20]) if low_fields else f"fuzzy={fuzzy:.3f}"
    return FeedbackScore(fuzzy, feedback)


def dspy_metric(example, prediction, trace=None) -> float | bool:
    """DSPy-compatible metric.

    Returns:
        float (fuzzy score) when trace is None (evaluation mode)
        bool (fuzzy >= 0.3) when trace is set (bootstrapping mode)
    """
    pred_dict = parse_prediction_document(prediction)
    gt_dict = parse_gt_document(example)

    if pred_dict is None or gt_dict is None:
        return False if trace else 0.0

    scores = score_single_prediction(pred_dict, gt_dict)
    fuzzy = scores["fuzzy"]

    if trace is not None:
        return fuzzy >= 0.3
    return fuzzy


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
