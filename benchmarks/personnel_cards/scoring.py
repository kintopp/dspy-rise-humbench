"""F1-based scoring logic for the Personnel Cards benchmark."""

import logging

from benchmarks.shared.scoring_helpers import (
    get_all_keys,
    get_nested_value,
    calculate_fuzzy_score,
    parse_prediction_document,
    parse_gt_document,
    FeedbackScore,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-image scoring
# ---------------------------------------------------------------------------

MATCH_THRESHOLD = 0.92
def _filter_keys(all_keys: list[str]) -> list[str]:
    """Filter out row_number and parent keys that have children."""
    # Remove row_number fields (structural, not scored)
    filtered_temp = []
    for key in all_keys:
        if key.endswith(".row_number") or key == "row_number":
            continue
        filtered_temp.append(key)

    # Filter out parent keys when child keys exist
    filtered_keys = []
    for key in filtered_temp:
        has_children = any(
            other.startswith(key + ".") for other in filtered_temp if other != key
        )
        if not has_children:
            filtered_keys.append(key)

    return filtered_keys


def score_single_prediction(pred_dict: dict, gt_dict: dict) -> dict:
    """Score a single prediction against its ground truth.

    Returns dict with f1_score, precision, recall, tp, fp, fn, field_scores.
    """
    response_keys = get_all_keys(pred_dict)
    gt_keys = get_all_keys(gt_dict)
    all_keys = set(response_keys + gt_keys)

    filtered_keys = _filter_keys(list(all_keys))

    tp = fp = fn = 0
    field_scores = {}

    for key in filtered_keys:
        response_value = get_nested_value(pred_dict, key)
        gt_value = get_nested_value(gt_dict, key)

        # Convert empty strings to None for consistent comparison
        if response_value == "":
            response_value = None
        if gt_value == "":
            gt_value = None

        field_score = calculate_fuzzy_score(response_value, gt_value)
        field_scores[key] = {
            "response": response_value,
            "ground_truth": gt_value,
            "score": field_score,
        }

        if response_value is not None and gt_value is not None:
            if field_score >= MATCH_THRESHOLD:
                tp += 1
            else:
                fp += 1
                fn += 1
        elif response_value is not None and gt_value is None:
            fp += 1
        elif response_value is None and gt_value is not None:
            fn += 1
        # Both None -> skip

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "f1_score": round(f1, 4),
        "precision": precision,
        "recall": recall,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "field_scores": field_scores,
        "total_fields": len(all_keys),
    }


# ---------------------------------------------------------------------------
# DSPy metric wrappers
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"rows"}


def refine_reward_fn(example, prediction, trace=None) -> float:
    """Reward function for dspy.Refine: 1.0 if output is valid JSON with required keys, else 0.0."""
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
    f1 = scores["f1_score"]

    if f1 >= 1.0:
        return FeedbackScore(f1, "Perfect score")

    low_fields = []
    for key, info in scores["field_scores"].items():
        if info["score"] < MATCH_THRESHOLD:
            low_fields.append(
                f"  - {key}: predicted={info['response']!r}, expected={info['ground_truth']!r}, fuzzy={info['score']:.2f}"
            )

    feedback = f"f1={f1:.3f}. Low-scoring fields:\n" + "\n".join(low_fields) if low_fields else f"f1={f1:.3f}"
    return FeedbackScore(f1, feedback)


def dspy_metric(example, prediction, trace=None) -> float | bool:
    """DSPy-compatible metric.

    Returns:
        float (f1 score) when trace is None (evaluation mode)
        bool (f1 >= 0.5) when trace is set (bootstrapping mode)
    """
    pred_dict = parse_prediction_document(prediction)
    gt_dict = parse_gt_document(example)

    if pred_dict is None or gt_dict is None:
        return False if trace else 0.0

    scores = score_single_prediction(pred_dict, gt_dict)
    f1 = scores["f1_score"]

    if trace is not None:
        return f1 >= 0.5
    return f1


# ---------------------------------------------------------------------------
# Aggregate scoring
# ---------------------------------------------------------------------------


def compute_aggregate_scores(all_scores: list[dict]) -> dict:
    """Compute micro and macro F1 across all scored images."""
    if not all_scores:
        return {"f1_micro": 0.0, "f1_macro": 0.0}

    total_tp = total_fp = total_fn = 0
    f1_scores = []

    for s in all_scores:
        if isinstance(s, dict) and "f1_score" in s:
            total_tp += s["true_positives"]
            total_fp += s["false_positives"]
            total_fn += s["false_negatives"]
            f1_scores.append(s["f1_score"])

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_micro = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )
    f1_macro = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {
        "f1_micro": round(f1_micro, 4),
        "f1_macro": round(f1_macro, 4),
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "total_instances": len(f1_scores),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }
