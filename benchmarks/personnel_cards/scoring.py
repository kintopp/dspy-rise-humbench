"""F1-based scoring logic for the Personnel Cards benchmark."""

import logging

from benchmarks.shared.scoring_helpers import (
    get_all_keys,
    get_nested_value,
    calculate_fuzzy_score,
    compute_f1,
    filter_parent_keys,
    f1_refine_reward_fn,
    f1_dspy_metric,
    f1_gepa_feedback_metric,
    f1_compute_aggregate_scores,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-image scoring
# ---------------------------------------------------------------------------

MATCH_THRESHOLD = 0.92


def _filter_keys(all_keys: list[str]) -> list[str]:
    """Filter out row_number and parent keys that have children."""
    without_row_number = [
        key for key in all_keys
        if key != "row_number" and not key.endswith(".row_number")
    ]
    return filter_parent_keys(without_row_number)


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

    precision, recall, f1 = compute_f1(tp, fp, fn)

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
# DSPy metric wrappers (delegated to shared helpers)
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"rows"}

refine_reward_fn = f1_refine_reward_fn(REQUIRED_KEYS)
dspy_metric = f1_dspy_metric(score_single_prediction, bootstrap_threshold=0.5)
gepa_feedback_metric = f1_gepa_feedback_metric(score_single_prediction, match_threshold=MATCH_THRESHOLD)
compute_aggregate_scores = f1_compute_aggregate_scores
