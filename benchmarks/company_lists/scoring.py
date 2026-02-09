"""F1-based scoring logic for the Company Lists benchmark.

Hybrid pattern: F1 with 0.92 fuzzy threshold (like Library Cards) but with
Blacklist Cards-style null normalization (None/"null" -> "" and != "" checks).
This is needed because GT locations use the string "null" for missing values.

Reimplements CompanyLists.score_request_answer from the upstream benchmark.
"""

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
# Per-image scoring (reimplements CompanyLists.score_request_answer)
# ---------------------------------------------------------------------------

MATCH_THRESHOLD = 0.92
SKIP_PREFIXES = ("metadata",)


def _normalize_null(value):
    """Normalize None and "null" to empty string, matching upstream behavior."""
    if value is None or value == "null":
        return ""
    return value


def score_single_prediction(pred_dict: dict, gt_dict: dict) -> dict:
    """Score a single prediction against its ground truth.

    Returns dict with f1_score, precision, recall, tp, fp, fn, field_scores.
    Uses Blacklist Cards-style null normalization: None/"null" -> "".
    Uses Library Cards-style F1 thresholding at 0.92.
    """
    # Wrap bare list in entries dict (upstream compatibility)
    if isinstance(pred_dict, list):
        pred_dict = {"entries": pred_dict}

    response_keys = get_all_keys(pred_dict)
    gt_keys = get_all_keys(gt_dict)
    all_keys = set(response_keys + gt_keys)

    # Filter out metadata fields and parent keys
    filtered_temp = [
        key for key in all_keys
        if not any(key.startswith(p) for p in SKIP_PREFIXES)
    ]
    filtered_keys = filter_parent_keys(filtered_temp)

    tp = fp = fn = 0
    field_scores = {}

    for key in filtered_keys:
        response_value = _normalize_null(get_nested_value(pred_dict, key))
        gt_value = _normalize_null(get_nested_value(gt_dict, key))

        field_score = calculate_fuzzy_score(response_value, gt_value)
        field_scores[key] = {
            "response": response_value,
            "ground_truth": gt_value,
            "score": field_score,
        }

        if response_value != "" and gt_value != "":
            if field_score >= MATCH_THRESHOLD:
                tp += 1
            else:
                fp += 1
                fn += 1
        elif response_value != "" and gt_value == "":
            fp += 1
        elif response_value == "" and gt_value != "":
            fn += 1
        # Both empty -> skip (neither TP, FP, nor FN)

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

REQUIRED_KEYS = {"entries"}

refine_reward_fn = f1_refine_reward_fn(REQUIRED_KEYS)
dspy_metric = f1_dspy_metric(score_single_prediction, bootstrap_threshold=0.3)
gepa_feedback_metric = f1_gepa_feedback_metric(score_single_prediction, match_threshold=MATCH_THRESHOLD)
compute_aggregate_scores = f1_compute_aggregate_scores
