"""Average fuzzy scoring logic for the Blacklist Cards benchmark.

Reimplements BlacklistCards.score_request_answer from the upstream benchmark.
Key difference from shared calculate_fuzzy_score: normalizes None/"null" to ""
before fuzzy comparison (upstream pattern), so both-null fields score 1.0.
"""

import logging

from benchmarks.shared.scoring_helpers import (
    get_all_keys,
    get_nested_value,
    calculate_fuzzy_score,
    f1_refine_reward_fn,
    fuzzy_dspy_metric,
    fuzzy_gepa_feedback_metric,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-image scoring (reimplements BlacklistCards.score_request_answer)
# ---------------------------------------------------------------------------


def _normalize_null(value):
    """Normalize None and "null" to empty string, matching upstream behavior."""
    if value is None or value == "null":
        return ""
    return value


def score_single_prediction(pred_dict: dict, gt_dict: dict) -> dict:
    """Score a single prediction against its ground truth.

    Returns dict with fuzzy (average fuzzy score across non-metadata leaf keys).
    Normalizes None/"null" to "" before comparison to match upstream benchmark.
    """
    gt_keys = get_all_keys(gt_dict)

    total_score = 0
    total_keys = 0
    field_scores = {}

    for k in gt_keys:
        if k.startswith("metadata"):
            continue

        test_value = _normalize_null(get_nested_value(pred_dict, k))
        gold_value = _normalize_null(get_nested_value(gt_dict, k))

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

REQUIRED_KEYS = {"company", "location", "b_id"}
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
