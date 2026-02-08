"""Category-level set-matching scoring for the Business Letters benchmark.

Scoring is fundamentally different from Library/Personnel Cards:
- 3 scored categories: send_date, sender_persons, receiver_persons
- TP/FP/FN via set intersection/difference per category per letter
- Person matching uses persons.json alias table
- Aggregate F1: sum TP/FP/FN per category across all letters, compute
  per-category F1, then average for f1_macro.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

from benchmarks.shared.config import DATA_DIR
from benchmarks.shared.scoring_helpers import (
    compute_f1,
    parse_prediction_document,
    parse_gt_document,
    FeedbackScore,
)

logger = logging.getLogger(__name__)

PERSONS_PATH = DATA_DIR / "business_letters" / "ground_truths" / "persons.json"

# Categories that are scored (letter_title and has_signatures are NOT scored)
SCORED_CATEGORIES = ("send_date", "sender_persons", "receiver_persons")

# Keys that must be present for a valid prediction
REQUIRED_KEYS = {"send_date", "sender_persons", "receiver_persons"}


# ---------------------------------------------------------------------------
# Person alias lookup
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_persons() -> list[dict]:
    """Load the persons.json alias table (cached)."""
    try:
        with open(PERSONS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("persons.json not found at %s", PERSONS_PATH)
        return []


def _parse_gt_persons(
    person_names: list[str],
    include_inferred_function: bool = False,
    include_inferred_correspondence: bool = False,
) -> list[str]:
    """Parse GT person strings, filtering out inferred persons by default.

    Angle-bracket conventions:
      <Name> = inferred from function
      <<Name>> = inferred from correspondence
    """
    result = []
    for name in person_names:
        name = name.strip()
        if name.startswith("<<") and name.endswith(">>"):
            if include_inferred_correspondence:
                result.append(name[2:-2].strip())
        elif name.startswith("<") and name.endswith(">"):
            if include_inferred_function:
                result.append(name[1:-1].strip())
        else:
            result.append(name)
    return result


def _match_person(persons_db: list[dict], gt_person_name: str, predicted_name: str) -> str | None:
    """Check if predicted_name is an alternate name for gt_person_name in persons.json.

    Returns the GT canonical name if matched, None otherwise.
    """
    for person in persons_db:
        if person.get("name") == gt_person_name:
            if predicted_name in person.get("alternateName", []):
                return gt_person_name
    return None


# ---------------------------------------------------------------------------
# Per-letter scoring
# ---------------------------------------------------------------------------

_NULL_SENTINELS = {"null", "None", "", None}


def _score_send_date(pred_dict: dict, gt_dict: dict) -> dict[str, int]:
    """Score send_date category via set comparison."""
    gt_dates = {d for d in (gt_dict.get("send_date") or []) if d not in _NULL_SENTINELS}
    pred_dates = {d for d in (pred_dict.get("send_date") or []) if d not in _NULL_SENTINELS}

    return {
        "send_date_tp": len(gt_dates & pred_dates),
        "send_date_fp": len(pred_dates - gt_dates),
        "send_date_fn": len(gt_dates - pred_dates),
    }


def _score_persons(
    category: str,
    pred_dict: dict,
    gt_dict: dict,
    persons_db: list[dict],
) -> dict[str, int]:
    """Score sender_persons or receiver_persons via set matching with alias lookup."""
    gt_raw = gt_dict.get(category) or []
    gt_names = _parse_gt_persons(gt_raw)

    pred_raw = pred_dict.get(category) or []

    # Normalize: remove null sentinels from predictions
    pred_names = []
    for name in pred_raw:
        if isinstance(name, str) and name.strip() not in _NULL_SENTINELS:
            pred_names.append(name.strip())

    # Remove "None" entries from GT
    gt_names = [n for n in gt_names if n not in _NULL_SENTINELS]

    # Match predicted persons to canonical GT names via alias table
    matched_predicted = []
    for pred_name in pred_names:
        matched = False
        for gt_name in gt_names:
            # Direct match
            if gt_name == pred_name:
                matched_predicted.append(pred_name)
                matched = True
                break
            # Alias match
            canonical = _match_person(persons_db, gt_name, pred_name)
            if canonical is not None:
                matched_predicted.append(canonical)
                matched = True
                break
        if not matched:
            matched_predicted.append(pred_name)

    gt_set = set(gt_names)
    pred_set = set(matched_predicted)

    return {
        f"{category}_tp": len(gt_set & pred_set),
        f"{category}_fp": len(pred_set - gt_set),
        f"{category}_fn": len(gt_set - pred_set),
    }


def score_single_prediction(pred_dict: dict, gt_dict: dict) -> dict:
    """Score a single letter prediction against its ground truth.

    Returns dict with per-category TP/FP/FN and overall f1_score.
    """
    # Handle metadata wrapper defensively (model may wrap output)
    if "metadata" in pred_dict and isinstance(pred_dict["metadata"], dict):
        inner = pred_dict["metadata"]
        if any(k in inner for k in SCORED_CATEGORIES):
            pred_dict = inner

    persons_db = _load_persons()

    score = _score_send_date(pred_dict, gt_dict)
    score |= _score_persons("sender_persons", pred_dict, gt_dict, persons_db)
    score |= _score_persons("receiver_persons", pred_dict, gt_dict, persons_db)

    # Compute per-letter F1 from summed TP/FP/FN across all 3 categories
    total_tp = sum(score[f"{cat}_tp"] for cat in SCORED_CATEGORIES)
    total_fp = sum(score[f"{cat}_fp"] for cat in SCORED_CATEGORIES)
    total_fn = sum(score[f"{cat}_fn"] for cat in SCORED_CATEGORIES)

    precision, recall, f1 = compute_f1(total_tp, total_fp, total_fn)
    score["f1_score"] = round(f1, 4)
    score["precision"] = precision
    score["recall"] = recall

    return score


# ---------------------------------------------------------------------------
# DSPy metric wrappers
# ---------------------------------------------------------------------------


def refine_reward_fn(example, prediction, trace=None) -> float:
    """Reward function for dspy.Refine: 1.0 if output is valid JSON with all required keys."""
    doc = parse_prediction_document(prediction)
    if doc is None:
        return 0.0
    # Handle metadata wrapper
    if "metadata" in doc and isinstance(doc["metadata"], dict):
        doc = doc["metadata"]
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

    parts = []
    for cat in SCORED_CATEGORIES:
        tp = scores[f"{cat}_tp"]
        fp = scores[f"{cat}_fp"]
        fn = scores[f"{cat}_fn"]
        if fp > 0 or fn > 0:
            parts.append(f"  - {cat}: tp={tp}, fp={fp}, fn={fn}")

    feedback = f"f1={f1:.3f}. Category errors:\n" + "\n".join(parts) if parts else f"f1={f1:.3f}"
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
# Aggregate scoring — category-level F1
# ---------------------------------------------------------------------------


def compute_aggregate_scores(all_scores: list[dict]) -> dict:
    """Compute category-level F1 macro/micro across all scored letters.

    F1 macro = average of per-category F1 (each from summed TP/FP/FN).
    F1 micro = global TP/FP/FN F1 across all categories.
    """
    if not all_scores:
        return {"f1_micro": 0.0, "f1_macro": 0.0}

    # Accumulate per-category TP/FP/FN
    cat_tp = {cat: 0 for cat in SCORED_CATEGORIES}
    cat_fp = {cat: 0 for cat in SCORED_CATEGORIES}
    cat_fn = {cat: 0 for cat in SCORED_CATEGORIES}
    n_valid = 0

    for s in all_scores:
        if s is None:
            continue
        n_valid += 1
        for cat in SCORED_CATEGORIES:
            cat_tp[cat] += s.get(f"{cat}_tp", 0)
            cat_fp[cat] += s.get(f"{cat}_fp", 0)
            cat_fn[cat] += s.get(f"{cat}_fn", 0)

    # Per-category F1
    cat_f1s = []
    for cat in SCORED_CATEGORIES:
        _, _, f1 = compute_f1(cat_tp[cat], cat_fp[cat], cat_fn[cat])
        cat_f1s.append(f1)

    f1_macro = sum(cat_f1s) / len(cat_f1s) if cat_f1s else 0.0

    # Global micro F1
    total_tp = sum(cat_tp.values())
    total_fp = sum(cat_fp.values())
    total_fn = sum(cat_fn.values())
    micro_p, micro_r, f1_micro = compute_f1(total_tp, total_fp, total_fn)

    return {
        "f1_micro": round(f1_micro, 4),
        "f1_macro": round(f1_macro, 4),
        "micro_precision": round(micro_p, 4),
        "micro_recall": round(micro_r, 4),
        "total_instances": n_valid,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }
