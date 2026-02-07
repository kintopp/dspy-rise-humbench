"""Scoring: field-level fuzzy F1 metric matching the RISE benchmark exactly."""

import json
import logging
from typing import Any, Union

from pydantic import BaseModel
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers (ported from benchmark's scoring_helper.py)
# ---------------------------------------------------------------------------


def get_all_keys(obj: Any, parent_key: str = "") -> list[str]:
    """Recursively get all terminal (leaf) keys in a nested dict/list."""
    keys = []
    if isinstance(obj, BaseModel):
        obj = obj.model_dump()
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, (dict, list, BaseModel)):
                keys.extend(get_all_keys(value, full_key))
            else:
                keys.append(full_key)
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            full_key = f"{parent_key}[{index}]"
            if isinstance(item, (dict, list, BaseModel)):
                keys.extend(get_all_keys(item, full_key))
            else:
                keys.append(full_key)
    else:
        keys.append(parent_key)
    return keys


def get_nested_value(obj: Union[dict, BaseModel], path: str) -> Any:
    """Retrieve a value from a nested dict based on a dot/bracket path."""
    keys = path.replace("[", ".").replace("]", "").split(".")
    for key in keys:
        if isinstance(obj, BaseModel):
            obj = obj.model_dump()
        if isinstance(obj, dict):
            obj = obj.get(key, None)
        elif isinstance(obj, list):
            try:
                obj = obj[int(key)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if obj is None:
            return None
    return obj


def calculate_fuzzy_score(test_value: Any, gold_value: Any) -> float:
    """Fuzzy match score between two values (0.0–1.0)."""
    if test_value == gold_value:
        return 1.0
    if test_value is None or gold_value is None:
        return 0.0
    test_str = str(test_value)
    gold_str = str(gold_value)
    if test_str == gold_str:
        return 1.0
    if not isinstance(test_value, (str, int, float)) or not isinstance(gold_value, (str, int, float)):
        return 0.0
    return fuzz.ratio(test_str, gold_str) / 100.0


# ---------------------------------------------------------------------------
# Per-image scoring (reimplements LibraryCards.score_request_answer)
# ---------------------------------------------------------------------------

MATCH_THRESHOLD = 0.92
SKIP_PREFIXES = ("examination.", "examination")
SKIP_FIELDS = {"publication.reprint_note", "library_reference.publication_number"}


def score_single_prediction(pred_dict: dict, gt_dict: dict) -> dict:
    """Score a single prediction against its ground truth.

    Returns dict with f1_score, precision, recall, tp, fp, fn, field_scores.
    """
    response_keys = get_all_keys(pred_dict)
    gt_keys = get_all_keys(gt_dict)
    all_keys = set(response_keys + gt_keys)

    # Filter out examination fields and removed fields
    filtered_temp = []
    for key in all_keys:
        if any(key.startswith(p) for p in SKIP_PREFIXES):
            continue
        if key in SKIP_FIELDS:
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
        # Both None → skip

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
# DSPy metric wrapper
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from a string."""
    s = text.strip()
    if s.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = s.index("\n") if "\n" in s else len(s)
        s = s[first_newline + 1:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def _parse_prediction_document(prediction) -> dict | None:
    """Extract the document dict from a DSPy prediction."""
    doc = prediction.document
    if isinstance(doc, dict):
        return doc
    if isinstance(doc, str):
        try:
            return json.loads(doc)
        except json.JSONDecodeError:
            pass
        # Retry after stripping markdown code fences
        try:
            return json.loads(_strip_code_fences(doc))
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse prediction document as JSON")
            return None
    return None


def _parse_gt_document(example) -> dict | None:
    """Extract the ground truth dict from a dspy.Example."""
    doc = example.document
    if isinstance(doc, dict):
        return doc
    if isinstance(doc, str):
        try:
            return json.loads(doc)
        except json.JSONDecodeError:
            logger.warning("Failed to parse GT document as JSON")
            return None
    return None


REQUIRED_KEYS = {"type", "author", "publication", "library_reference"}


def refine_reward_fn(example, prediction, trace=None) -> float:
    """Reward function for dspy.Refine: 1.0 if output is valid JSON with required keys, else 0.0."""
    doc = _parse_prediction_document(prediction)
    if doc is None:
        return 0.0
    if not REQUIRED_KEYS.issubset(doc.keys()):
        return 0.0
    return 1.0


class FeedbackScore(dict):
    """A dict with score/feedback that supports arithmetic for DSPy's parallelizer.

    DSPy's Evaluate uses sum() on metric results for progress tracking.
    GEPA expects dict-like access with "score" and "feedback" keys, and
    checks hasattr(s, "score") before doing s["score"].
    This class satisfies both by being a dict that also supports + and has
    a .score attribute.
    """

    def __init__(self, score: float, feedback: str = ""):
        super().__init__(score=score, feedback=feedback)
        self.score = score
        self.feedback = feedback

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return self["score"] + other
        if isinstance(other, dict) and "score" in other:
            return self["score"] + other["score"]
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return other + self["score"]
        return NotImplemented

    def __float__(self):
        return float(self["score"])

    def __repr__(self):
        return f"FeedbackScore(score={self['score']:.4f})"


def gepa_feedback_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """GEPA-compatible metric returning score + textual feedback.

    GEPA passes 5 args: (gold, pred, trace, pred_name, pred_trace) and expects
    a dict with ``{"score": float, "feedback": str}``.  Returns a FeedbackScore
    that also supports arithmetic so DSPy's parallelizer can call sum().
    """
    pred_dict = _parse_prediction_document(pred)
    gt_dict = _parse_gt_document(gold)

    if pred_dict is None or gt_dict is None:
        return FeedbackScore(0.0, "Failed to parse JSON output")

    scores = score_single_prediction(pred_dict, gt_dict)
    f1 = scores["f1_score"]

    if f1 >= 1.0:
        return FeedbackScore(f1, "Perfect score")

    # Build field-level feedback for low-scoring fields
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
    pred_dict = _parse_prediction_document(prediction)
    gt_dict = _parse_gt_document(example)

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
