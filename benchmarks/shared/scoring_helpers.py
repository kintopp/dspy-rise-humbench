"""Shared scoring helpers used across benchmarks.

Ported from the RISE benchmark's scoring_helper.py. Contains generic
fuzzy matching, nested key traversal, code-fence stripping, and
metric-factory helpers (F1-based DSPy metric, GEPA feedback metric)
shared across Library Cards, Personnel Cards, and Company Lists.
"""

import json
import logging
from typing import Any

import dspy
from pydantic import BaseModel
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Key traversal helpers
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


def get_nested_value(obj: dict | BaseModel, path: str) -> Any:
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
    """Fuzzy match score between two values (0.0-1.0)."""
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
# Code fence stripping
# ---------------------------------------------------------------------------


def strip_code_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from a string."""
    s = text.strip()
    if s.startswith("```"):
        first_newline = s.index("\n") if "\n" in s else len(s)
        s = s[first_newline + 1:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


# ---------------------------------------------------------------------------
# Prediction / GT parsing
# ---------------------------------------------------------------------------


def parse_prediction_document(prediction) -> dict | None:
    """Extract the document dict from a DSPy prediction."""
    doc = prediction.document
    if isinstance(doc, dict):
        return doc
    if isinstance(doc, str):
        try:
            return json.loads(doc)
        except json.JSONDecodeError:
            pass
        try:
            return json.loads(strip_code_fences(doc))
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse prediction document as JSON")
            return None
    return None


def parse_gt_document(example) -> dict | None:
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


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def compute_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, and F1 from TP/FP/FN counts.

    Returns (precision, recall, f1).
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def filter_parent_keys(keys: list[str]) -> list[str]:
    """Remove parent keys when child keys exist.

    Checks both dot notation (drop 'a' if 'a.b' exists) and bracket
    notation (drop 'items' if 'items[0]' exists).
    """
    return [
        key for key in keys
        if not any(
            other.startswith(key + ".") or other.startswith(key + "[")
            for other in keys if other != key
        )
    ]


# ---------------------------------------------------------------------------
# F1-based DSPy metric wrappers (shared by Library Cards, Personnel Cards)
# ---------------------------------------------------------------------------


def f1_refine_reward_fn(required_keys: set):
    """Create a Refine reward function that checks for valid JSON with required keys.

    Args:
        required_keys: Set of top-level keys that must be present.

    Returns:
        A reward function compatible with dspy.Refine.
    """

    def reward_fn(example, prediction, trace=None) -> float:
        doc = parse_prediction_document(prediction)
        if doc is None:
            return 0.0
        if not required_keys.issubset(doc.keys()):
            return 0.0
        return 1.0

    return reward_fn


def _make_scalar_metric(score_fn, score_key: str):
    """Pure-float metric returning ``score_fn(pred, gt)[score_key]``.

    Pass ``metric_threshold=BOOTSTRAP_THRESHOLD`` to MIPROv2/BootstrapFewShot
    for demo filtering — DSPy itself applies the threshold.
    """

    def metric(example, prediction, trace=None) -> float:
        pred_dict = parse_prediction_document(prediction)
        gt_dict = parse_gt_document(example)
        if pred_dict is None or gt_dict is None:
            return 0.0
        return score_fn(pred_dict, gt_dict)[score_key]

    return metric


def _make_gepa_feedback_metric(
    score_fn,
    score_key: str,
    low_field_threshold: float,
    max_low_fields: int | None = None,
):
    """GEPA metric returning ``dspy.Prediction(score=, feedback=)``.

    Fields with ``field_scores[k]["score"] < low_field_threshold`` are listed
    in the feedback up to ``max_low_fields`` (unlimited if None).
    """

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        pred_dict = parse_prediction_document(pred)
        gt_dict = parse_gt_document(gold)
        if pred_dict is None or gt_dict is None:
            return dspy.Prediction(score=0.0, feedback="Failed to parse JSON output")

        scores = score_fn(pred_dict, gt_dict)
        score = scores[score_key]
        if score >= 1.0:
            return dspy.Prediction(score=score, feedback="Perfect score")

        low_fields = [
            f"  - {key}: predicted={info['response']!r}, expected={info['ground_truth']!r}, fuzzy={info['score']:.2f}"
            for key, info in scores["field_scores"].items()
            if info["score"] < low_field_threshold
        ]
        if max_low_fields is not None:
            low_fields = low_fields[:max_low_fields]

        if low_fields:
            feedback = f"{score_key}={score:.3f}. Low-scoring fields:\n" + "\n".join(low_fields)
        else:
            feedback = f"{score_key}={score:.3f}"
        return dspy.Prediction(score=score, feedback=feedback)

    return metric


def f1_dspy_metric(score_fn):
    return _make_scalar_metric(score_fn, score_key="f1_score")


def f1_gepa_feedback_metric(score_fn, match_threshold: float = 0.92):
    return _make_gepa_feedback_metric(
        score_fn, score_key="f1_score", low_field_threshold=match_threshold
    )


def fuzzy_dspy_metric(score_fn):
    return _make_scalar_metric(score_fn, score_key="fuzzy")


def fuzzy_gepa_feedback_metric(
    score_fn, low_field_threshold: float = 0.8, max_low_fields: int = 20
):
    return _make_gepa_feedback_metric(
        score_fn,
        score_key="fuzzy",
        low_field_threshold=low_field_threshold,
        max_low_fields=max_low_fields,
    )


def f1_compute_aggregate_scores(all_scores: list[dict]) -> dict:
    """Compute micro and macro F1 across all scored images.

    Expects each score dict to have f1_score, true_positives, false_positives,
    false_negatives keys (as returned by F1-based score_single_prediction).
    """
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

    micro_precision, micro_recall, f1_micro = compute_f1(total_tp, total_fp, total_fn)
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
