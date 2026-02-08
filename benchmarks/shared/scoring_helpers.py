"""Shared scoring helpers used across benchmarks.

Ported from the RISE benchmark's scoring_helper.py. Contains generic
fuzzy matching, nested key traversal, code-fence stripping, and the
FeedbackScore class needed by GEPA.
"""

import json
import logging
from typing import Any

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
    """Remove parent keys when child keys exist (e.g. drop 'a' if 'a.b' is present)."""
    return [
        key for key in keys
        if not any(other.startswith(key + ".") for other in keys if other != key)
    ]


# ---------------------------------------------------------------------------
# FeedbackScore (for GEPA)
# ---------------------------------------------------------------------------


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
