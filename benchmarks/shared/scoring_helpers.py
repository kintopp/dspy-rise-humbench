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
    """Extract the document dict from a DSPy prediction.

    Returns None if the prediction lacks a ``document`` field (e.g. when a
    parallel-evaluator error produced a stub Prediction) — callers must
    handle None to score these as failures without crashing.
    """
    doc = getattr(prediction, "document", None)
    if doc is None:
        return None
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


# ---------------------------------------------------------------------------
# CER-based DSPy metric wrappers
#
# Used by transcription benchmarks where the upstream RISE metric is Character
# Error Rate (CER, lower is better) — fraktur_adverts, medieval_manuscripts.
# score_fn must return a dict with a "cer" key in [0, 1] (higher CER = worse)
# and a "field_scores" dict mapping leaf keys to {"response", "ground_truth",
# "score"} where "score" is the per-field fuzzy similarity (NOT CER).
#
# The DSPy metric converts CER to a similarity score via `1 - cer` so that
# higher is better (DSPy maximises metrics).
# ---------------------------------------------------------------------------


def _cer_to_similarity(cer: float) -> float:
    """Convert CER (0=perfect, higher=worse) to a 0..1 similarity (1=perfect)."""
    return max(0.0, min(1.0, 1.0 - cer))


def cer_dspy_metric(score_fn):
    """DSPy metric that maximises ``1 - cer`` as reported by ``score_fn``."""

    def metric(example, prediction, trace=None) -> float:
        pred_dict = parse_prediction_document(prediction)
        gt_dict = parse_gt_document(example)
        if pred_dict is None or gt_dict is None:
            return 0.0
        return _cer_to_similarity(score_fn(pred_dict, gt_dict)["cer"])

    return metric


def cer_gepa_feedback_metric(
    score_fn, low_field_threshold: float = 0.8, max_low_fields: int = 20
):
    """GEPA feedback metric for CER-primary benchmarks.

    Score is ``1 - cer`` (so GEPA maximises toward 1.0). Feedback enumerates
    fields whose per-field fuzzy similarity falls below ``low_field_threshold``,
    so the reflection LM sees *what* went wrong character-by-character even
    though the aggregate metric is CER.
    """

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        pred_dict = parse_prediction_document(pred)
        gt_dict = parse_gt_document(gold)
        if pred_dict is None or gt_dict is None:
            return dspy.Prediction(score=0.0, feedback="Failed to parse JSON output")

        scores = score_fn(pred_dict, gt_dict)
        cer = scores["cer"]
        similarity = _cer_to_similarity(cer)
        if cer <= 0.0:
            return dspy.Prediction(score=similarity, feedback="Perfect CER (0.0)")

        low_fields = [
            f"  - {key}: predicted={info['response']!r}, expected={info['ground_truth']!r}, fuzzy={info['score']:.2f}"
            for key, info in scores.get("field_scores", {}).items()
            if info["score"] < low_field_threshold
        ]
        if max_low_fields is not None:
            low_fields = low_fields[:max_low_fields]

        if low_fields:
            feedback = f"cer={cer:.3f} (similarity={similarity:.3f}). Low-scoring fields:\n" + "\n".join(low_fields)
        else:
            feedback = f"cer={cer:.3f} (similarity={similarity:.3f})"
        return dspy.Prediction(score=similarity, feedback=feedback)

    return metric


def cer_compute_aggregate_scores(all_scores: list[dict]) -> dict:
    """Macro-average CER and derived similarity across all scored images."""
    if not all_scores:
        return {"cer": 0.0, "similarity": 0.0, "total_instances": 0}

    cers = [s["cer"] for s in all_scores if isinstance(s, dict) and "cer" in s]
    if not cers:
        return {"cer": 0.0, "similarity": 0.0, "total_instances": 0}

    macro_cer = sum(cers) / len(cers)
    return {
        "cer": round(macro_cer, 4),
        "similarity": round(_cer_to_similarity(macro_cer), 4),
        "total_instances": len(cers),
    }


# ---------------------------------------------------------------------------
# IoU-based spatial metric (bounding-box F1)
#
# Used by benchmarks that predict spatial regions rather than text — currently
# magazine_pages (advertisement detection with PASCAL-VOC-style IoU=0.5 greedy
# matching). Factories here let a benchmark's score_single_prediction wrap an
# IoU computation without re-implementing the aggregation logic.
# ---------------------------------------------------------------------------


def box_iou(a, b) -> float:
    """Intersection-over-union of two (x0, y0, x1, y1) boxes. 0..1.

    Defensively returns 0.0 if either box is malformed (not 4 finite numbers)
    — models occasionally emit 3-element boxes or nulls when grounding fails.
    """
    try:
        ax0, ay0, ax1, ay1 = [float(v) for v in a]
        bx0, by0, bx1, by1 = [float(v) for v in b]
    except (TypeError, ValueError):
        return 0.0
    # Defensive: ensure x1>=x0, y1>=y0
    ax0, ax1 = min(ax0, ax1), max(ax0, ax1)
    ay0, ay1 = min(ay0, ay1), max(ay0, ay1)
    bx0, bx1 = min(bx0, bx1), max(bx0, bx1)
    by0, by1 = min(by0, by1), max(by0, by1)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_box_match(pred_boxes: list, gt_boxes: list, iou_threshold: float = 0.5) -> dict:
    """Greedy one-to-one IoU matching between predicted and GT boxes.

    Returns dict with tp, fp, fn, precision, recall, f1, mean_iou (of matched pairs).
    Any pair with IoU >= threshold counts as a TP; unmatched predictions are FP;
    unmatched GT boxes are FN. Pairing is greedy highest-IoU-first.
    """
    pairs = [(i, j, box_iou(tuple(p), tuple(g)))
             for i, p in enumerate(pred_boxes) for j, g in enumerate(gt_boxes)]
    pairs.sort(key=lambda t: -t[2])

    matched_pred, matched_gt = set(), set()
    matched_ious: list[float] = []
    for i, j, iou in pairs:
        if iou < iou_threshold:
            break
        if i in matched_pred or j in matched_gt:
            continue
        matched_pred.add(i); matched_gt.add(j); matched_ious.append(iou)

    tp = len(matched_ious)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    precision, recall, f1 = compute_f1(tp, fp, fn)
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": round(f1, 4),
        "mean_iou": round(mean_iou, 4),
    }


def iou_f1_dspy_metric(score_fn):
    """DSPy metric wrapping a spatial score_fn whose output has an f1_score."""

    def metric(example, prediction, trace=None) -> float:
        pred_dict = parse_prediction_document(prediction)
        gt_dict = parse_gt_document(example)
        if pred_dict is None or gt_dict is None:
            return 0.0
        return score_fn(pred_dict, gt_dict)["f1_score"]

    return metric


def iou_f1_gepa_feedback_metric(score_fn):
    """GEPA feedback metric for spatial tasks: reports TP/FP/FN + mean IoU."""

    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        pred_dict = parse_prediction_document(pred)
        gt_dict = parse_gt_document(gold)
        if pred_dict is None or gt_dict is None:
            return dspy.Prediction(score=0.0, feedback="Failed to parse JSON output")
        s = score_fn(pred_dict, gt_dict)
        if s["f1_score"] >= 1.0:
            return dspy.Prediction(score=1.0, feedback="Perfect IoU-F1")
        feedback = (
            f"f1={s['f1_score']:.3f} (tp={s['true_positives']}, "
            f"fp={s['false_positives']}, fn={s['false_negatives']}, "
            f"mean_iou_of_matches={s['mean_iou']:.3f}). "
            "Improve by: tightening box coordinates for matched regions "
            "(boost IoU above 0.5 threshold), removing spurious detections "
            "that drive false positives, and adding missed regions that "
            "drive false negatives."
        )
        return dspy.Prediction(score=s["f1_score"], feedback=feedback)

    return metric


def iou_f1_compute_aggregate_scores(all_scores: list[dict]) -> dict:
    """Macro-F1 and micro-F1 over per-image spatial scores."""
    if not all_scores:
        return {"f1_micro": 0.0, "f1_macro": 0.0}
    total_tp = total_fp = total_fn = 0
    f1s, ious = [], []
    for s in all_scores:
        if isinstance(s, dict) and "f1_score" in s and "mean_iou" in s:
            total_tp += s["true_positives"]
            total_fp += s["false_positives"]
            total_fn += s["false_negatives"]
            f1s.append(s["f1_score"])
            ious.append(s["mean_iou"])
    micro_precision, micro_recall, f1_micro = compute_f1(total_tp, total_fp, total_fn)
    return {
        "f1_macro": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "f1_micro": round(f1_micro, 4),
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "mean_iou_macro": round(sum(ious) / len(ious), 4) if ious else 0.0,
        "total_instances": len(f1s),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }
