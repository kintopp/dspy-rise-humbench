"""Scoring for the Magazine Pages benchmark.

Ported from humanities_data_benchmark/benchmarks/magazine_pages/benchmark.py.

Per-image: greedy 1:1 IoU matching at IoU >= 0.5, computing TP/FP/FN and
mean_iou of matched pairs. Uses the shared ``greedy_box_match`` helper.

Primary metric: F1 (higher is better). Per-image score also exposes
precision, recall, mean_iou, TP/FP/FN for richer GEPA feedback.

Perfect-empty case: if both GT and prediction have zero ads, score is 1.0.
"""

import logging

from benchmarks.shared.scoring_helpers import (
    greedy_box_match,
    iou_f1_dspy_metric,
    iou_f1_gepa_feedback_metric,
    iou_f1_compute_aggregate_scores,
    parse_prediction_document,
)

logger = logging.getLogger(__name__)

IOU_THRESHOLD = 0.5
BOOTSTRAP_THRESHOLD = 0.2
REQUIRED_KEYS = {"advertisements"}


def _extract_boxes(doc) -> list:
    """Pull a flat list of [x0,y0,x1,y1] from either 'advertisements' or 'detections'.

    Silently drops malformed boxes (non-list, wrong length, non-numeric) so
    downstream IoU computation never sees a partial coordinate tuple.
    """
    if not isinstance(doc, dict):
        return []
    for key in ("advertisements", "detections"):
        items = doc.get(key)
        if isinstance(items, list) and items:
            boxes = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                box = it.get("box")
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    continue
                try:
                    boxes.append([float(v) for v in box])
                except (TypeError, ValueError):
                    continue
            if boxes:
                return boxes
    return []


def score_single_prediction(pred_dict: dict, gt_dict: dict) -> dict:
    """Per-image IoU-F1 score with TP/FP/FN + mean_iou."""
    pred_boxes = _extract_boxes(pred_dict)
    gt_boxes = _extract_boxes(gt_dict)

    # Perfect empty: model correctly said "no ads"
    if not gt_boxes and not pred_boxes:
        return {
            "f1_score": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "mean_iou": 1.0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "iou_threshold": IOU_THRESHOLD,
        }

    result = greedy_box_match(pred_boxes, gt_boxes, iou_threshold=IOU_THRESHOLD)
    result["iou_threshold"] = IOU_THRESHOLD
    return result


# DSPy metric wrappers built from the shared IoU-F1 factories.
dspy_metric = iou_f1_dspy_metric(score_single_prediction)
gepa_feedback_metric = iou_f1_gepa_feedback_metric(score_single_prediction)
compute_aggregate_scores = iou_f1_compute_aggregate_scores


def refine_reward_fn(example, prediction, trace=None) -> float:
    """Valid-JSON + required-keys check for dspy.Refine."""
    doc = parse_prediction_document(prediction)
    if doc is None or not isinstance(doc, dict):
        return 0.0
    if not REQUIRED_KEYS.issubset(doc.keys()):
        return 0.0
    return 1.0
