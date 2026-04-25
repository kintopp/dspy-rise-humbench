"""Scoring logic for the Book Advert XML benchmark.

Reimplements BookAdvertXml.score_request_answer / score_benchmark from the
upstream RISE benchmark (humanities_data_benchmark/benchmarks/book_advert_xml/
benchmark.py) for byte-for-byte leaderboard parity:

- Per-sample score is rapidfuzz.fuzz.ratio of pred and GT fixed_xml strings,
  each whitespace-stripped via .replace("\\n","").replace("\\r","").replace(" ","").
  The whitespace-stripping must match upstream EXACTLY.
- Result is divided by 100 to convert from upstream's 0-100 scale to this
  project's internal 0-1 scale (other 'fuzzy' benchmarks use 0-1 internally;
  the leaderboard reports 0-100 = internal × 100).
- Aggregate is the unweighted mean across samples.
- Upstream applies no rounding at any stage. We round per-sample and aggregate
  at 4 decimal places (matching benchmarks/blacklist_cards/scoring.py); the
  resulting drift vs upstream is bounded at ~5e-5 (~0.005 pts in 0-100), well
  below test-split sampling noise.
"""

import logging

import dspy
from rapidfuzz import fuzz

from benchmarks.shared.scoring_helpers import (
    parse_gt_document,
    parse_prediction_document,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_whitespace(s: str) -> str:
    """Match upstream's whitespace-stripping exactly (newlines, carriage returns, spaces)."""
    return s.replace("\n", "").replace("\r", "").replace(" ", "")


# ---------------------------------------------------------------------------
# Per-sample scoring (reimplements BookAdvertXml.score_request_answer)
# ---------------------------------------------------------------------------


def score_single_prediction(pred_dict: dict, gt_dict: dict) -> dict:
    """Score a single prediction against its ground truth.

    Args:
        pred_dict: parsed prediction document with a ``fixed_xml`` key (str).
        gt_dict:   parsed ground truth document with a ``fixed_xml`` key (str).

    Returns dict with keys:
        - fuzzy: float in [0, 1] — rapidfuzz.fuzz.ratio / 100 on whitespace-stripped strings.
        - field_scores: dict of one entry mirroring the per-key structure used
          by the GEPA feedback metric factory.
    """
    pred_xml = _strip_whitespace((pred_dict or {}).get("fixed_xml", "") or "")
    gt_xml = _strip_whitespace((gt_dict or {}).get("fixed_xml", "") or "")

    if not gt_xml:
        return {
            "fuzzy": 0.0,
            "field_scores": {
                "fixed_xml": {"response": pred_xml, "ground_truth": gt_xml, "score": 0.0},
            },
        }

    score = fuzz.ratio(pred_xml, gt_xml) / 100.0

    return {
        "fuzzy": round(score, 4),
        "field_scores": {
            "fixed_xml": {
                "response": pred_xml,
                "ground_truth": gt_xml,
                "score": score,
            },
        },
    }


# ---------------------------------------------------------------------------
# DSPy metric wrappers
#
# Mirrors the structure of benchmarks/blacklist_cards/scoring.py.
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"fixed_xml"}
BOOTSTRAP_THRESHOLD = 0.5


def dspy_metric(example, prediction, trace=None) -> float:
    """Pass ``metric_threshold=BOOTSTRAP_THRESHOLD`` to MIPROv2/BootstrapFewShot for demo filtering."""
    pred_dict = parse_prediction_document(prediction)
    gt_dict = parse_gt_document(example)
    if pred_dict is None or gt_dict is None:
        return 0.0
    return score_single_prediction(pred_dict, gt_dict)["fuzzy"]


def gepa_feedback_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Rich-feedback GEPA metric.

    Returns a dspy.Prediction with score in [0, 1] and a feedback string that
    flags structural anomalies the reflection LM can act on.
    """
    pred_dict = parse_prediction_document(pred)
    gt_dict = parse_gt_document(gold)
    if pred_dict is None:
        return dspy.Prediction(score=0.0, feedback="Failed to parse prediction as JSON with a fixed_xml field")
    if gt_dict is None:
        return dspy.Prediction(score=0.0, feedback="Ground truth document missing or unparseable")
    if "fixed_xml" not in pred_dict:
        return dspy.Prediction(score=0.0, feedback="Prediction JSON has no fixed_xml field")

    pred_xml = pred_dict.get("fixed_xml", "") or ""
    gt_xml = gt_dict.get("fixed_xml", "") or ""

    result = score_single_prediction(pred_dict, gt_dict)
    score = result["fuzzy"]

    parts = [f"fuzzy={score:.3f}"]
    pred_stripped_len = len(_strip_whitespace(pred_xml))
    gt_stripped_len = len(_strip_whitespace(gt_xml))
    if pred_stripped_len == 0:
        parts.append("Output was empty after whitespace-stripping")
    elif gt_stripped_len > 0 and abs(pred_stripped_len - gt_stripped_len) / gt_stripped_len > 0.10:
        parts.append(
            f"Output length ({pred_stripped_len} chars stripped) differs from GT "
            f"({gt_stripped_len} chars stripped) by >10% — likely missing or extra content."
        )
    if "<" in gt_xml and "<" not in pred_xml:
        parts.append("Output contains no XML tags — model may have returned plain text")

    return dspy.Prediction(score=score, feedback=". ".join(parts))


def refine_reward_fn(example, prediction, trace=None) -> float:
    """Reward function for dspy.Refine.

    Cheap structural validity check: 1.0 if the prediction parses as JSON,
    contains a ``fixed_xml`` key, the value is non-empty after whitespace-
    stripping, and contains at least one XML tag. Otherwise 0.0.

    Quality-aware refinement (using the fuzzy metric as reward) should be
    layered on top via EvalReward at evaluation time, not here.
    """
    pred_dict = parse_prediction_document(prediction)
    if pred_dict is None or "fixed_xml" not in pred_dict:
        return 0.0
    fixed_xml = pred_dict.get("fixed_xml", "") or ""
    stripped = _strip_whitespace(fixed_xml)
    if not stripped:
        return 0.0
    if "<" not in fixed_xml or ">" not in fixed_xml:
        return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# Aggregate scoring (reimplements BookAdvertXml.score_benchmark)
# ---------------------------------------------------------------------------


def compute_aggregate_scores(all_scores: list[dict]) -> dict:
    """Compute mean fuzzy score across all scored samples.

    Matches upstream's score_benchmark: unweighted mean, no per-sample weighting.
    """
    if not all_scores:
        return {"fuzzy": 0.0, "total_instances": 0}

    fuzzy_scores = [s["fuzzy"] for s in all_scores if isinstance(s, dict) and "fuzzy" in s]
    avg = sum(fuzzy_scores) / len(fuzzy_scores) if fuzzy_scores else 0.0

    return {
        "fuzzy": round(avg, 4),
        "total_instances": len(fuzzy_scores),
    }
