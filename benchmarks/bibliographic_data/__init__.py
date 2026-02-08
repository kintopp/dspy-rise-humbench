"""Bibliographic Data benchmark package.

Public API re-exported for convenience:
    Extractor, load_and_split, dspy_metric, gepa_feedback_metric,
    refine_reward_fn, score_single_prediction, compute_aggregate_scores
"""

from benchmarks.bibliographic_data.data import load_and_split
from benchmarks.bibliographic_data.module import Extractor
from benchmarks.bibliographic_data.scoring import (
    compute_aggregate_scores,
    dspy_metric,
    gepa_feedback_metric,
    refine_reward_fn,
    score_single_prediction,
)

__all__ = [
    "Extractor",
    "load_and_split",
    "dspy_metric",
    "gepa_feedback_metric",
    "refine_reward_fn",
    "score_single_prediction",
    "compute_aggregate_scores",
]
