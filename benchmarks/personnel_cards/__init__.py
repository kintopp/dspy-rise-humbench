"""Personnel Cards benchmark package.

Public API re-exported for convenience:
    Extractor, load_and_split, dspy_metric, gepa_feedback_metric,
    refine_reward_fn, score_single_prediction, compute_aggregate_scores
"""

from benchmarks.personnel_cards.data import load_and_split
from benchmarks.personnel_cards.module import Extractor
from benchmarks.personnel_cards.scoring import (
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
