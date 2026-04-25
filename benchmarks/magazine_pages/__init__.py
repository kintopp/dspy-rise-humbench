"""Magazine Pages benchmark package."""

from benchmarks.magazine_pages.data import load_and_split
from benchmarks.magazine_pages.module import Extractor
from benchmarks.magazine_pages.scoring import (
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
