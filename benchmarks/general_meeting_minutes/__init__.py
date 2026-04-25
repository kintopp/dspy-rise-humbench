"""General Meeting Minutes benchmark package."""

from benchmarks.general_meeting_minutes.data import load_and_split
from benchmarks.general_meeting_minutes.module import Extractor
from benchmarks.general_meeting_minutes.scoring import (
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
