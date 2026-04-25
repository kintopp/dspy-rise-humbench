"""Fraktur Adverts benchmark package."""

from benchmarks.fraktur_adverts.data import load_and_split, load_loo_folds
from benchmarks.fraktur_adverts.module import Extractor
from benchmarks.fraktur_adverts.scoring import (
    compute_aggregate_scores,
    dspy_metric,
    gepa_feedback_metric,
    refine_reward_fn,
    score_single_prediction,
)

__all__ = [
    "Extractor",
    "load_and_split",
    "load_loo_folds",
    "dspy_metric",
    "gepa_feedback_metric",
    "refine_reward_fn",
    "score_single_prediction",
    "compute_aggregate_scores",
]
