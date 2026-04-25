"""Book Advert XML benchmark package.

Public API re-exported for convenience:
    Extractor, load_and_split, dspy_metric, gepa_feedback_metric,
    refine_reward_fn, score_single_prediction, compute_aggregate_scores
"""

from benchmarks.book_advert_xml.data import load_and_split
from benchmarks.book_advert_xml.module import Extractor
from benchmarks.book_advert_xml.scoring import (
    compute_aggregate_scores,
    dspy_metric,
    gepa_feedback_metric,
    refine_reward_fn,
    score_single_prediction,
    xml_refine_reward_fn,
)

__all__ = [
    "Extractor",
    "load_and_split",
    "dspy_metric",
    "gepa_feedback_metric",
    "refine_reward_fn",
    "score_single_prediction",
    "compute_aggregate_scores",
    "xml_refine_reward_fn",
]
