"""Quality-aware reward for dspy.Refine shared across eval/export/demo scripts."""

from benchmarks.shared.scoring_helpers import parse_prediction_document


class EvalReward:
    """Reward function that uses the actual benchmark metric when GT is available.

    DSPy's Refine calls ``reward_fn(kwargs, outputs)`` where *kwargs* is the
    input dict and *outputs* is a Prediction object.  At evaluation time we
    know the ground truth for each image, so we inject it via ``set_gt()``
    before calling the extractor and ``clear_gt()`` afterwards.
    """

    def __init__(self, scoring_mod):
        self._score_single = scoring_mod.score_single_prediction
        self._gt = None
        # Auto-detect primary metric key: bibliographic_data → "fuzzy", others → "f1_score"
        probe = scoring_mod.score_single_prediction({}, {})
        self._metric_key = "fuzzy" if "fuzzy" in probe else "f1_score"

    @property
    def metric_key(self) -> str:
        return self._metric_key

    def set_gt(self, gt_dict):
        self._gt = gt_dict

    def clear_gt(self):
        self._gt = None

    def __call__(self, kwargs, outputs):
        if self._gt is None:
            raise RuntimeError("EvalReward called without GT; call set_gt() first")
        try:
            pred_dict = parse_prediction_document(outputs)
        except Exception:
            return 0.0
        if pred_dict is None:
            return 0.0
        score = self._score_single(pred_dict, self._gt)
        return score.get(self._metric_key, 0.0)
