"""Foundation 4 (2026-04-25) — Composable reward-function validators.

## Why this isn't called ``asserts.py``

The original foundations plan called for ``dspy.Assert`` / ``dspy.Suggest``
helpers. Those primitives existed in dspy 2.x and let a ``forward`` method
declare soft constraints; dspy automatically caught violations and re-prompted
the LM. **They were removed in dspy 3.x.** The replacement idiom is:

- Define soft constraints as **reward functions** that map a prediction to a
  scalar in [0, 1].
- Pass those reward functions to ``dspy.Refine``, which retries with a higher
  temperature when the reward falls below a threshold.

This module provides composable reward functions for common constraint
shapes (well-formed JSON, regex match on a field, set membership, etc.) plus
``combine_rewards`` to AND them together. Benchmarks then expose a single
``refine_reward_fn`` built from these primitives.

## Existing usage to follow

``benchmarks/shared/scoring_helpers.py::f1_refine_reward_fn(required_keys)``
is the canonical example: it returns a reward function that yields 1.0 if
the prediction parses to JSON with all ``required_keys`` and 0.0 otherwise.
This module generalises that pattern.

## Reward-function shape

Every reward function in this module conforms to dspy.Refine's expected
signature::

    reward(example, prediction, trace=None) -> float

where the float is in [0, 1]. ``1.0`` means "the prediction passes this
constraint." ``0.0`` means "violated." Intermediate values are allowed for
graded constraints (rare; usually constraints are binary).
"""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable

from benchmarks.shared.scoring_helpers import (
    parse_prediction_document,
)


# ---------------------------------------------------------------------------
# Single-constraint reward factories
# ---------------------------------------------------------------------------


def json_well_formed_reward(required_keys: Iterable[str]) -> Callable:
    """Reward 1.0 iff prediction.document parses as JSON with all required keys.

    Args:
        required_keys: iterable of top-level key names that must be present.

    Returns:
        ``reward(example, prediction, trace=None) -> float``.
    """
    required = set(required_keys)

    def reward(example, prediction, trace=None) -> float:
        doc = parse_prediction_document(prediction)
        if doc is None:
            return 0.0
        return 1.0 if required.issubset(doc.keys()) else 0.0

    return reward


def regex_match_reward(field_path: str, pattern: str) -> Callable:
    """Reward 1.0 iff the value at ``field_path`` matches the regex pattern.

    ``field_path`` uses dotted notation (e.g. ``"author.last_name"``). Returns
    0.0 if the path does not exist or the value does not match.

    Args:
        field_path: dotted key path into the parsed prediction document.
        pattern: a regex that ``re.fullmatch`` will run against the value.
    """
    compiled = re.compile(pattern)
    parts = field_path.split(".")

    def reward(example, prediction, trace=None) -> float:
        doc = parse_prediction_document(prediction)
        if doc is None:
            return 0.0
        cur: Any = doc
        for p in parts:
            if not isinstance(cur, dict) or p not in cur:
                return 0.0
            cur = cur[p]
        if not isinstance(cur, str):
            return 0.0
        return 1.0 if compiled.fullmatch(cur) else 0.0

    return reward


def field_in_set_reward(field_path: str, allowed: Iterable[str]) -> Callable:
    """Reward 1.0 iff the value at ``field_path`` is one of ``allowed``.

    Useful for enum-shaped fields (e.g. document type, language code).
    """
    allowed_set = set(allowed)
    parts = field_path.split(".")

    def reward(example, prediction, trace=None) -> float:
        doc = parse_prediction_document(prediction)
        if doc is None:
            return 0.0
        cur: Any = doc
        for p in parts:
            if not isinstance(cur, dict) or p not in cur:
                return 0.0
            cur = cur[p]
        return 1.0 if cur in allowed_set else 0.0

    return reward


def callable_predicate_reward(predicate: Callable[[dict], bool]) -> Callable:
    """Reward 1.0 iff ``predicate(parsed_document)`` returns truthy.

    Catch-all for constraints that don't fit the other factories. The
    predicate receives the parsed JSON dict and may inspect arbitrary
    structure. It must return a boolean (or truthy/falsy value).
    """
    def reward(example, prediction, trace=None) -> float:
        doc = parse_prediction_document(prediction)
        if doc is None:
            return 0.0
        try:
            return 1.0 if predicate(doc) else 0.0
        except Exception:
            return 0.0

    return reward


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def combine_rewards(*reward_fns: Callable, mode: str = "min") -> Callable:
    """Combine multiple reward functions into one.

    Args:
        reward_fns: any number of reward functions with the standard
            ``(example, prediction, trace=None) -> float`` signature.
        mode: how to combine the per-constraint rewards.
            - ``"min"`` (default): the combined reward is the minimum across
              constraints — i.e. all constraints must pass to yield 1.0.
              Equivalent to AND for binary rewards.
            - ``"mean"``: the combined reward is the unweighted mean —
              partial credit for partial compliance.
            - ``"product"``: the combined reward is the product —
              equivalent to AND but penalises near-misses harder than mean.

    Returns:
        A single reward function of the standard signature.
    """
    if not reward_fns:
        raise ValueError("combine_rewards requires at least one reward_fn")
    if mode not in ("min", "mean", "product"):
        raise ValueError(f"mode must be one of 'min'/'mean'/'product', got {mode!r}")

    def reward(example, prediction, trace=None) -> float:
        scores = [fn(example, prediction, trace) for fn in reward_fns]
        if mode == "min":
            return min(scores)
        if mode == "mean":
            return sum(scores) / len(scores)
        # product
        out = 1.0
        for s in scores:
            out *= s
        return out

    return reward
