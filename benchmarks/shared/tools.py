"""Foundation 3 (2026-04-25) — `dspy.ReAct` tool-wrapper convention.

Per offline/2026-04-25_phase-a-foundations-plan.md, this module establishes
the project's convention for tool-augmented benchmarks.

## Convention

Per-benchmark tool implementations live in ``benchmarks/<name>/tools.py``.
Each tool is a Python callable with:
- type-annotated arguments (dspy infers the JSON schema from these),
- a one-line docstring describing what it does (the LLM sees this).

Example:

    # benchmarks/magazine_pages/tools.py
    def opencv_contour_tool(image_path: str) -> list[list[int]]:
        '''Return candidate bounding boxes [[x0,y0,x1,y1], ...] from OpenCV contour detection.'''
        ...

The benchmark's ``module.py`` then builds the ReAct module via
``make_react_module``::

    from benchmarks.shared.tools import make_react_module
    from benchmarks.magazine_pages.tools import opencv_contour_tool
    extractor = make_react_module(MagazineSignature, [opencv_contour_tool])

## Why a thin wrapper around dspy.ReAct?

dspy's native tool support (function with type-annotated args + docstring)
already carries the schema; we don't re-implement it. ``make_react_module``
exists so the project has *one* construction site we can add cross-cutting
concerns to (logging, retry policy, tool-call telemetry) without touching
every benchmark. ``log_tool_calls`` is one such concern; later additions
land here without per-benchmark churn.

## When NOT to use ReAct

If a tool always runs once per inference and its output is just an extra
input to a single Predict call, prefer a multi-module pipeline (separate
``dspy.Predict`` modules wired in ``forward``) over ReAct. ReAct's value is
in *iterative* tool use where the LM decides when to call. For one-shot
augmentation (e.g. always feed OpenCV contours into a single bbox-emitter),
the pipeline pattern is simpler and more deterministic.
"""

from __future__ import annotations

import functools
import logging
from typing import Callable

import dspy

logger = logging.getLogger(__name__)


def make_react_module(
    signature_cls,
    tools: list[Callable],
    *,
    max_iters: int = 4,
) -> dspy.ReAct:
    """Build a ``dspy.ReAct`` module wired with the given tools.

    Args:
        signature_cls: a ``dspy.Signature`` subclass defining the inputs and
            outputs of the overall task. ReAct emits the signature's outputs
            as the final step after any tool calls.
        tools: list of Python callables. Each should have type-annotated
            arguments and a docstring; dspy infers the tool schema and the
            description shown to the LM from these.
        max_iters: maximum number of tool-call rounds before ReAct must emit
            the final output. Default 4. Lower values cap inference cost;
            higher values let the LM explore more before committing.

    Returns:
        A ``dspy.ReAct`` instance ready to be called as ``extractor(**inputs)``.
    """
    return dspy.ReAct(signature_cls, tools=tools, max_iters=max_iters)


def log_tool_calls(func: Callable) -> Callable:
    """Decorator: log every call to a tool function with args + result.

    Useful during prompt iteration to see what the LM is actually asking each
    tool for. Strip when going to production if call volume is high.

    Example:

        @log_tool_calls
        def fraktur_lookup(char: str) -> str | None:
            '''Resolve a Fraktur glyph to its modern equivalent.'''
            return _FRAKTUR_TABLE.get(char)
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            logger.warning(
                "tool %s raised %s on args=%r kwargs=%r",
                func.__name__, type(exc).__name__, args, kwargs,
            )
            raise
        # Truncate result repr so logs stay readable for big returns
        result_repr = repr(result)
        if len(result_repr) > 120:
            result_repr = result_repr[:117] + "…"
        logger.debug(
            "tool %s args=%r kwargs=%r -> %s",
            func.__name__, args, kwargs, result_repr,
        )
        return result
    return wrapped
