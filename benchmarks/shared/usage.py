"""Aggregate per-prediction token usage into a cost-estimator-friendly summary.

DSPy attaches a per-call usage dict to every prediction when
``dspy.configure(track_usage=True)``. Each entry is keyed by the LiteLLM
model id and carries LiteLLM's raw fields (prompt_tokens, completion_tokens,
and optional cache/reasoning breakdowns).

``aggregate_usage`` normalises those keys (``prompt_tokens`` →
``input_tokens``, ``completion_tokens`` → ``output_tokens``) and sums across
a list of predictions, yielding a shape that ``dspy-costs/estimate_costs.py``
can ingest directly via its ``read_measured_usage`` helper.
"""


def aggregate_usage(predictions) -> dict:
    """Sum per-prediction usage into ``{model_id: {input_tokens, output_tokens, calls, ...}}``.

    ``None`` entries (from failed inferences) are skipped. Predictions with
    no usage (e.g. cached hits) contribute ``calls=1`` with zero tokens.
    """
    summary: dict[str, dict] = {}
    for pred in predictions:
        if pred is None:
            continue
        per_call = getattr(pred, "get_lm_usage", lambda: None)() or {}
        for model_id, u in per_call.items():
            if not isinstance(u, dict):
                continue
            dst = summary.setdefault(
                model_id,
                {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                 "cache_read_tokens": 0, "reasoning_tokens": 0},
            )
            dst["calls"] += 1
            dst["input_tokens"] += int(u.get("prompt_tokens", 0) or 0)
            dst["output_tokens"] += int(u.get("completion_tokens", 0) or 0)
            # Surface provider-specific breakdowns when present. LiteLLM nests
            # these under sub-dicts for some providers; tolerate both shapes.
            prompt_details = u.get("prompt_tokens_details") or {}
            if isinstance(prompt_details, dict):
                dst["cache_read_tokens"] += int(prompt_details.get("cached_tokens", 0) or 0)
            completion_details = u.get("completion_tokens_details") or {}
            if isinstance(completion_details, dict):
                dst["reasoning_tokens"] += int(completion_details.get("reasoning_tokens", 0) or 0)

    # Drop optional zero-value breakdowns so the summary stays compact.
    for model_id, u in list(summary.items()):
        for k in ("cache_read_tokens", "reasoning_tokens"):
            if u.get(k, 0) == 0:
                u.pop(k, None)
    return summary
