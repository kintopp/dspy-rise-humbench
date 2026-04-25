"""Scoring for the Medieval Manuscripts benchmark.

Ported from humanities_data_benchmark/benchmarks/medieval_manuscripts/benchmark.py.

The ground-truth JSON is a dict keyed by folio reference (e.g. "[3r]") where
each value is a single-element list of entry dicts. The LLM output wraps
folios in a flat list under "folios". Scoring:

1. Sort GT items by key and pair positionally with response["folios"][i].
2. For each field (folio, text, addition1..N): if either side non-empty,
   compute fuzzy similarity + CER.
3. Macro-average fuzzy and CER across all scored fields (not across folios).

Primary metric: CER (lower is better). Radar-chart metric: fuzzy.
Downstream DSPy wrappers report ``1 - cer`` as the maximisation target.
"""

import logging

from rapidfuzz.distance import Levenshtein

from benchmarks.shared.scoring_helpers import (
    calculate_fuzzy_score,
    cer_dspy_metric,
    cer_gepa_feedback_metric,
    cer_compute_aggregate_scores,
    parse_prediction_document,
)

logger = logging.getLogger(__name__)

BOOTSTRAP_THRESHOLD = 0.3
REQUIRED_KEYS = {"folios"}
MAX_ADDITIONS = 10  # scan addition1..addition9


def _normalize_empty(value):
    if value is None:
        return ""
    if isinstance(value, str) and not value.strip():
        return ""
    return value


def _calculate_cer(ref: str, hyp: str) -> float:
    ref = _normalize_empty(ref)
    hyp = _normalize_empty(hyp)
    if not ref and not hyp:
        return 0.0
    if not ref or not hyp:
        return 1.0
    ref_n = " ".join(ref.lower().split())
    hyp_n = " ".join(hyp.lower().split())
    if ref_n == hyp_n:
        return 0.0
    return min(1.0, Levenshtein.distance(ref_n, hyp_n) / max(1, len(ref_n)))


def _compare_folios(response: dict, ground_truth: dict) -> list[dict]:
    """Ported from upstream compare_folios: positional match, per-field scoring."""
    results: list[dict] = []
    response_folios = (response or {}).get("folios", []) or []
    if not isinstance(response_folios, list):
        response_folios = []

    gt_items = sorted((ground_truth or {}).items())

    for idx, (folio_ref, gt_entries) in enumerate(gt_items):
        if not gt_entries:
            continue
        gt_entry = gt_entries[0] if isinstance(gt_entries, list) else gt_entries

        response_entry = None
        match_found = False
        if idx < len(response_folios) and isinstance(response_folios[idx], dict):
            response_entry = response_folios[idx]
            match_found = True

        # folio field
        gt_folio = _normalize_empty(gt_entry.get("folio", ""))
        resp_folio = _normalize_empty((response_entry or {}).get("folio", ""))
        if gt_folio or resp_folio:
            results.append({
                "folio_ref": folio_ref,
                "field": "folio",
                "match_found": match_found,
                "similarity": calculate_fuzzy_score(resp_folio, gt_folio) if match_found else 0.0,
                "cer": _calculate_cer(gt_folio, resp_folio) if match_found else 1.0,
                "response_text": resp_folio if match_found else None,
                "ground_truth_text": gt_folio,
            })

        # main text field (always scored)
        gt_text = _normalize_empty(gt_entry.get("text", ""))
        resp_text = _normalize_empty((response_entry or {}).get("text", ""))
        results.append({
            "folio_ref": folio_ref,
            "field": "text",
            "match_found": match_found,
            "similarity": calculate_fuzzy_score(resp_text, gt_text) if match_found else 0.0,
            "cer": _calculate_cer(gt_text, resp_text) if match_found else 1.0,
            "response_text": resp_text if match_found else None,
            "ground_truth_text": gt_text,
        })

        # addition1..addition9 (only scored if at least one side non-empty)
        for i in range(1, MAX_ADDITIONS):
            key = f"addition{i}"
            if key not in gt_entry:
                break
            gt_add = _normalize_empty(gt_entry.get(key))
            resp_add = _normalize_empty((response_entry or {}).get(key, ""))
            if not gt_add and not resp_add:
                continue
            results.append({
                "folio_ref": folio_ref,
                "field": key,
                "match_found": match_found,
                "similarity": calculate_fuzzy_score(resp_add, gt_add) if match_found else 0.0,
                "cer": _calculate_cer(gt_add, resp_add) if match_found else 1.0,
                "response_text": resp_add if match_found else None,
                "ground_truth_text": gt_add,
            })

    return results


def score_single_prediction(pred_dict: dict, gt_dict: dict) -> dict:
    results = _compare_folios(pred_dict or {}, gt_dict or {})
    if not results:
        return {"fuzzy": 0.0, "cer": 1.0, "field_scores": {}, "total_fields": 0}

    total_fuzzy = sum(r["similarity"] for r in results)
    total_cer = sum(r["cer"] for r in results)
    n = len(results)

    field_scores = {
        f"{r['folio_ref']}.{r['field']}": {
            "response": r["response_text"],
            "ground_truth": r["ground_truth_text"],
            "score": r["similarity"],  # fuzzy used for low-field GEPA feedback
        }
        for r in results
    }

    return {
        "fuzzy": round(total_fuzzy / n, 4),
        "cer": round(total_cer / n, 4),
        "field_scores": field_scores,
        "total_fields": n,
    }


# DSPy metric wrappers built from the shared CER factories.
dspy_metric = cer_dspy_metric(score_single_prediction)
gepa_feedback_metric = cer_gepa_feedback_metric(score_single_prediction)
compute_aggregate_scores = cer_compute_aggregate_scores


def refine_reward_fn(example, prediction, trace=None) -> float:
    """Minimal valid-JSON + required-keys check for dspy.Refine."""
    doc = parse_prediction_document(prediction)
    if doc is None or not isinstance(doc, dict):
        return 0.0
    if not REQUIRED_KEYS.issubset(doc.keys()):
        return 0.0
    return 1.0
