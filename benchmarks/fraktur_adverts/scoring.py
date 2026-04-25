"""Scoring for the Fraktur Adverts benchmark.

Ported from humanities_data_benchmark/benchmarks/fraktur_adverts/benchmark.py.

Per-ad matching is section × number-prefix (not positional), so two models
that extract the same ads in a different order still score identically.
Fuzzy similarity runs on the ``text`` field; CER is Levenshtein distance on
normalized (lowercased, whitespace-collapsed) text divided by reference
length, capped at 1.0.

Returns a score dict compatible with the shared ``cer_*`` factories:
    {"fuzzy": float, "cer": float, "field_scores": {key: {...}}}
"""

import logging
import re
from collections import defaultdict

from rapidfuzz.distance import Levenshtein

from benchmarks.shared.scoring_helpers import (
    calculate_fuzzy_score,
    cer_dspy_metric,
    cer_gepa_feedback_metric,
    cer_compute_aggregate_scores,
)

logger = logging.getLogger(__name__)

DEFAULT_SECTION = "Es wird zum Verkauf angetragen"
SECTION_MATCH_THRESHOLD = 0.95

# Used by optimizers that filter demonstrations by metric threshold.
BOOTSTRAP_THRESHOLD = 0.3
REQUIRED_KEYS = {"advertisements"}


def _calculate_cer(reference_text: str, hypothesis_text: str) -> float:
    if not reference_text or not hypothesis_text:
        return 1.0
    ref = " ".join(reference_text.lower().split())
    hyp = " ".join(hypothesis_text.lower().split())
    if ref == hyp:
        return 0.0
    dist = Levenshtein.distance(ref, hyp)
    return min(1.0, dist / max(1, len(ref)))


def _extract_number_prefix(text: str) -> int | None:
    m = re.match(r"^\s*(\d+)\.", text or "")
    return int(m.group(1)) if m else None


def _group_by_section_and_number(ad_list: list, image_name: str | None = None) -> dict:
    grouped: dict = defaultdict(dict)
    if not isinstance(ad_list, list):
        return grouped
    for ad in ad_list:
        if not isinstance(ad, dict):
            continue
        section = (ad.get("tags_section") or "").strip()
        if image_name == "image_4" and not section:
            section = DEFAULT_SECTION
            ad["tags_section"] = DEFAULT_SECTION
        number = _extract_number_prefix(ad.get("text", ""))
        if section and number:
            grouped[section][number] = ad
    return grouped


def _flatten_gt(ground_truth) -> list:
    if isinstance(ground_truth, dict):
        return [entry for ads in ground_truth.values() for entry in ads]
    return ground_truth or []


def _compare_ads(response: dict, ground_truth, image_name: str | None) -> list:
    gt_flat = _flatten_gt(ground_truth)
    resp_ads = (response or {}).get("advertisements", []) or []

    # Auto-detect the image_4 special case: when every GT ad has an empty
    # tags_section, the upstream benchmark applies DEFAULT_SECTION as the
    # fallback heading. Detect this from the GT itself so callers (LOO driver,
    # dspy.Evaluate metric) don't have to thread image_name through.
    needs_default_section = bool(gt_flat) and all(
        not (ad.get("tags_section") or "").strip()
        for ad in gt_flat
        if isinstance(ad, dict)
    )

    if image_name == "image_4" or needs_default_section:
        for ad in resp_ads:
            if isinstance(ad, dict) and not (ad.get("tags_section") or "").strip():
                ad["tags_section"] = DEFAULT_SECTION
        for ad in gt_flat:
            if isinstance(ad, dict) and not (ad.get("tags_section") or "").strip():
                ad["tags_section"] = DEFAULT_SECTION

    response_grouped = _group_by_section_and_number(resp_ads, image_name)
    gt_grouped = _group_by_section_and_number(gt_flat, image_name)

    # When the GT uses DEFAULT_SECTION (the upstream image_4 convention —
    # ads not tied to any printed heading), fall back to number-only matching
    # across all predicted sections rather than requiring section similarity
    # ≥ 0.95. The model often emits a section heading borrowed from other
    # pages (e.g. "Es werden zum Verkauff offerirt"), which would otherwise
    # tank the score even when every ad's text is correct.
    use_section_blind_fallback = DEFAULT_SECTION in gt_grouped or image_name == "image_4"

    results = []
    for section, gt_ads in gt_grouped.items():
        # Exact section match first; fall back to fuzzy section match at >= 0.95
        resp_ads_for_section = response_grouped.get(section, {})
        if not resp_ads_for_section:
            for resp_section, ra in response_grouped.items():
                if calculate_fuzzy_score(resp_section, section) >= SECTION_MATCH_THRESHOLD:
                    resp_ads_for_section = ra
                    break

        for number, gt_ad in gt_ads.items():
            resp_ad = resp_ads_for_section.get(number)

            # If no in-section match and DEFAULT_SECTION is in play, search
            # for the same ad number under any predicted section.
            if not resp_ad and use_section_blind_fallback and section == DEFAULT_SECTION:
                for _, ra in response_grouped.items():
                    if number in ra:
                        resp_ad = ra[number]
                        break

            if resp_ad:
                similarity = calculate_fuzzy_score(resp_ad.get("text"), gt_ad.get("text"))
                results.append({
                    "section": section,
                    "number": number,
                    "match_found": True,
                    "similarity": round(similarity, 3),
                    "response_text": resp_ad.get("text"),
                    "ground_truth_text": gt_ad.get("text"),
                })
            else:
                results.append({
                    "section": section,
                    "number": number,
                    "match_found": False,
                    "similarity": 0.0,
                    "response_text": None,
                    "ground_truth_text": gt_ad.get("text"),
                })
    return results


def score_single_prediction(pred_dict: dict, gt_dict, image_name: str | None = None) -> dict:
    """Score a single predicted ads list against the ground-truth ads list.

    ``image_name`` may be passed by the benchmark driver to enable the
    image_4 special-case; it's not required for unit-level use.
    """
    results = _compare_ads(pred_dict or {}, gt_dict, image_name)

    total_fuzzy = 0.0
    total_cer = 0.0
    field_scores: dict[str, dict] = {}

    for r in results:
        total_fuzzy += r["similarity"]
        if r["match_found"]:
            cer = _calculate_cer(r["ground_truth_text"], r["response_text"])
        else:
            cer = 1.0
        r["cer"] = round(cer, 3)
        total_cer += cer

        key = f"{r['section']}/{r['number']}"
        field_scores[key] = {
            "response": r["response_text"],
            "ground_truth": r["ground_truth_text"],
            "score": r["similarity"],  # fuzzy similarity for per-field feedback
        }

    n = len(results) if results else 1
    avg_fuzzy = total_fuzzy / n if results else 0.0
    avg_cer = total_cer / n if results else 1.0

    return {
        "fuzzy": round(avg_fuzzy, 3),
        "cer": round(avg_cer, 3),
        "field_scores": field_scores,
        "total_ads": len(results),
        "matched_ads": sum(1 for r in results if r["match_found"]),
    }


# DSPy metric wrappers built from the shared CER factories.
dspy_metric = cer_dspy_metric(score_single_prediction)
gepa_feedback_metric = cer_gepa_feedback_metric(score_single_prediction)
compute_aggregate_scores = cer_compute_aggregate_scores


def refine_reward_fn(example, prediction, trace=None) -> float:
    """Minimal valid-JSON + required-keys check for dspy.Refine."""
    from benchmarks.shared.scoring_helpers import parse_prediction_document

    doc = parse_prediction_document(prediction)
    if doc is None or not isinstance(doc, dict):
        return 0.0
    if not REQUIRED_KEYS.issubset(doc.keys()):
        return 0.0
    return 1.0
