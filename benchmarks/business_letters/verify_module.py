"""Verify-and-correct module for Business Letters.

Post-hoc name correction: runs the base extractor, parses the output,
and replaces predicted person names with canonical aliases from
persons.json when a match is found. Zero additional LM calls.
"""

import json
import logging

import dspy

from benchmarks.business_letters.tools import _load_persons
from benchmarks.shared.scoring_helpers import parse_prediction_document, strip_code_fences

logger = logging.getLogger(__name__)


def _find_canonical(predicted_name: str, persons_db: list[dict]) -> str | None:
    """Find the GT-format canonical name for a predicted name via alias lookup.

    Only uses exact alias matches (case-sensitive) to avoid false positives.
    Returns the person's ``name`` field (\"Last, First\" format) which matches
    how GT person names are stored.
    """
    predicted_stripped = predicted_name.strip()
    if not predicted_stripped:
        return None

    for person in persons_db:
        canonical = person.get("name", "")
        all_names = [canonical] + person.get("alternateName", [])
        if predicted_stripped in all_names:
            return canonical

    return None


def _correct_person_list(names: list, persons_db: list[dict]) -> list:
    """Correct a list of person names using the alias table."""
    corrected = []
    for name in names:
        if not isinstance(name, str):
            corrected.append(name)
            continue
        canonical = _find_canonical(name, persons_db)
        if canonical is not None and canonical != name:
            logger.debug(f"  Corrected: {name!r} → {canonical!r}")
            corrected.append(canonical)
        else:
            corrected.append(name)
    return corrected


class VerifyExtractor(dspy.Module):
    """Wraps a base extractor with post-hoc person name verification.

    After the base module produces a prediction, this module parses the
    JSON output, looks up each person name in persons.json, and replaces
    with the canonical form when a match is found.
    """

    def __init__(self, base_module):
        super().__init__()
        self.base = base_module

    def forward(self, page_images):
        prediction = self.base(page_images=page_images)

        # Parse the prediction JSON
        pred_dict = parse_prediction_document(prediction)
        if pred_dict is None:
            return prediction

        persons_db = _load_persons()
        modified = False

        for key in ("sender_persons", "receiver_persons"):
            names = pred_dict.get(key)
            if isinstance(names, list):
                corrected = _correct_person_list(names, persons_db)
                if corrected != names:
                    pred_dict[key] = corrected
                    modified = True

        if modified:
            # Replace the document field with corrected JSON
            prediction.document = json.dumps(pred_dict)

        return prediction
