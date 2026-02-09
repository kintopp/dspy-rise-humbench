"""Tool functions for Business Letters extraction.

Provides person lookup and date validation tools using the persons.json
alias table. Used by both the verify-and-correct module and the ReAct agent.
"""

import json
import re
from functools import lru_cache
from pathlib import Path

from benchmarks.shared.config import DATA_DIR

PERSONS_PATH = DATA_DIR / "business_letters" / "ground_truths" / "persons.json"


@lru_cache(maxsize=1)
def _load_persons() -> list[dict]:
    """Load the persons.json alias table (cached)."""
    with open(PERSONS_PATH) as f:
        return json.load(f)


def lookup_person(partial_name: str) -> list[str]:
    """Look up a person name in the alias table.

    Args:
        partial_name: Full or partial name to search for (case-insensitive).

    Returns:
        List of matching canonical names (\"First Last\" format) from
        the alias table. Empty list if no matches found.
    """
    partial_lower = partial_name.strip().lower()
    if not partial_lower:
        return []

    matches = []
    for person in _load_persons():
        canonical = person.get("name", "")
        all_names = [canonical] + person.get("alternateName", [])
        for name in all_names:
            if partial_lower in name.lower():
                if canonical not in matches:
                    matches.append(canonical)
                break
    return matches[:10]


def get_all_persons() -> list[str]:
    """Return all known person names from the alias table.

    Returns:
        List of all canonical person names (\"First Last\" format).
    """
    return [p.get("name", "") for p in _load_persons()]


def validate_date(date_str: str) -> bool:
    """Check whether a date string is in valid YYYY-MM-DD format.

    Args:
        date_str: Date string to validate.

    Returns:
        True if the date matches YYYY-MM-DD format, False otherwise.
    """
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", date_str.strip()))
