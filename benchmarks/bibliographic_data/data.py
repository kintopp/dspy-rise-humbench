"""Data loading for the Bibliographic Data benchmark."""

import json
from typing import Any

import dspy

from benchmarks.shared.config import DATA_DIR
from benchmarks.shared.data_helpers import split_data as _split_data
from benchmarks.shared.data_helpers import load_and_split as _load_and_split

IMAGES_DIR = DATA_DIR / "bibliographic_data" / "images"
GROUND_TRUTHS_DIR = DATA_DIR / "bibliographic_data" / "ground_truths"


# page_10.json uses CSL-JSON conventions that differ from the other pages.
# Normalize both keys and type values at load time.
_TYPE_MAP = {
    "article-journal": "journal-article",
    "chapter": "book",  # no "chapter" type in schema; closest is "book"
    "review": "journal-article",  # page_2 has "review"; treat as article
}


def _normalize_keys(obj: Any) -> Any:
    """Recursively replace hyphens with underscores in dict keys.

    page_10.json uses CSL-JSON hyphenated keys (publisher-place, container-title)
    while other pages use underscored keys. Normalize to underscored.
    """
    if isinstance(obj, dict):
        return {k.replace("-", "_"): _normalize_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_keys(item) for item in obj]
    return obj


def _normalize_type_values(obj: Any) -> Any:
    """Normalize entry type values to match the schema.

    page_10 uses CSL-JSON 'article-journal' instead of 'journal-article',
    and 'chapter' which isn't in our schema. page_2 has 'review'.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "type" and isinstance(v, str) and v in _TYPE_MAP:
                out[k] = _TYPE_MAP[v]
            else:
                out[k] = _normalize_type_values(v)
        return out
    if isinstance(obj, list):
        return [_normalize_type_values(item) for item in obj]
    return obj


def load_matched_samples() -> list[dict]:
    """Load all image/GT pairs for the bibliographic data benchmark."""
    samples = []
    for gt_path in sorted(GROUND_TRUTHS_DIR.glob("*.json")):
        stem = gt_path.stem
        img_path = IMAGES_DIR / f"{stem}.jpeg"
        if not img_path.exists():
            continue
        with open(gt_path) as f:
            gt = json.load(f)
        gt = _normalize_type_values(_normalize_keys(gt))
        samples.append({
            "id": stem,
            "image_path": str(img_path),
            "ground_truth": gt,
        })
    return samples


def split_data(
    samples: list[dict],
    seed: int = 42,
    train_frac: float = 0.4,
    dev_frac: float = 0.2,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split samples into train/dev/test (default 40/20/40 for 5 images -> 2/1/2)."""
    return _split_data(
        samples, seed=seed, train_frac=train_frac, dev_frac=dev_frac, min_per_split=1,
    )


def samples_to_examples(samples: list[dict]) -> list[dspy.Example]:
    """Convert raw samples to dspy.Examples with image input and document output."""
    return [
        dspy.Example(
            page_image=dspy.Image(s["image_path"]),
            document=json.dumps(s["ground_truth"]),
        ).with_inputs("page_image")
        for s in samples
    ]


def load_loo_folds() -> list[tuple[list[dict], list[dict], list[dict]]]:
    """Return leave-one-out folds: [(train_raw, dev_raw, test_raw), ...].

    Each fold holds out 1 image as test, uses 1 as dev, and the rest as train.
    Images sorted by ID for deterministic ordering.
    """
    samples = sorted(load_matched_samples(), key=lambda s: s["id"])
    folds = []
    for i in range(len(samples)):
        test = [samples[i]]
        remaining = [s for j, s in enumerate(samples) if j != i]
        # Rotate dev assignment so each fold uses a different dev image
        dev = [remaining[i % len(remaining)]]
        train = [s for s in remaining if s is not dev[0]]
        folds.append((train, dev, test))
    return folds


def load_and_split(seed: int = 42):
    """Convenience: load samples, split, and convert all to dspy.Examples."""
    return _load_and_split(load_matched_samples, split_data, samples_to_examples, seed=seed)
