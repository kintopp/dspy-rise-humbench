"""Data loading for the Bibliographic Data benchmark."""

import json
import random
from pathlib import Path
from typing import Any

import dspy

from benchmarks.shared.config import DATA_DIR

IMAGES_DIR = DATA_DIR / "bibliographic_data" / "images"
GROUND_TRUTHS_DIR = DATA_DIR / "bibliographic_data" / "ground_truths"


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
        gt = _normalize_keys(gt)
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
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = max(1, int(n * train_frac))
    n_dev = max(1, int(n * dev_frac))
    return shuffled[:n_train], shuffled[n_train : n_train + n_dev], shuffled[n_train + n_dev :]


def samples_to_examples(samples: list[dict]) -> list[dspy.Example]:
    """Convert raw samples to dspy.Examples with image input and document output."""
    examples = []
    for s in samples:
        ex = dspy.Example(
            page_image=dspy.Image(s["image_path"]),
            document=json.dumps(s["ground_truth"]),
        ).with_inputs("page_image")
        examples.append(ex)
    return examples


def load_and_split(seed: int = 42):
    """Convenience: load samples, split, and convert all to dspy.Examples."""
    samples = load_matched_samples()
    train_raw, dev_raw, test_raw = split_data(samples, seed=seed)
    return (
        samples_to_examples(train_raw),
        samples_to_examples(dev_raw),
        samples_to_examples(test_raw),
        train_raw,
        dev_raw,
        test_raw,
    )
