"""Data loading for the Medieval Manuscripts benchmark.

The upstream ground-truth JSON is keyed by folio-reference (e.g. "[3r]") at
the top level, each value being a single-element list of entry dicts — NOT
the same shape as the expected LLM output (which wraps a flat "folios" list).
The scoring routine handles that conversion.
"""

import json

import dspy

from benchmarks.shared.config import DATA_DIR
from benchmarks.shared.data_helpers import split_data as _split_data
from benchmarks.shared.data_helpers import load_and_split as _load_and_split

IMAGES_DIR = DATA_DIR / "medieval_manuscripts" / "images"
GROUND_TRUTHS_DIR = DATA_DIR / "medieval_manuscripts" / "ground_truths"


def load_matched_samples() -> list[dict]:
    """Load all image/GT pairs for medieval_manuscripts."""
    samples = []
    for gt_path in sorted(GROUND_TRUTHS_DIR.glob("*.json")):
        stem = gt_path.stem
        # Upstream images use .jpg
        img_path = IMAGES_DIR / f"{stem}.jpg"
        if not img_path.exists():
            continue
        with open(gt_path) as f:
            gt = json.load(f)
        samples.append({
            "id": stem,
            "image_path": str(img_path),
            "ground_truth": gt,
        })
    return samples


def split_data(
    samples: list[dict],
    seed: int = 42,
    train_frac: float = 0.25,
    dev_frac: float = 0.25,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split samples into train/dev/test (default 25/25/50 -> 3/3/6 for 12 images)."""
    return _split_data(
        samples, seed=seed, train_frac=train_frac, dev_frac=dev_frac, min_per_split=1,
    )


def samples_to_examples(samples: list[dict]) -> list[dspy.Example]:
    """Convert raw samples to dspy.Examples with page_image input and document output."""
    return [
        dspy.Example(
            page_image=dspy.Image(s["image_path"]),
            document=json.dumps(s["ground_truth"]),
        ).with_inputs("page_image")
        for s in samples
    ]


def load_loo_folds() -> list[tuple[list[dict], list[dict], list[dict]]]:
    """Leave-one-out folds for small-data experiments (12 images).

    Each fold: 1 test, 1 dev, 10 train. Matches the bibliographic_data pattern.
    """
    samples = sorted(load_matched_samples(), key=lambda s: s["id"])
    folds = []
    for i in range(len(samples)):
        test = [samples[i]]
        remaining = [s for j, s in enumerate(samples) if j != i]
        dev = [remaining[i % len(remaining)]]
        train = [s for s in remaining if s is not dev[0]]
        folds.append((train, dev, test))
    return folds


def load_and_split(seed: int = 42):
    """Convenience: load samples, split, and convert all to dspy.Examples."""
    return _load_and_split(load_matched_samples, split_data, samples_to_examples, seed=seed)
