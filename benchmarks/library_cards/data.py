"""Data loading: match images to ground truths, split, convert to dspy.Example."""

import json

import dspy

from benchmarks.shared.config import DATA_DIR
from benchmarks.shared.data_helpers import split_data as _split_data
from benchmarks.shared.data_helpers import load_and_split as _load_and_split

IMAGES_DIR = DATA_DIR / "library_cards" / "images"
GROUND_TRUTHS_DIR = DATA_DIR / "library_cards" / "ground_truths"

# Two GTs have no matching image -- exclude them
ORPHAN_GTS = {"00604370", "00604375"}


def load_matched_samples() -> list[dict]:
    """Load all image/GT pairs, skipping orphan GTs."""
    samples = []
    for gt_path in sorted(GROUND_TRUTHS_DIR.glob("*.json")):
        stem = gt_path.stem
        if stem in ORPHAN_GTS:
            continue
        img_path = IMAGES_DIR / f"{stem}.jpg"
        if not img_path.exists():
            continue
        with open(gt_path) as f:
            gt = json.load(f)
        samples.append({
            "id": stem,
            "image_path": str(img_path),
            "ground_truth": gt["response_text"],
        })
    return samples


def split_data(
    samples: list[dict],
    seed: int = 42,
    train_frac: float = 0.15,
    dev_frac: float = 0.15,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split samples into train/dev/test (default 15/15/70)."""
    return _split_data(samples, seed=seed, train_frac=train_frac, dev_frac=dev_frac)


def samples_to_examples(samples: list[dict]) -> list[dspy.Example]:
    """Convert raw samples to dspy.Examples with image input and document output."""
    return [
        dspy.Example(
            card_image=dspy.Image(s["image_path"]),
            document=json.dumps(s["ground_truth"]),
        ).with_inputs("card_image")
        for s in samples
    ]


def load_and_split(seed: int = 42):
    """Convenience: load samples, split, and convert all to dspy.Examples."""
    return _load_and_split(load_matched_samples, split_data, samples_to_examples, seed=seed)
