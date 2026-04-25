"""Data loading for the General Meeting Minutes benchmark."""

import json

import dspy

from benchmarks.shared.config import DATA_DIR
from benchmarks.shared.data_helpers import split_data as _split_data
from benchmarks.shared.data_helpers import load_and_split as _load_and_split

IMAGES_DIR = DATA_DIR / "general_meeting_minutes" / "images"
GROUND_TRUTHS_DIR = DATA_DIR / "general_meeting_minutes" / "ground_truths"


def load_matched_samples() -> list[dict]:
    """Load all image/GT pairs for general_meeting_minutes."""
    samples = []
    for gt_path in sorted(GROUND_TRUTHS_DIR.glob("*.json")):
        stem = gt_path.stem
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
    """Split samples into train/dev/test (default 25/25/50 -> 2/2/5 for 9 images)."""
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


def load_and_split(seed: int = 42):
    """Convenience: load samples, split, and convert all to dspy.Examples."""
    return _load_and_split(load_matched_samples, split_data, samples_to_examples, seed=seed)
