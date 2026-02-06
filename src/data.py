"""Data loading: match images to ground truths, split, convert to dspy.Example."""

import json
import random
from pathlib import Path
from typing import Optional

import dspy

from src.config import IMAGES_DIR, GROUND_TRUTHS_DIR

# Two GTs have no matching image — exclude them
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
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_dev = int(n * dev_frac)
    return shuffled[:n_train], shuffled[n_train : n_train + n_dev], shuffled[n_train + n_dev :]


def samples_to_examples(samples: list[dict]) -> list[dspy.Example]:
    """Convert raw samples to dspy.Examples with image input and document output."""
    examples = []
    for s in samples:
        ex = dspy.Example(
            card_image=dspy.Image(s["image_path"]),
            document=json.dumps(s["ground_truth"]),
        ).with_inputs("card_image")
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
