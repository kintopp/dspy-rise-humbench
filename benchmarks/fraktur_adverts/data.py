"""Data loading for the Fraktur Adverts benchmark."""

import json

import dspy

from benchmarks.shared.config import DATA_DIR
from benchmarks.shared.data_helpers import split_data as _split_data
from benchmarks.shared.data_helpers import load_and_split as _load_and_split

IMAGES_DIR = DATA_DIR / "fraktur_adverts" / "images"
GROUND_TRUTHS_DIR = DATA_DIR / "fraktur_adverts" / "ground_truths"


def _find_image_for_stem(stem: str):
    """Images mix .jpg / .jpeg extensions in the upstream repo."""
    for ext in (".jpg", ".jpeg", ".png"):
        p = IMAGES_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_matched_samples() -> list[dict]:
    """Load all image/GT pairs for the fraktur_adverts benchmark."""
    samples = []
    for gt_path in sorted(GROUND_TRUTHS_DIR.glob("*.json")):
        stem = gt_path.stem
        img_path = _find_image_for_stem(stem)
        if img_path is None:
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
    train_frac: float = 0.4,
    dev_frac: float = 0.2,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split samples into train/dev/test (default 40/20/40 for 5 images -> 2/1/2)."""
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
    """Leave-one-out folds: each fold holds out 1 image as test, uses 1 as dev, rest as train.

    Matches the pattern established by bibliographic_data — the 5-image budget
    is too small for a fixed train/dev/test split to yield meaningful numbers.
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
