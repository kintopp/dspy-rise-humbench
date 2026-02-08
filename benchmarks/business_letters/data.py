"""Data loading: group multi-page images by letter, match to GTs, split, convert to dspy.Example."""

import json
import re

import dspy

from benchmarks.shared.config import DATA_DIR
from benchmarks.shared.data_helpers import split_data as _split_data
from benchmarks.shared.data_helpers import load_and_split as _load_and_split

IMAGES_DIR = DATA_DIR / "business_letters" / "images"
GROUND_TRUTHS_DIR = DATA_DIR / "business_letters" / "ground_truths"

# Regex to parse letterNN_pX.jpg filenames
_IMAGE_RE = re.compile(r"^letter(\d+)_p(\d+)\.jpg$")


def _group_images_by_letter() -> dict[str, list[str]]:
    """Group page images by letter ID, sorted by page number.

    Returns:
        dict mapping letter ID (e.g. "01") to sorted list of image paths.
    """
    groups: dict[str, list[tuple[int, str]]] = {}
    for img_path in IMAGES_DIR.iterdir():
        m = _IMAGE_RE.match(img_path.name)
        if not m:
            continue
        letter_id = m.group(1)
        page_num = int(m.group(2))
        groups.setdefault(letter_id, []).append((page_num, str(img_path)))

    return {
        lid: [path for _, path in sorted(pages)]
        for lid, pages in groups.items()
    }


def load_matched_samples() -> list[dict]:
    """Load all letter/GT pairs, grouping page images per letter."""
    letter_images = _group_images_by_letter()
    samples = []

    for gt_path in sorted(GROUND_TRUTHS_DIR.glob("letter*.json")):
        m = re.match(r"^letter(\d+)\.json$", gt_path.name)
        if not m:
            continue  # skip letterFALSE.json, persons.json, etc.
        letter_id = m.group(1)

        if letter_id not in letter_images:
            continue

        with open(gt_path) as f:
            gt = json.load(f)

        samples.append({
            "id": f"letter{letter_id}",
            "image_paths": letter_images[letter_id],
            "ground_truth": gt,
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
    """Convert raw samples to dspy.Examples with multi-image input and document output."""
    return [
        dspy.Example(
            page_images=[dspy.Image(p) for p in s["image_paths"]],
            document=json.dumps(s["ground_truth"]),
        ).with_inputs("page_images")
        for s in samples
    ]


def load_and_split(seed: int = 42):
    """Convenience: load samples, split, and convert all to dspy.Examples."""
    return _load_and_split(load_matched_samples, split_data, samples_to_examples, seed=seed)
