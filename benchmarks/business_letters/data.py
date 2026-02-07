"""Data loading: group multi-page images by letter, match to GTs, split, convert to dspy.Example."""

import json
import random
import re
from pathlib import Path

import dspy

from benchmarks.shared.config import DATA_DIR

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
        letter_id = m.group(1)  # e.g. "01", "65"
        page_num = int(m.group(2))
        groups.setdefault(letter_id, []).append((page_num, str(img_path)))

    # Sort pages within each letter and return just paths
    return {
        lid: [path for _, path in sorted(pages)]
        for lid, pages in groups.items()
    }


def load_matched_samples() -> list[dict]:
    """Load all letter/GT pairs, grouping page images per letter."""
    letter_images = _group_images_by_letter()
    samples = []

    for gt_path in sorted(GROUND_TRUTHS_DIR.glob("letter*.json")):
        # Extract letter number from filename (e.g. "letter01.json" -> "01")
        m = re.match(r"^letter(\d+)\.json$", gt_path.name)
        if not m:
            continue  # skip letterFALSE.json, persons.json, etc.
        letter_id = m.group(1)

        if letter_id not in letter_images:
            continue  # no images for this letter

        with open(gt_path) as f:
            gt = json.load(f)

        samples.append({
            "id": f"letter{letter_id}",
            "image_paths": letter_images[letter_id],  # list of page paths
            "ground_truth": gt,  # root dict with send_date, sender_persons, etc.
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
    """Convert raw samples to dspy.Examples with multi-image input and document output."""
    examples = []
    for s in samples:
        ex = dspy.Example(
            page_images=[dspy.Image(p) for p in s["image_paths"]],
            document=json.dumps(s["ground_truth"]),
        ).with_inputs("page_images")
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
