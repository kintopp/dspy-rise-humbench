"""Data loading for the Book Advert XML benchmark.

Text-only benchmark (no images). Inputs live under data/book_advert_xml/texts/
as JSON files of the shape {"xml_string": "..."}; ground truths live under
data/book_advert_xml/ground_truths/ as JSON files of the shape
{"fixed_xml": "...", "number_of_fixes": N, "explanation": "..."}.

The Example carries:
- xml_string: the malformed XML (input to the model)
- document:   the GT JSON string {"fixed_xml": "..."} (project convention —
  matches the shape every other benchmark uses, so the shared evaluator can
  call parse_prediction_document / parse_gt_document without special-casing).
"""

import json

import dspy

from benchmarks.shared.config import DATA_DIR
from benchmarks.shared.data_helpers import split_data as _split_data
from benchmarks.shared.data_helpers import load_and_split as _load_and_split

TEXTS_DIR = DATA_DIR / "book_advert_xml" / "texts"
GROUND_TRUTHS_DIR = DATA_DIR / "book_advert_xml" / "ground_truths"


def load_matched_samples() -> list[dict]:
    """Load all input/GT pairs for the Book Advert XML benchmark."""
    samples = []
    for gt_path in sorted(GROUND_TRUTHS_DIR.glob("*.json")):
        stem = gt_path.stem
        text_path = TEXTS_DIR / f"{stem}.json"
        if not text_path.exists():
            continue
        with open(text_path) as f:
            text = json.load(f)
        with open(gt_path) as f:
            gt = json.load(f)
        samples.append({
            "id": stem,
            "xml_string": text["xml_string"],
            "ground_truth": {"fixed_xml": gt["fixed_xml"]},
        })
    return samples


def split_data(
    samples: list[dict],
    seed: int = 42,
    train_frac: float = 0.15,
    dev_frac: float = 0.15,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split samples into train/dev/test (default 15/15/70 -> ~7/7/36 for 50 samples)."""
    return _split_data(samples, seed=seed, train_frac=train_frac, dev_frac=dev_frac)


def samples_to_examples(samples: list[dict]) -> list[dspy.Example]:
    """Convert raw samples to dspy.Examples with xml_string input and document (JSON) output."""
    return [
        dspy.Example(
            xml_string=s["xml_string"],
            document=json.dumps(s["ground_truth"]),
        ).with_inputs("xml_string")
        for s in samples
    ]


def load_and_split(seed: int = 42):
    """Convenience: load samples, split, and convert all to dspy.Examples."""
    return _load_and_split(load_matched_samples, split_data, samples_to_examples, seed=seed)
