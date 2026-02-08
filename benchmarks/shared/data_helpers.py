"""Shared data loading helpers used across benchmarks.

Provides generic split_data and load_and_split functions that each
benchmark's data.py can call instead of duplicating.
"""

import random


def split_data(
    samples: list[dict],
    seed: int = 42,
    train_frac: float = 0.15,
    dev_frac: float = 0.15,
    min_per_split: int = 0,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split samples into train/dev/test by fraction.

    Args:
        samples: List of sample dicts.
        seed: Random seed for reproducibility.
        train_frac: Fraction for training set.
        dev_frac: Fraction for dev set.
        min_per_split: Minimum samples per split (useful for tiny datasets).

    Returns:
        (train, dev, test) tuple of sample lists.
    """
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n = len(shuffled)

    if min_per_split > 0:
        n_train = max(min_per_split, int(n * train_frac))
        n_dev = max(min_per_split, int(n * dev_frac))
    else:
        n_train = int(n * train_frac)
        n_dev = int(n * dev_frac)

    assert n_train + n_dev < n, (
        f"train ({n_train}) + dev ({n_dev}) >= total ({n}); test split would be empty"
    )

    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_dev],
        shuffled[n_train + n_dev :],
    )


def load_and_split(
    load_fn,
    split_fn,
    examples_fn,
    seed: int = 42,
) -> tuple:
    """Generic load-split-convert pipeline.

    Args:
        load_fn: Callable returning list[dict] of matched samples.
        split_fn: Callable(samples, seed) returning (train, dev, test) raw splits.
        examples_fn: Callable(samples) returning list[dspy.Example].
        seed: Random seed.

    Returns:
        (train_examples, dev_examples, test_examples, train_raw, dev_raw, test_raw)
    """
    samples = load_fn()
    train_raw, dev_raw, test_raw = split_fn(samples, seed=seed)
    return (
        examples_fn(train_raw),
        examples_fn(dev_raw),
        examples_fn(test_raw),
        train_raw,
        dev_raw,
        test_raw,
    )
