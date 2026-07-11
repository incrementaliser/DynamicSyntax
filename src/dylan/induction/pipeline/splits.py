"""Corpus shuffle and train/val/test / k-fold split helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from dylan.induction.em_learner.record_type_corpus import RecordTypeCorpus


@dataclass(frozen=True)
class CorpusSplit:
    """Named corpus slices produced by a split strategy."""

    train: RecordTypeCorpus
    test: RecordTypeCorpus
    val: RecordTypeCorpus | None = None
    fold_index: int | None = None


def _copy_examples(examples: list) -> RecordTypeCorpus:
    """Build a new corpus containing *examples*."""
    corpus = RecordTypeCorpus()
    corpus.extend(examples)
    return corpus


def shuffle_corpus(corpus: RecordTypeCorpus, seed: int) -> RecordTypeCorpus:
    """Return a new corpus with examples shuffled by *seed*."""
    examples = list(corpus)
    rng = random.Random(seed)
    rng.shuffle(examples)
    return _copy_examples(examples)


def holdout_split(
    corpus: RecordTypeCorpus,
    *,
    test_ratio: float,
    seed: int,
) -> CorpusSplit:
    """Shuffle then hold out ``test_ratio`` of examples; train is the remainder."""
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")
    shuffled = shuffle_corpus(corpus, seed)
    n = len(shuffled)
    if n < 2:
        raise ValueError(f"Need at least 2 examples for holdout split, got {n}")
    test_size = int(n * test_ratio)
    test_size = max(1, min(test_size, n - 1))
    train_size = n - test_size
    return CorpusSplit(
        train=_copy_examples(list(shuffled[:train_size])),
        test=_copy_examples(list(shuffled[train_size:])),
    )


def train_val_test_split(
    corpus: RecordTypeCorpus,
    *,
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> CorpusSplit:
    """Shuffle then split: train = remainder after test and val slices."""
    if test_ratio <= 0 or val_ratio < 0 or test_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"Need test_ratio > 0, val_ratio >= 0, and test+val < 1; "
            f"got test_ratio={test_ratio}, val_ratio={val_ratio}",
        )
    shuffled = shuffle_corpus(corpus, seed)
    n = len(shuffled)
    if n < 2:
        raise ValueError(f"Need at least 2 examples for train/val/test split, got {n}")
    test_size = max(1, min(int(n * test_ratio), n - 1))
    val_size = int(n * val_ratio) if val_ratio > 0 else 0
    if test_size + val_size >= n:
        raise ValueError("test+val leave no training examples; lower ratios")
    train_end = n - test_size - val_size
    val_end = train_end + val_size
    train = _copy_examples(list(shuffled[:train_end]))
    val = _copy_examples(list(shuffled[train_end:val_end])) if val_size > 0 else None
    test = _copy_examples(list(shuffled[val_end:]))
    return CorpusSplit(train=train, test=test, val=val)


def kfold_splits(
    corpus: RecordTypeCorpus,
    *,
    folds: int,
    seed: int,
) -> list[CorpusSplit]:
    """Return disjoint k-fold splits after shuffling (Java k-fold parity)."""
    if folds < 2:
        raise ValueError(f"folds must be >= 2, got {folds}")
    shuffled = shuffle_corpus(corpus, seed)
    n = len(shuffled)
    if n < folds:
        raise ValueError(f"Need at least {folds} examples for {folds}-fold CV, got {n}")
    fold_size = n // folds
    if fold_size < 1:
        raise ValueError(f"Fold size is 0 for n={n}, folds={folds}")
    examples = list(shuffled)
    out: list[CorpusSplit] = []
    for i in range(folds):
        start = i * fold_size
        end = start + fold_size
        test_examples = examples[start:end]
        train_examples = examples[:start] + examples[end:]
        out.append(
            CorpusSplit(
                train=_copy_examples(train_examples),
                test=_copy_examples(test_examples),
                fold_index=i,
            ),
        )
    return out


def load_pre_split(
    *,
    train_path: str | Path,
    test_path: str | Path,
    val_path: str | Path | None = None,
) -> CorpusSplit:
    """Load corpora from explicit train/test/(val) file paths."""
    train = RecordTypeCorpus(corpus_path=train_path)
    test = RecordTypeCorpus(corpus_path=test_path)
    val = RecordTypeCorpus(corpus_path=val_path) if val_path else None
    if len(train) == 0:
        raise ValueError(f"Train corpus empty: {train_path}")
    if len(test) == 0:
        raise ValueError(f"Test corpus empty: {test_path}")
    return CorpusSplit(train=train, test=test, val=val)


def save_split_corpora(split: CorpusSplit, directory: str | Path) -> dict[str, Path]:
    """Write train/test/(val) corpora under *directory*; return written paths."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    train_path = root / "train.txt"
    test_path = root / "test.txt"
    split.train.save_corpus(train_path)
    split.test.save_corpus(test_path)
    written["train"] = train_path
    written["test"] = test_path
    if split.val is not None and len(split.val) > 0:
        val_path = root / "val.txt"
        split.val.save_corpus(val_path)
        written["val"] = val_path
    return written
