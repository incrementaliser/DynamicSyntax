"""Unit tests for induction pipeline config, splits, and metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.em_learner.record_type_corpus import RecordTypeCorpus
from dylan.induction.pipeline.config import apply_overrides, load_config, load_yaml
from dylan.induction.pipeline.metrics import EvalResult, SplitMetrics, write_metrics_tsv
from dylan.induction.pipeline.splits import holdout_split, kfold_splits, shuffle_corpus
from dylan.induction.pipeline.timing import format_hh_mm_ss


def _tiny_corpus(n: int = 6) -> RecordTypeCorpus:
    """Build a small synthetic corpus with empty TTR targets."""
    corpus = RecordTypeCorpus()
    empty = TTRRecordType.parse("[]")
    assert empty is not None
    for i in range(n):
        corpus.add_example(f"word{i} token", empty)
    return corpus


def words_to_str(example: tuple) -> str:
    """Stable string key for a corpus example."""
    words, _ = example
    return " ".join(w.word() for w in words)


def test_format_hh_mm_ss() -> None:
    """Seconds format as zero-padded HH-MM-SS."""
    assert format_hh_mm_ss(0) == "00-00-00"
    assert format_hh_mm_ss(65) == "00-01-05"
    assert format_hh_mm_ss(3661) == "01-01-01"


def test_load_yaml_and_overrides(tmp_path: Path) -> None:
    """YAML load plus dotted ``--set`` overrides update nested fields."""
    path = tmp_path / "cfg.yaml"
    path.write_text(
        "data:\n  seed: 1\n  split: holdout\n  test_ratio: 0.2\nmodel:\n  top_n: 2\n",
        encoding="utf-8",
    )
    raw = apply_overrides(load_yaml(path), ["data.seed=99", "logging.level=DEBUG", "model.top_n=5"])
    cfg = load_config(path, overrides=["data.seed=99", "logging.level=DEBUG", "model.top_n=5"])
    assert cfg.data.seed == 99
    assert cfg.logging.level == "DEBUG"
    assert cfg.model.top_n == 5
    assert raw["data"]["seed"] == 99


def test_holdout_split_deterministic() -> None:
    """Same seed yields identical holdout membership."""
    corpus = _tiny_corpus(10)
    a = holdout_split(corpus, test_ratio=0.3, seed=47)
    b = holdout_split(corpus, test_ratio=0.3, seed=47)
    assert len(a.train) == len(b.train)
    assert len(a.test) == len(b.test)
    assert [words_to_str(x) for x in a.train] == [words_to_str(x) for x in b.train]
    assert len(a.test) == int(10 * 0.3) or len(a.test) >= 1


def test_kfold_disjoint_test_sets() -> None:
    """No example appears in more than one fold's test set."""
    corpus = _tiny_corpus(9)
    folds = kfold_splits(corpus, folds=3, seed=7)
    assert len(folds) == 3
    seen: set[str] = set()
    for fold in folds:
        keys = {words_to_str(ex) for ex in fold.test}
        assert keys.isdisjoint(seen)
        seen |= keys
        train_keys = {words_to_str(ex) for ex in fold.train}
        assert keys.isdisjoint(train_keys)


def test_shuffle_changes_order() -> None:
    """Different seeds produce different orders (with high probability)."""
    corpus = _tiny_corpus(20)
    a = shuffle_corpus(corpus, 1)
    b = shuffle_corpus(corpus, 2)
    assert [words_to_str(x) for x in a] != [words_to_str(x) for x in b]


def test_write_metrics_tsv(tmp_path: Path) -> None:
    """TSV writer emits header and one row per top-N/split."""
    result = EvalResult()
    result.add(
        1,
        "test",
        SplitMetrics(
            precision=10.0,
            recall=20.0,
            f1=15.0,
            coverage=50.0,
            exact_match=25.0,
            parsed_count=1,
            total_count=2,
            exact_match_count=0,
        ),
    )
    out = tmp_path / "metrics.tsv"
    write_metrics_tsv(result, out)
    text = out.read_text(encoding="utf-8")
    assert "TopN" in text
    assert "test" in text
    assert "15.0000" in text


def test_load_example_yaml() -> None:
    """Repo example holdout YAML loads with expected defaults."""
    path = Path("configs/induction/holdout.yaml")
    if not path.is_file():
        pytest.skip("example config not present")
    cfg = load_config(path)
    assert cfg.data.split == "holdout"
    assert cfg.data.test_ratio == 0.10
    assert cfg.model.top_n == 3
    assert cfg.train.force_train is False
    assert cfg.train.reuse_existing_model is False
    assert cfg.model.use_previous_model is False
    assert cfg.output.write_tsv is True
    assert cfg.output.name == "induction-holdout"
