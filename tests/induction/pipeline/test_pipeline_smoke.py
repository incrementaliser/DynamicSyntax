"""Smoke tests for train/evaluate induction pipeline pieces."""

from __future__ import annotations

from pathlib import Path

import pytest

from dylan.induction.pipeline.config import (
    DataConfig,
    EvalConfig,
    InductionConfig,
    LoggingConfig,
    ModelConfig,
    OutputConfig,
    TrainConfig,
)
from dylan.induction.pipeline.logging_setup import configure_induction_logging
from dylan.induction.pipeline.metrics import EvalResult, SplitMetrics
from dylan.induction.pipeline.report import build_report_text, write_report_file
from dylan.induction.pipeline.train import lexicon_files_exist, prepare_previous_model_seed, train_model
from dylan.induction.em_learner.record_type_corpus import RecordTypeCorpus
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.action.lexicon import Lexicon
from dylan.induction.pipeline.evaluate import build_eval_parser, evaluate_corpus


def test_configure_logging_file(tmp_path: Path) -> None:
    """File sink is created when ``to_file`` is true."""
    cfg = LoggingConfig(level="INFO", to_cli=False, to_file=True, file_name="test.log")
    configure_induction_logging(cfg, run_dir=tmp_path)
    from loguru import logger

    logger.info("hello pipeline")
    log_path = tmp_path / "test.log"
    assert log_path.is_file()
    assert "hello pipeline" in log_path.read_text(encoding="utf-8")


def test_report_contains_scores_timing_config_metadata(tmp_path: Path) -> None:
    """Report text includes F1, timing, config, and metadata sections."""
    result = EvalResult()
    result.add(
        1,
        "train",
        SplitMetrics(80.0, 70.0, 75.0, 100.0, 50.0, 2, 2, 1),
    )
    result.add(
        1,
        "test",
        SplitMetrics(60.0, 50.0, 55.0, 80.0, 40.0, 1, 2, 0),
    )
    result.metadata.update(
        {
            "run_dir": str(tmp_path),
            "split": "holdout",
            "seed": 47,
            "train_time_s": 65.0,
            "eval_time_s": 12.0,
            "train_time": "00-01-05",
            "eval_time": "00-00-12",
        },
    )
    config = InductionConfig()
    text = build_report_text(result, config)
    assert "75.00" in text or "75.0" in text
    assert "seed_grammar" in text
    assert "00-01-05" in text
    assert "Metadata" in text or "run_dir" in text
    out = tmp_path / "report.txt"
    write_report_file(result, config, out)
    assert out.is_file()
    assert "manifest.json" not in text


def test_prepare_previous_model_seed(tmp_path: Path) -> None:
    """Continue-learning seed staging copies lexicon-top-N.txt under the staging dir."""
    prev = tmp_path / "prev"
    prev.mkdir()
    (prev / "lexicon-top-2.txt").write_text("dummy", encoding="utf-8")
    staging = tmp_path / "staging"
    prepare_previous_model_seed(prev, top_n=2, staging_dir=staging)
    assert (staging / "lexicon-top-2.txt").read_text(encoding="utf-8") == "dummy"


def test_lexicon_files_exist(tmp_path: Path) -> None:
    """Detect presence of lexicon-top-1..N under a model dir."""
    assert not lexicon_files_exist(tmp_path, 2)
    (tmp_path / "lexicon-top-1.txt").write_text("a", encoding="utf-8")
    (tmp_path / "lexicon-top-2.txt").write_text("b", encoding="utf-8")
    assert lexicon_files_exist(tmp_path, 2)


def test_reuse_existing_model_skips_train(tmp_path: Path) -> None:
    """reuse_existing_model skips EM when current-dir lexicons exist."""
    grammar = Path("resources/2025-seed-grammar")
    if not (grammar / "computational-actions.txt").is_file():
        pytest.skip("seed grammar missing")
    (tmp_path / "lexicon-top-1.txt").write_text("existing", encoding="utf-8")
    empty = TTRRecordType.parse("[]")
    assert empty is not None
    corpus = RecordTypeCorpus()
    corpus.add_example("open a door", empty)
    prefix, elapsed = train_model(
        train_corpus=corpus,
        seed_grammar=grammar,
        model_dir=tmp_path,
        top_n=1,
        force_train=False,
        reuse_existing_model=True,
        show_progress=False,
    )
    assert elapsed == 0.0
    assert prefix == tmp_path / "lexicon"
    assert (tmp_path / "lexicon-top-1.txt").read_text(encoding="utf-8") == "existing"
    assert not (tmp_path / "computational-actions.txt").exists()


@pytest.mark.timeout(120)
def test_pipeline_holdout_smoke(tmp_path: Path) -> None:
    """End-to-end holdout on the tiny induction-test corpus."""
    corpus = Path("data/induction-test/train.txt")
    grammar = Path("resources/2025-seed-grammar")
    if not corpus.is_file() or not (grammar / "computational-actions.txt").is_file():
        pytest.skip("induction fixtures missing")

    from dylan.induction.pipeline.runner import TrainEvalRunner

    config = InductionConfig(
        data=DataConfig(
            corpus=str(corpus),
            split="holdout",
            test_ratio=0.3,
            seed=47,
            save_splits=True,
        ),
        model=ModelConfig(
            seed_grammar=str(grammar),
            top_n=1,
            use_previous_model=False,
        ),
        train=TrainConfig(show_progress=False, force_train=True),
        eval=EvalConfig(evaluate_on=["test"], top_n_start=1, top_n_end=1),
        logging=LoggingConfig(level="WARNING", to_cli=False, to_file=True),
        output=OutputConfig(
            run_dir=str(tmp_path / "runs"),
            name="smoke",
            write_tsv=True,
            write_report=True,
        ),
    )
    result = TrainEvalRunner(config).run(report_tui=False)
    run_dir = Path(result.metadata["run_dir"])
    assert (run_dir / "metrics.tsv").is_file()
    assert (run_dir / "report.txt").is_file()
    assert not (run_dir / "manifest.json").exists()
    assert (run_dir / "lexicon-top-1.txt").is_file()
    assert not (run_dir / "computational-actions.txt").exists()
    assert result.get(1, "test") is not None
    assert "train_time_s" in result.metadata
    assert "eval_time_s" in result.metadata
    report = (run_dir / "report.txt").read_text(encoding="utf-8")
    assert "00-" in report  # HH-MM-SS style timing


@pytest.mark.timeout(60)
def test_eval_loads_learnt_lexicon_nonzero_words() -> None:
    """Eval parser must load IF/THEN lexicon-top-N.txt (not seed template reader)."""
    grammar = Path("resources/2025-seed-grammar")
    holdout = Path("out/runs/20260710-200128_induction-holdout")
    lex_file = holdout / "lexicon-top-1.txt"
    if not grammar.is_dir() or not lex_file.is_file():
        pytest.skip("holdout run or seed grammar missing")

    lex = Lexicon(holdout, 1, load_learnt_lexicon=True)
    assert len(lex) > 0
    assert lex.load_stats.word_entries_loaded > 0

    parser = build_eval_parser(lexicon_dir=holdout, seed_grammar=grammar, top_n=1)
    try:
        assert len(parser.lexicon) > 0
        # hyp-* must be excluded from eval optional grammar (Java model dirs).
        assert not any(n.lower().startswith("hyp") for n in parser.optional_grammar)
    finally:
        parser.close()


# Minimal Java-parity learnt lexicon (single-IF verb; determiner builds NP).
_JAVA_STYLE_OPEN_A_DOOR_LEXICON = "\n".join(
    [
        "[1.0,0]",
        "a",
        "IF    ?Ty(e)",
        "      ¬<\\/1>Ex.x",
        "      ¬<\\/0>Ex.x",
        "THEN  make(\\/0)",
        "      go(<\\/0>)",
        "      put(?Ty(cn))",
        "      go(</\\0>)",
        "      make(\\/1)",
        "      go(<\\/1>)",
        "      put(?Ty(cn>e))",
        "ELSE  abort",
        "IF    ?Ty(cn>e)",
        "THEN  put(Ty(cn>e))",
        "      ttrput(R1^[r0 : R1|x1==epsilon(r0.head, r0) : e|head==x1 : e])",
        "ELSE  abort",
        "",
        "[1.0,0]",
        "door",
        "IF    ?Ty(cn)",
        "THEN  put(Ty(cn))",
        "      ttrput([x0 : e|head==x0 : e|p0==obj_door(x0) : t])",
        "ELSE  abort",
        "",
        "[1.0,0]",
        "open",
        "IF    ?Ty(e>t)",
        "THEN  put(Ty(e>t))",
        "      ttrput(R1^(R1 ++ [e0==state_opened : es|head==e0 : es|p2==obj(e0, R1.head) : t]))",
        "ELSE  abort",
        "",
    ],
)


@pytest.mark.timeout(60)
def test_eval_java_style_lexicon_parses(tmp_path: Path) -> None:
    """Eval with a Java-style learnt lexicon must parse and get non-zero coverage."""
    grammar = Path("resources/2025-seed-grammar")
    if not (grammar / "computational-actions.txt").is_file():
        pytest.skip("seed grammar missing")

    (tmp_path / "lexicon-top-1.txt").write_text(_JAVA_STYLE_OPEN_A_DOOR_LEXICON, encoding="utf-8")
    parser = build_eval_parser(lexicon_dir=tmp_path, seed_grammar=grammar, top_n=1)
    try:
        gold = TTRRecordType.parse(
            "[e0==state_opened : es|x0 : e|p0==obj_door(x0) : t|p1==obj(e0, x0) : t|head==e0 : es]",
        )
        assert gold is not None
        corpus = RecordTypeCorpus()
        corpus.add_example("open a door", gold)
        metrics = evaluate_corpus(parser, corpus)
        assert metrics.parsed_count >= 1
        assert metrics.coverage > 0.0
    finally:
        parser.close()
