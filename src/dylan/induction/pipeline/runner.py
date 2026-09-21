"""Orchestrate split → train → evaluate → report for TTR induction."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from dylan.induction.corpus_profile import set_active_profile
from dylan.induction.em_learner.record_type_corpus import RecordTypeCorpus
from dylan.induction.pipeline.config import InductionConfig, dump_config
from dylan.induction.pipeline.evaluate import evaluate_model_dir
from dylan.induction.pipeline.logging_setup import configure_induction_logging
from dylan.induction.pipeline.metrics import EvalResult, write_metrics_tsv
from dylan.induction.pipeline.report import launch_report_tui, print_report, write_report_file
from dylan.induction.pipeline.splits import (
    CorpusSplit,
    holdout_split,
    kfold_splits,
    load_pre_split,
    save_split_corpora,
    train_val_test_split,
)
from dylan.induction.pipeline.timing import format_hh_mm_ss
from dylan.induction.pipeline.train import train_model


def _git_commit() -> str | None:
    """Return the current git commit hash if available."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _make_run_dir(config: InductionConfig) -> Path:
    """Create a timestamped run directory under ``output.run_dir``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    root = Path(config.output.run_dir)
    run_dir = root / f"{stamp}_{config.output.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_source_corpus(config: InductionConfig) -> RecordTypeCorpus:
    """Load the single corpus file used for auto-splitting."""
    if not config.data.corpus:
        raise ValueError("data.corpus is required for split modes other than pre_split")
    path = Path(config.data.corpus)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus not found: {path}")
    corpus = RecordTypeCorpus(corpus_path=path)
    if len(corpus) == 0:
        raise ValueError(f"Corpus is empty: {path}")
    return corpus


def _corpora_for_eval(split: CorpusSplit, evaluate_on: list[str]) -> dict[str, RecordTypeCorpus]:
    """Select named corpora listed in *evaluate_on* (train may be omitted)."""
    available: dict[str, RecordTypeCorpus | None] = {
        "train": split.train,
        "test": split.test,
        "val": split.val,
    }
    out: dict[str, RecordTypeCorpus] = {}
    for name in evaluate_on:
        corp = available.get(name)
        if corp is None:
            if name == "val":
                logger.warning("Skipping eval on val: no validation split")
                continue
            if name == "train":
                logger.warning("Skipping eval on train: train corpus unavailable")
                continue
            raise ValueError(f"Unknown evaluate_on split: {name!r}")
        out[name] = corp
    if not out:
        raise ValueError("evaluate_on produced no corpora to evaluate")
    return out


def _resolve_previous_model(config: InductionConfig) -> Path | None:
    """Return previous-model path when ``use_previous_model`` is enabled.

    The path must be a directory that directly contains ``lexicon-top-N.txt``
    (typically a prior run's ``models/`` folder); nested paths are not searched.
    """
    if not config.model.use_previous_model:
        return None
    if not config.model.previous_model:
        raise ValueError("model.use_previous_model is true but model.previous_model is unset")
    path = Path(config.model.previous_model)
    if not path.exists():
        raise FileNotFoundError(f"previous_model not found: {path}")
    return path


def _print_run_stage(config: InductionConfig, run_dir: Path) -> None:
    """Print a one-line summary of run identity and continual-learning status."""
    parts: list[str] = [f"dir={run_dir}", f"name={config.output.name}"]
    if config.model.use_previous_model:
        parts.append(f"continual_learning=true previous_model={config.model.previous_model}")
    else:
        parts.append("continual_learning=false")
    print(f"[Run]   {'  '.join(parts)}", flush=True)


def _print_data_stage(
    config: InductionConfig,
    split: CorpusSplit,
    *,
    n_folds: int | None = None,
) -> None:
    """Print a one-line summary after data load / split."""
    parts: list[str] = [f"split={config.data.split}"]
    if split.fold_index is not None and n_folds is not None:
        parts.append(f"fold={split.fold_index + 1}/{n_folds}")
    mode = config.data.split
    if mode == "holdout":
        parts.append(f"test_ratio={config.data.test_ratio}")
    elif mode == "train_val_test":
        parts.append(f"test_ratio={config.data.test_ratio}")
        parts.append(f"val_ratio={config.data.val_ratio}")
    parts.append(f"train={len(split.train)}")
    if split.val is not None and len(split.val) > 0:
        parts.append(f"val={len(split.val)}")
    parts.append(f"test={len(split.test)}")
    print(f"[Data]  {'  '.join(parts)}", flush=True)


def _print_train_stage(lexicon_prefix: Path, train_s: float, *, reused: bool) -> None:
    """Print a one-line summary after training (or reuse)."""
    timing = format_hh_mm_ss(train_s)
    if reused:
        print(f"[Train] reused existing model  ({timing})", flush=True)
    else:
        print(f"[Train] done  ({timing})  lexicons -> {lexicon_prefix}-top-*.txt", flush=True)


def _print_eval_stage(result: EvalResult, eval_s: float, corpora: dict) -> None:
    """Print a one-line summary after evaluation."""
    sets = ",".join(corpora.keys())
    timing = format_hh_mm_ss(eval_s)
    line = f"[Eval]  done  ({timing})  sets={sets}"
    test_metrics = result.get(1, "test") or (
        result.get(result.top_ns()[0], "test") if result.top_ns() else None
    )
    if test_metrics is not None:
        line += f"  test_F1={test_metrics.f1:.2f}"
    print(line, flush=True)


def _run_one(
    config: InductionConfig,
    split: CorpusSplit,
    work_dir: Path,
    *,
    n_folds: int | None = None,
) -> EvalResult:
    """Train and evaluate a single split under *work_dir*."""
    _print_data_stage(config, split, n_folds=n_folds)
    if config.data.save_splits:
        save_split_corpora(split, work_dir / "data")

    seed_grammar = Path(config.model.seed_grammar)
    previous = _resolve_previous_model(config)
    models_dir = work_dir / "models"
    lexicon_prefix, train_s = train_model(
        train_corpus=split.train,
        seed_grammar=seed_grammar,
        model_dir=models_dir,
        top_n=config.model.top_n,
        previous_model=previous,
        show_progress=config.train.show_progress,
        force_train=config.train.force_train,
        reuse_existing_model=config.train.reuse_existing_model,
        corpus_profile=config.data.profile,
    )
    reused = train_s == 0.0 and config.train.reuse_existing_model
    _print_train_stage(lexicon_prefix, train_s, reused=reused)

    corpora = _corpora_for_eval(split, config.eval.evaluate_on)
    result, eval_s = evaluate_model_dir(
        seed_grammar=seed_grammar,
        lexicon_prefix=lexicon_prefix,
        corpora=corpora,
        top_n_start=config.eval.top_n_start,
        top_n_end=config.resolved_top_n_end(),
        staging_root=work_dir / "_eval_staging",
    )
    result.metadata["train_time_s"] = train_s
    result.metadata["eval_time_s"] = eval_s
    result.metadata["train_time"] = format_hh_mm_ss(train_s)
    result.metadata["eval_time"] = format_hh_mm_ss(eval_s)
    _print_eval_stage(result, eval_s, corpora)
    return result


class TrainEvalRunner:
    """Run a full train/evaluate induction from an :class:`InductionConfig`."""

    def __init__(self, config: InductionConfig) -> None:
        """Store *config* for :meth:`run`."""
        self.config = config

    def run(self, *, report_tui: bool = False) -> EvalResult:
        """Execute the configured pipeline and return metrics."""
        set_active_profile(self.config.data.profile)
        run_dir = _make_run_dir(self.config)
        configure_induction_logging(self.config.logging, run_dir=run_dir)
        dump_config(self.config, run_dir / "run_config.yaml")
        logger.info("Run directory: {}", run_dir)
        logger.info("Induction corpus profile: {}", self.config.data.profile)
        _print_run_stage(self.config, run_dir)

        mode = self.config.data.split
        if mode == "pre_split":
            if not self.config.data.train or not self.config.data.test:
                raise ValueError("pre_split requires data.train and data.test paths")
            split = load_pre_split(
                train_path=self.config.data.train,
                test_path=self.config.data.test,
                val_path=self.config.data.val,
            )
            result = _run_one(self.config, split, run_dir)
        elif mode == "kfold":
            corpus = _load_source_corpus(self.config)
            folds = kfold_splits(
                corpus,
                folds=self.config.data.folds,
                seed=self.config.data.seed,
            )
            fold_results: list[EvalResult] = []
            total_train = 0.0
            total_eval = 0.0
            for fold in folds:
                assert fold.fold_index is not None
                fold_dir = run_dir / f"fold_{fold.fold_index}"
                logger.info("=== Fold {} / {} ===", fold.fold_index + 1, len(folds))
                fold_result = _run_one(
                    self.config,
                    fold,
                    fold_dir,
                    n_folds=len(folds),
                )
                total_train += float(fold_result.metadata.get("train_time_s", 0.0))
                total_eval += float(fold_result.metadata.get("eval_time_s", 0.0))
                if self.config.output.write_tsv:
                    write_metrics_tsv(fold_result, fold_dir / "eval-scores.tsv")
                if self.config.output.write_report:
                    write_report_file(fold_result, self.config, fold_dir / "full_run_report.txt")
                fold_results.append(fold_result)
            bag = EvalResult(fold_results=fold_results, metadata={"split": "kfold"})
            result = bag.mean_over_folds()
            result.metadata["train_time_s"] = total_train
            result.metadata["eval_time_s"] = total_eval
            result.metadata["train_time"] = format_hh_mm_ss(total_train)
            result.metadata["eval_time"] = format_hh_mm_ss(total_eval)
        elif mode == "train_val_test":
            corpus = _load_source_corpus(self.config)
            split = train_val_test_split(
                corpus,
                test_ratio=self.config.data.test_ratio,
                val_ratio=self.config.data.val_ratio,
                seed=self.config.data.seed,
            )
            result = _run_one(self.config, split, run_dir)
        elif mode == "holdout":
            corpus = _load_source_corpus(self.config)
            split = holdout_split(
                corpus,
                test_ratio=self.config.data.test_ratio,
                seed=self.config.data.seed,
            )
            result = _run_one(self.config, split, run_dir)
        else:
            raise ValueError(f"Unknown data.split: {mode!r}")

        result.metadata.setdefault("run_dir", str(run_dir))
        result.metadata.setdefault("split", mode)
        result.metadata.setdefault("seed", self.config.data.seed)
        result.metadata["git_commit"] = _git_commit()
        result.metadata.setdefault("dataset_sizes", dict(result.dataset_sizes))

        if self.config.output.write_tsv:
            write_metrics_tsv(result, run_dir / "eval-scores.tsv")
            logger.info("Wrote {}", run_dir / "eval-scores.tsv")

        if self.config.output.write_report:
            write_report_file(result, self.config, run_dir / "full_run_report.txt")
            print_report(result, self.config)
            logger.info("Wrote {}", run_dir / "full_run_report.txt")
            if report_tui:
                launch_report_tui(result, self.config)

        return result


def run_induction(
    config: InductionConfig,
    *,
    report_tui: bool = False,
) -> EvalResult:
    """Convenience wrapper around :class:`TrainEvalRunner`."""
    return TrainEvalRunner(config).run(report_tui=report_tui)


# Backward-compatible alias
run_experiment = run_induction
