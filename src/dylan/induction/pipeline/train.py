"""Training wrapper around :class:`~dylan.induction.em_learner.ttr_word_learner.TTRWordLearner`."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from loguru import logger

from dylan.induction.em_learner.record_type_corpus import RecordTypeCorpus
from dylan.induction.em_learner.ttr_word_learner import TTRWordLearner


def _find_lexicon_file(model_dir: Path, top_n: int) -> Path | None:
    """Locate a saved lexicon-top-N file directly under *model_dir* (no nested search)."""
    candidates = [
        model_dir / f"lexicon-top-{top_n}.txt",
        Path(f"{model_dir / 'lexicon'}-top-{top_n}.txt"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(model_dir.glob("lexicon-top-*.txt"))
    if matches:
        return matches[-1]
    return None


def prepare_previous_model_seed(
    previous_model: Path,
    *,
    top_n: int,
    staging_dir: Path,
) -> Path:
    """Copy a prior learnt lexicon into *staging_dir* as ``lexicon-top-{top_n}.txt`` for continue learning.

    *previous_model* must be the directory that directly contains ``lexicon-top-N.txt``
    (for new runs, typically ``.../models/``). Nested directories are not searched.
    """
    src = _find_lexicon_file(previous_model, top_n)
    if src is None:
        raise FileNotFoundError(
            f"No lexicon-top-*.txt found under previous_model={previous_model}",
        )
    staging_dir.mkdir(parents=True, exist_ok=True)
    dest = staging_dir / f"lexicon-top-{top_n}.txt"
    shutil.copy2(src, dest)
    logger.info("Continue learning from previous lexicon {}", src)
    return staging_dir


def lexicon_files_exist(model_dir: Path, top_n: int) -> bool:
    """Return whether lexicon files for ranks 1..*top_n* already exist."""
    prefix = model_dir / "lexicon"
    return all(Path(f"{prefix}-top-{n}.txt").is_file() for n in range(1, top_n + 1))


def train_model(
    *,
    train_corpus: RecordTypeCorpus,
    seed_grammar: Path,
    model_dir: Path,
    top_n: int = 3,
    previous_model: Path | None = None,
    show_progress: bool = True,
    force_train: bool = False,
    reuse_existing_model: bool = False,
    corpus_profile: str | None = None,
) -> tuple[Path, float]:
    """Train (or reuse) and save ``lexicon-top-N.txt`` under *model_dir*.

    Returns ``(lexicon_prefix, train_seconds)``.

    *previous_model* is an optional seed lexicon in another directory (continue
    learning). It is independent of whether *model_dir* already has outputs.

    *corpus_profile* when set activates the named induction corpus profile
    (``babyds`` / ``childes``) before learning.

    Precedence: ``force_train`` always trains/overwrites. ``reuse_existing_model``
    skips EM only when ``force_train`` is false and current-dir lexicons exist.
    """
    if corpus_profile is not None:
        from dylan.induction.corpus_profile import set_active_profile

        set_active_profile(corpus_profile)

    model_dir.mkdir(parents=True, exist_ok=True)
    comp_src = seed_grammar / "computational-actions.txt"
    if not comp_src.is_file():
        raise FileNotFoundError(f"Grammar missing computational-actions.txt: {seed_grammar}")

    lexicon_prefix = model_dir / "lexicon"
    exists = lexicon_files_exist(model_dir, top_n)

    if exists and reuse_existing_model and not force_train:
        logger.info("Reusing existing lexicons in {} (reuse_existing_model=true)", model_dir)
        return lexicon_prefix, 0.0

    if exists and not force_train:
        logger.warning(
            "Lexicon files already exist in {}; training will overwrite them "
            "(set train.force_train=true to silence this warning)",
            model_dir,
        )

    seed_resource: Path | None = None
    if previous_model is not None:
        seed_resource = prepare_previous_model_seed(
            Path(previous_model),
            top_n=top_n,
            staging_dir=model_dir / "_previous_seed",
        )

    t0 = time.perf_counter()
    learner = TTRWordLearner(
        seed_resource_dir=seed_resource,
        corpus=train_corpus,
        learner_comp_actions_path=seed_grammar,
        top_n=top_n,
        load_learnt_lexicon=seed_resource is not None,
    )
    learner.learn(show_progress=show_progress)
    learner.save_model(lexicon_prefix, top_n=top_n)
    elapsed = time.perf_counter() - t0
    if not lexicon_files_exist(model_dir, top_n):
        raise RuntimeError(f"Training finished but no lexicon-top-*.txt under {model_dir}")
    # Empty files (all examples skipped) break eval with a confusing 0-word load.
    top1 = model_dir / "lexicon-top-1.txt"
    if top1.is_file() and top1.stat().st_size == 0:
        skipped = len(getattr(learner, "skipped", []) or [])
        raise RuntimeError(
            f"Learnt lexicon is empty after training ({skipped} examples skipped). "
            "Check induction corpus profile, seed grammar, and that gold RTs have "
            "(or can deem) a head field for TypeLattice.",
        )
    logger.info("Saved lexicons under {}-top-*.txt (train {})", lexicon_prefix, elapsed)
    return lexicon_prefix, elapsed
