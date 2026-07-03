"""CLI entry point for full EM induction (TTRWordLearner) on a Java-format corpus file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dylan.induction.em_learner.record_type_corpus import RecordTypeCorpus
from dylan.induction.em_learner.ttr_word_learner import TTRWordLearner

DEFAULT_GRAMMAR = Path("resources") / "2025-seed-grammar"
DEFAULT_TOP_N = 3


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for EM induction."""
    parser = argparse.ArgumentParser(
        description="Run full EM induction on a TTR corpus and write lexicon files.",
    )
    parser.add_argument(
        "corpus",
        type=Path,
        help="Path to input corpus (GoldSent/Sent + Sem blocks, blank line between examples)",
    )
    parser.add_argument(
        "out",
        type=Path,
        help="Output path prefix; writes {out}-top-1.txt … {out}-top-{top_n}.txt",
    )
    parser.add_argument(
        "--grammar",
        type=Path,
        default=DEFAULT_GRAMMAR,
        help=f"Seed grammar directory containing computational-actions.txt (default: {DEFAULT_GRAMMAR})",
    )
    parser.add_argument(
        "--top-n",
        "--top_n",
        type=int,
        default=DEFAULT_TOP_N,
        dest="top_n",
        metavar="N",
        help=f"Number of top lexicon ranks to save (default: {DEFAULT_TOP_N})",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the corpus progress bar during learning",
    )
    return parser.parse_args(argv)


def run_em_learn(
    corpus_path: Path,
    out_prefix: Path,
    *,
    grammar: Path = DEFAULT_GRAMMAR,
    top_n: int = DEFAULT_TOP_N,
    show_progress: bool = True,
) -> list[Path]:
    """Load *corpus_path*, run EM learning, and save lexicon files at *out_prefix*."""
    if not corpus_path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    comp_actions = grammar / "computational-actions.txt"
    if not comp_actions.is_file():
        raise FileNotFoundError(f"Grammar missing computational-actions.txt: {grammar}")
    if top_n < 1:
        raise ValueError(f"--top-n must be >= 1, got {top_n}")

    corpus = RecordTypeCorpus(corpus_path=corpus_path)
    if len(corpus) == 0:
        raise ValueError(f"Corpus is empty or could not be parsed: {corpus_path}")

    learner = TTRWordLearner(
        seed_resource_dir=None,
        corpus=corpus,
        learner_comp_actions_path=grammar,
        top_n=top_n,
        load_learnt_lexicon=False,
    )
    learner.learn(show_progress=show_progress)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    learner.save_model(out_prefix, top_n=top_n)

    return [Path(f"{out_prefix}-top-{n}.txt") for n in range(1, top_n + 1)]


def main(argv: list[str] | None = None) -> int:
    """Run EM induction from the command line."""
    args = _parse_args(argv)
    try:
        written = run_em_learn(
            args.corpus,
            args.out,
            grammar=args.grammar,
            top_n=args.top_n,
            show_progress=not args.no_progress,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
