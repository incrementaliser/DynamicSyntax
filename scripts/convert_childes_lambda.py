"""Convert Eve ``trainPairs`` lambda formulae to TTR record types.

The rules are the Java ``qmul.ds.learn.CorpusConverter`` port in
``dylan.induction.em_learner.lambda_ttr_converter``. Raw CHAT transcripts
are not accepted: each example needs a ``Sem:`` lambda line.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dylan.induction.em_learner.lambda_ttr_converter import (
    LambdaTTRConversionError,
    convert_lambda,
    convert_train_pair_folder,
    iter_train_pairs,
    repair_lambda,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for a lambda-to-TTR batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        type=Path,
        help="Directory of trainPairs_1 … trainPairs_20",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Where to write Sent/Sem/File TTR blocks",
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=None,
        help="Optional path for utterances the existing rules reject",
    )
    parser.add_argument(
        "--repairs",
        type=Path,
        default=None,
        help="Optional path for ill-formed formulae that were repaired",
    )
    return parser.parse_args(argv)


def write_failures(folder: Path, path: Path) -> int:
    """Rewrite *folder* and store every failure line at *path*.

    :returns: Number of non-commented pairs that did not convert.
    """
    failed = 0
    lines: list[str] = []
    for pair in iter_train_pairs(folder):
        if pair.commented:
            continue
        try:
            convert_lambda(pair.semantics, pair.utterance)
        except LambdaTTRConversionError as exc:
            failed += 1
            lines.append(
                f"Sent : {pair.utterance} // {pair.source_comment()}\n"
                f"Sem : {pair.semantics} // {pair.source_comment()}\n"
                f"File : {pair.file_index}\nError : {exc}\n"
            )
        except Exception as exc:  # noqa: BLE001 — record unexpected gaps in the batch log
            failed += 1
            lines.append(
                f"Sent : {pair.utterance} // {pair.source_comment()}\n"
                f"Sem : {pair.semantics} // {pair.source_comment()}\n"
                f"File : {pair.file_index}\nError : {type(exc).__name__}: {exc}\n"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not lines:
        lines.append("No conversion failures.\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    return failed


def write_repairs(folder: Path, path: Path) -> int:
    """Record ill-formed source formulae that were repaired before conversion.

    :returns: Number of repaired non-commented pairs.
    """
    lines = [
        "Ill-formed Eve formulae repaired before TTR conversion.",
        "truncated-not: not($0,) has no embedded proposition.",
        "unbalanced: missing parenthesis or an empty conjunct.",
        "",
    ]
    count = 0
    for pair in iter_train_pairs(folder):
        if pair.commented:
            continue
        _repaired, tag = repair_lambda(pair.semantics)
        if tag is None:
            continue
        count += 1
        lines.append(
            f"Sent : {pair.utterance} // {pair.source_comment()}\n"
            f"Sem : {pair.semantics} // {pair.source_comment()}\n"
            f"File : {pair.file_index}\nRepair : {tag}\n"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return count


def main(argv: list[str] | None = None) -> None:
    """Convert a trainPairs directory and print source/converted/failed counts."""
    args = _parse_args(argv)
    report = convert_train_pair_folder(args.folder, args.output, failure_limit=80)
    print(report.summary())
    if args.failures is not None:
        print(f"failures_written={write_failures(args.folder, args.failures)}")
    if args.repairs is not None:
        print(f"repairs_written={write_repairs(args.folder, args.repairs)}")


if __name__ == "__main__":
    main()
