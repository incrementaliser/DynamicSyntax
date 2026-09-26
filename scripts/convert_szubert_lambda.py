"""Convert published Adam logical forms to TTR.

The forms are the UD-to-lambda output in ``adam.all_lf.txt`` (Szubert et al.,
via ``Lou1sM/CHILDES_UD2LF_2``). Each block records the CoNLL split file and
sample index the sentence came from.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from dylan.induction.em_learner.lambda_ttr_converter import LambdaTTRConversionError
from dylan.induction.em_learner.szubert_lambda import (
    align_lf_to_conll,
    convert_szubert_lambda,
    iter_conll_sentences,
    iter_szubert_pairs,
    load_comparison_utterances,
    normalize_szubert_lambda,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse paths for the Adam lambda-to-TTR batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lf", type=Path, required=True, help="adam.all_lf.txt")
    parser.add_argument("--conll", type=Path, required=True, help="Directory of adamN.conll.txt")
    parser.add_argument("--easy", type=Path, required=True, help="Adam.all_easy_lf.txt comparison list")
    parser.add_argument("--output", type=Path, required=True, help="TTR blocks to write")
    parser.add_argument("--failures", type=Path, required=True, help="Rejected blocks")
    parser.add_argument("--report", type=Path, required=True, help="Coverage summary")
    parser.add_argument(
        "--zip",
        type=Path,
        default=None,
        help="Optional Eng-NA MOR zip, used when a sentence matches one Brown Adam line",
    )
    return parser.parse_args(argv)


def _brown_index(zip_path: Path | None) -> dict[str, list[str]]:
    """Map a normalized utterance to Brown Adam ``session.cha`` locations.

    :param zip_path: Eng-NA MOR zip, or ``None`` to skip.
    :returns: Utterance text to a list of ``path index`` strings. Adult lines only.
    """
    if zip_path is None or not zip_path.exists():
        return {}
    import zipfile

    found: dict[str, list[str]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        names = [
            name
            for name in archive.namelist()
            if "/Brown/Adam/" in name and name.endswith(".cha")
        ]
        for name in names:
            sample = 0
            for raw in archive.read(name).decode("utf-8", errors="replace").splitlines():
                if not raw.startswith("*") or raw.startswith("*CHI"):
                    continue
                sample += 1
                utterance = raw.split(":", 1)[1]
                text = " ".join(utterance.replace(".", " ").replace("?", " ").replace("!", " ").split())
                text = text.lower()
                location = f"{name} {sample}"
                found.setdefault(text, []).append(location)
    return found


def _location(filename: str | None, sample: int, lf_index: int) -> str:
    """Format the source comment's file and sample number."""
    if filename is None:
        return f"adam.all_lf.txt {lf_index}"
    return f"{filename} {sample}"


def main(argv: list[str] | None = None) -> None:
    """Normalize Adam logical forms, convert them, and write source comments."""
    args = _parse_args(argv)
    pairs = iter_szubert_pairs(args.lf)
    sentences = iter_conll_sentences(args.conll)
    located = align_lf_to_conll(pairs, sentences)
    comparison = load_comparison_utterances(args.easy)
    brown = _brown_index(args.zip)
    converted_lines: list[str] = []
    failure_lines: list[str] = []
    reasons: Counter[str] = Counter()
    converted = 0
    compared = 0
    compared_ok = 0
    aligned = 0
    for pair, place in zip(pairs, located, strict=True):
        utterance_key = " ".join(pair.utterance.split())
        in_paper = utterance_key in comparison
        if in_paper:
            compared += 1
        if place is None:
            comment = pair.source_comment(
                _location(None, 0, pair.sample_index), in_paper
            )
        else:
            aligned += 1
            comment = pair.source_comment(
                _location(place.filename, place.sample_index, pair.sample_index),
                in_paper,
            )
            brown_key = " ".join(
                pair.utterance.replace("?", " ").replace("!", " ").replace(".", " ").split()
            ).lower()
            hits = brown.get(brown_key, [])
            if len(hits) == 1:
                comment = f"{comment} {hits[0]}"
        try:
            normalized = normalize_szubert_lambda(pair.semantics)
            ttr = convert_szubert_lambda(pair.semantics, pair.utterance.rstrip(" .?!"))
        except (LambdaTTRConversionError, Exception) as exc:  # noqa: BLE001
            reason = str(exc).split(" for ")[0][:160]
            reasons[reason] += 1
            failure_lines.append(
                f"Sent : {pair.utterance} // {comment}\n"
                f"Sem : {pair.semantics} // {comment}\n"
                f"Error : {reason}\n"
            )
            continue
        converted += 1
        if in_paper:
            compared_ok += 1
        converted_lines.append(
            f"Sent : {pair.utterance} // {comment}\n"
            f"Sem : {normalized} // {comment}\n"
            f"TTR : {ttr}\n"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(converted_lines), encoding="utf-8")
    args.failures.parent.mkdir(parents=True, exist_ok=True)
    args.failures.write_text("\n".join(failure_lines), encoding="utf-8")
    report = [
        "Adam logical forms to TTR",
        f"source_lf={len(pairs)}",
        f"conll_sentences={len(sentences)}",
        f"aligned={aligned}",
        f"converted={converted}",
        f"failed={len(pairs) - converted}",
        f"comparison_utterances_in_easy_file={len(comparison)}",
        f"comparison_rows={compared}",
        f"comparison_converted={compared_ok}",
        "The papers report 5320 Adam utterances after dropping communicator words.",
        "The public learner file Adam.all_easy_lf.txt has "
        f"{len(comparison)} distinct sentences; that list is the comparison flag.",
        "Higher-order adjuncts written as $n($m) are removed before conversion.",
        "A wh-adjunct with no lexical preposition is therefore absent from the TTR.",
        "Top failure reasons:",
    ]
    for reason, count in reasons.most_common(15):
        report.append(f"  {count}\t{reason}")
    args.report.write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report[:12]))


if __name__ == "__main__":
    main()
