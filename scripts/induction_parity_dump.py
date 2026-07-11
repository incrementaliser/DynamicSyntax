"""Differential dumps for induction IF/THEN parity (hypothesiser sequences + lexicon cores).

Usage::

    uv run python scripts/induction_parity_dump.py --corpus data/induction-test/one.txt
    uv run python scripts/induction_parity_dump.py --corpus data/induction-test/class1.txt --learn
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from dylan.induction.em_learner.hypothesise_parity_util import normalise_sequence_line
from dylan.induction.em_learner.record_type_corpus import RecordTypeCorpus
from dylan.induction.em_learner.ttr_hypothesiser import TTRHypothesiser
from dylan.induction.em_learner.ttr_word_learner import TTRWordLearner


def _repo_root() -> Path:
    """Return the dynamicsyntax repository root."""
    return Path(__file__).resolve().parents[1]


def dump_hypothesis_multisets(
    corpus_path: Path,
    seed_grammar: Path,
    out_dir: Path,
) -> None:
    """Write per-example hypothesiser sequence multiset dumps under *out_dir*."""
    corpus = RecordTypeCorpus(corpus_path=corpus_path)
    hyp = TTRHypothesiser(str(seed_grammar))
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (sent, sem) in enumerate(corpus):
        words = [str(w) for w in sent]
        hyp.load_training_example(words, sem)
        sequences = hyp.hypothesise()
        counts = Counter(normalise_sequence_line(s.to_short_string()) for s in sequences)
        dest = out_dir / f"example-{i:03d}-{' '.join(words)}.txt".replace(" ", "_")
        lines = [f"# words: {' '.join(words)}", f"# n_sequences={len(sequences)} unique={len(counts)}"]
        for sig, n in sorted(counts.items()):
            lines.append(f"{n}\t{sig}")
        dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {dest} ({len(sequences)} sequences)")


def _strip_prob_rank(text: str) -> str:
    """Drop ``[prob,rank]`` lines so IF/THEN bodies can be compared."""
    out: list[str] = []
    for line in text.splitlines():
        if re.fullmatch(r"\[[0-9.eE+-]+,\s*\d+\]", line.strip()):
            continue
        out.append(line)
    return "\n".join(out).strip() + "\n"


def dump_learned_lexicon(
    corpus_path: Path,
    seed_grammar: Path,
    out_prefix: Path,
    top_n: int,
) -> None:
    """Run ``TTRWordLearner`` and write ``{out_prefix}-top-{k}.txt`` plus IF/THEN-only views."""
    corpus = RecordTypeCorpus(corpus_path=corpus_path)
    learner = TTRWordLearner(str(seed_grammar), corpus)
    learner.learn()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    learner.save_model(str(out_prefix), top_n)
    for k in range(1, top_n + 1):
        path = Path(f"{out_prefix}-top-{k}.txt")
        if not path.is_file():
            alt = out_prefix.parent / f"{out_prefix.name}-top-{k}.txt"
            path = alt if alt.is_file() else path
        if path.is_file():
            body = _strip_prob_rank(path.read_text(encoding="utf-8"))
            body_path = path.with_name(path.stem + "-ifthen-only.txt")
            body_path.write_text(body, encoding="utf-8")
            print(f"wrote {path} and {body_path}")


def compare_ifthen(python_path: Path, java_path: Path) -> int:
    """Print a simple word-keyed IF/THEN body diff; return number of mismatched words."""
    py = _parse_lexicon_bodies(python_path)
    jv = _parse_lexicon_bodies(java_path)
    mismatches = 0
    for word in sorted(set(py) | set(jv)):
        a = py.get(word, [])
        b = jv.get(word, [])
        if a != b:
            mismatches += 1
            print(f"MISMATCH word={word!r} py_n={len(a)} java_n={len(b)}")
            if a and b and a[0] != b[0]:
                print("--- python top ---")
                print(a[0][:500])
                print("--- java top ---")
                print(b[0][:500])
    print(f"mismatched_words={mismatches} python_words={len(py)} java_words={len(jv)}")
    return mismatches


def _parse_lexicon_bodies(path: Path) -> dict[str, list[str]]:
    """Parse lexicon text into word -> list of IF/THEN bodies (prob lines stripped)."""
    text = path.read_text(encoding="utf-8")
    entries: dict[str, list[str]] = {}
    chunks = re.split(r"\n\s*\n", text.strip())
    for chunk in chunks:
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        if re.fullmatch(r"\[[0-9.eE+-]+,\s*\d+\]", lines[0].strip()):
            word = lines[1].strip()
            body = "\n".join(lines[2:]).strip()
        else:
            word = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
        entries.setdefault(word, []).append(body)
    return entries


def main() -> None:
    """CLI entry point for induction parity dumps."""
    root = _repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=root / "data/induction-test/one.txt")
    p.add_argument("--seed-grammar", type=Path, default=root / "resources/2025-seed-grammar")
    p.add_argument("--out-dir", type=Path, default=root / "out/parity-dumps")
    p.add_argument("--learn", action="store_true", help="Also run EM and dump lexicon-top-N")
    p.add_argument("--top-n", type=int, default=3)
    p.add_argument("--compare-java", type=Path, default=None, help="Java lexicon.lex-top-N.txt to compare")
    args = p.parse_args()

    dump_hypothesis_multisets(args.corpus, args.seed_grammar, args.out_dir / "sequences")
    if args.learn:
        dump_learned_lexicon(
            args.corpus,
            args.seed_grammar,
            args.out_dir / "lexicon",
            args.top_n,
        )
    if args.compare_java is not None:
        py_lex = args.out_dir / "lexicon-top-1.txt"
        if not py_lex.is_file():
            py_lex = args.out_dir / "lexicon-top-1-ifthen-only.txt"
        compare_ifthen(py_lex, args.compare_java)


if __name__ == "__main__":
    main()
