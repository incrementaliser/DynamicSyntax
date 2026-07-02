"""GS2013 disambiguation dataset loading and KS-format conversion."""

from __future__ import annotations

import csv
from pathlib import Path

from dylan.vss.types import GS2013Pair, GS2013Sentence

_VSS_DIR = Path(__file__).resolve().parent
DEFAULT_GS2013_RAW = _VSS_DIR / "GS2013data.txt"
DEFAULT_GS2013_KS = _VSS_DIR / "GS2013data-KSformat.txt"
_PAIR_KEYS = ("adj_subj", "subj", "landmark", "adj_obj", "obj")


def convert_gs2013_to_ks_format(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Convert raw GS2013 TSV to KS-format rows (ported from fix.py)."""
    inp = input_path or DEFAULT_GS2013_RAW
    out = output_path or DEFAULT_GS2013_KS
    datas: list[dict[str, str]] = []
    sentences: dict[str, int] = {}
    with inp.open(encoding="utf-8") as fstr:
        for line in csv.DictReader(fstr, delimiter=" "):
            sent = line["verb"] + line["subject"] + line["object"] + line["landmark"]
            if sent in sentences:
                sid = sentences[sent]
            else:
                sid = len(sentences) + 1
                sentences[sent] = sid
            datas.append(
                {
                    "sentence_id": str(sid),
                    "annotator_id": line["participant"].replace("participant", ""),
                    "adj_subj": "dummy",
                    "adj_obj": "dummy",
                    "subj": line["subject"],
                    "obj": line["object"],
                    "landmark": line["verb"],
                    "verb": line["landmark"],
                    "annotator_score": line["input"],
                }
            )
    datas.sort(
        key=lambda x: (
            int(x["annotator_id"]),
            int(x["sentence_id"]),
            x["subj"],
            x["obj"],
        )
    )
    fieldnames = [
        "sentence_id",
        "annotator_id",
        "adj_subj",
        "subj",
        "landmark",
        "verb",
        "adj_obj",
        "obj",
        "annotator_score",
    ]
    with out.open("w", encoding="utf-8", newline="") as fstr:
        wr = csv.DictWriter(fstr, fieldnames=fieldnames, delimiter=" ")
        wr.writeheader()
        for data in datas:
            wr.writerow(data)
    return out


def load_gs2013_ks_rows(path: Path | None = None) -> list[GS2013Sentence]:
    """Load KS-format GS2013 annotations aggregated by sentence_id."""
    data_path = path or DEFAULT_GS2013_KS
    if not data_path.is_file():
        convert_gs2013_to_ks_format(output_path=data_path)
    sentence_data: list[GS2013Sentence] = []
    sentence_id: str | None = None
    scores: list[float] = []
    fields: dict[str, str] = {}
    with data_path.open(encoding="utf-8") as fstr:
        for line in csv.DictReader(fstr, delimiter=" "):
            if sentence_id is None or line["sentence_id"] != sentence_id:
                if sentence_id is not None:
                    sentence_data.append(
                        GS2013Sentence(
                            sentence_id=sentence_id,
                            subj=fields["subj"],
                            landmark=fields["landmark"],
                            verb=fields["verb"],
                            obj=fields["obj"],
                            adj_subj=fields.get("adj_subj", "dummy"),
                            adj_obj=fields.get("adj_obj", "dummy"),
                            similarity_scores=tuple(scores),
                        )
                    )
                sentence_id = line["sentence_id"]
                fields = {k: line[k] for k in line if k not in ("annotator_id", "annotator_score")}
                scores = []
            scores.append(float(line["annotator_score"]))
    if sentence_id is not None:
        sentence_data.append(
            GS2013Sentence(
                sentence_id=sentence_id,
                subj=fields["subj"],
                landmark=fields["landmark"],
                verb=fields["verb"],
                obj=fields["obj"],
                adj_subj=fields.get("adj_subj", "dummy"),
                adj_obj=fields.get("adj_obj", "dummy"),
                similarity_scores=tuple(scores),
            )
        )
    return sentence_data


def load_sentence_pairs(path: Path | None = None) -> list[GS2013Pair]:
    """Pair consecutive KS rows sharing context; gold category from mean similarity (jolli)."""
    rows = load_gs2013_ks_rows(path)
    pairs: list[GS2013Pair] = []
    last: GS2013Sentence | None = None
    for sd in rows:
        if last is None:
            last = sd
            continue
        matching = all(getattr(sd, k) == getattr(last, k) for k in _PAIR_KEYS)
        if matching:
            m0 = sum(last.similarity_scores) / max(len(last.similarity_scores), 1)
            m1 = sum(sd.similarity_scores) / max(len(sd.similarity_scores), 1)
            gold = 0 if m0 > m1 else 1
            pairs.append(GS2013Pair(first=last, second=sd, gold_category=gold))
            last = None
        else:
            last = sd
    return pairs


def candidate_sets_for_landmark(
    pairs: list[GS2013Pair],
    landmark: str,
) -> tuple[set[str], set[str], set[str]]:
    """Return subject, verb, and object lemma sets for all pairs with the same landmark."""
    ss: set[str] = set()
    vs: set[str] = set()
    os: set[str] = set()
    for pair in pairs:
        if pair.first.landmark != landmark:
            continue
        for sent in (pair.first, pair.second):
            ss.add(sent.subj)
            vs.add(sent.verb)
            os.add(sent.obj)
    return ss, vs, os
