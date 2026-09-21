"""Encode legal CandidateSequence support as discrete word-hypothesis variables."""

from __future__ import annotations

from dataclasses import dataclass, field

from dylan.induction.em_learner.candidate_sequence import CandidateSequence
from dylan.induction.em_learner.common import Word, words_to_string
from dylan.induction.em_learner.word_hypothesis_base import WordHypothesisBase


@dataclass
class EncodedExample:
    """One training utterance with legal hyp-id tuples and word alignment.

    :param sentence: surface string.
    :param words: per-position :class:`Word` tokens.
    :param legal_tuples: each tuple is a list of ``hyp_id`` aligned to words.
    :param word_hyp_ids: map ``word_text -> sorted hyp_ids`` seen in this example.
    """

    sentence: str
    words: list[Word]
    legal_tuples: list[list[int]]
    word_hyp_ids: dict[str, list[int]] = field(default_factory=dict)


def encode_hypothesis_base(hb: WordHypothesisBase, words: list[Word]) -> EncodedExample:
    """Snapshot the current example's legal tuples from *hb* as hyp-id matrices.

    Call after :meth:`WordHypothesisBase.add_sequence_tuples` (and before or
    after EM — we only need the discrete support structure).
    """
    legal: list[list[int]] = []
    word_hyp_ids: dict[str, set[int]] = {}
    for row in hb.get_hypothesis_tuples():
        ids: list[int] = []
        for hyp in row:
            ids.append(int(hyp.hyp_id))
            try:
                key = hyp.get_word().word()
            except Exception:  # noqa: BLE001
                key = ""
            if key:
                word_hyp_ids.setdefault(key, set()).add(int(hyp.hyp_id))
        if ids:
            legal.append(ids)
    return EncodedExample(
        sentence=words_to_string(words),
        words=list(words),
        legal_tuples=legal,
        word_hyp_ids={k: sorted(v) for k, v in word_hyp_ids.items()},
    )


def ingest_sequences(
    hb: WordHypothesisBase,
    sequences: list[CandidateSequence],
    words: list[Word],
) -> EncodedExample:
    """Forget current dist, ingest *sequences*, return an :class:`EncodedExample`."""
    hb.forget_current_dist()
    for cs in sequences:
        hb.add_sequence_tuples(cs.split())
    return encode_hypothesis_base(hb, words)


def hyp_key(word: str, hyp_id: int) -> str:
    """Stable string key for a (word, hyp_id) pair in global parameter tables."""
    return f"{word}::{hyp_id}"


def collect_global_hyp_catalog(
    examples: list[EncodedExample],
) -> dict[str, list[int]]:
    """Union of hyp ids per surface word across *examples*."""
    catalog: dict[str, set[int]] = {}
    for ex in examples:
        for w, ids in ex.word_hyp_ids.items():
            catalog.setdefault(w, set()).update(ids)
    return {w: sorted(ids) for w, ids in catalog.items()}
