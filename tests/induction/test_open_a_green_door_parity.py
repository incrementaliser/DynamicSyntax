"""Integration checks for open-a-green-door induction parity (``z-myfiles/one4/one4.txt``)."""

from __future__ import annotations

from pathlib import Path

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.em_learner.ttr_hypothesiser import TTRHypothesiser
from dylan.induction.em_learner.hypothesise_parity_util import (
    load_reference_sequences,
    sequences_from_hypothesiser,
    sequences_match_reference,
)

_REPO = Path(__file__).resolve().parents[2]
_DATA = _REPO / "z-myfiles" / "one4" / "one4.txt"
_GRAMMAR = _REPO / "resources" / "2025-seed-grammar"
_HYP_REF = _REPO / "data" / "induction-test" / "hypotheses-one4-reference.txt"


def _gold_from_one4_txt() -> TTRRecordType:
    """Parse the gold TTR record from ``z-myfiles/one4/one4.txt``."""
    sem_line = next(l for l in _DATA.read_text(encoding="utf-8").splitlines() if l.strip().startswith("Sem"))
    sem = sem_line.split(":", 1)[1].strip()
    rt = TTRRecordType.parse(sem)
    assert rt is not None
    return rt


def test_ttr_hypothesiser_open_a_green_door_sequences() -> None:
    """End-to-end hypothesiser should yield sixteen sequences with four ``hyp-sem`` actions each."""
    rt = _gold_from_one4_txt()
    hyp = TTRHypothesiser(str(_GRAMMAR))
    hyp.load_training_example("open a green door", rt)
    hyps = hyp.hypothesise()
    got = sequences_from_hypothesiser(hyps)
    ref = load_reference_sequences(_HYP_REF)
    assert len(ref) == 16, "reference file should list exactly sixteen sequences"
    assert all(line.count("hyp-sem") == 4 for line in got), "each sequence must contain four hyp-sem actions"
    ok, missing = sequences_match_reference(got, ref)
    assert len(hyps) == 16, f"expected 16 hypotheses, got {len(hyps)}"
    assert ok, f"missing reference sequences: {missing[:1]}"
