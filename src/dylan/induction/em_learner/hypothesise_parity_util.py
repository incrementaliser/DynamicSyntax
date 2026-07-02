"""Utilities to compare TTR hypothesiser output with Java reference sequences."""

from __future__ import annotations

import re
from pathlib import Path

from dylan.induction.em_learner.candidate_sequence import CandidateSequence


def _canonicalise_hyp_sem_action(action: str) -> str:
    """Parse embedded TTR records in ``hyp-sem([...])`` for stable field order and spacing."""
    if not action.startswith("hyp-sem(") or not action.endswith(")"):
        return action
    body = action[len("hyp-sem(") : -1]
    if body.startswith("R1^") or "++" in body:
        return f"hyp-sem({body})"
    try:
        from dylan.formula.ttr_record_type import TTRRecordType

        rt = TTRRecordType.parse(body)
        if rt is not None:
            from dylan.formula.ttr_field import TTRField

            ordered = TTRRecordType()
            for field in sorted(rt.get_fields(), key=lambda f: str(f.label)):
                ordered.add_field(field.clone() if hasattr(field, "clone") else field)  # type: ignore[arg-type]
            return f"hyp-sem({ordered})"
    except Exception:  # noqa: BLE001
        pass
    return action


def _split_sequence_actions(line: str) -> list[str]:
    """Split on top-level ``|`` only (not inside ``[...]`` record literals)."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in line:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        elif ch == "|" and depth == 0:
            if cur:
                parts.append("".join(cur).strip())
                cur = []
            continue
        cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p]


def normalise_sequence_line(line: str) -> str:
    """Normalise whitespace around pipes and inside ``hyp-sem(...)`` record literals."""
    line = re.sub(r"\s*,\s*", ",", line)
    parts = [_canonicalise_hyp_sem_action(p) for p in _split_sequence_actions(line)]
    return "|".join(parts)


def load_reference_sequences(path: Path) -> list[str]:
    """Load pipe-separated reference sequences from ``hypotheses-example.txt`` style files."""
    text = path.read_text(encoding="utf-8")
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("Got") or line.startswith("---"):
            continue
        if "Incorrect" in line or line.lower().startswith("now splitting"):
            break
        if "|" in line:
            out.append(normalise_sequence_line(line))
    return out


def sequences_from_hypothesiser(hyps: list[CandidateSequence]) -> list[str]:
    """Convert hypothesiser output to normalised ``to_short_string`` lines."""
    return [normalise_sequence_line(cs.to_short_string()) for cs in hyps]


def sequences_match_reference(
    got: list[str],
    ref: list[str],
) -> tuple[bool, list[str]]:
    """Return whether *got* matches *ref* as a set, plus missing reference lines."""
    got_set = set(got)
    missing = [r for r in ref if r not in got_set]
    return len(missing) == 0 and len(got_set) == len(ref), missing
