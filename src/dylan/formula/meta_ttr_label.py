"""Meta TTR label (Java `MetaTTRLabel` extends `TTRLabel`)."""

from __future__ import annotations

from dataclasses import dataclass

from dylan.formula.ttr_label import TTRLabel


@dataclass(frozen=True, slots=True)
class MetaTTRLabel(TTRLabel):
    """Meta-level TTR field label (partial port)."""
