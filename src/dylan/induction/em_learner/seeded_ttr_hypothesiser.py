"""Seeded TTR hypothesiser variant (Java ``qmul.ds.learn.SeededTTRHypothesiser``).

Pure subclass of :class:`TTRHypothesiser` for parity; today it inherits all
behaviour from the base class.
"""

from __future__ import annotations

from pathlib import Path

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.em_learner.ttr_hypothesiser import TTRHypothesiser


class SeededTTRHypothesiser(TTRHypothesiser):
    """Seeded TTR hypothesiser (subclass shim for Java parity)."""

    def __init__(
        self,
        resource_dir_or_url: "str | Path | None" = None,
        rt: TTRRecordType | None = None,
        sent: str | None = None,
        top_n: int = 3,
        load_learnt_lexicon: bool = False,
        learner_comp_actions_path: "str | Path | None" = None,
    ) -> None:
        """Forward all parameters to :class:`TTRHypothesiser`."""
        super().__init__(
            resource_dir_or_url=resource_dir_or_url,
            top_n=top_n,
            load_learnt_lexicon=load_learnt_lexicon,
            learner_comp_actions_path=learner_comp_actions_path,
            rt=rt,
            sent=sent,
        )
