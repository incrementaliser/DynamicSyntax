"""Shared learning protocol for induction algorithms."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Learner(Protocol):
    """Standard training lifecycle implemented by induction learners."""

    def reset(self) -> None:
        """Reset model and corpus state."""

    def load_corpus(self, corpus_file: str | Path) -> None:
        """Load training examples from *corpus_file*."""

    def learn_once(self) -> bool:
        """Process one training example; return ``False`` at end of corpus."""

    def learn(self) -> None:
        """Run the learner until :meth:`learn_once` returns ``False``."""

    def get_hypothesis_base(self) -> Any:
        """Return the learner's hypothesis/probability store."""

    def save_model(self, save_path: str | Path, top_n: int, save_top_n_start: int = 1) -> None:
        """Persist the learned model."""

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the current model, if supported."""
