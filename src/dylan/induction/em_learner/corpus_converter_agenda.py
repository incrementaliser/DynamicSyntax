"""Agenda for running corpus converters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dylan.induction.em_learner.corpus_converter import CorpusConverter


@dataclass
class CorpusConverterAgenda:
    """Queue of corpus conversion jobs."""

    jobs: list[tuple[Path, Path | None]] = field(default_factory=list)
    converter: CorpusConverter = field(default_factory=CorpusConverter)

    def add(self, source: str | Path, target: str | Path | None = None) -> None:
        """Add a conversion job."""
        self.jobs.append((Path(source), Path(target) if target is not None else None))

    def run(self) -> None:
        """Run all queued conversions."""
        for source, target in self.jobs:
            self.converter.convert(source, target)
