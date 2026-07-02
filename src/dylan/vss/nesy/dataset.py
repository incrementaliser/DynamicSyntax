"""Datasets for NeSy parse learning (supervised and latent future)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True, slots=True)
class LatentParseRecord:
    """Sentence-only record for future unsupervised EM."""

    sentence: str
    grammar_path: Path


class LatentParseDataset:
    """Sentence-only corpus for future unsupervised parse marginalization."""

    def __init__(self, records: list[LatentParseRecord]) -> None:
        """Store *records* for iteration."""
        self._records = list(records)

    def __len__(self) -> int:
        """Number of records."""
        return len(self._records)

    def __iter__(self) -> Iterator[LatentParseRecord]:
        """Iterate records."""
        yield from self._records

    @classmethod
    def from_sentences(
        cls,
        sentences: list[str],
        *,
        grammar_path: Path,
    ) -> LatentParseDataset:
        """Build a dataset from raw sentences."""
        return cls(
            [LatentParseRecord(sentence=s.strip(), grammar_path=grammar_path) for s in sentences]
        )
