"""Corpus of sentences paired with TTR record types."""

from __future__ import annotations

from pathlib import Path

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.em_learner.corpus import Corpus


class RecordTypeCorpus(Corpus[TTRRecordType]):
    """Corpus whose targets are :class:`TTRRecordType` objects."""

    def load_corpus(self, file_name: str | Path) -> None:
        """Load a text corpus with ``sentence -> [record]`` or Java-style blocks."""
        path = Path(file_name)
        text = path.read_text(encoding="utf-8")
        self.clear()
        sentence: str | None = None
        semantics: str | None = None
        for raw in text.splitlines() + [""]:
            line = raw.strip()
            if not line:
                if sentence and semantics:
                    rt = TTRRecordType.parse(semantics)
                    if rt is not None:
                        self.add_example(sentence, rt)
                sentence, semantics = None, None
                continue
            lower = line.lower()
            if lower.startswith("sent"):
                sentence = line.split(":", 1)[1].strip() if ":" in line else line[4:].strip()
            elif lower.startswith("sem"):
                semantics = line.split(":", 1)[1].strip() if ":" in line else line[3:].strip()
            elif "->" in line:
                sentence, semantics = [part.strip() for part in line.split("->", 1)]
            elif "\t" in line:
                sentence, semantics = [part.strip() for part in line.split("\t", 1)]


RecordTypeCorpus.loadCorpus = RecordTypeCorpus.load_corpus  # type: ignore[attr-defined]
