"""Corpus of sentences paired with TTR record types (Java ``qmul.ds.learn.RecordTypeCorpus``)."""

from __future__ import annotations

import logging
from pathlib import Path

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.em_learner.corpus import Corpus

logger = logging.getLogger(__name__)


class RecordTypeCorpus(Corpus[TTRRecordType]):
    """Corpus whose targets are :class:`TTRRecordType` objects."""

    CORPUS_FOLDER = "corpus/CHILDES/eveTrainPairs/"
    WORD_SEP_PATTERN = r"\s"

    def __init__(self, corpus_name: "str | None" = None, corpus_path: "str | Path | None" = None) -> None:
        """Construct an empty TTR corpus, optionally loading from *corpus_path*."""
        super().__init__()
        self.sentence_indices: list[str] = []
        self.corpus_name: str | None = corpus_name
        if corpus_path is not None:
            try:
                self.load_corpus(corpus_path)
            except Exception as exc:  # noqa: BLE001
                logger.error("Couldn't load corpus from %s: %s", corpus_path, exc)

    # ---------------- index bookkeeping ----------------

    def add_index(self, index: str) -> None:
        """Append an external sentence id (Java ``addIndex``)."""
        self.sentence_indices.append(index)

    def get_index_number(self, i: int) -> str:
        """Return the index string at position *i* (Java ``getIndexNumber``)."""
        return self.sentence_indices[i]

    # ---------------- file IO ----------------

    def load_corpus(self, file_name: "str | Path") -> None:
        """Java ``loadCorpus``: parse ``Sent : ... / Sem : ...`` blocks separated by blank lines."""
        path = Path(file_name)
        try:
            self.corpus_name = path.stem
        except Exception:  # noqa: BLE001
            self.corpus_name = "default"
        text = path.read_text(encoding="utf-8")
        self.clear()
        block: list[str] = []
        i = 0
        for raw in text.splitlines() + [""]:
            line = raw.rstrip()
            if line.strip().startswith("//"):
                continue
            if line.strip().upper().startswith("END"):
                break
            if not line.strip():
                if block:
                    if self._consume_block(block):
                        i += 1
                    block = []
                continue
            block.append(line)
        if block and self._consume_block(block):
            i += 1
        logger.info("Successfully loaded TTR corpus with %d entries", i)

    def _consume_block(self, block: list[str]) -> bool:
        if len(block) < 2:
            return False
        sent_line = block[0]
        sem_line = block[1]
        if ":" not in sent_line or ":" not in sem_line:
            return False
        sentence = sent_line.split(":", 1)[1].strip()
        semantics = sem_line.split(":", 1)[1].strip()
        target = TTRRecordType.parse(semantics)
        if target is None:
            return False
        self.add_example(sentence, target)
        return True

    def save_corpus(self, file_dir: "str | Path") -> None:
        """Java ``saveCorpus``: write ``GoldSent`` / ``Sem`` blocks back to disk."""
        path = Path(file_dir)
        from dylan.induction.em_learner.corpus_stats import CorpusStats

        stats = CorpusStats()
        with path.open("w", encoding="utf-8") as out:
            for words, target in self:
                sent_str = " ".join(w.word() for w in words)
                out.write(f"GoldSent : {sent_str}\nSem : {target}\n\n")
                stats.add_sentence(words)
            try:
                out.write(stats.stat_reporter())
            except Exception:  # noqa: BLE001
                pass
        logger.info("Successfully saved TTR corpus (size: %d) to %s", len(self), path)

    def load_corpus_no_record_types(self, file_name: "str | Path") -> None:
        """Java ``loadCorpusNoRecordTypes``: each line ``utterance (id)`` -> empty RT."""
        path = Path(file_name)
        text = path.read_text(encoding="utf-8")
        i = 0
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            normalised = line.lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
            tokens = normalised.split()
            if not tokens:
                continue
            sentence_id = tokens[-1]
            tokens = tokens[:-1]
            target = TTRRecordType.parse("[]")
            self.add_example(" ".join(tokens), target)
            self.add_index(sentence_id)
            i += 1
        logger.info("loaded TTR corpus with %d entries and all empty RTs", i)

    # ---------------- corpus combination ----------------

    def merge_corpora(self, other: "RecordTypeCorpus") -> "RecordTypeCorpus":
        """Java ``mergeCorpora``: return new corpus with both contents."""
        merged = RecordTypeCorpus()
        merged.extend(self)
        merged.extend(other)
        return merged

    @staticmethod
    def merge_all_corpora(corpora: "list[RecordTypeCorpus]") -> "RecordTypeCorpus":
        """Java ``mergeAllCorpora``: fold a list of corpora into one."""
        merged = RecordTypeCorpus()
        for c in corpora:
            merged = merged.merge_corpora(c)
        return merged

    def remove_corpus(self, other: "RecordTypeCorpus") -> None:
        """Java ``removeCorpus``: drop entries present in *other*."""
        for entry in list(other):
            try:
                self.remove(entry)
            except ValueError:
                continue

    def get_sub_corpus(self, start: int, end: int) -> "RecordTypeCorpus":
        """Java ``getSubCorpus``: slice ``[start, end)`` into a new corpus."""
        subset = RecordTypeCorpus()
        subset.extend(self[start:end])
        return subset


RecordTypeCorpus.loadCorpus = RecordTypeCorpus.load_corpus  # type: ignore[attr-defined]
RecordTypeCorpus.saveCorpus = RecordTypeCorpus.save_corpus  # type: ignore[attr-defined]
RecordTypeCorpus.loadCorpusNoRecordTypes = RecordTypeCorpus.load_corpus_no_record_types  # type: ignore[attr-defined]
RecordTypeCorpus.mergeCorpora = RecordTypeCorpus.merge_corpora  # type: ignore[attr-defined]
RecordTypeCorpus.mergeAllCorpora = staticmethod(RecordTypeCorpus.merge_all_corpora)  # type: ignore[method-assign]
RecordTypeCorpus.removeCorpus = RecordTypeCorpus.remove_corpus  # type: ignore[attr-defined]
RecordTypeCorpus.getSubCorpus = RecordTypeCorpus.get_sub_corpus  # type: ignore[attr-defined]
RecordTypeCorpus.addIndex = RecordTypeCorpus.add_index  # type: ignore[attr-defined]
RecordTypeCorpus.getIndexNumber = RecordTypeCorpus.get_index_number  # type: ignore[attr-defined]
