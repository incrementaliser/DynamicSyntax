"""Corpus read/write helpers (Java ``qmul.ds.learn.CorpusReaderWriter``).

Reads CHILDES-style sentence/semantics pairs into the in-memory ``corpus_source``
list and writes them back out together with corpus statistics.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from dylan.induction.em_learner.corpus import Corpus

if TYPE_CHECKING:
    from dylan.induction.em_learner.corpus_statistics import CorpusStatistics

logger = logging.getLogger(__name__)


class CorpusReaderWriter:
    """CHILDES-style corpus reader/writer (Java parity).

    Parity with Java ``CorpusReaderWriter``: shared static lists collect
    successfully read examples and missed ones for downstream conversion.
    """

    corpus_source: list[list[str | None]] = []
    missed: list[list[str | None]] = []
    omit_list: list[str] = []
    max_length: int = 7
    sentence_pattern: re.Pattern[str] = re.compile(r"([^\.\?]*)(\s+[\.\?])$")

    def __init__(self, corpus_source_folder: "str | Path") -> None:
        """Populate the omit list and read CHILDES files from *corpus_source_folder*."""
        self.populate_omit()
        type(self).read_in_childes(corpus_source_folder)

    @classmethod
    def read_in_childes(cls, corpus_source_folder: "str | Path") -> None:
        """Java ``readInCHILDES``: read 20 numbered ``trainPairs_*`` files."""
        folder = Path(corpus_source_folder)
        for i in range(1, 21):
            path = folder / f"trainPairs_{i}"
            if not path.exists():
                logger.error("missing CHILDES file: %s", path)
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                logger.error("Error reading templates from %s: %s", path, exc)
                continue

            pair: list[str | None] = [None, None, str(i)]
            for raw in lines:
                line = raw.strip()
                if not line:
                    continue
                pair[2] = str(i)
                if line.startswith("Sent:"):
                    sent = line.split(":", 1)[1].strip()
                    if not cls.sentence_pattern.match(sent):
                        sent = "STRING PUNCT!" + sent
                    else:
                        morphemes = sent.split()
                        rebuilt = ""
                        for m, morph in enumerate(morphemes):
                            rebuilt += morph if morph == "ed" else " " + morph
                            if m > 0 and morph == morphemes[m - 1]:
                                rebuilt += "_REPEAT!_"
                        sent = rebuilt[1:].strip() if rebuilt else ""
                    pair[0] = sent
                elif line.startswith("Sem:"):
                    pair[1] = line.split(":", 1)[1].strip()
                elif line.startswith("//Sent:"):
                    pair[0] = line.split(":", 1)[1].strip()
                elif line.startswith("//Sem:"):
                    pair[1] = line.split(":", 1)[1].strip()
                elif line.startswith("//example_end"):
                    cls.missed.append(pair)
                    pair = [None, None, str(i)]
                elif line.startswith("example_end"):
                    if pair[0] is None or pair[1] is None:
                        logger.error("Hasn't read in example! %r", pair)
                    elif (
                        pair[1] in cls.omit_list
                        or pair[0].startswith("whose")
                        or len(pair[0].split()) > cls.max_length
                    ):
                        logger.debug("missed in %s", pair[2])
                        cls.missed.append(pair)
                    else:
                        cls.corpus_source.append(pair)
                        logger.debug("Adding example %r", pair)
                    pair = [None, None, str(i)]
        logger.info("Read %d examples from %s", len(cls.corpus_source), corpus_source_folder)

    def write_to_file(
        self,
        corpus_source_folder: "str | Path",
        done: list[list[str | None]],
        corpus_stats: "CorpusStatistics | None" = None,
    ) -> None:
        """Java ``writeToFile``: dump converted/missed pairs and stats."""
        folder = Path(corpus_source_folder)
        folder.mkdir(parents=True, exist_ok=True)
        out_path = folder / "CHILDESconversion.txt"
        miss_path = folder / "missedCHILDES.txt"
        try:
            with miss_path.open("w", encoding="utf-8") as miss_fh:
                m = 0
                for entry in type(self).missed:
                    miss_fh.write(f"{entry[0]}\n : {entry[1]}\n :{entry[2]}\n\n")
                    m += 1
                miss_fh.write(str(m))
            with out_path.open("w", encoding="utf-8") as out_fh:
                c = 0
                for entry in done:
                    out_fh.write(f"Sent : {entry[0]}\nSem : {entry[1]}\nFile : {entry[2]}\n\n")
                    c += 1
                out_fh.write("END_OF_CORPUS\n\n")
                out_fh.write(str(c))
                if corpus_stats is not None:
                    out_fh.write("\n" + corpus_stats.final_utt_length_distribution())
                    out_fh.write("\n" + corpus_stats.final_word_distribution())
        except OSError as exc:
            logger.error("Couldn't write to file: %s", exc)

    def populate_omit(self) -> None:
        """Java ``populateOmit``: bootstrap omit-list with hard-coded entries."""
        type(self).omit_list = [
            "lambda $0_{ev}.not($0,)",
            "lambda $0_{ev}.not(and(pro|me,,$0)",
            "lambda $0_{e}.and($0)",
        ]

    @staticmethod
    def read(path: "str | Path") -> Corpus[object]:
        """Compatibility helper used by older callers; delegates to :class:`Corpus`."""
        corpus: Corpus[object] = Corpus()
        corpus.load_corpus(path)
        return corpus

    @staticmethod
    def write(corpus: Corpus[object], path: "str | Path") -> None:
        """Compatibility helper that delegates to :meth:`Corpus.save_corpus`."""
        corpus.save_corpus(path)


CorpusReaderWriter.readInCHILDES = CorpusReaderWriter.read_in_childes  # type: ignore[attr-defined]
CorpusReaderWriter.writeToFile = CorpusReaderWriter.write_to_file  # type: ignore[attr-defined]
CorpusReaderWriter.populateOmit = CorpusReaderWriter.populate_omit  # type: ignore[attr-defined]
