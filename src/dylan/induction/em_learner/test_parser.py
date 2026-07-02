"""Test parser for evaluating learnt grammars (Java ``qmul.ds.learn.TestParser``)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from dylan.action.grammar import Grammar
from dylan.action.lexicon import Lexicon
from dylan.dag.uttered_word import UtteredWord
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.em_learner.common import Word
from dylan.induction.em_learner.record_type_corpus import RecordTypeCorpus
from dylan.parser.interactive_context_parser import InteractiveContextParser

logger = logging.getLogger(__name__)


class TestParser(InteractiveContextParser):
    """Java ``TestParser``: parse a held-out :class:`RecordTypeCorpus` against a learnt grammar."""

    def __init__(
        self,
        resource_dir_or_lexicon: "str | Path | Lexicon",
        grammar: "Grammar | None" = None,
        top_n: int = 1,
    ) -> None:
        """Construct the parser, accepting either a resource directory or pre-loaded lexicon+grammar."""
        if isinstance(resource_dir_or_lexicon, Lexicon):
            super().__init__()
            self.lexicon = resource_dir_or_lexicon
            if grammar is not None:
                self.grammar = grammar
        else:
            super().__init__(resource_dir_or_lexicon)
        self.test_corpus: RecordTypeCorpus | None = None
        self.unknown_words_encountered: list[Word] = []
        self.top_n = top_n

    # ---------------- corpus loading ----------------

    def load_test_corpus(
        self,
        corpus_file_or_corpus: "str | Path | RecordTypeCorpus",
        has_target_formulae: bool = True,
    ) -> None:
        """Java ``loadTestCorpus``: load a corpus or accept a pre-built one."""
        if isinstance(corpus_file_or_corpus, RecordTypeCorpus):
            self.test_corpus = corpus_file_or_corpus
            return
        self.test_corpus = RecordTypeCorpus()
        if has_target_formulae:
            self.test_corpus.load_corpus(corpus_file_or_corpus)
        else:
            self.test_corpus.load_corpus_no_record_types(corpus_file_or_corpus)

    # ---------------- analysis ----------------

    def contains_unknown(self, sentence: "Iterable[Word | str]") -> bool:
        """Java ``containsUnknown``: track unknown words and return whether any were seen."""
        unknown = False
        for w in sentence:
            key = w.word() if hasattr(w, "word") else str(w)
            if key not in self.lexicon:
                if w not in self.unknown_words_encountered:
                    self.unknown_words_encountered.append(w if isinstance(w, Word) else Word(key))
                unknown = True
        return unknown

    # ---------------- output ----------------

    def parse_corpus_to_file(
        self,
        output_file: "str | Path",
        error_file: "str | Path | None" = None,
        beam: int = 30,
        has_utterance_indices: bool = False,
        incremental_output: bool = False,
    ) -> None:
        """Java ``parseCorpusToFile``: parse the loaded corpus and write incremental TTR output."""
        if self.test_corpus is None:
            logger.warning("No test corpus loaded")
            return
        out_path = Path(output_file)
        err_path = Path(error_file) if error_file is not None else None
        not_parsed = 0
        out_lines: list[str] = []
        err_lines: list[str] = []
        for count, (words, target) in enumerate(self.test_corpus):
            sentence_id = (
                self.test_corpus.get_index_number(count) if has_utterance_indices else str(count)
            )
            self.contains_unknown(words)
            try:
                self.init()
            except Exception:  # noqa: BLE001
                continue
            successful_prefix: list[str] = []
            word_ttr: list[TTRRecordType | None] = []
            parsed_ok = True
            for w in words:
                key = w.word()
                try:
                    parse_state = self.parse_word(UtteredWord(key))
                except Exception:  # noqa: BLE001
                    parse_state = None
                if parse_state is None:
                    parsed_ok = False
                    err_lines.extend([sentence_id, " ".join(successful_prefix), str(words), ""])
                    word_ttr.append(TTRRecordType.parse("[]"))
                    continue
                try:
                    semantics = self.get_final_semantics() if hasattr(self, "get_final_semantics") else None
                except Exception:  # noqa: BLE001
                    semantics = None
                word_ttr.append(semantics if isinstance(semantics, TTRRecordType) else TTRRecordType.parse("[]"))
                successful_prefix.append(key)
            if not parsed_ok:
                not_parsed += 1
            start = 0 if incremental_output else max(0, len(word_ttr) - 1)
            for r in range(start, len(word_ttr)):
                ttr = word_ttr[r] or TTRRecordType.parse("[]")
                if incremental_output:
                    out_lines.append(f"{sentence_id}\t{words[r]}\t{ttr}")
                else:
                    out_lines.append(f"{sentence_id}\t{' '.join(successful_prefix)}\t{ttr}")
        out_path.write_text("\n".join(out_lines), encoding="utf-8")
        if err_path is not None:
            err_lines.append(f"Number of unknown words = {len(self.unknown_words_encountered)}")
            err_lines.extend(str(w) for w in self.unknown_words_encountered)
            err_lines.append(f"Number of bad parses = {not_parsed}")
            err_path.write_text("\n".join(err_lines), encoding="utf-8")


TestParser.loadTestCorpus = TestParser.load_test_corpus  # type: ignore[attr-defined]
TestParser.containsUnknown = TestParser.contains_unknown  # type: ignore[attr-defined]
TestParser.parseCorpusToFile = TestParser.parse_corpus_to_file  # type: ignore[attr-defined]
