"""Parser helper used by learning experiments."""

from __future__ import annotations

from pathlib import Path

from dylan.dag.uttered_word import UtteredWord
from dylan.parser.interactive_context_parser import InteractiveContextParser


class TestParser(InteractiveContextParser):
    """Compatibility subclass for corpus parsing experiments."""

    def parse_corpus_to_file(self, sentences: list[str], output_file: str | Path) -> None:
        """Parse *sentences* and write final semantics lines."""
        lines: list[str] = []
        for sentence in sentences:
            self.init()
            ok = True
            for word in sentence.split():
                if self.parse_word(UtteredWord(word, self.get_name())) is None:
                    ok = False
                    break
            if ok:
                lines.append(str(self.get_final_semantics()))
        Path(output_file).write_text("\n".join(lines), encoding="utf-8")


TestParser.parseCorpusToFile = TestParser.parse_corpus_to_file  # type: ignore[attr-defined]
