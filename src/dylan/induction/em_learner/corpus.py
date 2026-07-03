"""Training corpus containers for induction (Java ``qmul.ds.learn.Corpus``).

Generic list of ``(words, target)`` examples; mirrors Java's
``Corpus<T> extends ArrayList<Pair<Sentence<Word>, T>>`` API.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Generic, Iterable, TypeVar

from dylan.induction.em_learner.common import Word, sentence_from_text, words_to_string

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Corpus(list[tuple[list[Word], T]], Generic[T]):
    """Java ``Corpus<T>``: ``ArrayList<Pair<Sentence<Word>, T>>`` analogue."""

    CORPUS_FOLDER = "corpus"
    WORD_SEP_PATTERN = r"\s"

    def add_example(self, sentence: "str | Iterable[str | Word]", target: T) -> None:
        """Append one ``(words, target)`` example."""
        if isinstance(sentence, str):
            words = sentence_from_text(sentence)
        else:
            words = [Word(str(w)) if not isinstance(w, Word) else w for w in sentence]
        self.append((words, target))

    def load_corpus(self, file_name: "str | Path") -> None:
        """Load a pickled or text corpus (Java ``loadCorpus``)."""
        path = Path(file_name)
        data = path.read_bytes()
        try:
            loaded = pickle.loads(data)
            self.clear()
            self.extend(loaded)
            return
        except Exception:  # noqa: BLE001
            pass
        self.clear()
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            if "\t" in stripped:
                sent, target = stripped.split("\t", 1)
            elif "->" in stripped:
                sent, target = stripped.split("->", 1)
            else:
                sent, target = stripped, ""
            self.add_example(sent.strip(), target.strip())  # type: ignore[arg-type]

    def save_corpus(self, file_name: "str | Path") -> None:
        """Persist corpus as a pickle (Java ``saveCorpus`` analogue)."""
        Path(file_name).write_bytes(pickle.dumps(list(self)))

    def load_and_parse_corpus_from_file(self, sentences_file: "str | Path", resource_dir: "str | Path") -> None:
        """Java ``loadAndParseCorpusFromFile``: parse plain sentences via the DyLan parser.

        The Python port currently logs a warning and falls back to per-line text loading
        because :class:`InteractiveContextParser` integration with the corpus loader is
        out of scope here.
        """
        logger.warning(
            "load_and_parse_corpus_from_file: parser-based loading not ported; falling back to text load",
        )
        _ = resource_dir
        self.load_corpus(sentences_file)

    def __str__(self) -> str:
        """Java ``toString``: list ``words -> target`` lines."""
        return "\n".join(f"{words_to_string(words)} -> {target}" for words, target in self)


Corpus.addExample = Corpus.add_example  # type: ignore[attr-defined]
Corpus.loadCorpus = Corpus.load_corpus  # type: ignore[attr-defined]
Corpus.saveCorpus = Corpus.save_corpus  # type: ignore[attr-defined]
Corpus.loadAndParseCorpusFromFile = Corpus.load_and_parse_corpus_from_file  # type: ignore[attr-defined]
