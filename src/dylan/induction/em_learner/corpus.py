"""Training corpus containers for induction."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Generic, Iterable, TypeVar

from dylan.induction.em_learner.common import Word, sentence_from_text, words_to_string

T = TypeVar("T")


class Corpus(list[tuple[list[Word], T]], Generic[T]):
    """List of sentence/target pairs, matching Java ``Corpus<T>``."""

    CORPUS_FOLDER = "corpus"

    def add_example(self, sentence: str | Iterable[str | Word], target: T) -> None:
        """Append a sentence/target example."""
        words = sentence_from_text(sentence) if isinstance(sentence, str) else [Word(str(w)) for w in sentence]
        self.append((words, target))

    def load_corpus(self, file_name: str | Path) -> None:
        """Load a pickled or text corpus from *file_name*."""
        path = Path(file_name)
        data = path.read_bytes()
        try:
            loaded = pickle.loads(data)
            self.clear()
            self.extend(loaded)
            return
        except Exception:
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

    def save_corpus(self, file_name: str | Path) -> None:
        """Save corpus as pickle."""
        Path(file_name).write_bytes(pickle.dumps(list(self)))

    def __str__(self) -> str:
        """Return readable corpus lines."""
        return "\n".join(f"{words_to_string(words)} -> {target}" for words, target in self)


Corpus.addExample = Corpus.add_example  # type: ignore[attr-defined]
Corpus.loadCorpus = Corpus.load_corpus  # type: ignore[attr-defined]
Corpus.saveCorpus = Corpus.save_corpus  # type: ignore[attr-defined]
