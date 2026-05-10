"""Bounded brute-force language derivation from a loaded interactive parser."""

from __future__ import annotations

import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, TextIO

from dylan.dag.uttered_word import UtteredWord
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.nlp.types import DEFAULT_SPEAKER, RELEASE_TURN_TOKEN, WAIT_TOKEN

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE_OUTPUT_DIR = Path("data/languages_output")
_INCOMPLETE_LANGUAGE_SENTINEL = "<<incomplete>>"
_SEMANTICS_ERROR_LANGUAGE_SENTINEL = "<<semantics-error>>"
_COMPLETION_ABORT_SENTINEL = "<<completion-aborted>>"
_WORKER_PARSER: Any = None


@dataclass(frozen=True)
class LanguageDerivationRecord:
    """Structured outcome from one candidate sentence in language derivation."""

    kind: Literal["success", "failure"]
    sentence: str
    semantics: str | None = None
    failure_line: str | None = None


def _language_candidate_sequences(
    vocab: tuple[str, ...],
    *,
    min_len: int,
    max_len: int,
    max_candidates: int | None,
) -> Iterator[tuple[str, ...]]:
    """Yield candidate word tuples in deterministic Cartesian-product order."""
    emitted = 0
    for length in range(min_len, max_len + 1):
        for words in itertools.product(vocab, repeat=length):
            if max_candidates is not None and emitted >= max_candidates:
                return
            emitted += 1
            yield words


def _default_language_workers() -> int:
    """Return the safe default worker count, leaving one logical core free."""
    return max(1, (os.cpu_count() or 1) - 1)


def _next_language_output_paths(target_dir: Path, resolved_name: str) -> tuple[Path, Path]:
    """Return the next unused ``(language_path, failures_path)`` pair under *target_dir*.

    Uses ``{name}_language.txt`` / ``{name}_language_failures.txt`` when both are
    absent; otherwise ``_1``, ``_2``, … before the ``.txt`` extension until both
    paths are free.
    """
    for n in range(0, 1_000_000):
        if n == 0:
            language_path = target_dir / f"{resolved_name}_language.txt"
            failures_path = target_dir / f"{resolved_name}_language_failures.txt"
        else:
            language_path = target_dir / f"{resolved_name}_language_{n}.txt"
            failures_path = target_dir / f"{resolved_name}_language_failures_{n}.txt"
        if not language_path.exists() and not failures_path.exists():
            return (language_path, failures_path)
    msg = "exhausted language output path suffixes under"
    raise RuntimeError(f"{msg} {target_dir!r}")


def _derive_language_with_parser(
    parser: Any,
    words: tuple[str, ...],
    *,
    speaker: str,
    addressee: str,
) -> LanguageDerivationRecord:
    """Parse and complete one candidate sentence with an already-loaded parser.

    If ``complete_tree`` raises, the candidate is recorded as a failure with
    ``<<completion-aborted>>`` (logged at DEBUG with traceback).
    """
    # Local import: this module is imported from interactive_context_parser.derive_language.
    from dylan.parser.interactive_context_parser import InteractiveContextParser

    if not isinstance(parser, InteractiveContextParser):
        raise TypeError("parser must be InteractiveContextParser")
    parser.init()
    so_far: list[str] = []
    sentence = " ".join(words)
    for word in words:
        if parser.parse_word(UtteredWord(word, speaker, addressee)) is None:
            return LanguageDerivationRecord(
                kind="failure",
                sentence=sentence,
                failure_line=f"{word} | {' '.join(so_far)}",
            )
        so_far.append(word)

    try:
        _, completed = parser.complete_tree(parser.get_best_tuple().get_tree().clone())
    except Exception:
        logger.debug(
            "derive_language: complete_tree aborted for %r",
            sentence,
            exc_info=True,
        )
        return LanguageDerivationRecord(
            kind="failure",
            sentence=sentence,
            failure_line=f"{_COMPLETION_ABORT_SENTINEL} | {sentence}",
        )
    if not completed.is_complete():
        return LanguageDerivationRecord(
            kind="failure",
            sentence=sentence,
            failure_line=f"{_INCOMPLETE_LANGUAGE_SENTINEL} | {sentence}",
        )

    sem = completed.get_maximal_semantics(parser.context).evaluate()
    if not isinstance(sem, TTRRecordType):
        return LanguageDerivationRecord(
            kind="failure",
            sentence=sentence,
            failure_line=f"{_SEMANTICS_ERROR_LANGUAGE_SENTINEL} | {sentence}",
        )
    return LanguageDerivationRecord(kind="success", sentence=sentence, semantics=str(sem))


def _init_language_worker(
    grammar_path: str,
    repairing: bool,
    top_n: int,
    participants: tuple[str, ...],
) -> None:
    """Initialise one parser per worker process for language derivation."""
    global _WORKER_PARSER
    from dylan.parser.interactive_context_parser import InteractiveContextParser

    _WORKER_PARSER = InteractiveContextParser(
        Path(grammar_path),
        repairing=repairing,
        top_n=top_n,
        participants=participants,
    )


def _derive_language_worker(
    task: tuple[tuple[str, ...], str, str],
) -> LanguageDerivationRecord:
    """Run one derivation task in a worker process."""
    if _WORKER_PARSER is None:
        raise RuntimeError("language worker parser is not initialised")
    words, speaker, addressee = task
    return _derive_language_with_parser(
        _WORKER_PARSER,
        words,
        speaker=speaker,
        addressee=addressee,
    )


class LanguageDerivation:
    """Brute-force bounded candidate sentences and write language / failure reports."""

    def __init__(self, parser: Any) -> None:
        """Wire a loaded :class:`~dylan.parser.interactive_context_parser.InteractiveContextParser`."""
        from dylan.parser.interactive_context_parser import InteractiveContextParser

        if not isinstance(parser, InteractiveContextParser):
            raise TypeError("parser must be InteractiveContextParser")
        self._parser = parser

    def run(
        self,
        *,
        max_len: int,
        min_len: int = 1,
        max_candidates: int | None = None,
        max_successful: int | None = None,
        out_dir: str | Path = DEFAULT_LANGUAGE_OUTPUT_DIR,
        grammar_name: str | None = None,
        max_workers: int | None = None,
        speaker: str = DEFAULT_SPEAKER,
        addressee: str = "you",
    ) -> tuple[Path, Path]:
        """Derive bounded language files by parsing candidates in parallel.

        The success file contains ``Sent:`` / ``Sem:`` blocks only after a
        candidate parses and ``complete_tree`` yields a complete tree. The
        failure file contains ``WORD | SO_FAR`` lines, plus ``<<incomplete>>``
        when all words parse but completion leaves requirements unresolved,
        ``<<semantics-error>>`` when evaluation does not yield a record type,
        and ``<<completion-aborted>>`` when ``complete_tree`` raises (e.g.
        effect errors during completion); the latter is logged at DEBUG with
        traceback. ``max_workers=None`` uses all logical cores minus one; ``1``
        runs sequentially for deterministic debugging and tests. Output file
        names pick the lowest unused numeric suffix (``_1``, ``_2``, …) when the
        default pair already exists on disk.
        """
        parser = self._parser
        if parser.context is None:
            raise ValueError("grammar not loaded; call set_grammar(...) first")
        if min_len < 1:
            raise ValueError("min_len must be at least 1")
        if max_len < min_len:
            raise ValueError("max_len must be greater than or equal to min_len")
        if max_candidates is not None and max_candidates < 0:
            raise ValueError("max_candidates must be non-negative")
        if max_successful is not None and max_successful < 0:
            raise ValueError("max_successful must be non-negative")

        grammar_path = getattr(parser.lexicon, "_resource_dir", None)
        resolved_name = grammar_name or (Path(grammar_path).name if grammar_path else None)
        if not resolved_name:
            raise ValueError("grammar_name is required when the loaded lexicon has no resource dir")

        workers = _default_language_workers() if max_workers is None else max_workers
        if workers < 1:
            raise ValueError("max_workers must be at least 1")
        if workers > 1 and (grammar_path is None or not Path(grammar_path).is_dir()):
            raise ValueError("parallel derivation requires a filesystem grammar directory")

        vocab = tuple(
            sorted(
                word
                for word in parser.lexicon.keys()
                if word not in {WAIT_TOKEN, RELEASE_TURN_TOKEN}
            )
        )
        if not vocab:
            raise ValueError("lexicon has no derivable words")

        target_dir = Path(out_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        language_path, failures_path = _next_language_output_paths(target_dir, resolved_name)

        candidates = _language_candidate_sequences(
            vocab,
            min_len=min_len,
            max_len=max_len,
            max_candidates=max_candidates,
        )
        success_count = 0

        with language_path.open("w", encoding="utf-8") as language_file, failures_path.open(
            "w",
            encoding="utf-8",
        ) as failures_file:
            if max_successful == 0:
                return (language_path, failures_path)
            if workers == 1:
                for words in candidates:
                    record = _derive_language_with_parser(
                        parser,
                        words,
                        speaker=speaker,
                        addressee=addressee,
                    )
                    success_count += self._write_language_record(
                        record,
                        language_file=language_file,
                        failures_file=failures_file,
                    )
                    if max_successful is not None and success_count >= max_successful:
                        break
                return (language_path, failures_path)

            assert grammar_path is not None
            batch_size = max(1, workers * 4)
            tasks_iter = ((words, speaker, addressee) for words in candidates)
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_language_worker,
                initargs=(
                    str(Path(grammar_path)),
                    parser._default_repairing,
                    parser._top_n,
                    parser._participants,
                ),
            ) as executor:
                while True:
                    batch = list(itertools.islice(tasks_iter, batch_size))
                    if not batch:
                        break
                    for record in executor.map(_derive_language_worker, batch):
                        success_count += self._write_language_record(
                            record,
                            language_file=language_file,
                            failures_file=failures_file,
                        )
                        if max_successful is not None and success_count >= max_successful:
                            return (language_path, failures_path)
        return (language_path, failures_path)

    @staticmethod
    def _write_language_record(
        record: LanguageDerivationRecord,
        *,
        language_file: TextIO,
        failures_file: TextIO,
    ) -> int:
        """Write one derivation record and return 1 when it is a success."""
        if record.kind == "success":
            language_file.write(f"Sent: {record.sentence}\n")
            language_file.write(f"Sem: {record.semantics}\n\n")
            return 1
        if record.failure_line is not None:
            failures_file.write(f"{record.failure_line}\n")
        return 0
