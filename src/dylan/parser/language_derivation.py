"""Bounded brute-force language derivation from a loaded interactive parser."""

from __future__ import annotations

import itertools
import logging
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, TextIO

from rich.console import Console as _RichConsole

from dylan.action.lexicon import Lexicon
from dylan.dag.uttered_word import UtteredWord
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.nlp.types import DEFAULT_SPEAKER, RELEASE_TURN_TOKEN, WAIT_TOKEN

logger = logging.getLogger(__name__)
_layered_console = _RichConsole()

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


def _next_layered_run_paths(
    target_dir: Path,
    resolved_name: str,
    *,
    max_layer: int,
) -> tuple[int, dict[int, tuple[Path, Path]]]:
    """Return ``(run_suffix, layer -> (language_path, failures_path))`` for an unused layered run.

    Chooses the smallest numeric suffix *n* such that every layer file pair is absent on disk.
    """
    if max_layer < 1:
        raise ValueError("max_layer must be at least 1")
    for n in range(0, 1_000_000):
        paths: dict[int, tuple[Path, Path]] = {}
        collision = False
        for i in range(1, max_layer + 1):
            if n == 0:
                lang_p = target_dir / f"{resolved_name}_layer_{i}_language.txt"
                fail_p = target_dir / f"{resolved_name}_layer_{i}_language_failures.txt"
            else:
                lang_p = target_dir / f"{resolved_name}_layer_{i}_language_{n}.txt"
                fail_p = target_dir / f"{resolved_name}_layer_{i}_language_failures_{n}.txt"
            if lang_p.exists() or fail_p.exists():
                collision = True
                break
            paths[i] = (lang_p, fail_p)
        if not collision:
            return (n, paths)
    msg = "exhausted layered language output path suffixes under"
    raise RuntimeError(f"{msg} {target_dir!r}")


def _parser_for_layered_forward(base: Any) -> Any:
    """Return a fresh parser with ``repairing=False`` and ``top_n=1`` for layered derivation."""
    from dylan.parser.interactive_context_parser import InteractiveContextParser

    if not isinstance(base, InteractiveContextParser):
        raise TypeError("parser must be InteractiveContextParser")
    grammar_path = getattr(base.lexicon, "_resource_dir", None)
    if grammar_path is None or not Path(grammar_path).is_dir():
        msg = "layered derivation requires a filesystem grammar directory on the lexicon"
        raise ValueError(msg)
    return InteractiveContextParser(
        Path(grammar_path),
        repairing=False,
        top_n=1,
        participants=base._participants,
        log_level=getattr(base, "_log_level", "off"),
        log_output=getattr(base, "_log_output", "terminal"),
        log_dir=getattr(base, "_log_dir", None),
    )


def _derivable_vocab(parser: Any) -> tuple[str, ...]:
    """Return sorted derivable surface forms excluding dialogue-control tokens."""
    return tuple(
        sorted(
            w
            for w in parser.lexicon.keys()
            if w not in {WAIT_TOKEN, RELEASE_TURN_TOKEN}
        )
    )


def _template_representative_vocab(lex: Lexicon) -> tuple[str, ...]:
    """Pick one surface word per lexical template name (``LexicalAction.action_type``)."""
    by_template: dict[str, list[str]] = defaultdict(list)
    for word in lex.keys():
        for la in lex[word]:
            tmpl = la.action_type or ""
            by_template[tmpl].append(word)
    reps = [min(by_template[k]) for k in sorted(by_template) if by_template[k]]
    return tuple(sorted(reps))


def _template_vocab_groups(lex: Lexicon) -> tuple[tuple[str, ...], ...]:
    """Return one sorted group of words per lexical template; groups are in template-name order.

    Unlike :func:`_template_representative_vocab`, every word in each template is included.
    The BFS uses first-success semantics within each group: the first word that parses wins,
    so morphological variants (e.g. ``arrive`` vs ``arrives``) are all tried in sorted order.
    """
    by_template: dict[str, list[str]] = defaultdict(list)
    for word in lex.keys():
        if word in {WAIT_TOKEN, RELEASE_TURN_TOKEN}:
            continue
        for la in lex[word]:
            tmpl = la.action_type or ""
            if word not in by_template[tmpl]:
                by_template[tmpl].append(word)
    return tuple(
        tuple(sorted(by_template[k]))
        for k in sorted(by_template)
        if by_template[k]
    )


def _replay_prefix(
    parser: Any,
    prefix: tuple[str, ...],
    *,
    speaker: str,
    addressee: str,
) -> bool:
    """Reset the parser and replay *prefix*; return False if any word fails or raises."""
    parser.init()
    for w in prefix:
        try:
            result = parser.parse_word(UtteredWord(w, speaker, addressee))
        except Exception:
            logger.debug("_replay_prefix: parse_word raised for %r in prefix %r", w, prefix, exc_info=True)
            return False
        if result is None:
            return False
    return True


def _record_completion_from_current_state(parser: Any, *, sentence: str) -> LanguageDerivationRecord:
    """Run ``complete_tree`` on the current tuple and classify the outcome."""
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

    return _record_completion_from_current_state(parser, sentence=sentence)


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

    def run_layered(
        self,
        *,
        max_len: int,
        min_len: int = 1,
        max_successful: int | None = None,
        out_dir: str | Path = DEFAULT_LANGUAGE_OUTPUT_DIR,
        grammar_name: str | None = None,
        speaker: str = DEFAULT_SPEAKER,
        addressee: str = "you",
    ) -> dict[int, tuple[Path, Path]]:
        """Layered prefix expansion with completion on failed extensions (forward-only ``top_n=1`` parser).

        Each ``layer_i`` language file holds ``Sent:``/``Sem:`` when ``complete_tree`` succeeds,
        including immediately after each successful ``parse_word`` on a prefix.
        Each failures file holds deduplicated ``WORD | PREFIX`` lines for ``parse_word`` extension
        failures only; ``complete_tree`` failures are not written there.
        """
        fw = _parser_for_layered_forward(self._parser)
        vocab = _derivable_vocab(fw)
        if not vocab:
            raise ValueError("lexicon has no derivable words")
        # Each word is its own singleton group — identical to the original flat-vocab behaviour.
        vocab_groups = tuple((w,) for w in vocab)
        return self._run_layered_bfs(
            fw,
            vocab_groups,
            max_len=max_len,
            min_len=min_len,
            max_successful=max_successful,
            out_dir=out_dir,
            grammar_name=grammar_name,
            speaker=speaker,
            addressee=addressee,
        )

    def run_layered_category(
        self,
        *,
        max_len: int,
        min_len: int = 1,
        max_successful: int | None = None,
        out_dir: str | Path = DEFAULT_LANGUAGE_OUTPUT_DIR,
        grammar_name: str | None = None,
        speaker: str = DEFAULT_SPEAKER,
        addressee: str = "you",
    ) -> dict[int, tuple[Path, Path]]:
        """Same as :meth:`run_layered` but groups words by lexical template.

        For each template group the BFS uses first-success semantics: all words in the group
        are tried in sorted order and the first that parses is used.  This ensures that
        morphological variants (e.g. ``arrive`` vs ``arrives``) are all considered, so a
        category is only marked as failing when *no* form in the template can extend the
        current prefix.
        """
        fw = _parser_for_layered_forward(self._parser)
        vocab_groups = _template_vocab_groups(fw.lexicon)
        if not vocab_groups:
            raise ValueError("lexicon has no template groups")
        return self._run_layered_bfs(
            fw,
            vocab_groups,
            max_len=max_len,
            min_len=min_len,
            max_successful=max_successful,
            out_dir=out_dir,
            grammar_name=grammar_name,
            speaker=speaker,
            addressee=addressee,
        )

    def run_layered_random(
        self,
        *,
        max_len: int,
        max_paths: int,
        max_steps: int | None = None,
        seed: int | None = None,
        min_len: int = 1,
        max_successful: int | None = None,
        out_dir: str | Path = DEFAULT_LANGUAGE_OUTPUT_DIR,
        grammar_name: str | None = None,
        speaker: str = DEFAULT_SPEAKER,
        addressee: str = "you",
        use_category_vocab: bool = False,
    ) -> dict[int, tuple[Path, Path]]:
        """Random walks over feasible prefixes; completion uses the same layer files as :meth:`run_layered`.

        When ``use_category_vocab=True`` the vocabulary groups words by lexical template and
        the walk picks a random group at each step, trying all forms until one parses.
        """
        fw = _parser_for_layered_forward(self._parser)
        vocab_groups: tuple[tuple[str, ...], ...]
        if use_category_vocab:
            vocab_groups = _template_vocab_groups(fw.lexicon)
            if not vocab_groups:
                raise ValueError("lexicon has no template groups")
        else:
            vocab = _derivable_vocab(fw)
            if not vocab:
                raise ValueError("lexicon has no derivable words")
            vocab_groups = tuple((w,) for w in vocab)
        return self._run_layered_random_walk(
            fw,
            vocab_groups,
            max_len=max_len,
            max_paths=max_paths,
            max_steps=max_steps,
            seed=seed,
            min_len=min_len,
            max_successful=max_successful,
            out_dir=out_dir,
            grammar_name=grammar_name,
            speaker=speaker,
            addressee=addressee,
        )

    def _run_layered_bfs(
        self,
        parser: Any,
        vocab_groups: tuple[tuple[str, ...], ...],
        *,
        max_len: int,
        min_len: int,
        max_successful: int | None,
        out_dir: str | Path,
        grammar_name: str | None,
        speaker: str,
        addressee: str,
    ) -> dict[int, tuple[Path, Path]]:
        """Breadth-first layered derivation; writes deduplicated records per depth.

        ``vocab_groups`` is a tuple of word groups sharing a lexical template.  For each
        group the BFS uses *first-success* semantics: words in the group are tried in order
        and the first one that parses is used; the group is recorded as a failure only when
        *every* word fails.  Singleton groups reproduce the original flat-vocab behaviour.

        After every successful ``parse_word`` (seed and extensions), ``complete_tree`` is
        attempted immediately; successes are written to the matching layer language file.
        Extension failures (all words in a group fail) go to the layer failure file for the
        current prefix length.
        """
        base = self._parser
        if base.context is None:
            raise ValueError("grammar not loaded; call set_grammar(...) first")
        if min_len < 1:
            raise ValueError("min_len must be at least 1")
        if max_len < 1:
            raise ValueError("max_len must be at least 1")
        if max_len < min_len:
            raise ValueError("max_len must be greater than or equal to min_len")
        if max_successful is not None and max_successful < 0:
            raise ValueError("max_successful must be non-negative")

        grammar_path = getattr(parser.lexicon, "_resource_dir", None)
        resolved_name = grammar_name or (Path(grammar_path).name if grammar_path else None)
        if not resolved_name:
            raise ValueError("grammar_name is required when the loaded lexicon has no resource dir")

        n_groups = len(vocab_groups)
        n_words = sum(len(g) for g in vocab_groups)
        _layered_console.print(
            f"\n[bold]Layered BFS[/bold] — "
            f"[cyan]{n_groups}[/cyan] vocabulary groups, "
            f"[cyan]{n_words}[/cyan] total words, "
            f"max_len=[cyan]{max_len}[/cyan]"
        )

        target_dir = Path(out_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        _, paths_by_layer = _next_layered_run_paths(target_dir, resolved_name, max_layer=max_len)

        success_total = 0
        seen_success_by_layer: dict[int, set[str]] = {i: set() for i in range(1, max_len + 1)}
        seen_fail_by_layer: dict[int, set[str]] = {i: set() for i in range(1, max_len + 1)}

        with ExitStack() as stack:
            handles: dict[int, tuple[TextIO, TextIO]] = {}
            for i in range(1, max_len + 1):
                lp, fp = paths_by_layer[i]
                handles[i] = (
                    stack.enter_context(lp.open("w", encoding="utf-8")),
                    stack.enter_context(fp.open("w", encoding="utf-8")),
                )

            def emit_completion_success_only(layer_idx: int, record: LanguageDerivationRecord) -> None:
                """Write ``Sent:``/``Sem:`` only when completion succeeds; drop completion failures."""
                nonlocal success_total
                if record.kind != "success":
                    return
                lang_f = handles[layer_idx][0]
                ss = seen_success_by_layer[layer_idx]
                success_total += LanguageDerivation._write_layered_deduped(
                    record,
                    language_file=lang_f,
                    seen_success=ss,
                )

            stop = False
            frontier: list[tuple[str, ...]] = []
            seed_fail_groups = 0

            with _layered_console.status("[bold green]Layer 1 (seed) running...[/bold green]"):
                for group in vocab_groups:
                    if stop:
                        break
                    for w in group:
                        parser.init()
                        try:
                            seed_result = parser.parse_word(UtteredWord(w, speaker, addressee))
                        except Exception:
                            logger.debug("_run_layered_bfs: seed parse_word raised for %r", w, exc_info=True)
                            seed_result = None
                        if seed_result is not None:
                            frontier.append((w,))
                            if 1 >= min_len:
                                emit_completion_success_only(
                                    1,
                                    _record_completion_from_current_state(parser, sentence=w),
                                )
                            if max_successful is not None and success_total >= max_successful:
                                stop = True
                            break  # first success in group wins
                        # else: word failed — try next word in same group
                    else:
                        # every word in this group failed for the seed
                        seed_fail_groups += 1
                        LanguageDerivation._write_extension_failure_deduped(
                            handles[1][1],
                            seen_fail_by_layer[1],
                            word=group[0],
                            prefix_words=(),
                        )

            _layered_console.print(
                f"  Layer 1 — "
                f"[green]{len(frontier)} seeds parseable[/green], "
                f"[red]{seed_fail_groups} group failures[/red], "
                f"[bold yellow]{len(seen_success_by_layer[1])} completions[/bold yellow]"
            )

            for depth in range(1, max_len):
                if stop or not frontier:
                    break
                next_frontier: list[tuple[str, ...]] = []
                layer_ext_fail_groups = 0
                success_before_depth = success_total

                with _layered_console.status(
                    f"[bold green]Layer {depth + 1} (extending {len(frontier)} prefixes)...[/bold green]"
                ):
                    for P in sorted(set(frontier)):
                        if stop:
                            break
                        had_failure = False
                        for group in vocab_groups:
                            if stop:
                                break
                            replay_failed = False
                            group_success = False
                            for w in group:
                                if not _replay_prefix(parser, P, speaker=speaker, addressee=addressee):
                                    replay_failed = True
                                    break
                                try:
                                    ext_result = parser.parse_word(UtteredWord(w, speaker, addressee))
                                except Exception:
                                    logger.debug(
                                        "_run_layered_bfs: extension parse_word raised for %r after %r",
                                        w,
                                        P,
                                        exc_info=True,
                                    )
                                    ext_result = None
                                if ext_result is not None:
                                    group_success = True
                                    q = P + (w,)
                                    if depth + 1 <= max_len:
                                        next_frontier.append(q)
                                    if len(q) >= min_len and len(q) <= max_len:
                                        emit_completion_success_only(
                                            len(q),
                                            _record_completion_from_current_state(
                                                parser,
                                                sentence=" ".join(q),
                                            ),
                                        )
                                    if max_successful is not None and success_total >= max_successful:
                                        stop = True
                                    break  # first success in group wins
                                # else: word failed — try next word in same group
                            if not replay_failed and not group_success:
                                # every word in this group failed to extend P
                                had_failure = True
                                layer_ext_fail_groups += 1
                                LanguageDerivation._write_extension_failure_deduped(
                                    handles[len(P)][1],
                                    seen_fail_by_layer[len(P)],
                                    word=group[0],
                                    prefix_words=P,
                                )
                        if had_failure and len(P) >= min_len:
                            if _replay_prefix(parser, P, speaker=speaker, addressee=addressee):
                                emit_completion_success_only(
                                    len(P),
                                    _record_completion_from_current_state(parser, sentence=" ".join(P)),
                                )
                            if max_successful is not None and success_total >= max_successful:
                                stop = True

                frontier = sorted(set(next_frontier))
                new_successes = success_total - success_before_depth
                _layered_console.print(
                    f"  Layer {depth + 1} — "
                    f"[bold yellow]+{new_successes} new completions[/bold yellow], "
                    f"[red]{layer_ext_fail_groups} group failures[/red], "
                    f"[cyan]{len(frontier)} in fringe[/cyan]"
                )

            if not stop:
                for P in sorted(set(frontier)):
                    if stop:
                        break
                    if len(P) == max_len and len(P) >= min_len:
                        if _replay_prefix(parser, P, speaker=speaker, addressee=addressee):
                            emit_completion_success_only(
                                max_len,
                                _record_completion_from_current_state(parser, sentence=" ".join(P)),
                            )
                        if max_successful is not None and success_total >= max_successful:
                            stop = True

        total_successes = sum(len(ss) for ss in seen_success_by_layer.values())
        _layered_console.print(
            f"\n[bold green]Done.[/bold green] "
            f"Total completions written: [bold yellow]{total_successes}[/bold yellow]"
        )
        return paths_by_layer

    def _run_layered_random_walk(
        self,
        parser: Any,
        vocab_groups: tuple[tuple[str, ...], ...],
        *,
        max_len: int,
        max_paths: int,
        max_steps: int | None,
        seed: int | None,
        min_len: int,
        max_successful: int | None,
        out_dir: str | Path,
        grammar_name: str | None,
        speaker: str,
        addressee: str,
    ) -> dict[int, tuple[Path, Path]]:
        """Random walks over feasible prefixes; same layer output format as :meth:`_run_layered_bfs`.

        At each step a random group is chosen.  All words in the group are tried in order;
        the first that parses advances the prefix.  If no word in the group parses, a failure
        is recorded and the walk restarts.  After each successful ``parse_word``,
        ``complete_tree`` is attempted immediately.
        """
        base = self._parser
        if base.context is None:
            raise ValueError("grammar not loaded; call set_grammar(...) first")
        if min_len < 1:
            raise ValueError("min_len must be at least 1")
        if max_len < 1:
            raise ValueError("max_len must be at least 1")
        if max_len < min_len:
            raise ValueError("max_len must be greater than or equal to min_len")
        if max_paths < 1:
            raise ValueError("max_paths must be at least 1")
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps must be at least 1 when set")
        if max_successful is not None and max_successful < 0:
            raise ValueError("max_successful must be non-negative")

        grammar_path = getattr(parser.lexicon, "_resource_dir", None)
        resolved_name = grammar_name or (Path(grammar_path).name if grammar_path else None)
        if not resolved_name:
            raise ValueError("grammar_name is required when the loaded lexicon has no resource dir")

        target_dir = Path(out_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        _, paths_by_layer = _next_layered_run_paths(target_dir, resolved_name, max_layer=max_len)

        rng = random.Random(seed)
        groups_list = list(vocab_groups)
        success_total = 0
        seen_success_by_layer: dict[int, set[str]] = {i: set() for i in range(1, max_len + 1)}
        seen_fail_by_layer: dict[int, set[str]] = {i: set() for i in range(1, max_len + 1)}

        with ExitStack() as stack:
            handles: dict[int, tuple[TextIO, TextIO]] = {}
            for i in range(1, max_len + 1):
                lp, fp = paths_by_layer[i]
                handles[i] = (
                    stack.enter_context(lp.open("w", encoding="utf-8")),
                    stack.enter_context(fp.open("w", encoding="utf-8")),
                )

            def emit_completion_success_only(layer_idx: int, record: LanguageDerivationRecord) -> None:
                """Write ``Sent:``/``Sem:`` only when completion succeeds; drop completion failures."""
                nonlocal success_total
                if record.kind != "success":
                    return
                lang_f = handles[layer_idx][0]
                ss = seen_success_by_layer[layer_idx]
                success_total += LanguageDerivation._write_layered_deduped(
                    record,
                    language_file=lang_f,
                    seen_success=ss,
                )

            def _try_completion_on_prefix(pfx: list[str]) -> None:
                """Attempt completion on the current prefix if it meets length requirements."""
                lp = len(pfx)
                if lp >= min_len:
                    if _replay_prefix(parser, tuple(pfx), speaker=speaker, addressee=addressee):
                        emit_completion_success_only(
                            lp,
                            _record_completion_from_current_state(parser, sentence=" ".join(pfx)),
                        )

            stop = False
            for _ in range(max_paths):
                if stop:
                    break
                prefix: list[str] = []
                steps = 0
                while True:
                    if stop:
                        break
                    if len(prefix) >= max_len:
                        _try_completion_on_prefix(prefix)
                        if max_successful is not None and success_total >= max_successful:
                            stop = True
                        break
                    if max_steps is not None and steps >= max_steps:
                        _try_completion_on_prefix(prefix)
                        if max_successful is not None and success_total >= max_successful:
                            stop = True
                        break
                    steps += 1
                    group = rng.choice(groups_list)
                    replay_failed = False
                    group_success = False
                    for w in group:
                        if not _replay_prefix(parser, tuple(prefix), speaker=speaker, addressee=addressee):
                            replay_failed = True
                            break
                        try:
                            rw_result = parser.parse_word(UtteredWord(w, speaker, addressee))
                        except Exception:
                            logger.debug(
                                "_run_layered_random_walk: parse_word raised for %r after %r",
                                w,
                                prefix,
                                exc_info=True,
                            )
                            rw_result = None
                        if rw_result is not None:
                            group_success = True
                            prefix.append(w)
                            lp = len(prefix)
                            if lp >= min_len and lp <= max_len:
                                emit_completion_success_only(
                                    lp,
                                    _record_completion_from_current_state(
                                        parser,
                                        sentence=" ".join(prefix),
                                    ),
                                )
                            if max_successful is not None and success_total >= max_successful:
                                stop = True
                            break
                        # else: word failed — try next in group
                    if replay_failed:
                        break  # prefix invalid; restart walk
                    if not group_success:
                        # every word in the group failed — record failure and end walk
                        fail_layer = max(1, len(prefix))
                        LanguageDerivation._write_extension_failure_deduped(
                            handles[fail_layer][1],
                            seen_fail_by_layer[fail_layer],
                            word=group[0],
                            prefix_words=tuple(prefix),
                        )
                        _try_completion_on_prefix(prefix)
                        if max_successful is not None and success_total >= max_successful:
                            stop = True
                        break

        return paths_by_layer

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
    def _write_layered_deduped(
        record: LanguageDerivationRecord,
        *,
        language_file: TextIO,
        seen_success: set[str],
    ) -> int:
        """Write a successful ``Sent:``/``Sem:`` block if unseen; discard non-success records.

        Flushes the handle after writing so layered runs show incremental output on disk.
        """
        if record.kind != "success":
            return 0
        key = record.sentence
        if key in seen_success:
            return 0
        seen_success.add(key)
        language_file.write(f"Sent: {record.sentence}\n")
        language_file.write(f"Sem: {record.semantics}\n\n")
        language_file.flush()
        return 1

    @staticmethod
    def _write_extension_failure_deduped(
        failures_file: TextIO,
        seen_failure: set[str],
        *,
        word: str,
        prefix_words: tuple[str, ...],
    ) -> int:
        """Write one ``WORD | PREFIX`` extension-failure line when that line has not been written; return 1 if new.

        Flushes the handle after writing so layered runs show incremental output on disk.
        """
        so_far = " ".join(prefix_words)
        line = f"{word} | {so_far}"
        if line in seen_failure:
            return 0
        seen_failure.add(line)
        failures_file.write(f"{line}\n")
        failures_file.flush()
        return 1

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
