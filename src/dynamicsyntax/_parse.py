"""High-level :func:`parse` using bundled grammars and ``dylan`` parser core."""

from __future__ import annotations

from pathlib import Path
from typing import overload

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.dag.dag_tuple import DAGTuple
from dylan.dag.groundable_edge import GroundableEdge
from dylan.nlp.types import DEFAULT_SPEAKER, utterance_from_text
from dylan.parser.interactive_context_parser import InteractiveContextParser
from dylan.tree.tree import Tree

from dynamicsyntax._session import resolved_grammar_path, session_parser
from dynamicsyntax.parse_trace import ParseActionStep
from dynamicsyntax.parse_result import ParseResult


def _active_path_edges(parser: InteractiveContextParser) -> list[GroundableEdge]:
    """Return active DAG edges from root to the parser's current tuple."""
    dag = parser.get_state()
    cur = dag.get_current_tuple()
    edges: list[GroundableEdge] = []
    while True:
        edge = dag.get_parent_edge(cur)
        parent = dag.get_parent(cur)
        if edge is None or parent is None:
            break
        edges.append(edge)
        cur = parent
    edges.reverse()
    return edges


def _steps_from_edge(parser: InteractiveContextParser, edge: GroundableEdge) -> list[ParseActionStep]:
    """Replay one active edge into action-level tree transitions."""
    if not isinstance(edge.src, DAGTuple) or not isinstance(edge.dst, DAGTuple):
        return []
    word = edge.word.word if edge.word is not None else None
    actions = edge.get_actions()
    if not actions:
        return [
            ParseActionStep(
                word=word,
                action_name="(no action)",
                before_tree=edge.src.get_tree().clone(),
                after_tree=edge.dst.get_tree().clone(),
                edge_id=edge.edge_id,
            ),
        ]
    steps: list[ParseActionStep] = []
    cur = edge.src.get_tree().clone()
    for action in actions:
        before = cur.clone()
        after = parser.apply_actions(cur, [action])
        if after is None:
            return [
                ParseActionStep(
                    word=word,
                    action_name="; ".join(a.get_name() for a in actions),
                    before_tree=edge.src.get_tree().clone(),
                    after_tree=edge.dst.get_tree().clone(),
                    edge_id=edge.edge_id,
                ),
            ]
        steps.append(
            ParseActionStep(
                word=word,
                action_name=action.get_name(),
                before_tree=before,
                after_tree=after.clone(),
                edge_id=edge.edge_id,
            ),
        )
        cur = after
    return steps


def _run_parse_core(
    parser: InteractiveContextParser,
    stripped: str,
    *,
    speaker: str,
    trace: bool,
) -> ParseResult:
    """Run ``init`` / ``new_sentence`` / parse on *parser* and return a :class:`ParseResult`."""
    parser.init()
    parser.new_sentence()
    utt = utterance_from_text(speaker, stripped)
    if not trace:
        ok = parser.parse_utterance(utt)
        tree = parser.get_best_tuple().get_tree()
        semantics: TTRRecordType | None = parser.get_final_semantics() if ok else None
        return ParseResult(ok=ok, semantics=semantics, tree=tree, sentence=stripped, parser=parser)
    trace_list: list[Tree] = [parser.get_best_tuple().get_tree().clone()]
    labels: list[str] = []
    action_steps: list[ParseActionStep] = []
    seen_edge_ids: set[int] = set()
    ok = True
    for uw in utt.words:
        labels.append(uw.word)
        if parser.parse_word(uw) is None:
            ok = False
        for edge in _active_path_edges(parser):
            if edge.edge_id in seen_edge_ids:
                continue
            action_steps.extend(_steps_from_edge(parser, edge))
            seen_edge_ids.add(edge.edge_id)
        trace_list.append(parser.get_best_tuple().get_tree().clone())
    tree = parser.get_best_tuple().get_tree()
    semantics = parser.get_final_semantics() if ok else None
    return ParseResult(
        ok=ok,
        semantics=semantics,
        tree=tree,
        sentence=stripped,
        trace_trees=tuple(trace_list),
        trace_step_labels=tuple(labels),
        action_steps=tuple(action_steps),
        parser=parser,
    )


def _parse_one(
    parser: InteractiveContextParser,
    raw: str,
    *,
    speaker: str,
    trace: bool,
) -> ParseResult:
    """Strip *raw*, return a blank failure without parsing, or run the parse pipeline on *parser*."""
    stripped = raw.strip()
    if not stripped:
        return ParseResult(ok=False, semantics=None, tree=None, sentence="", parser=parser)
    return _run_parse_core(parser, stripped, speaker=speaker, trace=trace)


def _parse_at_path(grammar_path: Path, sentence: str, *, speaker: str, trace: bool) -> ParseResult:
    """Run parse at *grammar_path* and build a :class:`ParseResult`."""
    parser = InteractiveContextParser(grammar_path)
    return _parse_one(parser, sentence, speaker=speaker, trace=trace)


@overload
def parse(
    sentence: str,
    /,
    *,
    speaker: str = ...,
    trace: bool = ...,
) -> ParseResult: ...


@overload
def parse(
    sentences: list[str],
    /,
    *,
    speaker: str = ...,
    trace: bool = ...,
) -> list[ParseResult]: ...


@overload
def parse(
    sentence: str,
    grammar: str | Path | None,
    /,
    *,
    speaker: str = ...,
    trace: bool = ...,
) -> ParseResult: ...


@overload
def parse(
    sentences: list[str],
    grammar: str | Path | None,
    /,
    *,
    speaker: str = ...,
    trace: bool = ...,
) -> list[ParseResult]: ...


def parse(
    sentence_or_sentences: str | list[str],
    grammar: str | Path | None = None,
    /,
    *,
    speaker: str = DEFAULT_SPEAKER,
    trace: bool = False,
) -> ParseResult | list[ParseResult]:
    """Parse one or many sentences and return :class:`~dynamicsyntax.parse_result.ParseResult` objects.

    :param sentence_or_sentences: A single whitespace-tokenised surface string, or a list of
        such strings (lowercased by the tokenizer). An empty list returns ``[]`` without using
        a grammar. Per-item blank or whitespace-only strings yield a failed result for that slot.
    :param grammar: Bundled id or alias (e.g. ``\"ttr\"``), a grammar directory path, or
        ``None`` to use the parser from :func:`~dynamicsyntax.set_grammar`.
    :param speaker: Dialogue participant id passed to the parser (default matches ``dylan``).
    :param trace: If ``True``, record one DS tree after ``new_sentence`` and after each word
        (for :meth:`~dynamicsyntax.parse_result.ParseResult.to_latex` ``incremental``).
    :returns: One :class:`~dynamicsyntax.parse_result.ParseResult`, or a list of them in input
        order; ``semantics`` is ``None`` on failure or blank input for that item.
        Each result may include ``parser`` (the
        :class:`~dylan.parser.interactive_context_parser.InteractiveContextParser` used), except when
        the facade returns early for whitespace-only single-string input without a parse.
    :raises ValueError: If *grammar* is omitted or ``None`` but no session grammar was set with
        :func:`~dynamicsyntax.set_grammar` (non-empty sentence, or any list item non-blank after strip).
    :raises FileNotFoundError: If *grammar* is unknown or not a directory.

    Packaged grammars: ``dynamicsyntax/grammars/`` in the library, and the project
    :mod:`dynamicsyntax.resources` tree (the repository ``resources/`` directory at build time), read
    via :mod:`importlib.resources`. A single string with explicit
    *grammar* uses one fresh parser; a list with explicit *grammar* reuses one parser for all items.
    """
    if isinstance(sentence_or_sentences, list):
        sentences = sentence_or_sentences
        if not sentences:
            return []
        if grammar is not None:
            with resolved_grammar_path(grammar) as grammar_path:
                parser = InteractiveContextParser(grammar_path)
                return [_parse_one(parser, s, speaker=speaker, trace=trace) for s in sentences]
        parser = session_parser()
        if parser is None:
            if any(s.strip() for s in sentences):
                raise ValueError(
                    "no grammar set; call dynamicsyntax.set_grammar(...) first or pass grammar= to parse(...)",
                )
            return [
                ParseResult(ok=False, semantics=None, tree=None, sentence="", parser=None)
                for _ in sentences
            ]
        return [_parse_one(parser, s, speaker=speaker, trace=trace) for s in sentences]

    sentence = sentence_or_sentences
    stripped = sentence.strip()
    if not stripped:
        return ParseResult(ok=False, semantics=None, tree=None, sentence="", parser=None)

    if grammar is not None:
        with resolved_grammar_path(grammar) as grammar_path:
            return _parse_at_path(grammar_path, stripped, speaker=speaker, trace=trace)

    parser = session_parser()
    if parser is None:
        raise ValueError(
            "no grammar set; call dynamicsyntax.set_grammar(...) first or pass grammar= to parse(...)",
        )
    return _run_parse_core(parser, stripped, speaker=speaker, trace=trace)
