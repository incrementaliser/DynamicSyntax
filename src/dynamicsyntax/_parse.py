"""High-level :func:`parse` using bundled grammars and ``dylan`` parser core."""

from __future__ import annotations

from pathlib import Path

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
        return ParseResult(ok=ok, semantics=semantics, tree=tree, sentence=stripped)
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
    )


def _parse_at_path(grammar_path: Path, sentence: str, *, speaker: str, trace: bool) -> ParseResult:
    """Run parse at *grammar_path* and build a :class:`ParseResult`."""
    parser = InteractiveContextParser(grammar_path)
    return _run_parse_core(parser, sentence, speaker=speaker, trace=trace)


def parse(
    sentence: str,
    grammar: str | Path | None = None,
    /,
    *,
    speaker: str = DEFAULT_SPEAKER,
    trace: bool = False,
) -> ParseResult:
    """Parse *sentence* and return a :class:`~dynamicsyntax.parse_result.ParseResult`.

    :param sentence: Whitespace-tokenised surface string (lowercased by the tokenizer).
    :param grammar: Bundled id or alias (e.g. ``\"ttr\"``), a grammar directory path, or
        ``None`` to use the parser from :func:`~dynamicsyntax.load_grammar`.
    :param speaker: Dialogue participant id passed to the parser (default matches ``dylan``).
    :param trace: If ``True``, record one DS tree after ``new_sentence`` and after each word
        (for :meth:`~dynamicsyntax.parse_result.ParseResult.to_latex` ``incremental``).
    :returns: :class:`~dynamicsyntax.parse_result.ParseResult` with ``ok``, ``semantics``,
        and ``tree``; ``semantics`` is ``None`` on failure or blank input.
    :raises ValueError: If *grammar* is ``None`` but no grammar was loaded.
    :raises FileNotFoundError: If *grammar* is unknown or not a directory.

    Bundled grammars are read via :mod:`importlib.resources`; each one-shot parse with an
    explicit *grammar* uses a fresh parser under a short-lived extract path.
    """
    stripped = sentence.strip()
    if not stripped:
        return ParseResult(ok=False, semantics=None, tree=None, sentence="")

    if grammar is not None:
        with resolved_grammar_path(grammar) as grammar_path:
            return _parse_at_path(grammar_path, stripped, speaker=speaker, trace=trace)

    parser = session_parser()
    if parser is None:
        raise ValueError("no grammar loaded; call load_grammar(...) first or pass grammar= to parse(...)")
    return _run_parse_core(parser, stripped, speaker=speaker, trace=trace)
