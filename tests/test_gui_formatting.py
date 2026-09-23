"""Tests for GUI text formatters (no Flet dependency)."""

from __future__ import annotations

from pathlib import Path

from dylan.formula.opaque_formula import OpaqueFormula
from dylan.gui.formatting import (
    format_dag_overview,
    format_ds_tree,
    format_semantics_display,
    node_address_type_formula_strings,
)
from dylan.gui.parse_session import (
    FLET_INFO_HELP_TEXT,
    INTERPRETATION_CAP,
    NO_NEW_WORDS_LOG,
    ParseSession,
    format_action_log_lines,
    format_event_log,
    format_grammar_load_report,
    format_interpretation_log,
    format_interpretation_readout,
    format_noncontinuation,
    format_parse_event,
    format_session_info,
    format_session_status,
    format_word_event,
    is_action_log_line,
    resolve_grammar_directory,
)
from dylan.tree.label.labels import FormulaLabel, TypeLabel
from dylan.tree.node import Node
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree
from dylan.dag.word_level_context_dag import WordLevelContextDAG


def test_format_ds_tree_includes_root() -> None:
    t = Tree()
    s = format_ds_tree(t)
    assert "0" in s
    assert "?Ty" in s or "Ty" in s


def test_node_address_type_formula_strings_order() -> None:
    """Type-like labels precede formula segment in the middle field vs last field."""
    addr = NodeAddress("00")
    node = Node(addr, [TypeLabel.t, FormulaLabel(OpaqueFormula("sem"))])
    a, ty, fo = node_address_type_formula_strings(addr, node)
    assert a == "00"
    assert "Ty" in ty
    assert "Fo" in fo
    assert "sem" in fo


def test_format_semantics_display_spaces_pipes() -> None:
    """Pipe separators are normalized to `` ``...`` for the semantics tab."""
    assert format_semantics_display("a|b|c") == "a | b | c"
    assert format_semantics_display("a |b|  c") == "a | b | c"


def test_format_dag_overview_root_only() -> None:
    """Root-only DAG reports the current tuple id and marks that tuple in the list."""
    dag = WordLevelContextDAG()
    s = format_dag_overview(dag)
    assert "Tuple #" in s
    assert "Current tuple id" in s
    assert "← current" in s
    assert "▶ " in s


def test_address_order_keeps_full_labels() -> None:
    """Address-order text includes complete type and formula strings, not ellipsis."""
    t = Tree()
    formula = "semcontent" * 6
    t[t.root_addr] = Node(t.root_addr, [TypeLabel.t, FormulaLabel(OpaqueFormula(formula))])
    text = format_ds_tree(t)
    assert formula in text
    assert "*" in text or "0" in text


def test_format_parse_event_has_no_section_banner() -> None:
    """Parse log lines name the sentence and omit ``===`` banners."""
    ok = format_parse_event(sentence="john likes mary", ok=True)
    bad = format_parse_event(sentence="zzz", ok=False)
    assert "Parsed" in ok and "OK" in ok and "john likes mary" in ok
    assert "failed" in bad
    assert "===" not in ok and "===" not in bad


def test_format_parse_event_names_missing_lexicon_words() -> None:
    """Failed parses list words that are absent from the loaded lexicon."""
    msg = format_parse_event(
        sentence="he sees me",
        ok=False,
        missing_words=["sees"],
    )
    assert "sees" in msg
    assert "not in lexicon" in msg
    assert "he sees me" in msg


def test_resolve_grammar_directory_uses_parent_of_picked_file() -> None:
    """A picked file (e.g. lexicon.txt) resolves to its containing grammar folder."""
    lex = Path(__file__).resolve().parents[1] / "resources" / "2026-english-ttr" / "lexicon.txt"
    folder = resolve_grammar_directory(str(lex))
    assert folder.is_dir()
    assert folder.name == "2026-english-ttr"
    assert resolve_grammar_directory(str(folder)) == folder


def test_format_interpretation_readout_and_log() -> None:
    """The box and the log line share the 1 / N count, with a plus when capped."""
    assert format_interpretation_readout(0, 0, capped=False) == "#interpretations: 0"
    assert format_interpretation_readout(1, 5, capped=False) == "#interpretations: 1 / 5"
    assert (
        format_interpretation_readout(2, INTERPRETATION_CAP, capped=True)
        == "#interpretations: 2 / 30+"
    )
    assert format_interpretation_log(2, 5, capped=False) == "Interpretation 2 / 5"
    assert format_interpretation_log(2, INTERPRETATION_CAP, capped=True) == "Interpretation 2 / 30+"


class _PosDag:
    """DAG stand-in that only resets a position counter."""

    def __init__(self, owner: "_PosParser") -> None:
        self.owner = owner

    def reset_to_first_tuple_after_last_word(self) -> None:
        """Return the fake parser to interpretation 1."""
        self.owner.pos = 0


class _PosParser:
    """Parser stand-in with a fixed number of successful forward steps."""

    def __init__(self, extra_steps: int) -> None:
        self.extra_steps = extra_steps
        self.pos = 0
        self._dag = _PosDag(self)

    def get_state(self) -> _PosDag:
        """Return the fake DAG."""
        return self._dag

    def parse_goal(self, goal: object) -> bool:
        """Succeed *extra_steps* times from the anchor, then fail."""
        del goal
        if self.pos >= self.extra_steps:
            return False
        self.pos += 1
        return True


def test_interpretation_count_caps_at_thirty() -> None:
    """The walk stops at 30 and marks 30+ only when another step would succeed."""
    session = ParseSession()
    session.parser = _PosParser(4)  # type: ignore[assignment]
    session.refresh_interpretations()
    assert session.interpretation_index == 1
    assert session.interpretation_count == 5
    assert session.interpretation_capped is False
    assert session.parser.pos == 0  # type: ignore[attr-defined]

    capped = ParseSession()
    capped.parser = _PosParser(INTERPRETATION_CAP)  # type: ignore[assignment]
    capped.refresh_interpretations()
    assert capped.interpretation_count == INTERPRETATION_CAP
    assert capped.interpretation_capped is True

    exact = ParseSession()
    exact.parser = _PosParser(INTERPRETATION_CAP - 1)  # type: ignore[assignment]
    exact.refresh_interpretations()
    assert exact.interpretation_count == INTERPRETATION_CAP
    assert exact.interpretation_capped is False

    err, log = session.select_interpretation(3)
    assert err is None
    assert log == "Interpretation 3 / 5"
    assert session.interpretation_index == 3
    assert session.parser.pos == 2  # type: ignore[attr-defined]
    assert session.select_interpretation(3) == (None, None)
    assert session.select_interpretation(9) == (None, None)


def test_format_word_event_parsed_and_failed() -> None:
    """Each word is its own log line, with lexicon vs derivation failures distinguished."""
    assert format_word_event(word="a", ok=True) == "a parsed"
    assert format_word_event(word="b", ok=True) == "b parsed"
    assert format_word_event(word="c", ok=False, reason="lexicon") == "c failed (not in lexicon)"
    assert format_word_event(word="c", ok=False) == "c failed (no derivation)"


def _word_status_lines(events: list[str]) -> list[str]:
    """Keep parsed and failed lines, dropping computational-action lines."""
    return [line for line in events if line.endswith(" parsed") or " failed" in line]


def test_run_parse_logs_each_word() -> None:
    """A sentence with a missing last word yields parsed lines then one failed line."""
    grammar = Path(__file__).resolve().parents[1] / "resources" / "2026-english-ttr"
    if not (grammar / "lexicon.txt").is_file():
        return
    session = ParseSession()
    session.set_grammar(str(grammar), repairing=False)
    err, ok, events = session.run_parse("a man zzz", reset_before=True)
    assert err is None
    assert ok is False
    assert _word_status_lines(events) == [
        "a parsed",
        "man parsed",
        "zzz failed (not in lexicon)",
    ]
    assert events[-1] == "zzz failed (not in lexicon)"
    assert any(is_action_log_line(line) for line in events)


def test_format_event_log_strips_whitespace() -> None:
    """Event history lines are stripped, not wrapped in section headers."""
    assert format_event_log("  Init — axiom state.  ") == "Init — axiom state."


def test_format_session_info_lists_live_facts() -> None:
    """Info card includes grammar path, last event, warnings, pointer, and tuple id."""
    text = format_session_info(
        grammar_path="/tmp/g",
        repairing=False,
        last_event="Parsed “a” — OK",
        load_warning_count=2,
        pointer="01",
        tuple_id=3,
    )
    assert "Grammar: /tmp/g" in text
    assert "Repair: off" in text
    assert "Last: Parsed" in text
    assert "Load warnings: 2" in text
    assert "Pointer: 01" in text
    assert "Current DAG tuple: #3" in text


def test_format_action_log_lines_collapses_shared_sequences() -> None:
    """One shared sequence has no prefix; identical subsets share an interpretation line."""
    assert format_action_log_lines([("intro-pred",)]) == ["intro-pred"]
    assert format_action_log_lines([("*thinning",), ("*thinning",)]) == ["*thinning"]
    assert format_action_log_lines([(), ()]) == []
    assert format_action_log_lines([("*thinning",), ("*thinning",), ()]) == [
        "(interp 1, 2) *thinning"
    ]
    assert format_action_log_lines(
        [("completion", "elimination"), ("completion",)],
    ) == [
        "(interp 1) completion elimination",
        "(interp 2) completion",
    ]


def test_is_action_log_line_accepts_interp_prefix() -> None:
    """Action lines are recognised with and without an interpretation prefix."""
    assert is_action_log_line("intro-pred")
    assert is_action_log_line("(interp 1, 2) *thinning completion")
    assert not is_action_log_line("a parsed")
    assert not is_action_log_line(NO_NEW_WORDS_LOG)
    assert not is_action_log_line(format_noncontinuation("the dog"))


def test_format_session_status_tones() -> None:
    """Status rows colour repair, warnings, and the last event by outcome."""
    ok = format_session_status(
        grammar_path="/tmp/g",
        repairing=False,
        last_event="Parsed “a” — OK",
        load_warning_count=0,
        pointer="01",
        tuple_id=3,
    )
    assert ok.grammar.value == "/tmp/g" and ok.grammar.tone == "muted"
    assert ok.repair.value == "off" and ok.repair.tone == "muted"
    assert ok.last.tone == "ok"
    assert ok.warnings.value == "0" and ok.warnings.tone == "muted"
    assert ok.pointer.value == "01" and ok.pointer.tone == "mono"
    assert ok.dag_tuple.value == "#3" and ok.dag_tuple.tone == "mono"
    failed = format_session_status(
        grammar_path=None,
        repairing=True,
        last_event="Parsed “zzz” — failed (zzz not in lexicon)",
        load_warning_count=2,
        pointer=None,
        tuple_id=None,
    )
    assert failed.grammar.value == "(none)"
    assert failed.repair.tone == "amber"
    assert failed.last.tone == "error"
    assert failed.warnings.tone == "amber"
    assert failed.pointer.value == "—"
    assert "Reset" in FLET_INFO_HELP_TEXT
    assert "does not continue" in FLET_INFO_HELP_TEXT


def test_run_parse_logs_only_new_words_and_reset_clears_them() -> None:
    """A continuation logs the new word; a different sentence is refused until Reset."""
    grammar = Path(__file__).resolve().parents[1] / "resources" / "2026-english-ttr"
    if not (grammar / "lexicon.txt").is_file():
        return
    session = ParseSession()
    session.set_grammar(str(grammar), repairing=False)
    err, ok, events = session.run_parse("a man", reset_before=False)
    assert err is None and ok is True
    assert _word_status_lines(events) == ["a parsed", "man parsed"]

    err, ok, events = session.run_parse("a man knows", reset_before=False)
    assert err is None and ok is True
    assert _word_status_lines(events) == ["knows parsed"]
    assert "a parsed" not in events and "man parsed" not in events

    err, ok, events = session.run_parse("a man knows", reset_before=False)
    assert err is None and ok is None
    assert events == [NO_NEW_WORDS_LOG]

    err, ok, events = session.run_parse("the dog", reset_before=False)
    assert ok is None and events == []
    assert err == format_noncontinuation("the dog")

    assert session.run_init(success_event="Reset — axiom state.") is None
    assert session.last_event == "Reset — axiom state."
    err, ok, events = session.run_parse("a man", reset_before=False)
    assert err is None and ok is True
    assert _word_status_lines(events) == ["a parsed", "man parsed"]

    err, ok, events = session.run_parse("a man knows", reset_before=True)
    assert err is None and ok is True
    assert _word_status_lines(events) == ["a parsed", "man parsed", "knows parsed"]


def test_run_parse_emits_interpretation_action_lines() -> None:
    """A sentence with several readings prefixes differing computational actions."""
    grammar = Path(__file__).resolve().parents[1] / "resources" / "2026-english-ttr"
    if not (grammar / "lexicon.txt").is_file():
        return
    session = ParseSession()
    session.set_grammar(str(grammar), repairing=False)
    err, ok, events = session.run_parse("a man knows you", reset_before=True)
    assert err is None and ok is True
    assert _word_status_lines(events) == ["a parsed", "man parsed", "knows parsed", "you parsed"]
    interp_lines = [line for line in events if line.startswith("(interp ")]
    assert len(interp_lines) >= 2
    assert all(is_action_log_line(line) for line in interp_lines)
    assert session.interpretation_count >= 2


def test_format_grammar_load_report_drops_success_fluff() -> None:
    """Successful load reports the path and captured warnings, not init boilerplate."""
    report = format_grammar_load_report(
        Path("."),
        ["WARNING dylan.action.lexicon: skipped a line"],
        ok=True,
    )
    assert "Grammar loaded:" in report
    assert "Parser object created" not in report
    assert "skipped a line" in report
    assert "warning(s)" in report
