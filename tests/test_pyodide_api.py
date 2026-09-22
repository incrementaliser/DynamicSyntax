"""Smoke tests for :mod:`dylan.pyodide_api` (Pyodide JSON bridge for the static web UI)."""

from __future__ import annotations

import json
from importlib import resources

from dylan import pyodide_api


def test_dispatch_load_init_parse_bundled_grammar() -> None:
    """Load bundled TTR grammar via filesystem path, init, and parse a known sentence."""
    root = resources.files("dynamicsyntax")
    node = root / "grammars" / "2015-english-ttr"
    with resources.as_file(node) as grammar_path:
        load_out = pyodide_api.dispatch(
            "set_grammar",
            {"path": str(grammar_path), "repairing": False},
        )
    assert load_out.get("parser_ready") is True
    assert "grammar_log" in load_out
    assert "session_info" in load_out
    assert "Parser object created" not in (load_out.get("grammar_log") or "")

    init_out = pyodide_api.dispatch("init", {})
    assert init_out.get("error") is None
    views = init_out.get("views")
    assert isinstance(views, dict)
    for key in ("semantics", "parse_tree_ascii", "dag", "address_order"):
        assert key in views
    assert "session_info" in init_out
    assert "Grammar:" in (init_out.get("session_info") or "")

    parse_out = pyodide_api.dispatch(
        "parse",
        {"sentence": "a man arrives", "reset_before": True},
    )
    assert parse_out.get("error") is None
    assert parse_out.get("parse_ok") is True
    pv = parse_out.get("views")
    assert isinstance(pv, dict)
    assert "man(" in pv.get("semantics", "") and "arrive" in pv.get("semantics", "")
    log_msg = parse_out.get("log_message") or ""
    lines = parse_out.get("log_messages")
    assert "=== " not in log_msg
    assert isinstance(lines, list)
    assert lines == ["a parsed", "man parsed", "arrives parsed"]
    assert "session_info" in parse_out
    assert parse_out.get("interpretation_index") == 1
    assert int(parse_out.get("interpretation_count") or 0) >= 1
    assert isinstance(parse_out.get("interpretation_capped"), bool)

    count = int(parse_out["interpretation_count"])
    sel = pyodide_api.dispatch("select_interpretation", {"index": 2})
    assert sel.get("error") is None
    if count >= 2:
        assert str(sel.get("log_message") or "").startswith("Interpretation 2 /")
        assert sel.get("interpretation_index") == 2
    else:
        assert sel.get("log_message") is None
        assert sel.get("interpretation_index") == 1


def test_api_json_round_trip() -> None:
    """``api_json`` returns valid JSON and echoes dispatch results."""
    raw = pyodide_api.api_json("info_help", "{}")
    data = json.loads(raw)
    assert "help" in data
