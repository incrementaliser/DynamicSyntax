"""JSON ``api_json`` / ``dispatch`` façade for the static Pyodide browser UI.

Uses :class:`dylan.gui.parse_session.ParseSession` (same headless logic as the Flet
desktop app). This module is intentionally **not** under ``dylan.gui`` so the
browser import stays short: ``from dylan.pyodide_api import api_json``.
"""

from __future__ import annotations

import json
from typing import Any

from dylan.gui.parse_session import ParseSession, format_parse_state_log, GUI_INFO_HELP_TEXT

_session = ParseSession()


def _view_dict() -> dict[str, Any] | None:
    """Serialise current tab content; ``None`` if no parser."""
    vs = _session.current_view_strings()
    if vs is None:
        return None
    return {
        "address_order": vs.address_order,
        "parse_tree_ascii": vs.parse_tree_ascii,
        "semantics": vs.semantics,
        "dag": vs.dag,
    }


def dispatch(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Run *action* with *payload*; always returns a JSON-serialisable dict for the web UI."""
    if action == "info_help":
        return {"help": GUI_INFO_HELP_TEXT}
    if action == "current_views":
        return {"views": _view_dict()}
    if action == "load_grammar":
        path = str(payload.get("path", ""))
        repairing = bool(payload.get("repairing", False))
        log_text = _session.load_grammar(path, repairing=repairing)
        return {
            "grammar_log": log_text,
            "parser_ready": _session.parser is not None,
        }
    if action == "init":
        err = _session.run_init()
        if err is not None:
            return {"error": err, "views": None, "log_message": None}
        return {
            "error": None,
            "views": _view_dict(),
            "log_message": format_parse_state_log("Parser re-initialised (axiom state)."),
        }
    if action == "new_sentence":
        err = _session.run_new_sentence()
        if err is not None:
            return {"error": err, "views": None, "log_message": None}
        return {
            "error": None,
            "views": _view_dict(),
            "log_message": format_parse_state_log("New sentence — DAG reset to axiom."),
        }
    if action == "parse":
        sentence = str(payload.get("sentence", ""))
        reset_before = bool(payload.get("reset_before", True))
        err, ok = _session.run_parse(sentence, reset_before=reset_before)
        if err is not None:
            return {"error": err, "parse_ok": None, "views": None, "log_message": None}
        msg = (
            "Parse OK."
            if ok
            else "Parse finished with failures (check sentence / lexicon)."
        )
        return {
            "error": None,
            "parse_ok": ok,
            "log_message": format_parse_state_log(msg),
            "views": _view_dict(),
        }
    return {"error": f"unknown action: {action}"}


def api_json(action: str, payload_json: str) -> str:
    """Parse *payload_json*, dispatch *action*, return a JSON object string for JavaScript."""
    try:
        payload = json.loads(payload_json) if payload_json else {}
        if not isinstance(payload, dict):
            payload = {}
        out = dispatch(action, payload)
    except json.JSONDecodeError as ex:
        out = {"error": f"invalid JSON payload: {ex}"}
    return json.dumps(out)
