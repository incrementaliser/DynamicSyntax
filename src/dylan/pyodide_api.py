"""JSON ``api_json`` / ``dispatch`` façade for the static Pyodide browser UI.

Uses :class:`dylan.gui.parse_session.ParseSession` (same headless logic as the Flet
desktop app). This module is intentionally **not** under ``dylan.gui`` so the
browser import stays short: ``from dylan.pyodide_api import api_json``.
"""

from __future__ import annotations

import json
from typing import Any

from dylan.gui.parse_session import ParseSession

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


def _session_info() -> str:
    """Live Info-card text for the web UI."""
    return _session.session_info_text()


def _interpretation_payload() -> dict[str, Any]:
    """Index, count, and cap flag for the ``#interpretations`` readout."""
    return {
        "interpretation_index": _session.interpretation_index,
        "interpretation_count": _session.interpretation_count,
        "interpretation_capped": _session.interpretation_capped,
    }


def dispatch(action: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Run *action* with *payload*; always returns a JSON-serialisable dict for the web UI."""
    if action == "info_help":
        info = _session_info()
        return {"help": info, "session_info": info}
    if action == "current_views":
        return {
            "views": _view_dict(),
            "session_info": _session_info(),
            **_interpretation_payload(),
        }
    if action == "set_grammar":
        path = str(payload.get("path", ""))
        repairing = bool(payload.get("repairing", False))
        log_text = _session.set_grammar(path, repairing=repairing)
        return {
            "grammar_log": log_text,
            "parser_ready": _session.parser is not None,
            "session_info": _session_info(),
            **_interpretation_payload(),
        }
    if action == "init":
        err = _session.run_init()
        if err is not None:
            return {
                "error": err,
                "views": None,
                "log_message": err,
                "session_info": _session_info(),
                **_interpretation_payload(),
            }
        return {
            "error": None,
            "views": _view_dict(),
            "log_message": _session.last_event,
            "session_info": _session_info(),
            **_interpretation_payload(),
        }
    if action == "new_sentence":
        err = _session.run_new_sentence()
        if err is not None:
            return {
                "error": err,
                "views": None,
                "log_message": err,
                "session_info": _session_info(),
                **_interpretation_payload(),
            }
        return {
            "error": None,
            "views": _view_dict(),
            "log_message": _session.last_event,
            "session_info": _session_info(),
            **_interpretation_payload(),
        }
    if action == "parse":
        sentence = str(payload.get("sentence", ""))
        reset_before = bool(payload.get("reset_before", True))
        err, ok, events = _session.run_parse(sentence, reset_before=reset_before)
        if err is not None:
            return {
                "error": err,
                "parse_ok": None,
                "views": None,
                "log_message": err,
                "log_messages": [err],
                "session_info": _session_info(),
                **_interpretation_payload(),
            }
        return {
            "error": None,
            "parse_ok": ok,
            "log_message": "\n".join(events),
            "log_messages": events,
            "views": _view_dict(),
            "session_info": _session_info(),
            **_interpretation_payload(),
        }
    if action == "select_interpretation":
        try:
            index = int(payload.get("index", 0))
        except (TypeError, ValueError):
            index = 0
        err, log = _session.select_interpretation(index)
        if err is not None:
            return {
                "error": err,
                "views": None,
                "log_message": err,
                "session_info": _session_info(),
                **_interpretation_payload(),
            }
        return {
            "error": None,
            "log_message": log,
            "views": _view_dict(),
            "session_info": _session_info(),
            **_interpretation_payload(),
        }
    if action == "step_through":
        err, ok = _session.run_step_through()
        if err is not None:
            return {
                "error": err,
                "step_ok": None,
                "views": None,
                "log_message": err,
                "session_info": _session_info(),
                **_interpretation_payload(),
            }
        return {
            "error": None,
            "step_ok": ok,
            "log_message": _session.last_event,
            "views": _view_dict(),
            "session_info": _session_info(),
            **_interpretation_payload(),
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
