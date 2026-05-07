"""Escape plain text for safe inclusion in LaTeX math mode."""

from __future__ import annotations


def latex_escape_math(s: str) -> str:
    """Escape *s* for use inside ``$...$`` / ``\\(...\\)`` (ASCII subset used by DS / TTR strings)."""
    out: list[str] = []
    for ch in s:
        if ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch == "$":
            out.append(r"\$")
        elif ch == "%":
            out.append(r"\%")
        elif ch == "#":
            out.append(r"\#")
        elif ch == "&":
            out.append(r"\&")
        elif ch == "_":
            out.append(r"\_")
        elif ch == "^":
            out.append(r"\^{}")
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "{":
            out.append(r"\{")
        elif ch == "}":
            out.append(r"\}")
        else:
            out.append(ch)
    return "".join(out)
