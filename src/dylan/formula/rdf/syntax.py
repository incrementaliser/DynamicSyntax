"""Surface syntax for RDF formulae: ``rdf{ Turtle }`` and ``G`` / ``G1`` binders."""

from __future__ import annotations

import re

_RDF_OPEN = "rdf{"
_BINDER = re.compile(r"^(G\d*)\s*\^(.*)$", re.DOTALL)


class RDFSyntaxError(ValueError):
    """A delimited RDF formula is not well-formed Turtle or not a closed ``rdf{ }``."""


def split_rdf_graph(text: str) -> str | None:
    """Return the Turtle inside a whole-string ``rdf{ ... }``.

    Returns ``None`` when *text* is not an ``rdf{ }`` formula, so other formula
    parsers can try it. Raises :class:`RDFSyntaxError` when the delimiter is
    opened but the braces, strings, or IRIs do not close, or when text follows
    the closing brace.
    """
    source = text.strip()
    if not source.startswith(_RDF_OPEN):
        return None
    depth = 1
    index = len(_RDF_OPEN)
    while index < len(source):
        char = source[index]
        if char == "#":
            index = _consume_comment(source, index)
            continue
        if char == "<":
            index = _consume_iri(source, index)
            continue
        if char in "\"'":
            index = _consume_string(source, index)
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                rest = source[index + 1 :].strip()
                if rest:
                    raise RDFSyntaxError(f"trailing text after rdf{{ }}: {rest!r}")
                return source[len(_RDF_OPEN) : index]
        index += 1
    raise RDFSyntaxError("unclosed rdf{ }")


def split_rdf_lambda(text: str) -> tuple[str, str] | None:
    """If *text* is ``G^body`` or ``G1^body``, return ``(binder, body)``."""
    match = _BINDER.match(text.strip())
    if match is None:
        return None
    return match.group(1), match.group(2).strip()


def _consume_comment(source: str, index: int) -> int:
    """Advance past a Turtle ``#`` comment starting at *index*."""
    while index < len(source) and source[index] != "\n":
        index += 1
    return index


def _consume_iri(source: str, index: int) -> int:
    """Advance past a Turtle IRI ``<...>`` starting at *index*."""
    index += 1
    while index < len(source) and source[index] != ">":
        index += 1
    if index >= len(source):
        raise RDFSyntaxError("unclosed IRI in rdf{ }")
    return index + 1


def _consume_string(source: str, index: int) -> int:
    """Advance past a Turtle quoted string starting at *index*."""
    if source.startswith('"""', index) or source.startswith("'''", index):
        quote = source[index : index + 3]
        index += 3
        while index < len(source) and not source.startswith(quote, index):
            if source[index] == "\\":
                index += 2
                continue
            index += 1
        if index >= len(source):
            raise RDFSyntaxError("unclosed Turtle string in rdf{ }")
        return index + 3
    quote = source[index]
    index += 1
    while index < len(source) and source[index] != quote:
        if source[index] == "\\":
            index += 2
            continue
        index += 1
    if index >= len(source):
        raise RDFSyntaxError("unclosed Turtle string in rdf{ }")
    return index + 1
