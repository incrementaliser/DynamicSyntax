"""Tests for lexicon-backed VSS indexing."""

from __future__ import annotations

from pathlib import Path

from dylan.parser.interactive_context_parser import InteractiveContextParser
from dylan.vss.lexicon_vss import LexicalVSSRole, LexiconVSSIndex

_VSS_GRAMMAR = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "dylan"
    / "vss"
    / "resources"
    / "vss-transitive"
)


def test_lexicon_vss_index_lookup() -> None:
    """Indexed lexical actions expose embedding keys and coarse roles."""
    parser = InteractiveContextParser.from_resource_dir(_VSS_GRAMMAR)
    index = LexiconVSSIndex(parser.lexicon)
    bindings = index.lookup_by_word("draw")
    assert bindings
    assert bindings[0].embedding_key == "draw"
    assert bindings[0].role in (LexicalVSSRole.verb, LexicalVSSRole.noun, LexicalVSSRole.other)
