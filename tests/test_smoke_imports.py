"""Package import smoke tests."""

from __future__ import annotations

import dylan
from dylan.parser.interactive_context_parser import InteractiveContextParser
from dylan.type.dstype import DSType


def test_version_and_imports() -> None:
    assert dylan.__version__
    assert DSType.t is not None
    assert InteractiveContextParser is not None
