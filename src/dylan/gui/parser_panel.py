"""Headless compatibility stub for Java ``ParserPanel``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ParserPanel:
    """Placeholder GUI panel object accepted by parser constructor-style APIs."""

    parser: Any | None = None

    def set_parser(self, parser: Any) -> None:
        """Attach a parser instance to this panel."""
        self.parser = parser


ParserPanel.setParser = ParserPanel.set_parser  # type: ignore[attr-defined]
