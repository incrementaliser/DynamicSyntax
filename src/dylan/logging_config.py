"""Central logging setup (replaces log4j usage in Java)."""

from __future__ import annotations

import logging
from typing import Final

_DEFAULT_FORMAT: Final = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for CLI/tests."""
    logging.basicConfig(level=level, format=_DEFAULT_FORMAT)
