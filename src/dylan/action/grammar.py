"""Computational grammar from ``computational-actions.txt`` (Java ``Grammar``)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from dylan.action.computational_action import ComputationalAction
from dylan.action.lexicon import strip_block_comments

logger = logging.getLogger(__name__)

_URL_PREFIX = re.compile(r"^https?://", re.IGNORECASE)
_FILE_PREFIX = re.compile(r"^file:", re.IGNORECASE)


class Grammar(dict[str, ComputationalAction]):
    """Map of computational action name → ``ComputationalAction``."""

    FILE_NAME = "computational-actions.txt"

    def __init__(self, dir_or_url: str | Path | None = None) -> None:
        super().__init__()
        if dir_or_url is None:
            return
        self._load_from_disk(dir_or_url)

    def _load_from_disk(self, dir_or_url: str | Path) -> None:
        """Read computational-actions.txt from *dir_or_url*."""
        s = str(dir_or_url)
        if _URL_PREFIX.match(s) or _FILE_PREFIX.match(s):
            logger.warning("URL grammar loading not implemented for %s", s)
            return
        path = Path(dir_or_url) / self.FILE_NAME
        if not path.is_file():
            logger.error("Missing computational actions file %s", path)
            return
        cleaned = strip_block_comments(path.read_text(encoding="utf-8").splitlines())
        self._init_actions(cleaned)
        logger.info("Loaded %s computational actions", len(self))

    def _init_actions(self, cleaned_lines: list[str | None]) -> None:
        """Parse computational-action blocks (already block-comment-stripped)."""
        name: str | None = None
        always_good = False
        backtrack_on_success = False
        lines: list[str] = []
        for raw in cleaned_lines:
            if raw is None:
                continue
            line = raw.strip()
            if not line and not lines:
                continue
            if not line and lines:
                if name is not None:
                    self._commit(name, lines, always_good, backtrack_on_success)
                name = None
                lines = []
                always_good = False
                backtrack_on_success = False
                continue
            if name is None:
                name = line
                always_good = name.startswith("*")
                if always_good:
                    name = name[1:]
                backtrack_on_success = name.startswith("+")
                if backtrack_on_success:
                    name = name[1:]
                logger.debug(
                    "New computational action: %s (always_good=%s, backtrack_on_success=%s)",
                    name, always_good, backtrack_on_success,
                )
            else:
                lines.append(line)
        if name is not None and lines:
            self._commit(name, lines, always_good, backtrack_on_success)

    def _commit(
        self,
        name: str,
        lines: list[str],
        always_good: bool,
        backtrack_on_success: bool,
    ) -> None:
        action = ComputationalAction(name, lines, always_good, backtrack_on_success)
        self[name] = action
        logger.debug("Added computational action %s", action)
