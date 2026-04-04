"""Word → lexical actions (Java ``Lexicon``)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection

from dylan.action.atomic.effect_factory import EffectFactory
from dylan.action.lexical_action import LexicalAction

logger = logging.getLogger(__name__)

_TEMPLATE_SPEC = re.compile(r"(.+?)\((.+)\)")
_LINE_COMMENT = "//"
_BEGIN_COMMENT = "//*"
_END_COMMENT = "*//"
_INLINE_BLOCK_RE = re.compile(
    re.escape(_BEGIN_COMMENT) + r".*?" + re.escape(_END_COMMENT),
)


def strip_block_comments(raw_lines: list[str]) -> list[str | None]:
    """Remove multi-line ``//*`` … ``*//`` block comments, mirroring Java ``Lexicon.comment`` state machine.

    Also strips single-line ``//`` comments and inline ``//*…*//`` blocks.

    Returns a list the same length as *raw_lines*.  Entries that were
    entirely inside a block comment or entirely a line comment become
    ``None`` (mirrors Java returning ``null``).  Genuine blank lines
    remain as ``""`` so that callers can use them as block delimiters.
    """
    result: list[str | None] = []
    in_block = False
    for raw in raw_lines:
        line = raw
        if in_block:
            if _END_COMMENT in line:
                in_block = False
                line = line[line.index(_END_COMMENT) + len(_END_COMMENT) :]
            else:
                result.append(None)
                continue

        line = _INLINE_BLOCK_RE.sub("", line)

        if _BEGIN_COMMENT in line:
            in_block = True
            line = line[: line.index(_BEGIN_COMMENT)]

        if _LINE_COMMENT in line:
            line = line[: line.index(_LINE_COMMENT)]

        stripped = line.rstrip()
        if not stripped and raw.strip():
            result.append(None)
        else:
            result.append(stripped)
    return result


def _single_line_comment_strip(line: str) -> str:
    """Strip ``//`` and inline ``//*…*//`` on one line only (no multi-line block state)."""
    s = _INLINE_BLOCK_RE.sub("", line)
    if s.strip() == _END_COMMENT:
        return ""
    if _LINE_COMMENT in s:
        s = s[: s.index(_LINE_COMMENT)]
    return s.rstrip()


def _referenced_template_names_from_lexicon(cleaned_lines: list[str | None]) -> set[str]:
    """Collect template identifiers (second column) from lexicon lines."""
    names: set[str] = set()
    for raw in cleaned_lines:
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            names.add(parts[1])
    return names


class Lexicon(dict[str, list[LexicalAction]]):
    """Maps surface words to a list of instantiated `LexicalAction` objects."""

    WORD_FILE_NAME = "lexicon.txt"
    ACTION_FILE_NAME = "lexical-actions.txt"
    MACRO_FILE_NAME = "lexical-macros.txt"

    def __init__(self, resource_dir: str | Path | None = None, _top_n: int = 3) -> None:
        super().__init__()
        self._templates: dict[str, _LexicalTemplate] = {}
        if resource_dir is None:
            return
        root = Path(resource_dir)
        macro_path = root / self.MACRO_FILE_NAME
        if macro_path.is_file():
            EffectFactory.init_macro_templates(
                strip_block_comments(macro_path.read_text(encoding="utf-8").splitlines()),
            )
        else:
            EffectFactory.clear_macro_templates()
        action_path = root / self.ACTION_FILE_NAME
        action_raw_lines: list[str] = []
        if action_path.is_file():
            action_raw_lines = action_path.read_text(encoding="utf-8").splitlines()
            self._init_lexical_templates(strip_block_comments(list(action_raw_lines)))
        word_path = root / self.WORD_FILE_NAME
        cleaned_word_lines: list[str | None] = []
        if word_path.is_file():
            cleaned_word_lines = strip_block_comments(
                word_path.read_text(encoding="utf-8").splitlines(),
            )
            _lexicon_recover_missing_templates(self, action_raw_lines, cleaned_word_lines)
            self._read_words(cleaned_word_lines)

    @staticmethod
    def strip_comment(line: str) -> str | None:
        """Strip a single ``//`` line comment (no block-comment tracking).

        For full block-comment support, use :func:`strip_block_comments` on
        the whole file first.
        """
        if not line:
            return line
        line = _INLINE_BLOCK_RE.sub("", line)
        if _LINE_COMMENT in line:
            line = line[: line.index(_LINE_COMMENT)]
        line = line.strip()
        return line if line else None

    def lookup(self, word: str) -> Collection[LexicalAction]:
        """Return lexical entries for `word`, or an empty sequence (Java `Lexicon.get`)."""
        return super().get(word, [])

    def _init_lexical_templates(self, cleaned_lines: list[str | None]) -> None:
        """Parse lexical-action template blocks (already block-comment-stripped)."""
        name: str | None = None
        metavars: list[str] = []
        lines: list[str] = []
        no_left_adjustment = False
        for raw in cleaned_lines:
            if raw is None:
                continue
            line = raw.strip()
            if not line and not lines:
                continue
            if not line and lines and name is not None:
                self._templates[name] = _LexicalTemplate(
                    name, metavars, list(lines), no_left_adjustment,
                )
                name = None
                metavars = []
                lines = []
                no_left_adjustment = False
            elif name is None:
                m = _TEMPLATE_SPEC.match(line)
                if not m:
                    raise ValueError(f"unrecognised template spec {line!r}")
                raw_name = m.group(1).strip()
                if raw_name.startswith("*"):
                    raw_name = raw_name[1:]
                    no_left_adjustment = True
                name = raw_name
                metavars = [x.strip() for x in m.group(2).split(",") if x.strip()]
            else:
                lines.append(line)
        if name is not None and lines:
            self._templates[name] = _LexicalTemplate(
                name, metavars, list(lines), no_left_adjustment,
            )
        logger.info("Read %s lexical action templates", len(self._templates))

    def _read_words(self, cleaned_lines: list[str | None]) -> None:
        """Parse lexicon word entries (already block-comment-stripped)."""
        for raw in cleaned_lines:
            if raw is None:
                continue
            line = raw.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 2:
                logger.warning("Skipping lexicon line (need word and template name): %s", line)
                continue
            word, template = fields[0], fields[1]
            lt = self._templates.get(template)
            if lt is None:
                logger.debug("No template %s, skipping word %s", template, word)
                continue
            metavals = fields[2:]
            if lt.metavar_count != len(metavals):
                logger.warning(
                    "Skipping lexicon line for %r template %r: expected %s metavar(s), found %s",
                    word,
                    template,
                    lt.metavar_count,
                    len(metavals),
                )
                continue
            try:
                action = lt.create(word, metavals)
            except (ValueError, RuntimeError) as ex:
                logger.warning("Could not instantiate lexical template %s for %s: %s", template, word, ex)
                continue
            self.setdefault(word, []).append(action)
        logger.info("Read lexicon with %s words.", len(self))


@dataclass
class _LexicalTemplate:
    """Template for `LexicalAction` bodies with metavar substitution."""

    name: str
    metavars: list[str]
    lines: list[str] = field(default_factory=list)
    no_left_adjustment: bool = False

    @property
    def metavar_count(self) -> int:
        return len(self.metavars)

    def create(self, word: str, metavals: list[str]) -> LexicalAction:
        if len(self.metavars) != len(metavals):
            raise ValueError("metavar count mismatch")
        out_lines: list[str] = []
        for template_line in self.lines:
            line = template_line
            for i, mv in enumerate(self.metavars):
                line = line.replace(mv, metavals[i])
            out_lines.append(line)
        return LexicalAction(word, out_lines, self.name, self.no_left_adjustment)


def _recover_template_from_raw_lines(raw_lines: list[str], name: str) -> _LexicalTemplate | None:
    """Parse one template block from *raw_lines* ignoring multi-line ``//*`` block state.

    Used when ``strip_block_comments`` removed a template (e.g. ``det_quant`` inside a
    deprecated block) but ``lexicon.txt`` still references it — matches how maintainers
    expect the bundled English grammar to work.
    """
    i = 0
    while i < len(raw_lines):
        sl = _single_line_comment_strip(raw_lines[i]).strip()
        if not sl:
            i += 1
            continue
        m = _TEMPLATE_SPEC.match(sl)
        if not m:
            i += 1
            continue
        raw_name = m.group(1).strip()
        no_left_adjustment = False
        if raw_name.startswith("*"):
            raw_name = raw_name[1:]
            no_left_adjustment = True
        if raw_name != name:
            i += 1
            continue
        metavars = [x.strip() for x in m.group(2).split(",") if x.strip()]
        body: list[str] = []
        i += 1
        while i < len(raw_lines):
            ln = _single_line_comment_strip(raw_lines[i])
            stripped = ln.strip()
            if not stripped and body:
                return _LexicalTemplate(name, metavars, body, no_left_adjustment)
            if stripped:
                body.append(stripped)
            i += 1
        if body:
            return _LexicalTemplate(name, metavars, body, no_left_adjustment)
        return None
    return None


def _lexicon_recover_missing_templates(
    lex: Lexicon,
    action_raw_lines: list[str],
    cleaned_word_lines: list[str | None],
) -> None:
    """If lexicon names a template dropped by block-comment stripping, recover it from raw actions."""
    if not action_raw_lines:
        return
    referenced = _referenced_template_names_from_lexicon(cleaned_word_lines)
    missing = referenced - set(lex._templates)
    for tpl_name in sorted(missing):
        rec = _recover_template_from_raw_lines(action_raw_lines, tpl_name)
        if rec is not None:
            lex._templates[tpl_name] = rec
            logger.info(
                "Recovered lexical template %r from raw lexical-actions (referenced by lexicon)",
                tpl_name,
            )
