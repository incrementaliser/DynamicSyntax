"""Word → lexical actions (Java ``Lexicon``)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import shorten
from typing import Collection, Literal, TextIO

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
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")

# Above this length, Jupyter HTML escape + browser layout can freeze the UI; use plain MIME or a short HTML notice.
NOTEBOOK_MULTILINE_HTML_MAX_CHARS = 80_000


def _mime_filter_allows_html(include: object, exclude: object) -> bool:
    """Return whether IPython ``include`` / ``exclude`` filters request a ``text/html`` representation."""
    if exclude is not None and "text/html" in exclude:
        return False
    if include is None:
        return True
    if isinstance(include, (set, frozenset, dict)):
        return "text/html" in include
    return True


class NotebookMultilineText(str):
    """Multiline string that Jupyter/IPython displays as HTML ``<pre>`` (preserves line breaks).

    Returned by :meth:`Lexicon.get_vocab` so notebook cells show formatted vocabulary instead of
    one long escaped line. Still a normal ``str`` for printing, files, and equality tests.
    """

    def _repr_html_(self) -> str:
        """HTML preformatted block; ANSI sequences from Rich backends are stripped for display."""
        import html

        plain = str(self)
        if len(plain) > NOTEBOOK_MULTILINE_HTML_MAX_CHARS:
            notice = (
                f"(Output is {len(plain)} characters; use print(...) or the text/plain MIME view for the full text.)"
            )
            body = html.escape(notice)
            return (
                '<pre style="white-space: pre-wrap; font-family: ui-monospace, Consolas, monospace; '
                'font-size: 0.88em; line-height: 1.45; margin: 0; overflow-x: auto;">'
                f"{body}</pre>"
            )
        visible = _ANSI_ESCAPE_RE.sub("", plain)
        body = html.escape(visible)
        return (
            '<pre style="white-space: pre-wrap; font-family: ui-monospace, Consolas, monospace; '
            'font-size: 0.88em; line-height: 1.45; margin: 0; overflow-x: auto;">'
            f"{body}</pre>"
        )

    def _repr_mimebundle_(
        self,
        include: object = None,
        exclude: object = None,
        **kwargs: object,
    ) -> dict[str, str]:
        """Expose HTML (preferred in notebooks) and plain text MIME; large bodies skip heavy HTML."""
        plain = str(self)
        bundle: dict[str, str] = {"text/plain": plain}
        if not _mime_filter_allows_html(include, exclude):
            return bundle
        if len(plain) > NOTEBOOK_MULTILINE_HTML_MAX_CHARS:
            return bundle
        bundle["text/html"] = self._repr_html_()
        return bundle


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


@dataclass(frozen=True)
class LexiconLoadStats:
    """Counts recorded while loading a grammar directory into a `Lexicon`."""

    word_entries_loaded: int
    """Successful lexicon lines (each adds one lexical entry)."""
    words_unique: int
    """Distinct surface forms with at least one loaded entry."""
    words_failed: int
    """Lexicon lines skipped (bad shape, unknown template, arity, instantiation)."""
    macros_loaded: int
    """Macro definitions stored in `EffectFactory` for this load."""
    macros_failed: int
    """Macro headers that never received a non-empty body."""
    words_failed_names: tuple[str, ...] = field(default_factory=tuple)
    """Surface words (or best label) for each failed lexicon line, in order."""
    macros_failed_names: tuple[str, ...] = field(default_factory=tuple)
    """Macro base names that failed to load (incomplete body), in order."""


class Lexicon(dict[str, list[LexicalAction]]):
    """Maps surface words to a list of instantiated `LexicalAction` objects."""

    WORD_FILE_NAME = "lexicon.txt"
    ACTION_FILE_NAME = "lexical-actions.txt"
    MACRO_FILE_NAME = "lexical-macros.txt"

    def __init__(self, resource_dir: str | Path | None = None, _top_n: int = 3) -> None:
        super().__init__()
        self.top_n = _top_n
        self._templates: dict[str, _LexicalTemplate] = {}
        self._resource_dir: Path | None = None
        self._load_stats = LexiconLoadStats(0, 0, 0, 0, 0)
        self._vocab_cache: dict[tuple[str, str, int | None], str] = {}
        if resource_dir is None:
            return
        root = Path(resource_dir)
        self._resource_dir = root
        macro_path = root / self.MACRO_FILE_NAME
        macro_loaded, macro_failed, macros_failed_names = 0, 0, ()
        if macro_path.is_file():
            macro_loaded, macro_failed, macros_failed_names = EffectFactory.init_macro_templates(
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
        word_entries, word_failed, words_failed_names = 0, 0, ()
        if word_path.is_file():
            cleaned_word_lines = strip_block_comments(
                word_path.read_text(encoding="utf-8").splitlines(),
            )
            _lexicon_recover_missing_templates(self, action_raw_lines, cleaned_word_lines)
            word_entries, word_failed, words_failed_names = self._read_words(cleaned_word_lines)
        self._load_stats = LexiconLoadStats(
            word_entries_loaded=word_entries,
            words_unique=len(self),
            words_failed=word_failed,
            macros_loaded=macro_loaded,
            macros_failed=macro_failed,
            words_failed_names=words_failed_names,
            macros_failed_names=macros_failed_names,
        )

    @property
    def load_stats(self) -> LexiconLoadStats:
        """Parse counters snapshot from the last grammar load (zeros if no ``resource_dir``)."""
        return self._load_stats

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
        entries = list(super().get(word, []))
        return entries[: self.top_n] if self.top_n > 0 else entries

    def get(self, word: str, default: object = None) -> Collection[LexicalAction]:  # type: ignore[override]
        """Return lexical entries for *word* using Java ``get`` semantics."""
        entries = self.lookup(word)
        if entries:
            return entries
        return [] if default is None else default  # type: ignore[return-value]

    def invalidate_vocab_cache(self) -> None:
        """Clear memoised `get_vocab` output (needed after mutating entries post-load)."""
        self._vocab_cache.clear()

    @staticmethod
    def _export_lines_for_action(act: object) -> list[str]:
        """Return reloadable IF/THEN lines for *act*, or a single-line fallback when no effect is present."""
        src = getattr(act, "_source_lines", None)
        if isinstance(src, list) and src:
            return [str(line) for line in src]
        get_eff = getattr(act, "get_effect", None)
        eff = get_eff() if callable(get_eff) else None
        if eff is not None:
            from dylan.induction.em_learner.lexicon_export import effect_to_lexical_lines

            return effect_to_lexical_lines(eff)
        return [str(act)]

    def write_to_text_file(self, path: str | Path, *, encoding: str = "utf-8") -> None:
        """Write learnt entries like Java ``Lexicon.writeToTextFile``: ``[prob,rank]``, word line, then body (``_source_lines`` for reloadable IF/THEN text)."""
        out = Path(path)
        chunks: list[str] = []
        for word in sorted(self.keys()):
            entries = [a for a in self[word] if a is not None]
            entries.sort(key=lambda la: float(getattr(la, "prob", 0.0)), reverse=True)
            for act in entries:
                prob = getattr(act, "prob", 1.0)
                rank = getattr(act, "rank", 0)
                chunks.append(f"[{prob},{rank}]")
                chunks.append(word)
                chunks.extend(self._export_lines_for_action(act))
                chunks.append("")
        text = "\n".join(chunks)
        if text:
            text += "\n"
        out.write_text(text, encoding=encoding)

    def get_vocab(
        self,
        groupby: Literal["category", "alpha"] = "category",
        *,
        stream: TextIO | None = None,
        backend: Literal["plain", "rich"] = "plain",
        max_cell_width: int | None = 120,
    ) -> NotebookMultilineText:
        """Surface words only in a fixed-width grid (five columns); with ``groupby='category'``, template names appear as ``=== … ===`` sections only.

        Pass *groupby* positionally (e.g. ``get_vocab("alpha")``) or by keyword. ``stream``, ``backend``,
        and ``max_cell_width`` remain keyword-only.

        Cached per `(groupby, backend, max_cell_width)` until `invalidate_vocab_cache`.
        In Jupyter/IPython, the result renders as a monospace block with line breaks (see
        :class:`NotebookMultilineText`); elsewhere it behaves like a normal string.
        """
        if groupby not in ("category", "alpha"):
            raise ValueError(
                "groupby must be 'category' or 'alpha' "
                f"(got {groupby!r}); use get_vocab('alpha') or get_vocab(groupby='alpha').",
            )
        if backend not in ("plain", "rich"):
            raise ValueError(
                "backend must be 'plain' or 'rich' "
                f"(got {backend!r}); use get_vocab(..., backend='plain') or backend='rich'.",
            )
        cache_key = (groupby, backend, max_cell_width)
        if cache_key in self._vocab_cache:
            text = self._vocab_cache[cache_key]
            if stream is not None:
                stream.write(text)
            return NotebookMultilineText(text)
        rows = _collect_vocab_rows(self)
        if backend == "plain":
            text = _format_vocab_plain(self, rows, groupby=groupby, max_cell_width=max_cell_width)
        else:
            text = _format_vocab_rich(self, rows, groupby=groupby, max_cell_width=max_cell_width)
        self._vocab_cache[cache_key] = text
        if stream is not None:
            stream.write(text)
        return NotebookMultilineText(text)

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

    def _read_words(self, cleaned_lines: list[str | None]) -> tuple[int, int, tuple[str, ...]]:
        """Parse lexicon word entries; returns entries loaded, failure count, and failed surface-word labels."""
        entries_ok = 0
        failed = 0
        failed_names: list[str] = []
        for raw in cleaned_lines:
            if raw is None:
                continue
            line = raw.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) < 2:
                logger.warning("Skipping lexicon line (need word and template name): %s", line)
                failed += 1
                failed_names.append(fields[0] if fields else "(malformed line)")
                continue
            word, template = fields[0], fields[1]
            lt = self._templates.get(template)
            if lt is None:
                logger.debug("No template %s, skipping word %s", template, word)
                failed += 1
                failed_names.append(word)
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
                failed += 1
                failed_names.append(word)
                continue
            try:
                action = lt.create(word, metavals)
            except (ValueError, RuntimeError) as ex:
                logger.warning(
                    "Could not instantiate lexical template %s for %s: %s", template, word, ex,
                )
                failed += 1
                failed_names.append(word)
                continue
            entries_ok += 1
            logger.info('Created lexical action for "%s" with template "%s"', word, template)
            self.setdefault(word, []).append(action)
        logger.info("Read lexicon with %s words.", len(self))
        return entries_ok, failed, tuple(failed_names)


_VOCAB_GRID_COLUMNS = 5


def _vocab_grid_lines(cells: list[str]) -> list[str]:
    """Join display-ready word strings into grid lines (``_VOCAB_GRID_COLUMNS`` words per line)."""
    if not cells:
        return []
    col_w = max(len(c) for c in cells)
    lines: list[str] = []
    for i in range(0, len(cells), _VOCAB_GRID_COLUMNS):
        row = cells[i : i + _VOCAB_GRID_COLUMNS]
        padded = [c.ljust(col_w) for c in row]
        lines.append("  ".join(padded))
    return lines


def _truncate_cell(text: str, max_cell_width: int | None) -> str:
    """Shorten a table cell when *max_cell_width* is set (``None`` = no limit)."""
    if not text:
        return ""
    if max_cell_width is None or len(text) <= max_cell_width:
        return text
    return shorten(text, width=max_cell_width, placeholder="…")


def _collect_vocab_rows(lex: Lexicon) -> list[tuple[str, str]]:
    """Build ``(word, category)`` rows from loaded lexical entries (template name = category)."""
    rows: list[tuple[str, str]] = []
    for word in sorted(lex.keys()):
        for la in lex[word]:
            cat = la.action_type or ""
            rows.append((word, cat))
    return rows


_FAILED_WORDS_PER_LINE = 10


def _wrapped_comma_name_lines(names: tuple[str, ...], *, per_line: int, indent: str) -> list[str]:
    """Format *names* as comma-separated groups with at most *per_line* names per line, each prefixed by *indent*."""
    if not names:
        return []
    seq = list(names)
    return [f"{indent}{', '.join(seq[i : i + per_line])}" for i in range(0, len(seq), per_line)]


def _format_stats_header(lex: Lexicon) -> str:
    """Render grammar id line, path to ``lexicon.txt``, load statistics, and optional failure name lists."""
    st = lex._load_stats
    lines: list[str] = []
    if lex._resource_dir is not None:
        root = lex._resource_dir
        word_file = (root / Lexicon.WORD_FILE_NAME).resolve()
        lines.append(f"Lexicon: {root.name}")
        lines.append(f"Source: {word_file}")
        lines.append("")
    lines.extend(
        [
            "Load statistics:",
            f"  Word entries loaded:    {st.word_entries_loaded}",
            f"  Unique words:           {st.words_unique}",
            f"  Words failed:           {st.words_failed}",
            f"  Macros loaded:          {st.macros_loaded}",
            f"  Macros failed:          {st.macros_failed}",
            "",
        ],
    )
    if st.words_failed != 0:
        lines.append("Failed words:")
        if st.words_failed_names:
            lines.extend(_wrapped_comma_name_lines(st.words_failed_names, per_line=_FAILED_WORDS_PER_LINE, indent="  "))
        else:
            lines.append("  —")
        lines.append("")
    if st.macros_failed != 0:
        lines.append("Failed macros:")
        if st.macros_failed_names:
            lines.append(f"  {', '.join(st.macros_failed_names)}")
        else:
            lines.append("  —")
        lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    lines.append("")
    return "\n".join(lines)


def _format_vocab_plain(
    lex: Lexicon,
    rows: list[tuple[str, str]],
    *,
    groupby: Literal["category", "alpha"],
    max_cell_width: int | None,
) -> str:
    """Format vocabulary as a grid of words (category only in section titles when grouped)."""
    parts: list[str] = [_format_stats_header(lex)]
    if not rows:
        parts.append("(no lexical entries loaded)")
        return "\n".join(parts)

    disp: list[tuple[str, str]] = [
        (_truncate_cell(w, max_cell_width), _truncate_cell(c, max_cell_width)) for w, c in rows
    ]

    if groupby == "alpha":
        disp.sort(key=lambda t: (t[0], t[1]))
        words_only = [w for w, _ in disp]
        parts.extend(_vocab_grid_lines(words_only))
        return "\n".join(parts)

    by_cat: dict[str, list[tuple[str, str]]] = {}
    for w, c in disp:
        by_cat.setdefault(c, []).append((w, c))
    for cat in sorted(by_cat.keys(), key=lambda x: (x == "", x)):
        parts.append(f"=== {cat or '(empty category)'} ===")
        group_rows = sorted(by_cat[cat], key=lambda t: (t[0],))
        words_only = [w for w, _ in group_rows]
        parts.extend(_vocab_grid_lines(words_only))
        parts.append("")
    while parts and parts[-1] == "":
        parts.pop()
    return "\n".join(parts)


def _format_vocab_rich(
    lex: Lexicon,
    rows: list[tuple[str, str]],
    *,
    groupby: Literal["category", "alpha"],
    max_cell_width: int | None,
) -> str:
    """Render vocabulary as a word grid (Rich optional dependency)."""
    try:
        from rich.console import Console
        from rich.markup import escape
        from rich.table import Table
    except ImportError as exc:
        raise ImportError(
            "backend='rich' requires the 'rich' package (pip install rich or pip install "
            "dynamicsyntax[rich]).",
        ) from exc

    from io import StringIO

    buf = StringIO()
    header = _format_stats_header(lex)
    if not rows:
        return "\n".join([header, "(no lexical entries loaded)"])

    console = Console(
        file=buf,
        force_terminal=False,
        width=160,
        highlight=False,
        markup=False,
    )
    buf.write(header)
    buf.write("\n")

    def emit_vocab_grid(display_words: list[str]) -> None:
        """Print pre-formatted words in a borderless grid (``_VOCAB_GRID_COLUMNS`` columns)."""
        if not display_words:
            return
        table = Table(show_header=False, box=None, pad_edge=False, collapse_padding=True)
        for _ in range(_VOCAB_GRID_COLUMNS):
            table.add_column(justify="left")
        for i in range(0, len(display_words), _VOCAB_GRID_COLUMNS):
            chunk = display_words[i : i + _VOCAB_GRID_COLUMNS]
            padded = list(chunk) + [""] * (_VOCAB_GRID_COLUMNS - len(chunk))
            table.add_row(*padded[:_VOCAB_GRID_COLUMNS])
        console.print(table, highlight=False)

    disp = list(rows)
    if groupby == "alpha":
        disp.sort(key=lambda t: (t[0], t[1]))
        emit_vocab_grid([w for w, _ in disp])
        return buf.getvalue()

    by_cat: dict[str, list[tuple[str, str]]] = {}
    for w, c in disp:
        by_cat.setdefault(c, []).append((w, c))
    for cat in sorted(by_cat.keys(), key=lambda x: (x == "", x)):
        label = cat or "(empty category)"
        console.print(f"[bold]{escape(label)}[/bold]", highlight=False, markup=True)
        group_rows = sorted(by_cat[cat], key=lambda t: (t[0],))
        emit_vocab_grid([w for w, _ in group_rows])
        console.print("")
    return buf.getvalue()


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
