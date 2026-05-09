"""Bundled grammar discovery, optional filesystem paths, and module-level parser session."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from importlib import resources
from pathlib import Path
from typing import Any, TypeAlias

from dylan.parser.interactive_context_parser import InteractiveContextParser

# Public nicknames → directory name under ``dynamicsyntax/grammars/<name>``.
_GRAMMAR_ALIASES: dict[str, str] = {
    "ttr": "2015-english-ttr",
}

_AsFileCtx: TypeAlias = AbstractContextManager[Path]

# When a bundled grammar is loaded via :func:`load_grammar`, we keep ``as_file`` entered so
# wheel extract paths stay valid until the next load.
_bundled_as_file_cm: _AsFileCtx | None = None
_session_parser: InteractiveContextParser | None = None


def _is_grammar_dir(node: Any) -> bool:
    """Return whether an importlib resource node has the core grammar files."""
    return (
        node.is_dir()
        and (node / "lexicon.txt").is_file()
        and (node / "computational-actions.txt").is_file()
    )


def _canonical_grammar_id(name: str) -> str:
    """Map a user-facing grammar id or alias to the bundled directory name."""
    return _GRAMMAR_ALIASES.get(name, name)


def _bundled_package_grammar_traversable(canonical_id: str) -> Any | None:
    """Return traversable grammar dir: ``dynamicsyntax/grammars/<id>`` or repo ``resources/<id>`` package.

    ``grammars/<id>`` wins when both exist. Repo-root grammars are packaged as :mod:`dynamicsyntax.resources`.
    """
    root = resources.files("dynamicsyntax")
    grammars = root / "grammars" / canonical_id
    if _is_grammar_dir(grammars):
        return grammars
    res_root = resources.files("dynamicsyntax.resources")
    res = res_root / canonical_id
    return res if _is_grammar_dir(res) else None


def _iter_bundled_grammar_names(container: str) -> Iterator[str]:
    """Yield subdirectory names under packaged ``dynamicsyntax/<container>/`` (legacy ``grammars/`` only)."""
    root = resources.files("dynamicsyntax")
    node = root / container
    if not node.is_dir():
        return iter(())
    return (child.name for child in node.iterdir() if _is_grammar_dir(child))


def _iter_repo_resources_grammar_names() -> Iterator[str]:
    """Yield grammar directory names shipped under :mod:`dynamicsyntax.resources` (repo ``resources/``)."""
    root = resources.files("dynamicsyntax.resources")
    if not root.is_dir():
        return iter(())
    return (child.name for child in root.iterdir() if _is_grammar_dir(child))


def get_grammars() -> list[str]:
    """Return sorted ids for bundled grammars plus known aliases (e.g. ``\"ttr\"``)."""
    names: set[str] = set(_GRAMMAR_ALIASES.keys())
    names.update(_iter_bundled_grammar_names("grammars"))
    names.update(_iter_repo_resources_grammar_names())
    return sorted(names)


def get_datasets() -> list[str]:
    """Return bundled dataset identifiers; empty until datasets ship with the package."""
    return []


def _exit_bundled_context() -> None:
    """Release any active ``resources.as_file`` context from a prior :func:`load_grammar`."""
    global _bundled_as_file_cm
    if _bundled_as_file_cm is not None:
        _bundled_as_file_cm.__exit__(None, None, None)
        _bundled_as_file_cm = None


def load_grammar(grammar: str | Path, *, repairing: bool = False) -> None:
    """Load a grammar directory into the module-level parser (used by :func:`parse` with no grammar arg).

    :param grammar: Bundled id or alias (e.g. ``\"2015-english-ttr\"``, ``\"ttr\"``) or a filesystem
        directory path containing lexicon / grammar files.
    :param repairing: Passed to :class:`~dylan.parser.interactive_context_parser.InteractiveContextParser`.
    :raises FileNotFoundError: If a bundled id is unknown or a path is not a directory.
    """
    global _session_parser, _bundled_as_file_cm
    _exit_bundled_context()
    _session_parser = None

    if isinstance(grammar, Path):
        path = grammar
        if not path.is_dir():
            raise FileNotFoundError(f"not a grammar directory: {path}")
        parser = InteractiveContextParser(path, repairing=repairing)
        parser.init()
        _session_parser = parser
        return

    p = Path(grammar.strip())
    if p.is_dir():
        parser = InteractiveContextParser(p, repairing=repairing)
        parser.init()
        _session_parser = parser
        return

    canonical = _canonical_grammar_id(grammar.strip())
    node = _bundled_package_grammar_traversable(canonical)
    if node is None:
        known = ", ".join(get_grammars()) or "(none)"
        raise FileNotFoundError(
            f"unknown grammar {grammar!r} (resolved {canonical!r}); known: {known}",
        )
    cm = resources.as_file(node)
    path = cm.__enter__()
    _bundled_as_file_cm = cm
    try:
        parser = InteractiveContextParser(path, repairing=repairing)
        parser.init()
        _session_parser = parser
    except BaseException:
        _exit_bundled_context()
        raise


def session_parser() -> InteractiveContextParser | None:
    """Return the parser from the last successful :func:`load_grammar`, if any."""
    return _session_parser


def clear_grammar_session() -> None:
    """Clear the module-level parser and any active bundled grammar extract (for tests or tooling)."""
    global _session_parser
    _exit_bundled_context()
    _session_parser = None


@contextmanager
def resolved_grammar_path(grammar: str | Path) -> Iterator[Path]:
    """Yield a filesystem path to *grammar* (bundled or directory), exiting wheel extract on close."""
    if isinstance(grammar, Path):
        if not grammar.is_dir():
            raise FileNotFoundError(f"not a grammar directory: {grammar}")
        yield grammar
        return

    p = Path(str(grammar).strip())
    if p.is_dir():
        yield p
        return

    canonical = _canonical_grammar_id(str(grammar).strip())
    node = _bundled_package_grammar_traversable(canonical)
    if node is None:
        known = ", ".join(get_grammars()) or "(none)"
        raise FileNotFoundError(
            f"unknown grammar {grammar!r} (resolved {canonical!r}); known: {known}",
        )
    with resources.as_file(node) as path:
        yield path
