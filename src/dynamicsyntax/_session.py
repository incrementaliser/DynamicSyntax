"""Bundled grammar discovery and filesystem resolution for grammar directories."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Any

# Public nicknames → directory name under ``dynamicsyntax/grammars/<name>``.
_GRAMMAR_ALIASES: dict[str, str] = {
    "ttr": "2015-english-ttr",
}


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
