"""Setuptools entry: discover packages under ``src/`` plus ``dynamicsyntax.resources`` from repo ``resources/``."""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

_ROOT = Path(__file__).resolve().parent


def _packages() -> list[str]:
    """Return every package under ``src/`` plus :mod:`dynamicsyntax.resources`."""
    names = set(find_packages(where=str(_ROOT / "src")))
    names.add("dynamicsyntax.resources")
    return sorted(names)


setup(
    packages=_packages(),
    package_dir={
        "": "src",
        "dynamicsyntax.resources": "resources",
    },
)
