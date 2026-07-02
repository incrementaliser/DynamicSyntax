"""Morphological and spelling fallbacks for embedding lookup (ported from jolli.py)."""

from __future__ import annotations

import re

from dylan.vss.types import NOUN_SUFFIX, VERB_SUFFIX

_IS_VERB_RE = re.compile(r"is(e|es|ing|ed)\b")
_OFFENCE_RE = re.compile(r"offence\b")
_FAVOUR_RE = re.compile(r"favour\b")
_HYPHEN_RE = re.compile(r"-")


def alternate_keys(key: str, suffix: str) -> list[str]:
    """Return alternate lookup keys to try when *key*+suffix is missing in the space."""
    candidates: list[str] = []
    if _IS_VERB_RE.search(key) and suffix == VERB_SUFFIX:
        candidates.append(_IS_VERB_RE.sub(r"iz\1", key))
    if _OFFENCE_RE.search(key):
        candidates.append(_OFFENCE_RE.sub("offense", key))
    if _FAVOUR_RE.search(key):
        candidates.append(_FAVOUR_RE.sub("favor", key))
    if _HYPHEN_RE.search(key):
        candidates.append(re.sub(r"(\w+)-.*", r"\1", key))
    return candidates


def hack_morphology(verb: str, suffix: str) -> str:
    """Apply English morphological hacks used in jolli baseline experiments."""
    if suffix == "d":
        stem = re.sub(
            r"depositt$",
            "deposit",
            re.sub(r"([aeiou](t|l|p))$", r"\1\2", verb),
        )
    else:
        stem = verb
    stem = re.sub(r"(ch|s|sh|x|z)$", r"\1e", re.sub(r"(y)$", "ie", stem))
    if suffix == "d":
        stem = re.sub(r"([^aeiou])$", r"\1e", stem)
    return stem + suffix


def lookup_keys(word: str, *, noun: bool = True) -> list[str]:
    """Ordered keys to try in an embedding store for *word*."""
    suffix = NOUN_SUFFIX if noun else VERB_SUFFIX
    keys = [word + suffix, word]
    keys.extend(alternate_keys(word, suffix))
    return keys
