"""Small string helpers replacing trimmed Java `qmul.util` usage."""


def casefold_equal(a: str, b: str) -> bool:
    """Return True if `a` and `b` are the same modulo Unicode case-folding."""
    return a.casefold() == b.casefold()
