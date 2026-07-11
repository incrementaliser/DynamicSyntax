"""Timing helpers for induction runs."""

from __future__ import annotations


def format_hh_mm_ss(seconds: float) -> str:
    """Format *seconds* as ``HH-MM-SS`` (zero-padded)."""
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}-{minutes:02d}-{secs:02d}"
