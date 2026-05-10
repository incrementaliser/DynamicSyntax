"""Tests for bounded language derivation output."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from dylan.parser.interactive_context_parser import InteractiveContextParser
from dylan.parser.language_derivation import LanguageDerivation

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "parser_minimal"
GRAMMAR_2026 = Path(__file__).resolve().parents[1] / "resources" / "2026-english-ttr-test"


def test_derive_language_writes_completed_success_and_failures(tmp_path: Path) -> None:
    """Derivation records only completed parses as successes and logs failures."""
    if not (GRAMMAR_2026 / "lexicon.txt").is_file():
        pytest.skip("2026-english-ttr-test grammar not in resources")
    parser = InteractiveContextParser(GRAMMAR_2026)

    language_path, failures_path = parser.derive_language(
        min_len=4,
        max_len=4,
        max_candidates=256,
        max_successful=1,
        out_dir=tmp_path,
        max_workers=1,
    )

    language_text = language_path.read_text(encoding="utf-8")
    failures_text = failures_path.read_text(encoding="utf-8")
    assert language_path == tmp_path / "2026-english-ttr-test_language.txt"
    assert failures_path == tmp_path / "2026-english-ttr-test_language_failures.txt"
    assert "Sent: a man knows you\n" in language_text
    assert "Sem: " in language_text
    assert "?ty(t)" not in language_text
    assert " | " in failures_text


def test_derive_language_completion_runtime_error_becomes_failure_line(tmp_path: Path) -> None:
    """If ``complete_tree`` raises, derivation logs a failure line and continues."""
    parser = InteractiveContextParser(FIXTURE)
    with patch.object(
        InteractiveContextParser,
        "complete_tree",
        side_effect=RuntimeError("beta-reduce: formula at down1 is not a lambda: None"),
    ):
        _lang, fail_path = LanguageDerivation(parser).run(
            max_len=1,
            min_len=1,
            max_candidates=1,
            max_workers=1,
            out_dir=tmp_path,
        )
    assert fail_path.read_text(encoding="utf-8") == "<<completion-aborted>> | test\n"


def test_derive_language_output_path_increments_when_base_exists(tmp_path: Path) -> None:
    """If default output filenames exist, the next numeric suffix is used for both files."""
    name = Path(FIXTURE).name
    (tmp_path / f"{name}_language.txt").write_text("x", encoding="utf-8")
    (tmp_path / f"{name}_language_failures.txt").write_text("y", encoding="utf-8")
    parser = InteractiveContextParser(FIXTURE)

    lang_path, fail_path = parser.derive_language(
        max_len=1,
        min_len=1,
        max_candidates=1,
        max_workers=1,
        out_dir=tmp_path,
    )

    assert lang_path == tmp_path / f"{name}_language_1.txt"
    assert fail_path == tmp_path / f"{name}_language_failures_1.txt"
    assert lang_path.is_file() and fail_path.is_file()


def test_derive_language_parallel_smoke_writes_incomplete_failure(tmp_path: Path) -> None:
    """Parallel derivation uses worker parsers and reports incomplete candidates."""
    parser = InteractiveContextParser(FIXTURE)

    language_path, failures_path = parser.derive_language(
        max_len=1,
        max_candidates=1,
        out_dir=tmp_path,
        max_workers=2,
    )

    assert language_path.read_text(encoding="utf-8") == ""
    assert failures_path.read_text(encoding="utf-8") == "<<incomplete>> | test\n"
