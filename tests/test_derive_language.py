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


def test_derive_language_layered_parallel_matches_sequential(tmp_path: Path) -> None:
    """Multi-process layered BFS matches single-process output on the minimal fixture."""
    parser = InteractiveContextParser(FIXTURE)
    out_seq = tmp_path / "seq"
    out_par = tmp_path / "par"
    out_seq.mkdir()
    out_par.mkdir()
    paths_seq = parser.derive_language_layered(max_len=2, out_dir=out_seq, max_workers=1)
    paths_par = parser.derive_language_layered(max_len=2, out_dir=out_par, max_workers=2)
    for layer in paths_seq:
        lang_s, fail_s, fr_s = paths_seq[layer]
        lang_p, fail_p, fr_p = paths_par[layer]
        assert lang_s.read_text(encoding="utf-8") == lang_p.read_text(encoding="utf-8")
        assert fail_s.read_text(encoding="utf-8") == fail_p.read_text(encoding="utf-8")
        assert fr_s.read_text(encoding="utf-8") == fr_p.read_text(encoding="utf-8")


def test_derive_language_layered_writes_per_layer_files(tmp_path: Path) -> None:
    """Layered derivation creates ``layer_i`` language and failure files."""
    parser = InteractiveContextParser(FIXTURE)
    name = Path(FIXTURE).name

    paths = parser.derive_language_layered(max_len=1, out_dir=tmp_path)

    assert paths[1][0] == tmp_path / f"{name}_layer_1_language.txt"
    assert paths[1][1] == tmp_path / f"{name}_layer_1_language_failures.txt"
    assert paths[1][2] == tmp_path / f"{name}_layer_1_fringe.txt"
    assert paths[1][0].read_text(encoding="utf-8") == ""
    assert paths[1][1].read_text(encoding="utf-8") == ""
    assert paths[1][2].read_text(encoding="utf-8") == "test\n"


def test_derive_language_layered_category_matches_single_template_fixture(tmp_path: Path) -> None:
    """Category mode still derives when the fixture uses one lexical template."""
    parser = InteractiveContextParser(FIXTURE)
    name = Path(FIXTURE).name

    paths = parser.derive_language_layered_category(max_len=1, out_dir=tmp_path)

    assert paths[1][0].name == f"{name}_layer_1_language.txt"
    assert paths[1][1].read_text(encoding="utf-8") == ""
    assert paths[1][2].name == f"{name}_layer_1_fringe.txt"
    assert paths[1][2].read_text(encoding="utf-8") == "test\n"


def test_derive_language_layered_random_reproducible_with_seed(tmp_path: Path) -> None:
    """Random layered derivation is deterministic for a fixed seed and grammar."""
    parser = InteractiveContextParser(FIXTURE)
    name = Path(FIXTURE).name

    out_a = parser.derive_language_layered_random(
        max_len=1,
        max_paths=3,
        seed=12345,
        out_dir=tmp_path / "a",
    )
    out_b = parser.derive_language_layered_random(
        max_len=1,
        max_paths=3,
        seed=12345,
        out_dir=tmp_path / "b",
    )

    assert out_a[1][1].read_text(encoding="utf-8") == out_b[1][1].read_text(encoding="utf-8")
    assert out_a[1][1].name == f"{name}_layer_1_language_failures.txt"
    assert out_a[1][2].read_text(encoding="utf-8") == ""
    assert out_b[1][2].read_text(encoding="utf-8") == ""


def test_derive_language_layered_output_suffix_when_layer_files_exist(tmp_path: Path) -> None:
    """Layered run bumps numeric suffix when any layer output path collides."""
    name = Path(FIXTURE).name
    (tmp_path / f"{name}_layer_1_language.txt").write_text("x", encoding="utf-8")
    (tmp_path / f"{name}_layer_1_language_failures.txt").write_text("y", encoding="utf-8")
    (tmp_path / f"{name}_layer_1_fringe.txt").write_text("z", encoding="utf-8")
    (tmp_path / f"{name}_layer_2_language.txt").write_text("x", encoding="utf-8")
    (tmp_path / f"{name}_layer_2_language_failures.txt").write_text("y", encoding="utf-8")
    (tmp_path / f"{name}_layer_2_fringe.txt").write_text("z", encoding="utf-8")

    parser = InteractiveContextParser(FIXTURE)
    paths = parser.derive_language_layered(max_len=2, out_dir=tmp_path)

    assert paths[1][0] == tmp_path / f"{name}_layer_1_language_1.txt"
    assert paths[1][2] == tmp_path / f"{name}_layer_1_fringe_1.txt"
    assert paths[2][0] == tmp_path / f"{name}_layer_2_language_1.txt"
