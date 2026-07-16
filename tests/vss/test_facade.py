"""Integration tests: DS-VSS decoration of real DyLan parse trees."""

import numpy as np
import pytest

from dynamicsyntax.vss import (
    RequirementMode,
    VSSDirectSum,
    VSSLexicon,
    VSSParseResult,
    VectorSpace,
    parse_vss,
    vss_plausibility,
)

W = VectorSpace("W", 4, ("people", "affection", "motion", "action"))


@pytest.fixture()
def lexicon() -> VSSLexicon:
    lex = VSSLexicon(word_space=W)
    lex.add_entity("john", [5, 4, 1, 1])
    lex.add_entity("mary", [5, 4, 1, 1])
    lex.add_intransitive("arrive", [[6, 1], [1, 2], [8, 1], [2, 3]])
    cube = np.zeros((4, 2, 4))
    for i in (0, 1):
        for k in (0, 1):
            cube[i, 0, k] = 8.0
    cube[:, 1, :] = 1.0
    lex.add_transitive("like", cube)
    return lex


def test_parse_vss_transitive(lexicon):
    r = parse_vss("john likes mary", "ttr", lexicon=lexicon)
    assert isinstance(r, VSSParseResult)
    assert r.ok
    # one trace tree per word plus the initial axiom tree
    assert len(r.decorations) == 4
    assert len(r.plausibilities) == 4
    # final tree node tensor types match the paper's Figure 2
    final = r.decorations[-1].values
    assert final["00"].space_names() == ("W",)
    assert final["010"].space_names() == ("W",)
    assert final["011"].space_names() == ("W", "S", "W")
    assert final["01"].space_names() == ("W", "S")
    assert final["0"].space_names() == ("S",)
    # the root carries a real sentence vector with positive plausibility
    assert r.final_plausibility is not None
    assert 0.0 < r.final_plausibility <= 1.0


def test_parse_vss_intransitive(lexicon):
    r = parse_vss("john arrives", "ttr", lexicon=lexicon)
    assert r.ok
    assert r.final_plausibility is not None
    # incremental trajectory: neutral start, then subject-driven expectation
    traj = r.plausibilities
    assert traj[0] == pytest.approx(0.5)  # bare ?Ty(t) axiom tree
    assert all(p is not None for p in traj)


def test_plausibility_shortcut(lexicon):
    p = vss_plausibility("mary likes john", "ttr", lexicon=lexicon)
    assert p is not None and 0.0 < p <= 1.0


def test_batch_parse(lexicon):
    results = parse_vss(["john arrives", "john likes mary"], "ttr", lexicon=lexicon)
    assert len(results) == 2
    assert all(isinstance(r, VSSParseResult) for r in results)


def test_missing_predicates_recorded():
    lex = VSSLexicon(word_space=W)  # nothing registered
    lex.add_entity("john", [1, 0, 0, 0])
    r = parse_vss("john arrives", "ttr", lexicon=lex)
    assert "arrive" in r.missing_predicates


def test_requirement_modes(lexicon):
    r_sum = parse_vss("john likes mary", "ttr", lexicon=lexicon, requirement_mode="sum")
    r_unit = parse_vss("john likes mary", "ttr", lexicon=lexicon, requirement_mode="unit")
    r_dir = parse_vss(
        "john likes mary", "ttr", lexicon=lexicon, requirement_mode="direct_sum"
    )
    assert r_sum.final_plausibility is not None
    assert r_unit.final_plausibility is not None
    # direct sum keeps alternatives separate at the bare axiom tree
    first = r_dir.decorations[0].root_value
    assert isinstance(first, VSSDirectSum)
    assert RequirementMode("sum") is RequirementMode.SUM
