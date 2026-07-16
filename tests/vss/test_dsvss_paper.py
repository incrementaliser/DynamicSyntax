"""Replication of the worked example of Sadrzadeh, Purver, Hough & Kempson
(2018), "Exploring Semantic Incrementality with Dynamic Syntax and Vector
Space Semantics", section 4–5 (arXiv:1811.00614).

The ``baby``/``footballer`` plausibility model: a 4-dimensional word space
with basis (infant, nappy, pitch, goal) and the 2-dimensional plausibility
sentence space with basis (⊤, ⊥).
"""

import numpy as np
import pytest

from dylan.type.dstype import DSType
from dylan.vss import (
    RequirementMode,
    VSSDecorator,
    VSSLexicon,
    VectorSpace,
    contract,
    object_continuations,
    plausibility,
    plausibility_space,
    verb_continuations,
)

BASIS = ("infant", "nappy", "pitch", "goal")


@pytest.fixture()
def lexicon() -> VSSLexicon:
    lex = VSSLexicon(word_space=VectorSpace("W", 4, BASIS), sentence_space=plausibility_space())
    # entity vectors (co-occurrence counts)
    lex.add_entity("babies", [34, 10, 0, 0])
    lex.add_entity("footballers", [0, 0, 24, 17])
    lex.add_entity("balls", [0, 0, 10, 10])
    lex.add_entity("milk", [0, 5, 0, 0])
    # intransitive verb matrices, rows = W basis, columns = (⊤, ⊥)
    lex.add_intransitive("vomit", [[10, 2], [9, 3], [0, 30], [0, 25]])
    lex.add_intransitive("score", [[0, 8], [0, 6], [20, 1], [29, 1]])
    lex.add_intransitive("dribble", [[9, 2], [8, 2], [21, 5], [20, 4]])
    # sense vectors for the ambiguous "dribble" (section 5.1)
    lex.add_intransitive("dribble_drip", [[12, 1], [10, 1], [0, 20], [0, 18]])
    lex.add_intransitive("dribble_control", [[0, 20], [0, 18], [25, 2], [24, 2]])
    # transitive cube for "control": ⊤ only on (pitch|goal, ⊤, pitch|goal)
    cube = np.zeros((4, 2, 4))
    for i in (2, 3):
        for k in (2, 3):
            cube[i, 0, k] = 1.0
    lex.add_transitive("control", cube)
    return lex


def sent(lex, subj, verb):
    return contract(lex.lookup(verb, (lex.word_space, lex.sentence_space)),
                    lex.lookup(subj, (lex.word_space,)))


def test_paper_arithmetic_babies_vomit(lexicon):
    """Exact replication: T^babies_i T^vomit_ij = 430⊤ + 98⊥ (section 4)."""
    v = sent(lexicon, "babies", "vomit")
    assert np.allclose(v.array, [430, 98])


def test_plausibility_orderings(lexicon):
    """The paper's qualitative results: relative plausibility orderings."""
    assert plausibility(sent(lexicon, "babies", "vomit")) > plausibility(
        sent(lexicon, "footballers", "vomit")
    )
    assert plausibility(sent(lexicon, "footballers", "score")) > plausibility(
        sent(lexicon, "babies", "score")
    )
    # "dribble" is a mixture: both agents do it, in different senses
    assert plausibility(sent(lexicon, "babies", "dribble")) == pytest.approx(
        plausibility(sent(lexicon, "footballers", "dribble")), abs=0.05
    )


def test_transitive_cube_contraction(lexicon):
    """T_i^subj T_ijk^control T_k^obj: footballers ≫ babies (section 4)."""
    W, S = lexicon.word_space, lexicon.sentence_space
    cube = lexicon.lookup("control", (W, S, W))
    fb = lexicon.lookup("footballers", (W,))
    bb = lexicon.lookup("babies", (W,))
    balls = lexicon.lookup("balls", (W,))
    p_fb = plausibility(contract(contract(cube, balls), fb))
    p_bb = plausibility(contract(contract(cube, balls), bb))
    assert p_fb == pytest.approx(1.0)
    assert p_bb == pytest.approx(0.0)


def test_incremental_requirement_sum(lexicon):
    """Incomplete utterances: babies·T+ is less sharp than babies·vomit (§4)."""
    W = lexicon.word_space
    dec = VSSDecorator(lexicon, mode=RequirementMode.SUM)
    t_plus = dec._requirement_value(DSType.parse("e>t"))
    babies = lexicon.lookup("babies", (W,))
    incomplete = plausibility(contract(t_plus, babies))
    complete = plausibility(sent(lexicon, "babies", "vomit"))
    assert incomplete < complete
    # the incomplete vector is comparatively high-entropy (closer to uniform)
    assert incomplete == pytest.approx(0.5, abs=0.35)


def test_incremental_disambiguation(lexicon):
    """Section 5.1: sense vectors disambiguated by the subject alone."""
    assert plausibility(sent(lexicon, "babies", "dribble_drip")) > plausibility(
        sent(lexicon, "babies", "dribble_control")
    )
    assert plausibility(sent(lexicon, "footballers", "dribble_control")) > plausibility(
        sent(lexicon, "footballers", "dribble_drip")
    )


def test_incremental_expectation(lexicon):
    """Section 5.2: plausible continuations ranked by induced plausibility."""
    W = lexicon.word_space
    fb = lexicon.lookup("footballers", (W,))
    bb = lexicon.lookup("babies", (W,))
    verbs = verb_continuations(bb, ["vomit", "score"], lexicon)
    assert list(verbs) == ["vomit", "score"]
    objs = object_continuations(fb, "control", ["balls", "milk"], lexicon)
    assert list(objs) == ["balls", "milk"]
    assert objs["balls"] > objs["milk"]
