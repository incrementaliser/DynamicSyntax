"""Tests for the DS-VSS distributional lexicon."""

import numpy as np
import pytest

from dylan.vss.lexicon import VSSLexicon, ppm
from dylan.vss.spaces import VectorSpace

BASIS = ("infant", "nappy", "pitch", "goal")


def make_lexicon() -> VSSLexicon:
    return VSSLexicon(word_space=VectorSpace("W", 4, BASIS))


def test_ppmi_known_values():
    counts = np.array([[2.0, 0.0], [0.0, 2.0]])
    out = ppm(counts)
    assert out.shape == (2, 2)
    assert np.all(out >= 0.0)
    assert out[0, 0] > 0 and out[1, 1] > 0
    assert out[0, 1] == 0 and out[1, 0] == 0
    assert ppm(np.zeros((2, 2))).sum() == 0.0


def test_from_cooccurrence_count_weighting():
    lex = VSSLexicon.from_cooccurrence(
        ["baby", "footballer"],
        BASIS,
        [[34, 10, 0, 0], [0, 0, 24, 17]],
        weighting="count",
    )
    vec = lex.lookup("baby", (lex.word_space,))
    assert vec is not None
    assert np.allclose(vec.array, [34, 10, 0, 0])
    with pytest.raises(ValueError):
        VSSLexicon.from_cooccurrence(["x"], BASIS, np.zeros((2, 4)))


def test_registration_and_lookup():
    lex = make_lexicon()
    lex.add_entity("baby", [34, 10, 0, 0])
    lex.add_intransitive("vomit", [[10, 2], [9, 3], [0, 42], [0, 40]])
    lex.add_transitive("control", np.ones((4, 2, 4)))
    W, S = lex.word_space, lex.sentence_space
    assert lex.lookup("baby", (W,)).array.shape == (4,)
    assert lex.lookup("vomit", (W, S)).array.shape == (4, 2)
    assert lex.lookup("control", (W, S, W)).array.shape == (4, 2, 4)
    assert lex.lookup("vomit", (W, S, W)) is None
    assert lex.lookup("nonexistent", (W,)) is None
    with pytest.raises(ValueError):
        lex.add_entity("bad", [1.0, 2.0])


def test_entries_of_type_includes_phrase_tensors():
    """The T+ requirement material includes verb–object phrase tensors."""
    lex = make_lexicon()
    lex.add_entity("baby", [1, 0, 0, 0])
    lex.add_entity("ball", [0, 0, 1, 1])
    lex.add_intransitive("vomit", np.ones((4, 2)))
    cube = np.zeros((4, 2, 4))
    cube[:, :, 0] = 5.0  # only object index 0 ("baby") carries weight
    lex.add_transitive("control", cube)
    entries = lex.entries_of_type((lex.word_space, lex.sentence_space))
    # 1 matrix + 2 phrase tensors (cube contracted with each entity vector)
    assert len(entries) == 3
    phrase_baby = np.einsum("ijk,k->ij", cube, [1, 0, 0, 0])
    assert any(np.allclose(e.array, phrase_baby) for e in entries)


def test_plausibility_matrix_from_basis_counts():
    lex = make_lexicon()
    mat = lex.plausibility_matrix({"infant": 10, "nappy": 9}, {"pitch": 42})
    assert np.allclose(mat[:, 0], [10, 9, 0, 0])
    assert np.allclose(mat[:, 1], [0, 0, 42, 0])


def test_learn_plausibility_verbs():
    lex = make_lexicon()
    contexts = [
        ({"infant", "nappy"}, "vomit"),
        ({"infant"}, "vomit"),
        ({"pitch", "goal"}, "vomit"),
        ({"pitch", "goal"}, "score"),
    ]
    lex.learn_plausibility_verbs(contexts)
    vomit = lex.lookup("vomit", (lex.word_space, lex.sentence_space))
    # co-occurrence (plausibility) vs absence (implausibility)
    assert np.allclose(vomit.array[:, 0], [2, 1, 1, 1])
    assert np.allclose(vomit.array[:, 1], [1, 2, 2, 2])
    score = lex.lookup("score", (lex.word_space, lex.sentence_space))
    assert np.allclose(score.array[:, 0], [0, 0, 1, 1])
    # transitive registration yields cubes
    lex2 = make_lexicon()
    lex2.learn_plausibility_verbs(contexts, transitive={"vomit"})
    assert lex2.lookup("vomit", (lex2.word_space, lex2.sentence_space, lex2.word_space)) is not None
