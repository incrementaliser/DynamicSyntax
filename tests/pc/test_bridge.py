"""Tests for the neuro-symbolic bridge: PCs over DS / DS-VSS outputs."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import dynamicsyntax as ds  # noqa: E402
from dylan.pc.bridge import (  # noqa: E402
    SemanticTuplePC,
    WordSequencePC,
    extract_svo,
    plausibility_bin,
)
from dynamicsyntax.pc import DSPlausibilityPC, PCWordModel  # noqa: E402
from dynamicsyntax.vss import VSSLexicon, VectorSpace  # noqa: E402

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


def test_extract_svo_transitive():
    tree = ds.parse("john likes mary", "ttr").tree
    svo = extract_svo(tree)
    assert (svo.subject, svo.verb, svo.obj) == ("john", "like", "mary")


def test_extract_svo_intransitive():
    tree = ds.parse("john arrives", "ttr").tree
    svo = extract_svo(tree)
    assert (svo.subject, svo.verb, svo.obj) == ("john", "arrive", None)


def test_plausibility_bin():
    assert plausibility_bin(0.0, 5) == 0
    assert plausibility_bin(0.999, 5) == 4
    assert plausibility_bin(0.5, 5) == 2


def test_word_sequence_pc_next_word():
    torch.manual_seed(0)
    model = WordSequencePC(max_len=3, seed=0)
    corpus = ["john likes mary", "john likes john", "mary likes mary"] * 8
    model.fit([s.split() for s in corpus], epochs=15, step_size=1.0, batch_size=24)
    probs = model.next_word_probs(["john", "likes"])
    # only mary/john continue the prefix in the corpus (with prob. ~2:1)
    top2 = sorted(probs, key=probs.get, reverse=True)[:2]
    assert top2 == ["mary", "john"]
    assert probs["mary"] > probs["john"]
    # probabilities over candidates sum to 1
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-4)


def test_word_model_facade():
    torch.manual_seed(0)
    wm = PCWordModel(max_len=3, seed=1)
    wm.fit(["john likes mary", "mary likes john"] * 5, epochs=5)
    ll_in = wm.log_likelihood("john likes mary")
    assert isinstance(ll_in, float)
    probs = wm.next_word_probs("mary")
    assert "likes" in probs


def test_semantic_tuple_pc_conditionals():
    torch.manual_seed(0)
    model = SemanticTuplePC(num_bins=4, seed=0)
    rows = [("baby", "vomit", None, 3)] * 6 + [("footballer", "vomit", None, 0)] * 6
    model.fit(rows, epochs=20, step_size=1.0, batch_size=12)
    dist_baby = model.plausibility_distribution("baby", "vomit")
    dist_fb = model.plausibility_distribution("footballer", "vomit")
    assert sum(dist_baby) == pytest.approx(1.0, abs=1e-4)
    assert np.argmax(dist_baby) == 3
    assert np.argmax(dist_fb) == 0
    assert model.expected_plausibility("baby", "vomit") > model.expected_plausibility(
        "footballer", "vomit"
    )


def test_semantic_tuple_pc_object_and_verb_probs():
    torch.manual_seed(0)
    model = SemanticTuplePC(num_bins=4, seed=0)
    rows = (
        [("john", "like", "mary", 3)] * 5
        + [("john", "like", "john", 3)]
        + [("john", "arrive", None, 2)] * 2
    )
    model.fit(rows, epochs=8)
    obj = model.object_probs("john", "like")
    assert list(obj)[0] == "mary"
    verbs = model.verb_probs("john")
    assert set(verbs) == {"like", "arrive"}


def test_ds_plausibility_pc_end_to_end(lexicon):
    torch.manual_seed(0)
    corpus = ["john likes mary", "mary likes john", "john arrives", "mary arrives"]
    pc = DSPlausibilityPC(num_bins=4, seed=0)
    rows = pc.build_tuples(corpus, "ttr", lexicon)
    assert ("john", "like", "mary", 3) in rows
    assert any(s == "john" and v == "arrive" and o is None for s, v, o, _ in rows)
    history = pc.fit(corpus, "ttr", lexicon, epochs=6)
    assert history[-1] < history[0]
    dist = pc.plausibility_distribution("john", "like", "mary")
    assert sum(dist) == pytest.approx(1.0, abs=1e-4)
    ranked = pc.rank_objects("john", "like")
    assert set(ranked) == {"john", "mary"}
