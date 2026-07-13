"""Tests for BabyDS vs CHILDES induction corpus profiles."""

from __future__ import annotations

from pathlib import Path

import pytest

from dylan.dag.type_lattice import TypeLattice
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.corpus_profile import (
    get_active_profile,
    set_active_profile,
)
from dylan.induction.em_learner.record_type_corpus import RecordTypeCorpus
from dylan.induction.em_learner.tree_filter import TreeFilter
from dylan.induction.pipeline.config import load_config
from dylan.tree.node_address import NodeAddress
from dylan.tree.underspecified_type_map import get_static_type_map
from dylan.type.dstype import DSType


@pytest.fixture(autouse=True)
def _restore_babyds_profile() -> None:
    """Reset the default BabyDS profile after each test."""
    yield
    set_active_profile("babyds")


def test_tree_filter_babyds_maps_00_to_obj() -> None:
    """BabyDS profile overwrites address 00 to ``obj`` (AE hack)."""
    set_active_profile("babyds")
    TreeFilter.init()
    field = TreeFilter._node_field_map[NodeAddress("00")]
    assert "obj(" in str(field)
    assert "subj(" not in str(field)


def test_tree_filter_childes_maps_00_to_subj() -> None:
    """CHILDES profile keeps address 00 as ``subj``."""
    set_active_profile("childes")
    TreeFilter.init()
    field = TreeFilter._node_field_map[NodeAddress("00")]
    assert "subj(" in str(field)


def test_childes_profile_type_map_embeds_subj_in_et() -> None:
    """CHILDES ``e>t`` underspec template includes a subject slot."""
    set_active_profile("childes")
    type_map = get_static_type_map()
    et = DSType.parse("e>t")
    assert et is not None
    formula = str(type_map[et])
    assert "subj(" in formula


def test_babyds_profile_type_map_et_has_no_subj() -> None:
    """BabyDS ``e>t`` underspec template stays head-only."""
    set_active_profile("babyds")
    type_map = get_static_type_map()
    et = DSType.parse("e>t")
    assert et is not None
    formula = str(type_map[et])
    assert "subj(" not in formula


def test_childes_lattice_priority_templates_omit_ind_obj() -> None:
    """CHILDES lattice priorities drop the ditransitive ``ind_obj`` tier."""
    set_active_profile("childes")
    TypeLattice()
    joined = " | ".join(str(t) for t in TypeLattice.priority_templates)
    assert "ind_obj" not in joined
    assert "subj" in joined


def test_childes_templates_are_classic_pre_babyds() -> None:
    """CHILDES Filtered templates match pre-BabyDS t/cn order (not cn>cn-only)."""
    set_active_profile("childes")
    profile = get_active_profile()
    assert profile.filtered_abstraction_templates_t == (
        "e>(e>(e>t))",
        "es>(e>(e>t))",
        "e>(e>t)",
        "e>t",
    )
    assert profile.filtered_abstraction_templates_cn == (
        "e>(es>cn)",
        "es>cn",
        "e>cn",
    )


def test_childes_fixture_abstractions_nonempty() -> None:
    """CHILDES-style subject/object RTs yield induction abstractions."""
    set_active_profile("childes")
    rt = TTRRecordType.parse(
        "[r : [x2 : e|head==x2 : e|p7==plane(x2) :t]|x1==epsilon(r.head, r) : e|"
        "x4==london : e|e6==leave : es|head==e6 : es|p4==past(e6) : t|"
        "p6==subj(e6, x1) : t|p5==obj(e6, x4) : t]",
    )
    assert rt is not None
    trees = rt.get_induction_abstractions(NodeAddress(), DSType.t, False)
    assert trees


def test_childes_i_took_it_has_binary_tree_shape() -> None:
    """Headed ``i took it`` under childes Filtered yields ``010`` + swapped ``R2^R1``."""
    set_active_profile("childes")
    rt = TTRRecordType.parse(
        "[x1==it : e|x==i : e|e1==take : es|p2==subj(e1, x) : t|"
        "p3==obj(e1, x1) : t|head==e1 : es]",
    )
    assert rt is not None
    trees = rt.get_induction_abstractions(NodeAddress(), DSType.t, False)
    assert trees
    addrs = {str(n.address) for t in trees for n in t.get_nodes()}
    formulas = [
        str(n.get_formula())
        for t in trees
        for n in t.get_nodes()
        if n.get_formula() is not None
    ]
    assert "010" in addrs
    assert any(f.startswith("R2^R1") for f in formulas)
    # Merge order matches typeMap after post-pass swap.
    assert any("R1 ++ (R2" in f for f in formulas)
    # Event must stay inside the verb formula (5-node e>(e>t), not R3 peel).
    assert all(t.get_num_nodes() == 5 for t in trees)


def test_childes_i_took_it_hypothesise_nonempty() -> None:
    """Headed transitive SVO yields at least one hypothesiser sequence."""
    set_active_profile("childes")
    gdir = Path("resources/2023-english-ttr-induction-seed")
    if not (gdir / "computational-actions.txt").is_file():
        return
    corpus = RecordTypeCorpus()
    rt = TTRRecordType.parse(
        "[x1==it : e|x==i : e|e1==take : es|p2==subj(e1, x) : t|"
        "p3==obj(e1, x1) : t|head==e1 : es]",
    )
    assert rt is not None
    corpus.add_example("i took it", rt)
    from dylan.induction.em_learner.ttr_word_learner import TTRWordLearner

    learner = TTRWordLearner(None, corpus, gdir, top_n=1)
    h = learner.hypothesiser
    h.load_training_example(corpus[0][0], corpus[0][1])
    seqs = h.hypothesise()
    assert len(seqs) >= 1


def test_childes_you_go_abstractions_nonempty() -> None:
    """Intransitive ``you go`` still yields Filtered induction abstractions."""
    set_active_profile("childes")
    rt = TTRRecordType.parse("[x==you : e|e1==go : es|p1==subj(e1, x) : t|head==e1 : es]")
    assert rt is not None
    trees = rt.get_induction_abstractions(NodeAddress(), DSType.t, False)
    assert trees


def test_babyds_ditransitive_fixture_abstractions_nonempty() -> None:
    """BabyDS class-2 ``ind_obj`` fixtures still abstract under babyds profile."""
    set_active_profile("babyds")
    rt = TTRRecordType.parse(
        "[r1 : [x1 : e|head==x1 : e|p1==obj_key(x1) : t]|x2==iota(r1.head, r1) : e|"
        "r2 : [x3 : e|head==x3 : e|p3==obj_ball(x3) : t]|x4==iota(r2.head, r2) : e|"
        "e1==state_beside : es|head==e1 : es|p10==obj(e1, x2) : t|p20==ind_obj(e1, x4) : t]",
    )
    assert rt is not None
    trees = rt.get_maximal_filtered_abstractions(NodeAddress(), DSType.t, False)
    assert trees


def test_goldsent_corpus_loads(tmp_path: Path) -> None:
    """``GoldSent`` / ``Sem`` blocks load like ``Sent`` / ``Sem``."""
    path = tmp_path / "gold.txt"
    path.write_text(
        "GoldSent : i took it\n"
        "Sem : [x1==it : e|x==i : e|e1==take : es|p2==subj(e1, x) : t|p3==obj(e1, x1) : t]\n\n",
        encoding="utf-8",
    )
    corpus = RecordTypeCorpus()
    corpus.load_corpus(path)
    assert len(corpus) == 1
    words, target = corpus[0]
    assert [w.word() for w in words] == ["i", "took", "it"]
    assert target is not None


def test_childes_config_loads_profile_and_seed() -> None:
    """CHILDES yaml selects childes profile and 2023 induction seed."""
    cfg = load_config("configs/induction/childes-holdout.yaml")
    assert cfg.data.profile == "childes"
    assert cfg.data.split == "holdout"
    assert "2023-english-ttr-induction-seed" in cfg.model.seed_grammar
    assert cfg.data.corpus is not None
    assert "CHILDES" in cfg.data.corpus
    set_active_profile(cfg.data.profile)
    assert get_active_profile().name == "childes"


def test_childes_config_holdout_split_sizes() -> None:
    """CHILDES holdout config splits the single corpus file into train and test."""
    from dylan.induction.pipeline.splits import holdout_split

    cfg = load_config("configs/induction/childes-holdout.yaml")
    path = Path(cfg.data.corpus or "")
    if not path.is_file():
        return
    corpus = RecordTypeCorpus(corpus_path=path)
    split = holdout_split(corpus, test_ratio=cfg.data.test_ratio, seed=cfg.data.seed)
    assert len(split.train) + len(split.test) == len(corpus)
    assert len(split.train) > 0
    assert len(split.test) > 0


def test_ensure_induction_head_deems_event() -> None:
    """Headless CHILDES-style RTs get ``head`` pointing at the event field."""
    rt = TTRRecordType.parse(
        "[x1==it : e|x==i : e|e1==take : es|p2==subj(e1, x) : t|p3==obj(e1, x1) : t]",
    )
    assert rt is not None
    assert rt.get_head_field() is None
    label = rt.ensure_induction_head()
    assert label is not None
    assert str(label) == "e1"
    assert rt.get_head_field() is not None
    assert "head==e1" in str(rt).replace(" ", "")


def test_childes_learn_once_with_headless_rt(tmp_path: Path) -> None:
    """Headless CHILDES RT learns at least one word hypothesis (not all-skipped)."""
    set_active_profile("childes")
    corpus = RecordTypeCorpus()
    rt = TTRRecordType.parse("[x==you : e|e1==go : es|p1==subj(e1, x) : t]")
    assert rt is not None
    corpus.add_example("you go", rt)
    gdir = Path("resources/2023-english-ttr-induction-seed")
    if not (gdir / "computational-actions.txt").is_file():
        return
    from dylan.induction.em_learner.ttr_word_learner import TTRWordLearner

    learner = TTRWordLearner(None, corpus, gdir, top_n=1)
    assert learner.learn_once() is True
    assert len(learner.skipped) == 0
    prior = learner.get_hypothesis_base().get_prior()
    assert prior


def test_childes_data_file_loads() -> None:
    """Repository headed CHILDES conversion gold loads Sent/Sem examples."""
    path = Path("data/CHILDES/CHILDESconversion400Final.txt")
    if not path.is_file():
        return
    corpus = RecordTypeCorpus(corpus_path=path)
    assert len(corpus) >= 1
    headed = sum(
        1
        for _words, rt in corpus
        if rt is not None and rt.get_head_field() is not None
    )
    assert headed >= 1
