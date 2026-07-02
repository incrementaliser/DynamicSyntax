"""Programmatic GS2013 disambiguation evaluation (ported from jolli ``testSentences``)."""

from __future__ import annotations

from collections.abc import Callable

from dylan.vss.compose_svo import compose_gs2013_pair, compose_gs2013_triple, compose_svo_baseline
from dylan.vss.composition import cosine_distance, pick_category
from dylan.vss.embedding_store import EmbeddingStore
from dylan.vss.gs2013_data import candidate_sets_for_landmark, load_sentence_pairs
from dylan.vss.types import (
    CompositionMethod,
    EvaluationMode,
    GS2013EvaluationResult,
    GS2013Pair,
    MethodAccuracy,
    TensorRep,
    UnderspecMethod,
    VSSConfig,
)

_COMPOSITION_METHODS = (
    CompositionMethod.gs,
    CompositionMethod.ks,
    CompositionMethod.ko,
    CompositionMethod.baseline,
)
_INCR_METHODS = (
    UnderspecMethod.identity,
    UnderspecMethod.sum,
    UnderspecMethod.directsum,
)


def _init_accuracy_grid() -> dict[str, dict[str, list[MethodAccuracy]]]:
    """Empty nested accuracy counters for all incremental/composition/stage cells."""
    grid: dict[str, dict[str, list[MethodAccuracy]]] = {}
    for im in _INCR_METHODS:
        grid[im.value] = {}
        for cm in _COMPOSITION_METHODS:
            grid[im.value][cm.value] = [MethodAccuracy() for _ in range(3)]
    return grid


def _record_prediction(
    acc: MethodAccuracy,
    gold: int,
    lead: int,
) -> None:
    """Update counts for one pairwise comparison."""
    acc.total += 1.0
    if lead < 0:
        acc.correct += 0.5
        acc.incorrect += 0.5
    elif gold == lead:
        acc.correct += 1.0
    else:
        acc.incorrect += 1.0


def _rep_at_stage(
    incremental: object,
    cm: CompositionMethod,
    stage: int,
) -> TensorRep:
    """Fetch a :class:`~dylan.vss.types.TensorRep` for baseline list or incremental dict stage."""
    if cm == CompositionMethod.baseline:
        stages = incremental  # type: ignore[assignment]
        return stages[stage]
    stages = incremental.stages  # type: ignore[union-attr]
    stage_val = stages[stage]
    if isinstance(stage_val, TensorRep):
        return stage_val
    return stage_val[cm]


def _triple_from_pair(
    pair: GS2013Pair,
    which: str,
) -> tuple[str, str, str]:
    """Return (subj, verb, obj) for sent, para0, or para1."""
    if which == "sent":
        return (pair.first.subj, pair.first.landmark, pair.first.obj)
    if which == "para0":
        return (pair.first.subj, pair.first.verb, pair.first.obj)
    return (pair.second.subj, pair.second.verb, pair.second.obj)


def evaluate_pair(
    store: EmbeddingStore,
    pair: GS2013Pair,
    *,
    candidate_subjects: set[str],
    candidate_verbs: set[str],
    candidate_objects: set[str],
    grid: dict[str, dict[str, list[MethodAccuracy]]],
    triples: tuple[tuple[str, str, str], tuple[str, str, str], tuple[str, str, str]] | None = None,
) -> None:
    """Score one GS2013 pair into *grid* (jolli inner loop)."""
    t_sent, t_p0, t_p1 = triples or (
        _triple_from_pair(pair, "sent"),
        _triple_from_pair(pair, "para0"),
        _triple_from_pair(pair, "para1"),
    )
    bs_sent = compose_svo_baseline(store, *t_sent)
    bs_para0 = compose_svo_baseline(store, *t_p0)
    bs_para1 = compose_svo_baseline(store, *t_p1)
    gold = pair.gold_category

    for im in _INCR_METHODS:
        if triples is None:
            sent, para0, para1 = compose_gs2013_pair(
                store,
                pair,
                candidate_subjects=candidate_subjects,
                candidate_verbs=candidate_verbs,
                candidate_objects=candidate_objects,
                underspec=im,
            )
        else:
            sent = compose_gs2013_triple(
                store,
                t_sent,
                candidate_subjects=candidate_subjects,
                candidate_verbs=candidate_verbs,
                candidate_objects=candidate_objects,
                underspec=im,
            )
            para0 = compose_gs2013_triple(
                store,
                t_p0,
                candidate_subjects=candidate_subjects,
                candidate_verbs=candidate_verbs,
                candidate_objects=candidate_objects,
                underspec=im,
            )
            para1 = compose_gs2013_triple(
                store,
                t_p1,
                candidate_subjects=candidate_subjects,
                candidate_verbs=candidate_verbs,
                candidate_objects=candidate_objects,
                underspec=im,
            )
        for stage in range(3):
            i2 = stage
            for cm in _COMPOSITION_METHODS:
                acc = grid[im.value][cm.value][stage]
                if cm == CompositionMethod.baseline:
                    d0 = cosine_distance(
                        bs_sent[stage].flatten(),
                        bs_para0[i2].flatten(),
                    )
                    d1 = cosine_distance(
                        bs_sent[stage].flatten(),
                        bs_para1[i2].flatten(),
                    )
                else:
                    r_sent = _rep_at_stage(sent, cm, stage)
                    r_p0 = _rep_at_stage(para0, cm, i2)
                    r_p1 = _rep_at_stage(para1, cm, i2)
                    d0 = cosine_distance(r_sent.flatten(), r_p0.flatten())
                    d1 = cosine_distance(r_sent.flatten(), r_p1.flatten())
                _record_prediction(acc, gold, pick_category(d0, d1))


def evaluate_gs2013(
    store: EmbeddingStore,
    config: VSSConfig | None = None,
    *,
    mode: EvaluationMode = EvaluationMode.tensor_only,
    pairs: list[GS2013Pair] | None = None,
    session: object | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
    progress_every: int = 100,
) -> GS2013EvaluationResult:
    """Run GS2013 verb disambiguation evaluation and return structured accuracies."""
    cfg = config or VSSConfig()
    all_pairs = pairs or load_sentence_pairs()
    grid = _init_accuracy_grid()
    skipped = 0
    parse_failures = 0
    pair_total = len(all_pairs)

    ds_session = None
    if mode == EvaluationMode.ds_vss:
        from dylan.vss.ds_vss_session import DSVSSSession

        ds_session = session or DSVSSSession(
            grammar=cfg.grammar_path,
            embedding_store=store,
            config=cfg,
        )

    for pair_index, pair in enumerate(all_pairs, start=1):
        try:
            ss, vs, os = candidate_sets_for_landmark(all_pairs, pair.first.landmark)
            triples = None
            if mode == EvaluationMode.ds_vss:
                from dylan.vss.ds_vss_session import DSVSSSession

                assert isinstance(ds_session, DSVSSSession)

                def _triple(sent: object, *, landmark: bool) -> tuple[str, str, str]:
                    nonlocal parse_failures
                    from dylan.vss.types import GS2013Sentence

                    assert isinstance(sent, GS2013Sentence)
                    pr = ds_session.parse_gs2013_sentence(sent, use_landmark=landmark)
                    if not pr.ok or pr.final_roles is None:
                        parse_failures += 1
                        if landmark:
                            return (sent.subj, sent.landmark, sent.obj)
                        return (sent.subj, sent.verb, sent.obj)
                    r = pr.final_roles
                    verb_lemma = r.landmark if landmark else (r.verb or sent.verb)
                    return (r.subj, verb_lemma, r.obj)

                triples = (
                    _triple(pair.first, landmark=True),
                    _triple(pair.first, landmark=False),
                    _triple(pair.second, landmark=False),
                )
            evaluate_pair(
                store,
                pair,
                candidate_subjects=ss,
                candidate_verbs=vs,
                candidate_objects=os,
                grid=grid,
                triples=triples,
            )
        except KeyError:
            skipped += 1
            continue
        if progress_callback is not None:
            progress_callback(pair_index, pair_total, skipped)
    return GS2013EvaluationResult(
        mode=mode,
        by_incremental=grid,
        metadata={
            "pairs_total": len(all_pairs),
            "pairs_skipped": skipped,
            "parse_failures": parse_failures,
        },
    )
