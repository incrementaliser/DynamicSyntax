"""High-level SVO composition using an :class:`~dylan.vss.embedding_store.EmbeddingStore`."""

from __future__ import annotations

from typing import Sequence

from dylan.vss.underspec import baseline_incremental
from dylan.vss.embedding_store import EmbeddingStore
from dylan.vss.types import GS2013Pair, IncrementalComposition, TensorRep, UnderspecMethod
from dylan.vss.underspec import compose_incremental


def compose_svo(
    store: EmbeddingStore,
    subj: str,
    verb: str,
    obj: str,
    *,
    candidate_subjects: Sequence[str] | None = None,
    candidate_verbs: Sequence[str] | None = None,
    candidate_objects: Sequence[str] | None = None,
    underspec: UnderspecMethod = UnderspecMethod.identity,
) -> IncrementalComposition:
    """Compose one transitive sentence incrementally with the given underspecification method."""
    del candidate_subjects  # reserved for future directsum context filtering
    s = store.get_noun(subj)
    v = store.get_verb_tensor(verb)
    o = store.get_noun(obj)
    cand_v = [store.get_verb_tensor(w) for w in (candidate_verbs or [verb])]
    cand_o = [store.get_noun(w) for w in (candidate_objects or [obj])]
    return compose_incremental(
        s,
        v,
        o,
        candidate_verbs=cand_v,
        candidate_objects=cand_o,
        method=underspec,
    )


def compose_svo_baseline(
    store: EmbeddingStore,
    subj: str,
    verb: str,
    obj: str,
) -> list[TensorRep]:
    """Additive baseline composition for SVO."""
    s = store.get_noun(subj)
    v = store.get_verb_vector(verb)
    o = store.get_noun(obj)
    return baseline_incremental(s, v, o)


def compose_gs2013_triple(
    store: EmbeddingStore,
    sent: tuple[str, str, str],
    *,
    candidate_subjects: set[str],
    candidate_verbs: set[str],
    candidate_objects: set[str],
    underspec: UnderspecMethod,
) -> IncrementalComposition:
    """Compose (subj, verb, obj) with shared candidate sets for underspecification."""
    subj, verb, obj = sent
    return compose_svo(
        store,
        subj,
        verb,
        obj,
        candidate_subjects=candidate_subjects,
        candidate_verbs=candidate_verbs,
        candidate_objects=candidate_objects,
        underspec=underspec,
    )


def compose_gs2013_pair(
    store: EmbeddingStore,
    pair: GS2013Pair,
    *,
    candidate_subjects: set[str],
    candidate_verbs: set[str],
    candidate_objects: set[str],
    underspec: UnderspecMethod,
) -> tuple[IncrementalComposition, IncrementalComposition, IncrementalComposition]:
    """Return sent, para0, para1 incremental compositions for a GS2013 pair (jolli)."""
    f, s = pair.first, pair.second
    sent = compose_gs2013_triple(
        store,
        (f.subj, f.landmark, f.obj),
        candidate_subjects=candidate_subjects,
        candidate_verbs=candidate_verbs,
        candidate_objects=candidate_objects,
        underspec=underspec,
    )
    para0 = compose_gs2013_triple(
        store,
        (f.subj, f.verb, f.obj),
        candidate_subjects=candidate_subjects,
        candidate_verbs=candidate_verbs,
        candidate_objects=candidate_objects,
        underspec=underspec,
    )
    para1 = compose_gs2013_triple(
        store,
        (s.subj, s.verb, s.obj),
        candidate_subjects=candidate_subjects,
        candidate_verbs=candidate_verbs,
        candidate_objects=candidate_objects,
        underspec=underspec,
    )
    return sent, para0, para1
