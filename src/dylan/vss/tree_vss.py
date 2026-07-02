"""Decorate DS trees with vector-space tensors at each node (transitive spine)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from dylan.tree.label.labels import Requirement
from dylan.tree.node import Node
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree
from dylan.type.dstype import DSType
from dylan.vss.composition import interpret_sentence
from dylan.vss.embedding_store import EmbeddingStore
from dylan.vss.types import CompositionMethod, TensorRep, UnderspecMethod
from dylan.vss.lexicon_vss import LexicalVSSBinding, LexicalVSSRole, LexiconVSSIndex
from dylan.vss.underspec import compose_incremental, identity_placeholders


@dataclass
class NodeVSSDecoration:
    """Vector-space annotation for one DS tree node."""

    address: str
    space: str
    tensor: torch.Tensor
    is_requirement: bool = False


@dataclass
class TreeVSSState:
    """VSS decorations and root-level composed representations for one tree snapshot."""

    decorations: dict[str, NodeVSSDecoration] = field(default_factory=dict)
    root_reps: dict[CompositionMethod, TensorRep] = field(default_factory=dict)
    words_seen: tuple[str, ...] = ()


def _space_for_type(ty: DSType | None, *, is_req: bool) -> str:
    """Map a DS type to word (W) or sentence (S) space label."""
    if is_req:
        return "?W"
    if ty is None:
        return "?W"
    if ty == DSType.t:
        return "S"
    return "W"


def _lemma_from_node(node: Node) -> str | None:
    """Try to read a lexical lemma from a node's formula label."""
    fl = node.get_formula_label()
    if fl is None:
        return None
    text = str(fl.get_formula()).lower()
    if "==" in text:
        text = text.split("==", 1)[0].strip()
    return text.strip("[]") or None


def _apply_lexicon_hints(
    *,
    subj: str | None,
    verb: str | None,
    obj: str | None,
    words: tuple[str, ...],
    lexicon_index: LexiconVSSIndex | None,
    step_bindings: list[LexicalVSSBinding],
) -> tuple[str | None, str | None, str | None]:
    """Fill missing SVO hints from lexicon roles and per-step action bindings."""
    bindings = list(step_bindings)
    if lexicon_index is not None:
        bindings.extend(lexicon_index.bindings_for_words(words).values())
    nouns = [b.embedding_key for b in bindings if b.role == LexicalVSSRole.noun]
    verbs = [b.embedding_key for b in bindings if b.role == LexicalVSSRole.verb]
    if subj is None and nouns:
        subj = nouns[0]
    if verb is None and verbs:
        verb = verbs[0]
    if obj is None and len(nouns) > 1:
        obj = nouns[1]
    return subj, verb, obj


def decorate_tree(
    tree: Tree,
    store: EmbeddingStore,
    words: tuple[str, ...],
    *,
    subj: str | None = None,
    verb: str | None = None,
    obj: str | None = None,
    underspec: UnderspecMethod = UnderspecMethod.identity,
    lexicon_index: LexiconVSSIndex | None = None,
    step_bindings: list[LexicalVSSBinding] | None = None,
) -> TreeVSSState:
    """Attach VSS tensors to tree nodes and compute root composition for the transitive spine."""
    subj, verb, obj = _apply_lexicon_hints(
        subj=subj,
        verb=verb,
        obj=obj,
        words=words,
        lexicon_index=lexicon_index,
        step_bindings=step_bindings or [],
    )
    state = TreeVSSState(words_seen=words)
    dim = store.dims
    noun_cache: dict[str, torch.Tensor] = {}
    verb_t: torch.Tensor | None = None
    subj_v: torch.Tensor | None = None
    obj_v: torch.Tensor | None = None

    if subj:
        try:
            subj_v = store.get_noun(subj)
            noun_cache[subj] = subj_v
        except KeyError:
            pass
    if verb:
        try:
            verb_t = store.get_verb_tensor(verb)
        except KeyError:
            pass
    if obj:
        try:
            obj_v = store.get_noun(obj)
            noun_cache[obj] = obj_v
        except KeyError:
            pass

    for addr, node in tree.items():
        ty_label = node.get_type_label()
        is_req = any(isinstance(lab, Requirement) for lab in node.labels)
        ty = ty_label.type if ty_label is not None else None
        space = _space_for_type(ty, is_req=is_req)
        tensor: torch.Tensor
        if is_req:
            _, placeholder = identity_placeholders(dim)
            tensor = placeholder
        elif ty == DSType.e:
            lemma = _lemma_from_node(node)
            if lemma and lemma in noun_cache:
                tensor = noun_cache[lemma]
            elif lemma:
                try:
                    tensor = store.get_noun(lemma)
                    noun_cache[lemma] = tensor
                except KeyError:
                    tensor = torch.zeros(dim)
            else:
                tensor = torch.zeros(dim)
        elif ty is not None and "e>" in str(ty):
            if verb_t is not None:
                tensor = verb_t
            else:
                tensor = identity_placeholders(dim)[0]
        else:
            tensor = torch.zeros(dim)
        state.decorations[str(addr.address)] = NodeVSSDecoration(
            address=str(addr.address),
            space=space,
            tensor=tensor,
            is_requirement=is_req,
        )

    if subj_v is not None and verb_t is not None and obj_v is not None:
        vi, oi = identity_placeholders(dim, device=subj_v.device)
        incr = compose_incremental(
            subj_v,
            verb_t,
            obj_v,
            candidate_verbs=[verb_t],
            candidate_objects=[obj_v],
            method=underspec,
        )
        final = incr.stages[-1]
        if isinstance(final, dict):
            state.root_reps = final
        else:
            state.root_reps = interpret_sentence(subj_v, verb_t, obj_v)
    return state
