"""Underspecified verb/object tensors for incremental composition (paper Sect. 3)."""

from __future__ import annotations

from typing import Sequence

import torch

from dylan.vss.composition import baseline_stages, interpret_sentence
from dylan.vss.types import (
    CompositionMethod,
    IncrementalComposition,
    TensorRep,
    UnderspecMethod,
)


def identity_placeholders(dim: int, *, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Unit verb matrix and ones object vector (jolli ``OI``, ``VI``)."""
    dev = device or torch.device("cpu")
    obj_i = torch.ones(dim, 1, device=dev)
    verb_i = torch.eye(dim, device=dev)
    return verb_i, obj_i


def sum_tensors(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Element-wise sum of tensors with matching shape."""
    if not tensors:
        raise ValueError("sum_tensors requires at least one tensor")
    out = torch.zeros_like(tensors[0].float())
    for t in tensors:
        out = out + t.float()
    return out


def sum_composition_dicts(
    dicts: Sequence[dict[CompositionMethod, TensorRep]],
) -> dict[CompositionMethod, TensorRep]:
    """Sum tensor reps per composition method across a list of interpretations."""
    if not dicts:
        raise ValueError("sum_composition_dicts requires at least one dict")
    keys = list(dicts[0].keys())
    out: dict[CompositionMethod, TensorRep] = {}
    for cm in keys:
        stacked = torch.stack([d[cm].flatten() for d in dicts])
        out[cm] = TensorRep(stacked.sum(dim=0), cm)
    return out


def compose_incremental(
    subject: torch.Tensor,
    verb: torch.Tensor,
    obj: torch.Tensor,
    *,
    candidate_verbs: Sequence[torch.Tensor],
    candidate_objects: Sequence[torch.Tensor],
    method: UnderspecMethod,
) -> IncrementalComposition:
    """Build three incremental stages under identity, sum, or directsum (jolli ``doSentence``)."""
    dim = subject.reshape(-1).numel()
    verb_i, obj_i = identity_placeholders(dim, device=subject.device)

    if method == UnderspecMethod.identity:
        stages = (
            interpret_sentence(subject, verb_i, obj_i),
            interpret_sentence(subject, verb, obj_i),
            interpret_sentence(subject, verb, obj),
        )
        return IncrementalComposition(stages=stages)

    cand_o = list(candidate_objects) or [obj]
    cand_v = list(candidate_verbs) or [verb]

    if method == UnderspecMethod.sum:
        sum_v = sum_tensors(cand_v)
        sum_o = sum_tensors([o.reshape(-1, 1) if o.dim() == 1 else o for o in cand_o])
        if sum_o.dim() == 1:
            sum_o = sum_o.unsqueeze(1)
        stages = (
            interpret_sentence(subject, sum_v, sum_o),
            interpret_sentence(subject, verb, sum_o),
            interpret_sentence(subject, verb, obj),
        )
        return IncrementalComposition(stages=stages)

    # directsum
    stage0 = sum_composition_dicts(
        [interpret_sentence(subject, cv, co) for cv in cand_v for co in cand_o]
    )
    stage1 = sum_composition_dicts([interpret_sentence(subject, verb, co) for co in cand_o])
    stage2 = interpret_sentence(subject, verb, obj)
    return IncrementalComposition(stages=(stage0, stage1, stage2))


def baseline_incremental(
    subject: torch.Tensor,
    verb_vec: torch.Tensor,
    obj: torch.Tensor,
) -> list[TensorRep]:
    """Three-stage additive baseline."""
    return baseline_stages(subject, verb_vec, obj)
