"""VSS semantic loss terms for supervised NeSy training (APC-style dict)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from dylan.vss.composition import cosine_distance
from dylan.vss.compose_svo import compose_svo
from dylan.vss.ds_vss_session import DSVSSSession
from dylan.vss.embedding_store import EmbeddingStore
from dylan.vss.svo_roles import SVORoles
from dylan.vss.types import TensorRep, UnderspecMethod

if TYPE_CHECKING:
    from dylan.vss.nesy.parse_lattice import SupervisedParseExample


@dataclass(frozen=True, slots=True)
class SemanticLossConfig:
    """Weights and options for semantic supervision."""

    lambda_semantic: float = 1.0
    underspec: UnderspecMethod = UnderspecMethod.identity


def tensor_from_roles(
    store: EmbeddingStore,
    roles: SVORoles,
    *,
    underspec: UnderspecMethod = UnderspecMethod.identity,
) -> TensorRep | None:
    """Compose a VSS tensor from gold SVO roles."""
    try:
        comp = compose_svo(
            store,
            roles.subj,
            roles.landmark,
            roles.obj,
            underspec=underspec,
        )
    except KeyError:
        return None
    if not comp.stages:
        return None
    last = comp.stages[-1]
    if not last:
        return None
    return next(iter(last.values()))


def semantic_loss_dict(
    example: SupervisedParseExample,
    *,
    store: EmbeddingStore,
    session: DSVSSSession,
    config: SemanticLossConfig | None = None,
) -> dict[str, float]:
    """Return APC-style loss terms ``parse_nll`` (caller) and ``semantic``."""
    cfg = config or SemanticLossConfig()
    out: dict[str, float] = {"semantic": 0.0}
    if example.gold_roles is None:
        return out
    gold_rep = tensor_from_roles(store, example.gold_roles, underspec=cfg.underspec)
    if gold_rep is None:
        return out
    parse_result = session.parse_incremental(example.sentence)
    pred_rep: TensorRep | None = None
    if parse_result.final_roles is not None:
        pred_rep = tensor_from_roles(
            store,
            parse_result.final_roles,
            underspec=cfg.underspec,
        )
    if pred_rep is None:
        return out
    out["semantic"] = cosine_distance(gold_rep.tensor, pred_rep.tensor)
    return out


def semantic_loss_tensor(
    gold: torch.Tensor,
    pred: torch.Tensor,
) -> torch.Tensor:
    """Differentiable semantic loss ``1 - cosine_similarity`` on flat vectors."""
    gf = gold.reshape(-1).float()
    pf = pred.reshape(-1).float()
    n = min(gf.numel(), pf.numel())
    gf, pf = gf[:n], pf[:n]
    denom = torch.linalg.norm(gf) * torch.linalg.norm(pf)
    if float(denom) < 1e-8:
        return torch.tensor(1.0, device=gold.device)
    sim = torch.dot(gf, pf) / denom
    return 1.0 - sim
