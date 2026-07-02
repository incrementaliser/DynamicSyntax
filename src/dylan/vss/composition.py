"""PyTorch compositional operators for transitive SVO (ported from jolli.py)."""

from __future__ import annotations

from typing import Mapping

import torch

from dylan.vss.types import CompositionMethod, TensorRep


def _column(v: torch.Tensor) -> torch.Tensor:
    """Ensure a 2-D column vector (n, 1)."""
    if v.dim() == 1:
        return v.unsqueeze(1)
    if v.dim() == 2 and v.shape[1] == 1:
        return v
    return v.reshape(-1, 1)


def _row(v: torch.Tensor) -> torch.Tensor:
    """Ensure a 2-D row vector (1, n)."""
    col = _column(v)
    return col.T


def interpret_sentence(
    subject: torch.Tensor,
    verb: torch.Tensor,
    obj: torch.Tensor,
) -> dict[CompositionMethod, TensorRep]:
    """Compose subject, verb tensor, and object using gs, ks, and ko (jolli ``interpretSentence``)."""
    s = _column(subject).float()
    o = _column(obj).float()
    v = verb.float()
    if v.dim() == 1:
        n = s.shape[0]
        v = v.reshape(n, n)
    gs_mat = (s @ o.T) * v
    s1 = s.reshape(-1)
    o1 = o.reshape(-1)
    ks_vec = (s1 * (v @ o1)).reshape(-1)
    ko_vec = (o1 * (v.T @ s1)).reshape(-1)
    return {
        CompositionMethod.gs: TensorRep(gs_mat, CompositionMethod.gs),
        CompositionMethod.ks: TensorRep(ks_vec, CompositionMethod.ks),
        CompositionMethod.ko: TensorRep(ko_vec, CompositionMethod.ko),
    }


def baseline_stages(
    subject: torch.Tensor,
    verb: torch.Tensor,
    obj: torch.Tensor,
) -> list[TensorRep]:
    """Additive baseline: S, S+V, S+V+O as vectors (jolli ``doBaseline``)."""
    s = subject.reshape(-1).float()
    v = verb.reshape(-1).float()
    o = obj.reshape(-1).float()
    n = min(s.numel(), v.numel(), o.numel())
    s, v, o = s[:n], v[:n], o[:n]
    return [
        TensorRep(s.clone(), CompositionMethod.baseline),
        TensorRep(s + v, CompositionMethod.baseline),
        TensorRep(s + v + o, CompositionMethod.baseline),
    ]


def cosine_distance(a: torch.Tensor, b: torch.Tensor, *, eps: float = 1e-8) -> float:
    """Cosine distance ``1 - cos_sim`` matching scipy.spatial.distance.cosine on flat vectors."""
    af = a.reshape(-1).float()
    bf = b.reshape(-1).float()
    n = min(af.numel(), bf.numel())
    af, bf = af[:n], bf[:n]
    denom = torch.linalg.norm(af) * torch.linalg.norm(bf)
    if float(denom) < eps:
        return 1.0
    sim = torch.dot(af, bf) / denom
    return float(1.0 - sim.item())


def pick_category(distance_a: float, distance_b: float) -> int:
    """Return winning category 0/1, or -1 on tie (half credit in evaluation)."""
    if distance_a < distance_b:
        return 0
    if distance_b < distance_a:
        return 1
    return -1


def flatten_rep(rep: TensorRep | Mapping[CompositionMethod, TensorRep], method: CompositionMethod) -> torch.Tensor:
    """Flatten one composition method's representation."""
    if isinstance(rep, TensorRep):
        return rep.flatten()
    return rep[method].flatten()
