"""Supervised training loop for NeSy DS-VSS parse circuits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import optim

from dylan.vss.nesy.circuit_spec import linear_spec_from_lattice
from dylan.vss.nesy.parse_lattice import SupervisedParseExample
from dylan.vss.nesy.parse_lattice_to_circuit import (
    ParseCircuitModule,
    compile_parse_circuit,
)
from dylan.vss.nesy.semantic_layer import SemanticLossConfig, semantic_loss_dict


@dataclass
class SupervisedTrainConfig:
    """Hyper-parameters for :func:`fit_supervised`."""

    epochs: int = 5
    lr: float = 0.05
    lambda_semantic: float = 0.0
    prefer_cirkit: bool = True


@dataclass
class SupervisedTrainResult:
    """Summary statistics from a supervised training run."""

    epoch_losses: list[float]
    final_parse_nll: float


def _circuit_parameters(circuit: ParseCircuitModule) -> list[torch.nn.Parameter]:
    """Collect trainable parameters from torch or cirkit-backed circuits."""
    if hasattr(circuit, "parameters") and callable(circuit.parameters):  # type: ignore[attr-defined]
        return [p for p in circuit.parameters() if p.requires_grad]  # type: ignore[attr-defined]
    if hasattr(circuit, "torch_circuit"):
        return list(circuit.torch_circuit.parameters())  # type: ignore[attr-defined]
    return []


def fit_supervised(
    examples: Iterable[SupervisedParseExample],
    circuit: ParseCircuitModule | None = None,
    *,
    store: object | None = None,
    session: object | None = None,
    config: SupervisedTrainConfig | None = None,
) -> SupervisedTrainResult:
    """Train on gold lexical paths; reuses *circuit* when lattice signatures match."""
    cfg = config or SupervisedTrainConfig()
    params = _circuit_parameters(circuit) if circuit is not None else []
    optimizer = optim.Adam(params, lr=cfg.lr) if params else None
    epoch_losses: list[float] = []
    sem_cfg = SemanticLossConfig(lambda_semantic=cfg.lambda_semantic)

    for _epoch in range(cfg.epochs):
        total = 0.0
        count = 0
        for ex in examples:
            if ex.lattice is None:
                continue
            spec = linear_spec_from_lattice(ex.lattice)
            step_circuit = circuit
            if step_circuit is None or (
                hasattr(step_circuit, "spec")
                and step_circuit.spec.lattice_signature != spec.lattice_signature  # type: ignore[attr-defined]
            ):
                step_circuit = compile_parse_circuit(
                    spec, prefer_cirkit=cfg.prefer_cirkit
                )
                params = _circuit_parameters(step_circuit)
                optimizer = optim.Adam(params, lr=cfg.lr) if params else None
            gold = torch.tensor(spec.gold_indices, dtype=torch.long)
            log_p = step_circuit.log_likelihood(gold)
            loss = -log_p.mean()
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if store is not None and session is not None and cfg.lambda_semantic > 0:
                terms = semantic_loss_dict(
                    ex,
                    store=store,  # type: ignore[arg-type]
                    session=session,  # type: ignore[arg-type]
                    config=sem_cfg,
                )
                loss = loss + cfg.lambda_semantic * torch.tensor(
                    terms["semantic"], device=loss.device
                )
            total += float(loss.detach())
            count += 1
        epoch_losses.append(total / max(count, 1))

    final_nll = epoch_losses[-1] if epoch_losses else 0.0
    return SupervisedTrainResult(epoch_losses=epoch_losses, final_parse_nll=final_nll)


def build_circuit_for_example(
    example: SupervisedParseExample,
    *,
    prefer_cirkit: bool = True,
) -> ParseCircuitModule:
    """Compile a fresh circuit from one example's lattice."""
    if example.lattice is None:
        raise ValueError("example.lattice is required")
    spec = linear_spec_from_lattice(example.lattice)
    return compile_parse_circuit(spec, prefer_cirkit=prefer_cirkit)
