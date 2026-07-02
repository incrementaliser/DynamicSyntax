"""Compile and query libcirkit parse circuits with optional disk cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from dylan.vss.nesy.circuit_spec import LinearLatticeCircuitSpec
from dylan.vss.nesy.parse_lattice_to_circuit import build_symbolic_circuit


def _default_cache_dir() -> Path:
    """Return the default circuit cache directory under the user home."""
    return Path.home() / ".cache" / "dylan-nesy"


def _spec_cache_key(spec: LinearLatticeCircuitSpec) -> str:
    """Hash a lattice signature for cache filenames."""
    payload = json.dumps(spec.lattice_signature)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class CompiledParseCircuit:
    """Thin wrapper around a compiled cirkit torch circuit."""

    torch_circuit: Any
    spec: LinearLatticeCircuitSpec

    def log_likelihood(self, gold_indices: torch.Tensor) -> torch.Tensor:
        """Evaluate log-likelihood for integer assignments ``(B, num_words)``."""
        if gold_indices.dim() == 1:
            gold_indices = gold_indices.unsqueeze(0)
        out = self.torch_circuit(gold_indices)
        while out.dim() > 1:
            out = out.squeeze(-1)
        return out.reshape(gold_indices.shape[0])


class CirkitParseSession:
    """Compile :class:`LinearLatticeCircuitSpec` with cirkit and optional cache."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        """Configure an optional on-disk compile cache."""
        self._cache_dir = cache_dir or _default_cache_dir()

    @classmethod
    def compile(cls, spec: LinearLatticeCircuitSpec) -> CompiledParseCircuit:
        """Compile *spec* to a torch backend circuit (lse-sum semiring)."""
        from cirkit.pipeline import PipelineContext

        symbolic = build_symbolic_circuit(spec)
        ctx = PipelineContext(backend="torch", semiring="lse-sum")
        torch_circuit = ctx.compile(symbolic)
        return CompiledParseCircuit(torch_circuit=torch_circuit, spec=spec)

    def compile_or_load(self, spec: LinearLatticeCircuitSpec) -> CompiledParseCircuit:
        """Compile *spec*, using a cache directory marker when present (v1: always compile)."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        _ = _spec_cache_key(spec)
        return self.compile(spec)
