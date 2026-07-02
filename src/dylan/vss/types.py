"""Configuration and result types for DS-VSS (vector space semantics)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch

NOUN_SUFFIX = "#NN"
VERB_SUFFIX = "#VB"
INCREMENTAL_STAGES = 3


class CompositionMethod(str, Enum):
    """Sentence composition operators from Grefenstette–Sadrzadeh and Kartsaklis et al."""

    gs = "gs"
    ks = "ks"
    ko = "ko"
    baseline = "bs"


class UnderspecMethod(str, Enum):
    """How missing verb/object information is represented during incremental parse."""

    identity = "identity"
    sum = "sum"
    directsum = "directsum"


class EvaluationMode(str, Enum):
    """Whether GS2013 evaluation uses dataset roles only or DS-parsed roles."""

    tensor_only = "tensor_only"
    ds_vss = "ds_vss"


@dataclass(frozen=True, slots=True)
class VSSConfig:
    """Runtime configuration for embedding lookup and composition."""

    dims: int = 300
    underspec: UnderspecMethod = UnderspecMethod.identity
    composition: CompositionMethod | None = None
    vector_shelve_path: Path | None = None
    tensor_shelve_path: Path | None = None
    grammar_path: Path | None = None
    allow_dataset_role_fallback: bool = True
    top_n: int = 3
    use_lexicon_vss_hints: bool = True


@dataclass(frozen=True, slots=True)
class TensorRep:
    """A composed representation that may be a vector or matrix (flattened for similarity)."""

    tensor: torch.Tensor
    composition_method: CompositionMethod

    def flatten(self) -> torch.Tensor:
        """Return a 1-D tensor suitable for cosine similarity."""
        return self.tensor.reshape(-1).float()


@dataclass(frozen=True, slots=True)
class IncrementalComposition:
    """Three-stage incremental compositions for one underspecification method."""

    stages: tuple[dict[CompositionMethod, TensorRep], ...]

    def __len__(self) -> int:
        """Number of incremental stages (subject, subject+verb, full SVO)."""
        return len(self.stages)


@dataclass(frozen=True, slots=True)
class SVORoles:
    """Subject, landmark (ambiguous) verb, and object lemmas for transitive evaluation."""

    subj: str
    landmark: str
    obj: str
    verb: str | None = None
    parse_ok: bool = True
    source: str = "dataset"


@dataclass(frozen=True, slots=True)
class GS2013Sentence:
    """One annotated GS2013 sentence row (KS format)."""

    sentence_id: str
    subj: str
    landmark: str
    verb: str
    obj: str
    adj_subj: str = "dummy"
    adj_obj: str = "dummy"
    similarity_scores: tuple[float, ...] = ()

    def surface_order(self) -> tuple[str, str, str]:
        """Return subject, landmark verb, object as surface triple."""
        return (self.subj, self.landmark, self.obj)


@dataclass(frozen=True, slots=True)
class GS2013Pair:
    """Paired sentences sharing context; gold label picks closer paraphrase."""

    first: GS2013Sentence
    second: GS2013Sentence
    gold_category: int


@dataclass
class MethodAccuracy:
    """Counts and accuracy for one incremental/composition cell."""

    total: float = 0.0
    correct: float = 0.0
    incorrect: float = 0.0

    @property
    def accuracy(self) -> float:
        """Fraction correct (0.5 split on ties)."""
        if self.total <= 0:
            return 0.0
        return self.correct / self.total


@dataclass
class GS2013EvaluationResult:
    """Structured outcome of :func:`~dylan.vss.evaluate.evaluate_gs2013`."""

    mode: EvaluationMode
    by_incremental: dict[str, dict[str, list[MethodAccuracy]]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def accuracy(
        self,
        incremental: UnderspecMethod | str,
        composition: CompositionMethod | str,
        stage: int,
    ) -> float:
        """Lookup accuracy for one incremental method, composition method, and stage index."""
        im = incremental.value if isinstance(incremental, UnderspecMethod) else incremental
        cm = composition.value if isinstance(composition, CompositionMethod) else composition
        return self.by_incremental[im][cm][stage].accuracy
