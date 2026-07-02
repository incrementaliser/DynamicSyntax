"""Compile-time specification for a linear lexical parse circuit."""

from __future__ import annotations

from dataclasses import dataclass

from dylan.vss.nesy.parse_lattice import LatticeSpec, ParseStateKey


@dataclass(frozen=True, slots=True)
class WordStepSpec:
    """One word position in a left-to-right parse circuit."""

    word: str
    num_categories: int
    gold_index: int
    legal_mask: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class LinearLatticeCircuitSpec:
    """Per-word categoricals chained by product (Kronecker) for one sentence."""

    sentence: str
    words: tuple[str, ...]
    steps: tuple[WordStepSpec, ...]

    @property
    def num_words(self) -> int:
        """Number of surface words."""
        return len(self.words)

    @property
    def gold_indices(self) -> tuple[int, ...]:
        """Gold categorical index per word."""
        return tuple(s.gold_index for s in self.steps)

    @property
    def lattice_signature(self) -> tuple[tuple[int, ...], ...]:
        """Shape signature for caching compiled circuits."""
        return tuple((s.num_categories,) for s in self.steps)


def linear_spec_from_lattice(spec: LatticeSpec) -> LinearLatticeCircuitSpec:
    """Walk *spec* along the gold path and collect per-step legal supports."""
    if spec.root is None:
        raise ValueError("LatticeSpec.root is required")
    if len(spec.gold_edge_indices) != len(spec.words):
        raise ValueError("gold_edge_indices must align with words")
    steps: list[WordStepSpec] = []
    parent: ParseStateKey = spec.root
    for word, gold_i in zip(spec.words, spec.gold_edge_indices):
        edges = spec.edges_at(parent)
        if not edges:
            raise ValueError(f"No outgoing edges at state {parent!r} for word {word!r}")
        mask = tuple(True for _ in edges)
        steps.append(
            WordStepSpec(
                word=word,
                num_categories=len(edges),
                gold_index=gold_i,
                legal_mask=mask,
            )
        )
        parent = edges[gold_i].child
    return LinearLatticeCircuitSpec(
        sentence=spec.sentence,
        words=spec.words,
        steps=tuple(steps),
    )
