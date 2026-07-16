"""Tractable probabilistic circuits (Einsum Networks) for neuro-symbolic learning.

Self-contained implementation of the probabilistic-circuit model of Peharz,
Lang, Vergari et al. (2020), "Einsum Networks: Fast and Scalable Learning of
Tractable Probabilistic Circuits" (ICML 2020), in the framework of Choi,
Vergari & Van den Broeck (2020), with bridges to Dynamic Syntax parses and
DS-VSS semantics.
"""

from dylan.pc.bridge import (
    NO_OBJ,
    PAD,
    SemanticTuplePC,
    SVOTuple,
    WordSequencePC,
    extract_svo,
    plausibility_bin,
)
from dylan.pc.einet import (
    UNKNOWN,
    CategoricalLeaf,
    EinsumLayer,
    EinsumNetwork,
    EiNetConfig,
)
from dylan.pc.regions import PCStructure, LayerSpec, random_region_graph

__all__ = [
    "NO_OBJ",
    "PAD",
    "UNKNOWN",
    "CategoricalLeaf",
    "EinsumLayer",
    "EinsumNetwork",
    "EiNetConfig",
    "LayerSpec",
    "PCStructure",
    "SVOTuple",
    "SemanticTuplePC",
    "WordSequencePC",
    "extract_svo",
    "plausibility_bin",
    "random_region_graph",
]
