"""DS-VSS: Dynamic Syntax with Vector Space Semantics.

Compositional distributional semantics for the DyLan Dynamic Syntax parser,
after Sadrzadeh, Purver, Hough & Kempson (2018), "Exploring Semantic
Incrementality with Dynamic Syntax and Vector Space Semantics"
(arXiv:1811.00614): DS types map to tensor products of vector spaces, DS
function application to tensor contraction, LINK adjunction to the Frobenius
``mu`` map, and DS requirements to unit/sum/direct-sum tensors.
"""

from dylan.vss.decorate import (
    RequirementMode,
    VSSDecoration,
    VSSDecorator,
    decorate_tree,
)
from dylan.vss.incremental import (
    decorate_traces,
    object_continuations,
    plausibility_trajectory,
    verb_continuations,
)
from dylan.vss.lexicon import VSSLexicon
from dylan.vss.spaces import (
    FALSE,
    TRUE,
    VSSDirectSum,
    VSSValue,
    VectorSpace,
    contract,
    mu,
    plausibility,
    plausibility_space,
)
from dylan.vss.typemap import TensorTypeMap

__all__ = [
    "FALSE",
    "TRUE",
    "RequirementMode",
    "TensorTypeMap",
    "VSSDecoration",
    "VSSDecorator",
    "VSSDirectSum",
    "VSSLexicon",
    "VSSValue",
    "VectorSpace",
    "contract",
    "decorate_traces",
    "decorate_tree",
    "mu",
    "object_continuations",
    "plausibility",
    "plausibility_space",
    "plausibility_trajectory",
    "verb_continuations",
]
