"""Region graphs for Einsum Networks.

A *region* is a set of variables; a *region graph* organises regions into
layers whose alternating sum/product structure defines a smooth and
decomposable probabilistic circuit (Peharz et al. 2020, sec. 4; Choi,
Vergari & Van den Broeck 2020, "Probabilistic Circuits").

:func:`random_region_graph` builds a random binary hierarchy: starting from
the singleton regions, each layer pairs up the regions of the previous layer
at random and introduces their unions as parent regions (a left-over region
is carried upward by a single-child sum).  Regions at every layer partition
the variable scope, which guarantees **decomposability** (product children
have disjoint scopes) and **smoothness** (sum children share their scope) —
hence exact, linear-time marginal and conditional inference in the circuit.

The construction mirrors the random region graphs of RAT-SPNs / EiNets
(Peharz et al. 2020a, 2020b); ``num_repetitions`` independent channel blocks
are created at circuit construction time and mixed at the root sum.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LayerSpec:
    """One product-sum layer, by child indices into the previous layer.

    ``left[p]`` / ``right[p]`` are the indices (in the previous layer's
    region list) of the two children of parent region ``p``; ``right[p]``
    is ``None`` for a carried-over single child.
    """

    left: list[int]
    right: list[int | None]

    @property
    def num_parents(self) -> int:
        return len(self.left)


@dataclass
class PCStructure:
    """Bottom-up layer structure of a probabilistic circuit."""

    num_vars: int
    num_repetitions: int
    layers: list[LayerSpec] = field(default_factory=list)

    def region_counts(self) -> list[int]:
        """Number of regions per layer, starting at the input (leaf) layer."""
        counts = [self.num_vars]
        counts.extend(spec.num_parents for spec in self.layers)
        return counts


def random_region_graph(
    num_vars: int,
    num_repetitions: int = 1,
    seed: int | None = None,
) -> PCStructure:
    """Random binary region hierarchy over ``{0, …, num_vars - 1}``.

    :param num_vars: number of random variables (leaf regions).
    :param num_repetitions: channel-block multiplier (mixed at the root).
    :param seed: RNG seed for reproducible structures.
    """
    if num_vars < 1:
        raise ValueError("region graph needs at least one variable")
    if num_repetitions < 1:
        raise ValueError("num_repetitions must be >= 1")
    rng = np.random.default_rng(seed)
    structure = PCStructure(num_vars, num_repetitions)
    count = num_vars
    while count > 1:
        order = rng.permutation(count).tolist()
        left: list[int] = []
        right: list[int | None] = []
        i = 0
        while i < count:
            a = order[i]
            if i + 1 < count:
                left.append(a)
                right.append(order[i + 1])
                i += 2
            else:
                left.append(a)
                right.append(None)
                i += 1
        structure.layers.append(LayerSpec(left, right))
        count = len(left)
    return structure
