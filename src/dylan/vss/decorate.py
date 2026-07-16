"""Fuse Dynamic Syntax trees with vector-space semantics (DS-VSS).

The :class:`VSSDecorator` walks a (possibly partial) DS tree bottom-up and
decorates every node with a tensor value:

- lexical nodes (``Ty(X)`` with a formula) receive the lexicon tensor of
  their predicate constant;
- requirement nodes (``?Ty(X)``) receive, at choice, the *unit* tensor, the
  *sum* ``T+`` of all lexicon tensors of that space, or the *direct sum*
  ``T⊕`` (see :class:`RequirementMode`) — the three options of Sadrzadeh et
  al. (2018), section 3;
- functor/argument mother nodes receive the **tensor contraction** of their
  daughters (DS function application ``O`` ↦ contraction);
- LINKed trees are combined with the matrix tree through the Frobenius
  ``mu`` map (pointwise product).

Because requirements carry real tensor content, *partial* trees compile to a
sentence-space vector at the root — the analogue of Hough & Purver (2012)'s
incremental type inference, and the basis of incremental plausibility and
expectation in the paper's section 4–5.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from dylan.tree.node import Node
from dylan.tree.tree import Tree
from dylan.type.dstype import ConstructedType, DSType
from dylan.vss import predicates
from dylan.vss.lexicon import VSSLexicon
from dylan.vss.spaces import (
    VSSDirectSum,
    VSSValue,
    contract,
    direct_sum,
    mu,
    plausibility,
    unit_value,
)
from dylan.vss.typemap import TensorTypeMap


class RequirementMode(str, Enum):
    """Interpretation of DS requirements ``?Ty(X)`` as tensors."""

    #: The unit tensor (neutral, no semantic information).
    UNIT = "unit"
    #: Sum of all lexicon tensors of the space ("average" expectation, ``T+``).
    SUM = "sum"
    #: Direct sum: keep the alternatives separate (``T⊕``).
    DIRECT_SUM = "direct_sum"


@dataclass
class VSSDecoration:
    """Result of decorating one DS tree with tensor values."""

    values: dict[str, VSSValue | VSSDirectSum]
    root_value: VSSValue | VSSDirectSum | None
    missing: tuple[str, ...] = ()

    def value_at(self, address: str) -> VSSValue | VSSDirectSum | None:
        """Value decorating the node at *address* (e.g. ``"0"``, ``"00"``)."""
        return self.values.get(address)

    def sentence_value(self) -> VSSValue | VSSDirectSum | None:
        """Value at the tree root: the (possibly partial) utterance meaning."""
        return self.root_value

    def plausibility(self) -> float | list[float] | None:
        """Normalised plausibility of the root sentence vector."""
        v = self.root_value
        if v is None:
            return None
        if isinstance(v, VSSDirectSum):
            return v.plausibilities()
        try:
            return plausibility(v)
        except ValueError:
            return None


class VSSDecorator:
    """Decorate DyLan DS trees with distributional tensor semantics.

    :param lexicon: the :class:`~dylan.vss.lexicon.VSSLexicon` supplying
        lexical tensors.
    :param mode: how to instantiate requirement nodes (default
        :attr:`RequirementMode.SUM`, the working example of the paper).
    :param type_map: optional custom :class:`TensorTypeMap`.
    """

    def __init__(
        self,
        lexicon: VSSLexicon,
        mode: RequirementMode | str = RequirementMode.SUM,
        type_map: TensorTypeMap | None = None,
    ) -> None:
        self.lexicon = lexicon
        self.mode = RequirementMode(mode)
        self.type_map = type_map or TensorTypeMap(
            lexicon.word_space, lexicon.sentence_space
        )
        self.missing: list[str] = []

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def decorate(self, tree: Tree) -> VSSDecoration:
        """Decorate *tree*; return values per node address plus the root value."""
        self.missing = []
        values: dict[str, VSSValue | VSSDirectSum] = {}
        root = tree.get_root_node()
        root_value = self._decorate_node(tree, root, values) if root is not None else None
        return VSSDecoration(values, root_value, tuple(self.missing))

    # ------------------------------------------------------------------
    # node semantics
    # ------------------------------------------------------------------
    def _node_type(self, node: Node) -> DSType | None:
        """The node's DS type, looking inside a type requirement if needed."""
        ty = node.get_type()
        if ty is not None:
            return ty
        return node.get_required_type()

    def _requirement_value(
        self, ds_type: DSType
    ) -> VSSValue | VSSDirectSum:
        """Tensor interpretation of a requirement ``?Ty(ds_type)``."""
        spaces = self.type_map(ds_type)
        if self.mode is RequirementMode.UNIT:
            return unit_value(spaces)
        entries = self.lexicon.entries_of_type(spaces)
        if self.mode is RequirementMode.DIRECT_SUM:
            return direct_sum(entries) if entries else direct_sum([unit_value(spaces)])
        # SUM
        if not entries:
            return unit_value(spaces)
        total = np.zeros(tuple(s.dim for s in spaces), dtype=float)
        for e in entries:
            total = total + e.array
        return VSSValue(spaces, total)

    def _lexical_value(self, node: Node, ds_type: DSType) -> VSSValue:
        """Tensor of a lexical node, from its formula's predicate constant."""
        spaces = self.type_map(ds_type)
        formula = node.get_formula()
        # Entity-typed leaves look up entity constants (``john``); verb and
        # sentence-typed leaves look up eventuality constants (``like``).
        if not isinstance(ds_type, ConstructedType) and spaces == (self.lexicon.word_space,):
            primary = predicates.extract_entity(formula)
        else:
            primary = predicates.extract_event(formula)
        candidates: list[str | None] = [
            primary,
            predicates.extract_constant(formula, "e"),
            predicates.extract_constant(formula, "es"),
        ]
        for pred in candidates:
            if pred is None:
                continue
            value = self.lexicon.lookup(pred, spaces)
            if value is not None:
                return value
            if pred not in self.missing:
                self.missing.append(pred)
            break
        return unit_value(spaces)

    def _daughter_groups(
        self, tree: Tree, node: Node
    ) -> tuple[list[Node], list[Node]]:
        """Split daughters into (composition daughters, LINK daughters)."""
        comp: list[Node] = []
        links: list[Node] = []
        for dtr in tree.get_daughters(node):
            addr = str(dtr.address)
            if addr[-1] == "L":
                links.append(dtr)
            else:
                comp.append(dtr)
        return comp, links

    def _compose(
        self,
        tree: Tree,
        node: Node,
        values: dict[str, VSSValue | VSSDirectSum],
    ) -> VSSValue | VSSDirectSum | None:
        """Value of a mother node from its daughters (contraction + ``mu``)."""
        comp, links = self._daughter_groups(tree, node)
        daughter_values = [
            (dtr, self._decorate_node(tree, dtr, values)) for dtr in comp
        ]
        value: VSSValue | VSSDirectSum | None = None
        if len(daughter_values) == 1:
            value = daughter_values[0][1]
        elif len(daughter_values) >= 2:
            (d0, v0), (d1, v1) = daughter_values[0], daughter_values[1]
            t0, t1 = self._node_type(d0), self._node_type(d1)
            if isinstance(t1, ConstructedType) or not isinstance(t0, ConstructedType):
                functor, argument = v1, v0
            else:
                functor, argument = v0, v1
            value = self._apply(functor, argument)
        if value is None:
            return None
        for link in links:
            link_value = self._decorate_node(tree, link, values)
            value = self._combine_link(value, link_value)
        return value

    def _apply(
        self,
        functor: VSSValue | VSSDirectSum,
        argument: VSSValue | VSSDirectSum,
    ) -> VSSValue | VSSDirectSum:
        """DS function application as tensor contraction (distributing sums)."""
        if isinstance(functor, VSSDirectSum):
            return VSSDirectSum(tuple(self._apply(f, argument) for f in functor.values))
        if isinstance(argument, VSSDirectSum):
            return VSSDirectSum(tuple(self._apply(functor, a) for a in argument.values))
        return contract(functor, argument)

    def _combine_link(
        self,
        value: VSSValue | VSSDirectSum,
        link_value: VSSValue | VSSDirectSum,
    ) -> VSSValue | VSSDirectSum:
        """Combine a matrix-tree value with its LINKed tree via the ``mu`` map."""
        if isinstance(link_value, VSSDirectSum) or isinstance(value, VSSDirectSum):
            # Keep direct sums intact; combine elementwise where possible.
            base = value.values if isinstance(value, VSSDirectSum) else (value,)
            others = (
                link_value.values
                if isinstance(link_value, VSSDirectSum)
                else (link_value,)
            )
            combined = []
            for v in base:
                for o in others:
                    if (
                        isinstance(v, VSSValue)
                        and isinstance(o, VSSValue)
                        and v.space_names() == o.space_names()
                    ):
                        combined.append(mu(v, o))
                    else:
                        combined.append(v)
            return VSSDirectSum(tuple(combined))
        if value.space_names() == link_value.space_names():
            return mu(value, link_value)
        return value

    def _decorate_node(
        self, tree: Tree, node: Node, values: dict[str, VSSValue | VSSDirectSum]
    ) -> VSSValue | VSSDirectSum:
        """Bottom-up decoration of *node*; returns and records its value."""
        addr = str(node.address)
        comp, links = self._daughter_groups(tree, node)
        ds_type = self._node_type(node)

        if comp:
            value = self._compose(tree, node, values)
            if value is None and ds_type is not None:
                value = self._requirement_value(ds_type)
        elif node.has_type() and node.get_formula() is not None and ds_type is not None:
            # Lexical head whose only daughters are LINKed trees.
            value = self._lexical_value(node, ds_type)
            for link in links:
                link_value = self._decorate_node(tree, link, values)
                value = self._combine_link(value, link_value)
        elif node.get_required_type() is not None and not node.has_type():
            value = self._requirement_value(node.get_required_type())
        elif ds_type is not None and node.get_formula() is not None:
            value = self._lexical_value(node, ds_type)
        elif ds_type is not None:
            value = unit_value(self.type_map(ds_type))
        else:
            # Label-less node (should not normally occur): neutral scalar.
            value = VSSValue((), np.array(1.0))
        values[addr] = value
        return value


def decorate_tree(
    tree: Tree,
    lexicon: VSSLexicon,
    mode: RequirementMode | str = RequirementMode.SUM,
) -> VSSDecoration:
    """Convenience one-shot decoration of *tree* with *lexicon*."""
    return VSSDecorator(lexicon, mode=mode).decorate(tree)


__all__ = [
    "RequirementMode",
    "VSSDecoration",
    "VSSDecorator",
    "decorate_tree",
]
