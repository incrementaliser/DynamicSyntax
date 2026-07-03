"""Tree filter used by TTR abstraction induction (Java ``qmul.ds.learn.TreeFilter``).

Filters generated abstraction trees against a target :class:`TTRRecordType`
using ``subj``/``obj``/``ind_obj`` argument-position templates.  The Java
class hard-codes node addresses for these positions; the port preserves them.
"""

from __future__ import annotations

from typing import Iterable

from dylan.formula.predicate_argument import PredicateArgumentFormula
from dylan.formula.ttr_field import TTRField
from dylan.formula.ttr_label import TTRLabel
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.formula.variable import Variable
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree


class TreeFilter:
    """Filter abstraction trees against a target record type using template fields."""

    _node_field_map: "dict[NodeAddress, TTRField]" = {}

    def __init__(self, target: TTRRecordType | None = None) -> None:
        """Build the filter; pre-populate :attr:`minimal_sub_types` against *target*."""
        type(self).init()
        self.root_type: TTRRecordType | None = target
        self.minimal_sub_types: dict[NodeAddress, TTRRecordType] = {}
        if target is None:
            return
        try:
            fields = target.get_fields()
        except Exception:  # noqa: BLE001
            fields = []
        for f in fields:
            for n, template in self._node_field_map.items():
                if hasattr(f, "subsumes") and f.subsumes(template):
                    if hasattr(target, "get_minimal_increment_with"):
                        try:
                            subtype = target.get_minimal_increment_with(f, None)
                        except Exception:  # noqa: BLE001
                            subtype = None
                    else:
                        subtype = None
                    if subtype is not None:
                        self.minimal_sub_types[n] = subtype

    @classmethod
    def init(cls) -> None:
        """Populate the static node->field template map (Java ``init``)."""
        if cls._node_field_map:
            return
        try:
            cls._node_field_map[NodeAddress("00")] = TTRField.parse("p==subj(e,x):t")
            cls._node_field_map[NodeAddress("010")] = TTRField.parse("p==obj(e,x):t")
            cls._node_field_map[NodeAddress("0110")] = TTRField.parse("p==ind_obj(e1,e2):t")
            cls._node_field_map[NodeAddress("00")] = TTRField.parse("p==obj(e,x):t")
        except Exception:  # noqa: BLE001
            pass

    def matches(self, tree: Tree) -> bool:
        """Java ``matches``: ``True`` if *tree* satisfies every template position."""
        if not self.minimal_sub_types:
            return True
        for address, template in self._node_field_map.items():
            if address in tree.key_set() if hasattr(tree, "key_set") else address in tree:
                node = tree[address] if address in tree else None
                if node is None:
                    continue
                argument = node.get_formula() if hasattr(node, "get_formula") else None
                if argument is None or not hasattr(argument, "head"):
                    continue
                head_field = argument.head()
                if head_field is None:
                    continue
                head_type = head_field.get_type() if hasattr(head_field, "get_type") else None
                if not isinstance(head_type, Variable):
                    continue
                from dylan.formula.ttr_label import ttr_label_from_variable

                argument_head = ttr_label_from_variable(head_type)
                if address not in self.minimal_sub_types:
                    return False
                second_arg = self._get_second_arg(self.minimal_sub_types[address], template)
                if (
                    not self.minimal_sub_types[address].has_label(argument_head)
                    or argument_head != second_arg
                ):
                    return False
        return True

    def accepts(self, tree: Tree) -> bool:
        """Alias for :meth:`matches`."""
        return self.matches(tree)

    def filter_tree(self, tree: Tree) -> "Tree | None":
        """Return *tree* if it matches, otherwise ``None``."""
        return tree if self.matches(tree) else None

    def filter(self, trees: "Iterable[Tree]") -> list[Tree]:
        """Java ``filter``: return only the trees that satisfy :meth:`matches`."""
        return [tree for tree in trees if self.matches(tree)]

    @staticmethod
    def _get_second_arg(minimal: TTRRecordType, template: TTRField) -> "TTRLabel | None":
        """Java ``getSecondArg``: extract the second predicate argument's label."""
        for f in minimal.get_fields() if hasattr(minimal, "get_fields") else []:
            if hasattr(f, "subsumes") and f.subsumes(template):
                paf = f.get_type() if hasattr(f, "get_type") else None
                if isinstance(paf, PredicateArgumentFormula) and hasattr(paf, "get_arguments"):
                    args = paf.get_arguments()
                    if len(args) >= 2 and isinstance(args[1], Variable):
                        from dylan.formula.ttr_label import ttr_label_from_variable

                        return ttr_label_from_variable(args[1])
        return None


TreeFilter.filterTree = TreeFilter.filter_tree  # type: ignore[attr-defined]
TreeFilter.getSecondArg = staticmethod(TreeFilter._get_second_arg)  # type: ignore[method-assign]
