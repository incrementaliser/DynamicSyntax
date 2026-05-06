"""Parse outcome wrapper for the :mod:`dynamicsyntax` facade."""

from __future__ import annotations

from dataclasses import dataclass

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.gui.formatting import format_ds_tree
from dylan.tree.tree import Tree


@dataclass(frozen=True)
class ParseResult:
    """Result of :func:`dynamicsyntax.parse` with semantics, DS tree, and GUI-style tree view."""

    ok: bool
    semantics: TTRRecordType | None
    tree: Tree | None

    @property
    def address_order(self) -> str:
        """Address-ordered tree text (same panel as the Flet GUI ``address_order`` view)."""
        if self.tree is None:
            return ""
        return format_ds_tree(self.tree)

    def vis(self) -> None:
        """Print the address-order parse tree (GUI ``address_order`` panel); no-op message if no tree."""
        if self.tree is None:
            print("(no parse tree)")
            return
        print(format_ds_tree(self.tree))
