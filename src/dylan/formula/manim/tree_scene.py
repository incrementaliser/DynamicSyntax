"""Serialize DS trees and action steps into data consumed by generated Manim scenes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.gui.tree_viz import compute_tree_layout
from dylan.tree.tree import Tree


def _point(x_px: float, y_px: float, *, width_px: float, height_px: float) -> tuple[float, float]:
    """Map GUI-layout pixels to Manim scene coordinates."""
    x = -5.8 + (x_px / width_px) * 7.0
    y = 2.4 - (y_px / height_px) * 4.8
    return (round(x, 3), round(y, 3))


def serialize_tree(tree: Tree, *, width_px: float = 1000.0, height_px: float = 640.0) -> dict[str, Any]:
    """Return JSON-friendly node/edge layout data for *tree*."""
    layout = compute_tree_layout(tree, width_px, height_px, font_size=12.0)
    nodes: list[dict[str, Any]] = []
    for node in layout.nodes:
        x, y = _point(node.cx, node.cy, width_px=width_px, height_px=height_px)
        nodes.append(
            {
                "id": node.addr.address or "root",
                "label": node.label[:260],
                "x": x,
                "y": y,
                "w": round(max(0.55, min(2.25, node.w / 120.0)), 3),
                "h": round(max(0.35, min(1.4, node.h / 80.0)), 3),
            },
        )
    edges: list[dict[str, Any]] = []
    for edge in layout.edges:
        x1, y1 = _point(edge.x1, edge.y1, width_px=width_px, height_px=height_px)
        x2, y2 = _point(edge.x2, edge.y2, width_px=width_px, height_px=height_px)
        edges.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "style": edge.style})
    return {"nodes": nodes, "edges": edges}


def serialize_action_steps(
    steps: Sequence[Any],
    *,
    semantics: TTRRecordType | None,
    sentence: str,
) -> dict[str, Any]:
    """Return JSON-friendly animation data from action-step objects."""
    out_steps: list[dict[str, Any]] = []
    for step in steps:
        out_steps.append(
            {
                "word": getattr(step, "word", None) or "",
                "action": getattr(step, "action_name", ""),
                "before": serialize_tree(getattr(step, "before_tree")),
                "after": serialize_tree(getattr(step, "after_tree")),
            },
        )
    return {
        "sentence": sentence,
        "steps": out_steps,
        "semantics_tex": semantics.to_latex() if semantics is not None else "",
    }

