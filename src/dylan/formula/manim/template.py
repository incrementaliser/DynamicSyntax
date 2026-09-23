"""Generate standalone Manim scene code from serialized parse-animation data."""

from __future__ import annotations

import json
import re
from typing import Any

_SCENE_TEMPLATE = '''"""Auto-generated Manim scene for a dynamicsyntax parse."""

from __future__ import annotations

import json

from manim import *

DATA = json.loads(__PAYLOAD__)


class __CLASS_NAME__(Scene):
    """Action-level Dynamic Syntax parse animation."""

    def _card(self, lines: list[str], *, pointer: bool) -> VGroup:
        """Return a labelled node card; the pointer node uses a gold stroke."""
        text = VGroup(
            *[Text(line, font_size=20, color=WHITE) for line in lines],
        ).arrange(DOWN, buff=0.05, aligned_edge=LEFT)
        rect = SurroundingRectangle(
            text,
            buff=0.12,
            color="#ffca28" if pointer else "#b0bec5",
            fill_color="#102027" if pointer else "#263238",
            fill_opacity=1,
            stroke_width=3 if pointer else 1.6,
        )
        return VGroup(rect, text)

    def _edge(self, parent: Mobject, child: Mobject, style: str) -> Mobject:
        """Return a connector from the bottom of *parent* to the top of *child*."""
        start = parent.get_bottom()
        end = child.get_top()
        if style == "dashed":
            line = DashedLine(start, end, dash_length=0.08, stroke_width=2)
        elif style == "dotted":
            line = DashedLine(start, end, dash_length=0.04, stroke_width=2)
        else:
            line = Line(start, end, stroke_width=2)
        line.set_color("#b2dfdb")
        return line

    def _tree_group(self, tree_data: dict) -> VGroup:
        """Lay *tree_data* out from measured cards and scale it into the left pane."""
        specs = {node["id"]: node for node in tree_data.get("nodes", [])}
        children: dict[str, list[str]] = {}
        for node in tree_data.get("nodes", []):
            parent = node.get("parent")
            if parent:
                children.setdefault(parent, []).append(node["id"])
        for kids in children.values():
            kids.sort(key=lambda item: (len(item), item))
        root = tree_data.get("root") or ""
        if root not in specs:
            return VGroup(Text("empty tree", font_size=24, color=RED))

        def build(node_id: str) -> tuple[VGroup, VGroup]:
            """Return ``(card, subtree)`` for *node_id*."""
            spec = specs[node_id]
            card = self._card(spec.get("lines") or ["·"], pointer=bool(spec.get("pointer")))
            kid_ids = children.get(node_id, [])
            if not kid_ids:
                return card, VGroup(card)
            built = [build(kid) for kid in kid_ids]
            row = VGroup(*[group for _card, group in built]).arrange(RIGHT, buff=0.55)
            card.next_to(row, UP, buff=0.48)
            card.set_x(row.get_center()[0])
            edges = VGroup()
            for kid_id, (kid_card, _group) in zip(kid_ids, built, strict=True):
                edges.add(self._edge(card, kid_card, specs[kid_id].get("edge") or "solid"))
            return card, VGroup(edges, row, card)

        _root_card, tree = build(root)
        max_w, max_h = 9.0, 5.15
        if tree.width > max_w:
            tree.scale_to_fit_width(max_w)
        if tree.height > max_h:
            tree.scale_to_fit_height(max_h)
        tree.move_to([-1.85, -0.55, 0])
        return tree

    def _fit_block(self, block: VGroup, width: float) -> VGroup:
        """Scale *block* down so it stays inside the action column."""
        if block.width > width:
            block.scale_to_fit_width(width)
        return block

    def construct(self) -> None:
        """Animate the parse: utterance across the top, tree on the left, actions on the right."""
        self.camera.background_color = "#101820"
        latest = "#69f0ae"
        ink = "#e0f2f1"
        muted = "#eceff1"
        rule = "#80cbc4"

        steps = DATA.get("steps", [])
        if not steps:
            empty = Text("No action trace available", font_size=28).set_color(RED)
            self.play(Write(empty))
            self.wait(1)
            return

        utterance_label = Text("Utterance", font_size=26, weight=BOLD).set_color(ink)
        utterance_label.to_corner(UP + LEFT)
        hline = Line(start=LEFT * 7.1, end=RIGHT * 7.1)
        hline.next_to(utterance_label, DOWN, buff=0.18).set_color(rule)
        vline = Line(start=UP * 3.15, end=DOWN * 3.7)
        vline.set_x(3.15).set_color(rule)
        actions_header = Text("Actions", font_size=26, weight=BOLD).set_color(ink)
        actions_header.next_to(vline, RIGHT, buff=0.18).align_to(utterance_label, UP)
        self.play(FadeIn(utterance_label), Create(hline), Create(vline), FadeIn(actions_header))

        current_tree = self._tree_group(steps[0]["before"])
        self.play(FadeIn(current_tree))
        self.wait(0.3)

        words: list[Mobject] = []
        action_group = VGroup()
        column_width = 3.4
        for step in steps:
            if step.get("show_word") and step.get("word"):
                word_ref = words[-1] if words else utterance_label
                new_word = Text(step["word"], font_size=28).set_color(latest)
                new_word.next_to(word_ref, RIGHT, buff=0.16)
                anims: list[Animation] = [FadeIn(new_word)]
                if words:
                    anims.append(words[-1].animate.set_color(muted))
                self.play(*anims)
                words.append(new_word)

            action_name = step.get("action") or ""
            parts = [part.strip() for part in action_name.split(";") if part.strip()]
            if parts:
                lines = VGroup(
                    *[Text(part, font_size=18, color=latest) for part in parts],
                ).arrange(DOWN, buff=0.06, aligned_edge=LEFT)
                self._fit_block(lines, column_width)
                anchor = action_group if len(action_group) else actions_header
                lines.next_to(anchor, DOWN, buff=0.16, aligned_edge=LEFT)
                self.play(FadeIn(lines))
                if len(action_group):
                    action_group.set_color(muted)
                action_group.add(lines)
                if action_group.get_bottom()[1] < -3.45:
                    shift = -3.45 - action_group.get_bottom()[1]
                    self.play(action_group.animate.shift(UP * shift))
                self.wait(0.35)

            next_tree = self._tree_group(step["after"])
            self.play(FadeOut(current_tree), FadeIn(next_tree), run_time=0.6)
            current_tree = next_tree
            self.wait(0.7)

        semantics_lines = DATA.get("semantics_lines") or []
        if semantics_lines:
            self.play(FadeOut(current_tree))
            title = Text("Final semantics", font_size=28, weight=BOLD).set_color(ink)
            body = VGroup(
                *[Text(line, font_size=24, color=muted) for line in semantics_lines],
            ).arrange(DOWN, buff=0.08, aligned_edge=LEFT)
            if body.width > 10.5:
                body.scale_to_fit_width(10.5)
            panel = VGroup(title, body).arrange(DOWN, buff=0.28)
            panel.move_to([-1.7, -0.45, 0])
            self.play(FadeIn(panel))
        self.wait(1)
'''


def scene_class_name(name: str) -> str:
    """Return a valid Manim scene class name derived from *name*."""
    parts = re.findall(r"[A-Za-z0-9]+", name)
    base = "".join(part[:1].upper() + part[1:] for part in parts) or "DynamicSyntaxParse"
    if base[0].isdigit():
        base = "Scene" + base
    return base + "Scene"


def build_manim_scene_code(
    data: dict[str, Any], *, class_name: str = "DynamicSyntaxParseScene"
) -> str:
    """Return Python source for a Manim ``Scene`` rendering *data*."""
    payload = json.dumps(data, ensure_ascii=False)
    return _SCENE_TEMPLATE.replace("__CLASS_NAME__", class_name).replace(
        "__PAYLOAD__", repr(payload)
    )
