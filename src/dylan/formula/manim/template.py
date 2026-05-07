"""Generate standalone Manim scene code from serialized parse-animation data."""

from __future__ import annotations

import json
import re
from typing import Any


def scene_class_name(name: str) -> str:
    """Return a valid Manim scene class name derived from *name*."""
    parts = re.findall(r"[A-Za-z0-9]+", name)
    base = "".join(p[:1].upper() + p[1:] for p in parts) or "DynamicSyntaxParse"
    if base[0].isdigit():
        base = "Scene" + base
    return base + "Scene"


def build_manim_scene_code(data: dict[str, Any], *, class_name: str = "DynamicSyntaxParseScene") -> str:
    """Return Python source for a Manim ``Scene`` rendering *data*."""
    payload = json.dumps(data, ensure_ascii=True)
    return f'''"""Auto-generated Manim scene for a dynamicsyntax parse."""

from __future__ import annotations

import json
from manim import *

DATA = json.loads(r"""{payload}""")


class {class_name}(Scene):
    """Action-level Dynamic Syntax parse animation."""

    def _tex_escape(self, value: str) -> str:
        """Return *value* escaped for simple Manim Tex text labels."""
        parts = []
        for char in value:
            if char == "\\\\":
                parts.append(r"\\textbackslash{{}}")
            elif char == "&":
                parts.append(r"\\&")
            elif char == "%":
                parts.append(r"\\%")
            elif char == "$":
                parts.append(r"\\$")
            elif char == "#":
                parts.append(r"\\#")
            elif char == "_":
                parts.append(r"\\_")
            elif char == "{{":
                parts.append(r"\\{{")
            elif char == "}}":
                parts.append(r"\\}}")
            elif char == "^":
                parts.append(r"\\textasciicircum{{}}")
            elif char == "~":
                parts.append(r"\\textasciitilde{{}}")
            else:
                parts.append(char)
        return "".join(parts)

    def _tree_group(self, tree_data: dict) -> VGroup:
        """Return a Manim group containing plain tree labels and connecting edges."""
        group = VGroup()
        edge_group = VGroup()
        for edge in tree_data.get("edges", []):
            start = [edge["x1"], edge["y1"], 0]
            end = [edge["x2"], edge["y2"], 0]
            line = Line(start=start, end=end, stroke_width=2)
            if edge.get("style") == "dashed":
                line = DashedLine(start=start, end=end, stroke_width=2)
            elif edge.get("style") == "dotted":
                line = DashedLine(start=start, end=end, dash_length=0.04, stroke_width=2)
            line.set_color("#9dfcf9")
            edge_group.add(line)
        node_group = VGroup()
        for node in tree_data.get("nodes", []):
            label = Text(node["label"], font_size=16, line_spacing=0.8)
            label.set_color(WHITE)
            label.move_to([node["x"], node["y"], 0])
            label.scale_to_fit_width(max(0.3, node["w"]))
            if label.height > node["h"]:
                label.scale_to_fit_height(max(0.1, node["h"]))
            node_group.add(label)
        group.add(edge_group, node_group)
        return group

    def construct(self) -> None:
        """Animate the parse in the handwritten example layout."""
        self.camera.background_color = "#263238"
        green_latest = "#49b372"
        blue_lines = "#9dfcf9"
        blue_font = "#1e6b69"

        steps = DATA.get("steps", [])
        if not steps:
            empty = Text("No action trace available", font_size=28).set_color(RED)
            self.play(Write(empty))
            self.wait(1)
            return

        utterance_label = MarkupText("Utterance: ", weight=HEAVY)
        utterance_label.set_color(blue_font).scale(0.6).to_corner(UP + LEFT)
        hline = Line(start=LEFT * 7.5, end=RIGHT * 7.5)
        hline.next_to(utterance_label, DOWN, buff=0.25).set_color(blue_lines)
        vline = Line(start=UP * 3.8, end=DOWN * 4.0)
        vline.set_x(3.2).set_color(blue_lines)
        actions_header = MarkupText("<u>Actions</u>", weight=HEAVY)
        actions_header.set_color(blue_font).scale(0.6)
        actions_header.next_to(vline, RIGHT, buff=0.2).align_to(utterance_label, UP)
        self.play(Write(utterance_label), Create(hline), Create(vline), Write(actions_header))

        current_tree = self._tree_group(steps[0]["before"])
        self.play(FadeIn(current_tree))
        self.wait(0.4)

        words = []
        action_labels = []
        for step in steps:
            word = step.get("word", "")
            if word:
                word_ref = words[-1] if words else utterance_label
                new_word = Tex(self._tex_escape(word)).scale(0.75).set_color(green_latest)
                new_word.next_to(word_ref, RIGHT, buff=0.1)
                word_anims = [Write(new_word)]
                if words:
                    word_anims.append(words[-1].animate.set_color(WHITE))
                self.play(*word_anims)
                words.append(new_word)

            action_name = step.get("action", "")
            if action_name:
                action_ref = action_labels[-1] if action_labels else actions_header
                new_action = Tex(self._tex_escape(action_name)).scale(0.38).set_color(green_latest)
                new_action.next_to(action_ref, DOWN, buff=0.22)
                action_anims = [Write(new_action)]
                if action_labels:
                    action_anims.append(action_labels[-1].animate.set_color(WHITE))
                self.play(*action_anims)
                action_labels.append(new_action)
                self.wait(1.0)

            next_tree = self._tree_group(step["after"])
            self.play(FadeOut(current_tree), FadeIn(next_tree), run_time=0.75)
            current_tree = next_tree
            self.wait(1.5)

        sem = DATA.get("semantics_tex", "")
        if sem:
            sem_title = Text("Final semantics", font_size=24, weight=BOLD).set_color("#1e6b69")
            sem_tex = MathTex(sem, font_size=30)
            sem_tex.scale_to_fit_width(6.0)
            sem_group = VGroup(sem_title, sem_tex).arrange(DOWN, buff=0.25)
            sem_group.to_edge(DOWN)
            self.play(FadeIn(sem_group))
        self.wait(1)
'''

