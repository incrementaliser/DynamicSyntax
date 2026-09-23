"""Render DS types, TTR formulae, and node labels as LaTeX math or plain display text.

Tree and semantics export used to paste ``str(formula)`` into math mode (so ``==``
became a broken ``\\mathrel{ : =}`` and ``arrive`` was a product of letters).
These helpers walk the formula objects instead.
"""

from __future__ import annotations

import re

from dylan.formula.atomic_formula import AtomicFormula
from dylan.formula.bound_formula_variable import BoundFormulaVariable
from dylan.formula.fol_lambda import FOLLambdaAbstract
from dylan.formula.formula import Formula
from dylan.formula.formula_metavariable import FormulaMetavariable
from dylan.formula.opaque_formula import OpaqueFormula
from dylan.formula.opaque_ttr_spec import OpaqueTTRSpec
from dylan.formula.predicate_argument import PredicateArgumentFormula
from dylan.formula.ttr_field import TTRField
from dylan.formula.ttr_infix_expression import TTRInfixExpression
from dylan.formula.ttr_lambda import TTRLambdaAbstract
from dylan.formula.ttr_path import TTRAbsolutePath, TTRRelativePath
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.formula.variable import Variable
from dylan.tree.label.labels import (
    BottomLabel,
    FeatureLabel,
    FormulaLabel,
    NegatedLabel,
    Requirement,
    TypeLabel,
    UnaryPredicateLabel,
)
from dylan.tree.node import Node
from dylan.type.dstype import BasicType, ConstructedType, DSType

from dylan.formula.latex.escape import latex_escape_math

_NAME_RE = re.compile(r"^([A-Za-z]+)(\d*)$")
_DIGIT_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_BASIC_TYPE_TEX = {
    "e": "e",
    "t": "t",
    "cn": r"\mathit{cn}",
    "es": "e_{s}",
    "cnev": r"\mathit{cnev}",
}
_BASIC_TYPE_DISPLAY = {
    "e": "e",
    "t": "t",
    "cn": "cn",
    "es": "eₛ",
    "cnev": "cnev",
}
_MERGE = "++"


def math_symbol(name: str) -> str:
    """Return *name* as a math identifier, with trailing digits in a subscript."""
    match = _NAME_RE.fullmatch(name)
    if match is None:
        return r"\mbox{" + latex_escape_math(name) + "}"
    base, digits = match.group(1), match.group(2)
    body = base if len(base) == 1 else r"\mathit{" + base + "}"
    if digits:
        return body + "_{" + digits + "}"
    return body


def _subscripted_symbol(base_tex: str, manifest_tex: str) -> str:
    """Attach a manifest as a subscript without stacking two raw subscripts."""
    return "{" + base_tex + "}_{=" + manifest_tex + "}"


def display_symbol(name: str) -> str:
    """Return *name* with trailing digits as Unicode subscripts."""
    match = _NAME_RE.fullmatch(name)
    if match is None:
        return name
    base, digits = match.group(1), match.group(2)
    if not digits:
        return base
    return base + digits.translate(_DIGIT_SUB)


def dstype_to_latex(ds_type: DSType) -> str:
    """Return math-mode LaTeX for a dynamic-syntax type (``e_{s}``, ``e \\rightarrow t``)."""
    if isinstance(ds_type, BasicType):
        return _BASIC_TYPE_TEX.get(ds_type.name, math_symbol(ds_type.name))
    if isinstance(ds_type, ConstructedType):
        left = dstype_to_latex(ds_type.from_type)
        right = dstype_to_latex(ds_type.to_type)
        if isinstance(ds_type.from_type, ConstructedType):
            left = "(" + left + ")"
        if isinstance(ds_type.to_type, ConstructedType):
            right = "(" + right + ")"
        return left + r" \rightarrow " + right
    return r"\mbox{" + latex_escape_math(str(ds_type)) + "}"


def dstype_to_display(ds_type: DSType) -> str:
    """Return a Unicode display string for a dynamic-syntax type."""
    if isinstance(ds_type, BasicType):
        return _BASIC_TYPE_DISPLAY.get(ds_type.name, ds_type.name)
    if isinstance(ds_type, ConstructedType):
        left = dstype_to_display(ds_type.from_type)
        right = dstype_to_display(ds_type.to_type)
        if isinstance(ds_type.from_type, ConstructedType):
            left = "(" + left + ")"
        if isinstance(ds_type.to_type, ConstructedType):
            right = "(" + right + ")"
        return left + "→" + right
    return str(ds_type)


def _is_compact_manifest(formula: Formula) -> bool:
    """Whether *formula* is short enough to sit in a field subscript."""
    return isinstance(
        formula,
        (
            AtomicFormula,
            Variable,
            PredicateArgumentFormula,
            TTRAbsolutePath,
            TTRRelativePath,
            FormulaMetavariable,
        ),
    )


def record_to_latex(record: TTRRecordType) -> str:
    """Return a two-column TTR record in math mode (label / manifest subscript, type)."""
    if record.is_empty():
        return r"\left[\,\right]"
    rows = [_field_to_latex(field) for field in record._fields]
    body = r" \\ ".join(rows)
    return r"\left[\begin{array}{@{}l@{\,:\,}l@{}}" + body + r"\end{array}\right]"


def _field_to_latex(field: TTRField) -> str:
    """Return one ``label & type`` row for *field*."""
    label = math_symbol(str(field.label))
    type_tex = dstype_to_latex(field.ds_type) if field.ds_type is not None else ""
    manifest = field.manifest_type
    if manifest is None:
        return f"{label} & {type_tex}"
    manifest_tex = formula_to_latex(manifest)
    if type_tex and _is_compact_manifest(manifest):
        return _subscripted_symbol(label, manifest_tex) + " & " + type_tex
    if type_tex:
        return label + " & " + manifest_tex + r"\!:\!" + type_tex
    return label + " & " + manifest_tex


def _infix_operator_tex(name: str) -> str:
    """Return a math-mode operator for a TTR infix functor name."""
    if name == _MERGE:
        return r"\mathbin{+\mkern-6mu+}"
    if name == "||":
        return r"\mathbin{||}"
    return r"\mathbin{\mbox{" + latex_escape_math(name) + "}}"


def formula_to_latex(formula: Formula) -> str:
    """Return math-mode LaTeX for *formula* (records, lambdas, predicates, paths)."""
    if isinstance(formula, TTRRecordType):
        return record_to_latex(formula)
    if isinstance(formula, (TTRLambdaAbstract, FOLLambdaAbstract)):
        var = math_symbol(formula.variable.name)
        return r"\lambda " + var + "." + formula_to_latex(formula.body)
    if isinstance(formula, TTRInfixExpression):
        op = _infix_operator_tex(formula.functor.name)
        left = formula_to_latex(formula.arg1)
        right = formula_to_latex(formula.arg2)
        return r"\left(" + left + " " + op + " " + right + r"\right)"
    if isinstance(formula, PredicateArgumentFormula):
        pred = math_symbol(formula.predicate.name)
        if not formula.arguments:
            return pred
        args = ", ".join(formula_to_latex(arg) for arg in formula.arguments)
        return pred + "(" + args + ")"
    if isinstance(formula, TTRAbsolutePath):
        head = math_symbol(formula.name.label) if formula.name is not None else ""
        rest = ".".join(math_symbol(label.label) for label in formula.labels)
        return (head + "." + rest) if head else "." + rest
    if isinstance(formula, TTRRelativePath):
        if not formula.labels:
            return ""
        parts = [math_symbol(label.label) for label in formula.labels]
        if len(formula.labels) == 1:
            return "." + parts[0]
        return ".".join(parts)
    if isinstance(formula, TTRField):
        return _field_to_latex(formula)
    if isinstance(formula, (Variable, AtomicFormula, FormulaMetavariable)):
        name = formula.name
        return math_symbol(name)
    if isinstance(formula, BoundFormulaVariable):
        return math_symbol(formula.name)
    if isinstance(formula, (OpaqueFormula, OpaqueTTRSpec)):
        raw = formula.text if isinstance(formula, OpaqueFormula) else formula.source
        return r"\mbox{" + latex_escape_math(raw) + "}"
    return r"\mbox{" + latex_escape_math(str(formula)) + "}"


def _field_to_display(field: TTRField) -> str:
    """Return one plain-text field, using ``≔`` for a manifest value."""
    label = display_symbol(str(field.label))
    type_txt = dstype_to_display(field.ds_type) if field.ds_type is not None else ""
    manifest = field.manifest_type
    if manifest is None:
        return f"{label} : {type_txt}" if type_txt else label
    manifest_txt = formula_to_display(manifest)
    if type_txt:
        return f"{label} ≔ {manifest_txt} : {type_txt}"
    return f"{label} ≔ {manifest_txt}"


def record_display_lines(record: TTRRecordType) -> list[str]:
    """Return bracketed record lines, one field per line."""
    if record.is_empty():
        return ["[ ]"]
    fields = list(record._fields)
    if len(fields) == 1:
        return ["[ " + _field_to_display(fields[0]) + " ]"]
    lines = ["[ " + _field_to_display(fields[0])]
    lines.extend("  " + _field_to_display(field) for field in fields[1:-1])
    lines.append("  " + _field_to_display(fields[-1]) + " ]")
    return lines


def formula_to_display(formula: Formula) -> str:
    """Return a single-line Unicode rendering of *formula*."""
    return " ".join(formula_display_lines(formula))


def formula_display_lines(formula: Formula) -> list[str]:
    """Return readable Unicode lines for *formula* (records break one field per line)."""
    if isinstance(formula, TTRRecordType):
        return record_display_lines(formula)
    if isinstance(formula, (TTRLambdaAbstract, FOLLambdaAbstract)) and isinstance(
        formula.body, TTRInfixExpression
    ):
        body = formula.body
        if body.functor.name == _MERGE and isinstance(body.arg2, TTRRecordType):
            var = display_symbol(formula.variable.name)
            left = formula_to_display(body.arg1)
            rec_lines = record_display_lines(body.arg2)
            head = f"λ{var}.({left} ++"
            if len(rec_lines) == 1:
                return [head + " " + rec_lines[0] + ")"]
            return [head + " " + rec_lines[0], *rec_lines[1:-1], rec_lines[-1] + ")"]
    if isinstance(formula, (TTRLambdaAbstract, FOLLambdaAbstract)):
        var = display_symbol(formula.variable.name)
        body_lines = formula_display_lines(formula.body)
        if len(body_lines) == 1:
            return [f"λ{var}.{body_lines[0]}"]
        return [f"λ{var}.{body_lines[0]}", *body_lines[1:]]
    if isinstance(formula, TTRInfixExpression):
        op = "++" if formula.functor.name == _MERGE else formula.functor.name
        return [f"({formula_to_display(formula.arg1)} {op} {formula_to_display(formula.arg2)})"]
    if isinstance(formula, PredicateArgumentFormula):
        pred = display_symbol(formula.predicate.name)
        if not formula.arguments:
            return [pred]
        args = ", ".join(formula_to_display(arg) for arg in formula.arguments)
        return [f"{pred}({args})"]
    if isinstance(formula, TTRAbsolutePath):
        head = display_symbol(formula.name.label) if formula.name is not None else ""
        rest = ".".join(display_symbol(label.label) for label in formula.labels)
        return [(head + "." + rest) if head else "." + rest]
    if isinstance(formula, TTRRelativePath):
        if not formula.labels:
            return [""]
        parts = [display_symbol(label.label) for label in formula.labels]
        if len(parts) == 1:
            return ["." + parts[0]]
        return [".".join(parts)]
    if isinstance(formula, (Variable, AtomicFormula, FormulaMetavariable, BoundFormulaVariable)):
        return [display_symbol(formula.name)]
    if isinstance(formula, (OpaqueFormula, OpaqueTTRSpec)):
        raw = formula.text if isinstance(formula, OpaqueFormula) else formula.source
        return [raw]
    return [str(formula)]


def _decoration_latex(label: object) -> str | None:
    """Return a math fragment for a non-formula node label, or ``None`` to skip."""
    if isinstance(label, Requirement) and isinstance(label.inner, TypeLabel):
        return "?Ty(" + dstype_to_latex(label.inner.type) + ")"
    if isinstance(label, TypeLabel):
        return "Ty(" + dstype_to_latex(label.type) + ")"
    if isinstance(label, Requirement) and isinstance(label.inner, FeatureLabel):
        return "?+" + math_symbol(label.inner.name)
    if isinstance(label, FeatureLabel):
        return "+" + math_symbol(label.name)
    if isinstance(label, BottomLabel):
        return r"\langle\downarrow\rangle\bot"
    if isinstance(label, UnaryPredicateLabel):
        return math_symbol(label.predicate) + "(" + math_symbol(label.arg) + ")"
    if isinstance(label, NegatedLabel):
        inner = _decoration_latex(label.inner)
        if inner is None:
            inner = r"\mbox{" + latex_escape_math(str(label.inner)) + "}"
        return r"\neg " + inner
    if isinstance(label, FormulaLabel):
        return None
    return r"\mbox{" + latex_escape_math(str(label)) + "}"


def _decoration_display(label: object) -> str | None:
    """Return a Unicode fragment for a non-formula node label, or ``None`` to skip."""
    if isinstance(label, Requirement) and isinstance(label.inner, TypeLabel):
        return "?Ty(" + dstype_to_display(label.inner.type) + ")"
    if isinstance(label, TypeLabel):
        return "Ty(" + dstype_to_display(label.type) + ")"
    if isinstance(label, Requirement) and isinstance(label.inner, FeatureLabel):
        return "?+" + display_symbol(label.inner.name)
    if isinstance(label, FeatureLabel):
        return "+" + display_symbol(label.name)
    if isinstance(label, BottomLabel):
        return "⟨↓⟩⊥"
    if isinstance(label, UnaryPredicateLabel):
        return display_symbol(label.predicate) + "(" + display_symbol(label.arg) + ")"
    if isinstance(label, NegatedLabel):
        inner = _decoration_display(label.inner)
        return "¬" + (inner if inner is not None else str(label.inner))
    if isinstance(label, FormulaLabel):
        return None
    return str(label)


def _formula_from_label(label: object) -> tuple[Formula, bool] | None:
    """Return ``(formula, required)`` when *label* carries a semantic formula."""
    if isinstance(label, FormulaLabel):
        return label.get_formula(), False
    if isinstance(label, Requirement) and isinstance(label.inner, FormulaLabel):
        return label.inner.get_formula(), True
    return None


def node_math_lines(node: Node, *, pointed: bool) -> list[str]:
    """Return ``$...$`` lines for one DS node: decorations, then formulae.

    Addresses are omitted; the tree geometry already shows where the node sits.
    The pointed node is marked with ``\\ptr``.
    """
    decorations: list[str] = []
    formulas: list[Formula] = []
    required: list[bool] = []
    for label in node.labels:
        carried = _formula_from_label(label)
        if carried is not None:
            formulas.append(carried[0])
            required.append(carried[1])
            continue
        piece = _decoration_latex(label)
        if piece:
            decorations.append(piece)
    if pointed:
        decorations.append(r"\ptr")
    lines: list[str] = []
    if decorations:
        lines.append("$" + ", ".join(decorations) + "$")
    for formula, is_required in zip(formulas, required, strict=True):
        math = formula_to_latex(formula)
        if is_required:
            math = "?" + math
        lines.append("$" + math + "$")
    if not lines:
        lines.append(r"$\cdot$")
    return lines


def node_display_lines(node: Node, *, pointed: bool) -> list[str]:
    """Return plain-text lines for one DS node (decorations, then formulae)."""
    decorations: list[str] = []
    formula_lines: list[str] = []
    for label in node.labels:
        carried = _formula_from_label(label)
        if carried is not None:
            formula, is_required = carried
            lines = formula_display_lines(formula)
            if is_required and lines:
                lines = ["?" + lines[0], *lines[1:]]
            formula_lines.extend(lines)
            continue
        piece = _decoration_display(label)
        if piece:
            decorations.append(piece)
    if pointed:
        decorations.append("◆")
    lines: list[str] = []
    if decorations:
        lines.append(", ".join(decorations))
    lines.extend(formula_lines)
    return lines or ["·"]
