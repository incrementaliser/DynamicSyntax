"""Extract subject / verb / object roles from TTR semantics, trees, or GS2013 fields."""

from __future__ import annotations

import re

from dylan.formula.formula import Formula
from dylan.formula.predicate_argument import PredicateArgumentFormula
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.tree.label.labels import FormulaLabel, TypeLabel
from dylan.tree.tree import Tree
from dylan.type.dstype import DSType
from dylan.vss.types import GS2013Sentence, SVORoles

_SUBJ_RE = re.compile(r"^subj$", re.I)
_OBJ_RE = re.compile(r"^obj$", re.I)


def surface_svo_sentence(subj: str, verb: str, obj: str) -> str:
    """Build a space-separated surface string in GS2013 evaluation order (subj verb obj)."""
    return f"{subj} {verb} {obj}".strip()


def roles_from_gs2013(sent: GS2013Sentence, *, landmark_as_verb: bool = True) -> SVORoles:
    """Build :class:`~dylan.vss.types.SVORoles` from a dataset row."""
    verb = sent.landmark if landmark_as_verb else sent.verb
    return SVORoles(
        subj=sent.subj,
        landmark=sent.landmark,
        obj=sent.obj,
        verb=sent.verb,
        parse_ok=True,
        source="dataset",
    )


def _walk_predicates(formula: Formula | None) -> list[PredicateArgumentFormula]:
    """Collect predicate applications in a formula tree."""
    if formula is None:
        return []
    found: list[PredicateArgumentFormula] = []
    if isinstance(formula, PredicateArgumentFormula):
        found.append(formula)
        for arg in formula.arguments:
            found.extend(_walk_predicates(arg))
    elif isinstance(formula, TTRRecordType):
        for field in formula.get_fields():
            mt = field.manifest_type
            if isinstance(mt, Formula):
                found.extend(_walk_predicates(mt))
    return found


def _entity_name_from_arg(arg: Formula) -> str | None:
    """Extract a lemma-like string from a TTR argument."""
    text = str(arg).strip()
    if not text:
        return None
    text = text.strip("[]")
    if ":" in text:
        text = text.split(":", 1)[0].strip()
    return text.lower() if text else None


def roles_from_ttr(semantics: TTRRecordType | None) -> SVORoles | None:
    """Extract SVO from ``subj``/``obj`` predicates and event head in maximal TTR."""
    if semantics is None:
        return None
    subj_name: str | None = None
    obj_name: str | None = None
    event_pred: str | None = None
    for pred in _walk_predicates(semantics):
        name = pred.predicate.name.lower()
        if _SUBJ_RE.match(name) and pred.arguments:
            subj_name = _entity_name_from_arg(pred.arguments[-1])
        elif _OBJ_RE.match(name) and pred.arguments:
            obj_name = _entity_name_from_arg(pred.arguments[-1])
    for field in semantics.get_fields():
        if "es" in str(field) and field.manifest_type is not None:
            mt = field.manifest_type
            if isinstance(mt, PredicateArgumentFormula):
                event_pred = mt.predicate.name.lower()
            else:
                inner = _walk_predicates(mt if isinstance(mt, Formula) else None)
                if inner:
                    event_pred = inner[0].predicate.name.lower()
    if subj_name and obj_name and event_pred:
        return SVORoles(
            subj=subj_name,
            landmark=event_pred,
            obj=obj_name,
            verb=event_pred,
            parse_ok=True,
            source="ttr",
        )
    return None


def roles_from_tree_heuristic(tree: Tree, words: tuple[str, ...]) -> SVORoles | None:
    """Heuristic SVO from tree types and utterance words (transitive spine)."""
    if len(words) < 3:
        return None
    has_tran = False
    for node in tree.values():
        ty = node.get_type()
        if ty is not None and "e>" in str(ty) and "t" in str(ty):
            has_tran = True
            break
    if not has_tran and len(words) == 3:
        return SVORoles(
            subj=words[0],
            landmark=words[1],
            obj=words[2],
            verb=words[1],
            parse_ok=True,
            source="word_order",
        )
    e_nodes: list[str] = []
    for node in tree.values():
        ty = node.get_type()
        if ty == DSType.e:
            fo = node.get_formula_label()
            if fo is not None:
                e_nodes.append(str(fo.get_formula()).lower())
    if len(e_nodes) >= 2 and len(words) >= 3:
        return SVORoles(
            subj=e_nodes[0] if e_nodes else words[0],
            landmark=words[1],
            obj=e_nodes[-1] if len(e_nodes) > 1 else words[2],
            verb=words[1],
            parse_ok=True,
            source="tree",
        )
    return None


def roles_from_parse(
    semantics: TTRRecordType | None,
    tree: Tree | None,
    words: tuple[str, ...],
    *,
    fallback: GS2013Sentence | None = None,
    allow_dataset_fallback: bool = True,
) -> SVORoles:
    """Resolve roles using TTR, tree, then optional dataset fallback."""
    if semantics is not None:
        ttr_roles = roles_from_ttr(semantics)
        if ttr_roles is not None:
            return ttr_roles
    if tree is not None:
        tree_roles = roles_from_tree_heuristic(tree, words)
        if tree_roles is not None:
            return tree_roles
    if allow_dataset_fallback and fallback is not None:
        roles = roles_from_gs2013(fallback)
        return SVORoles(
            subj=roles.subj,
            landmark=roles.landmark,
            obj=roles.obj,
            verb=roles.verb,
            parse_ok=False,
            source="dataset_fallback",
        )
    if len(words) >= 3:
        return SVORoles(
            subj=words[0],
            landmark=words[1],
            obj=words[2],
            verb=words[1],
            parse_ok=False,
            source="words",
        )
    raise ValueError("could not extract SVO roles")
