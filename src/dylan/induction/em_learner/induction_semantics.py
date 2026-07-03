"""Helpers for maximal TTR semantics during EM induction (Java DAG / ``ParserTuple`` context)."""

from __future__ import annotations

import re
from dataclasses import replace

from dylan.formula.formula import Formula
from dylan.formula.ttr_record_type import TTRRecordType

_RECORD_MV = re.compile(r"^[rR]\d+$")


def _cn_restrictor_manifest(gold: TTRRecordType) -> TTRRecordType | None:
    """Return the embedded CN restrictor record from *gold* (first ``r*`` manifest record)."""
    for f in gold.get_fields():
        lab_s = str(f.label) if f.label is not None else ""
        if lab_s.lower() == "r0" and isinstance(f.manifest_type, TTRRecordType):
            return f.manifest_type.clone()
    for f in gold.get_fields():
        if f.ds_type is None and isinstance(f.manifest_type, TTRRecordType):
            return f.manifest_type.clone()
    return None


def _field_record_for_metavar(gold: TTRRecordType, metavar: str) -> TTRRecordType | None:
    """Return the manifest record at *gold* for label ``r0`` / ``R1`` style metavar, or ``None``."""
    key = metavar.strip()
    key_lower = key.lower()
    if _RECORD_MV.match(key):
        return _cn_restrictor_manifest(gold)
    for f in gold.get_fields():
        lab_s = str(f.label) if f.label is not None else ""
        if lab_s.lower() == key_lower or lab_s == key:
            mt = f.manifest_type
            if isinstance(mt, TTRRecordType):
                return mt.clone()
    return None


def bind_metavar_path_domains(formula: Formula, gold: TTRRecordType) -> Formula:
    """Clone *formula* and attach ``TTRAbsolutePath.domain`` from *gold* where paths lack domains (Java context bind)."""
    from dylan.formula.disjunctive_type import DisjunctiveType
    from dylan.formula.epsilon_term import EpsilonTerm
    from dylan.formula.predicate_argument import PredicateArgumentFormula
    from dylan.formula.ttr_field import TTRField
    from dylan.formula.ttr_infix_expression import TTRInfixExpression
    from dylan.formula.ttr_path import TTRAbsolutePath

    if isinstance(formula, TTRAbsolutePath):
        if formula.domain is not None or formula.name is None:
            return formula.clone()
        dom = _field_record_for_metavar(gold, formula.name.label)
        if dom is not None:
            return replace(formula, domain=dom)
        return formula.clone()

    if isinstance(formula, TTRRecordType):
        out = TTRRecordType()
        for f in formula.get_fields():
            mt = f.manifest_type
            if mt is not None:
                mt2 = bind_metavar_path_domains(mt, gold)
                out.add_field(TTRField(f.label, f.ds_type, mt2))
            else:
                out.add_field(f.clone())
        return out

    if isinstance(formula, DisjunctiveType):
        return DisjunctiveType(
            bind_metavar_path_domains(formula.arg1, gold),
            bind_metavar_path_domains(formula.arg2, gold),
        )

    if isinstance(formula, TTRInfixExpression):
        return TTRInfixExpression(
            formula.functor,
            bind_metavar_path_domains(formula.arg1, gold),
            bind_metavar_path_domains(formula.arg2, gold),
        )

    if isinstance(formula, EpsilonTerm):
        return EpsilonTerm(
            formula.predicate.name,
            bind_metavar_path_domains(formula.restrictor_head, gold),
            bind_metavar_path_domains(formula.restrictor, gold),
        )

    if isinstance(formula, PredicateArgumentFormula):
        return PredicateArgumentFormula(
            formula.predicate,
            tuple(bind_metavar_path_domains(a, gold) for a in formula.arguments),
        )

    from dylan.formula.ttr_lambda import TTRLambdaAbstract

    if isinstance(formula, TTRLambdaAbstract):
        return TTRLambdaAbstract(
            formula.variable,
            bind_metavar_path_domains(formula.body, gold),  # type: ignore[arg-type]
        )

    if hasattr(formula, "clone"):
        return formula.clone()
    return formula
