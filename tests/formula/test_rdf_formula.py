"""Tests for delimited RDF formulae: parse, equality, union, and application."""

from __future__ import annotations

import pytest
from rdflib import Literal, URIRef, XSD

from dylan.formula.fol_lambda import FOLLambdaAbstract
from dylan.formula.formula import Formula
from dylan.formula.rdf.demo import john_likes_mary
from dylan.formula.rdf.graph import ROOT, RDFGraph
from dylan.formula.rdf.lambda_abstract import RDFLambdaAbstract
from dylan.formula.ttr_lambda import TTRLambdaAbstract
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.formula.variable import Variable

_EX = "http://example.org/"


def _iri(local: str) -> URIRef:
    """Return ``http://example.org/{local}``."""
    return URIRef(_EX + local)


def test_round_trip_is_isomorphic() -> None:
    """Printing an RDF graph and parsing it again yields an equal graph."""
    source = "rdf{ @prefix ex: <http://example.org/> . ex:a ex:p ex:b . }"
    graph = Formula.create(source)
    assert isinstance(graph, RDFGraph)
    again = Formula.create(str(graph))
    assert again == graph
    assert hash(again) == hash(graph)


def test_prefix_names_do_not_affect_equality() -> None:
    """The same triples under different prefixes are one formula."""
    prefixed = Formula.create(
        "rdf{ @prefix ex: <http://example.org/> . ex:a ex:p ex:b . }",
    )
    renamed = Formula.create(
        "rdf{ @prefix ns: <http://example.org/> . ns:a ns:p ns:b . }",
    )
    assert isinstance(prefixed, RDFGraph)
    assert prefixed == renamed
    assert hash(prefixed) == hash(renamed)
    assert hash(prefixed) == hash(prefixed)
    assert len({prefixed, renamed}) == 1


def test_blank_node_labels_do_not_affect_equality() -> None:
    """A Turtle blank-node list and an explicit blank node are the same graph."""
    listed = Formula.create(
        'rdf{ <http://example.org/a> <http://example.org/p> [ <http://example.org/q> "x" ] . }',
    )
    named = Formula.create(
        "rdf{ <http://example.org/a> <http://example.org/p> _:b1 . "
        '_:b1 <http://example.org/q> "x" . }',
    )
    assert isinstance(listed, RDFGraph)
    assert listed == named
    assert hash(listed) == hash(named)
    assert Formula.create(str(listed)) == listed


def test_datatype_caret_stays_inside_turtle() -> None:
    """``^^`` inside ``rdf{ }`` is a literal datatype, not a lambda binder."""
    graph = Formula.create(
        "rdf{ <http://example.org/a> <http://example.org/p> "
        '"hi"^^<http://www.w3.org/2001/XMLSchema#string> . }',
    )
    assert isinstance(graph, RDFGraph)
    literals = [obj for _, _, obj in graph.to_rdflib() if isinstance(obj, Literal)]
    assert literals == [Literal("hi", datatype=XSD.string)]


def test_brace_inside_literal_is_one_triple() -> None:
    """A ``}`` inside a Turtle string does not close ``rdf{ }``."""
    graph = Formula.create(
        'rdf{ <http://example.org/a> <http://example.org/p> "see } this" . }',
    )
    assert isinstance(graph, RDFGraph)
    assert len(graph.to_rdflib()) == 1


def test_rdf_lambda_is_not_a_fol_lambda() -> None:
    """``G1^rdf{ ... }`` is an RDF lambda, and a bare ``G1`` is not a DS variable."""
    lam = Formula.create(
        "G1^rdf{ <urn:dylan:var:G1> <http://example.org/p> <http://example.org/o> . }",
    )
    assert isinstance(lam, RDFLambdaAbstract)
    assert not isinstance(lam, FOLLambdaAbstract)
    assert lam.variable.name == "G1"
    assert Formula.create("G1") is None
    assert Variable.is_variable_string("G1") is False


def test_ttr_and_fol_parses_unchanged() -> None:
    """Existing TTR records, TTR lambdas, and FOL lambdas still parse as themselves."""
    fol = Formula.create("x^y")
    assert isinstance(fol, FOLLambdaAbstract)
    record = Formula.create("[x1 : e|head==x1 : e]")
    assert isinstance(record, TTRRecordType)
    ttr_lambda = Formula.create("R1^[x1 : e|head==x1 : e]")
    assert isinstance(ttr_lambda, TTRLambdaAbstract)
    assert Formula.create(
        "<http://example.org/a> <http://example.org/p> <http://example.org/b> .",
    ) is None


def test_conjoin_unions_graphs_and_keeps_a_lambda_spine() -> None:
    """Conjoin is triple union. A lambda conjoined with a graph stays a lambda."""
    left = Formula.create(
        "rdf{ <http://example.org/a> <http://example.org/p> <http://example.org/b> . }",
    )
    right = Formula.create(
        "rdf{ <http://example.org/c> <http://example.org/q> <http://example.org/d> . }",
    )
    both = Formula.create(
        "rdf{ <http://example.org/a> <http://example.org/p> <http://example.org/b> . "
        "<http://example.org/c> <http://example.org/q> <http://example.org/d> . }",
    )
    assert isinstance(left, RDFGraph)
    assert isinstance(right, RDFGraph)
    assert left.conjoin(right) == both
    lam = Formula.create(
        "G1^rdf{ <urn:dylan:var:G1> <http://example.org/p> <http://example.org/o> . }",
    )
    assert isinstance(lam, RDFLambdaAbstract)
    merged = lam.conjoin(right)
    assert isinstance(merged, RDFLambdaAbstract)
    assert merged.variable.name == "G1"
    assert merged.get_core() == Formula.create(
        "rdf{ <urn:dylan:var:G1> <http://example.org/p> <http://example.org/o> . "
        "<http://example.org/c> <http://example.org/q> <http://example.org/d> . }",
    )
    other = Formula.create(
        "G2^rdf{ <urn:dylan:var:G2> <http://example.org/p> <http://example.org/o> . }",
    )
    with pytest.raises(TypeError):
        lam.conjoin(other)


def test_two_applications_do_not_identify_roots() -> None:
    """Each argument root becomes its own blank node, linked by the verb."""
    verb = Formula.create(
        "G1^G2^rdf{ <urn:dylan:var:G1> <http://example.org/likes> <urn:dylan:var:G2> . }",
    )
    john = Formula.create('rdf{ <urn:dylan:root> <http://example.org/name> "John" . }')
    mary = Formula.create('rdf{ <urn:dylan:root> <http://example.org/name> "Mary" . }')
    assert isinstance(verb, RDFLambdaAbstract)
    once = verb.beta_reduce(john)
    assert isinstance(once, RDFLambdaAbstract)
    assert once.variable.name == "G2"
    result = once.beta_reduce(mary)
    assert isinstance(result, RDFGraph)
    triples = list(result.to_rdflib())
    name = _iri("name")
    likes = _iri("likes")
    name_subjects = {subj for subj, pred, _obj in triples if pred == name}
    like_triples = [(subj, obj) for subj, pred, obj in triples if pred == likes]
    assert len(name_subjects) == 2
    assert len(like_triples) == 1
    subject, object_ = like_triples[0]
    assert subject != object_
    assert {subject, object_} == name_subjects
    assert all(ROOT not in (subj, pred, obj) for subj, pred, obj in triples)
    assert all(
        not (isinstance(term, URIRef) and str(term).startswith("urn:dylan:var:"))
        for subj, pred, obj in triples
        for term in (subj, pred, obj)
    )


def test_john_likes_mary_links_two_names() -> None:
    """The demo graph joins two distinct name nodes with likes."""
    graph = john_likes_mary()
    assert isinstance(graph, RDFGraph)
    triples = list(graph.to_rdflib())
    name = _iri("name")
    likes = _iri("likes")
    name_subjects = {subj for subj, pred, _obj in triples if pred == name}
    like_triples = [(subj, obj) for subj, pred, obj in triples if pred == likes]
    assert len(like_triples) == 1
    subject, object_ = like_triples[0]
    assert subject != object_
    assert {subject, object_} == name_subjects


def test_missing_root_and_lambda_argument_raise() -> None:
    """Application requires a ground graph that mentions the root."""
    lam = Formula.create(
        "G1^rdf{ <urn:dylan:var:G1> <http://example.org/p> <http://example.org/o> . }",
    )
    other = Formula.create(
        "G2^rdf{ <urn:dylan:var:G2> <http://example.org/p> <http://example.org/o> . }",
    )
    bare = Formula.create(
        "rdf{ <http://example.org/a> <http://example.org/p> <http://example.org/b> . }",
    )
    assert isinstance(lam, RDFLambdaAbstract)
    assert isinstance(other, RDFLambdaAbstract)
    assert isinstance(bare, RDFGraph)
    with pytest.raises(ValueError):
        lam.beta_reduce(bare)
    with pytest.raises(TypeError):
        lam.beta_reduce(other)
    with pytest.raises(ValueError):
        Formula.create("rdf{ not turtle }")
