"""Ground RDF graph formulae (Java ``RDFGraph``), stored as RDFLib graphs."""

from __future__ import annotations

from loguru import logger
from rdflib import BNode, Graph, URIRef
from rdflib.compare import to_isomorphic
from rdflib.term import Node

from dylan.formula.formula import Formula
from dylan.formula.rdf.formula import RDFFormula
from dylan.formula.rdf.syntax import RDFSyntaxError
from dylan.formula.variable import Variable

# Fixed base so relative IRIs in lexicon Turtle do not depend on the process cwd.
PUBLIC_ID = "urn:dylan:"
ROOT = URIRef("urn:dylan:root")
_VAR_PREFIX = "urn:dylan:var:"


def placeholder_iri(name: str) -> URIRef:
    """Return the IRI that stands for the RDF-lambda binder *name*."""
    return URIRef(f"{_VAR_PREFIX}{name}")


def copy_graph(source: Graph) -> Graph:
    """Return a new graph with the same triples and prefix bindings as *source*."""
    out = Graph()
    for prefix, namespace in source.namespaces():
        out.bind(prefix, namespace, override=True)
    for triple in source:
        out.add(triple)
    return out


def replace_term(source: Graph, old: Node, new: Node) -> Graph:
    """Return a copy of *source* with every occurrence of *old* replaced by *new*."""
    out = Graph()
    for prefix, namespace in source.namespaces():
        out.bind(prefix, namespace, override=True)
    for subj, pred, obj in source:
        out.add(
            (
                new if subj == old else subj,
                new if pred == old else pred,
                new if obj == old else obj,
            ),
        )
    return out


def graph_mentions(source: Graph, term: Node) -> bool:
    """Return whether *term* occurs as a subject, predicate, or object."""
    for subj, pred, obj in source:
        if term in (subj, pred, obj):
            return True
    return False


class RDFApplicationError(ValueError):
    """An RDF graph was applied where the argument has no root node."""


class RDFGraph(RDFFormula):
    """A ground set of RDF triples."""

    def __init__(self, graph: Graph) -> None:
        """Copy *graph* so later edits to the caller's graph do not change this formula."""
        super().__init__()
        self._graph = copy_graph(graph)
        self._digest: int | None = None

    @classmethod
    def parse(cls, turtle: str) -> RDFGraph:
        """Parse Turtle *turtle* into an RDF graph formula.

        Relative IRIs resolve against ``PUBLIC_ID``. Invalid Turtle raises
        :class:`~dylan.formula.rdf.syntax.RDFSyntaxError`.
        """
        if not turtle.strip():
            logger.debug("parsed empty RDF graph")
            return cls(Graph())
        graph = Graph()
        try:
            graph.parse(data=turtle, format="turtle", publicID=PUBLIC_ID)
        except Exception as exc:
            logger.debug("RDF Turtle parse failed: {}", exc)
            raise RDFSyntaxError(f"invalid Turtle in rdf{{ }}: {exc}") from exc
        logger.debug("parsed RDF graph with {} triples", len(graph))
        return cls(graph)

    def to_rdflib(self) -> Graph:
        """Return a copy of the underlying RDFLib graph."""
        return copy_graph(self._graph)

    def clone(self) -> RDFGraph:
        """Return a deep copy. Blank-node ids are preserved."""
        return RDFGraph(self._graph)

    def substitute(self, var: Variable, arg: Formula) -> RDFFormula:
        """Apply *arg* at binder *var*: fresh blank node, then union.

        *arg* must be an :class:`RDFGraph` that mentions :data:`ROOT`. Every
        :func:`placeholder_iri` for ``var.name`` in this graph, and every root
        in *arg*, is rewritten to one new blank node. The shared root IRI is
        therefore absent from the result, and two applications do not identify
        their roots.
        """
        if not isinstance(arg, RDFGraph):
            raise TypeError(
                f"RDF graph substitute expects an RDF graph, got {type(arg).__name__}",
            )
        if not graph_mentions(arg._graph, ROOT):
            raise RDFApplicationError("RDF argument graph has no root (urn:dylan:root)")
        fresh = BNode()
        body = replace_term(self._graph, placeholder_iri(var.get_name()), fresh)
        argument = replace_term(arg._graph, ROOT, fresh)
        return RDFGraph(body + argument)

    def merge(self, other: RDFFormula) -> RDFFormula:
        """Union *other* into this graph. A lambda argument keeps that lambda's spine."""
        from dylan.formula.rdf.lambda_abstract import RDFLambdaAbstract

        if isinstance(other, RDFLambdaAbstract):
            return other.merge(self)
        if not isinstance(other, RDFGraph):
            raise TypeError(f"Cannot merge RDF graph with {type(other).__name__}")
        return RDFGraph(self._graph + other._graph)

    def get_variables(self) -> set[Variable]:
        """Return binders whose placeholder IRI occurs in this graph."""
        found: set[Variable] = set()
        for subj, pred, obj in self._graph:
            for term in (subj, pred, obj):
                if isinstance(term, URIRef) and str(term).startswith(_VAR_PREFIX):
                    found.add(Variable(str(term)[len(_VAR_PREFIX) :]))
        return found

    def _canonical_digest(self) -> int:
        """Return the RDFLib isomorphism digest, cached on this instance."""
        if self._digest is None:
            self._digest = int(to_isomorphic(self._graph).internal_hash())
        return self._digest

    def __eq__(self, other: object) -> bool:
        """True when *other* is an isomorphic RDF graph. Prefixes are ignored."""
        if not isinstance(other, RDFGraph):
            return NotImplemented
        return self._canonical_digest() == other._canonical_digest()

    def __hash__(self) -> int:
        """Hash the isomorphism digest so equal graphs share a hash."""
        return hash(self._canonical_digest())

    def __str__(self) -> str:
        """Render ``rdf{ Turtle }`` using this graph's namespace bindings."""
        raw = self._graph.serialize(format="turtle")
        text = raw if isinstance(raw, str) else bytes(raw).decode("utf-8")
        return "rdf{" + text.strip() + "}"

    def __repr__(self) -> str:
        """Same spelling as :meth:`__str__`."""
        return str(self)
