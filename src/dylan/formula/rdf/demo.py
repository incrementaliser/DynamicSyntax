"""Terminal demo: the RDF graph for "John likes Mary"."""

from __future__ import annotations

from dylan.formula.formula import Formula
from dylan.formula.rdf.graph import RDFGraph
from dylan.formula.rdf.lambda_abstract import RDFLambdaAbstract
from dylan.logging_config import configure_logging

_VERB = (
    "G1^G2^rdf{ <urn:dylan:var:G1> <http://example.org/likes> <urn:dylan:var:G2> . }"
)
_JOHN = 'rdf{ <urn:dylan:root> <http://example.org/name> "John" . }'
_MARY = 'rdf{ <urn:dylan:root> <http://example.org/name> "Mary" . }'


def john_likes_mary() -> RDFGraph:
    """Apply the verb lambda to the John and Mary graphs."""
    verb = Formula.create(_VERB)
    john = Formula.create(_JOHN)
    mary = Formula.create(_MARY)
    if not isinstance(verb, RDFLambdaAbstract):
        raise TypeError(f"verb formula must be an RDF lambda, got {type(verb).__name__}")
    once = verb.beta_reduce(john)
    if not isinstance(once, RDFLambdaAbstract):
        raise TypeError(f"one application must leave an RDF lambda, got {type(once).__name__}")
    result = once.beta_reduce(mary)
    if not isinstance(result, RDFGraph):
        raise TypeError(f"John likes Mary must be an RDF graph, got {type(result).__name__}")
    return result


def main() -> None:
    """Print the Turtle graph for "John likes Mary"."""
    configure_logging("WARNING")
    raw = john_likes_mary().to_rdflib().serialize(format="turtle")
    text = raw if isinstance(raw, str) else bytes(raw).decode("utf-8")
    print(text, end="" if text.endswith("\n") else "\n")


if __name__ == "__main__":
    main()
