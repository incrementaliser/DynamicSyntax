# DyLan / Dynamic Syntax

Python port of DyLan, an incremental Dynamic Syntax (DS) parser with Type Theory with Records (TTR). The package parses utterances into decorated trees and a word-level derivation DAG, and can induce a DS-TTR lexicon from examples.

## Structure

- `src/dylan/` — parser, tree, DAG, actions, TTR, induction, Flet GUI, Pyodide façade
- `src/dynamicsyntax/` — public package entry (`icp`, bundled grammars)
- `web/` — static Pyodide UI that calls `dylan.pyodide_api` (same `ParseSession` as the Flet app)
- `tests/` — pytest suite

## Language

**DS Tree**:
A decorated Dynamic Syntax tree: node addresses, type and formula labels, and a distinguished pointer.
_Avoid_: parse tree, constituency tree

**DAG tuple**:
One interpretation node in the word-level context DAG; the current tuple is the parser’s live derivation state.
_Avoid_: parse, hypothesis (when you mean this DAG node)

**Viewport**:
The Output pane that shows the DS Tree canvas, not the OS screen or device.
_Avoid_: device, window (when you mean the tree pane)

**Camera**:
The Output pane's scroll position over the DS Tree at the current Zoom. A tree that fits the pane is centred; Fit scrolls a larger drawing so its middle is in view.
_Avoid_: scale-to-fit layout, stretch

**Zoom**:
A uniform scale of the natural-size DS Tree. Boxes and lettering grow or shrink together. 100% is intrinsic size.
_Avoid_: stretch-to-fill, reflow, fit

**Fit-to-pane**:
Scrolling the DS Tree at the current Zoom so its middle sits in the Viewport. It does not change Zoom.
_Avoid_: zoom, shrink-to-fit

**Interpretation**:
One DAG tuple in the sequence reached by stepping the parser after a sentence, shown as an index in 1 / N.
_Avoid_: step, hypothesis, reading

**Info**:
The how-to opened from the ? button: how to load a grammar, parse, move between interpretations, zoom, fit, and when to Reset.
_Avoid_: Status, Logs, loguru log

**Status**:
The live session card: grammar path, repair, last event, load warnings, pointer, and current DAG tuple.
_Avoid_: Info, Logs

**Logs**:
The append-only event history (grammar-load warnings, parse/step lines, computational actions).
_Avoid_: loguru log, Info, Status

**Reset**:
Returning the derivation to the empty axiom while the grammar stays loaded.
_Avoid_: Init

**Continuation**:
A sentence whose words start with the words already in the derivation.
_Avoid_: prefix

### RDF

**RDF formula**:
A semantic formula whose content is RDF: either an RDF graph or an RDF lambda.
_Avoid_: Jena model, triple store

**RDF graph**:
A ground RDF formula: a set of triples.
_Avoid_: record type, model

**RDF lambda**:
An RDF formula that binds ``G``, ``G1``, … over an RDF formula.
_Avoid_: FOL lambda, TTR lambda

**Root**:
The distinguished node of an RDF graph that is plugged into a placeholder when that graph is applied as an argument.
_Avoid_: head, TTR head

**Placeholder**:
The node in an RDF graph that stands for one bound RDF-lambda variable.
_Avoid_: blank node, SPARQL variable

