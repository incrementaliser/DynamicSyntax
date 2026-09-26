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

**DS Tree canvas**:
The Output pane's drawing of the current DS Tree. Layout is Reingold–Tilford (Buchheim) in `src/dylan/gui/tree_viz.py`: nodes in address order, parents centred on their children. Each link is one straight segment from the parent's bottom centre to the child's top centre, stroked with a Flet canvas path (`MoveTo`/`LineTo`, butt caps). Link, context, and unfixed steps stay dashed or dotted. If a straight segment would cross another node, that link is an orthogonal polyline in the gap between rows. Row spacing grows with sibling spread (capped) so branches are not flat gutters. Full canvas labels also break a formula record's ``|`` fields two per line; when the formula binds more than one R, that binder stays on its own line so a long predicate node is taller and narrower. The Pyodide page in `web/` still shows the address-order text dump, not this canvas.

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
