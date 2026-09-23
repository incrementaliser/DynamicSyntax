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

## LaTeX and Manim export

`ParseResult.to_latex` and `ParseResult.to_manim` render a DS-TTR parse. Formulae are walked as objects (`dylan.formula.latex.formula_tex`), not pasted from `str(...)`.

- `to_latex("semantics")` — two-column record: manifest values are subscripts (`e_{1}{}_{=\mathit{arrive}}`), types use `e_{s}` and `\rightarrow`.
- `to_latex("tree")` — one `rtrees` tree. Node cells are type/requirement decorations, `\ptr` on the pointer, and the formula. Node addresses are not printed.
- `to_latex("incremental")` — word snapshots stacked vertically (a horizontal row overflows once records are real). Each arrow caption is the actions for that word, then the word in quotes.
- Compilation is `latexmk -pdfps` (latex → dvips → ps2pdf). `pdflatex` does not draw the PSTricks trees. `dsttr.sty` loads `ecltree` only when that file exists.
- `to_manim` — each surface word is written once (`show_word` / `token_index`). The scene measures node cards and scales the tree into the left pane; actions sit in the right column and scroll upward if they would leave the frame. Final semantics replace the tree instead of covering it. Node text is the Unicode display form (`≔`, `→`, `eₛ`), not the address-order GUI dump.

