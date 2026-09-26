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

## Data and CHILDES

The parser targets **English** (bundled `2015-english-ttr` and the induction seeds). BabyDS (`data/BabyDS/`) is an English robot-command corpus already in TTR. It is not CHILDES.

**Raw CHILDES** (English–North American MOR zip) lives only at `data/raw/childes/Eng-NA-MOR.zip`. That path is gitignored. URL, date, size, and CC BY-NC-SA 3.0 terms are in `data/CHILDES/PROVENANCE.txt`. Brown Eve in that zip is 20 `.cha` files and 26920 utterances. Adam is 55 files and Sarah is 139. Those transcripts are CHAT (`*MOT` / `%mor`), not lambda formulae.

**childes-db 2026.1** (Redivis v1.4, 2026-07-25, DOI 10.57761/9yv6-c595) is the versioned chatter parse of the July 2026 TalkBank release: 24 collections, 437 corpora, 56579 transcripts, 9151 children, 24.2 million utterances, 89 million tokens, UD morphology, `%gra` dependencies, speech acts. Site: https://langcog.github.io/childes-db-website/data.html. It is not in this repo. Browser export needs no code; the R accessor is `childesr` on branch `redivis`; scripted download needs `REDIVIS_API_TOKEN` (unauthenticated REST returned 401). Utterance rows are gloss and stem, not the Eve lambda `Sem` tier, so `convert_lambda` does not apply. Pin version `2026.1` and cite Sanchez et al. (2019) *Behavior Research Methods* plus TalkBank.

**Eve lambda annotation** is `data/CHILDES/eve/lambda/trainPairs_1` … `trainPairs_20`, copied from DyLan `corpus/CHILDES/eveTrainPairs/`. 4645 `example_end` blocks, of which 28 are commented (`//example_end`). This is a semantically annotated subset of Brown Eve, not the 26920 CHAT lines.

**Lambda → TTR entry points**

- `convert_lambda(semantics, utterance)` in `src/dylan/induction/em_learner/lambda_ttr_converter.py` — Eve lambda to a TTR record type.
- `normalize_szubert_lambda(semantics)` in `src/dylan/induction/em_learner/szubert_lambda.py` — published Adam lambda (`_{r}`, `mod|`, `eat_4`, `BARE`, embedded `lambda`) into that Eve dialect. `convert_szubert_lambda` calls both.
- `scripts/convert_childes_lambda.py FOLDER OUTPUT --failures PATH --repairs PATH` reads `trainPairs_*` and writes `Sent` / `Sem` blocks with a `// trainPairs_N sample` comment.
- `scripts/convert_szubert_lambda.py` writes Adam `Sent` / `Sem` / `TTR` blocks plus `coverage.txt`.
- `CorpusConverter` and `RMRS_TTR_converter` do not implement this rewrite. `CorpusConverter.convert` only reloads an existing TTR corpus file. `RMRS_TTR_converter` is a stub in both the Python port and the Java source.

**Eve TTR coverage** (`data/CHILDES/eve/ttr/eve-ttr.txt`, failures in `eve-failures.txt`, repairs in `eve-repairs.txt`):

| | utterances |
|---|---|
| Brown Eve CHAT | 26920 |
| Lambda blocks | 4645 (28 commented) |
| Converted to TTR | 4617 |
| Rejected | 0 |
| Repaired before conversion | 12 |

`main` had no Eve files. The Java `CHILDESconvert` method stops after 400 successes; that partial file is on branch `CHILDES-TEST`, not on `main`. The Python converter now also covers object control with different subjects (`you want me to have it`), `whose icecream` as an equative question, truncated `not($0,)` (11, tag `truncated-not`), and one unbalanced `not(and(pro|me,,$0)` (tag `unbalanced`). Each `Sent` and `Sem` line ends with `// trainPairs_N sample`. Do not treat CHAT `%mor` lines as converter input: they are a different format. Do not emit the placeholder `[x : e|head==x : e]` for a failed formula.

**Adam logical forms → TTR**

Szubert et al. (arXiv 2109.10952, repo `Lou1sM/CHILDES_UD2LF_2` at `85004c98`) already turn manual Adam UD trees into lambda. Mahon, Johnson, and Steedman (2025, arXiv 2503.12832) train a CCG learner on that lambda; this repo does not port the learner. The TTR path is their released logical form, normalized into the Eve dialect, then `convert_lambda`.

- `normalize_szubert_lambda` / `convert_szubert_lambda` in `src/dylan/induction/em_learner/szubert_lambda.py`
- `scripts/convert_szubert_lambda.py` reads `adam.all_lf.txt`, the `adamN.conll.txt` split, and `Adam.all_easy_lf.txt` from `ida-szubert/ccg_acquisition_2`
- Output: `data/CHILDES/adam/ttr/adam-ttr.txt` (`Sent` / `Sem` / `TTR`), `adam-failures.txt`, `coverage.txt`
- Raw downloads stay in gitignored `data/raw/childes/szubert/`

Each comment is `adamN.conll.txt INDEX comparison=paper|extra`, plus a Brown `.cha` path and adult-line index when that string matches exactly one adult line in the Eng-NA zip. `comparison=paper` means the utterance is in `Adam.all_easy_lf.txt` (5734 distinct sentences). The papers’ 5320-utterance filter is not in the public files; that easy file is the comparison flag. Of 13593 released Adam LFs, 7859 convert (4629 of the 7612 rows whose text is in the easy file). Hebrew Hagar and childes-db are not converted here. A higher-order adjunct `$n($m)` with no lexical preposition is dropped before conversion.
