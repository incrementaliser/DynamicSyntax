# DS-TTR ↔ probabilistic circuits — design note (MVP)

This note is Workflow step 1 of the `ds-ttr-probabilistic-circuit-integration`
skill: map the **real** DyLan / DS-VSS / induction APIs onto circuit concepts
before committing to compilation structure. It records decisions for the
BabyDS SPL induction MVP in `dylan.nesy`.

## 1. What the codebase actually exposes

### Parser / trees / actions

| Concept | Concrete API |
|---------|----------------|
| Partial DS tree | `dylan.tree.tree.Tree`, nodes via `NodeAddress` |
| Computational vs lexical | `ComputationalAction` vs `LexicalAction` / `LexicalHypothesis` |
| Parse DAG (runtime) | `InteractiveContextParser` + `DAGTuple` |
| Induction search DAG | `DAGInductionState` / `DAGInductionTuple` |

### TTR

| Concept | Concrete API |
|---------|----------------|
| Record type | `TTRRecordType` (`dylan.formula.ttr_record_type`) |
| Subtyping | `subsumes` / `subsumes_mapped` (not raw equality) |
| “Intersection” | `mcs` / `most_specific_common_super_type` (no method named `intersection`) |
| Fields | `TTRField`, `get_fields`, `get_labels`, `head` |

### Induction (what EM does today)

```
RecordTypeCorpus → TTRWordLearner.learn_once
  → TTRHypothesiser.load_training_example(words, gold_RT)
  → hypothesise() → list[CandidateSequence]   # legal support
  → CandidateSequence.split() → per-word spines
  → WordHypothesisBase.add_sequence_tuples + update_dists_end_of_example (local EM)
  → save_learned_lexicon → lexicon-top-N.txt
  → build_eval_parser + Evaluation.precision_recall_macro
```

**Enumerable hook for MVP support:** `TTRHypothesiser.hypothesise() → list[CandidateSequence]`.
Canonicalization aid: `CandidateSequence.get_equivalence_class()` (≡ modulo flanking computational actions).

### DS-VSS

| Concept | Concrete API |
|---------|----------------|
| Decorate a parse | `dynamicsyntax.vss.parse_vss` / `VSSDecorator` |
| Per-word step vector | `VSSParseResult.decorations[t].root_value` (`VSSValue` in `S`) |
| Underspecification | `RequirementMode.SUM` (`T+`) / `DIRECT_SUM` (`T⊕`) |
| Lexicon keys | **predicate constants** via `dylan.vss.predicates`, not surface forms |

**Decision:** VSS `T+` / `VSSDirectSum` are *tensor* underspecification, **not**
circuit sum units. Circuit sum units mix mutually exclusive *derivation*
choices. Keep them separate: VSS supplies gating features `z`; the circuit
encodes discrete hyp choice.

## 2. Region graph / variables (MVP)

- **Variables:** for a sentence of length `n`, one categorical variable `X_t`
  per word position — the identity of the word-hypothesis at that position
  (from `CandidateSequence.split()`, keyed by `WordHypothesis.hyp_id` after
  ingest into `WordHypothesisBase`).
- **Support:** only assignments that appear as a legal tuple row in
  `WordHypothesisBase.tuples` for that example (hard constraint).
- **Region graph (MVP):** chain product over word positions; mixture
  (sum unit) only over the *enumerated legal sequences*, not over the full
  Cartesian product of per-word hyps. This is equivalent to a sum-of-products
  circuit with one product term per legal sequence.
- **Shared open LoFT CNF→SDD compilation** of all DS legality is **out of
  scope for MVP**; documented as the next KC step once BabyDS proves the
  training/export loop.

## 3. Evidence

- Gold target RT enters as: sequences whose completed semantics mutually
  subsume gold, or (fallback) all enumerated legal sequences from the
  hypothesiser (already lattice-constrained toward gold).
- Sentence word string is observed by construction (variables are aligned to
  the utterance).
- Subtyping as partial field evidence (full RT field encoding) is deferred;
  v1 uses hypothesiser-filtered support as the subtype constraint.

## 4. Gating `g(z)`

- `z_t` = root sentence-space vector at trace step `t` (dim = `|S|`, default 2),
  from induction-parser traces + `RequirementMode.SUM`, after a
  **predicate-aligned** lexicon (TTR constants such as `state_holding`,
  `obj_ball`, …).
- Optional concat: surface embedding / one-hot of the word token.
- `Ω_t = softmax(W z_t + b)` scores that word’s candidate hyps (or a global
  table `p(hyp | word)` refined by a gated residual).

## 5. Training objective (SPL)

\[
p(y \mid z) = C(y; \Omega = g(z)), \qquad
\mathcal{L} = -\log p(\hat{y} \mid z)
\]

with `C` supported only on legal sequences. Marginalize over sequences that
are equivalent under `get_equivalence_class` when multiple realize the same
tree.

## 6. Relation to `dylan.pc`

`dylan.pc` (hand-rolled EiNet) remains for demos / corpus-scale LM and
plausibility PCs. **Induction uses `dylan.nesy` + cirkit** (with a torch
enumerated-support backend that implements the same SPL equation when a
full symbolic compile is unnecessary for the finite support).

## 7. Eval

Reuse `Evaluation.precision_recall_macro` (maximal mapping), coverage, and
exact match via `build_eval_parser` on emitted `lexicon-top-N.txt`, same as
EM — so BabyDS numbers are comparable to `dsttr-induction` on
`configs/induction/presplit.yaml`.
