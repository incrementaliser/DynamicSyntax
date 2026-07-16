# 2013 CHILDES Filtered-path parity — findings

Restored the pre-BabyDS **Filtered** induction path under the `childes` profile (BabyDS stays on **Maximal**), then chased remaining coverage gaps (split replay, label disjunction, export).

## Seed ruling

| Seed | Role |
|------|------|
| `resources/2023-english-ttr-induction-seed` | **Use for CHILDES** — matches Java 2023; stripped actions ≈ Java `2013-english-ttr-induction-seed` |
| `resources/2013-english-ttr-induction-seed/NOTE.txt` | Alias note only (points at 2023 equivalence) |
| `resources/2025-seed-grammar` | **BabyDS only** — do not use for 2013 CHILDES parity |

[`docs/Induction-extension-guide.md`](Induction-extension-guide.md) mentions 2025 as the latest *BabyDS* reminder, not the CHILDES seed.

## Target vs measured (400 holdout)

| Source | Top-1 Cov | Top-2 Cov | Top-3 Cov |
|--------|-----------|-----------|-----------|
| 2013 (`z-myfiles/2013-results.txt`) | **59** | **85** | **92** |
| Prior Maximal (`20260713-162102`) | 15.0 | 30.0 | 30.0 |
| Filtered early-return (`20260713-204618`) | 13.9 | 24.7 | 26.4 |
| **After LabelDisjunction + export fixes (`20260714-234500`)** | **21.1** | **47.5** | **53.1** |
| Same run, test Cov | **15.0** | **25.0** | **30.0** |

F1 on successful parses stays high (train ~88–92%, test ~87–94%). Gap to 2013 is still coverage, but Top-2/3 roughly **doubled** vs the Filtered-only pass.

Training signal on `20260714-234500`:

- `NO SEQUENCES`: **83** (was 103)
- `failed to split`: **0** (was 189)

3200/800 LC eval was set up (`configs/induction/childes-3200-800.yaml` + copied corpora) but **not run** (stopped per request).

## What closed most of the remaining gap

1. **`LabelDisjunction`** ([`labels.py`](../src/dylan/tree/label/labels.py)): bracketed `||` IF specs (e.g. `hyp-adj-t-generic`) were parsed as `GenericLabel` (always false). Hypothesise applied `hyp-adj-t` (simple `ty(t)`) then stored `hyp-adj-t-generic` on the edge → split replay returned null. Port of Java `LabelDisjunction` fixes split.

2. **`substitute` first-wins on label collision** ([`ttr_record_type.py`](../src/dylan/formula/ttr_record_type.py)): renaming `e1`→`head` no longer emits duplicate `head` fields (`head==do` + `head==head`).

3. **`LexicalAction.from_action_spine`** ([`lexical_action.py`](../src/dylan/action/lexical_action.py)): keep Effect objects like Java `flatten` instead of re-parsing THEN lines (unblocks lexicon export after aux learning).

4. **TypeLattice `ind_obj` restored** on childes profile; **`TTR2TreeCorpusConverter`** uses `get_induction_abstractions`.

## Earlier Filtered-path work (still in place)

- Hypothesiser: childes → Filtered + `R2^R1` swap; babyds → Maximal
- Classic t/cn templates + early-return Filtered
- TreeFilter `arguments` fix + filter-aware early-return for SVO

## Pre-BabyDS peel / cn nest (D18)

Under `childes`, field peel and cn nesting now match Java **before** `62df057d` (not BabyDS tip):

- `TTRRecordType.get_abstractions_basic` → `_get_abstractions_basic_pre_babyds` (two branches; always `f.label → v.head`)
- `TTRFormula._abstractions_from_types` → nest cn only when `len(argument_abstracts) == 1` (no premature multi-cn trees)

`babyds` keeps tip peels + premature path. Re-measure holdout after this change if chasing 59/85/92 further.

## Remaining gap vs 59/85/92

- Holdout 360/40 ≠ original 2013 partition (optional LC 3200/800 config exists if needed later)
- ~83 train examples still yield no sequences (PP/`iota`, modals, negation-heavy)
- Do **not** loosen two-way subsumption; do **not** switch to 2025 seed

## Run artifacts

- Holdout (this pass): `out/runs/20260714-234500_induction-childes`
- Config: `configs/induction/childes-holdout.yaml`
- Optional LC protocol (unused): `configs/induction/childes-3200-800.yaml`
