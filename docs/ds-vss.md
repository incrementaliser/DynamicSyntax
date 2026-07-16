# DS-VSS: Dynamic Syntax with Vector Space Semantics

This package now ships an implementation of **DS-VSS** — the fusion of
Dynamic Syntax with compositional distributional (vector-space) semantics
introduced in:

> Sadrzadeh, M., Purver, M., Hough, J. & Kempson, R. (2018).
> *Exploring Semantic Incrementality with Dynamic Syntax and Vector Space
> Semantics.* [arXiv:1811.00614](https://arxiv.org/abs/1811.00614).
> Extended version: *Incremental Composition in Distributional Semantics*,
> JoLLI 30 (2021).

## Idea

Dynamic Syntax defines grammaticality as incremental *semantic tree growth*,
and is agnostic about the language its node decorations are drawn from.
DS-VSS instantiates that language with tensors:

| DS notion                        | DS-VSS counterpart                                   |
| -------------------------------- | ---------------------------------------------------- |
| `Ty(e)` (entities)               | vectors in a word space `W`                          |
| `Ty(t)` (propositions)           | vectors in a sentence space `S`                      |
| `Ty(⟨e,t⟩)` (one-place pred.)    | matrices in `W ⊗ S`                                  |
| `Ty(⟨e,⟨e,t⟩⟩)` (two-place pred.)| cubes in `W ⊗ S ⊗ W` — `(subject, sentence, object)` |
| composition `O` (application)    | **tensor contraction**                               |
| LINK adjunction                  | Frobenius `mu` map (pointwise product)               |
| requirements `?Ty(X)`            | unit tensor, sum `T+`, or direct sum `T⊕`            |

Because requirements carry real tensor content, even *partial* trees compile
to a sentence-space vector at the root — giving word-by-word **incremental
plausibility**, **incremental disambiguation**, and **expectation**
(plausibility of continuations), exactly as in sections 4–5 of the paper.

## Quick start

```python
from dynamicsyntax.vss import VSSLexicon, VectorSpace, parse_vss

lex = VSSLexicon(word_space=VectorSpace("W", 4, ("people", "affection", "motion", "action")))
lex.add_entity("john", [5, 4, 1, 1])
lex.add_entity("mary", [5, 4, 1, 1])
lex.add_intransitive("arrive", [[6, 1], [1, 2], [8, 1], [2, 3]])

r = parse_vss("john arrives", "ttr", lexicon=lex)
for word, p in r.trajectory():
    print(word, p)          # word-by-word plausibility
print(r.final_plausibility)
```

The sentence space defaults to the paper's two-dimensional *plausibility*
space with basis `(⊤, ⊥)`; `plausibility(v)` is the normalised `⊤` share.

## Learning the lexicon from data

```python
lex = VSSLexicon.from_cooccurrence(
    targets=["baby", "footballer"],
    contexts=["infant", "nappy", "pitch", "goal"],
    counts=[[34, 10, 0, 0], [0, 0, 24, 17]],
    weighting="ppmi",          # or "count", "pmi"
)
# verb plausibility matrices from (context words, verb) pairs:
lex.learn_plausibility_verbs([({"infant", "nappy"}, "vomit"), ({"pitch"}, "score")])
```

Plausibility is approximated from co-occurrence of verb and entity in the
same context, implausibility from occurrence of the verb without the entity
(section 4 of the paper).

## Expectation: ranking continuations

```python
from dylan.vss import verb_continuations, object_continuations

verb_continuations(lex.lookup("baby", (lex.word_space,)), ["vomit", "score"], lex)
object_continuations(lex.lookup("footballer", (lex.word_space,)), "control", ["ball", "milk"], lex)
```

## Requirement modes

`parse_vss(..., requirement_mode="sum" | "unit" | "direct_sum")` selects the
interpretation of `?Ty(X)` nodes (paper, section 3): the neutral unit
tensor, the sum `T+` of all lexicon tensors of the space (an "average"
expectation — the paper's working example), or the direct sum `T⊕` keeping
alternatives separate.

## Tests

`tests/vss/` replicates the paper's worked arithmetic (`babies vomit =
430⊤ + 98⊥`), its plausibility orderings, disambiguation and expectation
examples, and checks the decoration of real DyLan parse trees
(`john likes mary` with the bundled TTR grammar).
