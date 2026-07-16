"""Demo: DS-VSS incremental plausibility on the paper's worked example.

Reproduces the ``baby``/``footballer`` plausibility model of Sadrzadeh et
al. (2018), section 4, and shows DS-VSS decoration of DyLan parse trees
with the bundled TTR grammar.

Run:  python scripts/dsvss_plausibility_demo.py
"""

from __future__ import annotations

import numpy as np

from dylan.vss import (
    VSSLexicon,
    VectorSpace,
    contract,
    object_continuations,
    plausibility,
    plausibility_space,
    verb_continuations,
)
from dynamicsyntax.vss import parse_vss


def paper_example() -> None:
    print("== DS-VSS worked example (Sadrzadeh et al. 2018, §4–5) ==")
    W = VectorSpace("W", 4, ("infant", "nappy", "pitch", "goal"))
    lex = VSSLexicon(word_space=W, sentence_space=plausibility_space())
    lex.add_entity("babies", [34, 10, 0, 0])
    lex.add_entity("footballers", [0, 0, 24, 17])
    lex.add_entity("balls", [0, 0, 10, 10])
    lex.add_entity("milk", [0, 5, 0, 0])
    lex.add_intransitive("vomit", [[10, 2], [9, 3], [0, 30], [0, 25]])
    lex.add_intransitive("score", [[0, 8], [0, 6], [20, 1], [29, 1]])
    lex.add_intransitive("dribble", [[9, 2], [8, 2], [21, 5], [20, 4]])
    cube = np.zeros((4, 2, 4))
    for i in (2, 3):
        for k in (2, 3):
            cube[i, 0, k] = 1.0
    lex.add_transitive("control", cube)

    W_ = (lex.word_space,)
    WS = (lex.word_space, lex.sentence_space)

    def show(subj: str, verb: str) -> None:
        v = contract(lex.lookup(verb, WS), lex.lookup(subj, W_))
        print(f"  {subj:12s} {verb:8s} = {v.array!s:22s} plausibility {plausibility(v):.3f}")

    show("babies", "vomit")        # 430⊤ + 98⊥ in the paper
    show("babies", "score")
    show("footballers", "vomit")
    show("footballers", "score")
    show("babies", "dribble")
    show("footballers", "dribble")

    fb = lex.lookup("footballers", W_)
    bb = lex.lookup("babies", W_)
    print("\n  expectation — verb continuations for 'babies':",
          verb_continuations(bb, ["vomit", "score"], lex))
    print("  expectation — objects of 'footballers control':",
          object_continuations(fb, "control", ["balls", "milk"], lex))


def dylan_example() -> None:
    print("\n== DS-VSS on DyLan parses (bundled TTR grammar) ==")
    W = VectorSpace("W", 4, ("people", "affection", "motion", "action"))
    lex = VSSLexicon(word_space=W)
    lex.add_entity("john", [5, 4, 1, 1])
    lex.add_entity("mary", [5, 4, 1, 1])
    lex.add_intransitive("arrive", [[6, 1], [1, 2], [8, 1], [2, 3]])
    cube = np.zeros((4, 2, 4))
    for i in (0, 1):
        for k in (0, 1):
            cube[i, 0, k] = 8.0
    cube[:, 1, :] = 1.0
    lex.add_transitive("like", cube)

    for sentence in ["john arrives", "john likes mary", "mary likes john"]:
        r = parse_vss(sentence, "ttr", lexicon=lex)
        traj = " → ".join(
            f"{w}:{p:.3f}" if p is not None else f"{w}:—" for w, p in r.trajectory()
        )
        print(f"  {sentence:18s} ok={r.ok}  {traj}")
        print(f"    root value: {r.decorations[-1].root_value}")


if __name__ == "__main__":
    paper_example()
    dylan_example()
