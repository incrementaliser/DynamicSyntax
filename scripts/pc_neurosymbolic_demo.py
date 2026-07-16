"""Demo: neuro-symbolic learning with probabilistic circuits + DS-VSS.

Trains an Einsum Network (Peharz, Vergari et al. 2020) on DyLan parses and
DS-VSS plausibility scores, and queries it with exact conditional
inference.  Run:  python scripts/pc_neurosymbolic_demo.py
"""

from __future__ import annotations

import numpy as np
import torch

from dynamicsyntax.pc import DSPlausibilityPC, PCWordModel
from dynamicsyntax.vss import VSSLexicon, VectorSpace


def main() -> None:
    torch.manual_seed(0)
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

    corpus = [
        "john likes mary",
        "mary likes john",
        "john likes john",
        "mary likes mary",
        "john arrives",
        "mary arrives",
    ]

    print("== PC language model (exact-conditional next-word prediction) ==")
    wm = PCWordModel(max_len=3, seed=0)
    history = wm.fit(corpus * 8, epochs=12)
    print(f"  nll {history[0]:.3f} -> {history[-1]:.3f}")
    for prefix in ["john", "john likes", "mary likes"]:
        probs = wm.next_word_probs(prefix)
        top = {k: round(v, 3) for k, v in list(probs.items())[:3]}
        print(f"  p(w | {prefix!r}): {top}")

    print("\n== DS plausibility PC (DS-VSS tuples + exact inference) ==")
    pc = DSPlausibilityPC(num_bins=4, seed=0)
    rows = pc.build_tuples(corpus, "ttr", lex)
    print("  extracted tuples:", rows)
    history = pc.fit(corpus, "ttr", lex, epochs=12)
    print(f"  nll {history[0]:.3f} -> {history[-1]:.3f}")
    print("  p(bin | john, like, mary):",
          np.round(pc.plausibility_distribution("john", "like", "mary"), 3))
    print("  expected plausibility (john, arrive):",
          round(pc.expected_plausibility("john", "arrive"), 3))
    print("  p(object | john, like):",
          {k: round(v, 3) for k, v in pc.rank_objects("john", "like").items()})
    print("  p(verb | john):",
          {k: round(v, 3) for k, v in pc.rank_verbs("john").items()})


if __name__ == "__main__":
    main()
