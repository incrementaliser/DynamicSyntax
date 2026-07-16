# Probabilistic circuits for neuro-symbolic learning

This package ships a self-contained implementation of **Einsum Networks**
and two neuro-symbolic models built on them, in the probabilistic-circuits
framework of Antonio Vergari and colleagues:

> Peharz, R., Lang, S., **Vergari, A.**, Stelzner, K., Molina, A., Trapp,
> M., Van den Broeck, G., Kersting, K. & Ghahramani, Z. (2020).
> *Einsum Networks: Fast and Scalable Learning of Tractable Probabilistic
> Circuits.* ICML 2020.

> Choi, Y., **Vergari, A.** & Van den Broeck, G. (2020).
> *Probabilistic Circuits: A Unifying Framework for Tractable Probabilistic
> Models.* Tutorial.

Requires PyTorch: `pip install dynamicsyntax[pc]`.

## Why probabilistic circuits here?

Dynamic Syntax + DS-VSS give *symbolic, compositional, incremental*
semantics; probabilistic circuits add a *learned, calibrated, tractable*
joint distribution over the same structures.  Because EiNets are smooth and
decomposable, **marginals and conditionals are exact** and cost a single
feed-forward pass — so prediction during incremental parsing (the DS-VSS
"expectation" of continuations) becomes exact conditional inference in a
circuit learned from data.

## The EiNet implementation (`dylan.pc.einet`)

- random binary region graphs (`dylan.pc.regions`): regions at every layer
  partition the variable scope ⇒ decomposable products and smooth sums;
- monolithic product+sum layers with the log-einsum-exp trick, categorical
  input distributions (per-variable category counts supported);
- training by the paper's **EM via automatic differentiation** (stochastic
  online EM, Sato 1999) or by **SGD** (Adam on NLL);
- exact conditionals (`conditional_log_probs`), MAP prediction, and
  save/load.

Sanity properties covered by tests: the partition function is exactly 1
(even after training), and conditionals agree with brute-force enumeration.

```python
import torch
from dylan.pc.einet import EinsumNetwork, EiNetConfig, UNKNOWN

net = EinsumNetwork(EiNetConfig(num_vars=3, num_categories=3))
net.fit_em(torch.tensor([[0, 0, 0], [1, 1, 1]] * 20), epochs=8)
net.conditional_log_probs(torch.tensor([0, 0, UNKNOWN]), var=2).exp()
```

## Model 1 — PC language model (`PCWordModel`)

Each sequence position is a categorical variable; next-word prediction is
an exact conditional (later positions are marginalised):

```python
from dynamicsyntax.pc import PCWordModel

wm = PCWordModel(max_len=3)
wm.fit(["john likes mary", "mary likes john", "john arrives"] * 10, epochs=10)
wm.next_word_probs("john likes")   # {'mary': …, 'john': …, '<pad>': …}
```

## Model 2 — DS plausibility PC (`DSPlausibilityPC`)

The neuro-symbolic loop: DyLan parses a corpus, DS-VSS scores incremental
plausibility, `(subject, verb, object, plausibility-bin)` tuples train the
circuit, and plausibility/expectation queries become exact conditionals:

```python
from dynamicsyntax.pc import DSPlausibilityPC
from dynamicsyntax.vss import VSSLexicon, VectorSpace

lex = VSSLexicon(word_space=VectorSpace("W", 4, ("people", "affection", "motion", "action")))
# … register entity vectors and verb tensors …

pc = DSPlausibilityPC(num_bins=4)
pc.fit(["john likes mary", "john arrives"], "ttr", lex, epochs=8)
pc.plausibility_distribution("john", "like", "mary")  # posterior over bins
pc.expected_plausibility("john", "arrive")
pc.rank_objects("john", "like")   # p(object | subject, verb)
pc.rank_verbs("john")             # p(verb | subject)
```

This is the probabilistic counterpart of the DS-VSS expectation model
(Sadrzadeh et al. 2018, sec. 5.2): instead of raw tensor contraction, the
plausibility of a continuation is a learned, normalised posterior.

## Training notes

- EM (the paper's algorithm) is the default and works well at moderate
  widths; full-batch EM can plateau on very wide circuits with small data —
  use `method="sgd"` there (both are exact-likelihood methods).
- `init_scale` (default 1.0) keeps deep circuits away from the uniform
  initialisation that stalls learning.
