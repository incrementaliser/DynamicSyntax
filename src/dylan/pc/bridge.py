"""Neuro-symbolic bridge: probabilistic circuits over DS / DS-VSS outputs.

Two model families connect the tractable probabilistic circuits of
:mod:`dylan.pc.einet` (Einsum Networks; Peharz, Vergari et al. 2020) with
Dynamic Syntax parsing and DS-VSS semantics:

- :class:`WordSequencePC` — a PC language model over word sequences.  Every
  position is a categorical random variable; the circuit assigns exact
  likelihoods to (partial) evidence, so **next-word prediction is an exact
  conditional** — a probabilistic analogue of the DS-VSS expectation model
  (Sadrzadeh et al. 2018, sec. 5.2), but *learned* from data.

- :class:`SemanticTuplePC` — a PC over semantic tuples ``(subject, verb,
  object, plausibility-bin)`` extracted from DyLan parses and DS-VSS
  incremental plausibility.  It learns the joint distribution of semantic
  roles and plausibility, and answers queries such as "how plausible is
  *(baby, vomit)*?" or "which objects fit *(footballer, control, ?)*?" by
  exact conditional inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from dylan.pc.einet import UNKNOWN, EinsumNetwork, EiNetConfig
from dylan.tree.node_address import NodeAddress
from dylan.tree.tree import Tree
from dylan.vss import predicates

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "dylan.pc requires PyTorch; install the optional extra with "
        "`pip install dynamicsyntax[pc]` (or `pip install torch`)."
    ) from exc

#: Category index of the end-of-sequence / padding symbol in word models.
PAD = "<pad>"
#: Category for "no object" (intransitive tuples) in semantic models.
NO_OBJ = "<none>"


# ----------------------------------------------------------------------
# extraction of semantic tuples from DS trees
# ----------------------------------------------------------------------
@dataclass
class SVOTuple:
    """(subject, verb, object) predicate constants of a DS tree."""

    subject: str | None
    verb: str | None
    obj: str | None


def extract_svo(tree: Tree) -> SVOTuple:
    """Extract the semantic tuple of a (complete or partial) DS tree.

    Uses DS tree geometry: the subject sits at address ``00``, the object at
    ``010``, and the verb at ``011`` (transitive) or ``01`` (intransitive,
    where the predicate decorates the functor node itself).
    """

    def node_formula(addr: str):
        node = tree.get(NodeAddress(addr))
        return None if node is None else node.get_formula()

    subject = predicates.extract_entity(node_formula("00"))
    obj = predicates.extract_entity(node_formula("010"))
    verb = predicates.extract_event(node_formula("011"))
    if verb is None:
        verb = predicates.extract_event(node_formula("01"))
    return SVOTuple(subject, verb, obj)


def plausibility_bin(p: float, num_bins: int) -> int:
    """Discretise a plausibility in [0, 1] into ``num_bins`` bins."""
    return min(int(p * num_bins), num_bins - 1)


# ----------------------------------------------------------------------
# word-sequence PC language model
# ----------------------------------------------------------------------
class WordSequencePC:
    """PC language model over fixed-length word sequences.

    :param max_len: sequence length (shorter sentences are padded).
    :param einet_kwargs: structural hyper-parameters for the EiNet.
    """

    def __init__(self, max_len: int, **einet_kwargs) -> None:
        self.max_len = max_len
        self.vocab: list[str] = []
        # Moderate default width: robust for EM on small corpora.
        self._einet_kwargs = {
            "num_sums": 4,
            "num_input_dists": 2,
            "num_repetitions": 2,
            **einet_kwargs,
        }
        self.net: EinsumNetwork | None = None

    def _encode(self, tokens: Sequence[str]) -> list[int]:
        ids = [self.vocab.index(t) + 1 for t in tokens]
        if len(ids) > self.max_len:
            raise ValueError(f"sequence longer than max_len={self.max_len}")
        return ids + [0] * (self.max_len - len(ids))  # 0 = PAD

    def fit(
        self,
        sentences: Iterable[Sequence[str]],
        *,
        method: str = "em",
        epochs: int = 10,
        **fit_kwargs,
    ) -> list[float]:
        """Build the vocab and train the circuit on *sentences* (token lists)."""
        sentences = [list(s) for s in sentences]
        vocab = sorted({t for s in sentences for t in s})
        self.vocab = vocab
        num_categories = len(vocab) + 1  # + PAD at index 0
        self.net = EinsumNetwork(
            EiNetConfig(
                num_vars=self.max_len,
                num_categories=num_categories,
                **self._einet_kwargs,
            )
        )
        data = torch.tensor([self._encode(s) for s in sentences], dtype=torch.long)
        if method == "em":
            return self.net.fit_em(data, epochs=epochs, **fit_kwargs)
        return self.net.fit_sgd(data, epochs=epochs, **fit_kwargs)

    def _check_net(self) -> EinsumNetwork:
        if self.net is None:
            raise RuntimeError("WordSequencePC is not fitted yet")
        return self.net

    def log_likelihood(self, tokens: Sequence[str]) -> float:
        """Exact log-likelihood of a complete token sequence."""
        net = self._check_net()
        x = torch.tensor([self._encode(tokens)], dtype=torch.long)
        return float(net(x)[0])

    def next_word_probs(self, prefix: Sequence[str]) -> dict[str, float]:
        """Exact ``p(w | prefix)`` over the vocabulary (and PAD).

        Evidence: observed prefix; the queried position varies over
        candidates; all later positions are marginalised — a single exact
        conditional computation per candidate.
        """
        net = self._check_net()
        if len(prefix) >= self.max_len:
            raise ValueError("prefix already reaches max_len")
        position = len(prefix)
        evidence = [UNKNOWN] * self.max_len
        for i, tok in enumerate(prefix):
            evidence[i] = self.vocab.index(tok) + 1
        values = torch.arange(len(self.vocab) + 1)  # include PAD
        log_p = net.conditional_log_probs(
            torch.tensor(evidence, dtype=torch.long), position, values
        )
        probs = log_p.exp().tolist()
        out = {PAD: probs[0]}
        out.update({w: probs[i + 1] for i, w in enumerate(self.vocab)})
        return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


# ----------------------------------------------------------------------
# semantic-tuple PC
# ----------------------------------------------------------------------
class SemanticTuplePC:
    """PC over ``(subject, verb, object, plausibility-bin)`` tuples.

    Learns the joint distribution of semantic roles and DS-VSS
    plausibility; all queries are answered by exact conditional inference.
    """

    NUM_BINS_DEFAULT = 5

    def __init__(self, num_bins: int = NUM_BINS_DEFAULT, **einet_kwargs) -> None:
        self.num_bins = num_bins
        self._einet_kwargs = {
            "num_sums": 4,
            "num_input_dists": 2,
            "num_repetitions": 2,
            **einet_kwargs,
        }
        self.subjects: list[str] = []
        self.verbs: list[str] = []
        self.objects: list[str] = []
        self.net: EinsumNetwork | None = None

    def fit(
        self,
        tuples: Iterable[tuple[str, str, str | None, int]],
        *,
        method: str = "em",
        epochs: int = 10,
        **fit_kwargs,
    ) -> list[float]:
        """Train on ``(subject, verb, object|None, plausibility_bin)`` rows."""
        rows = [(s, v, o if o is not None else NO_OBJ, b) for s, v, o, b in tuples]
        self.subjects = sorted({s for s, _, _, _ in rows})
        self.verbs = sorted({v for _, v, _, _ in rows})
        self.objects = sorted({o for _, _, o, _ in rows} | {NO_OBJ})
        categories = (len(self.subjects), len(self.verbs), len(self.objects), self.num_bins)
        self.net = EinsumNetwork(
            EiNetConfig(num_vars=4, num_categories=categories, **self._einet_kwargs)
        )
        data = torch.tensor(
            [
                (
                    self.subjects.index(s),
                    self.verbs.index(v),
                    self.objects.index(o),
                    b,
                )
                for s, v, o, b in rows
            ],
            dtype=torch.long,
        )
        if method == "em":
            return self.net.fit_em(data, epochs=epochs, **fit_kwargs)
        return self.net.fit_sgd(data, epochs=epochs, **fit_kwargs)

    def _check(self) -> EinsumNetwork:
        if self.net is None:
            raise RuntimeError("SemanticTuplePC is not fitted yet")
        return self.net

    def _evidence(
        self,
        subject: str | None = None,
        verb: str | None = None,
        obj: str | None = None,
        bin_: int | None = None,
    ) -> list[int]:
        """Encode a partial tuple as an evidence vector (UNKNOWN = marginalised)."""
        return [
            UNKNOWN if subject is None else self.subjects.index(subject),
            UNKNOWN if verb is None else self.verbs.index(verb),
            UNKNOWN if obj is None else self.objects.index(obj if obj is not None else NO_OBJ),
            UNKNOWN if bin_ is None else bin_,
        ]

    def plausibility_distribution(
        self, subject: str, verb: str, obj: str | None = None
    ) -> list[float]:
        """Exact ``p(bin | subject, verb, object)`` over plausibility bins."""
        net = self._check()
        evidence = torch.tensor(self._evidence(subject, verb, obj), dtype=torch.long)
        log_p = net.conditional_log_probs(evidence, 3)
        return log_p.exp().tolist()

    def expected_plausibility(self, subject: str, verb: str, obj: str | None = None) -> float:
        """Expected plausibility (bin centres weighted by the posterior)."""
        dist = self.plausibility_distribution(subject, verb, obj)
        centres = [(i + 0.5) / self.num_bins for i in range(self.num_bins)]
        return sum(p * c for p, c in zip(dist, centres))

    def object_probs(self, subject: str, verb: str) -> dict[str, float]:
        """Exact ``p(object | subject, verb)`` over known objects."""
        net = self._check()
        evidence = torch.tensor(self._evidence(subject, verb), dtype=torch.long)
        values = torch.arange(len(self.objects))
        log_p = net.conditional_log_probs(evidence, 2, values)
        probs = log_p.exp().tolist()
        out = {o: probs[i] for i, o in enumerate(self.objects) if o != NO_OBJ}
        return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))

    def verb_probs(self, subject: str) -> dict[str, float]:
        """Exact ``p(verb | subject)`` over known verbs."""
        net = self._check()
        evidence = torch.tensor(self._evidence(subject), dtype=torch.long)
        values = torch.arange(len(self.verbs))
        log_p = net.conditional_log_probs(evidence, 1, values)
        probs = log_p.exp().tolist()
        out = {v: probs[i] for i, v in enumerate(self.verbs)}
        return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))


__all__ = [
    "NO_OBJ",
    "PAD",
    "SVOTuple",
    "SemanticTuplePC",
    "WordSequencePC",
    "extract_svo",
    "plausibility_bin",
]
