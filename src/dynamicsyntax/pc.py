"""High-level facade: probabilistic circuits for neuro-symbolic learning.

A neuro-symbolic learner combining the DyLan Dynamic Syntax parser, DS-VSS
distributional semantics, and Einsum-Network probabilistic circuits
(Peharz, Lang, Vergari et al. 2020, ICML; Choi, Vergari & Van den Broeck
2020).  Two model families are provided:

- :class:`PCWordModel` — an exact-inference PC language model: next-word
  prediction is a *tractable conditional* of the circuit;
- :class:`DSPlausibilityPC` — a PC over semantic tuples extracted from
  DyLan parses together with DS-VSS incremental plausibility, answering
  plausibility and expectation queries by exact conditional inference.

Requires the optional ``pc`` extra (PyTorch): ``pip install dynamicsyntax[pc]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

try:
    from dylan.pc.bridge import (
        NO_OBJ,
        PAD,
        SemanticTuplePC,
        SVOTuple,
        WordSequencePC,
        extract_svo,
        plausibility_bin,
    )
    from dylan.pc.einet import UNKNOWN, EinsumNetwork, EiNetConfig
    from dylan.pc.regions import PCStructure, random_region_graph
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "dynamicsyntax.pc requires PyTorch; install the optional extra with "
        "`pip install dynamicsyntax[pc]` (or `pip install torch`)."
    ) from exc

from dylan.nlp.types import DEFAULT_SPEAKER
from dylan.vss.lexicon import VSSLexicon

from dynamicsyntax.vss import VSSParseResult, parse_vss

__all__ = [
    "NO_OBJ",
    "PAD",
    "UNKNOWN",
    "DSPlausibilityPC",
    "EinsumNetwork",
    "EiNetConfig",
    "PCStructure",
    "PCWordModel",
    "SVOTuple",
    "SemanticTuplePC",
    "WordSequencePC",
    "extract_svo",
    "plausibility_bin",
    "random_region_graph",
]


class PCWordModel:
    """Word-sequence PC language model with a string-level API.

    :param max_len: sequence length (shorter sentences are padded).
    :param einet_kwargs: structural hyper-parameters of the EiNet
        (e.g. ``num_sums``, ``num_input_dists``, ``num_repetitions``,
        ``seed``).
    """

    def __init__(self, max_len: int, **einet_kwargs) -> None:
        self._model = WordSequencePC(max_len, **einet_kwargs)

    def fit(
        self,
        sentences: Iterable[str],
        *,
        method: str = "em",
        epochs: int = 10,
        **fit_kwargs,
    ) -> list[float]:
        """Train on whitespace-tokenised sentences."""
        return self._model.fit(
            (s.split() for s in sentences), method=method, epochs=epochs, **fit_kwargs
        )

    def next_word_probs(self, prefix: str) -> dict[str, float]:
        """Exact ``p(w | prefix)`` over the vocabulary (incl. ``<pad>``)."""
        return self._model.next_word_probs(prefix.split() if prefix.strip() else [])

    def log_likelihood(self, sentence: str) -> float:
        """Exact log-likelihood of a sentence."""
        return self._model.log_likelihood(sentence.split())


class DSPlausibilityPC:
    """Neuro-symbolic plausibility model: DS-VSS tuples + PC inference.

    Training data are ``(subject, verb, object, plausibility-bin)`` tuples
    obtained by parsing a corpus with DyLan and scoring each sentence with
    DS-VSS; the PC then learns the joint distribution, and expectation /
    plausibility queries become exact conditionals.
    """

    def __init__(self, num_bins: int = 5, **einet_kwargs) -> None:
        self._model = SemanticTuplePC(num_bins=num_bins, **einet_kwargs)

    def build_tuples(
        self,
        sentences: Iterable[str],
        grammar: str | Path,
        lexicon: VSSLexicon,
        *,
        speaker: str = DEFAULT_SPEAKER,
    ) -> list[tuple[str, str, str | None, int]]:
        """Parse + DS-VSS-score a corpus into semantic training tuples."""
        rows: list[tuple[str, str, str | None, int]] = []
        results = parse_vss(list(sentences), grammar, lexicon=lexicon, speaker=speaker)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, VSSParseResult)
            if not r.ok or r.parse.tree is None or r.final_plausibility is None:
                continue
            svo = extract_svo(r.parse.tree)
            if svo.subject is None or svo.verb is None:
                continue
            b = plausibility_bin(r.final_plausibility, self._model.num_bins)
            rows.append((svo.subject, svo.verb, svo.obj, b))
        return rows

    def fit(
        self,
        sentences: Iterable[str],
        grammar: str | Path,
        lexicon: VSSLexicon,
        *,
        speaker: str = DEFAULT_SPEAKER,
        method: str = "em",
        epochs: int = 10,
        **fit_kwargs,
    ) -> list[float]:
        """Extract DS-VSS tuples from *sentences* and train the circuit."""
        rows = self.build_tuples(sentences, grammar, lexicon, speaker=speaker)
        if not rows:
            raise ValueError("no usable (subject, verb, object) tuples extracted")
        return self._model.fit(rows, method=method, epochs=epochs, **fit_kwargs)

    def plausibility_distribution(
        self, subject: str, verb: str, obj: str | None = None
    ) -> list[float]:
        """Exact posterior over plausibility bins of a semantic tuple."""
        return self._model.plausibility_distribution(subject, verb, obj)

    def expected_plausibility(self, subject: str, verb: str, obj: str | None = None) -> float:
        """Expected plausibility of a semantic tuple (bin-centre weighted)."""
        return self._model.expected_plausibility(subject, verb, obj)

    def rank_objects(self, subject: str, verb: str) -> dict[str, float]:
        """Exact ``p(object | subject, verb)`` — the expectation model."""
        return self._model.object_probs(subject, verb)

    def rank_verbs(self, subject: str) -> dict[str, float]:
        """Exact ``p(verb | subject)`` — verb continuation expectation."""
        return self._model.verb_probs(subject)
