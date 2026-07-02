"""Public API: grammar-bounded NeSy DS-VSS learner."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch

from dylan.parser.interactive_context_parser import InteractiveContextParser
from dylan.vss.ds_vss_session import DSVSSSession
from dylan.vss.embedding_store import EmbeddingStore, embedding_store_from_config
from dylan.vss.nesy.circuit_spec import linear_spec_from_lattice
from dylan.vss.nesy.dataset import LatentParseDataset
from dylan.vss.nesy.parse_lattice import (
    ParseLatticeBuilder,
    SupervisedParseExample,
    examples_from_parse_result,
)
from dylan.vss.nesy.parse_lattice_to_circuit import ParseCircuitModule, compile_parse_circuit
from dylan.vss.nesy.trainer import SupervisedTrainConfig, SupervisedTrainResult, fit_supervised
from dylan.vss.types import TensorRep, VSSConfig


@dataclass(frozen=True, slots=True)
class CircuitConfig:
    """Options for compiling parse circuits."""

    prefer_cirkit: bool = True
    max_lattice_states: int = 500


@dataclass
class ParseLatticeMarginals:
    """Per-word marginal over legal lexical indices (MAP from logits)."""

    words: tuple[str, ...]
    marginals: tuple[tuple[float, ...], ...]
    map_indices: tuple[int, ...]


@dataclass
class NesyDSVSSLearner:
    """Grammar-respecting neuro-symbolic learner over DS parse lattices and VSS."""

    grammar: Path
    store: EmbeddingStore
    circuit_config: CircuitConfig = field(default_factory=CircuitConfig)
    vss_config: VSSConfig = field(default_factory=VSSConfig)
    _parser: InteractiveContextParser | None = field(default=None, repr=False)
    _session: DSVSSSession | None = field(default=None, repr=False)
    _circuit: ParseCircuitModule | None = field(default=None, repr=False)

    def _ensure_parser(self) -> InteractiveContextParser:
        """Lazily load the grammar parser."""
        if self._parser is None:
            self._parser = InteractiveContextParser.from_resource_dir(
                self.grammar,
                top_n=self.vss_config.top_n,
            )
        return self._parser

    def _ensure_session(self) -> DSVSSSession:
        """Lazily construct a :class:`~dylan.vss.ds_vss_session.DSVSSSession`."""
        if self._session is None:
            self._session = DSVSSSession(
                self.grammar,
                embedding_store=self.store,
                config=self.vss_config,
                parser=self._ensure_parser(),
            )
        return self._session

    def example_from_sentence(self, sentence: str, *, speaker: str = "Dylan") -> SupervisedParseExample | None:
        """Parse *sentence* with trace and build a supervised example with lattice."""
        from dynamicsyntax._parse import _run_parse_core

        parser = self._ensure_parser()
        result = _run_parse_core(parser, sentence.strip(), speaker=speaker, trace=True)
        ex = examples_from_parse_result(result, grammar_path=self.grammar)
        if ex is None:
            return None
        builder = ParseLatticeBuilder(parser, max_states=self.circuit_config.max_lattice_states)
        ex.lattice = builder.gold_lattice_from_parse(result)
        return ex

    def fit_supervised(
        self,
        examples: Iterable[SupervisedParseExample],
        *,
        config: SupervisedTrainConfig | None = None,
    ) -> SupervisedTrainResult:
        """Train parse-circuit parameters on gold lexical paths."""
        cfg = config or SupervisedTrainConfig(prefer_cirkit=self.circuit_config.prefer_cirkit)
        prepared = list(examples)
        if not prepared:
            raise ValueError("No supervised examples")
        first = prepared[0]
        if first.lattice is None:
            raise ValueError("Supervised examples must include lattice")
        spec = linear_spec_from_lattice(first.lattice)
        self._circuit = compile_parse_circuit(
            spec,
            prefer_cirkit=cfg.prefer_cirkit,
        )
        return fit_supervised(
            prepared,
            self._circuit,
            store=self.store,
            session=self._ensure_session(),
            config=cfg,
        )

    def fit_unsupervised(
        self,
        dataset: LatentParseDataset,
        *,
        epochs: int = 1,
    ) -> None:
        """Placeholder for future EM over parse marginals and VSS emissions."""
        raise NotImplementedError(
            "Unsupervised NeSy training is not implemented. Future work: E-step with "
            "cirkit IntegrateQuery marginals over latent lexical variables per word; "
            "M-step on gating logits and LexiconVSSIndex embeddings (see em_learner). "
            f"Received {len(dataset)} latent sentences for {epochs} epoch(s)."
        )

    def predict_parse(self, sentence: str) -> ParseLatticeMarginals:
        """Return MAP lexical indices from the trained circuit (uniform if untrained)."""
        ex = self.example_from_sentence(sentence)
        if ex is None or ex.lattice is None:
            raise ValueError(f"Could not parse sentence: {sentence!r}")
        spec = linear_spec_from_lattice(ex.lattice)
        if self._circuit is None:
            self._circuit = compile_parse_circuit(
                spec,
                prefer_cirkit=self.circuit_config.prefer_cirkit,
            )
        marginals: list[tuple[float, ...]] = []
        map_indices: list[int] = []
        for step in spec.steps:
            if hasattr(self._circuit, "logits"):
                logits = self._circuit.logits[len(map_indices)]  # type: ignore[attr-defined]
                probs = torch.softmax(logits, dim=-1).detach().tolist()
            else:
                probs = [1.0 / step.num_categories] * step.num_categories
            marginals.append(tuple(probs))
            map_indices.append(int(max(range(len(probs)), key=probs.__getitem)))
        return ParseLatticeMarginals(
            words=spec.words,
            marginals=tuple(marginals),
            map_indices=tuple(map_indices),
        )

    def predict_semantics(self, sentence: str) -> TensorRep:
        """Run DS-VSS composition on the parsed sentence."""
        session = self._ensure_session()
        result = session.parse_incremental(sentence)
        if result.steps and result.steps[-1].composition is not None:
            comp = result.steps[-1].composition
            if comp.stages:
                last = comp.stages[-1]
                if last:
                    return next(iter(last.values()))
        raise ValueError(f"No VSS composition for sentence: {sentence!r}")
