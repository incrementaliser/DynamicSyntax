"""Predicate-aligned DS-VSS features for gating.

BabyDS gold RTs use constants such as ``state_holding`` / ``obj_ball`` while
surface forms are ``pickup`` / ``ball``.  This module builds a lexicon keyed
by TTR predicate constants extracted from gold Sem strings (and optional
parse formulas), so ``parse_vss`` decorations are not stuck on unit tensors.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.em_learner.common import words_to_string
from dylan.induction.em_learner.record_type_corpus import RecordTypeCorpus
from dylan.nlp.types import DEFAULT_SPEAKER
from dylan.parser.interactive_context_parser import InteractiveContextParser
from dylan.vss.decorate import VSSDecorator
from dylan.vss.incremental import decorate_traces
from dylan.vss.lexicon import VSSLexicon
from dylan.vss.spaces import VSSValue
from dynamicsyntax._parse import _run_parse_core
from dynamicsyntax.vss import VSSParseResult

_MANIFEST_RE = re.compile(r"==\s*([A-Za-z][\w'-]*)\s*:\s*([A-Za-z]+)")

# Surface command → eventuality-style predicate used in BabyDS gold RTs.
_SURFACE_EVENT = {
    "pickup": "state_holding",
    "take": "state_holding",
    "grab": "state_holding",
    "goto": "state_facing",
    "open": "state_opened",
    "put": "state_holding",
    "drop": "state_holding",
    "move": "state_facing",
}


def predicates_from_rt(rt: TTRRecordType) -> list[tuple[str, str]]:
    """Return ``(constant, basic_type)`` pairs from a gold record type."""
    out: list[tuple[str, str]] = []
    for match in _MANIFEST_RE.finditer(str(rt)):
        out.append((match.group(1), match.group(2)))
    return out


def build_aligned_lexicon(
    corpus: RecordTypeCorpus | Iterable[tuple],
    *,
    context_dim: int = 8,
) -> VSSLexicon:
    """Build a PPMI-ish entity/event lexicon keyed by TTR predicate constants.

    Context basis is the set of all observed constants; each predicate gets a
    co-occurrence vector over that basis.  Eventualities (``es``) become
    intransitive plausibility matrices; entity-typed constants become vectors.
    """
    pairs: list[tuple[str, TTRRecordType]] = []
    if isinstance(corpus, RecordTypeCorpus):
        for words, rt in corpus:
            pairs.append((words_to_string(words), rt))
    else:
        for item in corpus:
            words, rt = item[0], item[1]
            if not isinstance(words, str):
                words = words_to_string(words)
            pairs.append((words, rt))

    all_consts: list[str] = []
    per_sent: list[list[tuple[str, str]]] = []
    for _sent, rt in pairs:
        preds = predicates_from_rt(rt)
        per_sent.append(preds)
        all_consts.extend(c for c, _ in preds)
    # Also register surface→event aliases so decoration can fall back if needed.
    for surface, ev in _SURFACE_EVENT.items():
        all_consts.append(ev)
    basis = sorted(set(all_consts)) or ["unk"]
    dim = max(context_dim, len(basis))
    # Pad basis if needed for a fixed dim.
    while len(basis) < dim:
        basis.append(f"_pad{len(basis)}")
    basis = basis[:dim]
    bix = {b: i for i, b in enumerate(basis)}

    counts = np.zeros((len(basis), dim), dtype=float)
    event_with: dict[str, np.ndarray] = {}
    event_without: dict[str, np.ndarray] = {}
    for preds in per_sent:
        consts = [c for c, _ in preds]
        types = {c: t for c, t in preds}
        for c in consts:
            if c not in bix:
                continue
            row = bix[c]
            for other in consts:
                if other in bix and other != c:
                    counts[row, bix[other]] += 1.0
            if types.get(c) == "es":
                hit = np.array([1.0 if b in consts else 0.0 for b in basis])
                event_with.setdefault(c, np.zeros(dim))
                event_without.setdefault(c, np.zeros(dim))
                event_with[c] += hit
                event_without[c] += 1.0 - hit

    # Ensure alias events get some mass even if rare in gold.
    for ev in set(_SURFACE_EVENT.values()):
        if ev in bix and ev not in event_with:
            event_with[ev] = np.ones(dim)
            event_without[ev] = np.ones(dim)

    lex = VSSLexicon.from_cooccurrence(
        basis, basis, counts + np.eye(dim), weighting="ppmi"
    )
    # Replace / add event matrices.
    for ev, with_c in event_with.items():
        mat = lex.plausibility_matrix(with_c, event_without.get(ev))
        lex.add_intransitive(ev, mat)
    return lex


def extract_z_trajectory(
    sentence: str,
    parser: InteractiveContextParser,
    lexicon: VSSLexicon,
) -> list[np.ndarray]:
    """Return one root ``S``-vector per word step (excluding the empty axiom)."""
    result = _run_parse_core(parser, sentence, speaker=DEFAULT_SPEAKER, trace=True)
    decorator = VSSDecorator(lexicon, mode="sum")
    trees = list(result.trace_trees) if result.trace_trees else (
        [result.tree] if result.tree else []
    )
    decorations = decorate_traces(trees, decorator)
    vss = VSSParseResult(
        parse=result, decorations=decorations, step_labels=result.trace_step_labels
    )
    vectors: list[np.ndarray] = []
    # Skip axiom (index 0); one vector per word.
    for dec in vss.decorations[1:]:
        root = dec.root_value
        if isinstance(root, VSSValue) and root.order == 1:
            vectors.append(np.asarray(root.array, dtype=float))
        else:
            # Fallback: uniform plausibility vector in default S.
            vectors.append(np.array([0.5, 0.5], dtype=float))
    return vectors


def z_tensors(
    sentence: str,
    parser: InteractiveContextParser,
    lexicon: VSSLexicon,
) -> list[torch.Tensor]:
    """Torch float tensors for :func:`extract_z_trajectory`."""
    return [torch.tensor(z, dtype=torch.float32) for z in extract_z_trajectory(sentence, parser, lexicon)]
