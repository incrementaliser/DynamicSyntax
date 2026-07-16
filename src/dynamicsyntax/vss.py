"""High-level DS-VSS facade: parse with DyLan, decorate with tensors.

DS-VSS (Dynamic Syntax with Vector Space Semantics; Sadrzadeh, Purver,
Hough & Kempson 2018, arXiv:1811.00614) assigns the incremental,
word-by-word parsing process of Dynamic Syntax a compositional
*distributional* semantics: DS types become tensor products of vector
spaces, DS function application becomes tensor contraction, and DS
requirements become unit/sum tensors — so even partial utterances compile
to a sentence vector whose plausibility can be tracked incrementally.

Typical usage::

    import dynamicsyntax as ds
    from dynamicsyntax.vss import VSSLexicon, parse_vss

    lex = VSSLexicon.from_cooccurrence(targets, contexts, counts)
    lex.add_intransitive("sleep", [[...], ...])
    r = parse_vss("john sleeps", "ttr", lexicon=lex)
    print(r.plausibilities)   # word-by-word plausibility trajectory
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import overload

from dylan.nlp.types import DEFAULT_SPEAKER
from dylan.vss.decorate import RequirementMode, VSSDecoration, VSSDecorator
from dylan.vss.incremental import decorate_traces, plausibility_trajectory
from dylan.vss.lexicon import VSSLexicon
from dylan.vss.spaces import (
    FALSE,
    TRUE,
    VSSDirectSum,
    VSSValue,
    VectorSpace,
    plausibility,
    plausibility_space,
)

from dynamicsyntax._parse import parse
from dynamicsyntax.parse_result import ParseResult

__all__ = [
    "FALSE",
    "TRUE",
    "RequirementMode",
    "VSSDecoration",
    "VSSDecorator",
    "VSSDirectSum",
    "VSSLexicon",
    "VSSParseResult",
    "VSSValue",
    "VectorSpace",
    "parse_vss",
    "plausibility",
    "plausibility_space",
    "vss_plausibility",
]


@dataclass
class VSSParseResult:
    """A DyLan parse together with its DS-VSS tensor semantics.

    :param parse: the underlying :class:`~dynamicsyntax.parse_result.ParseResult`.
    :param decorations: one :class:`VSSDecoration` per trace tree — the
        initial axiom tree, then one per parsed word.
    :param step_labels: surface words labelling the trace steps.
    """

    parse: ParseResult
    decorations: list[VSSDecoration] = field(default_factory=list)
    step_labels: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the underlying parse succeeded."""
        return self.parse.ok

    @property
    def sentence_values(self) -> list[VSSValue | VSSDirectSum | None]:
        """Sentence-space value at the root of each trace tree."""
        return [d.root_value for d in self.decorations]

    @property
    def plausibilities(self) -> list[float | None]:
        """Word-by-word root plausibility (the incremental trajectory)."""
        return plausibility_trajectory(self.decorations)

    @property
    def final_plausibility(self) -> float | None:
        """Plausibility of the complete utterance (last trace step)."""
        ps = self.plausibilities
        return ps[-1] if ps else None

    @property
    def missing_predicates(self) -> tuple[str, ...]:
        """Predicate constants with no lexicon tensor (per trace step)."""
        seen: list[str] = []
        for d in self.decorations:
            for m in d.missing:
                if m not in seen:
                    seen.append(m)
        return tuple(seen)

    def trajectory(self) -> list[tuple[str, float | None]]:
        """``(word, plausibility)`` pairs, starting with the empty utterance."""
        labels = ("∅",) + tuple(self.step_labels)
        return list(zip(labels, self.plausibilities))


def _parse_and_decorate(
    parser_sentence: str,
    grammar: str | Path,
    lexicon: VSSLexicon,
    requirement_mode: RequirementMode | str,
    speaker: str,
) -> VSSParseResult:
    result = parse(parser_sentence, grammar, speaker=speaker, trace=True)
    decorator = VSSDecorator(lexicon, mode=requirement_mode)
    trees = list(result.trace_trees) if result.trace_trees else ([result.tree] if result.tree else [])
    decorations = decorate_traces(trees, decorator)
    return VSSParseResult(
        parse=result,
        decorations=decorations,
        step_labels=result.trace_step_labels,
    )


@overload
def parse_vss(
    sentence: str,
    grammar: str | Path = "ttr",
    /,
    *,
    lexicon: VSSLexicon,
    requirement_mode: RequirementMode | str = RequirementMode.SUM,
    speaker: str = ...,
) -> VSSParseResult: ...


@overload
def parse_vss(
    sentence: list[str],
    grammar: str | Path = "ttr",
    /,
    *,
    lexicon: VSSLexicon,
    requirement_mode: RequirementMode | str = RequirementMode.SUM,
    speaker: str = ...,
) -> list[VSSParseResult]: ...


def parse_vss(
    sentence: str | list[str],
    grammar: str | Path = "ttr",
    /,
    *,
    lexicon: VSSLexicon,
    requirement_mode: RequirementMode | str = RequirementMode.SUM,
    speaker: str = DEFAULT_SPEAKER,
) -> VSSParseResult | list[VSSParseResult]:
    """Parse with DyLan and decorate the parse trace with DS-VSS tensors.

    :param sentence: surface string (or list of strings), as for
        :func:`dynamicsyntax.parse`.
    :param grammar: bundled grammar id/alias or grammar directory.
    :param lexicon: the distributional lexicon supplying word and
        predicate tensors.
    :param requirement_mode: interpretation of DS requirements — ``"sum"``
        (the paper's working example), ``"unit"`` or ``"direct_sum"``.
    :param speaker: dialogue participant id passed to the parser.
    """
    if isinstance(sentence, list):
        return [
            _parse_and_decorate(s, grammar, lexicon, requirement_mode, speaker)
            for s in sentence
        ]
    return _parse_and_decorate(sentence, grammar, lexicon, requirement_mode, speaker)


def vss_plausibility(
    sentence: str,
    grammar: str | Path = "ttr",
    /,
    *,
    lexicon: VSSLexicon,
    requirement_mode: RequirementMode | str = RequirementMode.SUM,
    speaker: str = DEFAULT_SPEAKER,
) -> float | None:
    """Final incremental plausibility of *sentence* under *lexicon*."""
    result = parse_vss(
        sentence,
        grammar,
        lexicon=lexicon,
        requirement_mode=requirement_mode,
        speaker=speaker,
    )
    assert isinstance(result, VSSParseResult)
    return result.final_plausibility
