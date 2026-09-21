"""Evaluate a circuit-learned lexicon with the standard induction metrics."""

from __future__ import annotations

from pathlib import Path

from dylan.induction.em_learner.evaluation import Evaluation
from dylan.induction.em_learner.record_type_corpus import RecordTypeCorpus
from dylan.induction.pipeline.evaluate import build_eval_parser, evaluate_corpus
from dylan.induction.pipeline.metrics import SplitMetrics


def evaluate_learned_lexicon(
    *,
    lexicon_dir: Path,
    seed_grammar: Path,
    corpus: RecordTypeCorpus | Path,
    top_n: int = 1,
) -> SplitMetrics:
    """Load ``lexicon-top-{top_n}`` and score *corpus* with maximal-mapping P/R."""
    if isinstance(corpus, Path):
        corpus = RecordTypeCorpus(corpus_path=corpus)
    parser = build_eval_parser(
        lexicon_dir=lexicon_dir, seed_grammar=seed_grammar, top_n=top_n
    )
    return evaluate_corpus(parser, corpus)


def metrics_to_dict(m: SplitMetrics) -> dict:
    """JSON-serialisable view of :class:`SplitMetrics`."""
    return {
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "coverage": m.coverage,
        "exact_match": m.exact_match,
        "parsed_count": m.parsed_count,
        "total_count": m.total_count,
        "exact_match_count": m.exact_match_count,
    }


# Re-export Evaluation for callers that want field-alignment helpers.
__all__ = ["Evaluation", "evaluate_learned_lexicon", "metrics_to_dict"]
