"""Held-out evaluation: parse with a learnt grammar and score against gold TTR."""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger

from dylan.action.grammar import Grammar
from dylan.action.lexicon import Lexicon
from dylan.action.speech_act_inference_grammar import SpeechActInferenceGrammar
from dylan.dag.uttered_word import UtteredWord
from dylan.formula.ttr_record_type import TTRRecordType
from dylan.induction.em_learner.common import words_to_string
from dylan.induction.em_learner.evaluation import Evaluation
from dylan.induction.em_learner.record_type_corpus import RecordTypeCorpus
from dylan.induction.pipeline.metrics import EvalResult, SplitMetrics
from dylan.nlp.types import DEFAULT_SPEAKER, Utterance
from dylan.parser.interactive_context_parser import InteractiveContextParser


def resolve_model_dir_for_top_n(*, lexicon_prefix: Path, top_n: int) -> Path:
    """Return the directory that contains the learnt ``lexicon-top-{top_n}.txt`` (or Java name)."""
    parent = Path(lexicon_prefix).parent
    if Lexicon.resolve_learnt_lexicon_path(parent, top_n) is not None:
        return parent
    # Prefix may already be the run dir itself (no trailing ``lexicon`` stem).
    if Lexicon.resolve_learnt_lexicon_path(lexicon_prefix, top_n) is not None:
        return Path(lexicon_prefix)
    raise FileNotFoundError(
        f"Learnt lexicon not found for top-{top_n} under {parent} "
        f"(expected lexicon-top-{top_n}.txt or lexicon.lex-top-{top_n}.txt)",
    )


_HYP_ACTION_PREFIX = "hyp"


def grammar_for_eval(seed_grammar: Path) -> Grammar:
    """Load computational actions for eval, excluding induction ``hyp-*`` rules.

    Java BabyDS eval loads ``InteractiveContextParser`` from the *model* directory.
    Those model ``computational-actions.txt`` files omit ``hyp-*`` hypothesis rules
    (training-only tree-building hyps). Using the full seed grammar at eval time
    puts those hyps into the optional grammar and explodes left-adjustment search.
    """
    raw = Grammar(seed_grammar)
    filtered = Grammar()
    for name, action in raw.items():
        if str(name).lower().startswith(_HYP_ACTION_PREFIX):
            continue
        filtered[name] = action
    return filtered


def build_eval_parser(
    *,
    lexicon_dir: Path,
    seed_grammar: Path,
    top_n: int,
) -> InteractiveContextParser:
    """Load a learnt IF/THEN lexicon from *lexicon_dir* and eval computational actions from *seed_grammar*."""
    if not (seed_grammar / "computational-actions.txt").is_file():
        raise FileNotFoundError(f"Missing computational-actions.txt in {seed_grammar}")
    lex = Lexicon(lexicon_dir, top_n, load_learnt_lexicon=True)
    if not lex:
        raise RuntimeError(f"Learnt lexicon loaded 0 words from {lexicon_dir} (top-{top_n})")
    grammar = grammar_for_eval(seed_grammar)
    sa = SpeechActInferenceGrammar(seed_grammar)
    return InteractiveContextParser.from_loaded(lex, grammar, sa=sa, log_level="off")


def _utterance_from_words(words: list) -> Utterance:
    """Build an :class:`Utterance` from induction :class:`Word` objects."""
    uttered = [
        UtteredWord(w.word() if hasattr(w, "word") else str(w), DEFAULT_SPEAKER)
        for w in words
    ]
    return Utterance(speaker=DEFAULT_SPEAKER, words=uttered)


def _collect_semantics(parser: InteractiveContextParser) -> list[TTRRecordType]:
    """Collect the current semantics plus alternate interpretations via ``parse()``."""
    all_sem: list[TTRRecordType] = []
    try:
        first = parser.get_state().get_current_tuple().get_semantics(parser.context)
        ev = first.evaluate() if hasattr(first, "evaluate") else first
        if isinstance(ev, TTRRecordType):
            all_sem.append(ev)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not read primary semantics: {}", exc)

    while True:
        try:
            if not parser.parse():
                break
            sem = parser.get_state().get_current_tuple().get_semantics(parser.context)
            ev = sem.evaluate() if hasattr(sem, "evaluate") else sem
            if isinstance(ev, TTRRecordType) and ev not in all_sem:
                all_sem.append(ev)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Stopped alternate-semantics loop: {}", exc)
            break
    return all_sem


def evaluate_corpus(
    parser: InteractiveContextParser,
    corpus: RecordTypeCorpus,
) -> SplitMetrics:
    """Parse *corpus* with *parser* and return macro P/R/F1, coverage, and EM."""
    pairs: list[tuple[TTRRecordType, TTRRecordType]] = []
    failed_parses: list[str] = []
    failed_em: list[str] = []
    parsed_count = 0
    exact_match_count = 0
    total = len(corpus)

    for words, gold in corpus:
        sent = words_to_string(words)
        try:
            parser.init()
            ok = parser.parse_utterance(_utterance_from_words(words))
            if not ok:
                failed_parses.append(sent)
                continue
            candidates = _collect_semantics(parser)
            if not candidates:
                failed_parses.append(sent)
                continue
            best = Evaluation.find_best_ttr_interpretation(gold, candidates)
            if best is None:
                failed_parses.append(sent)
                continue
            parsed_count += 1
            pairs.append((best, gold))
            if best.subsumes(gold) and gold.subsumes(best):
                exact_match_count += 1
            else:
                failed_em.append(sent)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error parsing {!r}: {}", sent, exc)
            failed_parses.append(sent)

    if pairs:
        p, r, f = Evaluation.precision_recall_macro(pairs)
        precision, recall, f1 = p * 100.0, r * 100.0, f * 100.0
    else:
        precision = recall = f1 = 0.0

    coverage = (parsed_count / total * 100.0) if total else 0.0
    em = (exact_match_count / parsed_count * 100.0) if parsed_count else 0.0

    return SplitMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        coverage=coverage,
        exact_match=em,
        parsed_count=parsed_count,
        total_count=total,
        exact_match_count=exact_match_count,
        failed_parses=failed_parses,
        failed_exact_matches=failed_em,
    )


def evaluate_model_dir(
    *,
    seed_grammar: Path,
    lexicon_prefix: Path,
    corpora: dict[str, RecordTypeCorpus],
    top_n_start: int,
    top_n_end: int,
    staging_root: Path | None = None,
) -> tuple[EvalResult, float]:
    """Evaluate *corpora* for each top-N; return ``(result, eval_seconds)``.

    *staging_root* is accepted for API compatibility but unused: learnt lexicons are
    loaded directly from the model directory via :meth:`Lexicon.load_learnt_lexicon_txt`.
    """
    del staging_root  # no longer staging as seed-style lexicon.txt
    result = EvalResult()
    t0 = time.perf_counter()
    for n in range(top_n_start, top_n_end + 1):
        model_dir = resolve_model_dir_for_top_n(lexicon_prefix=lexicon_prefix, top_n=n)
        parser = build_eval_parser(
            lexicon_dir=model_dir,
            seed_grammar=seed_grammar,
            top_n=n,
        )
        try:
            for split_name, corpus in corpora.items():
                logger.info("Evaluating {} with top-{} ({} examples)", split_name, n, len(corpus))
                metrics = evaluate_corpus(parser, corpus)
                result.add(n, split_name, metrics)
                logger.info(
                    "{} top-{}: P={:.2f} R={:.2f} F1={:.2f} cov={:.2f} EM={:.2f}",
                    split_name,
                    n,
                    metrics.precision,
                    metrics.recall,
                    metrics.f1,
                    metrics.coverage,
                    metrics.exact_match,
                )
        finally:
            parser.close()
    elapsed = time.perf_counter() - t0
    result.metadata["eval_time_s"] = elapsed
    return result, elapsed
