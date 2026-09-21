"""Corpus-scale evaluation of DS-VSS and probabilistic-circuit models.

Evaluates:
  - BabyDS (100 train / 12 test): PCWordModel, DS-VSS, DSPlausibilityPC
  - CHILDES (3200 / 800): PCWordModel only

Run from repo root::

    uv sync --extra pc
    uv run python scripts/nesy_corpus_eval.py

Artifacts land under ``out/runs/nesy-corpus-eval-<timestamp>/``.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from dylan.induction.em_learner.common import words_to_string
from dylan.induction.em_learner.record_type_corpus import RecordTypeCorpus
from dylan.induction.pipeline.evaluate import build_eval_parser
from dylan.nlp.types import DEFAULT_SPEAKER
from dylan.parser.interactive_context_parser import InteractiveContextParser
from dylan.pc.bridge import (
    NO_OBJ,
    PAD,
    SemanticTuplePC,
    extract_svo,
    plausibility_bin,
)
from dylan.vss.decorate import VSSDecorator
from dylan.vss.incremental import decorate_traces
from dylan.vss.lexicon import VSSLexicon
from dynamicsyntax._parse import _run_parse_core
from dynamicsyntax.pc import DSPlausibilityPC, PCWordModel
from dynamicsyntax.vss import VSSParseResult, parse_vss

ROOT = Path(__file__).resolve().parents[1]

BABYDS_TRAIN = ROOT / "data/BabyDS/class1_train_100.txt"
BABYDS_TEST = ROOT / "data/BabyDS/class1_test_12.txt"
CHILDES_TRAIN = ROOT / "data/CHILDES/LC-CHILDESconversion3200TrainFinal.txt"
CHILDES_TEST = ROOT / "data/CHILDES/LC-CHILDESconversion800TestFinal.txt"

BABYDS_VERBS = frozenset(
    {"pickup", "goto", "put", "open", "drop", "take", "grab", "move", "place"}
)
FUNCTION_WORDS = frozenset(
    {"a", "an", "the", "to", "of", "on", "in", "at", "and", "or", "'s", "is", "are"}
)
NOUNISH = frozenset(
    {
        "ball",
        "box",
        "key",
        "door",
        "cube",
        "cylinder",
        "pyramid",
        "red",
        "blue",
        "green",
        "yellow",
        "purple",
        "grey",
        "gray",
        "white",
        "black",
        "orange",
        "pink",
    }
)


# ---------------------------------------------------------------------------
# corpus I/O
# ---------------------------------------------------------------------------
def load_sentences(path: Path) -> list[str]:
    """Load surface sentences from a ``Sent`` / ``GoldSent`` TTR corpus file."""
    corpus = RecordTypeCorpus(corpus_path=path)
    return [words_to_string(words) for words, _ in corpus]


def tokenize(sentence: str) -> list[str]:
    """Whitespace-tokenise a surface sentence."""
    return sentence.split()


# ---------------------------------------------------------------------------
# co-occurrence lexicon
# ---------------------------------------------------------------------------
def window_cooccurrence(
    sentences: Sequence[str],
    *,
    window: int = 2,
) -> tuple[list[str], list[str], np.ndarray]:
    """Build a target×context count matrix from a sliding window.

    Content words (non-function) are row targets; all vocabulary words are
    context columns (PPMI basis of ``W``).
    """
    tokenised = [tokenize(s) for s in sentences]
    vocab = sorted({t for toks in tokenised for t in toks})
    targets = [t for t in vocab if t not in FUNCTION_WORDS] or list(vocab)
    target_ix = {t: i for i, t in enumerate(targets)}
    context_ix = {c: i for i, c in enumerate(vocab)}
    counts = np.zeros((len(targets), len(vocab)), dtype=float)
    for toks in tokenised:
        for i, w in enumerate(toks):
            if w not in target_ix:
                continue
            lo = max(0, i - window)
            hi = min(len(toks), i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                counts[target_ix[w], context_ix[toks[j]]] += 1.0
    return targets, vocab, counts


def build_babyds_lexicon(train: Sequence[str]) -> VSSLexicon:
    """PPMI entity lexicon + verb plausibility tensors from BabyDS train text."""
    targets, contexts, counts = window_cooccurrence(train)
    lex = VSSLexicon.from_cooccurrence(targets, contexts, counts, weighting="ppmi")
    verb_contexts: list[tuple[list[str], str]] = []
    transitive: list[str] = []
    for sent in train:
        toks = tokenize(sent)
        if not toks:
            continue
        verb = toks[0].lower()
        if verb not in BABYDS_VERBS:
            # also treat any first-token verb-ish word seen in train
            if verb in FUNCTION_WORDS:
                continue
        others = [t for t in toks[1:] if t != verb]
        verb_contexts.append((others, verb))
        if any(t in NOUNISH for t in others):
            transitive.append(verb)
    if verb_contexts:
        lex.learn_plausibility_verbs(verb_contexts, transitive=transitive)
    return lex


# ---------------------------------------------------------------------------
# PCWordModel metrics
# ---------------------------------------------------------------------------
@dataclass
class PCWordMetrics:
    """Held-out / train metrics for a word-sequence PC."""

    n_sentences: int = 0
    in_vocab_n: int = 0
    in_vocab_coverage: float = 0.0
    truncated_n: int = 0
    truncate_rate: float = 0.0
    mean_nll: float | None = None
    perplexity: float | None = None
    next_word_top1: float | None = None
    next_word_n: int = 0
    train_nll_history_first: float | None = None
    train_nll_history_last: float | None = None
    max_len: int = 0
    vocab_size: int = 0
    epochs: int = 0


def _in_vocab(tokens: Sequence[str], vocab: set[str]) -> bool:
    return all(t in vocab for t in tokens)


def evaluate_pc_word(
    model: PCWordModel,
    sentences: Sequence[str],
    *,
    max_len: int,
) -> PCWordMetrics:
    """Score sentences with NLL / perplexity / next-word top-1 under closed vocab."""
    vocab = set(model._model.vocab)
    m = PCWordMetrics(
        n_sentences=len(sentences),
        max_len=max_len,
        vocab_size=len(vocab),
    )
    nlls: list[float] = []
    correct = 0
    total = 0
    for sent in sentences:
        toks = tokenize(sent)
        if len(toks) > max_len:
            m.truncated_n += 1
            toks = toks[:max_len]
        if not toks or not _in_vocab(toks, vocab):
            continue
        m.in_vocab_n += 1
        try:
            nlls.append(-model.log_likelihood(" ".join(toks)))
        except Exception:  # noqa: BLE001
            continue
        for i in range(len(toks)):
            prefix = toks[:i]
            gold = toks[i]
            if gold not in vocab:
                continue
            try:
                probs = model.next_word_probs(" ".join(prefix) if prefix else "")
            except Exception:  # noqa: BLE001
                continue
            ranked = [(w, p) for w, p in probs.items() if w != PAD]
            if not ranked:
                continue
            pred = max(ranked, key=lambda kv: kv[1])[0]
            total += 1
            if pred == gold:
                correct += 1
    m.truncate_rate = m.truncated_n / m.n_sentences if m.n_sentences else 0.0
    m.in_vocab_coverage = m.in_vocab_n / m.n_sentences if m.n_sentences else 0.0
    if nlls:
        m.mean_nll = float(sum(nlls) / len(nlls))
        m.perplexity = float(math.exp(min(m.mean_nll, 50.0)))
    if total:
        m.next_word_top1 = correct / total
        m.next_word_n = total
    return m


def train_pc_word(
    train: Sequence[str],
    test: Sequence[str],
    *,
    max_len: int,
    epochs: int,
    seed: int = 0,
) -> tuple[PCWordModel, PCWordMetrics, PCWordMetrics, list[float]]:
    """Fit a PCWordModel and evaluate on train and test."""
    torch.manual_seed(seed)
    # Truncate train sentences that exceed max_len so fit does not raise.
    train_fit = []
    for s in train:
        toks = tokenize(s)
        if not toks:
            continue
        train_fit.append(" ".join(toks[:max_len]))
    model = PCWordModel(max_len=max_len, seed=seed)
    history = model.fit(train_fit, method="em", epochs=epochs)
    train_m = evaluate_pc_word(model, train, max_len=max_len)
    test_m = evaluate_pc_word(model, test, max_len=max_len)
    for m in (train_m, test_m):
        m.epochs = epochs
        if history:
            m.train_nll_history_first = float(history[0])
            m.train_nll_history_last = float(history[-1])
    return model, train_m, test_m, history


# ---------------------------------------------------------------------------
# DS-VSS metrics
# ---------------------------------------------------------------------------
@dataclass
class VSSMetrics:
    """Parse-and-decorate metrics for DS-VSS on a corpus split."""

    n_sentences: int = 0
    ok_n: int = 0
    ok_rate: float = 0.0
    mean_final_plausibility: float | None = None
    no_missing_predicates_rate: float = 0.0
    samples: list[dict[str, Any]] = field(default_factory=list)
    grammar: str = ""


def parse_vss_with_parser(
    sentence: str,
    parser: InteractiveContextParser,
    lexicon: VSSLexicon,
) -> VSSParseResult:
    """Parse with an existing ICP (e.g. induction eval parser) and decorate with DS-VSS."""
    result = _run_parse_core(parser, sentence, speaker=DEFAULT_SPEAKER, trace=True)
    decorator = VSSDecorator(lexicon, mode="unit")
    trees = list(result.trace_trees) if result.trace_trees else (
        [result.tree] if result.tree else []
    )
    decorations = decorate_traces(trees, decorator)
    return VSSParseResult(
        parse=result,
        decorations=decorations,
        step_labels=result.trace_step_labels,
    )


def evaluate_vss(
    sentences: Sequence[str],
    lexicon: VSSLexicon,
    grammar: str | Path,
    *,
    parser: InteractiveContextParser | None = None,
    sample_n: int = 5,
) -> VSSMetrics:
    """Run DS-VSS decoration over *sentences* and aggregate coverage / plausibility."""
    m = VSSMetrics(n_sentences=len(sentences), grammar=str(grammar))
    if not sentences:
        return m
    results: list[VSSParseResult] = []
    if parser is not None:
        for sent in sentences:
            results.append(parse_vss_with_parser(sent, parser, lexicon))
    else:
        batch = parse_vss(
            list(sentences),
            grammar,
            lexicon=lexicon,
            requirement_mode="unit",
        )
        assert isinstance(batch, list)
        results = batch
    plaus: list[float] = []
    no_missing = 0
    for r in results:
        if r.ok:
            m.ok_n += 1
        if not r.missing_predicates:
            no_missing += 1
        if r.final_plausibility is not None:
            plaus.append(float(r.final_plausibility))
    m.ok_rate = m.ok_n / m.n_sentences if m.n_sentences else 0.0
    m.no_missing_predicates_rate = no_missing / m.n_sentences if m.n_sentences else 0.0
    if plaus:
        m.mean_final_plausibility = float(sum(plaus) / len(plaus))
    for sent, r in zip(sentences[:sample_n], results[:sample_n]):
        m.samples.append(
            {
                "sentence": sent,
                "ok": r.ok,
                "final_plausibility": r.final_plausibility,
                "missing_predicates": list(r.missing_predicates),
                "trajectory": [
                    {"word": w, "plausibility": p} for w, p in r.trajectory()
                ],
            }
        )
    return m


# ---------------------------------------------------------------------------
# DSPlausibilityPC — hybrid tuples for BabyDS imperatives
# ---------------------------------------------------------------------------
def hybrid_tuple(
    sentence: str,
    lexicon: VSSLexicon,
    grammar: str | Path,
    *,
    num_bins: int,
    parser: InteractiveContextParser | None = None,
) -> tuple[str, str, str | None, int] | None:
    """Build one ``(subj, verb, obj, bin)`` row via tree SVO or surface fallback."""
    toks = tokenize(sentence)
    if not toks:
        return None
    if parser is not None:
        r = parse_vss_with_parser(sentence, parser, lexicon)
    else:
        r = parse_vss(sentence, grammar, lexicon=lexicon, requirement_mode="unit")
    subject: str | None = None
    verb: str | None = None
    obj: str | None = None
    if r.ok and r.parse.tree is not None:
        svo = extract_svo(r.parse.tree)
        subject, verb, obj = svo.subject, svo.verb, svo.obj
    if subject is None or verb is None:
        verb = toks[0].lower()
        subject = "agent"
        nouns = [t for t in toks[1:] if t.lower() in NOUNISH or t not in FUNCTION_WORDS]
        obj_candidates = [t for t in toks if t.lower() in NOUNISH]
        obj = obj_candidates[-1] if obj_candidates else (nouns[-1] if nouns else None)
    if r.final_plausibility is not None:
        b = plausibility_bin(r.final_plausibility, num_bins)
    else:
        b = num_bins // 2
    return (subject, verb, obj, b)


def build_hybrid_tuples(
    sentences: Sequence[str],
    lexicon: VSSLexicon,
    grammar: str | Path,
    *,
    num_bins: int,
    parser: InteractiveContextParser | None = None,
) -> list[tuple[str, str, str | None, int]]:
    """Extract hybrid SVO+bin tuples for a corpus."""
    rows: list[tuple[str, str, str | None, int]] = []
    for s in sentences:
        row = hybrid_tuple(
            s, lexicon, grammar, num_bins=num_bins, parser=parser
        )
        if row is not None:
            rows.append(row)
    return rows


@dataclass
class DSPMetrics:
    """Metrics for the DS-VSS + semantic-tuple PC."""

    train_yield: int = 0
    train_skip_rate: float = 0.0
    test_yield: int = 0
    test_nll: float | None = None
    object_rank_acc: float | None = None
    verb_rank_acc: float | None = None
    object_rank_n: int = 0
    verb_rank_n: int = 0
    nll_history_first: float | None = None
    nll_history_last: float | None = None
    note: str = ""


def _tuple_nll(model: SemanticTuplePC, row: tuple[str, str, str | None, int]) -> float | None:
    """Exact negative log-likelihood of one semantic tuple under the fitted PC."""
    s, v, o, b = row
    o_key = o if o is not None else NO_OBJ
    try:
        if s not in model.subjects or v not in model.verbs or o_key not in model.objects:
            return None
        if not (0 <= b < model.num_bins):
            return None
        net = model._check()
        import torch as _torch

        x = _torch.tensor(
            [[model.subjects.index(s), model.verbs.index(v), model.objects.index(o_key), b]],
            dtype=_torch.long,
        )
        return float(-net(x)[0])
    except Exception:  # noqa: BLE001
        return None


def evaluate_dsp(
    train: Sequence[str],
    test: Sequence[str],
    lexicon: VSSLexicon,
    grammar: str | Path,
    *,
    num_bins: int = 4,
    epochs: int = 20,
    seed: int = 0,
    parser: InteractiveContextParser | None = None,
) -> DSPMetrics:
    """Train DSPlausibilityPC on hybrid tuples and score held-out ranking / NLL."""
    torch.manual_seed(seed)
    m = DSPMetrics()
    train_rows = build_hybrid_tuples(
        train, lexicon, grammar, num_bins=num_bins, parser=parser
    )
    test_rows = build_hybrid_tuples(
        test, lexicon, grammar, num_bins=num_bins, parser=parser
    )
    m.train_yield = len(train_rows)
    m.test_yield = len(test_rows)
    m.train_skip_rate = 1.0 - (len(train_rows) / len(train) if train else 0.0)
    if not train_rows:
        m.note = "no training tuples extracted; skipped fit"
        return m
    pc = DSPlausibilityPC(num_bins=num_bins, seed=seed)
    history = pc._model.fit(train_rows, method="em", epochs=epochs)
    if history:
        m.nll_history_first = float(history[0])
        m.nll_history_last = float(history[-1])
    nlls = [_tuple_nll(pc._model, row) for row in test_rows]
    nlls_f = [x for x in nlls if x is not None]
    if nlls_f:
        m.test_nll = float(sum(nlls_f) / len(nlls_f))
    # Ranking accuracy
    obj_ok = obj_n = 0
    verb_ok = verb_n = 0
    for s, v, o, _b in test_rows:
        try:
            if o is not None and o != NO_OBJ and s in pc._model.subjects and v in pc._model.verbs:
                ranked = list(pc.rank_objects(s, v).keys())
                obj_n += 1
                if ranked and ranked[0] == o:
                    obj_ok += 1
            if s in pc._model.subjects and v in pc._model.verbs:
                ranked_v = list(pc.rank_verbs(s).keys())
                verb_n += 1
                if ranked_v and ranked_v[0] == v:
                    verb_ok += 1
        except Exception:  # noqa: BLE001
            continue
    m.object_rank_n = obj_n
    m.verb_rank_n = verb_n
    if obj_n:
        m.object_rank_acc = obj_ok / obj_n
    if verb_n:
        m.verb_rank_acc = verb_ok / verb_n
    m.note = "hybrid surface+parse tuples (subject=agent fallback for imperatives)"
    return m


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _round_metrics(obj: Any) -> Any:
    """JSON-friendly rounding of nested metric structures."""
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _round_metrics(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_metrics(v) for v in obj]
    return obj


def write_report(run_dir: Path, metrics: dict[str, Any]) -> None:
    """Write ``report.md`` summarising the evaluation run."""
    lines: list[str] = [
        "# Neuro-symbolic corpus evaluation",
        "",
        f"Run directory: `{run_dir}`",
        f"Timestamp: {metrics.get('timestamp', '')}",
        "",
        "## Scope",
        "",
        "- BabyDS (`class1_train_100` / `class1_test_12`): PCWordModel, DS-VSS, DSPlausibilityPC",
        "- CHILDES (3200 / 800): PCWordModel only",
        "",
        "## BabyDS — PCWordModel",
        "",
    ]
    bd = metrics.get("babyds", {})
    pcw = bd.get("pc_word", {})
    lines.extend(
        [
            "| Split | Coverage | Mean NLL | Perplexity | Next-word top-1 |",
            "|-------|----------|----------|------------|-----------------|",
        ]
    )
    for split in ("train", "test"):
        m = pcw.get(split, {})
        lines.append(
            f"| {split} | {m.get('in_vocab_coverage', '—'):.3f} | "
            f"{_fmt(m.get('mean_nll'))} | {_fmt(m.get('perplexity'))} | "
            f"{_fmt(m.get('next_word_top1'))} |"
        )
    lines.extend(
        [
            "",
            f"max_len={pcw.get('max_len')}, vocab={pcw.get('vocab_size')}, "
            f"epochs={pcw.get('epochs')}, "
            f"train NLL { _fmt(pcw.get('nll_history_first')) } → {_fmt(pcw.get('nll_history_last'))}",
            "",
            "## BabyDS — DS-VSS",
            "",
        ]
    )
    vss = bd.get("vss", {})
    lines.extend(
        [
            f"Grammar: `{vss.get('grammar', '')}`",
            "",
            "| Split | OK rate | Mean final plausibility | No missing predicates |",
            "|-------|---------|-------------------------|-----------------------|",
        ]
    )
    for split in ("train", "test"):
        m = vss.get(split, {})
        lines.append(
            f"| {split} | {_fmt(m.get('ok_rate'))} | "
            f"{_fmt(m.get('mean_final_plausibility'))} | "
            f"{_fmt(m.get('no_missing_predicates_rate'))} |"
        )
    samples = vss.get("test", {}).get("samples", [])
    if samples:
        lines.extend(["", "### Sample trajectories (test)", ""])
        for s in samples:
            lines.append(
                f"- `{s.get('sentence')}` ok={s.get('ok')} "
                f"plaus={_fmt(s.get('final_plausibility'))} "
                f"missing={s.get('missing_predicates')}"
            )
            traj = " → ".join(
                f"{t.get('word')}:{_fmt(t.get('plausibility'))}"
                for t in s.get("trajectory", [])
            )
            lines.append(f"  - {traj}")
    dsp = bd.get("dsp", {})
    lines.extend(
        [
            "",
            "## BabyDS — DSPlausibilityPC",
            "",
            f"Note: {dsp.get('note', '')}",
            "",
            f"- Train tuple yield: {dsp.get('train_yield')} "
            f"(skip rate {_fmt(dsp.get('train_skip_rate'))})",
            f"- Test tuple yield: {dsp.get('test_yield')}",
            f"- Test mean NLL: {_fmt(dsp.get('test_nll'))}",
            f"- Object rank top-1: {_fmt(dsp.get('object_rank_acc'))} "
            f"(n={dsp.get('object_rank_n')})",
            f"- Verb rank top-1: {_fmt(dsp.get('verb_rank_acc'))} "
            f"(n={dsp.get('verb_rank_n')})",
            f"- Train NLL {_fmt(dsp.get('nll_history_first'))} → "
            f"{_fmt(dsp.get('nll_history_last'))}",
            "",
            "## CHILDES — PCWordModel",
            "",
        ]
    )
    ch = metrics.get("childes", {}).get("pc_word", {})
    lines.extend(
        [
            "| Split | Coverage | Truncate rate | Mean NLL | Perplexity | Next-word top-1 |",
            "|-------|----------|---------------|----------|------------|-----------------|",
        ]
    )
    for split in ("train", "test"):
        m = ch.get(split, {})
        lines.append(
            f"| {split} | {_fmt(m.get('in_vocab_coverage'))} | "
            f"{_fmt(m.get('truncate_rate'))} | {_fmt(m.get('mean_nll'))} | "
            f"{_fmt(m.get('perplexity'))} | {_fmt(m.get('next_word_top1'))} |"
        )
    lines.extend(
        [
            "",
            f"max_len={ch.get('max_len')}, vocab={ch.get('vocab_size')}, "
            f"epochs={ch.get('epochs')}, "
            f"train NLL {_fmt(ch.get('nll_history_first'))} → "
            f"{_fmt(ch.get('nll_history_last'))}",
            "",
            "## Caveats",
            "",
            "- BabyDS is a robot-command domain; English SVO tree extraction often "
            "fails, so DSPlausibilityPC uses a hybrid subject=`agent` fallback.",
            "- BabyDS DyLan parses use an induction-learnt lexicon "
            "(`dsttr-induction` on `configs/induction/presplit.yaml`) with "
            "hyp-filtered computational actions from `2025-seed-grammar`.",
            "- PCWordModel uses a closed train vocabulary; OOV sentences are "
            "excluded from NLL / accuracy (coverage is reported separately).",
            "- CHILDES was not run through DyLan / `parse_vss` in this pass.",
            "",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def resolve_babyds_induction_parser() -> tuple[InteractiveContextParser, str]:
    """Build an induction eval parser for BabyDS (learnt lexicon + filtered hyps).

    Prefers run dirs whose name contains ``babyds``; otherwise the newest
    ``out/runs/*/models/lexicon-top-1.txt``. Requires a prior
    ``dsttr-induction`` run on the BabyDS presplit config.
    """
    seed = ROOT / "resources" / "2025-seed-grammar"
    runs = ROOT / "out" / "runs"
    all_models = [
        p for p in runs.glob("*/models") if (p / "lexicon-top-1.txt").is_file()
    ]
    babyds = [p for p in all_models if "babyds" in p.parent.name.lower()]
    pool = babyds or all_models
    if not pool:
        raise FileNotFoundError(
            "No BabyDS induction model found under out/runs/*/models/. "
            "Run: uv run dsttr-induction -c configs/induction/presplit.yaml"
        )
    models = max(pool, key=lambda p: p.stat().st_mtime)
    parser = build_eval_parser(lexicon_dir=models, seed_grammar=seed, top_n=1)
    label = f"{models.parent.name}/models + 2025-seed-grammar (top-1, hyp-filtered)"
    return parser, label


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    """Run BabyDS (all models) and CHILDES (PCWordModel) evaluations."""
    t0 = time.perf_counter()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "out" / "runs" / f"nesy-corpus-eval-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print("Loading BabyDS…")
    bd_train = load_sentences(BABYDS_TRAIN)
    bd_test = load_sentences(BABYDS_TEST)
    print(f"  train={len(bd_train)} test={len(bd_test)}")

    # --- PCWordModel BabyDS ---
    bd_max_len = max((len(tokenize(s)) for s in bd_train), default=4)
    print(f"\n== BabyDS PCWordModel (max_len={bd_max_len}, epochs=20) ==")
    _, bd_tr, bd_te, bd_hist = train_pc_word(
        bd_train, bd_test, max_len=bd_max_len, epochs=20, seed=0
    )
    print(
        f"  train cov={bd_tr.in_vocab_coverage:.3f} nll={bd_tr.mean_nll} "
        f"ppl={bd_tr.perplexity} top1={bd_tr.next_word_top1}"
    )
    print(
        f"  test  cov={bd_te.in_vocab_coverage:.3f} nll={bd_te.mean_nll} "
        f"ppl={bd_te.perplexity} top1={bd_te.next_word_top1}"
    )

    # --- DS-VSS BabyDS ---
    parser, grammar_label = resolve_babyds_induction_parser()
    print(f"\n== BabyDS DS-VSS (grammar={grammar_label}) ==")
    lex = build_babyds_lexicon(bd_train)
    print(
        f"  lexicon entities={len(lex._entities)} "
        f"matrices={len(lex._matrices)} cubes={len(lex._cubes)}"
    )
    vss_train = evaluate_vss(bd_train, lex, grammar_label, parser=parser)
    vss_test = evaluate_vss(
        bd_test, lex, grammar_label, parser=parser, sample_n=5
    )
    print(f"  train ok_rate={vss_train.ok_rate:.3f} mean_plaus={vss_train.mean_final_plausibility}")
    print(f"  test  ok_rate={vss_test.ok_rate:.3f} mean_plaus={vss_test.mean_final_plausibility}")

    # --- DSPlausibilityPC BabyDS ---
    print("\n== BabyDS DSPlausibilityPC ==")
    dsp = evaluate_dsp(
        bd_train,
        bd_test,
        lex,
        grammar_label,
        num_bins=4,
        epochs=20,
        seed=0,
        parser=parser,
    )
    print(
        f"  yield train={dsp.train_yield} test={dsp.test_yield} "
        f"test_nll={dsp.test_nll} obj_acc={dsp.object_rank_acc} verb_acc={dsp.verb_rank_acc}"
    )

    # --- CHILDES PCWordModel ---
    print("\nLoading CHILDES…")
    ch_train = load_sentences(CHILDES_TRAIN)
    ch_test = load_sentences(CHILDES_TEST)
    print(f"  train={len(ch_train)} test={len(ch_test)}")
    ch_max_len = 8
    print(f"\n== CHILDES PCWordModel (max_len={ch_max_len}, epochs=10) ==")
    _, ch_tr, ch_te, ch_hist = train_pc_word(
        ch_train, ch_test, max_len=ch_max_len, epochs=10, seed=0
    )
    print(
        f"  train cov={ch_tr.in_vocab_coverage:.3f} trunc={ch_tr.truncate_rate:.3f} "
        f"nll={ch_tr.mean_nll} ppl={ch_tr.perplexity} top1={ch_tr.next_word_top1}"
    )
    print(
        f"  test  cov={ch_te.in_vocab_coverage:.3f} trunc={ch_te.truncate_rate:.3f} "
        f"nll={ch_te.mean_nll} ppl={ch_te.perplexity} top1={ch_te.next_word_top1}"
    )

    metrics: dict[str, Any] = {
        "timestamp": stamp,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "babyds": {
            "pc_word": {
                "max_len": bd_max_len,
                "vocab_size": bd_tr.vocab_size,
                "epochs": 20,
                "nll_history_first": bd_tr.train_nll_history_first,
                "nll_history_last": bd_tr.train_nll_history_last,
                "train": asdict(bd_tr),
                "test": asdict(bd_te),
            },
            "vss": {
                "grammar": grammar_label,
                "train": asdict(vss_train),
                "test": asdict(vss_test),
            },
            "dsp": asdict(dsp),
        },
        "childes": {
            "pc_word": {
                "max_len": ch_max_len,
                "vocab_size": ch_tr.vocab_size,
                "epochs": 10,
                "nll_history_first": ch_tr.train_nll_history_first,
                "nll_history_last": ch_tr.train_nll_history_last,
                "train": asdict(ch_tr),
                "test": asdict(ch_te),
            },
        },
    }
    metrics = _round_metrics(metrics)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    write_report(run_dir, metrics)
    print(f"\nWrote {run_dir / 'metrics.json'}")
    print(f"Wrote {run_dir / 'report.md'}")
    print(f"Elapsed {metrics['elapsed_sec']}s")


if __name__ == "__main__":
    main()
