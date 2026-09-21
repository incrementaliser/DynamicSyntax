"""BabyDS MVP: circuit (SPL) induction vs EM baseline.

Reuses the same hypothesise → lexicon-top-N → maximal-mapping eval path as
``dsttr-induction``, but replaces local EM with :class:`CircuitWordLearner`.

Run::

    uv sync --extra nesy --extra pc
    uv run python scripts/nesy_induction_mvp.py
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from dylan.induction.em_learner.record_type_corpus import RecordTypeCorpus
from dylan.induction.em_learner.ttr_word_learner import TTRWordLearner
from dylan.induction.pipeline.evaluate import evaluate_corpus, build_eval_parser
from dylan.nesy.evaluate import metrics_to_dict
from dylan.nesy.learner import CircuitWordLearner

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/BabyDS/class1_train_100.txt"
TEST = ROOT / "data/BabyDS/class1_test_12.txt"
SEED = ROOT / "resources/2025-seed-grammar"


def _eval_models(models_dir: Path, top_ns: tuple[int, ...] = (1, 2, 3)) -> dict:
    """Score train/test for each top-N with a learnt lexicon directory."""
    train_c = RecordTypeCorpus(corpus_path=TRAIN)
    test_c = RecordTypeCorpus(corpus_path=TEST)
    out: dict = {}
    for top_n in top_ns:
        parser = build_eval_parser(
            lexicon_dir=models_dir, seed_grammar=SEED, top_n=top_n
        )
        out[f"top_{top_n}"] = {
            "train": metrics_to_dict(evaluate_corpus(parser, train_c)),
            "test": metrics_to_dict(evaluate_corpus(parser, test_c)),
        }
    return out


def run_em_baseline(run_dir: Path, top_n: int = 3) -> dict:
    """Train EM lexicon (or reuse existing BabyDS induction models)."""
    # Prefer a fresh EM run under this run_dir for a fair wall-clock comparison.
    models = run_dir / "em_models"
    models.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        (ROOT / "out/runs").glob("*babyds*/models/lexicon-top-1.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    t0 = time.perf_counter()
    if existing:
        src = existing[0].parent
        for f in src.glob("lexicon-top-*.txt"):
            shutil.copy(f, models / f.name)
        train_s = 0.0
        reused = True
    else:
        corpus = RecordTypeCorpus(corpus_path=TRAIN)
        learner = TTRWordLearner(
            seed_resource_dir=SEED,
            corpus=corpus,
            learner_comp_actions_path=SEED,
            top_n=top_n,
        )
        learner.learn(show_progress=True)
        learner.save_model(models / "lexicon", top_n)
        train_s = time.perf_counter() - t0
        reused = False
    metrics = _eval_models(models)
    return {
        "reused_existing": reused,
        "train_time_s": round(train_s if not reused else time.perf_counter() - t0, 3),
        "models_dir": str(models),
        "metrics": metrics,
    }


def run_circuit(run_dir: Path, em_models: Path, top_n: int = 3) -> dict:
    """Train CircuitWordLearner and evaluate."""
    models = run_dir / "circuit_models"
    models.mkdir(parents=True, exist_ok=True)
    corpus = RecordTypeCorpus(corpus_path=TRAIN)
    learner = CircuitWordLearner(
        seed_resource_dir=SEED,
        corpus=corpus,
        learner_comp_actions_path=SEED,
        top_n=top_n,
        epochs=15,
        lr=0.01,
        use_vss_gate=True,
        seed=0,
    )
    t0 = time.perf_counter()
    learner.learn(show_progress=True)
    learner.finish_training(models_dir=em_models, seed_grammar=SEED)
    learner.save_model(models / "lexicon", top_n)
    train_s = time.perf_counter() - t0
    metrics = _eval_models(models)
    return {
        "train_time_s": round(train_s, 3),
        "circuit_train_time_s": round(learner.train_time_s, 3),
        "n_parameters": learner.n_parameters,
        "n_hyp_entries": learner.n_hyp_entries,
        "models_dir": str(models),
        "metrics": metrics,
        "n_encoded": len(learner._encoded),
        "n_skipped": len(learner.skipped),
    }


def write_report(run_dir: Path, payload: dict) -> None:
    """Write a short markdown comparison report."""
    em = payload["em"]
    cir = payload["circuit"]
    lines = [
        "# BabyDS SPL induction MVP",
        "",
        f"Run: `{run_dir}`",
        f"Timestamp: {payload['timestamp']}",
        "",
        "## EM baseline",
        "",
        f"- Reused existing model: {em['reused_existing']}",
        f"- Wall time: {em['train_time_s']}s",
        "",
        "| Top-N | test Cov | test F1 | test EM |",
        "|-------|----------|---------|---------|",
    ]
    for n in (1, 2, 3):
        m = em["metrics"][f"top_{n}"]["test"]
        lines.append(
            f"| {n} | {m['coverage']:.2f} | {m['f1']:.2f} | {m['exact_match']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Circuit (SPL) learner",
            "",
            f"- Wall time: {cir['train_time_s']}s (circuit fit {cir['circuit_train_time_s']}s)",
            f"- Parameters: {cir['n_parameters']} (hyp catalog entries {cir['n_hyp_entries']})",
            f"- Encoded examples: {cir['n_encoded']} (skipped {cir['n_skipped']})",
            "",
            "| Top-N | test Cov | test F1 | test EM |",
            "|-------|----------|---------|---------|",
        ]
    )
    for n in (1, 2, 3):
        m = cir["metrics"][f"top_{n}"]["test"]
        lines.append(
            f"| {n} | {m['coverage']:.2f} | {m['f1']:.2f} | {m['exact_match']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Circuit support = legal `TTRHypothesiser` sequences (hard constraint).",
            "- Hypotheses are accumulated with EM bookkeeping; circuit MLE (warm-started",
            "  from EM log-probs) refines `p(hyp|word)`, gated by predicate-aligned DS-VSS.",
            "- Full shared LoFT CNF→cirkit compilation remains future work;",
            "  see `docs/ds-ttr-pc-design-note.md`.",
            "",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run EM vs circuit BabyDS induction comparison."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "out" / "runs" / f"nesy-induction-mvp-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")

    print("== EM baseline ==")
    em = run_em_baseline(run_dir)
    print(
        f"  test top1 F1={em['metrics']['top_1']['test']['f1']:.2f} "
        f"cov={em['metrics']['top_1']['test']['coverage']:.2f}"
    )

    print("== Circuit learner ==")
    cir = run_circuit(run_dir, Path(em["models_dir"]))
    print(
        f"  test top1 F1={cir['metrics']['top_1']['test']['f1']:.2f} "
        f"cov={cir['metrics']['top_1']['test']['coverage']:.2f} "
        f"top3 cov={cir['metrics']['top_3']['test']['coverage']:.2f} "
        f"params={cir['n_parameters']}"
    )

    payload = {"timestamp": stamp, "em": em, "circuit": cir}
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(run_dir, payload)
    print(f"Wrote {run_dir / 'metrics.json'}")
    print(f"Wrote {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
