"""Circuit (SPL) induction on the AA CHILDES holdout set.

Reuses the same train/test split and EM lexicons from
``out/runs/20260717-122528_induction-childes-aa396`` for a fair comparison,
then trains :class:`CircuitWordLearner` under the ``childes`` profile with the
2023 seed grammar.

Run::

    uv sync --extra nesy --extra pc
    uv run python scripts/nesy_induction_aa.py
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from dylan.induction.corpus_profile import set_active_profile
from dylan.induction.em_learner.record_type_corpus import RecordTypeCorpus
from dylan.induction.pipeline.evaluate import build_eval_parser, evaluate_corpus
from dylan.nesy.evaluate import metrics_to_dict
from dylan.nesy.learner import CircuitWordLearner

ROOT = Path(__file__).resolve().parents[1]
EM_RUN = ROOT / "out/runs/20260717-122528_induction-childes-aa396"
TRAIN = EM_RUN / "data/train.txt"
TEST = EM_RUN / "data/test.txt"
EM_MODELS = EM_RUN / "models"
SEED = ROOT / "resources/2023-english-ttr-induction-seed"


def _eval_models(models_dir: Path, seed: Path, top_ns: tuple[int, ...] = (1, 2, 3)) -> dict:
    """Score train/test for each top-N with a learnt lexicon directory."""
    train_c = RecordTypeCorpus(corpus_path=TRAIN)
    test_c = RecordTypeCorpus(corpus_path=TEST)
    out: dict = {}
    for top_n in top_ns:
        parser = build_eval_parser(
            lexicon_dir=models_dir, seed_grammar=seed, top_n=top_n
        )
        out[f"top_{top_n}"] = {
            "train": metrics_to_dict(evaluate_corpus(parser, train_c)),
            "test": metrics_to_dict(evaluate_corpus(parser, test_c)),
        }
    return out


def run_em_baseline(run_dir: Path) -> dict:
    """Copy AA EM lexicons into the run dir and re-evaluate (no retrain)."""
    models = run_dir / "em_models"
    models.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for f in EM_MODELS.glob("lexicon-top-*.txt"):
        shutil.copy(f, models / f.name)
    metrics = _eval_models(models, SEED)
    return {
        "reused_existing": True,
        "source_run": str(EM_RUN),
        "train_time_s": round(time.perf_counter() - t0, 3),
        "models_dir": str(models),
        "metrics": metrics,
    }


def run_circuit(run_dir: Path, em_models: Path, top_n: int = 3) -> dict:
    """Train CircuitWordLearner on AA train and evaluate train/test."""
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
    metrics = _eval_models(models, SEED)
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


def _row(metrics: dict, split: str, n: int) -> str:
    """Format one markdown table row for a split/top-N."""
    m = metrics[f"top_{n}"][split]
    return (
        f"| {n} | {m['coverage']:.2f} | {m['f1']:.2f} | {m['exact_match']:.2f} |"
    )


def write_report(run_dir: Path, payload: dict) -> None:
    """Write a short markdown EM vs circuit comparison report."""
    em = payload["em"]
    cir = payload["circuit"]
    lines = [
        "# AA CHILDES SPL induction (circuit vs EM)",
        "",
        f"Run: `{run_dir}`",
        f"Timestamp: {payload['timestamp']}",
        f"Profile: `{payload['profile']}`",
        f"Seed: `{payload['seed_grammar']}`",
        f"Train/test: `{TRAIN}` / `{TEST}` (same holdout as EM AA run)",
        "",
        "## EM baseline (reused)",
        "",
        f"- Source: `{em['source_run']}`",
        f"- Re-eval wall time: {em['train_time_s']}s",
        "",
        "### Test",
        "",
        "| Top-N | Cov | F1 | EM |",
        "|-------|-----|----|----|",
    ]
    for n in (1, 2, 3):
        lines.append(_row(em["metrics"], "test", n))
    lines.extend(
        [
            "",
            "### Train",
            "",
            "| Top-N | Cov | F1 | EM |",
            "|-------|-----|----|----|",
        ]
    )
    for n in (1, 2, 3):
        lines.append(_row(em["metrics"], "train", n))
    lines.extend(
        [
            "",
            "## Circuit (SPL) learner",
            "",
            f"- Wall time: {cir['train_time_s']}s (circuit fit {cir['circuit_train_time_s']}s)",
            f"- Parameters: {cir['n_parameters']} (hyp catalog entries {cir['n_hyp_entries']})",
            f"- Encoded examples: {cir['n_encoded']} (skipped {cir['n_skipped']})",
            "",
            "### Test",
            "",
            "| Top-N | Cov | F1 | EM |",
            "|-------|-----|----|----|",
        ]
    )
    for n in (1, 2, 3):
        lines.append(_row(cir["metrics"], "test", n))
    lines.extend(
        [
            "",
            "### Train",
            "",
            "| Top-N | Cov | F1 | EM |",
            "|-------|-----|----|----|",
        ]
    )
    for n in (1, 2, 3):
        lines.append(_row(cir["metrics"], "train", n))
    lines.append("")
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run EM re-eval vs circuit induction on the AA CHILDES holdout."""
    if not TRAIN.is_file() or not TEST.is_file():
        raise FileNotFoundError(
            f"AA holdout splits missing under {EM_RUN}; run dsttr-induction first."
        )
    if not (EM_MODELS / "lexicon-top-1.txt").is_file():
        raise FileNotFoundError(f"AA EM lexicons missing under {EM_MODELS}")
    if not SEED.is_dir():
        raise FileNotFoundError(f"Seed grammar missing: {SEED}")

    set_active_profile("childes")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "out" / "runs" / f"nesy-induction-aa-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {run_dir}")
    print("Profile: childes")
    print(f"Train={TRAIN} Test={TEST}")

    print("== EM baseline (reuse AA lexicons) ==")
    em = run_em_baseline(run_dir)
    print(
        f"  test top1 Cov={em['metrics']['top_1']['test']['coverage']:.2f} "
        f"F1={em['metrics']['top_1']['test']['f1']:.2f} "
        f"top3 Cov={em['metrics']['top_3']['test']['coverage']:.2f}"
    )

    print("== Circuit learner ==")
    cir = run_circuit(run_dir, Path(em["models_dir"]))
    print(
        f"  test top1 Cov={cir['metrics']['top_1']['test']['coverage']:.2f} "
        f"F1={cir['metrics']['top_1']['test']['f1']:.2f} "
        f"top3 Cov={cir['metrics']['top_3']['test']['coverage']:.2f} "
        f"params={cir['n_parameters']} encoded={cir['n_encoded']} skipped={cir['n_skipped']}"
    )

    payload = {
        "timestamp": stamp,
        "profile": "childes",
        "seed_grammar": str(SEED),
        "train": str(TRAIN),
        "test": str(TEST),
        "em": em,
        "circuit": cir,
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(run_dir, payload)
    print(f"Wrote {run_dir / 'metrics.json'}")
    print(f"Wrote {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
