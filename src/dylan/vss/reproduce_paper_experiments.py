"""Reproduce Sec. 5 disambiguation experiments (Purver et al., GS2013 / word2vec W2V).

Run from the repo root::

    uv sync --extra vss
    uv run python -m dylan.vss.reproduce_paper_experiments

Artifacts (log, JSON, CSV, Markdown report, plots) default to
``src/dylan/vss/output/runs/<timestamp>/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from loguru import logger

from dylan.vss.embedding_store import (
    BundledWord2VecStore,
    ShelveEmbeddingStore,
    embedding_store_from_config,
)
from dylan.vss.evaluate import evaluate_gs2013
from dylan.vss.experiment_run import (
    ExperimentRunContext,
    _COMPOSITION_LABELS,
    _INCR_LABELS,
    _STAGE_LABELS,
    best_methods_summary,
    format_accuracy_table,
    log_evaluation_progress,
    save_run_artifacts,
)
from dylan.vss.gs2013_data import load_sentence_pairs
from dylan.vss.types import EvaluationMode, GS2013EvaluationResult, VSSConfig


def save_accuracy_plot(result: GS2013EvaluationResult, path: Path) -> None:
    """Save incremental accuracy curves when matplotlib is installed."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib required for --plot (pip install matplotlib)") from exc

    fig, ax = plt.subplots(figsize=(10, 5))
    for im_key, comp_grid in sorted(result.by_incremental.items()):
        for cm_key, stages in sorted(comp_grid.items()):
            label = f"{_INCR_LABELS.get(im_key, im_key)}/{_COMPOSITION_LABELS.get(cm_key, cm_key)}"
            y = [stages[i].accuracy for i in range(3)]
            ax.plot(_STAGE_LABELS, y, marker="o", label=label)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Incremental parse position")
    ax.set_ylabel("Disambiguation accuracy")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance")
    ax.set_title(f"GS2013 mean accuracy ({result.mode.value})")
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_worked_example(store: BundledWord2VecStore) -> str:
    """Cosine similarities for the incremental dribble example (paper Sec. 5)."""
    from dylan.vss.composition import cosine_distance
    from dylan.vss.compose_svo import compose_svo
    from dylan.vss.types import CompositionMethod, UnderspecMethod

    lines: list[str] = ["", "Worked example (paper Sec. 5):", ""]
    examples = [
        ("footballer", "dribble", "ball", "control"),
        ("footballer", "dribble", "ball", "drip"),
        ("baby", "dribble", "milk", "drip"),
        ("baby", "dribble", "milk", "control"),
    ]
    try:
        amb = compose_svo(store, "footballer", "dribble", "ball", underspec=UnderspecMethod.identity)
        for subj, verb, obj, paraphrase in examples:
            para = compose_svo(store, subj, paraphrase, obj, underspec=UnderspecMethod.identity)
            for stage in (0, 1, 2):
                rep_amb = amb.stages[stage][CompositionMethod.gs]
                rep_para = para.stages[stage][CompositionMethod.gs]
                dist = cosine_distance(rep_amb.flatten(), rep_para.flatten())
                lines.append(
                    f"  cos({subj} {verb} {obj} vs {subj} {paraphrase} {obj}) "
                    f"stage {_STAGE_LABELS[stage]}: {1.0 - dist:.4f}"
                )
    except KeyError as exc:
        lines.append(f"  (skipped: {exc})")
    return "\n".join(lines)


def build_store(args: argparse.Namespace) -> object:
    """Construct embedding store from CLI paths or bundled defaults."""
    if args.vector_shelve and args.tensor_shelve:
        return ShelveEmbeddingStore(args.vector_shelve, args.tensor_shelve, dims=args.dims)
    cfg = VSSConfig(dims=args.dims)
    if args.vector_shelve or args.tensor_shelve:
        cfg = VSSConfig(
            dims=args.dims,
            vector_shelve_path=args.vector_shelve,
            tensor_shelve_path=args.tensor_shelve,
        )
    return embedding_store_from_config(cfg)


def _cli_args_dict(args: argparse.Namespace) -> dict[str, Any]:
    """Serialize argparse namespace for the run report."""
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


def main(argv: list[str] | None = None) -> int:
    """Entry point: load embeddings, run GS2013 eval, log and save analysis artifacts."""
    parser = argparse.ArgumentParser(
        description="Reproduce incremental disambiguation experiments (Purver et al., GS2013, W2V).",
    )
    parser.add_argument(
        "--mode",
        choices=("tensor_only", "ds_vss", "both"),
        default="tensor_only",
        help="tensor_only matches jolli; ds_vss uses DS parser roles (slower).",
    )
    parser.add_argument("--dims", type=int, default=300, help="Embedding dimensionality (paper: 300).")
    parser.add_argument("--vector-shelve", type=Path, default=None, help="Optional vector shelve path.")
    parser.add_argument("--tensor-shelve", type=Path, default=None, help="Optional tensor shelve path.")
    parser.add_argument("--grammar", type=Path, default=None, help="Grammar dir for ds_vss mode.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run output root (default: src/dylan/vss/output/runs/<timestamp>).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write JSON/CSV/report files (logging to stderr only).",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Loguru level for stderr and run.log.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save accuracy plots under the run plots/ directory (requires matplotlib).",
    )
    parser.add_argument("--worked-example", action="store_true", help="Include Sec. 5 dribble example in report.")
    parser.add_argument("--max-pairs", type=int, default=None, help="Limit pairs (smoke runs).")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Log evaluation progress every N pairs (0 disables).",
    )
    args = parser.parse_args(argv)

    ctx: ExperimentRunContext | None = None
    if not args.no_save:
        ctx = ExperimentRunContext.create(args.output_dir, log_level=args.log_level)
        ctx.setup_logging()
    else:
        from dylan.logging_config import configure_logging

        configure_logging(level=args.log_level)

    log = logger.bind(component="reproduce_paper_experiments")
    log.info("Purver et al. - Incremental Composition in Distributional Semantics")
    log.info("Dataset: GS2013 | Embeddings: bundled word2vec W2V (300-dim)")

    t0 = time.perf_counter()
    log.info("Loading embedding store (noun zip is large; may take several minutes)...")
    store = build_store(args)
    load_sec = time.perf_counter() - t0
    if ctx is not None:
        ctx.record_timing("load_embeddings", load_sec)
    log.info("Embeddings ready: dims={} elapsed={:.1f}s", store.dims, load_sec)
    if isinstance(store, BundledWord2VecStore):
        log.info("Verb tensors loaded: {}", len(store._verb_tensors))

    pairs = load_sentence_pairs()
    if args.max_pairs is not None:
        log.info("Using first {} pairs (--max-pairs)", args.max_pairs)
    else:
        log.info("Loaded {} sentence pairs", len(pairs))

    modes: list[EvaluationMode] = []
    if args.mode in ("tensor_only", "both"):
        modes.append(EvaluationMode.tensor_only)
    if args.mode in ("ds_vss", "both"):
        modes.append(EvaluationMode.ds_vss)

    all_results: dict[str, GS2013EvaluationResult] = {}
    extra_sections: list[str] = []
    subset = pairs[: args.max_pairs] if args.max_pairs is not None else None
    progress_cb = None
    if args.progress_every > 0:
        progress_cb = lambda i, total, skipped: log_evaluation_progress(
            i, total, skipped=skipped, every=args.progress_every
        )

    for mode in modes:
        log.info("Running evaluation: {}", mode.value)
        t1 = time.perf_counter()
        result = evaluate_gs2013(
            store,
            VSSConfig(dims=getattr(store, "dims", args.dims), grammar_path=args.grammar),
            mode=mode,
            pairs=subset,
            progress_callback=progress_cb,
            progress_every=args.progress_every,
        )
        eval_sec = time.perf_counter() - t1
        if ctx is not None:
            ctx.record_timing(f"evaluate_{mode.value}", eval_sec)
        all_results[mode.value] = result
        log.info("Evaluation {} finished in {:.1f}s", mode.value, eval_sec)

        print(format_accuracy_table(result))
        print(best_methods_summary(result))

        if args.plot:
            plot_path = (
                ctx.plots_path_for(mode.value)
                if ctx is not None
                else Path(f"gs2013_accuracy_{mode.value}.png")
            )
            try:
                save_accuracy_plot(result, plot_path)
                log.info("Plot saved: {}", plot_path)
            except ImportError as exc:
                log.warning("Plot skipped: {}", exc)

    if args.worked_example and isinstance(store, BundledWord2VecStore):
        section = run_worked_example(store)
        print(section)
        extra_sections.append(section)

    if ctx is not None and not args.no_save:
        save_run_artifacts(
            ctx,
            all_results,
            extra_report_sections=extra_sections or None,
            cli_args=_cli_args_dict(args),
        )
        log.info("Run artifacts directory: {}", ctx.config.output_dir)
        log.info("  log: {}", ctx.log_path)
        log.info("  report: {}", ctx.report_path)
        log.info("  json: {}", ctx.json_path)
        log.info("  csv: {}", ctx.csv_path)
        ctx.teardown_logging()

    log.info("Done. Stage S should be ~0.50 (chance). Compare tensor_only to jolli.py / paper Sec. 5.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
