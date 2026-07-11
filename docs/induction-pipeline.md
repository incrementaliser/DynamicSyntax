# Induction pipeline manual

Train and evaluate TTR induction with a YAML config: split a corpus, run EM learning, score selected splits, and write a report.

## Quick start

```bash
# From the repo root
uv run dsttr-induction --config configs/induction/holdout.yaml

# Equivalent script entry
uv run python scripts/dsttr_induction.py --config configs/induction/holdout.yaml

# Override fields without editing YAML
uv run dsttr-induction -c configs/induction/holdout.yaml --set data.seed=48 --set logging.level=DEBUG

# Browse the report in a Textual TUI after the run
uv run dsttr-induction -c configs/induction/holdout.yaml --report-tui
```

Example configs live under [`configs/induction/`](../configs/induction/):

| File | Split mode |
|------|------------|
| `holdout.yaml` | Shuffle + hold out `test_ratio` |
| `train-val-test.yaml` | Train / val / test from ratios |
| `kfold.yaml` | Disjoint k-fold CV |
| `presplit.yaml` | Explicit train/val/test files |

## Config sections

### `data`

- `corpus` — single file to split (`holdout`, `train_val_test`, `kfold`)
- `train` / `val` / `test` — paths for `split: pre_split`
- `split` — `holdout` | `train_val_test` | `kfold` | `pre_split`
- `test_ratio` — fraction held out as test (default `0.15`); train is the remainder
- `val_ratio` — validation fraction for `train_val_test` (default `0.0`)
- `folds`, `seed`, `save_splits`

### `model`

- `seed_grammar` — directory with `computational-actions.txt` (read in place; not copied)
- `use_previous_model` — if `true`, continue learning from `previous_model`
- `previous_model` — path to another run/model dir with `lexicon-top-N.txt` (ignored unless `use_previous_model`)
- `top_n` — lexicon ranks to save / evaluate up to

### `train`

- `show_progress`
- `force_train` — always train and overwrite lexicons in the **current** output dir (no warning). Wins over `reuse_existing_model`. Can still seed from `previous_model` elsewhere.
- `reuse_existing_model` — if `true` and current-dir lexicons exist and `force_train` is false, skip EM and reuse them for eval

### `eval`

- `evaluate_on` — e.g. `[test]`, `[val, test]`, or `[train, test]` (omit `train` to skip train-set eval)
- `top_n_start`, `top_n_end` (`null` → `model.top_n`)

### `logging`

- `level` — `DEBUG` | `INFO` | `WARNING` | `ERROR` (loguru + stdlib intercept)
- `to_cli` / `to_file` / `file_name`

### `output`

- `run_dir`, `name` — creates `run_dir/<timestamp>_<name>/`
- `write_tsv`, `write_report`

## What to expect

Each run creates a directory like `out/runs/20260710-161500_induction-holdout/` containing:

| Artifact | Contents |
|----------|----------|
| `config.yaml` | Resolved config (after `--set`) |
| `lexicon-top-N.txt` | Learnt lexicons |
| `splits/` | Saved train/test/(val) corpora (if enabled) |
| `metrics.tsv` | Tab-separated P/R/F1/coverage/EM |
| `report.txt` | Score tables, timing (`HH-MM-SS`), config, metadata |
| `run.log` | File log (if `logging.to_file`) |

For `kfold`, each fold is under `fold_0/`, … with its own `metrics.tsv` and `report.txt`; the top-level files average fold scores.

The console prints a Rich report (timing, scores, config, metadata). Use `--report-tui` for an interactive Textual view (`q` to quit).

## Library API

```python
from dylan.induction.pipeline import load_config, run_induction

config = load_config("configs/induction/holdout.yaml", overrides=["data.seed=1"])
result = run_induction(config)
print(result.get(1, "test"))
```
