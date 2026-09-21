"""YAML induction configuration for the train/evaluate pipeline."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml

SplitMode = Literal["holdout", "train_val_test", "kfold", "pre_split"]
LogLevelName = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
CorpusProfileName = Literal["babyds", "childes"]


@dataclass
class DataConfig:
    """Corpus path(s), split strategy, and induction corpus profile."""

    corpus: str | None = None
    train: str | None = None
    val: str | None = None
    test: str | None = None
    split: SplitMode = "holdout"
    profile: CorpusProfileName = "babyds"
    test_ratio: float = 0.15
    val_ratio: float = 0.0
    folds: int = 5
    seed: int = 47
    save_splits: bool = True


@dataclass
class ModelConfig:
    """Seed grammar and optional previously learned model for continue learning."""

    seed_grammar: str = "resources/2025-seed-grammar"
    use_previous_model: bool = False
    previous_model: str | None = None
    top_n: int = 3


@dataclass
class TrainConfig:
    """Training behaviour flags for the current output directory."""

    show_progress: bool = True
    force_train: bool = False
    reuse_existing_model: bool = False


@dataclass
class EvalConfig:
    """Which splits and top-N ranks to evaluate."""

    evaluate_on: list[str] = field(default_factory=lambda: ["train", "test"])
    top_n_start: int = 1
    top_n_end: int | None = None


@dataclass
class LoggingConfig:
    """Log level and sink toggles for the induction run."""

    level: LogLevelName = "INFO"
    to_cli: bool = True
    to_file: bool = True
    file_name: str = "run.log"


@dataclass
class OutputConfig:
    """Run directory naming and artifact writers."""

    run_dir: str = "out/runs"
    name: str = "induction"
    write_tsv: bool = True
    write_report: bool = True


@dataclass
class InductionConfig:
    """Full induction configuration loaded from YAML (+ CLI overrides)."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def resolved_top_n_end(self) -> int:
        """Return the inclusive top-N end rank for evaluation."""
        return self.eval.top_n_end if self.eval.top_n_end is not None else self.model.top_n

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict suitable for YAML dumping."""
        return asdict(self)


# Backward-compatible alias
ExperimentConfig = InductionConfig

_SECTION_TYPES: dict[str, type] = {
    "data": DataConfig,
    "model": ModelConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
    "logging": LoggingConfig,
    "output": OutputConfig,
}


def _merge_dataclass(cls: type, raw: dict[str, Any] | None) -> Any:
    """Build a dataclass instance from a partial dict, ignoring unknown keys."""
    if not raw:
        return cls()
    valid = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in valid}
    return cls(**filtered)


def config_from_dict(raw: dict[str, Any]) -> InductionConfig:
    """Build :class:`InductionConfig` from a nested mapping."""
    sections: dict[str, Any] = {}
    for name, cls in _SECTION_TYPES.items():
        sections[name] = _merge_dataclass(cls, raw.get(name))
    return InductionConfig(**sections)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict (empty dict if the file is empty)."""
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}")
    return data


def _parse_override_value(raw: str) -> Any:
    """Parse a CLI override value with YAML scalar rules."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` (dot-path) overrides onto a nested config dict."""
    result = copy.deepcopy(raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, value_str = item.split("=", 1)
        parts = key.split(".")
        if not parts or any(not p for p in parts):
            raise ValueError(f"Invalid override key: {key!r}")
        cursor: dict[str, Any] = result
        for part in parts[:-1]:
            nxt = cursor.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[part] = nxt
            cursor = nxt
        cursor[parts[-1]] = _parse_override_value(value_str)
    return result


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> InductionConfig:
    """Load YAML config from *path* and apply optional ``--set`` overrides."""
    raw: dict[str, Any] = load_yaml(path) if path is not None else {}
    if overrides:
        raw = apply_overrides(raw, overrides)
    return config_from_dict(raw)


def dump_config(config: InductionConfig, path: str | Path) -> None:
    """Write the resolved config to *path* as YAML."""
    Path(path).write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
