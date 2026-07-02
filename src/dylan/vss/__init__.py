"""DS-VSS: incremental Dynamic Syntax with vector space compositional semantics."""

from __future__ import annotations

from dylan.vss.compose_svo import compose_gs2013_pair, compose_svo, compose_svo_baseline
from dylan.vss.composition import cosine_distance, interpret_sentence
from dylan.vss.ds_vss_session import DSVSSParseResult, DSVSSSession, IncrementalVSSStep
from dylan.vss.embedding_store import (
    BundledWord2VecStore,
    DemoEmbeddingStore,
    EmbeddingStore,
    ShelveEmbeddingStore,
    build_demo_tensors_pt,
    embedding_store_from_config,
)
from dylan.vss.evaluate import evaluate_gs2013
from dylan.vss.experiment_run import (
    ExperimentRunConfig,
    ExperimentRunContext,
    build_analysis_report,
    format_accuracy_table,
    save_run_artifacts,
)
from dylan.vss.gs2013_data import convert_gs2013_to_ks_format, load_sentence_pairs
from dylan.vss.types import (
    CompositionMethod,
    EvaluationMode,
    GS2013EvaluationResult,
    GS2013Pair,
    GS2013Sentence,
    MethodAccuracy,
    SVORoles,
    TensorRep,
    UnderspecMethod,
    VSSConfig,
)

__all__ = [
    "BundledWord2VecStore",
    "CompositionMethod",
    "DSVSSParseResult",
    "DSVSSSession",
    "DemoEmbeddingStore",
    "EmbeddingStore",
    "EvaluationMode",
    "GS2013EvaluationResult",
    "GS2013Pair",
    "GS2013Sentence",
    "IncrementalVSSStep",
    "MethodAccuracy",
    "SVORoles",
    "ShelveEmbeddingStore",
    "TensorRep",
    "UnderspecMethod",
    "VSSConfig",
    "build_demo_tensors_pt",
    "compose_gs2013_pair",
    "compose_svo",
    "compose_svo_baseline",
    "convert_gs2013_to_ks_format",
    "cosine_distance",
    "embedding_store_from_config",
    "ExperimentRunConfig",
    "ExperimentRunContext",
    "build_analysis_report",
    "evaluate_gs2013",
    "format_accuracy_table",
    "save_run_artifacts",
    "interpret_sentence",
    "load_sentence_pairs",
]
