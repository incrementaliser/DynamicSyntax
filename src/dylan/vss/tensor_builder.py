"""Optional verb-tensor construction from subject–object co-occurrence (Grefenstette–Sadrzadeh)."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import torch

from dylan.vss.embedding_store import EmbeddingStore


def build_verb_tensor_from_cooccurrence(
    store: EmbeddingStore,
    verb: str,
    pairs: list[tuple[str, str]],
) -> torch.Tensor:
    """Sum outer products S⊗O over attested subject–object pairs for *verb*."""
    dim = store.dims
    acc = torch.zeros(dim, dim)
    for subj, obj in pairs:
        try:
            s = store.get_noun(subj).reshape(-1)
            o = store.get_noun(obj).reshape(-1)
            n = min(s.numel(), o.numel(), dim)
            acc[:n, :n] += torch.outer(s[:n], o[:n])
        except KeyError:
            continue
    return acc


def load_cooccurrence_tsv(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Load verb -> [(subj, obj), ...] from a simple TSV (verb, subject, object)."""
    by_verb: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            v = row.get("verb", "").strip()
            s = row.get("subject", "").strip()
            o = row.get("object", "").strip()
            if v and s and o:
                by_verb[v].append((s, o))
    return dict(by_verb)
