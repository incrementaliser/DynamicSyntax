"""Embedding backends: bundled word2vec text, demo tensors, and external shelve spaces."""

from __future__ import annotations

import os
import shelve
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import torch

from dylan.vss.morph import lookup_keys
from dylan.vss.tensor_utils import to_dense_tensor
from dylan.vss.types import NOUN_SUFFIX, VERB_SUFFIX, VSSConfig

_VSS_DIR = Path(__file__).resolve().parent
_DEFAULT_W2V_ZIP = _VSS_DIR / "word2vec_trained.zip"
_DEFAULT_TENSOR_ZIP = _VSS_DIR / "word2vec_tensors_trained.zip"
_DEMO_TENSORS = _VSS_DIR / "embeddings" / "demo_tensors.pt"


class EmbeddingStore(ABC):
    """Abstract noun vectors and verb tensors for compositional semantics."""

    @abstractmethod
    def get_noun(self, word: str, *, dims: int | None = None) -> torch.Tensor:
        """Return a noun vector for *word* (column or 1-D)."""

    @abstractmethod
    def get_verb_tensor(self, word: str) -> torch.Tensor:
        """Return an n×n verb tensor for *word*."""

    @abstractmethod
    def get_verb_vector(self, word: str, *, dims: int | None = None) -> torch.Tensor:
        """Return a verb word vector (for additive baseline)."""

    @property
    @abstractmethod
    def dims(self) -> int:
        """Embedding dimensionality."""


class BundledWord2VecStore(EmbeddingStore):
    """Load noun vectors and verb tensors from bundled word2vec zip archives."""

    def __init__(
        self,
        zip_path: Path | None = None,
        *,
        tensor_zip_path: Path | None = None,
        demo_tensors_path: Path | None = None,
        dims: int = 300,
        load_vectors: bool = True,
        load_tensors: bool = True,
    ) -> None:
        """Parse vectors from *zip_path* and verb matrices from *tensor_zip_path* by default."""
        self._zip_path = zip_path or _DEFAULT_W2V_ZIP
        self._tensor_zip_path = tensor_zip_path or _DEFAULT_TENSOR_ZIP
        self._demo_path = demo_tensors_path or _DEMO_TENSORS
        self._dims = dims
        self._vectors: dict[str, torch.Tensor] = {}
        self._verb_tensors: dict[str, torch.Tensor] = {}
        if load_vectors:
            self._load_word2vec()
        if load_tensors:
            self._load_verb_tensors()
        if not self._verb_tensors:
            self._load_demo_tensors()

    def _load_word2vec(self) -> None:
        """Parse ``word2vec_trained.txt`` from the zip archive."""
        if not self._zip_path.is_file():
            return
        with zipfile.ZipFile(self._zip_path) as zf:
            names = [n for n in zf.namelist() if n.endswith(".txt")]
            if not names:
                return
            with zf.open(names[0]) as raw:
                for line in raw:
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text:
                        continue
                    parts = text.split()
                    if len(parts) < 2:
                        continue
                    word = parts[0]
                    vec = torch.tensor([float(x) for x in parts[1 : 1 + self._dims]], dtype=torch.float32)
                    self._vectors[word] = vec
                    self._vectors[word + NOUN_SUFFIX] = vec
                    self._vectors[word + VERB_SUFFIX] = vec

    def _load_verb_tensors(self) -> None:
        """Parse ``word2vec_tensors_trained.txt`` (verb name + 300×300 floats) from the tensor zip."""
        if not self._tensor_zip_path.is_file():
            return
        with zipfile.ZipFile(self._tensor_zip_path) as zf:
            names = [n for n in zf.namelist() if n.endswith(".txt")]
            if not names:
                return
            with zf.open(names[0]) as raw:
                for line in raw:
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text:
                        continue
                    parts = text.split()
                    if len(parts) < 2:
                        continue
                    word = parts[0]
                    n = self._dims
                    expected = n * n
                    floats = parts[1:]
                    if len(floats) < expected:
                        continue
                    flat = torch.tensor(
                        [float(x) for x in floats[:expected]],
                        dtype=torch.float32,
                    )
                    self._verb_tensors[word] = flat.reshape(n, n)

    def _load_demo_tensors(self) -> None:
        """Fallback verb tensors from ``demo_tensors.pt`` when the tensor zip is absent."""
        if not self._demo_path.is_file():
            return
        data = torch.load(self._demo_path, map_location="cpu", weights_only=True)
        verbs: dict[str, torch.Tensor] = data.get("verb_tensors", {})
        for key, val in verbs.items():
            if key not in self._verb_tensors:
                self._verb_tensors[key] = val.float()

    def _lookup_vector(self, word: str, *, noun: bool, dims: int | None) -> torch.Tensor:
        """Try morphological key variants."""
        d = dims or self._dims
        for key in lookup_keys(word, noun=noun):
            if key in self._vectors:
                return self._vectors[key][:d]
        raise KeyError(f"no embedding for {word!r}")

    def get_noun(self, word: str, *, dims: int | None = None) -> torch.Tensor:
        """Return a noun vector for *word*."""
        return self._lookup_vector(word, noun=True, dims=dims)

    def get_verb_tensor(self, word: str) -> torch.Tensor:
        """Return the bundled verb tensor, trying morphological key variants."""
        for key in (word, *lookup_keys(word, noun=False)):
            if key in self._verb_tensors:
                t = self._verb_tensors[key]
                if t.dim() == 2:
                    return t
                n = int(t.numel() ** 0.5)
                return t.reshape(n, n)
        if not self._vectors:
            self._load_word2vec()
        try:
            v = self.get_verb_vector(word)
        except KeyError:
            raise KeyError(f"no verb tensor for {word!r}") from None
        n = v.numel()
        return torch.outer(v, v)

    def get_verb_vector(self, word: str, *, dims: int | None = None) -> torch.Tensor:
        """Return verb vector (word2vec #VB key or bare word)."""
        return self._lookup_vector(word, noun=False, dims=dims)

    @property
    def dims(self) -> int:
        """Configured vector dimensionality."""
        return self._dims


class ShelveEmbeddingStore(EmbeddingStore):
    """Load jolli-style shelve vector and tensor spaces."""

    def __init__(
        self,
        vector_path: Path | str,
        tensor_path: Path | str,
        *,
        dims: int | None = None,
    ) -> None:
        """Open shelve databases at *vector_path* and *tensor_path*."""
        self._vector_db = shelve.open(str(vector_path), flag="r")
        self._tensor_db = shelve.open(str(tensor_path), flag="r")
        self._dims = dims
        if self._dims is None:
            self._dims = self._infer_dims()

    def _infer_dims(self) -> int:
        """Guess dimensionality from the first shelve entry."""
        for _key, val in self._vector_db.items():
            t = to_dense_tensor(val)
            return int(t.reshape(-1).numel())
        return 300

    def _get_from_space(self, space: shelve.Shelf, word: str, *, noun: bool, dims: int | None) -> torch.Tensor:
        """Lookup with morph fallbacks."""
        d = dims or self._dims or 300
        for key in lookup_keys(word, noun=noun):
            if key in space:
                return to_dense_tensor(space[key], dims=d)
        raise KeyError(f"no shelve embedding for {word!r}")

    def get_noun(self, word: str, *, dims: int | None = None) -> torch.Tensor:
        """Return noun vector from vector shelve."""
        return self._get_from_space(self._vector_db, word, noun=True, dims=dims)

    def get_verb_tensor(self, word: str) -> torch.Tensor:
        """Return verb tensor from tensor shelve."""
        if word not in self._tensor_db:
            raise KeyError(f"no verb tensor for {word!r}")
        t = to_dense_tensor(self._tensor_db[word])
        if t.dim() == 1:
            n = int(t.numel() ** 0.5)
            return t.reshape(n, n)
        return t

    def get_verb_vector(self, word: str, *, dims: int | None = None) -> torch.Tensor:
        """Return verb vector from vector shelve."""
        return self._get_from_space(self._vector_db, word, noun=False, dims=dims)

    @property
    def dims(self) -> int:
        """Embedding dimensionality."""
        return int(self._dims or 300)

    def close(self) -> None:
        """Close shelve databases."""
        self._vector_db.close()
        self._tensor_db.close()


class DemoEmbeddingStore(EmbeddingStore):
    """Tiny fixed embeddings for unit tests (4-dimensional)."""

    def __init__(self) -> None:
        """Build deterministic demo vectors and verb matrices."""
        self._nouns = {
            "alpha": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "beta": torch.tensor([0.0, 1.0, 0.0, 0.0]),
            "gamma": torch.tensor([0.0, 0.0, 1.0, 0.0]),
        }
        self._verbs = {
            "like": torch.eye(4),
            "draw": torch.eye(4) * 0.9 + torch.ones(4, 4) * 0.025,
        }
        self._verb_vecs = {
            "like": torch.tensor([0.5, 0.5, 0.0, 0.0]),
            "draw": torch.tensor([0.0, 0.5, 0.5, 0.0]),
        }

    def get_noun(self, word: str, *, dims: int | None = None) -> torch.Tensor:
        """Return demo noun vector."""
        if word not in self._nouns:
            raise KeyError(word)
        return self._nouns[word]

    def get_verb_tensor(self, word: str) -> torch.Tensor:
        """Return demo verb matrix."""
        if word not in self._verbs:
            raise KeyError(word)
        return self._verbs[word]

    def get_verb_vector(self, word: str, *, dims: int | None = None) -> torch.Tensor:
        """Return demo verb vector."""
        if word not in self._verb_vecs:
            raise KeyError(word)
        return self._verb_vecs[word]

    @property
    def dims(self) -> int:
        """Demo dimensionality."""
        return 4


def build_demo_tensors_pt(path: Path | None = None, *, verbs: list[str] | None = None) -> Path:
    """Write ``demo_tensors.pt`` with verb matrices from the bundled tensor zip when possible."""
    out = path or _DEMO_TENSORS
    out.parent.mkdir(parents=True, exist_ok=True)
    store = BundledWord2VecStore(load_vectors=False)
    verb_tensors: dict[str, torch.Tensor] = {}
    for verb in verbs or ("draw", "meet", "run", "provide", "accept", "depict", "attract", "visit", "satisfy"):
        try:
            verb_tensors[verb] = store.get_verb_tensor(verb)
        except KeyError:
            continue
    torch.save({"verb_tensors": verb_tensors, "dims": store.dims}, out)
    return out


def embedding_store_from_config(config: VSSConfig | None = None) -> EmbeddingStore:
    """Construct an embedding store from config and environment variables."""
    cfg = config or VSSConfig()
    vec_env = os.environ.get("DYLAN_VSS_VECTOR_SHELVE")
    ten_env = os.environ.get("DYLAN_VSS_TENSOR_SHELVE")
    vec_path = cfg.vector_shelve_path or (Path(vec_env) if vec_env else None)
    ten_path = cfg.tensor_shelve_path or (Path(ten_env) if ten_env else None)
    if vec_path is not None and ten_path is not None:
        return ShelveEmbeddingStore(vec_path, ten_path, dims=cfg.dims)
    return BundledWord2VecStore(dims=cfg.dims)


def iter_gs2013_verbs(data_path: Path) -> Iterator[str]:
    """Yield unique verbs and landmarks from GS2013 TSV for demo tensor building."""
    import csv

    seen: set[str] = set()
    with data_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=" "):
            for col in ("verb", "landmark"):
                w = row.get(col, "").strip()
                if w and w not in seen:
                    seen.add(w)
                    yield w
