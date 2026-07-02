"""Convert shelve/sparse embedding payloads to PyTorch tensors."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def to_dense_tensor(value: Any, *, dims: int | None = None) -> torch.Tensor:
    """Convert numpy, scipy sparse, list, or tensor payloads to a float32 ``torch.Tensor``."""
    if value is None:
        raise KeyError("embedding value is None")
    if isinstance(value, torch.Tensor):
        t = value.float()
    elif hasattr(value, "toarray"):
        t = torch.from_numpy(np.asarray(value.toarray(), dtype=np.float32))
    elif hasattr(value, "todense"):
        t = torch.from_numpy(np.asarray(value.todense(), dtype=np.float32))
    else:
        arr = np.asarray(value, dtype=np.float32)
        t = torch.from_numpy(arr)
    if dims is not None and t.numel() > dims:
        t = t.reshape(-1)[:dims]
    return t.float()
