"""Integration tests for DS-VSS session (grammar-dependent)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from dylan.vss.ds_vss_session import DSVSSSession
from dylan.vss.embedding_store import DemoEmbeddingStore


@pytest.mark.timeout(60)
def test_parse_incremental_table_draw_eye() -> None:
    """Parse a short transitive sentence with demo store (may use dataset fallback)."""
    session = DSVSSSession(embedding_store=DemoEmbeddingStore())
    result = session.parse_incremental("table draw eye")
    assert len(result.steps) >= 1
    assert result.sentence == "table draw eye"
