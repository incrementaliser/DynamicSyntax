"""Shared pytest configuration."""

from __future__ import annotations

import pytest

from dylan.logging_config import configure_logging


def pytest_configure(config: pytest.Config) -> None:
    """Use loguru stderr for non-ICP records so unconfigured ICP messages are not duplicated."""
    configure_logging("WARNING")
