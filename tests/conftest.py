"""Shared pytest configuration."""

from __future__ import annotations

import os

import pytest

from dylan.logging_config import configure_logging


def pytest_configure(config: pytest.Config) -> None:
    """Use loguru stderr for non-ICP records so unconfigured ICP messages are not duplicated."""
    configure_logging("WARNING")
    os.environ["DYLAN_UNDER_PYTEST"] = "1"


def pytest_unconfigure(config: pytest.Config) -> None:
    """Clear the pytest marker env var so ad-hoc runs in the same interpreter stay unaffected."""
    os.environ.pop("DYLAN_UNDER_PYTEST", None)
