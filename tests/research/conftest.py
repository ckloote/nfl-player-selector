"""Tests of the research code: replays, benchmarks, calibration, diagnostics, the
published reports, and the after-the-fact checks on captured decisions.

Everything in this directory is marked `research`, so `pytest -m "not research"` runs the
everyday suite alone. CI runs both.
"""

from pathlib import Path

import pytest

HERE = Path(__file__).parent


def pytest_collection_modifyitems(items):
    for item in items:
        if HERE in item.path.parents:
            item.add_marker(pytest.mark.research)
