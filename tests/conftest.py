import os
from datetime import UTC, datetime

import pandas as pd
import pytest
import time_machine

from pool import db
from tests.support import reports
from tests.support.local import local_db
from tests.support.season import seed_season

# The CLI tests read Rich output as plain text at a known width, and whether it is plain was
# otherwise up to the environment. Typer forces colour whenever GITHUB_ACTIONS, FORCE_COLOR or
# PY_COLORS is set, Rich follows FORCE_COLOR, and both size tables from COLUMNS -- so a CI
# runner, or a shell exporting any of these, failed tests that were not broken. Typer reads
# its switch once, on import, and a Rich console created while COLUMNS is set keeps that
# width for good: this runs before any test module imports either, so keep `pool.cli` out of
# this file's imports. With COLUMNS unset, width defaults to 80 off a terminal, and a test
# that needs a wider console can still monkeypatch it.
os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] = "1"
os.environ["TTY_COMPATIBLE"] = "0"
os.environ.pop("COLUMNS", None)

# The fixtures are written as of week 2 of the 2026 season: a report in hand, week 2 not yet
# kicked off. Commands that read the clock -- `recommend`, `predict record` -- were run on the
# real one, so the suite passed until week 2 kicked off and failed every day after. It runs
# as of this instant instead, whatever today is. Session-scoped so module-scoped fixtures are
# covered too, and ticking so timestamps written one after another stay in order.
SUITE_NOW = datetime(2026, 9, 18, 16, 0, tzinfo=UTC)


@pytest.fixture(autouse=True, scope="session")
def suite_clock():
    with time_machine.travel(SUITE_NOW, tick=True):
        yield


@pytest.fixture
def make_proj():
    def _make(rows):
        return pd.DataFrame(rows)

    return _make


@pytest.fixture
def seeded(tmp_path):
    """A four-week, four-team season with a full prior season behind it."""
    return seed_season(db.connect(tmp_path / "bt.db"))


@pytest.fixture
def local(tmp_path):
    """The small 2026 database: week 1's games and five players. Yields `(conn, path)`."""
    yield from local_db(tmp_path)


@pytest.fixture
def reported(local):
    """`local` with week 1 scored and its pool report imported, so week 2 has rivals."""
    return reports.reported(local)
