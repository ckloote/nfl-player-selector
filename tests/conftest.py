import os
from datetime import UTC, datetime

import pandas as pd
import pytest
import time_machine

from pool import db

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


def proj_row(
    pid,
    name,
    slot,
    week,
    lam,
    kickoff="2026-09-13T13:00",
    team="AAA",
    opp="BBB",
    position=None,
    home=True,
    status=None,
):
    return dict(
        player_id=pid,
        player_name=name,
        position=position or {"QB": "QB", "RB": "RB", "FLEX": "WR"}[slot],
        slot=slot,
        team=team,
        week=week,
        opponent=opp,
        home=home,
        kickoff=kickoff,
        kickoff_known=1,
        game_id=f"g{week}",
        base_rate=lam,
        def_mult=1.0,
        vegas_mult=1.0,
        home_mult=1.0,
        avail_mult={"Out": 0.0, "Doubtful": 0.0, "Questionable": 0.85}.get(status, 1.0),
        hard_eligible=status not in ("Out", "Doubtful"),
        report_status=status,
        role_mult=1.0,
        depth_rank=1,
        lam=lam,
        prior_games=10,
        prior_tds=5,
        cur_games=0,
        cur_tds=0,
    )


@pytest.fixture
def make_proj():
    def _make(rows):
        return pd.DataFrame(rows)

    return _make


@pytest.fixture
def seeded(tmp_path):
    """A four-week, four-team season with a full prior season behind it.

    Shared by the backtest and evaluation suites; the builder lives with the
    backtest tests that define the schema expectations.
    """
    from tests.test_backtest import _seed

    return _seed(db.connect(tmp_path / "bt.db"))
