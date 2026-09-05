import pandas as pd
import pytest

from pool import db


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
