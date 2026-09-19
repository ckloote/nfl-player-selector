"""Weekly commands meeting a database that cannot answer yet: each says what is missing
and what to run, rather than failing with a traceback."""

import pytest
from typer.testing import CliRunner

from pool.cli import app
from tests.support.local import schedule_only_db

runner = CliRunner()


@pytest.fixture(autouse=True)
def wide_terminal(monkeypatch):
    """Rich wraps output at 80 columns under CliRunner, splitting the very
    phrases these tests assert on. Give it room."""
    monkeypatch.setenv("COLUMNS", "200")


@pytest.fixture
def dbfile(tmp_path):
    return schedule_only_db(tmp_path)


@pytest.mark.parametrize(
    "cmd",
    [
        ["recommend", "--week", "19"],
        ["plan", "--week", "19"],  # used to crash: int(NaN) on an empty frame
        ["players", "--pos", "QB", "--week", "19"],
        ["recommend", "--week", "0"],  # below the season still yields rows
        ["players", "--pos", "QB", "--week", "0"],
    ],
)
def test_out_of_range_week_is_explained_not_crashed(cmd, dbfile):
    result = runner.invoke(app, [*cmd, "--db", str(dbfile)])
    assert result.exit_code == 1
    assert "outside the 2026 season (weeks 1-18)" in result.stdout


def test_missing_schedule_points_at_refresh(tmp_path):
    result = runner.invoke(app, ["plan", "--week", "1", "--db", str(tmp_path / "empty.db")])
    assert result.exit_code == 1
    assert "No 2026 schedule loaded" in result.stdout


def test_schedule_but_no_player_data_is_reported_separately(dbfile):
    """A valid week with no roster or stats loaded is a different problem."""
    result = runner.invoke(app, ["recommend", "--week", "1", "--db", str(dbfile)])
    assert result.exit_code == 1
    assert "No projections for week 1" in result.stdout
