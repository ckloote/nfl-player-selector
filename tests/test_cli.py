import pytest
from typer.testing import CliRunner

from pool import db
from pool.cli import app

runner = CliRunner()


@pytest.fixture
def dbfile(tmp_path):
    """A database holding an 18-week schedule and nothing else."""
    path = tmp_path / "pool.db"
    conn = db.connect(path)
    with conn:
        conn.executemany(
            "INSERT INTO games(game_id, season, week, game_type, kickoff, home_team, away_team) "
            "VALUES (?,?,?,?,?,?,?)",
            [
                (f"g{w}", 2026, w, "REG", f"2026-09-{12 + w:02d}T13:00", "A", "B")
                for w in range(1, 19)
            ],
        )
    conn.close()
    return path


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
