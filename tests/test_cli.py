import pytest
from typer.testing import CliRunner

from pool import db
from pool.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def wide_terminal(monkeypatch):
    """Rich wraps output at 80 columns under CliRunner, splitting the very
    phrases these tests assert on. Give it room."""
    monkeypatch.setenv("COLUMNS", "200")


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


def test_backtest_without_prior_season_data_names_the_refresh_to_run(dbfile):
    """The prior season is the model's starting prior: without it every
    positional mean collapses to zero and the replay silently returns garbage
    instead of failing."""
    result = runner.invoke(app, ["backtest", "--season", "2026", "--db", str(dbfile)])
    assert result.exit_code == 1
    assert "No 2025 stats loaded" in result.stdout
    assert "pool refresh --season 2026" in result.stdout


def test_backtest_on_a_season_with_no_schedule_points_at_refresh(dbfile):
    result = runner.invoke(app, ["backtest", "--season", "2030", "--db", str(dbfile)])
    assert result.exit_code == 1
    assert "No 2030 schedule loaded" in result.stdout


def test_backtest_rejects_an_unknown_strategy(dbfile):
    result = runner.invoke(
        app, ["backtest", "--season", "2026", "--strategy", "bogus", "--db", str(dbfile)]
    )
    assert result.exit_code == 1
    assert "Unknown strategy 'bogus'" in result.stdout


@pytest.mark.parametrize("spec", ["twenty", "2024-", "20x4"])
def test_backtest_explains_an_unreadable_season_spec(spec, dbfile):
    result = runner.invoke(app, ["backtest", "--season", spec, "--db", str(dbfile)])
    assert result.exit_code == 1
    assert "as a season, list, or range" in result.stdout


def test_backtest_renders_a_summary_table(tmp_path):
    """The happy path: a replay over a seeded two-season database."""
    from tests.test_backtest import SEASON, _seed

    path = tmp_path / "bt.db"
    _seed(db.connect(path)).close()
    result = runner.invoke(
        app,
        [
            "backtest",
            "--season",
            str(SEASON),
            "--strategy",
            "optimizer,greedy,hindsight",
            "--db",
            str(path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert f"Backtest {SEASON}" in result.stdout
    for strategy in ("optimizer", "greedy", "hindsight"):
        assert strategy in result.stdout


def test_sweep_renders_the_grid_and_warns_about_noise(tmp_path):
    from tests.test_backtest import SEASON, _seed

    path = tmp_path / "bt.db"
    _seed(db.connect(path)).close()
    result = runner.invoke(
        app,
        [
            "sweep",
            "--season",
            str(SEASON),
            "--discount",
            "0.9,1.0",
            "--prior-weight",
            "5,7",
            "--db",
            str(path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Best cell" in result.stdout
    assert "noise" in result.stdout


def test_backtest_rejects_an_unknown_projection_model(dbfile):
    result = runner.invoke(
        app, ["backtest", "--season", "2026", "--projection", "bogus", "--db", str(dbfile)]
    )
    assert result.exit_code == 1
    assert "Unknown projection 'bogus'" in result.stdout
