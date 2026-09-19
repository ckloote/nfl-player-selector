"""`pool research backtest` and `sweep` at the command line: errors that name the
fix, and summaries that say what they are.
"""

import pytest
from typer.testing import CliRunner

from pool import db
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


def test_backtest_without_prior_season_data_names_the_refresh_to_run(dbfile):
    """The prior season is the model's starting prior: without it every
    positional mean collapses to zero and the replay silently returns garbage
    instead of failing."""
    result = runner.invoke(app, ["research", "backtest", "--season", "2026", "--db", str(dbfile)])
    assert result.exit_code == 1
    assert "No 2025 stats loaded" in result.stdout
    assert "pool refresh --season 2026" in result.stdout


def test_backtest_on_a_season_with_no_schedule_points_at_refresh(dbfile):
    result = runner.invoke(app, ["research", "backtest", "--season", "2030", "--db", str(dbfile)])
    assert result.exit_code == 1
    assert "No 2030 schedule loaded" in result.stdout


def test_backtest_rejects_an_unknown_strategy(dbfile):
    result = runner.invoke(
        app,
        ["research", "backtest", "--season", "2026", "--strategy", "bogus", "--db", str(dbfile)],
    )
    assert result.exit_code == 1
    assert "Unknown strategy 'bogus'" in result.stdout


@pytest.mark.parametrize("spec", ["twenty", "2024-", "20x4"])
def test_backtest_explains_an_unreadable_season_spec(spec, dbfile):
    result = runner.invoke(app, ["research", "backtest", "--season", spec, "--db", str(dbfile)])
    assert result.exit_code == 1
    assert "as a season, list, or range" in result.stdout


def test_backtest_renders_a_summary_table(tmp_path):
    """The happy path: a replay over a seeded two-season database."""
    from tests.support.season import SEASON, seed_season

    path = tmp_path / "bt.db"
    seed_season(db.connect(path)).close()
    result = runner.invoke(
        app,
        [
            "research",
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


def _backtest_db(tmp_path):
    from tests.support.season import seed_season

    path = tmp_path / "bt.db"
    seed_season(db.connect(path)).close()
    return path


def test_a_winprob_backtest_prints_the_invented_rivals_caveat(tmp_path):
    """Claim 25. The caveat goes under every table that has a finish column, not once in
    the help text, because the table is what gets copied into a message."""
    from tests.support.season import SEASON

    path = _backtest_db(tmp_path)
    result = runner.invoke(
        app,
        [
            "research",
            "backtest",
            "--season",
            str(SEASON),
            "--strategy",
            "winprob,optimizer",
            "--rivals",
            "3",
            "--db",
            str(path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "winprob" in result.stdout
    assert "3 invented rivals picking greedy" in result.stdout
    assert "not of the idea" in result.stdout
    assert "Finish" in result.stdout
    assert "Invented A" in result.stdout


def test_a_backtest_without_winprob_invents_nobody_and_says_nothing(tmp_path):
    from tests.support.season import SEASON

    path = _backtest_db(tmp_path)
    result = runner.invoke(
        app,
        [
            "research",
            "backtest",
            "--season",
            str(SEASON),
            "--strategy",
            "optimizer,greedy",
            "--db",
            str(path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "invented" not in result.stdout
    assert "Finish" not in result.stdout


@pytest.mark.parametrize(
    ("args", "expected"),
    [(["--rival-behaviour", "telepathic"], "telepathic"), (["--rivals", "0"], "at least one")],
)
def test_a_bad_invented_field_is_named_rather_than_crashing(tmp_path, args, expected):
    from tests.support.season import SEASON

    path = _backtest_db(tmp_path)
    result = runner.invoke(
        app,
        [
            "research",
            "backtest",
            "--season",
            str(SEASON),
            "--strategy",
            "winprob",
            "--db",
            str(path),
            *args,
        ],
    )
    assert result.exit_code == 1
    assert expected in result.stdout


def test_sweep_renders_the_grid_and_warns_about_noise(tmp_path):
    from tests.support.season import SEASON, seed_season

    path = tmp_path / "bt.db"
    seed_season(db.connect(path)).close()
    result = runner.invoke(
        app,
        [
            "research",
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
