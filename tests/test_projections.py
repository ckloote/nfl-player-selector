import pandas as pd
import pytest

from pool import config
from pool import projections as P


def pw(rows):
    cols = [
        "season",
        "week",
        "season_type",
        "player_id",
        "player_name",
        "position",
        "team",
        "opponent",
        "pass_td",
        "rush_td",
        "rec_td",
        "attempts",
        "carries",
        "targets",
    ]
    return pd.DataFrame(rows, columns=cols)


def test_baseline_rate_regresses_prior_and_blends_current():
    prior = pw(
        [[2025, w, "REG", "qb1", "QB One", "QB", "A", "B", 2, 0, 0, 30, 2, 0] for w in range(1, 18)]
    )
    pool = pd.DataFrame(
        {
            "player_id": ["qb1", "rook"],
            "player_name": ["QB One", "Rookie"],
            "position": ["QB", "QB"],
            "team": ["A", "C"],
        }
    )
    means = {"QB": 1.5}
    rates = P.baseline_rates(pool, prior, pw([]), means)
    k = config.PRIOR_SEASON_SHRINK_GAMES
    expected_prior = (34 + k * 1.5) / (17 + k)
    r = rates.set_index("player_id")
    assert r.loc["qb1", "prior_rate"] == pytest.approx(expected_prior)
    assert r.loc["qb1", "base_rate"] == pytest.approx(expected_prior)  # no current data
    assert r.loc["rook", "prior_rate"] == pytest.approx(config.NO_HISTORY_FACTOR * 1.5)

    cur = pw(
        [[2026, w, "REG", "qb1", "QB One", "QB", "A", "B", 0, 0, 0, 30, 2, 0] for w in range(1, 4)]
    )
    rates2 = P.baseline_rates(pool, prior, cur, means).set_index("player_id")
    w = config.PRIOR_WEIGHT_GAMES
    assert rates2.loc["qb1", "base_rate"] == pytest.approx((w * expected_prior + 0) / (w + 3))
    assert rates2.loc["qb1", "base_rate"] < expected_prior


def test_positional_means_use_regulars_only():
    rows = []
    for w in range(1, 18):
        rows.append([2025, w, "REG", "s", "Starter", "RB", "A", "B", 0, 1, 0, 0, 15, 3])
    for w in range(1, 3):
        rows.append([2025, w, "REG", "b", "Backup", "RB", "A", "B", 0, 0, 0, 0, 1, 0])
    means = P.positional_means(pw(rows))
    assert means["RB"] == pytest.approx(1.0)


def test_defense_multipliers_symmetric_data_is_one():
    rows = []
    for w in range(1, 10):
        rows.append([2025, w, "REG", "r1", "R1", "RB", "A", "B", 0, 1, 0, 0, 10, 0])
        rows.append([2025, w, "REG", "r2", "R2", "RB", "B", "A", 0, 1, 0, 0, 10, 0])
    d = P.defense_multipliers(pw(rows), pw([])).set_index(["team", "group"])
    assert d.loc[("A", "RB"), "def_mult"] == pytest.approx(1.0)


def test_defense_multiplier_soft_defense_above_one_and_regressed():
    rows = []
    for w in range(1, 10):
        rows.append([2025, w, "REG", "r1", "R1", "RB", "A", "B", 0, 3, 0, 0, 10, 0])  # B allows 3/g
        rows.append([2025, w, "REG", "r2", "R2", "RB", "B", "A", 0, 1, 0, 0, 10, 0])  # A allows 1/g
    d = P.defense_multipliers(pw(rows), pw([])).set_index(["team", "group"])
    assert 1.0 < d.loc[("B", "RB"), "def_mult"] < 1.5  # raw would be 1.5; regressed
    assert d.loc[("A", "RB"), "def_mult"] < 1.0


def test_game_context_implied_totals_and_fallbacks():
    games = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "season": [2026, 2026],
            "week": [1, 2],
            "game_type": ["REG"] * 2,
            "kickoff": ["2026-09-13T13:00", "2026-09-20T13:00"],
            "weekday": ["Sunday"] * 2,
            "home_team": ["H", "H"],
            "away_team": ["A", "A"],
            "home_score": [None] * 2,
            "away_score": [None] * 2,
            "spread_line": [3.0, None],
            "total_line": [50.0, None],
        }
    )
    ctx = P.game_context(games, 2026).set_index(["team", "week"])
    assert ctx.loc[("H", 1), "implied_total"] == pytest.approx(26.5)
    assert ctx.loc[("A", 1), "implied_total"] == pytest.approx(23.5)
    # week 2 has no line: falls back to the team's own average
    assert ctx.loc[("H", 2), "implied_total"] == pytest.approx(26.5)
    assert ctx.loc[("H", 1), "home_mult"] == config.HOME_MULT


def test_build_projections_excludes_bye_weeks_and_applies_injuries():
    games = pd.DataFrame(
        {
            "game_id": ["g1", "g3"],
            "season": [2026] * 2,
            "week": [1, 3],
            "game_type": ["REG"] * 2,
            "kickoff": ["2026-09-13T13:00", "2026-09-27T13:00"],
            "weekday": ["Sunday"] * 2,
            "home_team": ["H", "H"],
            "away_team": ["A", "A"],
            "home_score": [None] * 2,
            "away_score": [None] * 2,
            "spread_line": [0.0, 0.0],
            "total_line": [45.0, 45.0],
        }
    )
    rosters = pd.DataFrame(
        {
            "season": [2026],
            "week": [1],
            "player_id": ["p1"],
            "player_name": ["P"],
            "position": ["RB"],
            "team": ["H"],
            "status": ["ACT"],
            "depth_chart_position": ["RB"],
        }
    )
    prior = pw(
        [[2025, w, "REG", "p1", "P", "RB", "H", "A", 0, 1, 0, 0, 10, 2] for w in range(1, 18)]
    )
    inj = pd.DataFrame(
        {
            "season": [2026],
            "week": [3],
            "player_id": ["p1"],
            "player_name": ["P"],
            "team": ["H"],
            "position": ["RB"],
            "report_status": ["Out"],
            "practice_status": [None],
        }
    )
    fr = P.Frames(2026, games, prior, pw([]), rosters, inj)
    proj = P.build_projections(fr)
    assert sorted(proj.week.unique()) == [1, 3]  # week 2 is a bye
    assert proj[proj.week == 3].lam.iloc[0] == 0.0
    assert proj[proj.week == 1].lam.iloc[0] > 0


def injury_report(rows):
    cols = ["player_id", "week", "report_status"]
    return pd.DataFrame(rows, columns=cols)


def test_injury_multipliers_map_status_and_default_to_available():
    m = injury_report(
        [
            ["out", 1, "Out"],
            ["doubt", 1, "Doubtful"],
            ["quest", 1, "Questionable"],
            ["full", 1, "Full Participation in Practice"],  # not a ruling-out status
            ["none", 1, None],
        ]
    )
    r = P.injury_multipliers(m).set_index("player_id").avail_mult
    assert r["out"] == 0.0
    assert r["doubt"] == 0.0
    assert r["quest"] == pytest.approx(config.INJURY_MULT["Questionable"])
    # an unrecognised or absent status must not silently zero a player out
    assert r["full"] == 1.0
    assert r["none"] == 1.0


def test_injury_multipliers_are_keyed_by_week_not_player():
    """Being Out in week 3 must not follow a player into every other week."""
    r = P.injury_multipliers(injury_report([["p1", 3, "Out"], ["p1", 4, "Questionable"]]))
    by_week = r.set_index("week").avail_mult
    assert by_week[3] == 0.0
    assert by_week[4] == pytest.approx(config.INJURY_MULT["Questionable"])
    assert set(r.week) == {3, 4}  # other weeks carry no row and default to available


def test_injury_multipliers_on_empty_report_returns_mergeable_frame():
    r = P.injury_multipliers(injury_report([]))
    assert len(r) == 0
    assert {"player_id", "week", "avail_mult"} <= set(r.columns)
