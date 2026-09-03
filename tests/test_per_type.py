"""Tests for the per-type projection experiment (see src/pool/experiments/)."""

import pytest

from pool import backtest as B
from pool import db
from pool import projections as P
from pool.experiments import per_type

SEASON, PRIOR, WEEKS = 2024, 2023, [1, 2]

# Prior season establishes both the players and the defences; the replay season
# is where the two quarterbacks meet the rush-leaking defences.
PRIOR_GAMES = [("FIL", "DEF1"), ("FIL2", "DEF2"), ("POC", "OTH"), ("RUN", "OTH2")]
SEASON_GAMES = [("POC", "DEF1"), ("RUN", "DEF2"), ("FIL", "OTH"), ("FIL2", "OTH2")]

# player_id, team, opponent, pass_td, rush_td — all in the prior season.
# Filler quarterbacks make DEF1/DEF2 leak rushing scores and nothing else.
# Pocket and Runner reach an identical 2 TDs a game by different routes.
PRIOR_LINES = [
    ("filler", "FIL", "DEF1", 0, 3),
    ("filler2", "FIL2", "DEF2", 0, 3),
    ("pocket", "POC", "OTH", 2, 0),
    ("runner", "RUN", "OTH2", 1, 1),
]


def _seed(path):
    conn = db.connect(path)
    games, lines, rosters = [], [], []
    for season, pairs in ((PRIOR, PRIOR_GAMES), (SEASON, SEASON_GAMES)):
        for wk in WEEKS:
            for home, away in pairs:
                games.append(
                    (
                        f"g{season}{wk}{home}",
                        season,
                        wk,
                        "REG",
                        f"{season}-09-{10 + wk:02d}T13:00",
                        home,
                        away,
                        0.0,
                        44.0,
                    )
                )
    for wk in WEEKS:
        for pid, team, opp, ptd, rtd in PRIOR_LINES:
            lines.append(
                (PRIOR, wk, "REG", pid, pid.title(), "QB", team, opp, ptd, rtd, 0, 30, 4, 0)
            )
        for pid, team in (("pocket", "POC"), ("runner", "RUN")):
            for s in (PRIOR, SEASON):
                rosters.append((s, wk, pid, pid.title(), "QB", team, "ACT", "QB"))
    with conn:
        conn.executemany(
            "INSERT INTO games(game_id, season, week, game_type, kickoff, home_team, away_team,"
            " spread_line, total_line) VALUES (?,?,?,?,?,?,?,?,?)",
            games,
        )
        conn.executemany(
            "INSERT INTO player_weeks(season, week, season_type, player_id, player_name, position,"
            " team, opponent, pass_td, rush_td, rec_td, attempts, carries, targets)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            lines,
        )
        conn.executemany(
            "INSERT INTO rosters(season, week, player_id, player_name, position, team, status,"
            " depth_chart_position) VALUES (?,?,?,?,?,?,?,?)",
            rosters,
        )
    return conn


@pytest.fixture
def seeded(tmp_path):
    return _seed(tmp_path / "pt.db")


def _lam(frame, player_id, week=1):
    row = frame[(frame.player_id == player_id) & (frame.week == week)]
    return float(row.lam.iloc[0])


def test_per_type_frame_matches_the_projection_contract(seeded):
    """Downstream code consumes only the frame, so a variant model must produce
    exactly the same columns or it cannot be swapped in."""
    out = per_type.build(P.load_frames(seeded, SEASON, as_of_week=1), 1, role_source="usage")
    assert list(out.columns) == P.PROJECTION_COLUMNS
    assert out.lam.notna().all() and (out.lam >= 0).all()


def test_per_type_separates_a_rushing_quarterback_from_a_pocket_passer(seeded):
    """The whole point of splitting the rate. Two quarterbacks with identical
    touchdown totals face defences that leak rushing scores and stop passing
    ones. A combined rate cannot tell them apart; a per-type rate must."""
    frames = P.load_frames(seeded, SEASON, as_of_week=1)
    combined = P.build_projections(frames, 1, role_source="usage")
    split = per_type.build(frames, 1, role_source="usage")
    assert _lam(combined, "runner") == pytest.approx(_lam(combined, "pocket"), rel=0.02)
    assert _lam(split, "runner") > _lam(split, "pocket")


def test_the_builder_seam_actually_swaps_the_model(seeded):
    """Regression: the CLI once accepted --projection per-type, threaded the
    builder nowhere, and silently reported the shipped model's numbers."""
    shipped = B.weekly_projections(seeded, SEASON, WEEKS)
    swapped = B.weekly_projections(seeded, SEASON, WEEKS, builder=per_type.build)
    assert not shipped[1].lam.astype(float).equals(swapped[1].lam.astype(float))
