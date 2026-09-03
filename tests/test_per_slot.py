"""Tests for the per-slot base-rate experiment (see src/pool/experiments/)."""

import pytest

from pool import backtest as B
from pool import db
from pool import projections as P
from pool.experiments import per_slot

SEASON, PRIOR = 2024, 2023

# Two receivers, opposite profiles. Volume is targeted constantly and scores
# rarely; Lucky is barely targeted and found the end zone anyway. Touchdown
# history prefers Lucky, opportunity prefers Volume, and regression to the mean
# is the reason to think opportunity is the better bet.
PRIOR_LINES = [  # player, position, team, opponent, week, touchdowns, usage
    ("volume", "WR", "AAA", "OLD", 1, 0, 10),
    ("volume", "WR", "AAA", "OLD", 2, 1, 10),
    ("lucky", "WR", "BBB", "OLD2", 1, 2, 3),
    ("lucky", "WR", "BBB", "OLD2", 2, 1, 3),
    ("passer", "QB", "AAA", "OLD", 1, 2, 30),
    ("passer", "QB", "AAA", "OLD", 2, 1, 30),
]


def _seed(path):
    conn = db.connect(path)
    games = []
    for wk in (1, 2):
        for home, away in (("AAA", "OLD"), ("BBB", "OLD2")):
            games.append(
                (
                    f"p{wk}{home}",
                    PRIOR,
                    wk,
                    "REG",
                    f"{PRIOR}-09-{10 + wk:02d}T13:00",
                    home,
                    away,
                    0.0,
                    44.0,
                )
            )
    # Fresh opponents in the replay season, so neither receiver gets a defensive
    # edge: with no prior stat lines the multiplier falls back to 1.0 for both.
    for home, away in (("AAA", "DEF1"), ("BBB", "DEF2")):
        games.append(
            (f"s1{home}", SEASON, 1, "REG", f"{SEASON}-09-11T13:00", home, away, 0.0, 44.0)
        )
    lines = [
        (
            PRIOR,
            wk,
            "REG",
            pid,
            pid.title(),
            pos,
            team,
            opp,
            td if pos == "QB" else 0,
            0,
            0 if pos == "QB" else td,
            usage if pos == "QB" else 0,
            0,
            0 if pos == "QB" else usage,
        )
        for pid, pos, team, opp, wk, td, usage in PRIOR_LINES
    ]
    rosters = [
        (s, wk, pid, pid.title(), pos, team, "ACT", pos)
        for s in (PRIOR, SEASON)
        for wk in (1, 2)
        for pid, pos, team in (
            ("volume", "WR", "AAA"),
            ("lucky", "WR", "BBB"),
            ("passer", "QB", "AAA"),
        )
    ]
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
    return _seed(tmp_path / "ps.db")


def _lam(frame, player_id):
    return float(frame.loc[frame.player_id == player_id, "lam"].iloc[0])


def test_per_slot_frame_matches_the_projection_contract(seeded):
    out = per_slot.build(P.load_frames(seeded, SEASON, as_of_week=1), 1, role_source="usage")
    assert list(out.columns) == P.PROJECTION_COLUMNS
    assert out.lam.notna().all() and (out.lam >= 0).all()


def test_the_flex_slot_is_ranked_by_opportunity_not_touchdown_history(seeded):
    """The change this experiment exists to make: at WR/TE the heavily targeted
    receiver outranks the one who simply converted more of very little."""
    frames = P.load_frames(seeded, SEASON, as_of_week=1)
    shipped = P.build_projections(frames, 1, role_source="usage")
    swapped = per_slot.build(frames, 1, role_source="usage")
    assert _lam(shipped, "lucky") > _lam(shipped, "volume")
    assert _lam(swapped, "volume") > _lam(swapped, "lucky")


def test_the_quarterback_slot_is_left_alone(seeded):
    """Touchdown history works at QB - it is the one slot the shipped model
    ranks well - so this model must not touch it."""
    frames = P.load_frames(seeded, SEASON, as_of_week=1)
    shipped = P.build_projections(frames, 1, role_source="usage")
    swapped = per_slot.build(frames, 1, role_source="usage")
    qb = shipped.slot == "QB"
    assert qb.any(), "fixture must contain a quarterback for this to mean anything"
    assert _lam(shipped, "passer") == pytest.approx(_lam(swapped, "passer"))


def test_the_builder_seam_actually_swaps_the_model(seeded):
    """Regression: the CLI once accepted --projection, threaded the builder
    nowhere, and silently reported the shipped model's numbers."""
    shipped = B.weekly_projections(seeded, SEASON, [1])
    swapped = B.weekly_projections(seeded, SEASON, [1], builder=per_slot.build)
    assert not shipped[1].lam.astype(float).equals(swapped[1].lam.astype(float))
