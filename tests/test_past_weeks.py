"""Projections as of a past week: they see what was known then, from the rosters
and lines of that week, and nothing later.
"""

import pandas as pd

from pool import config
from pool import projections as P
from tests.support.season import SEASON, TEAMS


# --- point-in-time freezing -------------------------------------------------
def test_frozen_frames_hide_the_week_being_played(seeded):
    """Week W's own results are the answer; only earlier weeks may be visible.

    Injuries and rosters are different: both are published before kickoff, so
    week W's rows are legitimately in hand at the deadline.
    """
    frames = P.load_frames(seeded, SEASON, as_of_week=3)
    assert sorted(frames.pw_cur.week.unique()) == [1, 2]
    assert frames.pw_prior.week.max() == 4  # the prior season is complete
    assert frames.rosters.week.max() <= 3


def test_vegas_lines_beyond_the_horizon_are_hidden(seeded):
    """A finished season has every closing line; live, the database holds a few
    weeks and nothing beyond. Replaying with all of them would make far-future
    matchups look knowable and bias the future discount upward."""
    frames = P.load_frames(
        seeded, SEASON, as_of_week=1, vegas_horizon=1, input_policy="legacy-closing"
    )
    cur = frames.games[frames.games.season == SEASON]
    assert cur[cur.week <= 2].total_line.notna().all()
    assert cur[cur.week > 2].total_line.isna().all()


def test_usage_roles_rank_the_starter_ahead_of_the_backup(seeded):
    """The depth chart cannot be rewound, so role comes from usage to date."""
    frames = P.load_frames(seeded, SEASON, as_of_week=2)
    pool = P.player_pool(frames.rosters, frames.pw_prior, frames.pw_cur)
    roles = P.usage_roles(pool, frames.pw_prior, frames.pw_cur).set_index("player_id")
    assert roles.loc["AAA-QB1", "rank"] < roles.loc["AAA-QB2", "rank"]
    assert roles.loc["AAA-QB1", "role_mult"] == config.DEPTH_MULT["QB"][1]
    assert roles.loc["AAA-QB2", "role_mult"] == config.DEPTH_MULT["QB"][2]


# --- the candidate pool -----------------------------------------------------
def _pool(conn, week):
    frames = P.load_frames(conn, SEASON, as_of_week=week)
    return P.player_pool(frames.rosters, frames.pw_prior, frames.pw_cur).set_index("player_id")


def test_a_released_player_leaves_the_pool(seeded):
    """Weekly rosters are snapshots: a released player stops appearing rather
    than being marked CUT, so the union of every week to date never forgets
    anyone. Against the 2024 feed that union carried 640 candidates into week 17
    where the week-17 roster listed 443."""
    with seeded:
        seeded.execute(
            "DELETE FROM rosters WHERE season=? AND player_id='AAA-RB2' AND week > 2", (SEASON,)
        )
    assert "AAA-RB2" in _pool(seeded, 2).index
    assert "AAA-RB2" not in _pool(seeded, 4).index


def test_a_traded_player_carries_his_new_team(seeded):
    """Resolving duplicates by first appearance pinned a moved player to his old
    team, and so to the wrong opponent, defense and implied total, for the rest
    of the season. Every one of the 25 players who changed teams during 2024 was
    reported under the team they left."""
    with seeded:
        seeded.execute(
            "UPDATE rosters SET team='CCC' WHERE season=? AND player_id='AAA-WR1' AND week >= 3",
            (SEASON,),
        )
    assert _pool(seeded, 2).loc["AAA-WR1", "team"] == "AAA"
    assert _pool(seeded, 4).loc["AAA-WR1", "team"] == "CCC"


def test_one_inflated_snapshot_cannot_contaminate_later_weeks(seeded):
    """The nflverse feed labels a cutdown-era roster as 2016 week 1 — 24.3 pool
    actives per team against 11.9-16.5 in every other season-week from 2010 on,
    naming ~275 players who never took a snap. Under the union it inflated the
    candidate universe for all seventeen weeks of that season."""
    ghosts = [(SEASON, 1, f"ghost{i}", f"Ghost {i}", "WR", "AAA", "ACT", "WR") for i in range(50)]
    with seeded:
        seeded.executemany("INSERT INTO rosters VALUES (?,?,?,?,?,?,?,?)", ghosts)
    assert sum(p.startswith("ghost") for p in _pool(seeded, 1).index) == 50
    assert not any(p.startswith("ghost") for p in _pool(seeded, 4).index)


def test_an_unpublished_roster_week_reads_the_previous_one(seeded):
    """Live, week W's roster may not have landed by the pick deadline. Falling
    back to the latest week that did is what the picker is looking at anyway."""
    with seeded:
        seeded.execute("DELETE FROM rosters WHERE season=? AND week=4", (SEASON,))
    assert set(_pool(seeded, 4).index) == set(_pool(seeded, 3).index)


def _bye(conn, teams, week=4):
    """A team on its bye: the weekly roster feed publishes no row for it at all."""
    with conn:
        conn.executemany(
            "DELETE FROM rosters WHERE season=? AND week=? AND team=?",
            [(SEASON, week, team) for team in teams],
        )


def test_a_team_on_its_bye_keeps_its_players(seeded):
    """The feed publishes no roster for a team on its bye, so one league-wide
    latest week deletes it from the pool. In 2024 week 5 that removed DET, PHI,
    LAC and TEN; the worst decision weeks of 2016-2025 kept 26 of 32 teams."""
    _bye(seeded, ["CCC", "DDD"])
    pool = _pool(seeded, 4)
    assert set(pool.team) == set(TEAMS)
    for team in ("CCC", "DDD"):
        assert pool.loc[f"{team}-QB1", "team"] == team


def test_a_bye_team_stays_in_the_remaining_season_plan(seeded):
    """The current week never misses them — a bye team has no game, so no row.
    The forward surface does, and that is the half the optimizer chooses over:
    at decision week 5 of 2024 the four bye teams had no row at any week 6-18."""
    _bye(seeded, ["CCC", "DDD"])
    frames = P.load_frames(seeded, SEASON, as_of_week=4)
    proj = P.build_projections(frames, from_week=4)
    assert set(proj[proj.week == 4].team) == set(TEAMS)


def test_a_player_traded_off_a_bye_team_appears_once():
    """Reading each team's own latest week can offer the same player twice: his
    old team's snapshot is a week behind and still lists him. `load_frames` keeps
    only his latest row, but the snapshot rule has to hold on its own."""
    rosters = pd.DataFrame(
        [
            dict(week=3, player_id="p1", team="CCC", status="ACT"),
            dict(week=4, player_id="p1", team="AAA", status="ACT"),
            dict(week=4, player_id="p2", team="AAA", status="ACT"),
        ]
    )
    snapshot = P.active_snapshot(rosters).set_index("player_id")
    assert sorted(snapshot.index) == ["p1", "p2"]
    assert snapshot.loc["p1", "team"] == "AAA"
