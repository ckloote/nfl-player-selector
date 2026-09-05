import pandas as pd
import pytest

from pool import backtest as B
from pool import config, db, models
from pool import projections as P

PRIOR, SEASON, WEEKS = 2023, 2024, [1, 2, 3, 4]
TEAMS = ["AAA", "BBB", "CCC", "DDD"]
PAIRS = [("AAA", "BBB"), ("CCC", "DDD")]


def _players():
    """Two players per position per team, so every week has real alternatives."""
    out = []
    for team in TEAMS:
        for pos in ("QB", "RB", "WR"):
            for depth in (1, 2):
                pid = f"{team}-{pos}{depth}"
                out.append((pid, f"{team} {pos}{depth}", pos, team, depth))
    return out


def _seed(conn, *, season_tds=None):
    """A four-week, four-team season with a full prior season behind it."""
    people = _players()
    games, weeks_rows, rosters = [], [], []
    for season in (PRIOR, SEASON):
        for wk in WEEKS:
            for home, away in PAIRS:
                games.append(
                    (
                        f"g{season}-{wk}-{home}",
                        season,
                        wk,
                        "REG",
                        f"{season}-09-{10 + wk:02d}T13:00",
                        home,
                        away,
                        -3.0,
                        45.0,
                    )
                )
    opponent = {}
    for home, away in PAIRS:
        opponent[home], opponent[away] = away, home

    for pid, name, pos, team, depth in people:
        for wk in WEEKS:
            # Prior season: the starter scores, the backup does not.
            tds = 1 if depth == 1 else 0
            weeks_rows.append(
                (
                    PRIOR,
                    wk,
                    "REG",
                    pid,
                    name,
                    pos,
                    team,
                    opponent[team],
                    tds if pos == "QB" else 0,
                    0,
                    tds if pos != "QB" else 0,
                    30 if pos == "QB" else 0,
                    15 if pos == "RB" else 0,
                    8 if pos == "WR" else 0,
                )
            )
            actual = (season_tds or {}).get((wk, pid), 1 if depth == 1 else 0)
            weeks_rows.append(
                (
                    SEASON,
                    wk,
                    "REG",
                    pid,
                    name,
                    pos,
                    team,
                    opponent[team],
                    actual if pos == "QB" else 0,
                    0,
                    actual if pos != "QB" else 0,
                    30 if pos == "QB" else 0,
                    15 if pos == "RB" else 0,
                    8 if pos == "WR" else 0,
                )
            )
            for s in (PRIOR, SEASON):
                rosters.append((s, wk, pid, name, pos, team, "ACT", pos))

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
            weeks_rows,
        )
        conn.executemany(
            "INSERT INTO rosters(season, week, player_id, player_name, position, team, status,"
            " depth_chart_position) VALUES (?,?,?,?,?,?,?,?)",
            rosters,
        )
    # This synthetic fixture explicitly supplies complete scoring results. Legacy
    # databases in production must obtain these from the play-by-play importer.
    with conn:
        conn.execute("UPDATE games SET home_score = 21, away_score = 21, kickoff_known = 1")
        db.backfill_game_ids(conn)
        conn.execute(
            "INSERT INTO game_results SELECT game_id, season, week, 1, 'complete', "
            "home_score, away_score, '2026-09-01T00:00:00+00:00' FROM games"
        )
        credits = []
        for i, r in enumerate(conn.execute("SELECT * FROM player_weeks")):
            for j in range(r["pass_td"] + r["rush_td"] + r["rec_td"]):
                credits.append(
                    (
                        r["game_id"],
                        i * 100 + j,
                        r["player_id"],
                        "throwing" if r["position"] == "QB" else "scoring",
                    )
                )
        conn.executemany(
            "INSERT INTO touchdown_credits(game_id, play_id, player_id, kind) VALUES (?, ?, ?, ?)",
            credits,
        )
    return conn


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


@pytest.mark.parametrize("week", [1, 2, 3])
@pytest.mark.parametrize("model", sorted(models.BUILDERS))
def test_projections_are_identical_whether_or_not_the_future_exists(tmp_path, week, model):
    """The frozen frame must be a pure function of the visible slice.

    Stated this way the test catches any future leak, including through a
    column nobody thought about — which is how the depth-chart snapshot (a
    single end-of-season row, no week) slipped past review in the first place.

    Every benchmark model is checked, not just the shipped one: a baseline that
    quietly reads the whole season would flatter itself in the comparison, and
    the comparison is the entire point of having baselines.
    """
    full = _seed(db.connect(tmp_path / "full.db"))
    trimmed = _seed(db.connect(tmp_path / "trimmed.db"))
    with trimmed:
        trimmed.execute("DELETE FROM player_weeks WHERE season=? AND week>=?", (SEASON, week))
        trimmed.execute("DELETE FROM rosters WHERE season=? AND week>?", (SEASON, week))
        trimmed.execute(
            "UPDATE games SET spread_line=NULL, total_line=NULL WHERE season=? AND week>?",
            (SEASON, week + config.VEGAS_HORIZON_WEEKS),
        )
    builder = models.get(model)
    build = P.build_projections if builder is None else builder
    a = build(P.load_frames(full, SEASON, as_of_week=week), week, role_source="usage")
    b = build(P.load_frames(trimmed, SEASON, as_of_week=week), week, role_source="usage")
    pd.testing.assert_frame_equal(a, b)


def test_vegas_lines_beyond_the_horizon_are_hidden(seeded):
    """A finished season has every closing line; live, the database holds a few
    weeks and nothing beyond. Replaying with all of them would make far-future
    matchups look knowable and bias the future discount upward."""
    frames = P.load_frames(seeded, SEASON, as_of_week=1, vegas_horizon=1)
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


# --- scoring ----------------------------------------------------------------
def test_scoring_counts_every_touchdown_a_pick_throws_or_scores(seeded):
    actuals = B.actual_tds(seeded, SEASON)
    assert actuals[(1, "AAA-QB1")] == 1.0
    assert actuals[(1, "AAA-QB2")] == 0.0
    assert actuals.get((1, "nobody"), 0.0) == 0.0  # missing row is zero, not NaN


# --- replay invariants ------------------------------------------------------
@pytest.mark.parametrize("strategy", ["optimizer", "greedy", "random"])
def test_no_player_is_used_twice_and_every_slot_week_is_filled(seeded, strategy):
    import numpy as np

    frames = B.weekly_projections(seeded, SEASON, WEEKS)
    out = B.replay(
        frames,
        B.actual_tds(seeded, SEASON),
        SEASON,
        strategy,
        weeks=WEEKS,
        rng=np.random.default_rng(0),
    )
    assert len(out.picks) == len(WEEKS) * len(config.SLOTS)
    assert out.empty_slots == 0
    assert out.players_used == len(out.picks)  # the once-per-season rule


def test_hindsight_bounds_every_strategy(seeded):
    rows = {
        r.strategy: r
        for r in B.run_season(
            seeded, SEASON, ["optimizer", "greedy", "random", "hindsight"], weeks=WEEKS, trials=3
        )
    }
    ceiling = rows["hindsight"].total
    for name in ("optimizer", "greedy", "random"):
        assert rows[name].total <= ceiling


def test_the_random_baseline_is_reproducible_from_its_seed(seeded):
    kw = dict(weeks=WEEKS, trials=5, strategies=["random"])
    a = B.run_season(seeded, SEASON, seed=1, **kw)[0]
    b = B.run_season(seeded, SEASON, seed=1, **kw)[0]
    c = B.run_season(seeded, SEASON, seed=2, **kw)[0]
    assert [p.player_id for p in a.picks] == [p.player_id for p in b.picks]
    assert a.total != c.total or [p.player_id for p in a.picks] != [p.player_id for p in c.picks]


def test_a_backtest_never_touches_recorded_picks(seeded):
    """The harness keeps its state in memory; a replay must not be able to
    corrupt the real pool state it happens to share a database with."""
    with seeded:
        seeded.execute(
            "INSERT INTO my_picks(season, week, slot, player_id, player_name, recorded_at)"
            " VALUES (?,?,?,?,?,?)",
            (SEASON, 1, "QB", "AAA-QB1", "AAA QB1", "now"),
        )
    B.run_season(seeded, SEASON, ["optimizer", "greedy"], weeks=WEEKS)
    rows = seeded.execute("SELECT player_id FROM my_picks").fetchall()
    assert [r[0] for r in rows] == ["AAA-QB1"]


def test_the_optimizer_spends_the_star_in_his_best_week_and_greedy_does_not(make_proj):
    """The whole premise of the assignment approach, as the design doc puts it:
    greedy 'burns Josh Allen in week 1 against a top defense when week 9 offers
    him against the league's worst.'"""
    rows = []
    for wk, star, role in [(1, 1.0, 0.9), (2, 2.0, 0.2)]:
        rows += [proj_row("star", "Star", "QB", wk, star), proj_row("role", "Role", "QB", wk, role)]
    proj = make_proj(rows)
    frames = {1: proj, 2: proj}
    actuals = {(1, "star"): 1.0, (2, "star"): 5.0, (1, "role"): 1.0, (2, "role"): 0.0}
    greedy = B.replay(frames, actuals, 2024, "greedy", weeks=[1, 2])
    optimal = B.replay(frames, actuals, 2024, "optimizer", weeks=[1, 2], discount=1.0)
    assert [p.player_id for p in greedy.picks if p.slot == "QB"] == ["star", "role"]
    assert [p.player_id for p in optimal.picks if p.slot == "QB"] == ["role", "star"]
    assert optimal.total > greedy.total


def test_an_alternative_projection_model_can_be_swapped_in(seeded):
    """Candidate models are compared against the shipped one on identical frozen
    data; that seam is what the experiment branches hang off."""

    def half_speed(frames, from_week=1, role_source=None):
        out = P.build_projections(frames, from_week, role_source=role_source)
        return out.assign(lam=out.lam.astype(float) / 2)

    frames = B.weekly_projections(seeded, SEASON, WEEKS, builder=half_speed)
    assert all(
        (
            f.lam
            <= P.build_projections(
                P.load_frames(seeded, SEASON, as_of_week=w), w, role_source="usage"
            ).lam.max()
        ).all()
        for w, f in frames.items()
    )
    picks = B.replay(frames, B.actual_tds(seeded, SEASON), SEASON, "greedy", weeks=WEEKS)
    assert picks.players_used == len(picks.picks)  # still a valid replay


# --- config overrides -------------------------------------------------------
def test_override_restores_values_even_when_the_body_raises():
    before = config.FUTURE_DISCOUNT
    with pytest.raises(RuntimeError):
        with config.override(FUTURE_DISCOUNT=0.5):
            assert config.FUTURE_DISCOUNT == 0.5
            raise RuntimeError("boom")
    assert config.FUTURE_DISCOUNT == before


def test_override_rejects_unknown_names():
    """A typo'd sweep parameter that silently changed nothing would report a
    flat surface and read as a finding."""
    with pytest.raises(KeyError, match="FUTURE_DISCOUNTT"):
        with config.override(FUTURE_DISCOUNTT=0.5):
            pass


from tests.conftest import proj_row  # noqa: E402
