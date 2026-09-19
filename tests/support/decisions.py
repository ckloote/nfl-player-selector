"""A week-3 decision made on the Friday: the Thursday game played and published, the
Sunday games still to come, and a way to capture a decision at any instant.
"""

from pool import capture, config, db, state
from pool import projections as P
from pool.recommend import advise_week

from .season import SEASON, seed_season

THU_KICK, SUN_KICK = "2024-09-19T20:15", "2024-09-22T13:00"
PRE_WEEK3 = "2024-09-19T18:00:00+00:00"  # Thursday afternoon, before kickoff
POST_THU = "2024-09-20T12:00:00+00:00"  # the Thursday result is in the feed
DECISION = "2024-09-20T16:00:00+00:00"  # Friday: Thursday locked, Sunday open
POST_SUN = "2024-09-23T12:00:00+00:00"  # everything else has been played


def rows_as_dicts(conn, sql, params=()):
    return [dict(r) for r in conn.execute(sql, params)]


def insert_rows(conn, table, rows):
    if not rows:
        return
    cols = list(rows[0])
    conn.executemany(
        f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})",
        [tuple(r[c] for c in cols) for r in rows],
    )


def staged(tmp_path):
    """A week-3 decision with one game already played and the rest still to come.

    The shared fixture kicks every game off at the same hour, which is precisely the
    situation in which the live and snapshot stats rules cannot disagree. Give week 3
    a Thursday game and a Sunday game, then withhold everything from week 3 on so the
    feed can publish it in the order it really would.
    """
    conn = seed_season(db.connect(tmp_path / "parity.db"))
    with conn:
        for week, kick in ((1, "2024-09-08T13:00"), (2, "2024-09-15T13:00")):
            conn.execute(
                "UPDATE games SET kickoff = ? WHERE season = ? AND week = ?", (kick, SEASON, week)
            )
        conn.execute("UPDATE games SET kickoff = ? WHERE game_id = ?", (THU_KICK, "g2024-3-AAA"))
        conn.execute("UPDATE games SET kickoff = ? WHERE game_id = ?", (SUN_KICK, "g2024-3-CCC"))
        conn.execute(
            "UPDATE games SET kickoff = '2024-09-29T13:00' WHERE season = ? AND week = 4",
            (SEASON,),
        )
    unplayed = {
        "player_weeks": rows_as_dicts(
            conn, "SELECT * FROM player_weeks WHERE season = ? AND week >= 3", (SEASON,)
        ),
        "rosters": rows_as_dicts(
            conn, "SELECT * FROM rosters WHERE season = ? AND week >= 4", (SEASON,)
        ),
        "game_results": rows_as_dicts(
            conn, "SELECT * FROM game_results WHERE season = ? AND week >= 3", (SEASON,)
        ),
        "touchdown_credits": rows_as_dicts(
            conn,
            "SELECT t.* FROM touchdown_credits t JOIN games g USING(game_id) "
            "WHERE g.season = ? AND g.week >= 3",
            (SEASON,),
        ),
    }
    with conn:
        conn.execute("DELETE FROM player_weeks WHERE season = ? AND week >= 3", (SEASON,))
        conn.execute("DELETE FROM rosters WHERE season = ? AND week >= 4", (SEASON,))
        conn.execute(
            "DELETE FROM touchdown_credits WHERE game_id IN "
            "(SELECT game_id FROM games WHERE season = ? AND week >= 3)",
            (SEASON,),
        )
        conn.execute("DELETE FROM game_results WHERE season = ? AND week >= 3", (SEASON,))
    return conn, unplayed


def publish(conn, unplayed, which):
    """Publish the Thursday game's rows, or everything that follows them.

    Next week's roster is part of "everything that follows": a week-4 snapshot does not
    exist on the Friday of week 3, and letting one path see it would make the parity
    comparison pass for the wrong reason.
    """
    thursday = which == "thursday"
    with conn:
        for table in ("player_weeks", "game_results", "touchdown_credits"):
            rows = [r for r in unplayed[table] if (r["game_id"] == "g2024-3-AAA") == thursday]
            insert_rows(conn, table, rows)
        if not thursday:
            insert_rows(conn, "rosters", unplayed["rosters"])


def decide(conn, week, decided, used=None, locked=None):
    used = set() if used is None else used
    locked = {slot: {} for slot in config.SLOTS} if locked is None else locked
    now = state.eastern_now(decided)
    proj = P.projections_for(conn, SEASON, from_week=week)
    advice = advise_week(proj, week, used, locked, now=now)
    decision_id = capture.record_decision(
        conn, SEASON, week, proj, advice, used, locked, decision_at=decided
    )
    return decision_id, proj, advice
