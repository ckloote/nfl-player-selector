"""The small 2026 database most everyday tests run on, the play-by-play rows that score
it, and a way to record picks into it.
"""

from datetime import datetime

from pool import db, state


def play(gid="g1", pid="r1", play_id=1, **kw):
    row = dict(
        game_id=gid,
        play_id=play_id,
        touchdown=1,
        pass_touchdown=0,
        td_player_id=pid,
        passer_player_id=None,
        two_point_attempt=0,
        extra_point_attempt=0,
        play_type="run",
        desc="TOUCHDOWN",
        total_home_score=28,
        total_away_score=7,
        td_team="A",
        td_player_name=pid,
        posteam="A",
    )
    return row | kw


def end(gid="g1", home=28, away=7, **kw):
    return (
        play(
            gid,
            None,
            999,
            touchdown=0,
            desc="END GAME",
            play_type=None,
            total_home_score=home,
            total_away_score=away,
        )
        | kw
    )


def local_db(tmp_path):
    """Week 1 of 2026: a Thursday game that finished 28-7, a Sunday game, and week 2's
    game still to come, with five players. Yields `(conn, path)`."""
    path = tmp_path / "workflow.db"
    conn = db.connect(path)
    with conn:
        conn.executemany(
            "INSERT INTO games(game_id, season, week, game_type, kickoff, "
            "home_team, away_team, home_score, away_score, kickoff_known) "
            "VALUES (?, 2026, ?, 'REG', ?, ?, ?, ?, ?, 1)",
            [
                ("g1", 1, "2026-09-10T20:15", "A", "B", 28, 7),
                ("g2", 1, "2026-09-13T13:00", "C", "D", 0, 0),
                ("g3", 2, "2026-09-20T13:00", "A", "B", None, None),
            ],
        )
        conn.executemany(
            "INSERT INTO rosters(season, week, player_id, player_name, position, "
            "team, status) VALUES (2026, 1, ?, ?, ?, ?, ?)",
            [
                ("q1", "Quarter One", "QB", "A", "ACT"),
                ("q2", "Quarter Two", "QB", "A", "ACT"),
                ("r1", "Runner One", "RB", "A", "RES"),
                ("f1", "Flex One", "WR", "C", "ACT"),
                ("f2", "Flex Two", "TE", "A", "ACT"),
            ],
        )
    yield conn, path
    conn.close()


def record(conn, **slots):
    pool = state.historical_pool(conn, 2026, 1).set_index("player_id")
    return state.record_picks(
        conn,
        2026,
        1,
        [
            dict(
                slot=slot,
                player_id=pid,
                player_name=pool.loc[pid, "player_name"],
                position=pool.loc[pid, "position"],
            )
            for slot, pid in slots.items()
        ],
        now=datetime(2026, 9, 14),
    )
