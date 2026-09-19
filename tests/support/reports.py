"""Pool reports for the local database: who picked whom in weeks 1 and 2, imported the
way `pool report import` imports them, and enough history behind them to project from.
"""

from datetime import UTC, datetime

import pandas as pd

from pool import entrants, scoring

from .local import end, play

# g3 kicks off 2026-09-20T13:00 Eastern, which is 17:00 UTC. Every deadline below is
# stated against that instant, because it is the moment week 2 stops being unobservable.
KICKOFF = datetime(2026, 9, 20, 17, 0, tzinfo=UTC)
BEFORE_ALL = "2026-09-19T12:00:00+00:00"
AFTER_REPORT = "2026-09-19T16:00:00+00:00"
AFTER_KICKOFF = "2026-09-20T18:00:00+00:00"
REPORT_AT = "2026-09-19T15:00:00+00:00"

WEEK1 = {
    "Chris": ("Quarter One", "Runner One", "Flex One"),
    "Pat": ("Quarter One", "", "Flex Two"),
    "Jamie": ("Quarter Two", "", "Flex One"),
}
# Pat spent q1/f2 and Jamie spent q2/f1, so greedy and naive must disagree at QB and FLEX.
WEEK2 = {
    "Chris": ("Quarter One", "Runner One", "Flex One"),
    "Pat": ("Quarter Two", "Runner One", "Flex One"),
    "Jamie": ("Quarter One", "Runner One", "Flex Two"),
}


def report(conn, week, picks, observed_at, me="Chris"):
    frame = pd.DataFrame(
        [
            dict(week=week, entrant=name, slot=slot, player_name=player)
            for name, players in picks.items()
            for slot, player in zip(("QB", "RB", "FLEX"), players, strict=True)
        ]
    )
    result = entrants.import_report(
        conn,
        2026,
        week,
        frame.to_csv(index=False).encode(),
        me=me,
        allow_roster_change=True,
        observed_at=observed_at,
    )
    assert result.written, result.errors
    return result


PLAYERS = [
    ("q1", "Quarter One", "QB", "A"),
    ("q2", "Quarter Two", "QB", "A"),
    ("r1", "Runner One", "RB", "A"),
    ("f1", "Flex One", "WR", "C"),
    ("f2", "Flex Two", "TE", "A"),
]


def history(conn):
    """A prior season behind the fixture, so `pool predict record` can build real
    projections. The predictors themselves are tested on synthetic frames; this exists
    only so the command has something to run on."""
    with conn:
        conn.executemany(
            "INSERT INTO player_weeks(season, week, season_type, player_id, player_name, "
            "position, team, opponent, pass_td, rush_td, rec_td, attempts, carries, targets) "
            "VALUES (2025, ?, 'REG', ?, ?, ?, ?, 'Z', ?, ?, ?, ?, ?, ?)",
            [
                (
                    week,
                    pid,
                    name,
                    pos,
                    team,
                    *(
                        (2, 0, 0, 30, 2, 0)
                        if pos == "QB"
                        else (0, 1, 0, 0, 15, 2)
                        if pos == "RB"
                        else (0, 0, 1, 0, 0, 8)
                    ),
                )
                for week in range(1, 10)
                for pid, name, pos, team in PLAYERS
            ],
        )
        conn.executemany(
            "INSERT INTO games(game_id, season, week, game_type, kickoff, home_team, "
            "away_team, home_score, away_score, kickoff_known) "
            "VALUES (?, 2025, ?, 'REG', '2025-09-14T13:00', ?, 'Z', 21, 17, 1)",
            [(f"p{week}", week, "A") for week in range(1, 10)],
        )
        conn.executemany(
            "INSERT INTO rosters(season, week, player_id, player_name, position, team, status) "
            "VALUES (2026, 2, ?, ?, ?, ?, 'ACT')",
            PLAYERS,
        )


def reported(local):
    """The local database with week 1 scored and its report imported, so week 2 has
    rivals. Returns `(conn, path)`."""
    conn, path = local
    scoring.import_touchdowns(conn, 2026, pd.DataFrame([play(), end(), end("g2", 0, 0)]))
    history(conn)
    report(conn, 1, WEEK1, "2026-09-14T00:00:00+00:00")
    return conn, path
