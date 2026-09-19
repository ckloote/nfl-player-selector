"""What each recorded pick has scored, read from the results as they stand.

Nothing here is stored. A pick's touchdowns are read from the game results whenever they
are asked for, so a result corrected upstream shows on the next read and there is no step
to run first. They used to be kept in `my_picks.tds`, which only `pool score` filled, so
`pool picks` could say "not scored" about a game that had finished; that column stays in
the schema, and nothing reads it.

Standings, the rival predictions and `pool week` score picks through the same four
functions, which is why every view of a pick agrees. None of this is in the decision
closure: what a decision banks is recorded with it, not re-read from here.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import pandas as pd

from . import db, scoring, state


@dataclass(frozen=True)
class PickScore:
    game_id: str | None
    tds: int | None
    pending: str
    note: str = ""  # a final score that is worth explaining; never a reason to wait


@dataclass(frozen=True)
class ScoreBoard:
    season: int
    games: pd.DataFrame
    credits: dict[tuple[str, str], int]


def score_board(conn: sqlite3.Connection, season: int) -> ScoreBoard:
    """Read coverage and credits once, in the same database snapshot."""
    with db.transaction(conn):
        games = scoring.coverage(conn, season).set_index("game_id")
        credits = scoring.touchdown_totals(conn, season)
    return ScoreBoard(season, games, credits.set_index(["game_id", "player_id"]).pool_td.to_dict())


def resolve_pick_game(
    conn: sqlite3.Connection, season: int, week: int, player_id: str, recorded: str | None = None
) -> str | None:
    """Prefer this week's actual appearance to the game recorded with the pick."""
    stat = conn.execute(
        "SELECT game_id, team FROM player_weeks WHERE season = ? AND week = ? AND player_id = ?",
        (season, week, player_id),
    ).fetchone()
    gid = stat["game_id"] if stat else None
    if stat and not gid:
        actual_game = db.resolve_game(conn, season, week, stat["team"])
        gid = actual_game["game_id"] if actual_game else None
    return gid or recorded


def score_pick(board: ScoreBoard, week: int, player_id: str, game_id: str | None) -> PickScore:
    """A final zero is a score; absent or incomplete coverage is not.

    A pick with no game at all is the pool's own zero once that week has finished: the
    player did not play, so he scored nothing, and the pool treats a pick on a player who
    is not playing as worth nothing rather than as unfinished business. Waiting instead
    left one such pick holding the whole season at provisional forever.

    The zero waits for the week's games to be final, because before that "no game found"
    and "no game yet" are the same silence. It also carries a note, because the other way
    to reach it is a team abbreviation the schedule does not know -- which would otherwise
    zero every pick on that team without a word.
    """
    if game_id not in board.games.index or int(board.games.loc[game_id, "week"]) != week:
        played = board.games[board.games.week == week]
        if len(played) and played.complete.all():
            return PickScore(game_id, 0, "", "no game that week; scored zero")
        return PickScore(game_id, None, "game unresolved")
    reason = "" if board.games.loc[game_id, "complete"] else board.games.loc[game_id, "reason"]
    value = None if reason else int(board.credits.get((game_id, player_id), 0))
    return PickScore(game_id, value, reason)


def resolved_through(conn: sqlite3.Connection, season: int) -> int | None:
    """Last complete week in the unbroken prefix starting at week one."""
    complete = set(scoring.complete_weeks(conn, season))
    week = 0
    while week + 1 in complete:
        week += 1
    return week or None


def pick_results(conn: sqlite3.Connection, season: int, week: int | None = None) -> pd.DataFrame:
    """Recorded picks with `tds` and `pending_reason`, scored now; writes nothing.

    `tds` is empty exactly when `pending_reason` says why, and `game_id` is the game the
    pick was scored from, which is where the player actually played when that is known.
    """
    if week is not None:
        state.validate_week(conn, season, week)
    picks = state.picks(conn, season)
    if week is not None:
        picks = picks[picks.week == week].copy()
    board = score_board(conn, season)
    games, scores, reasons = [], [], []
    for row in picks.itertuples():
        state.validate_week(conn, season, int(row.week))
        gid = resolve_pick_game(conn, season, row.week, row.player_id, row.game_id)
        result = score_pick(board, row.week, row.player_id, gid)
        games.append(gid)
        scores.append(result.tds)
        reasons.append(result.pending)
    picks["game_id"] = games
    picks["tds"] = pd.Series(scores, index=picks.index, dtype="float64")
    picks["pending_reason"] = reasons
    return picks
