"""Pool state: my picks, used players, name lookup, current week."""

from __future__ import annotations

import difflib
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from . import config, db


class PickError(ValueError):
    pass


def picks(conn: sqlite3.Connection, season: int) -> pd.DataFrame:
    return db.read_df(
        conn, "SELECT * FROM my_picks WHERE season = ? ORDER BY week, slot", (season,)
    )


def used_ids(conn: sqlite3.Connection, season: int) -> set[str]:
    rows = conn.execute("SELECT player_id FROM my_picks WHERE season = ?", (season,)).fetchall()
    return {r[0] for r in rows}


def locked_by_slot(conn: sqlite3.Connection, season: int) -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = {slot: {} for slot in config.SLOTS}
    for r in conn.execute("SELECT week, slot, player_id FROM my_picks WHERE season = ?", (season,)):
        out[r["slot"]][int(r["week"])] = r["player_id"]
    return out


def record_pick(
    conn: sqlite3.Connection,
    season: int,
    week: int,
    slot: str,
    player_id: str,
    player_name: str,
    position: str,
) -> None:
    record_picks(
        conn,
        season,
        week,
        [
            {
                "slot": slot,
                "player_id": player_id,
                "player_name": player_name,
                "position": position,
            }
        ],
    )


def historical_pool(conn: sqlite3.Connection, season: int, week: int) -> pd.DataFrame:
    """Prefer the requested week's stats, then roster history as of that week.

    Inactive players are retained. Later roster changes cannot move a historical pick.
    """
    stats = db.read_df(
        conn, "SELECT * FROM player_weeks WHERE season = ? AND week <= ?", (season, week)
    )
    roster = db.read_df(
        conn, "SELECT * FROM rosters WHERE season = ? AND week <= ?", (season, week)
    )
    stats["priority"] = (stats.week == week).astype(int) + 1
    roster["priority"] = 1
    both = pd.concat([roster, stats], ignore_index=True)
    return (
        both.sort_values(["week", "priority"])
        .drop_duplicates("player_id", keep="last")
        .reset_index(drop=True)
    )


def record_picks(
    conn: sqlite3.Connection,
    season: int,
    week: int,
    entries: list[dict],
    now: datetime | None = None,
) -> list[str]:
    """Validate the whole submission before atomically replacing any slots."""
    validate_week(conn, season, week)
    decision = eastern_now(now)
    warnings = []
    slots = [e["slot"] for e in entries]
    if len(set(slots)) != len(slots):
        raise PickError("each slot may only be supplied once")
    ids = set()
    prepared = []
    history = historical_pool(conn, season, week).set_index("player_id")
    for e in entries:
        slot, pid, name, position = (e[k] for k in ("slot", "player_id", "player_name", "position"))
        if pid in history.index:
            position = history.loc[pid, "position"]
        if slot not in config.SLOTS:
            raise PickError(f"unknown slot {slot!r}; expected one of {list(config.SLOTS)}")
        if position not in config.SLOTS[slot]:
            raise PickError(f"{name} is a {position}; slot {slot} takes {config.SLOTS[slot]}")
        prior = conn.execute(
            "SELECT week, slot FROM my_picks WHERE season = ? AND player_id = ? "
            "AND NOT (week = ? AND slot = ?)",
            (season, pid, week, slot),
        ).fetchone()
        if prior or pid in ids:
            where = f"week {prior['week']} ({prior['slot']})" if prior else "this submission"
            raise PickError(f"{name} was already used in {where}")
        ids.add(pid)
        info = history.loc[pid] if pid in history.index else e
        team = info.get("team")
        game = db.resolve_game(conn, season, week, team)
        stat_game = conn.execute(
            "SELECT g.* FROM player_weeks p JOIN games g ON g.game_id = p.game_id "
            "WHERE p.season = ? AND p.week = ? AND p.player_id = ? "
            "AND g.season = p.season AND g.week = p.week AND g.game_type = 'REG'",
            (season, week, pid),
        ).fetchone()
        game = stat_game or game
        if game is None:
            warnings.append(f"{name}: game unresolved; apparent unavailability (bye or no team).")
        elif not game["kickoff_known"]:
            warnings.append(f"{name}: kickoff unconfirmed; deadline cannot be verified.")
        elif decision >= datetime.fromisoformat(game["kickoff"]) - timedelta(
            minutes=config.PICK_DEADLINE_MINUTES
        ):
            warnings.append(
                f"{name}: deadline has passed; recording a historical entry/correction."
            )
        status = info.get("status")
        injury = conn.execute(
            "SELECT report_status FROM injuries WHERE season = ? AND week = ? AND player_id = ?",
            (season, week, pid),
        ).fetchone()
        if (pd.notna(status) and status not in config.ACTIVE_ROSTER_STATUSES) or (
            injury and injury[0] in ("Out", "Doubtful")
        ):
            warnings.append(f"{name}: apparent unavailability; entry retained as submitted.")
        prepared.append(
            (
                season,
                week,
                slot,
                pid,
                name,
                datetime.now(UTC).isoformat(timespec="seconds"),
                game["game_id"] if game else None,
            )
        )
    # A savepoint, not `with conn:`: a plain commit would also commit whatever the
    # caller had open, so a pick could survive a failure in the same command that
    # was supposed to record its history alongside it.
    with db.transaction(conn):
        conn.executemany(
            "INSERT INTO my_picks(season, week, slot, player_id, player_name, "
            "recorded_at, game_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(season, week, slot) DO UPDATE SET "
            "tds = CASE WHEN my_picks.player_id = excluded.player_id THEN my_picks.tds END, "
            "game_id = CASE WHEN my_picks.player_id = excluded.player_id "
            "THEN COALESCE(excluded.game_id, my_picks.game_id) ELSE excluded.game_id END, "
            "player_id = excluded.player_id, player_name = excluded.player_name, "
            "recorded_at = excluded.recorded_at",
            prepared,
        )
    return warnings


def remove_pick(conn: sqlite3.Connection, season: int, week: int, slot: str) -> bool:
    validate_week(conn, season, week)
    if slot not in config.SLOTS:
        raise PickError(f"unknown slot {slot!r}")
    with db.transaction(conn):
        cur = conn.execute(
            "DELETE FROM my_picks WHERE season = ? AND week = ? AND slot = ?", (season, week, slot)
        )
    return cur.rowcount > 0


# --- name lookup ------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def find_player(
    pool: pd.DataFrame, query: str, positions: tuple[str, ...] | None = None
) -> pd.DataFrame:
    """Match a typed name against the player pool. Returns 0, 1, or several rows."""
    cand = pool if positions is None else pool[pool.position.isin(positions)]
    names = cand.player_name.map(_norm)
    q = _norm(query)
    exact = cand[names == q]
    if len(exact):
        return exact
    contains = cand[names.str.contains(rf"\b{re.escape(q)}", regex=True)]
    if len(contains):
        return contains
    close = difflib.get_close_matches(q, names.tolist(), n=5, cutoff=0.7)
    return cand[names.isin(close)]


# --- calendar ---------------------------------------------------------------
def eastern_now(now: datetime | None = None) -> datetime:
    """Naive Eastern wall-clock time — the form kickoffs are stored in.

    `None` means "now", read off this machine's clock and converted; an aware
    datetime is converted; a naive one is assumed to be Eastern already.
    """
    if now is None:
        return datetime.now(ZoneInfo(config.TIMEZONE)).replace(tzinfo=None)
    if now.tzinfo is not None:
        return now.astimezone(ZoneInfo(config.TIMEZONE)).replace(tzinfo=None)
    return now


def decision_instant(now: datetime | None = None) -> datetime:
    """The timezone-aware instant behind an Eastern wall-clock decision time.

    Deadlines compare against kickoffs, which are stored as Eastern wall clock, but
    which feed observations were available is a question about an instant. Both must
    describe the same moment, so derive one from the other rather than reading the
    clock twice. An ambiguous hour on the fall-back boundary resolves to the earlier
    offset, which can only make a decision look older than it was.
    """
    eastern = eastern_now(now)
    return eastern.replace(tzinfo=ZoneInfo(config.TIMEZONE)).astimezone(UTC)


def current_week(conn: sqlite3.Connection, season: int, now: datetime | None = None) -> int:
    """First week whose games haven't all finished (kickoff + 4h).

    Compared in Eastern time: kickoffs are stored as Eastern wall-clock, so a
    raw local `datetime.now()` would roll the week over by the caller's UTC
    offset — six hours early from Berlin, three hours late from Los Angeles.
    """
    now = eastern_now(now)
    rows = conn.execute(
        "SELECT week, MAX(kickoff) AS last FROM games WHERE season = ? AND game_type = 'REG' "
        "GROUP BY week ORDER BY week",
        (season,),
    ).fetchall()
    if not rows:
        raise PickError(f"no schedule loaded for {season}; run `pool refresh`")
    for r in rows:
        if datetime.fromisoformat(r["last"]) + timedelta(hours=4) > now:
            return int(r["week"])
    return int(rows[-1]["week"])


def validate_week(conn: sqlite3.Connection, season: int, week: int) -> int:
    weeks = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT week FROM games WHERE season = ? AND game_type = 'REG' ORDER BY week",
            (season,),
        )
    ]
    if not weeks:
        raise PickError(f"No {season} schedule loaded; run `pool refresh --season {season}` first.")
    if week not in weeks:
        raise PickError(
            f"Week {week} is outside the {season} season (weeks {weeks[0]}-{weeks[-1]})."
        )
    return week


def availability(row, week: int, now: datetime, used: set[str] | None = None) -> str:
    if used and row.player_id in used:
        return "already used"
    known = row.get("kickoff_known", 0)
    if pd.isna(known) or not known:
        if int(row.week) == week:
            return "kickoff unconfirmed"
    elif eastern_now(now) >= datetime.fromisoformat(row.kickoff) - timedelta(
        minutes=config.PICK_DEADLINE_MINUTES
    ):
        return "deadline passed"
    if row.get("avail_mult", 1) <= 0:
        return "unavailable"
    return "available"


def unavailable_cells(proj: pd.DataFrame, week: int, now: datetime) -> set[tuple[str, int]]:
    return {
        (r.player_id, int(r.week))
        for _, r in proj.iterrows()
        if availability(r, week, now) != "available"
    }
