"""Pool state: my picks, used players, name lookup, current week."""

from __future__ import annotations

import difflib
import re
import sqlite3
from datetime import datetime, timedelta
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
    if slot not in config.SLOTS:
        raise PickError(f"unknown slot {slot!r}; expected one of {list(config.SLOTS)}")
    if position not in config.SLOTS[slot]:
        raise PickError(f"{player_name} is a {position}; slot {slot} takes {config.SLOTS[slot]}")
    prior = conn.execute(
        "SELECT week, slot FROM my_picks WHERE season = ? AND player_id = ? "
        "AND NOT (week = ? AND slot = ?)",
        (season, player_id, week, slot),
    ).fetchone()
    if prior:
        raise PickError(f"{player_name} was already used in week {prior['week']} ({prior['slot']})")
    with conn:
        conn.execute(
            "INSERT INTO my_picks(season, week, slot, player_id, player_name, recorded_at) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(season, week, slot) DO UPDATE SET "
            "player_id = excluded.player_id, player_name = excluded.player_name, "
            "recorded_at = excluded.recorded_at",
            (
                season,
                week,
                slot,
                player_id,
                player_name,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )


def remove_pick(conn: sqlite3.Connection, season: int, week: int, slot: str) -> bool:
    with conn:
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
