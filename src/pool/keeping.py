"""Your own records, kept: a copy of the database, and your picks as a plain file.

The nflverse feeds can be downloaded again. Your recorded picks, the pool's archived reports
and the decisions `pool week` saved exist only in the database, so they are what a backup is
for.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from . import results, state

EXPORT_COLUMNS = [
    "season",
    "week",
    "slot",
    "player_name",
    "team",
    "position",
    "tds",
    "pending",
    "recorded_at",
    "player_id",
]


def backup(source: Path, target: Path) -> None:
    """Copy the database with SQLite's online backup, then check the copy reads back whole.

    The online backup copies a consistent state even if another command writes during
    it, which a plain file copy does not promise. The copy is written beside its final name
    and renamed only once it has passed `integrity_check`, so a failed backup never leaves a
    file that looks like a good one.
    """
    if not source.is_file():
        raise FileNotFoundError(f"No database at {source}")
    if target.exists():
        raise FileExistsError(f"{target} already exists; not overwritten")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    partial.unlink(missing_ok=True)
    src = sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)
    dst = sqlite3.connect(partial)
    try:
        src.backup(dst)
        check = dst.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        dst.close()
        src.close()
    if check != "ok":
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"The copy failed its integrity check: {check}")
    partial.rename(target)


def picks(conn: sqlite3.Connection, season: int) -> pd.DataFrame:
    """Every recorded pick of the season with what it has scored, as `pool picks` shows it.

    `tds` is empty exactly when `pending` says why. The team and position are the player's
    as of that week, the same ones `pool record` printed when the pick was logged.
    """
    scored = results.pick_results(conn, season)
    teams = {}
    for week in scored.week.unique():
        pool = state.historical_pool(conn, season, int(week))
        teams.update({(int(week), r.player_id): (r.team, r.position) for r in pool.itertuples()})
    known = [teams.get((int(r.week), r.player_id), (None, None)) for r in scored.itertuples()]
    out = scored.assign(
        team=[team for team, _ in known],
        position=[position for _, position in known],
        tds=scored.tds.astype("Int64"),
        pending=scored.pending_reason,
    )
    return out[EXPORT_COLUMNS].reset_index(drop=True)
