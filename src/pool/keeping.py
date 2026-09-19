"""Your own records, kept: a copy of the database, and your picks as a plain file.

The nflverse feeds can be downloaded again. Your recorded picks, the pool's archived reports
and the decisions `pool week` saved exist only in the database, so they are what a backup is
for.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import closing
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
    and published only once it has passed `integrity_check`. Linking the verified copy to
    its final name atomically refuses an existing destination.
    """
    if not source.is_file():
        raise FileNotFoundError(f"No database at {source}")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"{target} already exists; not overwritten")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".partial", dir=target.parent)
    partial = Path(name)
    try:
        os.close(fd)
        with (
            closing(sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro", uri=True)) as src,
            closing(sqlite3.connect(partial)) as dst,
        ):
            src.backup(dst)
            check = dst.execute("PRAGMA integrity_check").fetchone()[0]
            if check != "ok":
                raise RuntimeError(f"The copy failed its integrity check: {check}")
        os.link(partial, target)
    finally:
        partial.unlink(missing_ok=True)


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
