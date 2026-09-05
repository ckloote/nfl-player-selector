"""SQLite storage: imported nflverse tables plus pool state."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    game_id     TEXT PRIMARY KEY,
    season      INTEGER NOT NULL,
    week        INTEGER NOT NULL,
    game_type   TEXT NOT NULL,
    kickoff     TEXT NOT NULL,      -- ISO datetime, US Eastern
    weekday     TEXT,
    home_team   TEXT NOT NULL,
    away_team   TEXT NOT NULL,
    home_score  INTEGER,
    away_score  INTEGER,
    spread_line REAL,               -- positive = home favored
    total_line  REAL
);
CREATE INDEX IF NOT EXISTS games_season_week ON games(season, week);

CREATE TABLE IF NOT EXISTS player_weeks (
    season      INTEGER NOT NULL,
    week        INTEGER NOT NULL,
    season_type TEXT NOT NULL,
    player_id   TEXT NOT NULL,
    player_name TEXT NOT NULL,
    position    TEXT NOT NULL,
    team        TEXT NOT NULL,
    opponent    TEXT,
    pass_td     INTEGER NOT NULL DEFAULT 0,
    rush_td     INTEGER NOT NULL DEFAULT 0,
    rec_td      INTEGER NOT NULL DEFAULT 0,
    attempts    INTEGER NOT NULL DEFAULT 0,
    carries     INTEGER NOT NULL DEFAULT 0,
    targets     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (season, week, player_id)
);

CREATE TABLE IF NOT EXISTS rosters (
    season      INTEGER NOT NULL,
    week        INTEGER NOT NULL,
    player_id   TEXT NOT NULL,
    player_name TEXT NOT NULL,
    position    TEXT NOT NULL,
    team        TEXT NOT NULL,
    status      TEXT,
    depth_chart_position TEXT,
    PRIMARY KEY (season, week, player_id)
);

CREATE TABLE IF NOT EXISTS injuries (
    season          INTEGER NOT NULL,
    week            INTEGER NOT NULL,
    player_id       TEXT NOT NULL,
    player_name     TEXT,
    team            TEXT,
    position        TEXT,
    report_status   TEXT,
    practice_status TEXT,
    PRIMARY KEY (season, week, player_id)
);

CREATE TABLE IF NOT EXISTS depth_charts (
    season      INTEGER NOT NULL,
    player_id   TEXT NOT NULL,
    team        TEXT NOT NULL,
    position    TEXT NOT NULL,
    rank        INTEGER NOT NULL,
    as_of       TEXT,
    PRIMARY KEY (season, player_id)
);

CREATE TABLE IF NOT EXISTS my_picks (
    season      INTEGER NOT NULL,
    week        INTEGER NOT NULL,
    slot        TEXT NOT NULL,
    player_id   TEXT NOT NULL,
    player_name TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    tds         INTEGER,           -- filled in once scored
    PRIMARY KEY (season, week, slot)
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    """Open (and initialize) the database. Use ':memory:' for tests."""
    target = ":memory:" if path == ":memory:" else Path(path or config.DB_PATH)
    if target != ":memory:":
        target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    migrate(conn)
    return conn


def replace_season(conn: sqlite3.Connection, table: str, season: int, df: pd.DataFrame) -> int:
    """Delete a season's rows from `table` and insert `df` (must match columns)."""
    cols = [c[1] for c in conn.execute(f"PRAGMA table_info({table})")]
    missing = [c for c in df.columns if c not in cols]
    if missing:
        raise ValueError(f"{table}: unexpected columns {missing}")
    with conn:
        conn.execute(f"DELETE FROM {table} WHERE season = ?", (season,))
        if len(df):
            placeholders = ",".join("?" for _ in df.columns)
            conn.executemany(
                f"INSERT INTO {table} ({','.join(df.columns)}) VALUES ({placeholders})",
                _rows(df),
            )
    return len(df)


def _rows(df: pd.DataFrame) -> Iterable[tuple]:
    clean = df.astype(object).where(pd.notna(df), None)
    for row in clean.itertuples(index=False, name=None):
        yield tuple(v.item() if hasattr(v, "item") else v for v in row)


def read_df(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    with conn:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


# Migrations deliberately do not infer scoring completeness from legacy totals.
MIGRATIONS = {
    1: [
        "ALTER TABLE games ADD COLUMN kickoff_known INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE player_weeks ADD COLUMN game_id TEXT",
        "ALTER TABLE my_picks ADD COLUMN game_id TEXT",
        """CREATE TABLE touchdown_credits (
            game_id TEXT NOT NULL, play_id INTEGER NOT NULL, player_id TEXT NOT NULL,
            kind TEXT NOT NULL CHECK(kind IN ('throwing', 'scoring')),
            player_name TEXT, team TEXT,
            PRIMARY KEY(game_id, play_id, player_id, kind))""",
        """CREATE TABLE game_results (
            game_id TEXT PRIMARY KEY, season INTEGER NOT NULL, week INTEGER NOT NULL,
            complete INTEGER NOT NULL, reason TEXT NOT NULL,
            home_score INTEGER, away_score INTEGER, imported_at TEXT NOT NULL)""",
        """CREATE TABLE feed_status (
            season INTEGER NOT NULL, feed TEXT NOT NULL,
            last_attempt TEXT NOT NULL, last_success TEXT, outcome TEXT NOT NULL,
            coverage TEXT, source_timestamp TEXT, failure TEXT,
            PRIMARY KEY(season, feed))""",
    ],
}


def migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for target, statements in MIGRATIONS.items():
        if version < target:
            with conn:
                conn.execute("BEGIN")
                for statement in statements:
                    conn.execute(statement)
                backfill_game_ids(conn)
                conn.execute(f"PRAGMA user_version = {target}")


def resolve_game(conn: sqlite3.Connection, season: int, week: int, team: str | None):
    rows = conn.execute(
        "SELECT * FROM games WHERE season = ? AND week = ? AND game_type = 'REG' "
        "AND (home_team = ? OR away_team = ?)",
        (season, week, team, team),
    ).fetchall()
    return rows[0] if len(rows) == 1 else None


def backfill_game_ids(conn: sqlite3.Connection) -> None:
    """Associate only unique regular-season matches; never guess a team/game."""
    conn.execute("""UPDATE player_weeks SET game_id = (
        SELECT g.game_id FROM games g WHERE g.season = player_weeks.season
        AND g.week = player_weeks.week AND g.game_type = 'REG'
        AND player_weeks.team IN (g.home_team, g.away_team))
        WHERE game_id IS NULL AND 1 = (
        SELECT COUNT(*) FROM games g WHERE g.season = player_weeks.season
        AND g.week = player_weeks.week AND g.game_type = 'REG'
        AND player_weeks.team IN (g.home_team, g.away_team))""")
    conn.execute("""UPDATE my_picks SET game_id = (
        SELECT game_id FROM player_weeks p WHERE p.season = my_picks.season
        AND p.week = my_picks.week AND p.player_id = my_picks.player_id)
        WHERE game_id IS NULL""")
    conn.execute("""UPDATE my_picks SET game_id = (
        SELECT g.game_id FROM rosters r JOIN games g ON g.season = r.season
        AND g.week = r.week AND r.team IN (g.home_team, g.away_team)
        AND g.game_type = 'REG' WHERE r.season = my_picks.season
        AND r.week = my_picks.week AND r.player_id = my_picks.player_id)
        WHERE game_id IS NULL AND 1 = (
        SELECT COUNT(*) FROM rosters r JOIN games g ON g.season = r.season
        AND g.week = r.week AND r.team IN (g.home_team, g.away_team)
        AND g.game_type = 'REG' WHERE r.season = my_picks.season
        AND r.week = my_picks.week AND r.player_id = my_picks.player_id)""")
