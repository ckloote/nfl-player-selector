"""Content-addressed normalized inputs; observation time is availability, never source time."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import zlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from . import config, db, freshness

SCHEMA_VERSION = 1
TABLES = {
    **{f: (t,) for f, t in freshness.FEEDS.items()},
    "touchdowns": ("touchdown_credits", "game_results"),
}


def timestamp(value: str | datetime) -> str:
    instant = datetime.fromisoformat(value) if isinstance(value, str) else value
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError(f"Timestamp {value!r} must include a timezone (Z or UTC offset).")
    return instant.astimezone(UTC).isoformat(timespec="microseconds")


def decision_times(path: str | Path) -> dict[tuple[int, int], str]:
    frame = pd.read_csv(path)
    if not {"season", "week", "decision_at"} <= set(frame):
        raise ValueError("decision-times CSV requires season,week,decision_at columns")
    out = {}
    for row in frame.itertuples():
        key = (int(row.season), int(row.week))
        if key in out:
            raise ValueError(f"Duplicate decision timestamp for {key}")
        out[key] = timestamp(row.decision_at)
    return out


def archive(
    conn: sqlite3.Connection,
    season: int,
    feed: str,
    *,
    observed_at: str | datetime | None = None,
    source_timestamp: str | None = None,
) -> int:
    """Archive the current normalized replacement, inside its caller's transaction.

    The optional observation time supports import integrations and deterministic tests.
    Refresh always supplies the actual completion time, never a source timestamp.
    """
    payload = {}
    for table in TABLES[feed]:
        if table == "touchdown_credits":
            frame = db.read_df(
                conn,
                "SELECT * FROM touchdown_credits WHERE game_id IN "
                "(SELECT game_id FROM game_results WHERE season = ?)",
                (season,),
            )
        else:
            frame = db.read_df(conn, f"SELECT * FROM {table} WHERE season = ?", (season,))
        # Import time belongs to observations, not content identity.
        if table == "game_results":
            frame = frame.drop(columns="imported_at")
        frame = frame.sort_values(list(frame.columns), na_position="first").reset_index(drop=True)
        payload[table] = {"columns": list(frame.columns), "rows": list(db._rows(frame))}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    digest = hashlib.sha256(raw).hexdigest()
    with db.transaction(conn):
        conn.execute(
            "INSERT OR IGNORE INTO input_payloads VALUES (?, ?, 'zlib-json', ?)",
            (digest, SCHEMA_VERSION, zlib.compress(raw)),
        )
        # Stamped after payload preparation: data is available at successful commit.
        stamp = timestamp(observed_at or datetime.now(UTC))
        cursor = conn.execute(
            "INSERT INTO input_observations "
            "(season, feed, observed_at, source_timestamp, content_hash, coverage, schema_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                season,
                feed,
                stamp,
                source_timestamp,
                digest,
                json.dumps(freshness.feed_coverage(conn, season, feed), sort_keys=True),
                SCHEMA_VERSION,
            ),
        )
    return cursor.lastrowid


def coverage(conn: sqlite3.Connection) -> pd.DataFrame:
    return db.read_df(
        conn,
        "SELECT season, feed, COUNT(*) AS observations, "
        "MIN(observed_at) AS earliest, MAX(observed_at) AS latest "
        "FROM input_observations GROUP BY season, feed ORDER BY season, feed",
    )


def restore(
    conn: sqlite3.Connection, season: int, decision_at: str | datetime, week: int
) -> tuple[sqlite3.Connection, list[dict]]:
    """Resolve inputs into an isolated database. Never consult live tables for a fallback."""
    stamp = timestamp(decision_at)
    restored = db.connect(":memory:")
    provenance = []
    try:
        for yr in (season - 1, season):
            for feed, tables in TABLES.items():
                row = conn.execute(
                    "SELECT o.*, p.payload, p.codec FROM input_observations o "
                    "JOIN input_payloads p "
                    "USING(content_hash) WHERE season = ? AND feed = ? AND observed_at <= ? "
                    "ORDER BY observed_at DESC, observation_id DESC LIMIT 1",
                    (yr, feed, stamp),
                ).fetchone()
                essential = feed == "schedule" or (
                    feed in ("player_stats", "touchdowns") and (yr < season or week > 1)
                )
                if row is None:
                    if essential:
                        raise ValueError(
                            f"Missing snapshot: {yr} {feed} at or before {stamp}. "
                            "Import and archive essential history before the decision; "
                            "legacy rows have no historical observation time."
                        )
                    provenance.append(
                        dict(
                            season=yr,
                            feed=feed,
                            missing=True,
                            age_hours=None,
                            stale=None,
                            observation_id=None,
                            observed_at=None,
                            content_hash=None,
                        )
                    )
                    continue
                if row["schema_version"] != SCHEMA_VERSION or row["codec"] != "zlib-json":
                    raise ValueError("Unsupported snapshot payload schema or codec")
                raw = zlib.decompress(row["payload"])
                if hashlib.sha256(raw).hexdigest() != row["content_hash"]:
                    raise ValueError("Snapshot payload hash mismatch")
                payload = json.loads(raw)
                with db.transaction(restored):
                    for table in tables:
                        entry = payload[table]
                        frame = pd.DataFrame(entry["rows"], columns=entry["columns"])
                        if table == "game_results":
                            frame["imported_at"] = row["observed_at"]
                        restored.executemany(
                            f"INSERT INTO {table} ({','.join(frame.columns)}) "
                            f"VALUES ({','.join('?' for _ in frame.columns)})",
                            db._rows(frame),
                        )
                age = (
                    datetime.fromisoformat(stamp) - datetime.fromisoformat(row["observed_at"])
                ).total_seconds() / 3600
                provenance.append(
                    dict(
                        season=yr,
                        feed=feed,
                        missing=False,
                        age_hours=age,
                        stale=age > config.FRESHNESS_HOURS[feed],
                        observation_id=row["observation_id"],
                        observed_at=row["observed_at"],
                        source_timestamp=row["source_timestamp"],
                        content_hash=row["content_hash"],
                        coverage=json.loads(row["coverage"]),
                    )
                )
        return restored, provenance
    except BaseException:
        restored.close()
        raise
