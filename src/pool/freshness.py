"""Per-season import status, data coverage and model fallback reporting."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

from . import config, db, scoring, state

FEEDS = {
    "schedule": "games",
    "player_stats": "player_weeks",
    "rosters": "rosters",
    "injuries": "injuries",
    "depth_charts": "depth_charts",
    "touchdowns": "game_results",
}


def utc_now(now: datetime | None = None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    if now.tzinfo is None:
        from zoneinfo import ZoneInfo

        now = now.replace(tzinfo=ZoneInfo(config.TIMEZONE))
    return now.astimezone(UTC)


def feed_coverage(conn: sqlite3.Connection, season: int, feed: str) -> dict:
    table = FEEDS[feed]
    count = conn.execute(f"SELECT COUNT(*) FROM {table} WHERE season = ?", (season,)).fetchone()[0]
    result = {"rows": count}
    if feed == "depth_charts":
        result["as_of"] = conn.execute(
            "SELECT MAX(as_of) FROM depth_charts WHERE season = ?", (season,)
        ).fetchone()[0]
    else:
        reg = " AND game_type = 'REG'" if feed == "schedule" else ""
        result["weeks"] = [
            r[0]
            for r in conn.execute(
                f"SELECT DISTINCT week FROM {table} WHERE season = ?{reg} ORDER BY week", (season,)
            )
        ]
    if feed == "touchdowns":
        cov = scoring.coverage(conn, season)
        result["complete_games"] = int(cov.complete.sum())
        result["scheduled_games"] = len(cov)
    return result


def record_status(
    conn: sqlite3.Connection,
    season: int,
    feed: str,
    outcome: str,
    attempt: str,
    failure: str | None = None,
    source_timestamp: str | None = None,
):
    success = utc_now().isoformat(timespec="seconds") if outcome == "success" else None
    with conn:
        conn.execute(
            """INSERT INTO feed_status
            (season, feed, last_attempt, last_success, outcome, coverage, source_timestamp, failure)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(season, feed) DO UPDATE SET last_attempt = excluded.last_attempt,
            last_success = COALESCE(excluded.last_success, feed_status.last_success),
            outcome = excluded.outcome, coverage = excluded.coverage,
            source_timestamp = CASE WHEN excluded.outcome = 'success' THEN excluded.source_timestamp
                                    ELSE feed_status.source_timestamp END,
            failure = excluded.failure""",
            (
                season,
                feed,
                attempt,
                success,
                outcome,
                json.dumps(feed_coverage(conn, season, feed)),
                source_timestamp,
                failure,
            ),
        )


def report(conn: sqlite3.Connection, season: int, week: int, now: datetime | None = None):
    """Return compact feed rows and separate warnings for age, coverage and fallbacks."""
    instant = utc_now(now)
    eastern = state.eastern_now(instant)
    games = db.read_df(
        conn, "SELECT * FROM games WHERE season = ? AND game_type = 'REG'", (season,)
    )
    historical = bool(
        len(games)
        and games.home_score.notna().all()
        and games.away_score.notna().all()
        and max(games.kickoff) < eastern.isoformat()
    )
    rows, warnings = [], []
    for feed in FEEDS:
        status = conn.execute(
            "SELECT * FROM feed_status WHERE season = ? AND feed = ?", (season, feed)
        ).fetchone()
        cov = feed_coverage(conn, season, feed)
        outcome = status["outcome"] if status else "legacy / never fetched"
        last = status["last_success"] if status else None
        age = "unknown"
        if last:
            delta = instant - datetime.fromisoformat(last)
            age = f"{max(0, delta.total_seconds()) / 3600:.1f}h"
            if not historical and delta > timedelta(hours=config.FRESHNESS_HOURS[feed]):
                warnings.append(f"{feed}: stale ({age}; limit {config.FRESHNESS_HOURS[feed]:g}h).")
        if status and outcome in ("failed", "missing"):
            warnings.append(f"{feed}: {outcome}: {status['failure']}")
        if not cov["rows"]:
            warnings.append(f"{feed}: no local {season} coverage.")
        elif not last:
            warnings.append(
                f"{feed}: legacy import age unknown; refresh to establish UTC metadata."
            )
        if feed in ("schedule", "rosters", "injuries") and week not in cov["weeks"]:
            warnings.append(f"{feed}: no week {week} coverage.")
        if feed == "player_stats":
            needed = games[(games.week < week) & games.home_score.notna()].week.unique()
            missing = sorted(set(needed) - set(cov["weeks"]))
            if missing:
                warnings.append(f"player_stats: missing completed weeks {missing}.")
        rows.append(
            dict(
                feed=feed,
                outcome=outcome,
                age=age,
                coverage=cov,
                last_attempt=status["last_attempt"] if status else None,
                last_success=last,
                source_timestamp=status["source_timestamp"] if status else None,
                failure=status["failure"] if status else None,
            )
        )
    current = games[games.week == week]
    if len(current) and not current.kickoff_known.all():
        warnings.append(
            "Unconfirmed kickoff times: excluded from current-week advice; "
            "future planning uses estimates."
        )
    if len(current) and all(
        not r.kickoff_known
        or eastern
        >= datetime.fromisoformat(r.kickoff) - timedelta(minutes=config.PICK_DEADLINE_MINUTES)
        for r in current.itertuples()
    ):
        warnings.append(
            "All confirmed pick deadlines have passed or kickoffs are unresolved; "
            "future assignments remain available."
        )
    if len(current) and (current.total_line.isna() | current.spread_line.isna()).any():
        warnings.append("Missing betting lines: using team/league scoring estimates.")
    for yr in (season - 1, season):
        cov = scoring.coverage(conn, yr)
        completed = db.read_df(
            conn,
            "SELECT game_id FROM games WHERE season = ? "
            "AND game_type = 'REG' AND home_score IS NOT NULL "
            "AND away_score IS NOT NULL",
            (yr,),
        )
        missing = set(completed.game_id) - set(cov.loc[cov.complete.eq(1), "game_id"])
        has_stats = conn.execute(
            "SELECT 1 FROM player_weeks WHERE season = ? LIMIT 1", (yr,)
        ).fetchone()
        uncovered_stats = conn.execute(
            "SELECT 1 FROM player_weeks p LEFT JOIN game_results r ON r.game_id = p.game_id "
            "WHERE p.season = ? AND (r.complete IS NULL OR r.complete = 0) LIMIT 1",
            (yr,),
        ).fetchone()
        if missing or uncovered_stats or (has_stats and cov.empty):
            warnings.append(
                f"{yr}: incomplete touchdown history; model uses legacy offensive-TD "
                "estimates where coverage is missing. Refresh before finalized comparisons."
            )
    if not feed_coverage(conn, season, "depth_charts")["rows"]:
        warnings.append("Depth chart missing: model uses neutral role multipliers.")
    if not feed_coverage(conn, season, "injuries")["rows"]:
        warnings.append("Injuries missing: model assumes availability.")
    if not feed_coverage(conn, season, "rosters")["rows"]:
        warnings.append("Roster missing: player pool falls back to latest stat-line teams.")
    legacy = db.get_meta(conn, f"last_refresh_{season}")
    if legacy and "+" not in legacy and not legacy.endswith("Z"):
        warnings.append(f"Legacy refresh timestamp {legacy}: timezone unspecified.")
    return rows, list(dict.fromkeys(warnings))
