"""Pull nflverse data into SQLite.

Fetch functions talk to the network and return pandas frames (or None when a
season's file doesn't exist yet, e.g. current-season stats before Week 1).
Transform functions are pure and unit-tested. `refresh` wires them together.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from . import config, db, freshness, scoring, snapshots, state

NFLVERSE_REPO = "nflverse-data"


# --- fetch ------------------------------------------------------------------
def _download(path: str, columns: set[str] | None = None) -> pd.DataFrame | None:
    """Download a parquet release asset; None if it doesn't exist (404)."""
    from nflreadpy.downloader import get_downloader

    try:
        downloader = get_downloader()
        stamps = []

        def capture(response, *args, **kwargs):
            stamps.append(response.headers.get("Last-Modified"))

        downloader.session.hooks.setdefault("response", []).append(capture)
        try:
            frame = downloader.download(NFLVERSE_REPO, path)
            if columns is not None:
                frame = frame.select([c for c in frame.columns if c in columns])
            out = frame.to_pandas()
            out.attrs["source_timestamp"] = stamps[-1] if stamps else None
            return out
        finally:
            downloader.session.hooks["response"].remove(capture)
    except Exception as exc:  # nflreadpy wraps HTTP errors in ConnectionError
        if "404" in str(exc):
            return None
        raise


def fetch_touchdowns(season: int) -> pd.DataFrame | None:
    columns = scoring.PBP_REQUIRED | {
        "no_play",
        "game_end",
        "td_team",
        "td_player_name",
        "passer_player_name",
        "posteam",
    }
    return _download(f"pbp/play_by_play_{season}", columns)


def fetch_player_stats(season: int) -> pd.DataFrame | None:
    return _download(f"stats_player/stats_player_week_{season}")


def fetch_schedules() -> pd.DataFrame:
    raw = _download("schedules/games")
    if raw is None:
        raise FileNotFoundError("schedule source returned 404")
    return raw


def fetch_weekly_rosters(season: int) -> pd.DataFrame | None:
    return _download(f"weekly_rosters/roster_weekly_{season}")


def fetch_season_roster(season: int) -> pd.DataFrame | None:
    return _download(f"rosters/roster_{season}")


def fetch_depth_charts(season: int) -> pd.DataFrame | None:
    return _download(f"depth_charts/depth_charts_{season}")


def fetch_injuries(season: int) -> pd.DataFrame | None:
    return _download(f"injuries/injuries_{season}")


# --- transform --------------------------------------------------------------
def transform_schedule(raw: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    df = raw[raw["season"].isin(seasons)].copy()
    df["kickoff_known"] = df["gametime"].fillna("").str.fullmatch(r"\d{2}:\d{2}").astype(int)
    time = df["gametime"].where(df.kickoff_known.eq(1), "13:00")
    df["kickoff"] = pd.to_datetime(df["gameday"] + " " + time).dt.strftime("%Y-%m-%dT%H:%M")
    out = df[
        [
            "game_id",
            "season",
            "week",
            "game_type",
            "kickoff",
            "kickoff_known",
            "weekday",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "spread_line",
            "total_line",
        ]
    ].copy()
    return out.reset_index(drop=True)


def transform_player_stats(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[raw["position"].isin(config.POSITIONS)].copy()
    df = df[df["season_type"] == "REG"]
    out = pd.DataFrame(
        {
            "season": df["season"],
            "week": df["week"],
            "season_type": df["season_type"],
            "player_id": df["player_id"],
            "player_name": df["player_display_name"],
            "position": df["position"],
            "team": df["team"],
            "opponent": df["opponent_team"],
            "game_id": df.get("game_id"),
            "pass_td": df["passing_tds"].fillna(0).astype(int),
            "rush_td": df["rushing_tds"].fillna(0).astype(int),
            "rec_td": df["receiving_tds"].fillna(0).astype(int),
            "attempts": df["attempts"].fillna(0).astype(int),
            "carries": df["carries"].fillna(0).astype(int),
            "targets": df["targets"].fillna(0).astype(int),
        }
    )
    # A player can appear twice in a week only via data quirks; keep the max.
    out = out.sort_values(["season", "week", "player_id"]).drop_duplicates(
        ["season", "week", "player_id"], keep="last"
    )
    return out.reset_index(drop=True)


def transform_rosters(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[raw["position"].isin(config.POSITIONS) & raw["gsis_id"].notna()].copy()
    if "game_type" in df.columns:
        df = df[df["game_type"].fillna("REG") == "REG"]
    out = pd.DataFrame(
        {
            "season": df["season"],
            "week": df["week"] if "week" in df else 1,
            "player_id": df["gsis_id"],
            "player_name": df["full_name"],
            "position": df["position"],
            "team": df["team"],
            "status": df["status"],
            "depth_chart_position": df.get("depth_chart_position"),
        }
    )
    out = out.drop_duplicates(["season", "week", "player_id"], keep="last")
    return out.reset_index(drop=True)


def _season_type(df: pd.DataFrame) -> pd.Series:
    """REG/POST marker, whatever the file calls it.

    Seasons from 2025 carry `season_type`; 2024 and earlier only have
    `game_type` (REG/WC/DIV/CON/SB). Reading the wrong one raises KeyError, so
    `refresh --season 2025` used to die on the prior season it always imports.
    """
    col = "season_type" if "season_type" in df.columns else "game_type"
    return df[col].where(df[col].isin(("REG",)), other="POST")


def transform_injuries(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[raw["position"].isin(config.POSITIONS) & raw["gsis_id"].notna()].copy()
    df = df[_season_type(df) == "REG"]
    out = pd.DataFrame(
        {
            "season": df["season"],
            "week": df["week"],
            "player_id": df["gsis_id"],
            "player_name": df["full_name"],
            "team": df["team"],
            "position": df["position"],
            "report_status": df["report_status"],
            "practice_status": df["practice_status"],
        }
    )
    out = out.drop_duplicates(["season", "week", "player_id"], keep="last")
    return out.reset_index(drop=True)


DEPTH_COLUMNS = ["season", "player_id", "team", "position", "rank", "as_of"]


def transform_depth_charts(
    raw: pd.DataFrame, season: int, max_week: int | None = None
) -> pd.DataFrame:
    """Latest snapshot per team; a player's rank is their best listed rank.

    Two file formats exist upstream: 2025 onward publishes dated snapshots
    (`dt`, `pos_abb`, `pos_rank`), while 2024 and earlier publish one row per
    week (`week`, `position`, `depth_team`). Reading the newer columns off an
    older file raises KeyError, which is what made `refresh --season 2024` fail.
    """
    if "pos_abb" in raw.columns:
        return _depth_from_snapshots(raw, season)
    return _depth_from_weeks(raw, season, max_week)


def _depth_from_snapshots(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    df = raw[raw["pos_abb"].isin(config.POSITIONS) & raw["gsis_id"].notna()].copy()
    if not len(df):
        return pd.DataFrame(columns=DEPTH_COLUMNS)
    latest = df.groupby("team")["dt"].transform("max")
    df = df[df["dt"] == latest]
    out = (
        df.sort_values("pos_rank")
        .groupby("gsis_id", as_index=False)
        .agg(
            team=("team", "first"),
            position=("pos_abb", "first"),
            rank=("pos_rank", "min"),
            as_of=("dt", "max"),
        )
        .rename(columns={"gsis_id": "player_id"})
    )
    out["season"] = season
    out["rank"] = out["rank"].astype(int)
    return out[DEPTH_COLUMNS].reset_index(drop=True)


def _depth_from_weeks(raw: pd.DataFrame, season: int, max_week: int | None = None) -> pd.DataFrame:
    df = raw[raw["position"].isin(config.POSITIONS) & raw["gsis_id"].notna()].copy()
    df = df[_season_type(df) == "REG"]
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df["rank"] = pd.to_numeric(df["depth_team"], errors="coerce")
    df = df.dropna(subset=["week", "rank"])
    # 2024 labels 1821 week-19 rows "REG"; trust the schedule's week count, not
    # the file's own game_type, and never hardcode 18 (pre-2021 seasons had 17).
    if max_week is not None:
        df = df[df["week"] <= max_week]
    if not len(df):
        return pd.DataFrame(columns=DEPTH_COLUMNS)
    df = df[df["week"] == df["week"].max()]
    out = (
        df.sort_values("rank")
        .groupby("gsis_id", as_index=False)
        .agg(
            team=("club_code", "first"),
            position=("position", "first"),
            rank=("rank", "min"),
            week=("week", "max"),
        )
        .rename(columns={"gsis_id": "player_id"})
    )
    out["season"] = season
    out["rank"] = out["rank"].astype(int)
    out["as_of"] = f"{season}-W" + out.pop("week").astype(int).astype(str)
    return out[DEPTH_COLUMNS].reset_index(drop=True)


# --- refresh ----------------------------------------------------------------
class RefreshResult(dict):
    """Successful row counts plus independent failed/unpublished feed outcomes."""

    def __init__(self):
        super().__init__()
        self.failures: list[str] = []


@contextmanager
def uncached_downloads():
    from nflreadpy.config import CacheMode, get_config, update_config

    previous = get_config().cache_mode
    update_config(cache_mode=CacheMode.OFF)
    try:
        yield
    finally:
        update_config(cache_mode=previous)


def _unpublished(conn, season: int, feed: str) -> bool:
    if feed not in ("player_stats", "touchdowns", "injuries"):
        return False
    first = conn.execute(
        "SELECT MIN(kickoff) FROM games WHERE season = ? AND game_type = 'REG'", (season,)
    ).fetchone()[0]
    return bool(first and state.eastern_now() < datetime.fromisoformat(first))


def refresh(conn: sqlite3.Connection, season: int, log=print) -> RefreshResult:
    """Try independent feeds uncached; retain the last dataset on fetch/parse failures."""
    result = RefreshResult()

    def attempt(s, feed, fetch, transform=None, table=None, stamp=None):
        stamp = stamp or freshness.utc_now().isoformat(timespec="seconds")
        freshness.record_status(conn, s, feed, "attempting", stamp)
        try:
            raw = fetch()
            if raw is None:
                expected = _unpublished(conn, s, feed)
                outcome = "unpublished" if expected else "missing"
                detail = "not published yet" if expected else "source returned 404 / missing file"
                freshness.record_status(conn, s, feed, outcome, stamp, detail)
                log(f"  {feed} {s}: {detail}; previous data retained")
                if not expected:
                    result.failures.append(f"{feed} {s}: {detail}")
                return
            with db.transaction(conn):
                if feed == "touchdowns":
                    count = scoring.import_touchdowns(conn, s, raw)
                else:
                    frame = transform(raw)
                    if "season" in frame and not frame.season.eq(s).all():
                        raise ValueError("feed contains a different season")
                    count = db.replace_season(conn, table, s, frame)
                    db.backfill_game_ids(conn)
                freshness.record_status(
                    conn,
                    s,
                    feed,
                    "success",
                    stamp,
                    source_timestamp=raw.attrs.get("source_timestamp"),
                )
                snapshots.archive(conn, s, feed, source_timestamp=raw.attrs.get("source_timestamp"))
            result[f"{table or 'touchdown_credits'}_{s}"] = count
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            freshness.record_status(conn, s, feed, "failed", stamp, detail)
            result.failures.append(f"{feed} {s}: {detail}")
            log(f"  {feed} {s}: FAILED: {detail}; previous data retained")

    def roster(s):
        raw = fetch_weekly_rosters(s)
        if raw is None:
            if conn.execute("SELECT 1 FROM rosters WHERE season = ? LIMIT 1", (s,)).fetchone():
                log(f"  rosters {s}: weekly file missing; retaining imported roster history")
                return None
            raw = fetch_season_roster(s)
            if raw is not None:
                log(f"  rosters {s}: using season roster fallback")
        return raw

    with uncached_downloads():
        # Fetch once, but record schedule outcomes for each season independently.
        schedule_stamp = freshness.utc_now().isoformat(timespec="seconds")
        for s in (season - 1, season):
            freshness.record_status(conn, s, "schedule", "attempting", schedule_stamp)
        try:
            schedule = fetch_schedules()
            schedule_error = None
        except Exception as exc:
            schedule, schedule_error = None, exc

        def schedules():
            if schedule_error:
                raise schedule_error
            return schedule

        for s in (season - 1, season):
            attempt(
                s,
                "schedule",
                schedules,
                lambda raw, s=s: transform_schedule(raw, [s]),
                "games",
                stamp=schedule_stamp,
            )
        for s in (season - 1, season):
            attempt(
                s,
                "player_stats",
                lambda s=s: fetch_player_stats(s),
                transform_player_stats,
                "player_weeks",
            )
            attempt(s, "rosters", lambda s=s: roster(s), transform_rosters, "rosters")
            attempt(s, "injuries", lambda s=s: fetch_injuries(s), transform_injuries, "injuries")
            weeks = conn.execute(
                "SELECT MAX(week) FROM games WHERE season = ? AND game_type = 'REG'", (s,)
            ).fetchone()[0]
            attempt(
                s,
                "depth_charts",
                lambda s=s: fetch_depth_charts(s),
                lambda raw, s=s, weeks=weeks: transform_depth_charts(raw, s, weeks),
                "depth_charts",
            )
            attempt(s, "touchdowns", lambda s=s: fetch_touchdowns(s))
    log("Refresh partially succeeded." if result.failures else "Refresh complete.")
    return result
