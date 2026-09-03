"""Pull nflverse data into SQLite.

Fetch functions talk to the network and return pandas frames (or None when a
season's file doesn't exist yet, e.g. current-season stats before Week 1).
Transform functions are pure and unit-tested. `refresh` wires them together.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pandas as pd

from . import config, db

NFLVERSE_REPO = "nflverse-data"


# --- fetch ------------------------------------------------------------------
def _download(path: str) -> pd.DataFrame | None:
    """Download a parquet release asset; None if it doesn't exist (404)."""
    from nflreadpy.downloader import get_downloader

    try:
        return get_downloader().download(NFLVERSE_REPO, path).to_pandas()
    except Exception as exc:  # nflreadpy wraps HTTP errors in ConnectionError
        if "404" in str(exc):
            return None
        raise


def fetch_player_stats(season: int) -> pd.DataFrame | None:
    return _download(f"stats_player/stats_player_week_{season}")


def fetch_schedules() -> pd.DataFrame:
    import nflreadpy as nfl

    return nfl.load_schedules().to_pandas()


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
    time = df["gametime"].fillna("13:00")
    df["kickoff"] = pd.to_datetime(df["gameday"] + " " + time).dt.strftime("%Y-%m-%dT%H:%M")
    out = df[
        [
            "game_id",
            "season",
            "week",
            "game_type",
            "kickoff",
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
            "week": df["week"],
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
def refresh(conn: sqlite3.Connection, season: int, log=print) -> dict[str, int]:
    """Import prior-season and current-season data. Idempotent."""
    prior = season - 1
    counts: dict[str, int] = {}

    sched = transform_schedule(fetch_schedules(), [prior, season])
    for s in (prior, season):
        counts[f"games_{s}"] = db.replace_season(conn, "games", s, sched[sched.season == s])

    for s in (prior, season):
        raw = fetch_player_stats(s)
        if raw is None:
            log(f"  player stats {s}: not published yet")
            counts[f"player_weeks_{s}"] = db.replace_season(
                conn, "player_weeks", s, _empty("player_weeks", conn)
            )
            continue
        counts[f"player_weeks_{s}"] = db.replace_season(
            conn, "player_weeks", s, transform_player_stats(raw)
        )

    for s in (prior, season):
        raw = fetch_weekly_rosters(s)
        if raw is None:
            raw = fetch_season_roster(s)
        if raw is None:
            log(f"  rosters {s}: not available")
            continue
        counts[f"rosters_{s}"] = db.replace_season(conn, "rosters", s, transform_rosters(raw))

    for s in (prior, season):
        raw = fetch_injuries(s)
        if raw is None:
            log(f"  injuries {s}: not published yet")
            continue
        counts[f"injuries_{s}"] = db.replace_season(conn, "injuries", s, transform_injuries(raw))

    raw = fetch_depth_charts(season)
    if raw is None:
        log(f"  depth charts {season}: not available")
    else:
        reg = sched[(sched.season == season) & (sched.game_type == "REG")]
        max_week = int(reg.week.max()) if len(reg) else None
        counts[f"depth_charts_{season}"] = db.replace_season(
            conn, "depth_charts", season, transform_depth_charts(raw, season, max_week)
        )

    stamp = datetime.now().isoformat(timespec="seconds")
    # Per-season as well as global: backfilling an old season for a backtest
    # must not make the live season's pick sheet claim it was just refreshed.
    db.set_meta(conn, f"last_refresh_{season}", stamp)
    db.set_meta(conn, "last_refresh", stamp)
    return counts


def _empty(table: str, conn: sqlite3.Connection) -> pd.DataFrame:
    cols = [c[1] for c in conn.execute(f"PRAGMA table_info({table})")]
    return pd.DataFrame(columns=cols)
