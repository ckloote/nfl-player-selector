"""Projection model v1: expected touchdowns for every (player, week).

lambda = base_rate * defense_mult * vegas_mult * home_mult * availability

- base_rate: prior-season TD/game regressed toward the positional mean, then
  blended with current-season observations (Bayesian shrinkage).
- defense_mult: opponent's TDs allowed to the position group, regressed.
- vegas_mult: implied team total relative to league average.
- home_mult / availability: small home edge; byes, injuries, roster status.

Everything downstream consumes only the frame returned by `build_projections`.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, db
from .config import slot_for_position


@dataclass
class Frames:
    season: int
    games: pd.DataFrame  # both seasons, REG only
    pw_prior: pd.DataFrame
    pw_cur: pd.DataFrame
    rosters: pd.DataFrame  # current season, latest week per player
    injuries: pd.DataFrame  # current season
    depth: pd.DataFrame = field(default_factory=pd.DataFrame)  # current season, latest snapshot


def load_frames(conn: sqlite3.Connection, season: int) -> Frames:
    prior = season - 1
    games = db.read_df(
        conn, "SELECT * FROM games WHERE season IN (?, ?) AND game_type = 'REG'", (prior, season)
    )
    pw = db.read_df(conn, "SELECT * FROM player_weeks WHERE season IN (?, ?)", (prior, season))
    rosters = db.read_df(conn, "SELECT * FROM rosters WHERE season = ?", (season,))
    if len(rosters):
        rosters = rosters.sort_values("week").drop_duplicates("player_id", keep="last")
    injuries = db.read_df(conn, "SELECT * FROM injuries WHERE season = ?", (season,))
    depth = db.read_df(conn, "SELECT * FROM depth_charts WHERE season = ?", (season,))
    return Frames(
        season=season,
        games=games,
        pw_prior=pw[pw.season == prior].copy(),
        pw_cur=pw[pw.season == season].copy(),
        rosters=rosters,
        injuries=injuries,
        depth=depth,
    )


# --- player pool ------------------------------------------------------------
def player_pool(
    rosters: pd.DataFrame, pw_prior: pd.DataFrame, pw_cur: pd.DataFrame
) -> pd.DataFrame:
    """Active players at pool positions with their current team.

    Prefers the current-season roster (handles offseason team changes); falls
    back to the most recent stats row when no roster is loaded.
    """
    cols = ["player_id", "player_name", "position", "team"]
    if len(rosters):
        active = rosters[rosters["status"].isin(config.ACTIVE_ROSTER_STATUSES)]
        return active[cols].drop_duplicates("player_id").reset_index(drop=True)
    pw = pd.concat([pw_prior, pw_cur], ignore_index=True)
    latest = pw.sort_values(["season", "week"]).drop_duplicates("player_id", keep="last")
    return latest[cols].reset_index(drop=True)


# --- baseline rates ---------------------------------------------------------
def _usage_per_game(df: pd.DataFrame) -> pd.Series:
    pos = df["position"]
    usage = np.where(
        pos == "QB",
        df["attempts"],
        np.where(pos == "RB", df["carries"] + df["targets"], df["targets"]),
    )
    return pd.Series(usage, index=df.index, dtype=float)


def per_player_totals(pw: pd.DataFrame) -> pd.DataFrame:
    """games, tds, usage per player from a player_weeks frame."""
    if not len(pw):
        return pd.DataFrame(columns=["player_id", "position", "games", "tds", "usage"]).set_index(
            "player_id"
        )
    df = pw.copy()
    df["tds"] = df["pass_td"] + df["rush_td"] + df["rec_td"]
    df["usage"] = _usage_per_game(df)
    agg = df.groupby("player_id").agg(
        position=("position", "last"),
        games=("week", "nunique"),
        tds=("tds", "sum"),
        usage=("usage", "sum"),
    )
    return agg


def positional_means(pw_prior: pd.DataFrame) -> dict[str, float]:
    """Mean TDs/game among 'regulars' at each position (the shrinkage target)."""
    totals = per_player_totals(pw_prior)
    means: dict[str, float] = {}
    for pos in config.POSITIONS:
        sub = totals[totals.position == pos]
        if not len(sub):
            means[pos] = 0.0
            continue
        regs = sub[
            (sub.games >= config.REGULAR_MIN_GAMES)
            & (sub.usage / sub.games >= config.REGULAR_MIN_USAGE[pos])
        ]
        pool = regs if len(regs) else sub
        means[pos] = float((pool.tds / pool.games).mean())
    return means


def baseline_rates(
    pool: pd.DataFrame, pw_prior: pd.DataFrame, pw_cur: pd.DataFrame, pos_means: dict[str, float]
) -> pd.DataFrame:
    prior = per_player_totals(pw_prior)
    cur = per_player_totals(pw_cur)
    out = pool.copy()
    out["prior_games"] = out.player_id.map(prior.games).fillna(0).astype(float)
    out["prior_tds"] = out.player_id.map(prior.tds).fillna(0).astype(float)
    out["cur_games"] = out.player_id.map(cur.games).fillna(0).astype(float)
    out["cur_tds"] = out.player_id.map(cur.tds).fillna(0).astype(float)
    mean = out.position.map(pos_means).fillna(0.0)
    k = config.PRIOR_SEASON_SHRINK_GAMES
    regressed = (out.prior_tds + k * mean) / (out.prior_games + k)
    out["prior_rate"] = np.where(out.prior_games > 0, regressed, config.NO_HISTORY_FACTOR * mean)
    w = config.PRIOR_WEIGHT_GAMES
    out["base_rate"] = (w * out.prior_rate + out.cur_tds) / (w + out.cur_games)
    return out


# --- opponent defense -------------------------------------------------------
def _allowed_by_group(pw: pd.DataFrame) -> pd.DataFrame:
    """Per (team, group): TDs allowed and games played, from opponents' stat lines."""
    if not len(pw):
        return pd.DataFrame(columns=["team", "group", "allowed", "games"])
    df = pw[pw.opponent.notna()].copy()
    df["group"] = df.position.map(slot_for_position)
    df["tds"] = df.pass_td + df.rush_td + df.rec_td
    allowed = df.groupby(["opponent", "group"]).tds.sum().rename("allowed")
    games = df.groupby("opponent").week.nunique().rename("games")
    out = allowed.reset_index().rename(columns={"opponent": "team"})
    out["games"] = out.team.map(games).astype(float)
    return out


def defense_multipliers(pw_prior: pd.DataFrame, pw_cur: pd.DataFrame) -> pd.DataFrame:
    """Per (team, group) multiplier, regressed toward league average."""
    prior = _allowed_by_group(pw_prior)
    cur = _allowed_by_group(pw_cur)
    wp = config.DEF_PRIOR_SEASON_WEIGHT
    prior = prior.assign(allowed=prior.allowed * wp, games=prior.games * wp)
    both = pd.concat([prior, cur], ignore_index=True)
    if not len(both):
        return pd.DataFrame(columns=["team", "group", "def_mult"])
    agg = both.groupby(["team", "group"], as_index=False)[["allowed", "games"]].sum()
    league = agg.groupby("group").apply(
        lambda g: g.allowed.sum() / g.games.sum(), include_groups=False
    )
    avg = agg.group.map(league)
    k = config.DEF_SHRINK_GAMES
    agg["def_mult"] = ((agg.allowed + k * avg) / (agg.games + k)) / avg
    return agg[["team", "group", "def_mult"]]


# --- game context (schedule + Vegas) ---------------------------------------
def game_context(games: pd.DataFrame, season: int) -> pd.DataFrame:
    """One row per (team, week) for the season: opponent, home, kickoff, implied total."""
    g = games[games.season == season].copy()
    half_total = g.total_line / 2
    half_spread = g.spread_line / 2  # positive = home favored
    home = pd.DataFrame(
        {
            "team": g.home_team,
            "opponent": g.away_team,
            "home": True,
            "implied_total": half_total + half_spread,
        }
    )
    away = pd.DataFrame(
        {
            "team": g.away_team,
            "opponent": g.home_team,
            "home": False,
            "implied_total": half_total - half_spread,
        }
    )
    common = g[["game_id", "week", "kickoff"]]
    ctx = pd.concat(
        [pd.concat([common, home], axis=1), pd.concat([common, away], axis=1)], ignore_index=True
    )
    known = ctx.implied_total.dropna()
    league_avg = float(known.mean()) if len(known) else config.FALLBACK_TEAM_TOTAL
    team_avg = ctx.groupby("team").implied_total.mean()
    filled = ctx.implied_total.fillna(ctx.team.map(team_avg)).fillna(league_avg)
    ctx["implied_total"] = filled
    ctx["line_known"] = ctx.implied_total.notna() & known.reindex(ctx.index).notna()
    ctx["vegas_mult"] = ctx.implied_total / league_avg
    ctx["home_mult"] = np.where(ctx.home, config.HOME_MULT, config.AWAY_MULT)
    return ctx.sort_values(["week", "kickoff", "team"]).reset_index(drop=True)


# --- availability -----------------------------------------------------------
def injury_multipliers(injuries: pd.DataFrame) -> pd.DataFrame:
    """(player_id, week) -> avail_mult from the injury report."""
    if not len(injuries):
        return pd.DataFrame(columns=["player_id", "week", "avail_mult", "report_status"])
    df = injuries[["player_id", "week", "report_status"]].copy()
    df["avail_mult"] = df.report_status.map(config.INJURY_MULT).fillna(1.0)
    return df


def role_multipliers(depth: pd.DataFrame) -> pd.Series:
    """player_id -> multiplier from depth-chart rank."""
    if not len(depth):
        return pd.Series(dtype=float)

    def mult(pos: str, rank: int) -> float:
        table = config.DEPTH_MULT.get(pos)
        if not table:
            return 1.0
        return table.get(int(rank), table[max(table)])

    vals = [mult(p, r) for p, r in zip(depth.position, depth["rank"], strict=True)]
    return pd.Series(vals, index=depth.player_id.values, dtype=float)


# --- assemble ---------------------------------------------------------------
PROJECTION_COLUMNS = [
    "player_id",
    "player_name",
    "position",
    "slot",
    "team",
    "week",
    "opponent",
    "home",
    "kickoff",
    "game_id",
    "base_rate",
    "def_mult",
    "vegas_mult",
    "home_mult",
    "avail_mult",
    "report_status",
    "role_mult",
    "depth_rank",
    "lam",
    "prior_games",
    "prior_tds",
    "cur_games",
    "cur_tds",
]


def build_projections(frames: Frames, from_week: int = 1) -> pd.DataFrame:
    pool = player_pool(frames.rosters, frames.pw_prior, frames.pw_cur)
    pool = pool[pool.position.isin(config.POSITIONS)]
    means = positional_means(frames.pw_prior)
    rates = baseline_rates(pool, frames.pw_prior, frames.pw_cur, means)
    rates["slot"] = rates.position.map(slot_for_position)

    ctx = game_context(frames.games, frames.season)
    ctx = ctx[ctx.week >= from_week]
    proj = rates.merge(ctx, on="team", how="inner")  # bye weeks simply have no row

    dmult = defense_multipliers(frames.pw_prior, frames.pw_cur)
    proj = proj.merge(
        dmult.rename(columns={"team": "opponent", "group": "slot"}),
        on=["opponent", "slot"],
        how="left",
    )
    proj["def_mult"] = proj.def_mult.fillna(1.0)

    inj = injury_multipliers(frames.injuries)
    proj = proj.merge(inj, on=["player_id", "week"], how="left")
    proj["avail_mult"] = proj.avail_mult.fillna(1.0)

    role = role_multipliers(frames.depth)
    proj["role_mult"] = proj.player_id.map(role)
    proj["role_mult"] = proj.role_mult.fillna(config.DEPTH_DEFAULT_MULT if len(role) else 1.0)
    ranks = (
        frames.depth.set_index("player_id")["rank"] if len(frames.depth) else pd.Series(dtype=float)
    )
    proj["depth_rank"] = proj.player_id.map(ranks)

    proj["lam"] = (
        proj.base_rate
        * proj.def_mult
        * proj.vegas_mult
        * proj.home_mult
        * proj.avail_mult
        * proj.role_mult
    )
    return (
        proj[PROJECTION_COLUMNS]
        .sort_values(["week", "slot", "lam"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def projections_for(conn: sqlite3.Connection, season: int, from_week: int = 1) -> pd.DataFrame:
    return build_projections(load_frames(conn, season), from_week)
