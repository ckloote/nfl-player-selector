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
    as_of_week: int | None = None  # None = everything loaded (live use)


def load_frames(
    conn: sqlite3.Connection,
    season: int,
    as_of_week: int | None = None,
    vegas_horizon: int | None = None,
) -> Frames:
    """Every frame the model needs, optionally frozen at a past pick deadline.

    With `as_of_week` set the loader returns only what was knowable an hour
    before week W's first kickoff — which is the whole basis of a backtest:

    | source                       | visible at week W                        |
    |------------------------------|------------------------------------------|
    | games (teams, weeks, kickoff)| all REG weeks: the schedule is published  |
    |                              | in advance                               |
    | games.spread_line/total_line | weeks <= W + horizon; NaN beyond          |
    | player_weeks, prior season   | all: the prior season is complete         |
    | player_weeks, this season    | week < W — week W has not been played     |
    | rosters                      | week <= W, then the latest row per player |
    | injuries                     | week <= W: the report precedes kickoff    |

    Final scores are never read by the model, so they need no cutoff.
    """
    prior = season - 1
    games = db.read_df(
        conn, "SELECT * FROM games WHERE season IN (?, ?) AND game_type = 'REG'", (prior, season)
    )
    if as_of_week is not None:
        horizon = config.VEGAS_HORIZON_WEEKS if vegas_horizon is None else vegas_horizon
        beyond = (games.season == season) & (games.week > as_of_week + horizon)
        games.loc[beyond, ["spread_line", "total_line"]] = np.nan

    if as_of_week is None:
        pw = db.read_df(conn, "SELECT * FROM player_weeks WHERE season IN (?, ?)", (prior, season))
    else:
        pw = db.read_df(
            conn,
            "SELECT * FROM player_weeks WHERE season = ? OR (season = ? AND week < ?)",
            (prior, season, as_of_week),
        )

    rosters = _as_of(
        db.read_df(conn, "SELECT * FROM rosters WHERE season = ?", (season,)), as_of_week
    )
    if len(rosters):
        rosters = rosters.sort_values("week").drop_duplicates("player_id", keep="last")
    injuries = _as_of(
        db.read_df(conn, "SELECT * FROM injuries WHERE season = ?", (season,)), as_of_week
    )
    depth = db.read_df(conn, "SELECT * FROM depth_charts WHERE season = ?", (season,))
    return Frames(
        season=season,
        games=games,
        pw_prior=pw[pw.season == prior].copy(),
        pw_cur=pw[pw.season == season].copy(),
        rosters=rosters,
        injuries=injuries,
        depth=depth,
        as_of_week=as_of_week,
    )


def _as_of(df: pd.DataFrame, as_of_week: int | None) -> pd.DataFrame:
    """Rows published up to and including week W. Both reports precede kickoff."""
    if as_of_week is None or not len(df):
        return df
    return df[df.week <= as_of_week].copy()


def available_weeks(conn: sqlite3.Connection, season: int) -> list[int]:
    """REG weeks on the schedule. Never assume 18 — pre-2021 seasons had 17."""
    rows = conn.execute(
        "SELECT DISTINCT week FROM games WHERE season = ? AND game_type = 'REG' ORDER BY week",
        (season,),
    ).fetchall()
    return [int(r[0]) for r in rows]


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
    ctx["implied_total"] = ctx.implied_total.fillna(ctx.team.map(team_avg)).fillna(league_avg)
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


def depth_mult(position: str, rank: float) -> float:
    """Role multiplier for a depth-chart rank. Past the table, the last value."""
    table = config.DEPTH_MULT.get(position)
    if not table:
        return 1.0
    return table.get(int(rank), table[max(table)])


def role_multipliers(depth: pd.DataFrame) -> pd.Series:
    """player_id -> multiplier from depth-chart rank."""
    if not len(depth):
        return pd.Series(dtype=float)
    vals = [depth_mult(p, r) for p, r in zip(depth.position, depth["rank"], strict=True)]
    return pd.Series(vals, index=depth.player_id.values, dtype=float)


def usage_roles(pool: pd.DataFrame, pw_prior: pd.DataFrame, pw_cur: pd.DataFrame) -> pd.DataFrame:
    """Depth rank and role multiplier per player, inferred from usage to date.

    The `depth_charts` table holds one end-of-season snapshot per season and has
    no week column, so it cannot be rewound to a past pick deadline — a 2025
    backtest would be reading a chart published in March 2026. Usage share is
    reconstructible at any point in time and says much the same thing: rank
    players within their own (team, position) by touches per game, then read the
    same `DEPTH_MULT` table the depth chart feeds.

    Usage blends prior and current season with the model's own shrinkage, so a
    backup who takes over in week 3 climbs the ranking as the evidence arrives.
    """
    if not len(pool):
        return pd.DataFrame(columns=["player_id", "rank", "role_mult"])
    prior = per_player_totals(pw_prior)
    cur = per_player_totals(pw_cur)
    df = pool[["player_id", "team", "position"]].copy()
    prior_games = df.player_id.map(prior.games).fillna(0.0).to_numpy(dtype=float)
    prior_usage = df.player_id.map(prior.usage).fillna(0.0).to_numpy(dtype=float)
    cur_games = df.player_id.map(cur.games).fillna(0.0).to_numpy(dtype=float)
    cur_usage = df.player_id.map(cur.usage).fillna(0.0).to_numpy(dtype=float)
    prior_pg = np.divide(
        prior_usage, prior_games, out=np.zeros_like(prior_usage), where=prior_games > 0
    )
    w = config.PRIOR_WEIGHT_GAMES
    df["usage_pg"] = (w * prior_pg + cur_usage) / (w + cur_games)
    # Ties broken by player_id so a replay is reproducible run to run.
    df = df.sort_values(
        ["team", "position", "usage_pg", "player_id"], ascending=[True, True, False, True]
    )
    df["rank"] = df.groupby(["team", "position"]).cumcount() + 1
    df["role_mult"] = [depth_mult(p, r) for p, r in zip(df.position, df["rank"], strict=True)]
    return df[["player_id", "rank", "role_mult"]].reset_index(drop=True)


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


def _role_frames(
    frames: Frames, pool: pd.DataFrame, role_source: str | None
) -> tuple[pd.Series, pd.Series]:
    """(role multiplier, depth rank) per player_id for the configured source."""
    source = config.ROLE_SOURCE if role_source is None else role_source
    if source == "none":
        return pd.Series(dtype=float), pd.Series(dtype=float)
    if source == "usage":
        roles = usage_roles(pool, frames.pw_prior, frames.pw_cur).set_index("player_id")
        return roles["role_mult"], roles["rank"]
    if source != "depth":
        raise ValueError(f"unknown role source {source!r}; expected depth, usage, or none")
    depth = frames.depth
    ranks = depth.set_index("player_id")["rank"] if len(depth) else pd.Series(dtype=float)
    return role_multipliers(depth), ranks


def build_projections(
    frames: Frames, from_week: int = 1, role_source: str | None = None
) -> pd.DataFrame:
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

    role, ranks = _role_frames(frames, pool, role_source)
    proj["role_mult"] = proj.player_id.map(role)
    proj["role_mult"] = proj.role_mult.fillna(config.DEPTH_DEFAULT_MULT if len(role) else 1.0)
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


def projections_for(
    conn: sqlite3.Connection,
    season: int,
    from_week: int = 1,
    as_of_week: int | None = None,
    role_source: str | None = None,
    vegas_horizon: int | None = None,
) -> pd.DataFrame:
    frames = load_frames(conn, season, as_of_week, vegas_horizon)
    return build_projections(frames, from_week, role_source)
