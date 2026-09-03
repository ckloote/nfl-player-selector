"""Per-type touchdown rates: pass, rush and receive estimated separately.

The original design specified this; the shipped model collapses all three into
one rate. This is that design, built in full, including opponent-defence
multipliers split by touchdown type:

    lam = sum over t in {pass, rush, rec} of (rate_t * def_mult[slot, t])
          * vegas * home * avail * role

RESULT (2017-2025, measured against the shipped model on identical frozen data):

    greedy      -0.89 TD/season (SE 1.33), better in 4/9 seasons
    optimizer   -2.89 TD/season (SE 1.54), better in 2/9 seasons

No evidence of benefit. Two likely reasons: it refines the opponent-defence
multiplier, which carries no measurable signal to begin with (see
docs/BACKTEST.md section 4), and it splits already-sparse touchdown counts three
ways, adding estimation variance for no extra information. Rankings barely move
- Spearman 0.999 against the combined model - though it does change 23% of
picks.

If revisiting: this reuses the same shrinkage constants for every type, and
sparser per-type counts arguably want heavier shrinkage than
PRIOR_SEASON_SHRINK_GAMES. That is the first thing to try.

    uv run pool backtest --season 2017-2025 --projection per-type
"""

import numpy as np
import pandas as pd

from .. import config
from .. import projections as P
from ..config import slot_for_position

TYPES = ["pass_td", "rush_td", "rec_td"]


def _totals_by_type(pw):
    if not len(pw):
        return pd.DataFrame(columns=["position", "games", *TYPES]).rename_axis("player_id")
    return pw.groupby("player_id").agg(
        position=("position", "last"), games=("week", "nunique"), **{t: (t, "sum") for t in TYPES}
    )


def type_means(pw_prior):
    """Mean per-game rate of each TD type among 'regulars' at each position."""
    totals = P.per_player_totals(pw_prior)
    by_type = _totals_by_type(pw_prior)
    means = {}
    for pos in config.POSITIONS:
        sub = totals[totals.position == pos]
        regs = sub[
            (sub.games >= config.REGULAR_MIN_GAMES)
            & (sub.usage / sub.games >= config.REGULAR_MIN_USAGE[pos])
        ]
        pool = regs if len(regs) else sub
        tt = by_type.reindex(pool.index)
        for t in TYPES:
            means[(pos, t)] = float((tt[t] / pool.games).mean()) if len(pool) else 0.0
    return means


def rates_by_type(pool, pw_prior, pw_cur, means):
    prior, cur = _totals_by_type(pw_prior), _totals_by_type(pw_cur)
    out = pool.copy()
    pg = out.player_id.map(prior.games).fillna(0.0)
    cg = out.player_id.map(cur.games).fillna(0.0)
    k, w = config.PRIOR_SEASON_SHRINK_GAMES, config.PRIOR_WEIGHT_GAMES
    for t in TYPES:
        mean = pd.Series([means.get((p, t), 0.0) for p in out.position], index=out.index)
        zero = pd.Series(0.0, index=out.index)
        ptd = out.player_id.map(prior[t]).fillna(0.0) if len(prior) else zero
        ctd = out.player_id.map(cur[t]).fillna(0.0) if len(cur) else zero
        regressed = (ptd + k * mean) / (pg + k)
        prior_rate = np.where(pg > 0, regressed, config.NO_HISTORY_FACTOR * mean)
        out[f"rate_{t}"] = (w * prior_rate + ctd) / (w + cg)
    out["prior_games"], out["cur_games"] = pg, cg
    out["prior_tds"] = sum(
        (out.player_id.map(prior[t]).fillna(0.0) if len(prior) else 0.0) for t in TYPES
    )
    out["cur_tds"] = sum(
        (out.player_id.map(cur[t]).fillna(0.0) if len(cur) else 0.0) for t in TYPES
    )
    return out


def defense_by_type(pw_prior, pw_cur):
    """(team, slot, type) -> multiplier, regressed toward the league rate."""

    def allowed(pw):
        if not len(pw):
            return pd.DataFrame(columns=["team", "slot", "type", "allowed", "games"])
        df = pw[pw.opponent.notna()].copy()
        df["slot"] = df.position.map(slot_for_position)
        g = df.groupby("opponent").week.nunique()
        m = df.melt(
            id_vars=["opponent", "slot"], value_vars=TYPES, var_name="type", value_name="td"
        )
        out = m.groupby(["opponent", "slot", "type"], as_index=False).td.sum()
        out = out.rename(columns={"opponent": "team", "td": "allowed"})
        out["games"] = out.team.map(g).astype(float)
        return out

    wp = config.DEF_PRIOR_SEASON_WEIGHT
    pr = allowed(pw_prior)
    pr = pr.assign(allowed=pr.allowed * wp, games=pr.games * wp)
    both = pd.concat([pr, allowed(pw_cur)], ignore_index=True)
    if not len(both):
        return pd.DataFrame(columns=["team", "slot", "type", "def_mult"])
    agg = both.groupby(["team", "slot", "type"], as_index=False)[["allowed", "games"]].sum()
    league = agg.groupby(["slot", "type"]).apply(
        lambda g: g.allowed.sum() / g.games.sum() if g.games.sum() else 0.0, include_groups=False
    )
    avg = pd.MultiIndex.from_frame(agg[["slot", "type"]]).map(league).to_numpy(dtype=float)
    k = config.DEF_SHRINK_GAMES
    # Some (slot, type) pairs are structurally empty - nobody allows rushing
    # touchdowns "to the QB slot" via a receiver - so guard the divisor.
    safe = np.where(avg > 0, avg, 1.0)
    mult = ((agg.allowed.to_numpy(float) + k * safe) / (agg.games.to_numpy(float) + k)) / safe
    agg["def_mult"] = np.where(avg > 0, mult, 1.0)
    return agg[["team", "slot", "type", "def_mult"]]


def build(frames, from_week=1, role_source=None):
    pool = P.player_pool(frames.rosters, frames.pw_prior, frames.pw_cur)
    pool = pool[pool.position.isin(config.POSITIONS)]
    rates = rates_by_type(pool, frames.pw_prior, frames.pw_cur, type_means(frames.pw_prior))
    rates["slot"] = rates.position.map(slot_for_position)

    ctx = P.game_context(frames.games, frames.season)
    proj = rates.merge(ctx[ctx.week >= from_week], on="team", how="inner")

    dmult = defense_by_type(frames.pw_prior, frames.pw_cur)
    wide = dmult.pivot_table(index=["team", "slot"], columns="type", values="def_mult")
    proj = proj.merge(
        wide.rename(columns={t: f"d_{t}" for t in TYPES})
        .reset_index()
        .rename(columns={"team": "opponent"}),
        on=["opponent", "slot"],
        how="left",
    )

    base = 0.0
    for t in TYPES:
        base = base + proj[f"rate_{t}"].astype(float) * proj[f"d_{t}"].fillna(1.0).astype(float)

    inj = P.injury_multipliers(frames.injuries)
    proj = proj.merge(inj, on=["player_id", "week"], how="left")
    proj["avail_mult"] = proj.avail_mult.fillna(1.0)
    role, ranks = P._role_frames(frames, pool, role_source)
    proj["role_mult"] = proj.player_id.map(role).fillna(
        config.DEPTH_DEFAULT_MULT if len(role) else 1.0
    )
    proj["depth_rank"] = proj.player_id.map(ranks)
    proj["base_rate"] = base
    proj["def_mult"] = 1.0
    proj["lam"] = base * proj.vegas_mult * proj.home_mult * proj.avail_mult * proj.role_mult
    return (
        proj[P.PROJECTION_COLUMNS]
        .sort_values(["week", "slot", "lam"], ascending=[True, True, False])
        .reset_index(drop=True)
    )
