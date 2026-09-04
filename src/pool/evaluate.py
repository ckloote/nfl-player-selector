"""Player-week forecast evaluation: the projection layer measured on its own.

`docs/BACKTEST.md` §6 sets the constraint this module exists to escape. Replaying
a season scores 54 picks and yields one number; with fifteen seasons that
resolves effects larger than about +/-3 TD/season and nothing smaller, which is
every model refinement anyone has tried.

The same frozen projections contain ~7,500 player-week forecasts per season.
Scoring those directly — every candidate, not just the one that was picked —
is three to four orders of magnitude more observations, and it is the only known
way to measure a projection change at a resolution the pool decision cannot.

What that buys is layer A and layer B evidence (is lambda numerically right, and
does it rank the right player-weeks). It does **not** buy layer C: whether a
better forecast produces a better season still runs into the +/-3 TD floor, and
must not be claimed from anything in this module.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import backtest

FORECAST_COLUMNS = [
    "model",
    "season",
    "week",
    "slot",
    "position",
    "player_id",
    "player_name",
    "team",
    "opponent",
    "home",
    "lam",
    "actual_tds",
    "played",
    "rank_in_slot",
    "rank_available",
    "picked_greedy",
    "picked_optimizer",
]

# Reliability bins. Deliberately fine below 0.3, where most of the population
# lives, and open-ended above 1.0, which is essentially the starting quarterbacks.
LAMBDA_BINS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00, np.inf]

TOP_K = (1, 3, 5, 10)

# The pre-registered primary is top-10 of the *depleted* pool. Both halves of
# that were chosen against measured behaviour on the two ablations the season
# harness already sized, not from first principles:
#
#   - Depleted, not full-pool. The pool may use each player once, so by midseason
#     the decision is made well down the ranking, and that is where models
#     separate. Ranked in the full pool, top-1 is the same handful of stars every
#     model likes: it recovers only -4.9 of the -8.6 TD/season that dropping
#     Vegas actually costs. Ranked among still-available players, top-1 recovers
#     -8.8 — it *is* the greedy replay.
#   - k=10, not k=1. Sensitivity falls with k but noise falls faster: depleted
#     top-1 carries SE 1.8-2.7 TD/season, depleted top-10 carries SE 0.4-0.6 for
#     a comparable t. Top-1 is the decision but reproduces the season harness's
#     own resolution, so selecting on it would buy nothing.
PRIMARY_K = 10
PRIMARY_RANK = "rank_available"
# Depleted top-1 is the confirmatory gate: it is exactly what greedy scores, so
# it says whether a change reaches the decision the pool actually makes.
CONFIRM_K = 1

# Poisson deviance is reported over this subpopulation. Below it the metric is
# dominated by deep-bench player-weeks the pool will never select. It is a
# layer-A diagnostic and deliberately *not* a gate: `base-rate-only` scores
# better deviance and a better calibration slope than the shipped model while
# costing 10.3 TD/season, which is the whole reason A and C are kept apart.
DEVIANCE_FLOOR = 0.30


# --- the forecast set -------------------------------------------------------
def played_pairs(conn: sqlite3.Connection, season: int) -> set[tuple[int, str]]:
    """(week, player_id) that have a stat line — i.e. the player was on the field.

    Distinguishing "projected and scored zero" from "was never active" is what
    separates a real tail defect from a did-not-play artifact.
    """
    return {
        (int(w), p)
        for w, p in conn.execute(
            "SELECT week, player_id FROM player_weeks WHERE season = ?", (season,)
        )
    }


def greedy_used_before(
    frames: dict[int, pd.DataFrame],
    actuals: dict[tuple[int, str], float],
    season: int,
    weeks: Sequence[int],
) -> dict[int, set[str]]:
    """Week -> players a greedy walk has already spent before that week."""
    replay = backtest.replay(frames, actuals, season, "greedy", weeks=weeks)
    used_before: dict[int, set[str]] = {}
    seen: set[str] = set()
    for week in weeks:
        used_before[week] = set(seen)
        seen |= {p.player_id for p in replay.picks if p.week == week and p.player_id}
    return used_before


def forecasts(
    conn: sqlite3.Connection,
    season: int,
    *,
    model: str,
    builder: Callable[..., pd.DataFrame] | None = None,
    weeks: Sequence[int] | None = None,
    role_source: str = "usage",
    vegas_horizon: int | None = None,
    strategies: Sequence[str] = ("greedy",),
    frames: dict[int, pd.DataFrame] | None = None,
    available_from: dict[int, set[str]] | None = None,
) -> pd.DataFrame:
    """One row per (player, week) forecast for `season`, with the realised outcome.

    Only the week being forecast is kept from each frozen frame: the later weeks
    a frame also carries are forecasts the model will revise before they are
    picked, so scoring them would double-count and would not match any decision
    anyone actually makes.
    """
    weeks = list(weeks) if weeks is not None else backtest.scored_weeks(conn, season)
    if frames is None:
        frames = backtest.weekly_projections(
            conn,
            season,
            weeks,
            role_source=role_source,
            vegas_horizon=vegas_horizon,
            builder=builder,
        )
    actuals = backtest.actual_tds(conn, season)
    played = played_pairs(conn, season)

    picks: dict[str, set[tuple[int, str]]] = {}
    for name in strategies:
        replay = backtest.replay(frames, actuals, season, name, weeks=weeks)
        picks[name] = {(p.week, p.player_id) for p in replay.picks if p.player_id}

    # Which players are already spent, and therefore out of the running.
    #
    # This MUST come from one common walk, not from each model's own. A model
    # that picks badly leaves better players in the pool, which inflates its
    # own top-k on every later week — a mechanical advantage for being worse.
    # Measured: with per-model walks, 12 of 15 shrinkage perturbations "improved"
    # the primary metric; against one common pool the same sweep splits 8/7,
    # which is the signature of exactly that bias.
    used_before = (
        available_from
        if available_from is not None
        else greedy_used_before(frames, actuals, season, weeks)
    )

    rows = []
    for week in weeks:
        cur = frames[week]
        cur = cur[cur.week == week].copy()
        cur["model"] = model
        cur["season"] = season
        cur["actual_tds"] = [float(actuals.get((week, pid), 0.0)) for pid in cur.player_id]
        cur["played"] = [(week, pid) in played for pid in cur.player_id]
        # Ties broken by player_id so the ranking — and therefore every top-k
        # metric — is reproducible run to run.
        cur = cur.sort_values(["slot", "lam", "player_id"], ascending=[True, False, True])
        cur["rank_in_slot"] = cur.groupby("slot").cumcount() + 1
        # Rank among players not yet spent — the ranking the pool actually
        # chooses from. Players already used are NaN: they are not selectable,
        # so they belong in no top-k.
        spent = used_before.get(week, set())
        free = cur[~cur.player_id.isin(spent)]
        cur["rank_available"] = (free.groupby("slot").cumcount() + 1).reindex(cur.index)
        for name in ("greedy", "optimizer"):
            chosen = picks.get(name)
            cur[f"picked_{name}"] = (
                False if chosen is None else [(week, pid) in chosen for pid in cur.player_id]
            )
        rows.append(cur[FORECAST_COLUMNS])
    return pd.concat(rows, ignore_index=True)


def forecast_set(
    conn: sqlite3.Connection,
    seasons: Iterable[int],
    models: dict[str, Callable[..., pd.DataFrame] | None],
    *,
    role_source: str = "usage",
    vegas_horizon: int | None = None,
    strategies: Sequence[str] = ("greedy",),
    baseline: str = "shipped",
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """The full evaluation set: every model over every season, one frame.

    Every model is scored against the same depleted pool — the one the baseline
    model's greedy walk produces — so no model can gain from having spent its
    players badly. See `forecasts`.
    """
    out = []
    for season in seasons:
        weeks = backtest.scored_weeks(conn, season)
        common = None
        if baseline in models:
            frames = backtest.weekly_projections(
                conn,
                season,
                weeks,
                role_source=role_source,
                vegas_horizon=vegas_horizon,
                builder=models[baseline],
            )
            common = greedy_used_before(frames, backtest.actual_tds(conn, season), season, weeks)
        for name, builder in models.items():
            df = forecasts(
                conn,
                season,
                model=name,
                builder=builder,
                weeks=weeks,
                role_source=role_source,
                vegas_horizon=vegas_horizon,
                strategies=strategies,
                available_from=common,
            )
            out.append(df)
            if log:
                log(f"  {season} {name}: {len(df)} forecasts")
    return pd.concat(out, ignore_index=True)


# --- layer A: calibration ---------------------------------------------------
def reliability(df: pd.DataFrame, bins: Sequence[float] = LAMBDA_BINS) -> pd.DataFrame:
    """Mean projected vs mean actual by projection bin — the readable diagnostic.

    A pooled projected/actual ratio can sit at 1.0 while the model is badly
    tilted, because under-projection at the bottom cancels over-projection at
    the top. This table is what shows that; the pooled number hides it.
    """
    out = df.copy()
    out["bin"] = pd.cut(out.lam, bins=list(bins))
    g = out.groupby("bin", observed=True).agg(
        n=("lam", "size"),
        proj=("lam", "mean"),
        actual=("actual_tds", "mean"),
        played=("played", "mean"),
    )
    g["ratio"] = g.proj / g.actual.replace(0.0, np.nan)
    return g.reset_index()


def poisson_glm(
    y: np.ndarray, x: np.ndarray, cluster: np.ndarray, max_iter: int = 50, tol: float = 1e-10
) -> dict[str, float]:
    """Fit E[Y] = exp(a + b*x) by IRLS, with cluster-robust standard errors.

    `log(actual + eps) ~ a + b*log(lambda)` would be the easy thing to write and
    the wrong thing to report: touchdowns are zero-heavy counts, so the answer
    would depend on an arbitrary eps. A Poisson GLM on the log scale needs no
    such fudge — b = 1 means calibrated, b < 1 means the forecasts are too
    extreme.

    The standard errors must be clustered, not merely heteroskedasticity-robust.
    A player's rate error persists across every week of their season and
    team-mates share `vegas_mult` and game script, so the ~113k rows carry far
    less information than 113k independent observations; naive errors would make
    everything significant.
    """
    design = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(max_iter):
        mu = np.exp(np.clip(design @ beta, -30, 30))
        grad = design.T @ (y - mu)
        hess = design.T @ (design * mu[:, None])
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break

    mu = np.exp(np.clip(design @ beta, -30, 30))
    bread = np.linalg.inv(design.T @ (design * mu[:, None]))
    resid = (y - mu)[:, None] * design
    groups = pd.Series(cluster)
    meat = np.zeros((2, 2))
    for _, idx in groups.groupby(groups, sort=False).indices.items():
        s = resid[idx].sum(axis=0)
        meat += np.outer(s, s)
    n_groups = groups.nunique()
    n, k = len(y), 2
    correction = (n_groups / max(n_groups - 1, 1)) * ((n - 1) / (n - k))
    cov = bread @ meat @ bread * correction
    se = np.sqrt(np.diag(cov))
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "se_intercept": float(se[0]),
        "se_slope": float(se[1]),
        "slope_lo": float(beta[1] - 1.96 * se[1]),
        "slope_hi": float(beta[1] + 1.96 * se[1]),
        "n": int(n),
        "clusters": int(n_groups),
    }


def calibration(df: pd.DataFrame, cluster_on: str = "player_season") -> dict[str, float]:
    """Calibration slope/intercept for one model's forecasts."""
    sub = df[df.lam > 0]
    if not len(sub):
        return {"intercept": np.nan, "slope": np.nan, "n": 0}
    if cluster_on == "player_season":
        cluster = sub.player_id.astype(str) + "|" + sub.season.astype(str)
    elif cluster_on == "season":
        cluster = sub.season.astype(str)
    else:
        cluster = sub[cluster_on].astype(str)
    out = poisson_glm(
        sub.actual_tds.to_numpy(dtype=float),
        np.log(sub.lam.to_numpy(dtype=float)),
        cluster.to_numpy(),
    )
    out["dropped_zero_lam"] = int(len(df) - len(sub))
    return out


def poisson_deviance(actual: np.ndarray, lam: np.ndarray) -> float:
    """Mean Poisson deviance. A proper scoring rule, so it cannot be improved by
    flattening the ranking — which a naive calibration fix would happily do."""
    actual = np.asarray(actual, dtype=float)
    lam = np.clip(np.asarray(lam, dtype=float), 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(actual > 0, actual * np.log(actual / lam), 0.0)
    return float(2.0 * np.mean(term - (actual - lam)))


# --- layer B: ranking -------------------------------------------------------
def top_k(df: pd.DataFrame, ks: Sequence[int] = TOP_K, rank: str = PRIMARY_RANK) -> pd.DataFrame:
    """Projected vs realised for the top k of each (season, week, slot).

    This is where the pool actually lives: it never picks from the 95% of
    player-weeks that dominate a population-wide error metric.
    """
    rows = []
    for model, sub in df.groupby("model", sort=False):
        pop = sub.actual_tds.mean()
        for k in ks:
            s = sub[sub[rank] <= k]
            rows.append(
                {
                    "model": model,
                    "k": k,
                    "rank": rank,
                    "n": len(s),
                    "proj": s.lam.mean(),
                    "actual": s.actual_tds.mean(),
                    "ratio": s.lam.mean() / s.actual_tds.mean() if s.actual_tds.mean() else np.nan,
                    "played": s.played.mean(),
                    "lift": s.actual_tds.mean() / pop if pop else np.nan,
                }
            )
    return pd.DataFrame(rows)


def spearman(df: pd.DataFrame) -> pd.DataFrame:
    """Rank correlation of lambda against realised TDs, within each slot-week."""
    rows = []
    for model, sub in df.groupby("model", sort=False):
        rhos = []
        for _, g in sub.groupby(["season", "week", "slot"], sort=False):
            if g.actual_tds.nunique() > 1 and g.lam.nunique() > 1:
                # numpy arrays, not Series: scipy 1.18 / numpy 2.5 raises on
                # Series input here.
                rhos.append(
                    float(
                        stats.spearmanr(
                            g.lam.to_numpy(dtype=float),
                            g.actual_tds.to_numpy(dtype=float),
                        ).statistic
                    )
                )
        rows.append(
            {
                "model": model,
                "spearman": float(np.mean(rhos)) if rhos else np.nan,
                "se": float(np.std(rhos, ddof=1) / np.sqrt(len(rhos))) if len(rhos) > 1 else np.nan,
                "slot_weeks": len(rhos),
            }
        )
    return pd.DataFrame(rows)


# --- paired comparison ------------------------------------------------------
def paired_top_k(
    df: pd.DataFrame,
    baseline: str = "shipped",
    k: int = PRIMARY_K,
    rank: str = PRIMARY_RANK,
) -> pd.DataFrame:
    """Each model minus `baseline`, paired on (season, week, slot).

    Pairing is what makes the comparison sensitive: a week where everyone scores
    is a week where both models score, and that shared variation cancels. It is
    the same mechanism that gives the ablations in BACKTEST.md standard errors
    of 1.3-1.7 rather than ~3.

    Standard errors are clustered by season, which is conservative — slot-weeks
    within a season share players — and makes the number directly comparable to
    the per-season deltas the existing harness reports.
    """
    top = df[df[rank] <= k]
    per_cell = top.groupby(["model", "season", "week", "slot"], sort=False).actual_tds.mean()
    wide = per_cell.unstack("model")
    if baseline not in wide.columns:
        raise KeyError(f"baseline model {baseline!r} not in the forecast set")
    rows = []
    # 54 slot-weeks a season (3 slots x 18 weeks), so a per-slot-week delta
    # scales to a season by multiplying by that count.
    for model in wide.columns:
        if model == baseline:
            continue
        delta = (wide[model] - wide[baseline]).dropna()
        by_season = delta.groupby("season").mean()
        cells = delta.groupby("season").size()
        season_totals = by_season * cells
        n = len(by_season)
        rows.append(
            {
                "model": model,
                "k": k,
                "delta_per_pick": float(delta.mean()),
                "delta_per_season": float(season_totals.mean()),
                "se_per_season": float(season_totals.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
                "seasons_won": int((by_season > 0).sum()),
                "seasons": n,
            }
        )
    return pd.DataFrame(rows).sort_values("delta_per_season", ascending=False)


def paired_deviance(
    df: pd.DataFrame, baseline: str = "shipped", floor: float = DEVIANCE_FLOOR
) -> pd.DataFrame:
    """Poisson deviance on the decision-relevant tail, each model minus baseline.

    Restricted to lambda > `floor` because the population metric is otherwise
    dominated by deep-bench rows. Restricting the subpopulation costs strict
    propriety; the deviance is still a proper score conditional on the stratum,
    and it is used as a co-primary gate rather than alone.
    """
    keep = df[df.model == baseline]
    keep = keep.loc[keep.lam > floor, ["season", "week", "slot", "player_id"]]
    sub = df.merge(keep, on=["season", "week", "slot", "player_id"], how="inner")
    rows = []
    for model, s in sub.groupby("model", sort=False):
        per_season = s.groupby("season").apply(
            lambda g: poisson_deviance(g.actual_tds.to_numpy(), g.lam.to_numpy()),
            include_groups=False,
        )
        rows.append(
            {"model": model, "deviance": float(per_season.mean()), "_per_season": per_season}
        )
    out = pd.DataFrame(rows).set_index("model")
    base = out.loc[baseline, "_per_season"]
    res = []
    for model, r in out.iterrows():
        if model == baseline:
            continue
        d = r["_per_season"] - base
        res.append(
            {
                "model": model,
                "deviance": r["deviance"],
                # Negative is better: less deviance than the shipped model.
                "delta": float(d.mean()),
                "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan,
                "seasons_better": int((d < 0).sum()),
                "seasons": len(d),
            }
        )
    return pd.DataFrame(res).sort_values("delta")


def by_season(df: pd.DataFrame, k: int = 1, rank: str = PRIMARY_RANK) -> pd.DataFrame:
    """Top-k realised TDs per season per model — the stability check.

    A model is not preferred on a pooled mean. `BACKTEST.md` records what
    happens when it is: +3.67 TD/season on the seasons a variant was found on,
    +0.17 on a holdout.
    """
    top = df[df[rank] <= k]
    g = top.groupby(["model", "season"]).actual_tds.agg(["mean", "size"])
    g["season_total"] = g["mean"] * g["size"]
    return g.reset_index()
