"""Forecast diagnostics and actual season replays, with season-level uncertainty.

Common-pool rankings use one deterministic baseline's depletion history. They do
not describe a challenger's achieved season score. Seeds are repeated forecasts
of the same seasons, never independent observations.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import backtest, config, projections, scoring
from . import models as registry

FORECAST_COLUMNS = [
    "model",
    "seed",
    "input_policy",
    "decision_at",
    "hard_eligible",
    "baseline_spent",
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

# Ranking diagnostics, expressed in TDs per ranked candidate.
PRIMARY_K = 10
PRIMARY_RANK = "rank_available"
DEVIANCE_FLOOR = 0.30


# --- the forecast set -------------------------------------------------------
def played_pairs(conn: sqlite3.Connection, season: int) -> set[tuple[int, str]]:
    """(week, player_id) with a stat row or touchdown credit.

    This exported `played` flag is a data-presence proxy, not an independently
    observed game-day active status or snap-participation label.
    """
    return {
        (int(w), p)
        for w, p in conn.execute(
            "SELECT week, player_id FROM player_weeks WHERE season = ? "
            "UNION SELECT g.week, t.player_id FROM touchdown_credits t "
            "JOIN games g ON g.game_id = t.game_id WHERE g.season = ? AND g.game_type = 'REG'",
            (season, season),
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
    seed: int = -1,
    builder: Callable[..., pd.DataFrame] | None = None,
    weeks: Sequence[int] | None = None,
    role_source: str | None = None,
    vegas_horizon: int | None = None,
    input_policy: str = "historical",
    decision_times: dict[tuple[int, int], str] | None = None,
    strategies: Sequence[str] = ("greedy",),
    frames: dict[int, pd.DataFrame] | None = None,
    available_from: dict[int, set[str]] | None = None,
) -> pd.DataFrame:
    """One row per (player, week) forecast for `season`, with the realised outcome.

    This export keeps only the current decision-week slice. The benchmark saves
    later-week surfaces separately: they inform assignment now, even though later
    decisions revise them. Evaluating those surfaces requires an explicit horizon
    and weighting scheme because repeated forecasts share target outcomes.
    """
    weeks = list(weeks) if weeks is not None else projections.available_weeks(conn, season)
    scoring.require_complete(conn, season, weeks)
    if frames is None:
        frames = backtest.weekly_projections(
            conn,
            season,
            weeks,
            role_source=role_source,
            vegas_horizon=vegas_horizon,
            input_policy=input_policy,
            decision_times=decision_times,
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
        cur["seed"] = seed
        cur["input_policy"] = input_policy
        cur["decision_at"] = frames[week].attrs.get("decision_at")
        if "hard_eligible" not in cur:
            cur["hard_eligible"] = True
        cur["season"] = season
        cur["actual_tds"] = [float(actuals.get((week, pid), 0.0)) for pid in cur.player_id]
        cur["played"] = [(week, pid) in played for pid in cur.player_id]
        # Ties broken by player_id so the ranking — and therefore every top-k
        # metric — is reproducible run to run.
        cur = cur.sort_values(["slot", "lam", "player_id"], ascending=[True, False, True])
        eligible = cur[cur.hard_eligible]
        cur["rank_in_slot"] = (eligible.groupby("slot").cumcount() + 1).reindex(cur.index)
        # Rank among players not yet spent — the ranking the pool actually
        # chooses from. Players already used are NaN: they are not selectable,
        # so they belong in no top-k.
        spent = used_before.get(week, set())
        cur["baseline_spent"] = cur.player_id.isin(spent)
        free = cur[cur.hard_eligible & ~cur.baseline_spent]
        cur["rank_available"] = (free.groupby("slot").cumcount() + 1).reindex(cur.index)
        for name in ("greedy", "optimizer"):
            chosen = picks.get(name)
            cur[f"picked_{name}"] = (
                False if chosen is None else [(week, pid) in chosen for pid in cur.player_id]
            )
        rows.append(cur[FORECAST_COLUMNS])
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=FORECAST_COLUMNS)
    out.attrs["cells"] = [(season, w, slot) for w in weeks for slot in config.SLOTS]
    return out


def validate_models(models, baseline):
    for name in models:
        registry.get(name)
    if baseline not in models:
        raise ValueError(
            f"Baseline {baseline!r} must be included in --model; selected {list(models)}"
        )
    if baseline in registry.NULLS:
        raise ValueError(
            "A shuffled baseline has no single depletion history; choose a selected "
            "deterministic baseline such as shipped or player-vegas."
        )


def assert_comparable(reference, challenger):
    keys = ["season", "week", "slot", "player_id"]
    masks = ["hard_eligible", "baseline_spent"]
    a = reference.sort_values(keys).set_index(keys)[masks]
    b = challenger.sort_values(keys).set_index(keys)[masks]
    if a.index.has_duplicates or b.index.has_duplicates:
        raise ValueError("Duplicate candidate keys in forecast comparison")
    if not a.equals(b):
        raise ValueError("Candidate keys, hard-eligibility or baseline-depletion masks differ")


def season_frames(
    conn,
    season,
    models,
    *,
    baseline="shipped",
    seeds=range(20),
    role_source=None,
    input_policy="historical",
    decision_times=None,
    vegas_horizon=None,
):
    """Yield model, seed, shared-baseline history, inputs, and model frames."""
    validate_models(models, baseline)
    role = projections.validate_policy(input_policy, role_source, vegas_horizon)
    weeks = projections.available_weeks(conn, season)
    scoring.require_complete(conn, season, weeks)
    loaded = backtest.weekly_inputs(
        conn,
        season,
        weeks,
        input_policy=input_policy,
        decision_times=decision_times,
        vegas_horizon=vegas_horizon,
    )
    for week, frame in loaded.items():
        frame.scaffold = projections.build_projections(frame, week, role_source=role)
    build = models[baseline]
    base_frames = {
        w: f.scaffold.copy() if build is None else build(f, w, role_source=role)
        for w, f in loaded.items()
    }
    common = greedy_used_before(base_frames, backtest.actual_tds(conn, season), season, weeks)
    yield baseline, -1, common, loaded, base_frames
    for name in sorted(set(models) - {baseline}):
        for seed in sorted(set(seeds)) if name in registry.NULLS else [-1]:
            build = registry.seeded(name, seed) if name in registry.NULLS else models[name]
            frames = {
                w: f.scaffold.copy() if build is None else build(f, w, role_source=role)
                for w, f in loaded.items()
            }
            for week, frame in frames.items():
                keys = ["week", "slot", "player_id"]
                expected = base_frames[week].set_index(keys).hard_eligible.sort_index()
                observed = frame.set_index(keys).hard_eligible.sort_index()
                if not expected.equals(observed):
                    raise ValueError(
                        "Candidate keys or hard-eligibility differ in future forecasts"
                    )
            yield name, seed, common, loaded, frames


def forecast_set(
    conn: sqlite3.Connection,
    seasons: Iterable[int],
    models: dict[str, Callable[..., pd.DataFrame] | None],
    *,
    role_source=None,
    vegas_horizon=None,
    strategies=("greedy", "optimizer"),
    baseline="shipped",
    seeds=range(20),
    input_policy="historical",
    decision_times=None,
    log=None,
) -> pd.DataFrame:
    validate_models(models, baseline)
    projections.validate_policy(input_policy, role_source, vegas_horizon)
    out, cells, replay_rows, pick_rows, provenance = [], [], [], [], []
    for season in sorted(seasons):
        reference = None
        actuals = backtest.actual_tds(conn, season)
        for name, seed, common, _loaded, frames in season_frames(
            conn,
            season,
            models,
            baseline=baseline,
            seeds=seeds,
            role_source=role_source,
            vegas_horizon=vegas_horizon,
            input_policy=input_policy,
            decision_times=decision_times,
        ):
            df = forecasts(
                conn,
                season,
                model=name,
                seed=seed,
                frames=frames,
                weeks=list(frames),
                available_from=common,
                strategies=(),
                input_policy=input_policy,
            )
            if reference is None:
                reference = df
                cells.extend(df.attrs["cells"])
                provenance.extend(
                    dict(season=season, week=w, decision_at=f.decision_at, inputs=f.provenance)
                    for w, f in _loaded.items()
                )
            assert_comparable(reference, df)
            for strategy in strategies:
                run = backtest.replay(frames, actuals, season, strategy)
                row, picks = replay_records(run, name, seed)
                replay_rows.append(row)
                pick_rows.extend(picks)
                chosen = {(p.week, p.player_id) for p in run.picks if p.player_id}
                df[f"picked_{strategy}"] = [
                    (w, p) in chosen for w, p in zip(df.week, df.player_id, strict=True)
                ]
            out.append(df)
            if log:
                log(f"  {season} {name} seed={seed}: {len(df)} forecasts")
    result = pd.concat(out, ignore_index=True)
    result.attrs.update(
        cells=cells, replays=replay_rows, picks=pick_rows, input_provenance=provenance
    )
    return result


def replay_records(run, model, seed):
    from dataclasses import asdict

    identity = dict(model=model, seed=seed, season=run.season, strategy=run.strategy)
    row = dict(
        **identity,
        total=run.total,
        projected=run.projected,
        empty_slots=run.empty_slots,
        unique_players=run.players_used,
        zero_picks=run.zero_picks,
    )
    picks = [dict(**identity, **asdict(pick)) for pick in run.picks]
    return row, picks


# --- layer A: calibration ---------------------------------------------------
def reliability(df: pd.DataFrame, bins: Sequence[float] = LAMBDA_BINS) -> pd.DataFrame:
    """Mean projected vs mean actual by projection bin — the readable diagnostic.

    A pooled projected/actual ratio can sit at 1.0 while the model is badly
    tilted, because under-projection at the bottom cancels over-projection at
    the top. This table is what shows that; the pooled number hides it.
    """
    out = eligible_rows(df).copy()
    out["bin"] = pd.cut(out.lam, bins=list(bins), include_lowest=True)
    g = out.groupby("bin", observed=True).agg(
        n=("lam", "size"),
        proj=("lam", "mean"),
        actual=("actual_tds", "mean"),
        played=("played", "mean"),
    )
    g["ratio"] = g.proj / g.actual.replace(0.0, np.nan)
    return g.reset_index()


# A fit either identifies its coefficients or it does not. The failure modes below all
# used to return plausible-looking numbers: `pinv` splits a rank-deficient design into a
# minimum-norm intercept and slope, an all-zero outcome column drives the linear predictor
# into the clip at -30 and reports a finite interval around it, and a single cluster scores
# the sandwich at the MLE, where the gradient is zero, producing a zero-width interval.
# A number that survives into a table is worse than an explicit refusal to fit.
FIT_OK = "ok"
FIT_UNSUPPORTED = "unsupported"

# The linear predictor is clipped so a diverging IRLS step cannot overflow `exp`. A
# solution that comes to rest *on* that bound was stopped by the guard rather than by
# the data: the likelihood was still climbing, and the step only looked small because
# the clip had flattened it. On this population a rate of `exp(30)` touchdowns is not a
# large estimate, it is an absent one.
LINEAR_PREDICTOR_BOUND = 30.0
# Above this, the observed information is numerically singular and `pinv` is silently
# discarding a direction, so the "estimate" in that direction is whatever the
# pseudoinverse chose. The separation fixture reaches 1e15; interior fits sit near 1e0.
MAX_INFORMATION_CONDITION = 1e12

FIT_KEYS = (
    "intercept",
    "slope",
    "se_intercept",
    "se_slope",
    "slope_lo",
    "slope_hi",
    "n",
    "clusters",
    "fit_status",
    "reason",
    "converged",
    "iterations",
    "cluster_se",
)


def _unsupported_fit(reason: str, n: int, clusters: int, iterations: int = 0) -> dict:
    """An explicit non-fit: every coefficient NaN, with the reason travelling beside it."""
    return {
        "intercept": np.nan,
        "slope": np.nan,
        "se_intercept": np.nan,
        "se_slope": np.nan,
        "slope_lo": np.nan,
        "slope_hi": np.nan,
        "n": int(n),
        "clusters": int(clusters),
        "fit_status": FIT_UNSUPPORTED,
        "reason": reason,
        "converged": False,
        "iterations": int(iterations),
        "cluster_se": False,
    }


def poisson_glm(
    y: np.ndarray,
    x: np.ndarray,
    cluster: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-10,
    min_clusters: int | None = None,
) -> dict[str, float]:
    """Fit E[Y] = exp(a + b*x) by IRLS, with cluster-robust standard errors.

    `log(actual + eps) ~ a + b*log(lambda)` would be the easy thing to write and
    the wrong thing to report: touchdowns are zero-heavy counts, so the answer
    would depend on an arbitrary eps. A Poisson GLM on the log scale needs no
    such fudge. The identity reference is a = 0 and b = 1; slope alone does not
    establish calibration, and these coefficients do not test nonlinear departures.

    The standard errors must be clustered, not merely heteroskedasticity-robust.
    Player-season clustering groups a player's repeated weekly errors. It does
    not also account for shared teammate/game effects or the same player across
    seasons. The caller must choose a clustering scheme for its inference target.

    Every return carries `fit_status`, `reason`, `converged`, `iterations` and
    `cluster_se`, so a caller can never mistake a refusal for an estimate. Empty
    inputs, non-finite inputs, negative outcomes, `n <= 2`, a rank-deficient design
    (a constant log rate), all-zero outcomes, exhausted iterations and a solution with
    no finite maximum -- separation, where the likelihood climbs forever and IRLS still
    reports a tidy coefficient -- are unsupported and return NaN coefficients.
    Fewer than `min_clusters` clusters keeps the point estimates and suppresses the
    cluster-robust uncertainty: the coefficients are still identified, the sandwich
    is not.
    """
    min_clusters = config.MIN_INFERENCE_CLUSTERS if min_clusters is None else min_clusters
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    cluster = np.asarray(cluster)
    if not len(y) == len(x) == len(cluster):
        raise ValueError("poisson_glm needs outcomes, log rates and clusters of equal length")
    n = len(y)
    n_groups = int(pd.Series(cluster).nunique()) if n else 0

    if n == 0:
        return _unsupported_fit("empty input", n, n_groups)
    if not (np.isfinite(y).all() and np.isfinite(x).all()):
        return _unsupported_fit("non-finite inputs", n, n_groups)
    if (y < 0).any():
        return _unsupported_fit("negative outcomes", n, n_groups)
    if n <= 2:
        # The finite-sample correction divides by n - k with k = 2, and two points
        # cannot separate an intercept from a slope in any case.
        return _unsupported_fit("n <= 2 for a two-parameter fit", n, n_groups)
    design = np.column_stack([np.ones_like(x), x])
    if np.linalg.matrix_rank(design) < 2:
        return _unsupported_fit("design rank 1 (constant log rate)", n, n_groups)
    if y.sum() == 0:
        return _unsupported_fit("all-zero outcomes", n, n_groups)

    beta = np.zeros(2)
    converged, iterations = False, 0
    for iterations in range(1, max_iter + 1):
        mu = np.exp(np.clip(design @ beta, -LINEAR_PREDICTOR_BOUND, LINEAR_PREDICTOR_BOUND))
        grad = design.T @ (y - mu)
        hess = design.T @ (design * mu[:, None])
        if not (np.isfinite(grad).all() and np.isfinite(hess).all()):
            return _unsupported_fit("non-finite IRLS step", n, n_groups, iterations)
        step = np.linalg.pinv(hess) @ grad
        if not np.isfinite(step).all():
            return _unsupported_fit("non-finite IRLS step", n, n_groups, iterations)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    if not converged:
        return _unsupported_fit("iterations exhausted before convergence", n, n_groups, iterations)

    # Converging is not the same as landing on a maximum. Under separation -- every
    # outcome zero on one side of a threshold and positive on the other -- the
    # likelihood climbs without bound and the estimate does not exist, yet IRLS reports
    # a tidy slope with an interval of no width. Two checks catch it, both on the
    # solution rather than on the path taken to it.
    eta = design @ beta
    if np.max(np.abs(eta)) >= LINEAR_PREDICTOR_BOUND:
        return _unsupported_fit(
            "no finite maximum-likelihood estimate (separation)", n, n_groups, iterations
        )
    mu = np.exp(eta)
    information = design.T @ (design * mu[:, None])
    if (
        np.linalg.matrix_rank(design * np.sqrt(mu)[:, None]) < 2
        or not np.isfinite(np.linalg.cond(information))
        or np.linalg.cond(information) > MAX_INFORMATION_CONDITION
    ):
        return _unsupported_fit(
            "weighted design is rank deficient at the solution", n, n_groups, iterations
        )

    bread = np.linalg.pinv(information)
    resid = (y - mu)[:, None] * design
    groups = pd.Series(cluster)
    meat = np.zeros((2, 2))
    for _, idx in groups.groupby(groups, sort=False).indices.items():
        s = resid[idx].sum(axis=0)
        meat += np.outer(s, s)
    k = 2
    correction = (n_groups / max(n_groups - 1, 1)) * ((n - 1) / (n - k))
    cov = bread @ meat @ bread * correction
    se = np.sqrt(np.diag(cov))
    # One cluster reduces the meat to the score at the MLE, which is zero: the
    # "interval" would have no width. Below the predeclared minimum the sandwich
    # is not a usable inferential object, so it is withheld rather than shown.
    cluster_se = n_groups >= min_clusters
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "se_intercept": float(se[0]) if cluster_se else np.nan,
        "se_slope": float(se[1]) if cluster_se else np.nan,
        "slope_lo": float(beta[1] - 1.96 * se[1]) if cluster_se else np.nan,
        "slope_hi": float(beta[1] + 1.96 * se[1]) if cluster_se else np.nan,
        "n": int(n),
        "clusters": int(n_groups),
        "fit_status": FIT_OK,
        "reason": None if cluster_se else f"cluster SEs suppressed: {n_groups} < {min_clusters}",
        "converged": True,
        "iterations": int(iterations),
        "cluster_se": bool(cluster_se),
    }


def calibration(
    df: pd.DataFrame, cluster_on: str = "player_season", min_clusters: int | None = None
) -> dict[str, float]:
    """Calibration slope/intercept for one model's forecasts.

    Eligible zero rates cannot enter a fit on `log(lambda)`, but dropping them
    silently would hide the population the fit does not describe — including the
    zero forecasts that went on to score. They are counted, not disappeared.
    """
    df = eligible_rows(df)
    sub = df[df.lam > 0]
    dropped = df[~(df.lam > 0)]
    outcomes = dropped["actual_tds"] if "actual_tds" in dropped else pd.Series(dtype=float)
    zero_counts = {
        "dropped_zero_lam": int(len(df) - len(sub)),
        "zero_lam_n": int(len(dropped)),
        "zero_lam_positive_outcome_n": int((outcomes > 0).sum()),
    }
    if not len(sub):
        return {**_unsupported_fit("no positive-rate rows", 0, 0), **zero_counts}
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
        min_clusters=min_clusters,
    )
    return {**out, **zero_counts}


def poisson_deviance(actual: np.ndarray, lam: np.ndarray) -> float:
    """Mean Poisson deviance. A proper scoring rule, so it cannot be improved by
    flattening the ranking — which a naive calibration fix would happily do."""
    actual = np.asarray(actual, dtype=float)
    lam = np.clip(np.asarray(lam, dtype=float), 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(actual > 0, actual * np.log(actual / lam), 0.0)
    return float(2.0 * np.mean(term - (actual - lam)))


# --- season and seed aggregation -------------------------------------------
def eligible_rows(df):
    return df[df.hard_eligible] if "hard_eligible" in df else df


def with_seed(df):
    return df if "seed" in df else df.assign(seed=-1)


def uncertainty(values):
    values = pd.Series(values).dropna()
    return float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan


def summarize_seeds(df, value, group=("model",)):
    """Equal seed weight within season, then equal season weight; SD is separate."""
    rows = []
    for keys, sub in df.groupby(list(group), dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        seasons = sub.groupby("season")[value].mean()
        variability = sub.groupby("season")[value].std(ddof=1)
        rows.append(
            dict(zip(group, keys, strict=True))
            | {
                "mean": seasons.mean(),
                "se": uncertainty(seasons),
                "seasons": seasons.count(),
                "shuffle_sd": variability.mean() if variability.notna().any() else 0.0,
            }
        )
    return pd.DataFrame(rows)


def ranking_cells(df, k=PRIMARY_K, rank=PRIMARY_RANK):
    """Mean outcome per ranked candidate in each common comparison cell."""
    df = with_seed(df)
    cells = df.attrs.get("cells") or list(
        df[["season", "week"]].drop_duplicates().itertuples(index=False, name=None)
    )
    if cells and len(cells[0]) == 2:
        cells = [(s, w, slot) for s, w in cells for slot in config.SLOTS]
    index = pd.MultiIndex.from_tuples(cells, names=["season", "week", "slot"])
    rows = []
    for (model, seed), sub in df.groupby(["model", "seed"], sort=True):
        part = (
            sub[sub[rank] <= k]
            .groupby(["season", "week", "slot"])
            .agg(
                actual=("actual_tds", "mean"),
                proj=("lam", "mean"),
                played=("played", "mean"),
                n=("lam", "size"),
            )
        )
        part = part.reindex(index[index.get_level_values("season").isin(sub.season.unique())])
        part["n"] = part.n.fillna(0).astype(int)
        rows.append(part.reset_index().assign(model=model, seed=seed, k=k, rank=rank))
    return pd.concat(rows, ignore_index=True)


def ranking_seasons(df, ks=TOP_K, rank=PRIMARY_RANK):
    rows = []
    for k in ks:
        cells = ranking_cells(df, k, rank)
        part = cells.groupby(["model", "seed", "season"], as_index=False).agg(
            actual=("actual", "mean"),
            proj=("proj", "mean"),
            played=("played", "mean"),
            candidates=("n", "sum"),
            covered_cells=("actual", "count"),
            total_cells=("n", "size"),
        )
        part["empty_cells"] = part.total_cells - part.covered_cells
        rows.append(part.assign(k=k, rank=rank, units="TDs per ranked candidate"))
    return pd.concat(rows, ignore_index=True)


def top_k(df, ks=TOP_K, rank=PRIMARY_RANK):
    per = ranking_seasons(df, ks, rank)
    rows = []
    for (model, k), sub in per.groupby(["model", "k"]):
        means = sub.groupby("season")[["actual", "proj", "played"]].mean().mean()
        pop = eligible_rows(df[df.model == model]).actual_tds.mean()
        rows.append(
            dict(
                model=model,
                k=k,
                rank=rank,
                n=sub.candidates.sum(),
                **means.to_dict(),
                ratio=means.proj / means.actual if means.actual else np.nan,
                lift=means.actual / pop if pop else np.nan,
            )
        )
    return pd.DataFrame(rows)


def paired_ranking_seasons(df, baseline="shipped", k=PRIMARY_K, rank=PRIMARY_RANK):
    cells = ranking_cells(df, k, rank)
    base = cells[cells.model == baseline]
    if base.empty or base.seed.nunique() != 1:
        raise ValueError("Paired comparisons require one selected deterministic baseline")
    keys = ["season", "week", "slot"]
    rows = []
    for (model, seed), sub in cells[cells.model != baseline].groupby(["model", "seed"]):
        paired = sub.merge(
            base[keys + ["actual", "n"]], on=keys, suffixes=("", "_base"), validate="one_to_one"
        )
        if not paired.n.eq(paired.n_base).all():
            raise ValueError("Comparison cells have different candidate coverage")
        paired["delta"] = paired.actual - paired.actual_base
        for season, group in paired.groupby("season"):
            rows.append(
                dict(
                    model=model,
                    seed=seed,
                    season=season,
                    k=k,
                    rank=rank,
                    delta=group.delta.mean(),
                    covered_cells=group.delta.count(),
                    jointly_empty_cells=group.actual.isna().sum(),
                    total_cells=len(group),
                )
            )
    return pd.DataFrame(rows)


def paired_top_k(df, baseline="shipped", k=PRIMARY_K, rank=PRIMARY_RANK):
    per = paired_ranking_seasons(df, baseline, k, rank)
    out = summarize_seeds(per, "delta").rename(columns={"mean": "delta"})
    for i, row in out.iterrows():
        sub = per[per.model == row.model]
        out.loc[i, "seasons_won"] = sub.groupby("season").delta.mean().gt(0).sum()
        for column in ("jointly_empty_cells", "covered_cells", "total_cells"):
            out.loc[i, column] = sub.groupby("season")[column].mean().sum()
    return out.assign(k=k, rank=rank, units="TDs per ranked candidate")


def spearman_seasons(df):
    rows = []
    for (model, seed, season), sub in with_seed(eligible_rows(df)).groupby(
        ["model", "seed", "season"]
    ):
        rhos = []
        for _, g in sub.groupby(["week", "slot"]):
            if g.actual_tds.nunique() > 1 and g.lam.nunique() > 1:
                rhos.append(
                    float(
                        stats.spearmanr(
                            g.lam.to_numpy(dtype=float), g.actual_tds.to_numpy(dtype=float)
                        ).statistic
                    )
                )
        rows.append(
            dict(
                model=model,
                seed=seed,
                season=season,
                spearman=np.mean(rhos) if rhos else np.nan,
                slot_weeks=len(rhos),
            )
        )
    return pd.DataFrame(rows)


def spearman(df):
    return summarize_seeds(spearman_seasons(df), "spearman").rename(columns={"mean": "spearman"})


def deviance_seasons(df, baseline="shipped", floor=DEVIANCE_FLOOR):
    df = with_seed(eligible_rows(df))
    keys = ["season", "week", "slot", "player_id"]
    keep = df.loc[(df.model == baseline) & (df.lam > floor), keys]
    sub = df.merge(keep, on=keys, validate="many_to_one")
    rows = []
    for (model, seed, season), g in sub.groupby(["model", "seed", "season"]):
        rows.append(
            dict(
                model=model,
                seed=seed,
                season=season,
                n=len(g),
                deviance=poisson_deviance(g.actual_tds.to_numpy(), g.lam.to_numpy()),
            )
        )
    return pd.DataFrame(rows)


def paired_deviance(df, baseline="shipped", floor=DEVIANCE_FLOOR):
    per = deviance_seasons(df, baseline, floor)
    base = per[per.model == baseline].set_index("season").deviance
    per["delta"] = per.deviance - per.season.map(base)
    out = summarize_seeds(per[per.model != baseline], "delta").rename(columns={"mean": "delta"})
    return out


def by_season(df, k=1, rank=PRIMARY_RANK):
    """Ranking diagnostics by season and seed; no conversion to season scores."""
    return ranking_seasons(df, [k], rank)


def replay_summary(per, baseline="shipped"):
    keys = ["model", "strategy"]
    out = summarize_seeds(per, "total", keys).rename(columns={"mean": "tds_per_season"})
    base = per[(per.model == baseline) & (per.strategy == "greedy")].set_index("season").total
    paired = per.assign(delta=per.total - per.season.map(base))
    delta = summarize_seeds(paired, "delta", keys).rename(
        columns={"mean": "delta_vs_baseline_greedy", "se": "paired_se"}
    )
    return out.merge(delta[keys + ["delta_vs_baseline_greedy", "paired_se"]], on=keys)
