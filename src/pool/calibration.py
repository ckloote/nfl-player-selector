"""Phase 3B: past-only rate maps, fitted walk-forward and applied before the solver.

Phase 3A described the shipped model's rate errors by population, position, rate bin,
availability and forecast horizon, and stopped there on purpose: a slope is not a
correction. A pooled per-season slope near 0.87 in all fifteen seasons says the level is
wrong somewhere; it does not say which map to apply, to whom, or whether applying one
would change a single pick for the better.

This module answers that as an experiment rather than an adjustment. The specification is
frozen and dated before anything is fitted; each fold trains only on seasons that had
finished before the season it is applied to; the map is applied to the whole surface
before the optimizer prunes or discounts it; and every candidate replays its own
no-reuse history so an achieved-TD difference is a policy result rather than a rescaling
of the same picks. Production constants are untouched, and a candidate that fails to
improve anything is a completed result.

The map is `exp(a) * lam ** b`, which covers both declared families: `level` fixes `b`
at one and estimates the scale alone, `log_affine` estimates both. Zero rates map to
zero and hard exclusions stay masks, so the population the primary metric scores is
identical for every candidate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import config, diagnostics, models, recommend
from . import evaluate as ev

SCHEMA_VERSION = 1

IDENTITY = "shipped"
FAMILIES = ("level", "log_affine")
GROUPINGS = ("pooled", "position")
POOLED_GROUP = "all"
# WR and TE are separate everywhere. They share the FLEX slot, so one map fitted across
# both cannot reorder them against each other and two separate maps can -- which is a
# result to measure, not a detail to pool away.
POSITION_GROUPS = ("QB", "RB", "WR", "TE")
FALLBACK = ("own", "pooled", "identity")

MAP_TARGET = "availability_adjusted_lam"
FIT_HORIZONS = "pooled"
ROW_WEIGHT = "equal"
CLUSTER = "target_player_week"
ZERO_RATE_POLICY = "excluded_and_counted"
# The declared method, one supported value each. A specification that names a metric or
# an adjustment this module does not implement is a specification the run cannot keep, so
# it is refused before the first pair is read rather than silently executed as something
# else.
REFIT = "annual"
PRIMARY_METRIC = "paired_season_mean_poisson_deviance"
PRIMARY_POPULATION = "all_eligible_current_week"
UNCERTAINTY = "paired_season_t"
MULTIPLICITY = "holm"

COVERAGE_MARGINS = ("min_outcome_coverage_current", "min_outcome_coverage_future")
NONNEGATIVE_MARGINS = ("min_deviance_improvement", "max_policy_loss_td_per_season")
REQUIRED_MARGINS = NONNEGATIVE_MARGINS + COVERAGE_MARGINS
# The promotion rule as data. In comments it was outside the parsed configuration and so
# outside the run identity: "both strategies" could become "either strategy" without
# moving `config_hash`. Nothing here applies the rule -- that is a separate, dated
# authoring step -- but the conditions it will be applied under are now frozen with the
# rest of the specification.
PROMOTION = {
    "max_promoted": (1,),
    "require_deviance_margin": (True,),
    # The decision is the step-down's own result, not an interval. Holm's per-step levels
    # widen down the ranking, so a later step's local interval can exclude zero on a step
    # the procedure never reached; a rule written against those intervals would admit a
    # candidate Holm declined. `reject` is the adjusted answer and the rule reads it.
    "decision": ("holm_step_down",),
    "alpha": (0.05,),
    "direction": ("improvement",),
    "policy_strategies": ("both",),
    "require_coverage_floors": (True,),
}
# A comparison whose season differences never vary has an exactly determined bound and no
# sampling distribution. Declared here rather than discovered: a shared increasing map
# reproduces identity's whole pick history, so it is the expected case for the policy
# comparison and its treatment must be frozen with everything else.
ZERO_VARIANCE_POLICY = "degenerate_interval"
ALPHA = 0.05


# --- candidate naming -------------------------------------------------------
def candidate_name(family: str, grouping: str) -> str:
    return f"cal-{family.replace('_', '')}-{grouping}"


def candidates(spec: dict) -> list[tuple[str, str, str]]:
    """(family, grouping, candidate name) for every pre-registered candidate."""
    cal = spec["calibration_experiment"]
    return [
        (family, grouping, candidate_name(family, grouping))
        for family in cal["families"]
        for grouping in cal["groupings"]
    ]


# --- fitting ----------------------------------------------------------------
def _finish(fit: dict, family: str) -> dict:
    """Add the family and the intercept interval the level fit reports instead of a slope."""
    se = fit["se_intercept"]
    finite = bool(np.isfinite(se)) if se is not None else False
    return {
        **fit,
        "family": family,
        "intercept_lo": float(fit["intercept"] - 1.96 * se) if finite else np.nan,
        "intercept_hi": float(fit["intercept"] + 1.96 * se) if finite else np.nan,
    }


def fit_level(y, lam, cluster, min_clusters: int | None = None) -> dict:
    """Fit `E[Y] = c * lam` and report it as `exp(a) * lam ** 1`.

    `evaluate.poisson_glm` cannot express this: its design is `[1, log lam]`, so the
    exponent is always free. Holding it at one is not a special case of that fit, it is
    a different model -- the one that asks whether the shipped rates are merely the
    wrong size, and it has a closed-form maximum likelihood, `c = sum(y) / sum(lam)`.

    The standard error is the same cluster-robust sandwich `poisson_glm` uses, for the
    same reason: repeated forecasts of one target share its single outcome, so the
    rows are not independent and the model-based interval would be too narrow.
    """
    min_clusters = config.MIN_INFERENCE_CLUSTERS if min_clusters is None else min_clusters
    y = np.asarray(y, dtype=float)
    lam = np.asarray(lam, dtype=float)
    cluster = np.asarray(cluster)
    if not len(y) == len(lam) == len(cluster):
        raise ValueError("fit_level needs outcomes, rates and clusters of equal length")
    n = len(y)
    groups = pd.Series(cluster)
    n_groups = int(groups.nunique()) if n else 0

    def refuse(reason: str) -> dict:
        return _finish(ev.unsupported_fit(reason, n, n_groups), "level")

    if n == 0:
        return refuse("empty input")
    if not (np.isfinite(y).all() and np.isfinite(lam).all()):
        return refuse("non-finite inputs")
    if (y < 0).any():
        return refuse("negative outcomes")
    if (lam <= 0).any():
        return refuse("non-positive rates")
    if n <= 1:
        return refuse("n <= 1 for a one-parameter fit")
    total = float(y.sum())
    if total == 0.0:
        # c = 0 maps every rate to zero, which is not a calibration of the model.
        return refuse("all-zero outcomes")

    scale = total / float(lam.sum())
    mu = scale * lam
    # Observed information for the log scale is sum(mu), which at the MLE is sum(y).
    information = float(mu.sum())
    resid = y - mu
    meat = float(
        sum(resid[idx].sum() ** 2 for idx in groups.groupby(groups, sort=False).indices.values())
    )
    # `poisson_glm`'s finite-sample factor with one parameter instead of two: the
    # `(n - 1) / (n - k)` term is exactly one here, so only the cluster count remains.
    correction = n_groups / max(n_groups - 1, 1)
    se = float(np.sqrt(meat * correction) / information) if information > 0 else np.nan
    cluster_se = n_groups >= min_clusters and np.isfinite(se)
    return _finish(
        {
            "intercept": float(np.log(scale)),
            # Fixed by the family, not estimated: reported so an artifact reads the same
            # for both families, with no interval because there is no estimate.
            "slope": 1.0,
            "se_intercept": se if cluster_se else np.nan,
            "se_slope": np.nan,
            "slope_lo": np.nan,
            "slope_hi": np.nan,
            "n": int(n),
            "clusters": int(n_groups),
            "fit_status": ev.FIT_OK,
            "reason": None
            if cluster_se
            else f"cluster SEs suppressed: {n_groups} < {min_clusters}",
            "converged": True,
            "iterations": 0,
            "cluster_se": bool(cluster_se),
        },
        "level",
    )


def fit_log_affine(y, lam, cluster, min_clusters: int | None = None) -> dict:
    """Fit `E[Y] = exp(a) * lam ** b` -- `poisson_glm` on `log lam`, with `b > 0` required.

    A non-positive exponent is arithmetically a fit and semantically not a calibration:
    it ranks a better forecast below a worse one, so it is refused rather than applied.
    """
    lam = np.asarray(lam, dtype=float)
    if len(lam) and (lam <= 0).any():
        n_groups = int(pd.Series(np.asarray(cluster)).nunique())
        return _finish(ev.unsupported_fit("non-positive rates", len(lam), n_groups), "log_affine")
    with np.errstate(divide="ignore"):
        fit = ev.poisson_glm(
            np.asarray(y, dtype=float), np.log(lam), np.asarray(cluster), min_clusters=min_clusters
        )
    if fit["fit_status"] == ev.FIT_OK and not fit["slope"] > 0:
        fit = ev.unsupported_fit(
            "fitted exponent is not positive, so the map is not increasing",
            fit["n"],
            fit["clusters"],
            fit["iterations"],
        )
    return _finish(fit, "log_affine")


def fit_family(family: str, rows: pd.DataFrame, min_clusters: int | None = None) -> dict:
    """One family's fit on one group of forecast/outcome pairs."""
    y = rows.actual_tds.to_numpy(dtype=float)
    lam = rows.lam.to_numpy(dtype=float)
    cluster = diagnostics.cluster_key(rows)
    if family == "level":
        return fit_level(y, lam, cluster, min_clusters)
    if family == "log_affine":
        return fit_log_affine(y, lam, cluster, min_clusters)
    raise ValueError(f"unknown calibration family {family!r}; choose from {sorted(FAMILIES)}")


IDENTITY_COEF = {"a": 0.0, "b": 1.0, "source": "identity"}


def resolve_coefficients(
    fit: dict, family: str, min_rows: int, min_clusters: int, fallback: dict
) -> dict:
    """Use a group's own fit only if it is supported and big enough; else fall back.

    Declared before fitting, so a sparse group cannot be promoted by lowering the bar
    after its coefficient turns out to be the interesting one.
    """
    a = fit["intercept"]
    b = 1.0 if family == "level" else fit["slope"]
    supported = (
        fit["fit_status"] == ev.FIT_OK
        and fit["n"] >= min_rows
        and fit["clusters"] >= min_clusters
        and np.isfinite(a)
        and np.isfinite(b)
        and b > 0
    )
    if supported:
        return {"a": float(a), "b": float(b), "source": "own"}
    return dict(fallback)


# --- artifacts --------------------------------------------------------------
def artifact_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, indent=2, sort_keys=True, default=str).encode()
    ).hexdigest()


def load_artifact(path: Path) -> dict:
    artifact = json.loads(Path(path).read_text())
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported calibration artifact schema in {path}")
    stated = artifact.pop("artifact_hash")
    if artifact_hash(artifact) != stated:
        raise ValueError(f"Calibration artifact {path} does not match its recorded hash")
    artifact["artifact_hash"] = stated
    return artifact


def build_artifact(
    *,
    fold: int,
    family: str,
    grouping: str,
    spec: dict,
    rows: pd.DataFrame,
    training_digest: str,
    pooled_fit: dict,
    counts: dict | None = None,
    upstream: dict | None = None,
) -> dict:
    cal = spec["calibration_experiment"]
    min_rows, min_clusters = cal["min_rows"], cal["min_clusters"]
    pooled_coef = resolve_coefficients(pooled_fit, family, min_rows, min_clusters, IDENTITY_COEF)
    if grouping == "pooled":
        groups = {POOLED_GROUP: {**pooled_coef, "fit": pooled_fit}}
        default = pooled_coef
    else:
        # A position with no usable fit of its own inherits the pooled map, not identity:
        # the pooled fit is still evidence about that position, just not evidence about
        # how it differs. Only a pooled fit that is itself unusable falls through.
        inherited = (
            {"a": pooled_coef["a"], "b": pooled_coef["b"], "source": "pooled"}
            if pooled_coef["source"] == "own"
            else dict(IDENTITY_COEF)
        )
        groups = {}
        for position in POSITION_GROUPS:
            sub = rows[rows.position.eq(position)]
            fit = fit_family(family, sub, min_clusters)
            coef = resolve_coefficients(fit, family, min_rows, min_clusters, inherited)
            groups[position] = {**coef, "fit": fit}
        default = inherited
    payload = dict(
        schema_version=SCHEMA_VERSION,
        fold=int(fold),
        candidate=candidate_name(family, grouping),
        family=family,
        grouping=grouping,
        map_target=cal["map_target"],
        fit_horizons=cal["fit_horizons"],
        row_weight=cal["row_weight"],
        cluster=cal["cluster"],
        zero_rate_policy=cal["zero_rate_policy"],
        train_seasons=list(range(cal["train_start"], fold)),
        cutoff_season=fold - 1,
        training_digest=training_digest,
        training_counts=dict(counts or {}),
        training_upstream=dict(upstream or {}),
        training_rows=int(len(rows)),
        base_model=spec["baseline"],
        base_seed=-1,
        min_rows=min_rows,
        min_clusters=min_clusters,
        fallback=list(cal["fallback"]),
        default=default,
        groups=groups,
    )
    return {**payload, "artifact_hash": artifact_hash(payload)}


# --- applying ---------------------------------------------------------------
def mapped_lam(proj: pd.DataFrame, artifact: dict) -> np.ndarray:
    """`exp(a) * lam ** b` on positive rates; everything else is left exactly as it is.

    A zero rate stays zero whether it is an eligible zero forecast or a ruled-out
    player's mask, so the map changes neither who is eligible nor which rows the primary
    metric scores. Only `lam` moves: `avail_mult`, `hard_eligible` and the keys are the
    shipped model's own, and the optimizer has not yet pruned or discounted anything.
    """
    lam = proj.lam.to_numpy(dtype=float)
    if artifact["grouping"] == "pooled":
        coef = artifact["groups"][POOLED_GROUP]
        a = np.full(len(proj), float(coef["a"]))
        b = np.full(len(proj), float(coef["b"]))
    else:
        default = artifact["default"]
        table = {k: v for k, v in artifact["groups"].items()}
        a = proj.position.map(lambda p: float(table.get(p, default)["a"])).to_numpy(dtype=float)
        b = proj.position.map(lambda p: float(table.get(p, default)["b"])).to_numpy(dtype=float)
    positive = lam > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(positive, np.exp(a) * np.power(np.where(positive, lam, 1.0), b), lam)
    return out


def candidate_builders(folds: Path, spec: dict, season: int) -> dict:
    """Every candidate's builder for one apply season, bound to that season's fold.

    Built inside the process that will use them: a builder closes over a fitted artifact
    and cannot be pickled to a spawned worker, and binding the wrong fold would be an
    out-of-fold leak that nothing downstream could detect.
    """
    folds = Path(folds)
    out = {}
    for _family, _grouping, name in candidates(spec):
        artifact = load_artifact(folds / str(season) / f"{name}.json")
        if artifact["fold"] != season:
            raise ValueError(f"Artifact for {name} is fold {artifact['fold']}, not {season}")
        out[name] = models.calibrated(lambda proj, art=artifact: mapped_lam(proj, art))
    return out


# --- training pairs ---------------------------------------------------------
# What the fold digest covers. The keys alone said which rows were fitted but not what was
# in them, so two folds trained on identical keys and different rates, positions or
# outcomes hashed the same -- and one run's fold directory, checkpoint included, could be
# restored into another and pass every metadata and reconciliation check. These are the
# columns the coefficients are a function of, so a digest over them is a digest of the fit.
DIGEST_COLUMNS = ("season", "decision_week", "week", "player_id", "position", "lam", "actual_tds")

TRAINING_COUNTS = (
    "surface_rows",
    "hard_excluded",
    "eligible_zero_lam",
    "unresolved_outcome",
    "fitted_rows",
)


def training_pairs(pairs: Path, spec: dict, fold: int) -> tuple[pd.DataFrame, str, dict]:
    """The forecast/outcome pairs fold `fold` is allowed to see, their digest and its counts.

    Allowed means two things, and the second is the one that is easy to lose: the
    forecast was made before the cutoff, *and* the week it forecast has already been
    played and scored. A week-12 forecast of week 17 is an early timestamp attached to
    an outcome that did not exist yet; training on it would be reading the future
    through a row that looks past.

    Everything this drops is counted on the way past. "Excluded and counted" was declared
    of the training population and only implemented for the applied one, so the first
    fold's discarded rows -- 2011-2015 for fold 2016 -- appeared in no table at all. The
    counts reconcile: exclusions plus zero rates plus unresolved outcomes plus the fitted
    rows are the surface it started from.

    The digest covers the values the fit is a function of, not merely the keys, so "this
    artifact was fitted on these rows" is checkable and not just "on rows named these".
    """
    cal = spec["calibration_experiment"]
    seasons = list(range(cal["train_start"], fold))
    if not seasons:
        raise ValueError(f"Fold {fold} has no training seasons from {cal['train_start']}")
    rows = diagnostics.load_run(Path(pairs), model=spec["baseline"], seed=-1, seasons=seasons)
    eligible = rows.hard_eligible.fillna(False).astype(bool)
    positive = eligible & rows.lam.gt(0)
    counts = dict(
        surface_rows=int(len(rows)),
        hard_excluded=int((~eligible).sum()),
        eligible_zero_lam=int((eligible & ~rows.lam.gt(0)).sum()),
        unresolved_outcome=int((positive & ~rows.outcome_known).sum()),
    )
    rows = rows[positive & rows.outcome_known].reset_index(drop=True)
    counts["fitted_rows"] = int(len(rows))
    if rows.empty:
        raise ValueError(f"Fold {fold} has no eligible positive-rate training pairs")
    if int(rows.season.max()) >= fold:
        raise ValueError(f"Fold {fold} training pairs reach season {int(rows.season.max())}")
    if sum(counts[k] for k in TRAINING_COUNTS[1:]) != counts["surface_rows"]:
        raise ValueError(f"Fold {fold} training accounting does not reconcile: {counts}")
    values = rows[list(DIGEST_COLUMNS)].sort_values(list(DIGEST_COLUMNS))
    digest = hashlib.sha256(values.to_csv(index=False).encode()).hexdigest()
    return rows, digest, counts


def fit_folds(output: Path, spec: dict, log=print) -> Path:
    """Stage II: one fit per fold, from seasons that had finished before it."""
    from . import benchmark

    output = Path(output)
    folds = output / "folds"
    folds.mkdir(parents=True, exist_ok=True)
    for fold in spec["calibration_experiment"]["apply_seasons"]:
        directory = folds / str(fold)
        if benchmark.checkpoint_complete(directory, f"fold {fold}"):
            log(f"Resume: verified fitted fold {fold}")
            continue
        directory.mkdir(parents=True, exist_ok=True)
        rows, training_digest, counts = training_pairs(output / "pairs", spec, fold)
        log(f"Fitting fold {fold} on {len(rows)} pairs from {min(rows.season)}-{max(rows.season)}")
        # The stages that produced those pairs, named by the digest their checkpoints
        # recorded: a fold is then bound to the run it was fitted inside, not only to the
        # rows it saw.
        upstream = {
            str(season): benchmark.digest(output / "pairs" / str(season) / "checkpoint.json")
            for season in range(spec["calibration_experiment"]["train_start"], fold)
        }
        for family in spec["calibration_experiment"]["families"]:
            pooled_fit = fit_family(family, rows, spec["calibration_experiment"]["min_clusters"])
            for grouping in spec["calibration_experiment"]["groupings"]:
                artifact = build_artifact(
                    fold=fold,
                    family=family,
                    grouping=grouping,
                    spec=spec,
                    rows=rows,
                    training_digest=training_digest,
                    pooled_fit=pooled_fit,
                    counts=counts,
                    upstream=upstream,
                )
                benchmark.write_json(directory / f"{artifact['candidate']}.json", artifact)
        benchmark.write_checkpoint(directory)
    return folds


# --- static advice sensitivity ----------------------------------------------
def advice_rows(frames: dict, common: dict, model: str, seed: int, season: int) -> pd.DataFrame:
    """What `advise_slot` would have said at each decision, under one candidate's rates.

    Replay makes one decision per week through `plan_slot`, so it never exercises the
    hold-or-commit rule at all. That rule compares a season cost in touchdowns against a
    fixed `INFO_PREMIUM_TD`, so rescaling every rate rescales one side of the comparison
    and not the other, and holds can flip without any ranking changing.

    This is a static sensitivity and nothing more. Every candidate is given the shared
    baseline depletion, so the rows differ only by the map; and a flipped hold here is
    not a touchdown gained or lost, because measuring the value of waiting needs a
    multi-event replay that processes news in order. Phase 3C owns that.
    """
    rows = []
    for week in sorted(frames):
        proj = frames[week]
        used = set(common.get(week, set()))
        decided = proj.attrs.get("decision_at")
        now = pd.Timestamp(decided).to_pydatetime() if decided else None
        for slot in config.SLOTS:
            advice = recommend.advise_slot(proj, slot, week, used, {}, now=now)
            pick = advice.recommended
            alternative = advice.hold_alternative
            rows.append(
                dict(
                    model=model,
                    seed=seed,
                    season=season,
                    week=week,
                    slot=slot,
                    recommended=None if pick is None else pick.player_id,
                    recommended_lam=np.nan if pick is None else pick.lam,
                    early=bool(pick.early) if pick is not None else False,
                    hold=bool(advice.hold),
                    hold_alternative=None if alternative is None else alternative.player_id,
                    hold_alternative_cost=np.nan if alternative is None else alternative.cost,
                    premium=config.INFO_PREMIUM_TD,
                )
            )
    return pd.DataFrame(rows)


# --- run stages -------------------------------------------------------------
def run_stages(conn, frozen, output: Path, spec: dict, log=print) -> None:
    """Pairs, then folds, then apply -- in that order, with a barrier between each.

    Fold Y needs every season before Y, so the fitting stage cannot live inside a season
    worker. Stage III rebuilds the identity model for its own seasons rather than
    reusing Stage I's: comparing the two is how "identity reproduces unchanged rates,
    picks and scores" gets checked, and a check that shares its inputs with the thing it
    checks is not one.
    """
    from . import benchmark

    output = Path(output)
    pairs, apply_dir = output / "pairs", output / "apply"
    for directory in (pairs, apply_dir):
        directory.mkdir(parents=True, exist_ok=True)
        # `diagnostics.load_run` reads the study's own discount from here, so a stage
        # directory that is read as a run needs the resolved configuration beside it.
        (directory / "resolved-config.json").write_text(
            (output / "resolved-config.json").read_text()
        )
    log(f"Stage I: identity forecast/outcome pairs for {len(spec['seasons'])} seasons")
    benchmark.season_sweep(conn, frozen, spec, spec["seasons"], pairs, log)
    log("Stage II: fitting folds")
    folds = fit_folds(output, spec, log)
    apply_seasons = spec["calibration_experiment"]["apply_seasons"]
    log(f"Stage III: applying {len(candidates(spec))} candidates over {len(apply_seasons)} seasons")
    benchmark.season_sweep(conn, frozen, spec, apply_seasons, apply_dir, log, folds=folds)


# --- out-of-fold evidence ---------------------------------------------------
CURRENT, FUTURE = "current", "future"


def _out_of_fold(apply_dir: Path, model: str, season: int) -> pd.DataFrame:
    """One model's applied surface for one season, joined to its outcomes."""
    return diagnostics.load_run(Path(apply_dir), model=model, seed=-1, seasons=[season])


def training_accounting(folds: Path, spec: dict) -> pd.DataFrame:
    """Every training row each fold discarded, in the classes that mean different things.

    Read back from the fitted artifacts rather than recomputed, so the table describes the
    fit that happened. `fitted_rows` is the population the coefficients came from and the
    four counts before it are what stood between that and the surface it started with.
    """
    rows = []
    for fold in spec["calibration_experiment"]["apply_seasons"]:
        # One fit per fold reads one training population, so any candidate's artifact
        # carries it; take the first and check the rest agree rather than repeating it.
        counts = None
        for _f, _g, name in candidates(spec):
            artifact = load_artifact(Path(folds) / str(fold) / f"{name}.json")
            if counts is None:
                counts = artifact["training_counts"]
            elif artifact["training_counts"] != counts:
                raise ValueError(f"Fold {fold} candidates disagree about their training rows")
        rows.append(dict(fold=fold, **{k: counts[k] for k in TRAINING_COUNTS}))
    out = pd.DataFrame(rows)
    out["discarded"] = out.surface_rows - out.fitted_rows
    out["fitted_share"] = out.fitted_rows / out.surface_rows
    return out


def coverage_tables(apply_dir: Path, spec: dict) -> dict[str, pd.DataFrame]:
    """Where the applied surface has an outcome to be scored against, and where it has none.

    The declared coverage floors are conditions on this join, not on the schedule. Every
    game can be complete and scored while a player forecast for week 12 has left the pool
    by week 12 and has no target row at all -- and the further ahead the forecast, the
    more of that there is. A run that never measured it could satisfy its own game-count
    audit and still miss the floor it froze.

    Measured on identity alone, because a map preserves keys, masks and zeros: every
    candidate is scored on exactly these rows.
    """
    cal = spec["calibration_experiment"]
    by_horizon, zeros = [], []
    for season in cal["apply_seasons"]:
        rows = _out_of_fold(apply_dir, spec["baseline"], season)
        eligible = rows[rows.hard_eligible.fillna(False).astype(bool)]
        by_horizon.append(
            diagnostics.coverage(eligible).merge(
                eligible.groupby(["season", "horizon"], as_index=False).in_target_pool.sum(),
                on=["season", "horizon"],
            )
        )
        zeros.append(diagnostics.zero_accounting(rows).assign(season=season))
    coverage = pd.concat(by_horizon, ignore_index=True)
    # Collapse to the two populations the margins are stated over. The current week is
    # the primary metric's own population; everything beyond it is the surface the
    # optimizer plans against.
    coverage["population"] = np.where(coverage.horizon.eq("0"), CURRENT, FUTURE)
    counts = ["rows", "outcomes_known", "outcomes_missing", "in_target_pool"]
    grouped = coverage.groupby("population", as_index=False)[counts].sum()
    grouped["coverage"] = grouped.outcomes_known / grouped["rows"]
    # Retention is a different question and answering it with coverage is what made a
    # departed player look like an unavailable outcome. Both are reported.
    grouped["retention"] = grouped.in_target_pool / grouped["rows"]
    floors = {
        CURRENT: float(cal["margins"]["min_outcome_coverage_current"]),
        FUTURE: float(cal["margins"]["min_outcome_coverage_future"]),
    }
    grouped["declared_floor"] = grouped.population.map(floors)
    # A comparison of two measured numbers, not a promotion decision: whether the run met
    # a floor it froze is arithmetic, and hiding it would leave the floor unenforceable.
    grouped["meets_floor"] = grouped.coverage >= grouped.declared_floor
    return {
        "coverage_out_of_fold": coverage.drop(columns="population"),
        "coverage_margins": grouped,
        "zero_accounting": pd.concat(zeros, ignore_index=True),
    }


def stratified_scores(apply_dir: Path, spec: dict, baseline: str = IDENTITY) -> pd.DataFrame:
    """Out-of-fold proper score by forecast horizon and availability, per candidate.

    The frozen decisions pool every horizon into one fit with equal row weight, and note
    two consequences that only a stratified score can show: late target weeks are
    forecast, and so represented, more often than early ones, and for an exponent away
    from one the Questionable multiplier is rescaled nonlinearly and is no longer a clean
    multiplier on the calibrated rate. Reporting one pooled number would leave both
    declared and unmeasured.

    Season-paired against identity on the identical rows, one season at a time so the
    ten-season surface never has to be resident at once.
    """
    c = spec["calibration_experiment"]
    keys = ["decision_week", "week", "slot", "player_id"]
    per_season = []
    for season in c["apply_seasons"]:
        base = _out_of_fold(apply_dir, baseline, season)
        base = base[
            base.hard_eligible.fillna(False).astype(bool) & base.outcome_known & base.lam.gt(0)
        ]
        strata = base[keys + ["horizon", "availability", "actual_tds"]]
        for _f, _g, model in [(None, None, baseline), *candidates(spec)]:
            rows = base if model == baseline else _out_of_fold(apply_dir, model, season)
            frame = strata.merge(rows[keys + ["lam"]], on=keys, how="left", validate="one_to_one")
            # The map preserves keys, so a candidate missing one of identity's rows means
            # the two surfaces are not the same population and nothing paired below holds.
            if frame.lam.isna().any():
                raise ValueError(f"{model} is missing {int(frame.lam.isna().sum())} identity rows")
            frame = frame.assign(
                deviance=ev.poisson_deviance_terms(
                    frame.actual_tds.to_numpy(), frame.lam.to_numpy()
                )
            )
            per_season.append(
                frame.groupby(["horizon", "availability"], as_index=False)
                .agg(n=("deviance", "size"), deviance=("deviance", "mean"))
                .assign(model=model, season=season)
            )
    per = pd.concat(per_season, ignore_index=True)
    group = ["model", "horizon", "availability"]
    stratum = ["horizon", "availability", "season"]
    levels = ev.summarize_seeds(per, "deviance", group).rename(columns={"mean": "deviance"})
    base = per[per.model == baseline].set_index(stratum).deviance.rename("identity_deviance")
    paired = per.join(base, on=stratum)
    paired["delta"] = paired.deviance - paired.identity_deviance
    delta = ev.summarize_seeds(paired[paired.model != baseline], "delta", group).rename(
        columns={"mean": "delta", "se": "paired_se"}
    )
    rows = per.groupby(group, as_index=False).n.sum()
    return levels.merge(rows, on=group).merge(
        delta[group + ["delta", "paired_se"]], on=group, how="left"
    )


def fold_table(folds: Path, spec: dict) -> pd.DataFrame:
    """Every fitted group in every fold, with the fit that produced it."""
    rows = []
    for fold in spec["calibration_experiment"]["apply_seasons"]:
        for _family, _grouping, name in candidates(spec):
            artifact = load_artifact(Path(folds) / str(fold) / f"{name}.json")
            for group, entry in artifact["groups"].items():
                fit = entry["fit"]
                rows.append(
                    dict(
                        fold=artifact["fold"],
                        candidate=artifact["candidate"],
                        family=artifact["family"],
                        grouping=artifact["grouping"],
                        group=group,
                        a=entry["a"],
                        b=entry["b"],
                        source=entry["source"],
                        train_seasons=len(artifact["train_seasons"]),
                        cutoff_season=artifact["cutoff_season"],
                        training_rows=artifact["training_rows"],
                        training_digest=artifact["training_digest"],
                        artifact_hash=artifact["artifact_hash"],
                        **{k: fit[k] for k in ev.FIT_KEYS},
                    )
                )
    return pd.DataFrame(rows)


def policy_table(replays: pd.DataFrame, baseline: str = IDENTITY) -> pd.DataFrame:
    """Achieved TDs per season, and each candidate against its *own* strategy's identity.

    Not against identity greedy. A candidate's optimizer replay and the identity greedy
    replay differ by two things at once, and attributing that difference to the map
    would be attributing the solver to it as well.

    The paired interval travels with the difference because the declared margin is a
    non-inferiority bound: the rule compares the *lower* end of this interval against
    `-max_policy_loss_td_per_season`, and a standard error alone does not answer that.
    No Holm adjustment here -- the pre-registered family the correction applies to is the
    primary forecast comparison, and this is the policy check beside it.
    """
    per = replays[replays.strategy.isin(("greedy", "optimizer"))]
    keys = ["model", "strategy"]
    levels = ev.summarize_seeds(per, "total", keys).rename(columns={"mean": "tds_per_season"})
    base = (
        per[per.model == baseline].set_index(["strategy", "season"]).total.rename("identity_total")
    )
    paired = per.join(base, on=["strategy", "season"])
    paired["delta"] = paired.total - paired.identity_total
    delta = ev.summarize_seeds(paired[paired.model != baseline], "delta", keys)
    if not delta.empty:
        delta = paired_t(delta, "mean", "se")
    delta = delta.rename(columns={"mean": "delta_vs_own_identity", "se": "paired_se"})
    columns = (
        "delta_vs_own_identity",
        "paired_se",
        "seasons",
        "df",
        "t",
        "p_value",
        # Which kind of statement the bound is. A policy that reproduced identity's whole
        # history has a difference of exactly zero, and saying so is not the same as
        # having no bound to offer.
        "comparison",
        "lo",
        "hi",
    )
    picked = [c for c in columns if c in delta and c not in levels]
    return levels.merge(delta[keys + picked], on=keys, how="left")


def decision_changes(picks: pd.DataFrame, baseline: str = IDENTITY) -> pd.DataFrame:
    """How often a candidate's replay chose a different player than identity's did."""
    keys = ["season", "strategy", "week", "slot"]
    base = picks[picks.model == baseline].set_index(keys).player_id.rename("identity_player")
    joined = picks[picks.model != baseline].join(base, on=keys)
    joined["changed"] = joined.player_id.fillna("") != joined.identity_player.fillna("")
    grouped = joined.groupby(["model", "strategy", "season"], as_index=False).agg(
        decisions=("changed", "size"), changed=("changed", "sum")
    )
    grouped["changed_share"] = grouped.changed / grouped.decisions
    return grouped


def advice_changes(advice: pd.DataFrame, baseline: str = IDENTITY) -> pd.DataFrame:
    """Static `advise_slot` disagreements -- never reported as waiting-policy touchdowns."""
    keys = ["season", "week", "slot"]
    base = advice[advice.model == baseline].set_index(keys)[["recommended", "hold"]]
    base.columns = ["identity_recommended", "identity_hold"]
    joined = advice[advice.model != baseline].join(base, on=keys)
    joined["pick_changed"] = joined.recommended.fillna("") != joined.identity_recommended.fillna("")
    joined["hold_changed"] = joined.hold.astype(bool) != joined.identity_hold.astype(bool)
    return joined.groupby(["model", "season"], as_index=False).agg(
        decisions=("pick_changed", "size"),
        pick_changed=("pick_changed", "sum"),
        hold_changed=("hold_changed", "sum"),
    )


# How a row's interval was arrived at. `t` is the ordinary paired case. A comparison whose
# season differences are all identical has no sampling variation to describe: the bound is
# the difference itself, exactly, and that is a different statement from having no bound at
# all. Erasing it removed the non-inferiority bound from precisely the candidates that
# changed no decision -- a shared increasing map reproduces identity's whole pick history,
# so this is the expected case here, not a corner one.
COMPARISON_T = "t"
COMPARISON_DEGENERATE = "degenerate_zero_variance"
COMPARISON_UNAVAILABLE = "unavailable"


def _interval(frame: pd.DataFrame, column: str, se: str, level, prefix: str = "") -> pd.DataFrame:
    """A two-sided t interval on the season differences, at `level`, added in place.

    A degenerate comparison gets width zero at its own difference -- it is exact, not
    unknown -- and an unavailable one gets nothing.
    """
    out = frame
    delta = out[column].to_numpy(dtype=float)
    error = out[se].to_numpy(dtype=float)
    df = out.df.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        half = stats.t.isf(np.asarray(level, dtype=float) / 2.0, np.where(df > 0, df, 1.0)) * error
    degenerate = out.comparison.eq(COMPARISON_DEGENERATE).to_numpy()
    half = np.where(
        out.comparison.eq(COMPARISON_T).to_numpy(), half, np.where(degenerate, 0.0, np.nan)
    )
    out[f"{prefix}lo"], out[f"{prefix}hi"] = delta - half, delta + half
    return out


def paired_t(
    frame: pd.DataFrame, column: str = "delta", se: str = "se", alpha: float = ALPHA
) -> pd.DataFrame:
    """The paired season t, its p-value and its interval, for one comparison per row.

    Equal season weight is already in the mean and the standard error that reach here;
    what this adds is the reference distribution the specification declares. Ten seasons
    is nine degrees of freedom, and a normal would report a narrower interval than the
    evidence supports.

    A zero standard error is not a missing one. Every season differing by the same amount
    -- most often by nothing at all -- leaves the mean difference exactly determined and
    the t undefined, which is `degenerate_zero_variance`: no p-value, and an interval of
    width zero at the difference itself. `unavailable` is the genuinely uninformative
    case, a season too few or a difference that is not a number.
    """
    out = frame.copy()
    df = out.seasons.to_numpy(dtype=float) - 1.0
    error = out[se].to_numpy(dtype=float)
    delta = out[column].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["t"] = np.where(error > 0, delta / error, np.nan)
    out["df"] = df
    known = np.isfinite(delta) & np.isfinite(error) & (df > 0)
    out["comparison"] = np.where(
        known & (error > 0),
        COMPARISON_T,
        np.where(known, COMPARISON_DEGENERATE, COMPARISON_UNAVAILABLE),
    )
    safe = np.where(df > 0, df, 1.0)
    usable = out.comparison.eq(COMPARISON_T).to_numpy()
    out["p_value"] = np.where(usable, 2.0 * stats.t.sf(np.abs(out.t.to_numpy()), safe), np.nan)
    return _interval(out, column, se, np.full(len(out), alpha))


def paired_inference(
    frame: pd.DataFrame, column: str = "delta", se: str = "se", alpha: float = ALPHA
) -> pd.DataFrame:
    """Paired season t on each comparison, then a Holm step-down over the whole family.

    The specification declares `uncertainty = "paired_season_t"` and `multiplicity =
    "holm"`, and a standard error is neither. Ten seasons is a t with nine degrees of
    freedom, not a normal, and a rank beside an alpha is an ingredient rather than an
    adjusted inference: reading four comparisons each against its own displayed level is
    not the Holm procedure, because Holm stops at the first non-rejection.

    So this returns the finished quantities. `p_holm` is the step-down adjusted p-value
    with monotonicity enforced -- a later step can have a smaller raw p than an earlier
    step, and the adjusted sequence must not go back down -- and `reject` is the
    procedure's own decision, which is exactly "every step up to here rejected". `reject`
    is the adjusted result; nothing else here is.

    In particular `stage_lo`/`stage_hi` is not. It is the interval of each step's own
    local test, at that step's level, and Holm's levels *widen* down the ranking -- 0.0125
    then 0.0167 then 0.025 then 0.05 for four comparisons -- so a later step gets a
    narrower interval than an earlier one. It can therefore exclude zero on a step the
    procedure never reached. Four comparisons at delta = -0.02 with raw p of 0.020, 0.021,
    0.022 and 0.040 all have p_holm = 0.08 and reject nothing, yet the third and fourth
    local intervals lie entirely below zero. A promotion rule written against those
    intervals would admit a candidate the step-down declined, which is why the rule is
    written against `reject`. `lo`/`hi` is the ordinary unadjusted 95% interval, reported
    because it is what a single comparison would say and it is honest about being that.
    """
    out = paired_t(frame, column, se, alpha)
    # Sort by evidence, which for an unusable comparison is none: a fit that produced no
    # standard error must not take the first Holm step and stop the family behind it.
    # Every array below is read back from the sorted frame, because the step a row takes
    # and the interval it is given have to describe the same row.
    out = out.sort_values("p_value", ascending=True, kind="stable", na_position="last")
    out = out.reset_index(drop=True)
    total = len(out)
    out["holm_rank"] = np.arange(1, total + 1)
    out["holm_alpha"] = alpha / (total - out.holm_rank + 1)
    scaled = out.p_value.to_numpy(dtype=float) * (total - out.holm_rank.to_numpy() + 1)
    out["p_holm"] = np.minimum(np.maximum.accumulate(np.nan_to_num(scaled, nan=1.0)), 1.0)
    out["reject"] = (out.p_holm <= alpha) & out.p_value.notna()
    return _interval(out, column, se, out.holm_alpha.to_numpy(dtype=float), "stage_")
