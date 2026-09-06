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

REQUIRED_MARGINS = (
    "min_deviance_improvement",
    "max_policy_loss_td_per_season",
    "min_outcome_coverage_current",
    "min_outcome_coverage_future",
)


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
def training_pairs(pairs: Path, spec: dict, fold: int) -> tuple[pd.DataFrame, str]:
    """The forecast/outcome pairs fold `fold` is allowed to see, and their digest.

    Allowed means two things, and the second is the one that is easy to lose: the
    forecast was made before the cutoff, *and* the week it forecast has already been
    played and scored. A week-12 forecast of week 17 is an early timestamp attached to
    an outcome that did not exist yet; training on it would be reading the future
    through a row that looks past.

    The digest covers the exact keys that entered the fit, so "no evaluation-season row
    reached this artifact" is checkable afterwards rather than asserted.
    """
    cal = spec["calibration_experiment"]
    seasons = list(range(cal["train_start"], fold))
    if not seasons:
        raise ValueError(f"Fold {fold} has no training seasons from {cal['train_start']}")
    rows = diagnostics.load_run(Path(pairs), model=spec["baseline"], seed=-1, seasons=seasons)
    rows = rows[
        rows.hard_eligible.fillna(False).astype(bool) & (rows.lam > 0) & rows.outcome_known
    ].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"Fold {fold} has no eligible positive-rate training pairs")
    if int(rows.season.max()) >= fold:
        raise ValueError(f"Fold {fold} training pairs reach season {int(rows.season.max())}")
    keys = rows[["season", "decision_week", "week", "player_id"]].sort_values(
        ["season", "decision_week", "week", "player_id"]
    )
    return rows, hashlib.sha256(keys.to_csv(index=False).encode()).hexdigest()


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
        rows, training_digest = training_pairs(output / "pairs", spec, fold)
        log(f"Fitting fold {fold} on {len(rows)} pairs from {min(rows.season)}-{max(rows.season)}")
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


# --- metric tables ----------------------------------------------------------
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
    """
    per = replays[replays.strategy.isin(("greedy", "optimizer"))]
    keys = ["model", "strategy"]
    levels = ev.summarize_seeds(per, "total", keys).rename(columns={"mean": "tds_per_season"})
    base = (
        per[per.model == baseline].set_index(["strategy", "season"]).total.rename("identity_total")
    )
    paired = per.join(base, on=["strategy", "season"])
    paired["delta"] = paired.total - paired.identity_total
    delta = ev.summarize_seeds(paired[paired.model != baseline], "delta", keys).rename(
        columns={"mean": "delta_vs_own_identity", "se": "paired_se"}
    )
    return levels.merge(delta[keys + ["delta_vs_own_identity", "paired_se"]], on=keys, how="left")


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


def holm(pvalue_free: pd.DataFrame, column: str = "delta") -> pd.DataFrame:
    """Rank the pre-registered comparisons for a Holm adjustment on |delta| / se.

    The comparisons are declared in the frozen specification, so the family is fixed
    before the numbers exist. Reporting the rank and the adjusted level beside each
    comparison keeps the correction visible rather than folded into a verdict.
    """
    out = pvalue_free.copy()
    out["z"] = (out[column] / out["se"]).abs()
    out = out.sort_values("z", ascending=False, kind="stable").reset_index(drop=True)
    total = len(out)
    out["holm_rank"] = np.arange(1, total + 1)
    out["holm_alpha"] = 0.05 / (total - out.holm_rank + 1)
    return out
