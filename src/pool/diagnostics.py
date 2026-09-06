"""Descriptive calibration diagnostics over a saved study's forecast surface.

Phase 3A asks what the model's rate errors look like *by population*, because the
one number the published study reports -- a pooled per-season slope below one -- does
not say which players, which rates, which availability states or which forecast
horizons it describes, and therefore does not specify a correction. This module
computes the breakdown and nothing else: it fits no correction, selects no model and
recommends nothing.

Three rules make the breakdown honest rather than a search:

* Group definitions are frozen before any outcome is read. Nothing here selects on
  future touchdowns or on whether a player took the field.
* Horizons are never pooled. A week-6 forecast and a week-1 forecast of the same
  player-week are not two observations, and the repeated future forecasts of one
  target share its single outcome, so they are clustered on that outcome.
* Eligible zero rates cannot enter a fit on `log(lambda)` and are excluded from it and
  from the deviance. They are counted instead -- including the zero forecasts that
  went on to score -- because that population is exactly what such a fit does not
  describe.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import evaluate as ev

SCHEMA_VERSION = 1

# Frozen before any outcome is read. `depleted`, the common-pool top-k and the two
# selected populations are properties of a decision week, so they exist only at
# horizon 0; the surface carries no rank or pick indicator for a future week.
POPULATIONS = (
    "all_eligible",
    "available",
    "depleted",
    "common_top1",
    "common_top3",
    "common_top5",
    "common_top10",
    "selected_greedy",
    "selected_optimizer",
)

# WR and TE are separate everywhere: they share the FLEX slot, so a map fitted on the
# two together can reorder them against each other.
POSITIONS = ("QB", "RB", "WR", "TE")

HORIZON_BUCKETS = ((0, 0, "0"), (1, 1, "1"), (2, 3, "2-3"), (4, 6, "4-6"), (7, 99, "7+"))

AXES = ("overall", "position", "rate_bin", "availability")


def horizon_bucket(horizon: pd.Series) -> pd.Series:
    out = pd.Series("7+", index=horizon.index, dtype=object)
    for lo, hi, label in HORIZON_BUCKETS:
        out[(horizon >= lo) & (horizon <= hi)] = label
    return out


def availability(avail_mult: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(avail_mult <= 0, "excluded", np.where(avail_mult < 1, "questionable", "full")),
        index=avail_mult.index,
        dtype=object,
    )


def load_run(run: Path, model: str = "shipped", seed: int = -1, seasons=None) -> pd.DataFrame:
    """Join a study's future surface to the outcomes recorded on its forecast rows.

    The surface carries no position and no outcome, and the forecast export keeps only
    each week's own decision row. Player-week facts -- position, touchdowns, whether a
    row exists at all -- come from the forecast row for the *target* week. Decision-week
    facts -- depletion, rank, which policy picked the player -- are attached only where
    the target week is the decision week; a later decision's rank says nothing about an
    earlier decision's plan.
    """
    run = Path(run)
    years = sorted(int(p.name) for p in run.iterdir() if p.is_dir() and p.name.isdigit())
    if seasons:
        years = [y for y in years if y in set(seasons)]
    if not years:
        raise ValueError(f"No completed seasons under {run}")
    parts = []
    for year in years:
        directory = run / str(year)
        path = directory / f"surface-{model}-{seed}.parquet"
        if not path.exists():
            raise ValueError(f"No saved surface for {model} seed {seed} in {year}")
        surface = pd.read_parquet(path)
        forecasts = pd.read_parquet(directory / "forecasts.parquet")
        forecasts = forecasts[forecasts.model.eq(model) & forecasts.seed.eq(seed)]
        player_week = forecasts[
            ["season", "week", "player_id", "position", "actual_tds", "played"]
        ].drop_duplicates(["season", "week", "player_id"])
        decision = forecasts[
            [
                "season",
                "week",
                "player_id",
                "baseline_spent",
                "rank_available",
                "picked_greedy",
                "picked_optimizer",
            ]
        ].drop_duplicates(["season", "week", "player_id"])
        joined = surface.merge(player_week, on=["season", "week", "player_id"], how="left").merge(
            decision, on=["season", "week", "player_id"], how="left"
        )
        parts.append(joined)
    return prepare(pd.concat(parts, ignore_index=True))


def prepare(rows: pd.DataFrame) -> pd.DataFrame:
    """Derive the frozen grouping axes from a joined surface."""
    rows = rows.copy()
    rows["lead_horizon"] = rows.week.astype(int) - rows.decision_week.astype(int)
    rows["horizon"] = horizon_bucket(rows.lead_horizon)
    rows["availability"] = availability(rows.avail_mult)
    rows["rate_bin"] = pd.cut(rows.lam, list(ev.LAMBDA_BINS), include_lowest=True).astype(str)
    # The raw rate is what a calibration map would act on; the planning value is what
    # the optimizer compares. Diagnosing one as the other conflates a rate error with
    # a discount that was never claimed to be calibrated.
    rows["planning_lam"] = rows.lam * config.FUTURE_DISCOUNT ** rows.lead_horizon.clip(lower=0)
    # Decision-week facts describe the decision, not the player-week: a later decision's
    # depletion or rank says nothing about an earlier decision's plan for that week.
    # Nullable dtypes so "not a decision-week row" stays distinct from False.
    future = rows.lead_horizon > 0
    for column in ("baseline_spent", "picked_greedy", "picked_optimizer"):
        rows[column] = rows[column].astype("boolean")
        rows.loc[future, column] = pd.NA
    rows["rank_available"] = rows.rank_available.astype("Float64")
    rows.loc[future, "rank_available"] = pd.NA
    rows["outcome_known"] = rows.actual_tds.notna()
    # A player forecast for a week he was no longer in the pool for has no forecast row
    # and therefore no position. Label it rather than leave it null, or every groupby
    # would drop it and the missing coverage would disappear from the tables reporting it.
    rows["position"] = rows.position.fillna("unknown")
    return rows


def population_masks(rows: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks over the surface, defined without reference to any outcome."""

    def flag(column):
        return rows[column].fillna(False).astype(bool)

    eligible = flag("hard_eligible")
    masks = {
        "all_eligible": eligible,
        "available": eligible & rows.avail_mult.gt(0),
        "depleted": eligible & flag("baseline_spent"),
        "selected_greedy": eligible & flag("picked_greedy"),
        "selected_optimizer": eligible & flag("picked_optimizer"),
    }
    for k in (1, 3, 5, 10):
        masks[f"common_top{k}"] = eligible & rows.rank_available.le(k).fillna(False).astype(bool)
    return {name: masks[name] for name in POPULATIONS}


def _cluster(sub: pd.DataFrame) -> np.ndarray:
    """Repeated forecasts of one target share its outcome, so the outcome is the unit.

    At horizon 0 there is one forecast per player-week and the binding repetition is a
    player's own weeks within a season. Beyond it, a player-week is forecast again at
    every earlier decision, and treating those as independent rows would shrink every
    interval by the number of times the tool happened to look.
    """
    if sub.lead_horizon.max() == 0:
        return (sub.player_id.astype(str) + "|" + sub.season.astype(str)).to_numpy()
    return (
        sub.player_id.astype(str) + "|" + sub.season.astype(str) + "|" + sub.week.astype(str)
    ).to_numpy()


def _stratum_rows(rows: pd.DataFrame, masks: dict[str, pd.Series]):
    """Every (population, axis level, horizon) group, with its rows."""
    for population, mask in masks.items():
        base = rows[mask]
        if base.empty:
            continue
        for horizon, by_horizon in base.groupby("horizon", sort=True):
            for axis in AXES:
                if axis == "overall":
                    yield population, axis, "all", horizon, by_horizon
                    continue
                levels = POSITIONS if axis == "position" else sorted(by_horizon[axis].unique())
                for level in levels:
                    sub = by_horizon[by_horizon[axis].eq(level)]
                    if not sub.empty:
                        yield population, axis, str(level), horizon, sub


def _describe(sub: pd.DataFrame) -> dict:
    scored = sub[sub.outcome_known]
    positive = scored[scored.lam > 0]
    zero = scored[~(scored.lam > 0)]
    return {
        "n": int(len(sub)),
        "outcomes_known": int(len(scored)),
        "outcome_coverage": float(len(scored) / len(sub)) if len(sub) else np.nan,
        "players": int(sub.player_id.nunique()),
        "seasons": int(sub.season.nunique()),
        "forecast_mean": float(sub.lam.mean()),
        "planning_mean": float(sub.planning_lam.mean()),
        "actual_mean": float(scored.actual_tds.mean()) if len(scored) else np.nan,
        "forecast_over_actual": (
            float(scored.lam.sum() / scored.actual_tds.sum())
            if len(scored) and scored.actual_tds.sum() > 0
            else np.nan
        ),
        "played_fraction": float(scored.played.mean()) if len(scored) else np.nan,
        # Zero rates are excluded from the fit and from the deviance, and counted here.
        "deviance_positive_rates": (
            ev.poisson_deviance(positive.actual_tds.to_numpy(), positive.lam.to_numpy())
            if len(positive)
            else np.nan
        ),
        "zero_lam_n": int(len(zero)),
        "zero_lam_positive_outcome_n": int((zero.actual_tds > 0).sum()),
        "zero_lam_outcome_tds": float(zero.actual_tds.sum()) if len(zero) else 0.0,
    }


def strata(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The descriptive table and the fits keyed to it, in one pass over the groups."""
    masks = population_masks(rows)
    described, fitted = [], []
    for population, axis, level, horizon, sub in _stratum_rows(rows, masks):
        key = dict(population=population, axis=axis, level=level, horizon=horizon)
        described.append({**key, **_describe(sub)})
        scored = sub[sub.outcome_known & sub.lam.gt(0)]
        fit = ev.poisson_glm(
            scored.actual_tds.to_numpy(dtype=float),
            np.log(scored.lam.to_numpy(dtype=float)),
            _cluster(scored),
        )
        fitted.append({**key, **fit})
    return pd.DataFrame(described), pd.DataFrame(fitted)


def by_season(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-season fits for the whole of each population at each horizon.

    Season is the unit any later paired comparison would use, so the per-season fits
    are reported rather than a single pooled coefficient standing in for fifteen.
    """
    masks = population_masks(rows)
    out = []
    for population, mask in masks.items():
        base = rows[mask]
        for (season, horizon), sub in base.groupby(["season", "horizon"], sort=True):
            scored = sub[sub.outcome_known & sub.lam.gt(0)]
            fit = ev.poisson_glm(
                scored.actual_tds.to_numpy(dtype=float),
                np.log(scored.lam.to_numpy(dtype=float)),
                _cluster(scored),
            )
            out.append(dict(population=population, season=int(season), horizon=horizon, **fit))
    return pd.DataFrame(out)


def reliability(rows: pd.DataFrame) -> pd.DataFrame:
    """The bin table by population, position and horizon; a pooled ratio hides tilt."""
    masks = population_masks(rows)
    out = []
    for population, mask in masks.items():
        base = rows[mask & rows.outcome_known]
        for (position, horizon), sub in base.groupby(["position", "horizon"], sort=True):
            table = ev.reliability(sub.assign(hard_eligible=True))
            out.append(table.assign(population=population, position=position, horizon=horizon))
    return (
        pd.concat(out, ignore_index=True)
        if out
        else pd.DataFrame(columns=["population", "position", "horizon"])
    )


def zero_accounting(rows: pd.DataFrame) -> pd.DataFrame:
    """Every zero rate, in the two classes that mean different things.

    An *eligible* zero rate is a candidate the tool would let you pick and forecasts at
    zero; it cannot enter a fit on `log(lambda)` and is the population such a fit does
    not describe. A *hard exclusion* is a player ruled out by the injury report, whose
    zero is a mask rather than a forecast; it never enters any fit or any eligible
    population. Reporting one number for both would hide whichever is the real one, and
    a ruled-out player who scores anyway is not a calibration error.
    """
    out = []
    groups = dict(population_masks(rows))
    groups["hard_excluded"] = ~rows.hard_eligible.fillna(False).astype(bool)
    for population, mask in groups.items():
        base = rows[mask]
        for (position, horizon), sub in base.groupby(["position", "horizon"], sort=True):
            scored = sub[sub.outcome_known]
            zero = scored[~(scored.lam > 0)]
            out.append(
                dict(
                    population=population,
                    position=position,
                    horizon=horizon,
                    zero_class=(
                        "hard_exclusion" if population == "hard_excluded" else "eligible_zero_rate"
                    ),
                    n=int(len(sub)),
                    outcomes_known=int(len(scored)),
                    zero_lam_n=int(len(zero)),
                    zero_lam_share=float(len(zero) / len(scored)) if len(scored) else np.nan,
                    zero_lam_positive_outcome_n=int((zero.actual_tds > 0).sum()),
                    zero_lam_outcome_tds=float(zero.actual_tds.sum()) if len(zero) else 0.0,
                    excluded_from_fit=int(len(zero)),
                )
            )
    return pd.DataFrame(out)


def coverage(rows: pd.DataFrame) -> pd.DataFrame:
    """Where the surface has no outcome at all.

    A player forecast for a week he had left the pool by has no target row, so there is
    nothing to score him against. A missing target is not a zero, and the fraction of
    them grows with the horizon, which is a limit on what the far-horizon fits describe.
    """
    grouped = rows.groupby(["season", "horizon"], sort=True)
    return pd.DataFrame(
        {
            "rows": grouped.size(),
            "outcomes_known": grouped.outcome_known.sum(),
            "outcomes_missing": grouped.size() - grouped.outcome_known.sum(),
            "coverage": grouped.outcome_known.mean(),
        }
    ).reset_index()


def module_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def readiness_note(run: Path, rows: pd.DataFrame, fits: pd.DataFrame, identities: dict) -> str:
    unsupported = fits[fits.fit_status.eq(ev.FIT_UNSUPPORTED)]
    reasons = unsupported.reason.value_counts().to_dict()
    no_interval = fits[fits.fit_status.eq(ev.FIT_OK) & ~fits.cluster_se.astype(bool)]
    lines = [
        "# Phase 3A Readiness Note",
        "",
        f"Generated {datetime.now(UTC).date().isoformat()} from `{run}`. Measurements and "
        "identities only; interpretation belongs in a separately authored, dated analysis.",
        "",
        "## Identities",
        "",
    ]
    lines += [f"- {k}: `{v}`" for k, v in identities.items()]
    lines += [
        "",
        "## Population",
        "",
        f"- Surface rows: {len(rows):,} over {rows.season.nunique()} seasons, "
        f"{rows.player_id.nunique():,} players, lead horizons "
        f"{int(rows.lead_horizon.min())}-{int(rows.lead_horizon.max())}.",
        f"- Rows with a resolved outcome: {int(rows.outcome_known.sum()):,} "
        f"({rows.outcome_known.mean():.4f}). A missing outcome is never read as a zero.",
        f"- Eligible zero rates (forecast zero but selectable): "
        f"{int((rows.hard_eligible & ~rows.lam.gt(0)).sum()):,}. These are the rows a fit on "
        "`log(lambda)` cannot take; they are excluded from every fit and from the deviance, "
        "and counted in `zero-accounting.csv`.",
        f"- Hard exclusions (ruled out, so masked to zero rather than forecast at zero): "
        f"{int((~rows.hard_eligible & rows.outcome_known).sum()):,} with a known outcome, of "
        f"which {int((~rows.hard_eligible & rows.actual_tds.gt(0)).sum()):,} scored anyway. "
        "These never enter an eligible population or a fit; a player who was ruled out and "
        "played is a report error, not a calibration error.",
        "",
        "## Group definitions",
        "",
        f"- Populations: {', '.join(POPULATIONS)}. Depletion, common-pool rank and the two "
        "selected populations are decision-week properties and exist only at horizon 0.",
        f"- Positions: {', '.join(POSITIONS)}, WR and TE separately throughout.",
        f"- Rate bins: {list(ev.LAMBDA_BINS)}.",
        "- Availability: excluded, questionable, full, from the availability multiplier.",
        f"- Lead horizons: {', '.join(label for _, _, label in HORIZON_BUCKETS)}, never pooled.",
        "- Every definition is fixed before any outcome is read; none selects on future "
        "touchdowns or on participation.",
        "",
        "## Fits",
        "",
        f"- Strata fitted: {len(fits):,}; unsupported: {len(unsupported):,}; supported but "
        f"without cluster standard errors: {len(no_interval):,} "
        f"(fewer than {config.MIN_INFERENCE_CLUSTERS} clusters).",
    ]
    lines += [f"  - {reason}: {count:,}" for reason, count in sorted(reasons.items())]
    lines += [
        "- Horizon 0 clusters on player-season; later horizons cluster on the shared target "
        "player-week, because repeated forecasts of one target share its single outcome.",
        "- Raw rates are diagnosed separately from the discounted planning values; both are "
        "exported.",
        "",
        "## Limits",
        "",
        "- Descriptive and retrospective. These are in-sample diagnostic coefficients on "
        "seasons that have already been explored, not fitted correction artifacts and not "
        "an out-of-sample evaluation.",
        "- A slope is not a correction. Neither a stratum's coefficient nor the set of them "
        "identifies a mapping; Phase 3B specifies and freezes that separately, before fitting.",
        "- Nothing here authorizes a production change. Production constants are unchanged.",
        "",
    ]
    return "\n".join(lines)


def export(
    run: Path, out: Path, *, model: str = "shipped", seed: int = -1, seasons=None, log=print
) -> Path:
    run, out = Path(run), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_run(run, model=model, seed=seed, seasons=seasons)
    log(f"Loaded {len(rows):,} surface rows for {model} seed {seed}")
    described, fitted = strata(rows)
    tables = {
        "strata.csv": described,
        "fits.csv": fitted,
        "fits-by-season.csv": by_season(rows),
        "reliability.csv": reliability(rows),
        "zero-accounting.csv": zero_accounting(rows),
        "coverage.csv": coverage(rows),
    }
    for name, frame in tables.items():
        frame.insert(0, "schema_version", SCHEMA_VERSION)
        frame.to_csv(out / name, index=False, float_format="%.12g")
        log(f"  {name}: {len(frame):,} rows")

    manifest = json.loads((run / "manifest.json").read_text())
    identities = {
        "experiment": run.name,
        "model": f"{model} seed {seed}",
        "source fingerprint": manifest["code"]["code_hash"],
        "code revision": manifest["code"].get("revision"),
        "frozen dataset": manifest["dataset_hash"],
        "configuration": manifest["identity"]["config_hash"],
        "diagnostics module": module_hash(),
        "artifact schema": SCHEMA_VERSION,
    }
    (out / "identities.json").write_text(json.dumps(identities, indent=2, sort_keys=True) + "\n")
    (out / "READINESS.md").write_text(readiness_note(run, rows, fitted, identities))
    log(f"Wrote {out}")
    return out
