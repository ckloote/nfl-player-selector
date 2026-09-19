"""Descriptive diagnostics over a benchmark run's saved forecasts: strata defined
before any outcome is read, every row accounted for, and no conclusion stated.
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from pool import config, db
from pool.cli import app
from pool.research import benchmark, diagnostics
from pool.research import evaluate as ev
from tests.support.season import PRIOR, SEASON, seed_season

STUDY = Path("data/experiments/roster-snapshot-repair")


# --- descriptive diagnostics ------------------------------------------------
def _diagnosable_run(tmp_path, monkeypatch):
    spec = benchmark.resolve(Path("experiments/phase2-validation.toml"))
    spec.update(
        seasons=[SEASON],
        history_start=PRIOR,
        models=["shipped", "random"],
        baseline="shipped",
        seeds=[0],
        random_trials=1,
        workers=1,
        eras={"test retrospective": [SEASON, SEASON]},
        expected_games={str(PRIOR): 8, str(SEASON): 8},
    )
    monkeypatch.setattr(benchmark, "resolve", lambda path: spec)
    monkeypatch.setattr(
        benchmark, "code_identity", lambda: dict(code_hash="t", revision="t", dirty=False)
    )
    conn = seed_season(db.connect(tmp_path / "diag-source.db"))
    out = tmp_path / "run"
    out.mkdir()
    destination = sqlite3.connect(out / "research.db")
    conn.backup(destination)
    destination.close()
    benchmark.run("unused", out, log=lambda x: None)
    return out


def test_group_definitions_do_not_depend_on_any_outcome(tmp_path, monkeypatch):
    """A stratum chosen after seeing which rows scored is not a diagnosis. Perturbing
    every outcome must leave the groups, and their sizes, exactly where they were."""
    run = _diagnosable_run(tmp_path, monkeypatch)
    rows = diagnostics.load_run(run)
    perturbed = rows.assign(actual_tds=rows.actual_tds * 7 + 3)

    before = {name: mask.to_numpy() for name, mask in diagnostics.population_masks(rows).items()}
    after = diagnostics.population_masks(perturbed)
    assert sorted(before) == sorted(diagnostics.POPULATIONS)
    for name, mask in before.items():
        assert (mask == after[name].to_numpy()).all(), name

    keys = ["population", "axis", "level", "horizon"]
    a, _ = diagnostics.strata(rows)
    b, _ = diagnostics.strata(perturbed)
    pd.testing.assert_frame_equal(a[[*keys, "n", "players"]], b[[*keys, "n", "players"]])


def _synthetic_rows():
    """A surface with the cases the shipped study happens not to contain.

    The saved study has no eligible zero rate at all and no tight end in the four-team
    fixture, so the groups that matter most to a calibration diagnosis would go
    untested against real data alone.
    """
    base = dict(
        season=2024,
        decision_week=1,
        game_id="g",
        baseline_spent=False,
        rank_available=1.0,
        picked_greedy=False,
        picked_optimizer=False,
        played=True,
    )
    rows = [
        # A tight end and a wide receiver in the same FLEX slot, same week.
        {
            **base,
            "week": 1,
            "player_id": "te",
            "slot": "FLEX",
            "position": "TE",
            "lam": 0.4,
            "avail_mult": 1.0,
            "hard_eligible": True,
            "actual_tds": 1.0,
        },
        {
            **base,
            "week": 1,
            "player_id": "wr",
            "slot": "FLEX",
            "position": "WR",
            "lam": 0.5,
            "avail_mult": 1.0,
            "hard_eligible": True,
            "actual_tds": 0.0,
        },
        # An eligible zero rate that scored anyway: selectable, forecast at zero.
        {
            **base,
            "week": 1,
            "player_id": "zero",
            "slot": "RB",
            "position": "RB",
            "lam": 0.0,
            "avail_mult": 1.0,
            "hard_eligible": True,
            "actual_tds": 1.0,
        },
        # A ruled-out player who played: a report error, not a calibration error.
        {
            **base,
            "week": 1,
            "player_id": "out",
            "slot": "QB",
            "position": "QB",
            "lam": 0.0,
            "avail_mult": 0.0,
            "hard_eligible": False,
            "actual_tds": 2.0,
        },
        # Excluded because his deadline has passed, not because anything was wrong with
        # the forecast: available, positive rate, and unpickable all the same.
        {
            **base,
            "week": 1,
            "player_id": "late",
            "slot": "QB",
            "position": "QB",
            "lam": 0.6,
            "avail_mult": 1.0,
            "hard_eligible": False,
            "actual_tds": 1.0,
        },
        # Questionable, and a future row with no target to score against.
        {
            **base,
            "week": 1,
            "player_id": "q",
            "slot": "RB",
            "position": "RB",
            "lam": 0.3,
            "avail_mult": 0.85,
            "hard_eligible": True,
            "actual_tds": 0.0,
        },
        {
            **base,
            "week": 4,
            "player_id": "gone",
            "slot": "WR",
            "position": None,
            "lam": 0.2,
            "avail_mult": 1.0,
            "hard_eligible": True,
            "actual_tds": None,
        },
    ]
    return diagnostics.prepare(pd.DataFrame(rows))


def test_wide_receivers_and_tight_ends_are_never_merged(tmp_path, monkeypatch):
    """They share the FLEX slot, so a map fitted on them together can reorder them
    against each other -- which is the one way a shared map changes a pick."""
    assert "WR" in diagnostics.POSITIONS and "TE" in diagnostics.POSITIONS
    described, _ = diagnostics.strata(_synthetic_rows())
    positions = set(described[described.axis.eq("position")].level)
    assert {"WR", "TE"} <= positions
    assert not {"FLEX", "WR/TE"} & positions
    per_position = described[
        described.axis.eq("position") & described.population.eq("all_eligible")
    ].set_index("level")
    assert per_position.loc["WR", "n"] == 1 and per_position.loc["TE", "n"] == 1

    run = _diagnosable_run(tmp_path, monkeypatch)
    real, _ = diagnostics.strata(diagnostics.load_run(run))
    assert set(real[real.axis.eq("position")].level) <= set(diagnostics.POSITIONS)


def test_no_fit_pools_two_forecast_horizons(tmp_path, monkeypatch):
    """A week-6 and a week-1 forecast of the same player-week are one outcome seen
    twice. Every fit is inside one horizon bucket, and the far ones cluster on the
    target they share rather than on the row."""
    run = _diagnosable_run(tmp_path, monkeypatch)
    rows = diagnostics.load_run(run)
    _, fitted = diagnostics.strata(rows)
    labels = {label for _, _, label in diagnostics.HORIZON_BUCKETS}
    assert set(fitted.horizon) <= labels
    assert fitted.horizon.notna().all()

    current = rows[rows.lead_horizon.eq(0)]
    future = rows[rows.lead_horizon.gt(0)]
    assert (
        len(set(diagnostics.cluster_key(current)))
        == current.groupby(["player_id", "season"]).ngroups
    )
    assert (
        len(set(diagnostics.cluster_key(future)))
        == future.groupby(["player_id", "season", "week"]).ngroups
    )
    # The repetition is real: a target week is forecast from several decision weeks.
    assert len(future) > future.groupby(["player_id", "season", "week"]).ngroups


def test_a_stratum_too_small_to_fit_exports_its_reason(tmp_path, monkeypatch):
    """The export must say which cells it could not fit and why, rather than leaving a
    blank that reads as a missing measurement."""
    run = _diagnosable_run(tmp_path, monkeypatch)
    _, fitted = diagnostics.strata(diagnostics.load_run(run))
    unsupported = fitted[fitted.fit_status.eq(ev.FIT_UNSUPPORTED)]
    assert len(unsupported)
    assert unsupported.reason.notna().all()
    assert unsupported.slope.isna().all()
    assert set(diagnostics.AXES) == set(fitted.axis)


def test_zero_rates_are_split_by_what_the_zero_means():
    """Three different things used to share one number. An eligible zero rate is a
    forecast the fit cannot take, and one that scores is exactly the row it fails on.
    A ruled-out player masked to zero who plays anyway is a report error. A player
    excluded because his deadline passed keeps a perfectly good positive forecast --
    calling his touchdowns a report error, as the note did, describes neither."""
    rows = _synthetic_rows()
    table = diagnostics.zero_accounting(rows)
    assert set(table.zero_class) == {
        "eligible_zero_rate",
        "exclusion: ruled out",
        "exclusion: deadline or kickoff",
    }
    # What a stratum contributes to no fit is a different count in the two cases. An
    # eligible stratum loses its zero rates; an excluded one loses all of itself, rate
    # or no rate. This assertion used to be `excluded_from_fit == zero_lam_n` for every
    # row, which is the defect written down: a positive-rate exclusion then reported 0.
    excluded = table.zero_class.str.startswith("exclusion")
    assert (table.excluded_from_fit[excluded] == table.n[excluded]).all()
    assert (table.excluded_from_fit[~excluded] == table.zero_lam_n[~excluded]).all()

    eligible = table[table.population.eq("all_eligible")].set_index(["position", "horizon"])
    assert eligible.loc[("RB", "0"), "zero_lam_n"] == 1
    assert eligible.loc[("RB", "0"), "zero_lam_positive_outcome_n"] == 1
    assert eligible.loc[("RB", "0"), "zero_lam_outcome_tds"] == 1.0

    ruled_out = table[table.population.eq("excluded_unavailable")].set_index("position")
    assert ruled_out.loc["QB", "zero_lam_n"] == 1
    assert ruled_out.loc["QB", "zero_lam_positive_outcome_n"] == 1

    # The deadline exclusion carries a positive rate, so it is not a zero at all and
    # its touchdown is not evidence about the injury report.
    late = table[table.population.eq("excluded_undecidable")].set_index("position")
    assert late.loc["QB", "n"] == 1 and late.loc["QB", "zero_lam_n"] == 0
    assert late.loc["QB", "outcome_tds"] == 1.0
    # It is still excluded from every fit. Counting only zeros reported nothing here.
    assert late.loc["QB", "excluded_from_fit"] == 1

    # Neither exclusion is in any eligible population, so neither reaches a fit.
    masks = diagnostics.population_masks(rows)
    indexed = rows.reset_index(drop=True)
    for pid in ("out", "late"):
        where = int(indexed.index[indexed.player_id.eq(pid)][0])
        assert not any(mask.iloc[where] for mask in masks.values())


def test_an_eligible_zero_rate_never_reaches_a_fit_but_is_always_counted():
    rows = _synthetic_rows()
    described, fitted = diagnostics.strata(rows)
    overall = described[
        described.population.eq("all_eligible")
        & described.axis.eq("overall")
        & described.horizon.eq("0")
    ].iloc[0]
    assert overall.n == 4 and overall.zero_lam_n == 1
    assert overall.zero_lam_positive_outcome_n == 1
    fit = fitted[
        fitted.population.eq("all_eligible") & fitted.axis.eq("overall") & fitted.horizon.eq("0")
    ].iloc[0]
    assert fit.n == 3  # the three positive rates with a known outcome


@pytest.mark.skipif(not STUDY.exists(), reason="saved study artifacts are not checked in")
def test_a_forecast_is_grouped_by_the_position_it_was_made_under():
    """Position came from the target week, so a player who changed position had his
    earlier forecasts reclassified using information that did not exist when they were
    made. 563 rows of the saved study moved that way: B.J. Daniels was forecast as a
    quarterback in 2015 and diagnosed as a receiver."""
    rows = diagnostics.load_run(STUDY)
    forecasts = pd.concat(
        [
            pd.read_parquet(STUDY / str(year) / "forecasts.parquet")[
                ["model", "seed", "season", "week", "player_id", "position"]
            ]
            for year in sorted(int(p.name) for p in STUDY.iterdir() if p.name.isdigit())
        ],
        ignore_index=True,
    )
    at_decision = (
        forecasts[forecasts.model.eq("shipped") & forecasts.seed.eq(-1)]
        .drop_duplicates(["season", "week", "player_id"])
        .rename(columns={"week": "decision_week", "position": "expected"})[
            ["season", "decision_week", "player_id", "expected"]
        ]
    )
    known = rows[rows.position.ne("unknown")]
    check = known.merge(at_decision, on=["season", "decision_week", "player_id"], how="left")
    assert check.expected.notna().all()
    assert (check.position == check.expected).all()
    # Forecast-time position is also the more complete join, not a trade for accuracy.
    assert rows.position.eq("unknown").mean() < 0.06


def test_available_means_not_yet_spent(tmp_path, monkeypatch):
    """`avail_mult > 0` is already required by hard eligibility, so `available` was a
    copy of `all_eligible` -- 962,332 identical rows in the saved study, every depleted
    one included. The split that means something is against the baseline's own walk."""
    run = _diagnosable_run(tmp_path, monkeypatch)
    rows = diagnostics.load_run(run)
    masks = diagnostics.population_masks(rows)
    assert not masks["available"].equals(masks["all_eligible"])
    assert not (masks["available"] & masks["depleted"]).any()

    # Both are decision-week statements; a future cell has no depletion to report.
    current = rows.lead_horizon.eq(0)
    assert (masks["available"] | masks["depleted"]).loc[~current].sum() == 0
    assert int((masks["available"] | masks["depleted"]).sum()) == int(
        (masks["all_eligible"] & current).sum()
    )


def test_position_strata_account_for_every_row_in_their_population():
    """Enumerating only QB/RB/WR/TE dropped the explicit `unknown` group, so at horizon
    7+ of the saved study the position rows summed to 304,284 against an overall
    363,276, each reporting full outcome coverage. The rows that vanished were exactly
    the ones with no outcome to report."""
    rows = _synthetic_rows()
    assert "unknown" in set(rows.position), "fixture must contain a row with no position"
    described, _ = diagnostics.strata(rows)
    overall = described[described.axis.eq("overall")].set_index(["population", "horizon"]).n
    by_position = (
        described[described.axis.eq("position")].groupby(["population", "horizon"]).n.sum()
    )
    assert not overall.empty
    pd.testing.assert_series_equal(
        overall.sort_index(), by_position.sort_index(), check_names=False
    )


def test_a_zero_forecast_is_counted_whether_or_not_its_outcome_resolved():
    """Counting zeros only among resolved outcomes made a stratum of one unscored zero
    forecast report n=1 and zero_lam_n=0, contradicting the population it describes."""
    rows = _synthetic_rows()
    unresolved = rows[rows.player_id.eq("gone")].assign(lam=0.0, actual_tds=np.nan)
    unresolved["outcome_known"] = False
    combined = pd.concat([rows, unresolved], ignore_index=True)

    described, _ = diagnostics.strata(combined)
    horizon3 = described[
        described.population.eq("all_eligible")
        & described.axis.eq("overall")
        & described.horizon.eq("2-3")
    ].iloc[0]
    assert horizon3.n == 2 and horizon3.outcomes_known == 0
    assert horizon3.zero_lam_n == 1
    assert horizon3.zero_lam_outcomes_known == 0
    assert horizon3.zero_lam_positive_outcome_n == 0


def test_a_forecast_with_no_target_week_is_missing_not_zero(tmp_path, monkeypatch):
    """A player forecast for a week he had left the pool by has nothing to score
    against. Counting that as a zero would score the roster feed, not the model."""
    run = _diagnosable_run(tmp_path, monkeypatch)
    rows = diagnostics.load_run(run)
    assert rows.actual_tds[~rows.outcome_known].isna().all()
    table = diagnostics.coverage(rows)
    assert (table.rows == table.outcomes_known + table.outcomes_missing).all()
    described, _ = diagnostics.strata(rows)
    assert (described.outcomes_known <= described.n).all()


def test_the_export_records_identities_and_states_no_conclusion(tmp_path, monkeypatch):
    """Generated artifacts carry facts, methods and provenance. Choosing a correction,
    or saying a result is good, belongs to a dated authored analysis."""
    run = _diagnosable_run(tmp_path, monkeypatch)
    out = diagnostics.export(run, tmp_path / "readiness", log=lambda x: None)
    assert {p.name for p in out.iterdir()} == {
        "strata.csv",
        "fits.csv",
        "fits-by-season.csv",
        "reliability.csv",
        "zero-accounting.csv",
        "coverage.csv",
        "identities.json",
        "READINESS.md",
    }
    identities = json.loads((out / "identities.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    forecasts, computed = identities["forecast_provenance"], identities["diagnostic_provenance"]
    assert forecasts["source_fingerprint"] == manifest["code"]["code_hash"]
    assert forecasts["frozen_dataset"] == manifest["dataset_hash"]
    assert forecasts["future_discount_source"] == "study"
    assert computed["diagnostics_module"] == diagnostics.module_hash("diagnostics")
    assert computed["evaluate_module"] == diagnostics.module_hash("evaluate")

    note = (out / "READINESS.md").read_text()
    assert "Production constants are unchanged" in note
    assert "not fitted correction artifacts" in note
    for claim in ("recommend", "should apply", "is well calibrated", "ready to ship", "improves"):
        assert claim not in note


def test_the_recorded_identity_follows_the_code_that_computes_the_metrics(tmp_path, monkeypatch):
    """Only the study was fingerprinted, so changing an inference constant changed
    whether intervals were reported at all while leaving `identities.json` byte for
    byte identical. The implementation describing the forecasts is its own identity."""
    run = _diagnosable_run(tmp_path, monkeypatch)
    before = diagnostics.provenance(run, "shipped", -1)
    with config.override(MIN_INFERENCE_CLUSTERS=config.MIN_INFERENCE_CLUSTERS + 1):
        after = diagnostics.provenance(run, "shipped", -1)
    assert before["forecast_provenance"] == after["forecast_provenance"]
    assert before["diagnostic_provenance"] != after["diagnostic_provenance"]
    assert (
        after["diagnostic_provenance"]["min_inference_clusters"]
        == before["diagnostic_provenance"]["min_inference_clusters"] + 1
    )


def test_planning_values_use_the_discount_the_study_ran_under(tmp_path, monkeypatch):
    """The forecasts on disk were planned under the discount in force when the run
    happened; a later edit to `config` must not restate them under a different one."""
    run = _diagnosable_run(tmp_path, monkeypatch)
    saved = diagnostics.study_constants(run)["FUTURE_DISCOUNT"]
    with config.override(FUTURE_DISCOUNT=0.5):
        rows = diagnostics.load_run(run)
        recorded = diagnostics.provenance(run, "shipped", -1)["forecast_provenance"]
    future = rows[rows.lead_horizon.gt(0)].iloc[0]
    assert future.planning_lam == pytest.approx(future.lam * saved**future.lead_horizon)
    assert recorded["future_discount"] == saved
    assert recorded["future_discount_source"] == "study"


def test_the_cli_reports_a_missing_surface_instead_of_a_traceback(tmp_path, monkeypatch):
    run = _diagnosable_run(tmp_path, monkeypatch)
    result = CliRunner().invoke(
        app,
        [
            "research",
            "diagnose",
            "--run",
            str(run),
            "--out",
            str(tmp_path / "x"),
            "--model",
            "no-such",
        ],
    )
    assert result.exit_code == 1
    assert "No saved surface" in result.output
