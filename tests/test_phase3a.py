"""Phase 3A: evidence, capture and readiness repairs.

Each test states the failure it prevents. The guards here exist because the
unguarded fit returned plausible numbers for populations that identify nothing,
and a plausible number survives into a published table.
"""

import contextlib
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import optimize
from typer.testing import CliRunner

from pool import benchmark, capture, config, db, diagnostics, snapshots, state
from pool import evaluate as ev
from pool import projections as P
from pool.cli import app
from pool.recommend import advise_week
from tests.test_backtest import PRIOR, SEASON, WEEKS, _seed
from tests.test_phase2 import archive_all

STUDY = Path("data/experiments/roster-snapshot-repair")


# --- trusted reference ------------------------------------------------------
def _reference_mle(y, x):
    """An independent Poisson MLE, sharing no code with `evaluate.poisson_glm`.

    Minimising the negative log-likelihood with a general-purpose optimiser is a
    different route to the same estimand; agreeing with it is evidence the IRLS
    implementation is correct rather than merely self-consistent.
    """
    design = np.column_stack([np.ones_like(x), x])

    def nll(beta):
        eta = design @ beta
        return float(np.sum(np.exp(eta) - y * eta))

    return optimize.minimize(nll, np.zeros(2), method="BFGS", tol=1e-14).x


def _reference_model_se(y, x, beta):
    """Model-based SEs from a numerically differentiated Hessian of the same NLL."""
    design = np.column_stack([np.ones_like(x), x])

    def nll(b):
        eta = design @ b
        return float(np.sum(np.exp(eta) - y * eta))

    step, hess = 1e-5, np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            a, b = np.zeros(2), np.zeros(2)
            a[i] = b[i] = step
            a[j] += step
            b[j] -= step
            hess[i, j] = (nll(beta + a) - nll(beta + b) - nll(beta - b) + nll(beta - a)) / (
                4 * step * step
            )
    return np.sqrt(np.diag(np.linalg.inv(hess)))


def test_an_identifiable_fit_matches_an_independent_maximum_likelihood_estimate():
    rng = np.random.default_rng(11)
    x = rng.normal(-1.0, 0.8, 20000)
    y = rng.poisson(np.exp(0.2 + 0.75 * x))
    fit = ev.poisson_glm(y, x, np.arange(len(y)))
    beta = _reference_mle(y, x)
    assert fit["fit_status"] == ev.FIT_OK and fit["converged"]
    assert fit["intercept"] == pytest.approx(beta[0], abs=1e-6)
    assert fit["slope"] == pytest.approx(beta[1], abs=1e-6)
    # With one row per cluster the sandwich is a robust SE; on a correctly specified
    # Poisson it should sit close to the model-based SE, not an order of magnitude off.
    se = _reference_model_se(y, x, beta)
    assert fit["se_intercept"] == pytest.approx(se[0], rel=0.10)
    assert fit["se_slope"] == pytest.approx(se[1], rel=0.10)


def test_the_guards_do_not_move_a_supported_fit():
    """The repair must add refusals, not new numbers. This is the unguarded
    implementation inlined, so a future change to the fit itself is visible."""
    rng = np.random.default_rng(12)
    x = rng.normal(-1.0, 0.8, 5000)
    y = rng.poisson(np.exp(0.1 + 0.9 * x))
    cluster = np.repeat(np.arange(100), 50)

    design = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(50):
        mu = np.exp(np.clip(design @ beta, -30, 30))
        step = np.linalg.pinv(design.T @ (design * mu[:, None])) @ (design.T @ (y - mu))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break

    fit = ev.poisson_glm(y, x, cluster)
    assert fit["intercept"] == pytest.approx(float(beta[0]), abs=1e-12)
    assert fit["slope"] == pytest.approx(float(beta[1]), abs=1e-12)


@pytest.mark.skipif(not STUDY.exists(), reason="saved study artifacts are not checked in")
def test_a_published_calibration_row_reproduces_under_the_guards():
    saved = pd.read_csv("experiments/results/roster-snapshot-repair/calibration.csv")
    row = saved[saved.model.eq("shipped") & saved.season.eq(2019) & saved.seed.eq(-1)].iloc[0]
    df = pd.read_parquet(STUDY / "2019" / "forecasts.parquet")
    fit = ev.calibration(df[df.model.eq("shipped") & df.seed.eq(-1)])
    assert fit["fit_status"] == ev.FIT_OK and fit["cluster_se"]
    for key in ("intercept", "slope", "se_intercept", "se_slope"):
        assert fit[key] == pytest.approx(float(row[key]), abs=5e-7)
    for key in ("n", "clusters", "dropped_zero_lam"):
        assert int(fit[key]) == int(row[key])


# --- unsupported fits -------------------------------------------------------
def _varied(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(-1.0, 0.8, n)
    return rng.poisson(np.exp(0.2 + 0.75 * x)), x


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("empty", "empty input"),
        ("one_row", "n <= 2 for a two-parameter fit"),
        ("two_rows", "n <= 2 for a two-parameter fit"),
        ("constant_x", "design rank 1 (constant log rate)"),
        ("all_zero_y", "all-zero outcomes"),
        ("negative_y", "negative outcomes"),
        ("nan_x", "non-finite inputs"),
        ("neg_inf_x", "non-finite inputs"),
        ("nan_y", "non-finite inputs"),
    ],
)
def test_a_population_that_identifies_nothing_yields_a_reason_not_a_coefficient(case, reason):
    """Each of these used to return a finite slope and a finite interval."""
    y, x = _varied(200, seed=3)
    if case == "empty":
        y, x = np.array([]), np.array([])
    elif case == "one_row":
        y, x = y[:1], x[:1]
    elif case == "two_rows":
        y, x = y[:2], x[:2]
    elif case == "constant_x":
        x = np.full(len(y), 0.3)
    elif case == "all_zero_y":
        y = np.zeros(len(y))
    elif case == "negative_y":
        y = y.astype(float)
        y[0] = -1.0
    elif case == "nan_x":
        x = x.copy()
        x[5] = np.nan
    elif case == "neg_inf_x":
        x = x.copy()
        x[5] = -np.inf  # what log(0) would supply for an eligible zero rate
    elif case == "nan_y":
        y = y.astype(float)
        y[5] = np.nan

    fit = ev.poisson_glm(y, x, np.arange(len(y)))
    assert fit["fit_status"] == ev.FIT_UNSUPPORTED
    assert fit["reason"] == reason
    assert not fit["converged"] and not fit["cluster_se"]
    for key in ("intercept", "slope", "se_intercept", "se_slope", "slope_lo", "slope_hi"):
        assert np.isnan(fit[key]), key
    assert set(ev.FIT_KEYS) <= set(fit)


def test_separation_is_refused_rather_than_certified():
    """Every outcome zero on one side of a threshold and positive on the other: the
    likelihood climbs without bound and the estimate does not exist. IRLS still came to
    rest -- against the clip that keeps `exp` from overflowing -- and reported slope
    300 with `converged=True` and an interval of no width."""
    y, x = np.tile([0.0, 1.0], 100), np.tile([-0.1, 0.0], 100)
    fit = ev.poisson_glm(y, x, np.arange(len(y)))
    assert fit["fit_status"] == ev.FIT_UNSUPPORTED
    assert fit["reason"] == "no finite maximum-likelihood estimate (separation)"
    assert np.isnan(fit["slope"]) and np.isnan(fit["se_slope"])
    assert not fit["cluster_se"]


def test_a_separated_population_is_refused_however_it_fails():
    """A wider gap between the two levels diverges instead of settling on the clip, so
    it is the iteration guard that catches it rather than the separation check. Either
    way it must not come back as an estimate; the control proves the guards are
    refusing separation and not merely any two-level design."""
    rng = np.random.default_rng(21)
    x = np.repeat([0.0, 2.0], 500)
    separated = np.concatenate([np.zeros(500), rng.poisson(4.0, 500) + 1.0])
    fit = ev.poisson_glm(separated, x, np.arange(len(x)))
    assert fit["fit_status"] == ev.FIT_UNSUPPORTED
    assert np.isnan(fit["slope"])

    both_score = np.concatenate([rng.poisson(0.3, 500), rng.poisson(4.0, 500)])
    control = ev.poisson_glm(both_score, x, np.arange(len(x)))
    assert control["fit_status"] == ev.FIT_OK
    assert np.isfinite(control["slope"])


@pytest.mark.skipif(not STUDY.exists(), reason="saved study artifacts are not checked in")
def test_an_interior_fit_is_nowhere_near_the_refusal_bounds():
    """The guards must refuse a non-existent estimate without touching a real one. The
    published season rests four log units from the clip and thirteen orders of
    magnitude from the conditioning bound."""
    df = pd.read_parquet(STUDY / "2019" / "forecasts.parquet")
    sub = df[df.model.eq("shipped") & df.seed.eq(-1) & df.hard_eligible & df.lam.gt(0)]
    x = np.log(sub.lam.to_numpy(dtype=float))
    fit = ev.poisson_glm(
        sub.actual_tds.to_numpy(dtype=float),
        x,
        (sub.player_id + "|" + sub.season.astype(str)).to_numpy(),
    )
    assert fit["fit_status"] == ev.FIT_OK
    design = np.column_stack([np.ones_like(x), x])
    eta = design @ np.array([fit["intercept"], fit["slope"]])
    mu = np.exp(eta)
    assert np.max(np.abs(eta)) < ev.LINEAR_PREDICTOR_BOUND / 3
    condition = np.linalg.cond(design.T @ (design * mu[:, None]))
    assert condition < ev.MAX_INFORMATION_CONDITION / 1e6


def test_exhausted_iterations_are_a_refusal_not_a_partial_answer():
    """A fit stopped mid-Newton is not an estimate; the old loop returned it silently."""
    y, x = _varied(2000, seed=4)
    fit = ev.poisson_glm(y, x, np.arange(len(y)), max_iter=1)
    assert fit["fit_status"] == ev.FIT_UNSUPPORTED
    assert fit["reason"] == "iterations exhausted before convergence"
    assert fit["iterations"] == 1
    assert np.isnan(fit["slope"])


def test_a_converged_fit_reports_how_many_iterations_it_took():
    y, x = _varied(2000, seed=5)
    fit = ev.poisson_glm(y, x, np.arange(len(y)))
    assert fit["converged"] and 0 < fit["iterations"] < 50


# --- cluster-robust inference ----------------------------------------------
@pytest.mark.parametrize("n_groups", [1, 2, config.MIN_INFERENCE_CLUSTERS - 1])
def test_too_few_clusters_keeps_the_estimate_and_withholds_the_interval(n_groups):
    """One cluster scores the sandwich at the MLE, where the gradient is zero, so the
    interval had no width at all. Point estimates stay: they are still identified."""
    y, x = _varied(600, seed=6)
    fit = ev.poisson_glm(y, x, np.arange(len(y)) % n_groups)
    assert fit["fit_status"] == ev.FIT_OK and fit["converged"]
    assert fit["clusters"] == n_groups
    assert not fit["cluster_se"]
    assert str(n_groups) in fit["reason"] and str(config.MIN_INFERENCE_CLUSTERS) in fit["reason"]
    assert np.isfinite(fit["intercept"]) and np.isfinite(fit["slope"])
    for key in ("se_intercept", "se_slope", "slope_lo", "slope_hi"):
        assert np.isnan(fit[key]), key


def test_the_minimum_cluster_count_is_predeclared_and_met_by_the_saved_study():
    """A threshold chosen after seeing a stratum's fit is not a threshold. It lives in
    config, is recorded in every run's constants, and passes every published season."""
    assert config.MIN_INFERENCE_CLUSTERS == 30
    saved = pd.read_csv("experiments/results/roster-snapshot-repair/calibration.csv")
    assert saved.clusters.min() >= config.MIN_INFERENCE_CLUSTERS


def test_mismatched_input_lengths_are_a_caller_bug_not_a_data_state():
    with pytest.raises(ValueError, match="equal length"):
        ev.poisson_glm(np.zeros(5), np.zeros(4), np.zeros(5))


# --- zero-rate accounting ---------------------------------------------------
def test_eligible_zero_rates_are_counted_including_the_ones_that_scored():
    """A zero forecast whose player scored is the most informative row the GLM cannot
    take. Dropping it silently would hide exactly the population the fit misses."""
    df = pd.DataFrame(
        {
            "hard_eligible": True,
            "season": 2024,
            "player_id": [f"p{i}" for i in range(8)],
            "lam": [0.0, 0.0, 0.0, 0.4, 0.5, 0.6, 0.7, 0.8],
            "actual_tds": [1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        }
    )
    fit = ev.calibration(df)
    assert fit["zero_lam_n"] == 3
    assert fit["zero_lam_positive_outcome_n"] == 2
    assert fit["dropped_zero_lam"] == 3
    assert fit["n"] == 5


def test_a_population_with_no_positive_rates_reports_its_exclusions():
    fit = ev.calibration(pd.DataFrame(dict(hard_eligible=[True] * 3, lam=[0.0] * 3)))
    assert fit["fit_status"] == ev.FIT_UNSUPPORTED
    assert fit["reason"] == "no positive-rate rows"
    assert fit["n"] == 0 and fit["zero_lam_n"] == 3
    assert set(ev.FIT_KEYS) <= set(fit)


# --- one same-week stats contract -------------------------------------------
THU_KICK, SUN_KICK = "2024-09-19T20:15", "2024-09-22T13:00"
PRE_WEEK3 = "2024-09-19T18:00:00+00:00"  # Thursday afternoon, before kickoff
POST_THU = "2024-09-20T12:00:00+00:00"  # the Thursday result is in the feed
DECISION = "2024-09-20T16:00:00+00:00"  # Friday: Thursday locked, Sunday open
POST_SUN = "2024-09-23T12:00:00+00:00"  # everything else has been played


def _dicts(conn, sql, params=()):
    return [dict(r) for r in conn.execute(sql, params)]


def _insert(conn, table, rows):
    if not rows:
        return
    cols = list(rows[0])
    conn.executemany(
        f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})",
        [tuple(r[c] for c in cols) for r in rows],
    )


def _staged(tmp_path):
    """A week-3 decision with one game already played and the rest still to come.

    The shared fixture kicks every game off at the same hour, which is precisely the
    situation in which the live and snapshot stats rules cannot disagree. Give week 3
    a Thursday game and a Sunday game, then withhold everything from week 3 on so the
    feed can publish it in the order it really would.
    """
    conn = _seed(db.connect(tmp_path / "parity.db"))
    with conn:
        for week, kick in ((1, "2024-09-08T13:00"), (2, "2024-09-15T13:00")):
            conn.execute(
                "UPDATE games SET kickoff = ? WHERE season = ? AND week = ?", (kick, SEASON, week)
            )
        conn.execute("UPDATE games SET kickoff = ? WHERE game_id = ?", (THU_KICK, "g2024-3-AAA"))
        conn.execute("UPDATE games SET kickoff = ? WHERE game_id = ?", (SUN_KICK, "g2024-3-CCC"))
        conn.execute(
            "UPDATE games SET kickoff = '2024-09-29T13:00' WHERE season = ? AND week = 4",
            (SEASON,),
        )
    unplayed = {
        "player_weeks": _dicts(
            conn, "SELECT * FROM player_weeks WHERE season = ? AND week >= 3", (SEASON,)
        ),
        "rosters": _dicts(conn, "SELECT * FROM rosters WHERE season = ? AND week >= 4", (SEASON,)),
        "game_results": _dicts(
            conn, "SELECT * FROM game_results WHERE season = ? AND week >= 3", (SEASON,)
        ),
        "touchdown_credits": _dicts(
            conn,
            "SELECT t.* FROM touchdown_credits t JOIN games g USING(game_id) "
            "WHERE g.season = ? AND g.week >= 3",
            (SEASON,),
        ),
    }
    with conn:
        conn.execute("DELETE FROM player_weeks WHERE season = ? AND week >= 3", (SEASON,))
        conn.execute("DELETE FROM rosters WHERE season = ? AND week >= 4", (SEASON,))
        conn.execute(
            "DELETE FROM touchdown_credits WHERE game_id IN "
            "(SELECT game_id FROM games WHERE season = ? AND week >= 3)",
            (SEASON,),
        )
        conn.execute("DELETE FROM game_results WHERE season = ? AND week >= 3", (SEASON,))
    return conn, unplayed


def _publish(conn, unplayed, which):
    """Publish the Thursday game's rows, or everything that follows them.

    Next week's roster is part of "everything that follows": a week-4 snapshot does not
    exist on the Friday of week 3, and letting one path see it would make the parity
    comparison pass for the wrong reason.
    """
    thursday = which == "thursday"
    with conn:
        for table in ("player_weeks", "game_results", "touchdown_credits"):
            rows = [r for r in unplayed[table] if (r["game_id"] == "g2024-3-AAA") == thursday]
            _insert(conn, table, rows)
        if not thursday:
            _insert(conn, "rosters", unplayed["rosters"])


def _advice(proj, now):
    out = []
    for a in advise_week(proj, 3, set(), {slot: {} for slot in config.SLOTS}, now=now):
        out.append(
            (
                a.slot,
                None if a.recommended is None else a.recommended.player_id,
                None if a.recommended is None else round(a.recommended.cost, 10),
                a.hold,
                None if a.hold_alternative is None else a.hold_alternative.player_id,
                {w: a.plan.players.iloc[r].player_id for w, r in a.plan.assignment.items()},
            )
        )
    return out


MODEL_COLUMNS = [
    "base_rate",
    "def_mult",
    "vegas_mult",
    "home_mult",
    "avail_mult",
    "role_mult",
    "lam",
]


def _model_view(proj):
    return proj.set_index(["week", "slot", "player_id"])[MODEL_COLUMNS].sort_index()


def test_an_observed_early_game_reaches_the_same_decision_live_and_from_snapshots(tmp_path):
    """Live loading read every stat row it had, including Thursday's; snapshot replay
    cut at `week < W` and threw the same row away. Same instant, same archive, two
    different forecasts -- so nothing measured in replay described the live model."""
    conn, unplayed = _staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    _publish(conn, unplayed, "thursday")
    for feed in ("player_stats", "touchdowns"):
        snapshots.archive(conn, SEASON, feed, observed_at=POST_THU)

    now = datetime.fromisoformat(DECISION)
    live = P.build_projections(P.load_frames(conn, SEASON, input_policy="live"), 3, "usage")
    live_advice = _advice(live, now)

    # Everything after the decision arrives before the replay is run.
    _publish(conn, unplayed, "rest")
    for feed in ("player_stats", "touchdowns"):
        snapshots.archive(conn, SEASON, feed, observed_at=POST_SUN)

    frames = P.load_frames(conn, SEASON, 3, input_policy="snapshots", decision_at=DECISION)
    assert sorted(frames.pw_cur.week.unique()) == [1, 2, 3]
    assert set(frames.pw_cur[frames.pw_cur.week.eq(3)].team) == {"AAA", "BBB"}
    replayed = P.build_projections(frames, 3, "usage")

    pd.testing.assert_frame_equal(_model_view(live), _model_view(replayed))
    assert live_advice == _advice(replayed, now)
    # The Thursday game has locked; its players stay planned for a later week.
    thursday = replayed[replayed.week.eq(3) & replayed.team.isin(["AAA", "BBB"])]
    assert not thursday.hard_eligible.any()


def test_a_decision_before_the_feed_published_cannot_see_the_early_game(tmp_path):
    """Availability is the observation time, not the kickoff: one microsecond before
    the import, the Thursday result does not exist for the decision."""
    conn, unplayed = _staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    _publish(conn, unplayed, "thursday")
    for feed in ("player_stats", "touchdowns"):
        snapshots.archive(conn, SEASON, feed, observed_at=POST_THU)

    just_before = "2024-09-20T11:59:59.999999+00:00"
    before = P.load_frames(conn, SEASON, 3, input_policy="snapshots", decision_at=just_before)
    assert sorted(before.pw_cur.week.unique()) == [1, 2]
    at = P.load_frames(conn, SEASON, 3, input_policy="snapshots", decision_at=POST_THU)
    assert sorted(at.pw_cur.week.unique()) == [1, 2, 3]


def test_later_imports_cannot_change_an_earlier_decision(tmp_path):
    """A correction published after the pick was made must not rewrite the pick's
    inputs. Reconstructing the same instant twice, across an import, must not move."""
    conn, unplayed = _staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    _publish(conn, unplayed, "thursday")
    for feed in ("player_stats", "touchdowns"):
        snapshots.archive(conn, SEASON, feed, observed_at=POST_THU)
    before = P.build_projections(
        P.load_frames(conn, SEASON, 3, input_policy="snapshots", decision_at=DECISION), 3, "usage"
    )

    _publish(conn, unplayed, "rest")
    with conn:
        conn.execute(
            "UPDATE player_weeks SET rec_td = rec_td + 3 WHERE season = ? AND week = 3", (SEASON,)
        )
    for feed in ("player_stats", "touchdowns"):
        snapshots.archive(conn, SEASON, feed, observed_at=POST_SUN)

    after = P.build_projections(
        P.load_frames(conn, SEASON, 3, input_policy="snapshots", decision_at=DECISION), 3, "usage"
    )
    pd.testing.assert_frame_equal(before, after)


def test_historical_replay_still_stops_at_the_previous_week(tmp_path):
    """The historical policy has no observation times to consult, so it cannot know
    which of week W's games had finished. It keeps its documented approximation."""
    conn, _ = _staged(tmp_path)
    frames = P.load_frames(conn, SEASON, 3)
    assert sorted(frames.pw_cur.week.unique()) == [1, 2]
    assert P._stats_through("historical", 3) == 2
    assert P._stats_through("legacy-closing", 3) == 2
    assert P._stats_through("snapshots", 3) == 3
    assert P._stats_through("live", None) is None


# --- frozen decision-time inputs --------------------------------------------
def _times_csv(path, weeks=WEEKS, stamp="2024-09-01T00:00:00Z"):
    path.write_text("season,week,decision_at\n" + "".join(f"{SEASON},{w},{stamp}\n" for w in weeks))
    return path


def _snapshot_spec(csv_path, **overrides):
    spec = benchmark.resolve(Path("experiments/phase2-validation.toml"))
    spec.update(
        seasons=[SEASON],
        history_start=PRIOR,
        models=["shipped", "random"],
        baseline="shipped",
        seeds=[0, 1],
        random_trials=2,
        workers=1,
        eras={"test retrospective": [SEASON, SEASON]},
        expected_games={str(PRIOR): 8, str(SEASON): 8},
        input_policy="snapshots",
        role_source="depth",
        decision_times=str(csv_path),
    )
    spec.update(overrides)
    spec["resolved_decision_times"] = benchmark.decision_time_records(spec)
    return spec


def _research_db(tmp_path, name):
    conn = _seed(db.connect(tmp_path / f"{name}-source.db"))
    archive_all(conn, "2024-09-01T00:00:00Z")
    out = tmp_path / name
    out.mkdir()
    destination = sqlite3.connect(out / "research.db")
    conn.backup(destination)
    destination.close()
    return out


def test_editing_the_decision_csv_in_place_changes_the_run_identity(tmp_path):
    """The identity covered the path, so the same path with different timestamps was
    the same run. The contents are the input; the path is where it happened to live."""
    csv = _times_csv(tmp_path / "times.csv")
    before = _snapshot_spec(csv)
    _times_csv(csv, stamp="2024-09-02T00:00:00Z")
    after = _snapshot_spec(csv)

    assert before["decision_times"] == after["decision_times"]
    assert before["resolved_decision_times"] != after["resolved_decision_times"]
    digest = lambda spec: hashlib.sha256(benchmark.json_text(spec).encode()).hexdigest()  # noqa: E731
    assert digest(before) != digest(after)


@pytest.mark.parametrize(
    ("contents", "match"),
    [
        ("season,week,decision_at\n2024,1,2024-09-01T00:00:00\n", "timezone"),
        (
            "season,week,decision_at\n2024,1,2024-09-01T00:00:00Z\n2024,1,2024-09-02T00:00:00Z\n",
            "Duplicate",
        ),
        ("season,week\n2024,1\n", "decision_at"),
    ],
)
def test_an_unusable_decision_csv_fails_while_resolving_the_configuration(
    tmp_path, contents, match
):
    """These already failed -- inside a worker, after the manifest was written."""
    csv = tmp_path / "times.csv"
    csv.write_text(contents)
    with pytest.raises(ValueError, match=match):
        _snapshot_spec(csv)


def test_a_missing_week_fails_before_any_worker_starts(tmp_path, monkeypatch):
    """A fifteen-season run should not discover a gap in season eleven."""
    spec = _snapshot_spec(_times_csv(tmp_path / "times.csv", weeks=[1, 2]))
    monkeypatch.setattr(benchmark, "resolve", lambda path: spec)
    monkeypatch.setattr(
        benchmark, "code_identity", lambda: dict(code_hash="t", revision="t", dirty=False)
    )
    monkeypatch.setattr(
        benchmark, "evaluate_season", lambda *a: pytest.fail("started a worker anyway")
    )
    out = _research_db(tmp_path, "gap")
    with pytest.raises(ValueError, match="missing timestamps"):
        benchmark.run("unused", out, log=lambda x: None)
    assert not (out / "manifest.json").exists()


@pytest.mark.parametrize(
    ("policy", "path", "match"),
    [
        ("snapshots", None, "requires decision_times"),
        ("historical", "times.csv", "requires input_policy"),
    ],
)
def test_a_snapshot_policy_and_a_decision_csv_require_each_other(tmp_path, policy, path, match):
    _times_csv(tmp_path / "times.csv")
    spec = benchmark.resolve(Path("experiments/phase2-validation.toml"))
    spec["input_policy"] = policy
    spec["decision_times"] = None if path is None else str(tmp_path / path)
    with pytest.raises(ValueError, match=match):
        benchmark.decision_time_records(spec)


def test_workers_read_the_frozen_timestamps_not_the_path(tmp_path, monkeypatch):
    """Each worker is a spawned process with its own working directory, and re-read the
    path there. The frozen records travel in the pickled specification instead."""
    spec = _snapshot_spec(_times_csv(tmp_path / "times.csv"))
    monkeypatch.setattr(benchmark, "resolve", lambda path: spec)
    monkeypatch.setattr(
        benchmark, "code_identity", lambda: dict(code_hash="t", revision="t", dirty=False)
    )
    monkeypatch.setattr(
        snapshots, "decision_times", lambda path: pytest.fail("re-read the mutable path")
    )
    compact = benchmark.run("unused", _research_db(tmp_path, "frozen"), log=lambda x: None)
    assert (compact.parent / "decision-times.csv").exists()
    assert (compact / "EVALUATION.md").exists()


def test_resume_rejects_an_edited_decision_csv_at_the_same_path(tmp_path, monkeypatch):
    """The whole point: a run continued after its inputs changed under it."""
    csv = _times_csv(tmp_path / "times.csv")
    spec = _snapshot_spec(csv)
    monkeypatch.setattr(benchmark, "resolve", lambda path: spec)
    monkeypatch.setattr(
        benchmark, "code_identity", lambda: dict(code_hash="t", revision="t", dirty=False)
    )
    out = _research_db(tmp_path, "resume")
    benchmark.run("unused", out, log=lambda x: None)

    unchanged = _snapshot_spec(csv)
    monkeypatch.setattr(benchmark, "resolve", lambda path: unchanged)
    monkeypatch.setattr(
        benchmark, "evaluate_season", lambda *a: pytest.fail("recomputed a saved season")
    )
    benchmark.run("unused", out, resume=True, log=lambda x: None)

    _times_csv(csv, stamp="2024-09-02T00:00:00Z")
    edited = _snapshot_spec(csv)
    monkeypatch.setattr(benchmark, "resolve", lambda path: edited)
    with pytest.raises(ValueError, match="fingerprint"):
        benchmark.run("unused", out, resume=True, log=lambda x: None)


# --- append-only decision capture -------------------------------------------
def _decide(conn, week, decided, used=None, locked=None):
    used = set() if used is None else used
    locked = {slot: {} for slot in config.SLOTS} if locked is None else locked
    now = state.eastern_now(decided)
    proj = P.projections_for(conn, SEASON, from_week=week)
    advice = advise_week(proj, week, used, locked, now=now)
    decision_id = capture.record_decision(
        conn, SEASON, week, proj, advice, used, locked, decision_at=decided
    )
    return decision_id, proj, advice


def test_the_capture_stores_the_whole_surface_not_the_shortlist(tmp_path):
    """The optimizer prunes to eighty candidates and the recommender shows six. A
    diagnostic asking about eligible zero rates, or about a player at a four-week
    horizon, has to find them here, because nothing downstream keeps them."""
    conn, unplayed = _staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    _publish(conn, unplayed, "thursday")
    decision_id, proj, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))

    frame, _, _, _ = capture.reconstruct(conn, decision_id)
    assert len(frame) == len(proj)
    assert set(frame.columns) >= set(P.PROJECTION_COLUMNS) | set(capture.SURFACE_EXTRAS)
    assert sorted(frame.week.unique()) == [3, 4]  # the future surface, not just this week
    assert sorted(frame.slot.unique()) == sorted(config.SLOTS)
    assert set(frame.lead_horizon) == {0, 1}
    # The elapsed Thursday cells are kept and labelled, not dropped: live loading puts
    # the deadline in the recommender rather than in `hard_eligible`.
    assert frame.hard_eligible.all()
    assert set(frame.decision_status) == {"available", "deadline passed"}
    elapsed = frame[frame.decision_status.eq("deadline passed")]
    assert set(elapsed.week) == {3} and set(elapsed.team) == {"AAA", "BBB"}
    assert (frame.mapped_lam == frame.original_lam).all()  # identity calibrator in 3A


def test_a_captured_decision_reconstructs_its_advice_from_the_surface_alone(tmp_path):
    """The 3A acceptance case. If the re-derived advice differs from the recorded
    advice, something the decision depended on was never written down."""
    conn, unplayed = _staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    _publish(conn, unplayed, "thursday")
    decision_id, _, advice = _decide(conn, 3, datetime.fromisoformat(DECISION))

    _, redone, recorded, drift = capture.reconstruct(conn, decision_id)
    assert not drift["code_hash_changed"] and not drift["constants_changed"]
    assert sorted(recorded) == sorted(config.SLOTS)
    for original, again in zip(advice, redone, strict=True):
        detail = recorded[original.slot]
        assert detail["recommended"]["player_id"] == again.recommended.player_id
        assert detail["recommended"]["cost"] == pytest.approx(again.recommended.cost)
        assert detail["hold"] == again.hold
        assert detail["plan"] == {
            str(w): again.plan.players.iloc[r].player_id for w, r in again.plan.assignment.items()
        }
        assert [c["player_id"] for c in detail["alternatives"]] == [
            c.player_id for c in again.alternatives
        ]


def test_hold_and_commit_are_events_in_their_own_right(tmp_path):
    """The early deadline is the decision the pool actually forces; recording only the
    pick would lose whether the tool said to wait for Sunday's news."""
    conn, unplayed = _staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    _publish(conn, unplayed, "thursday")
    # Before the Thursday kickoff, so its players are still live and early.
    _decide(conn, 3, datetime.fromisoformat("2024-09-19T18:30:00+00:00"))
    kinds = set(capture.events(conn, SEASON).kind)
    assert {"surface", "advice"} <= kinds
    assert kinds & {"hold", "commit"}


def test_captured_inputs_name_the_observations_the_decision_could_see(tmp_path):
    conn, unplayed = _staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    _publish(conn, unplayed, "thursday")
    for feed in ("player_stats", "touchdowns"):
        snapshots.archive(conn, SEASON, feed, observed_at=POST_THU)
    decision_id, _, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))

    inputs = db.read_df(conn, "SELECT * FROM decision_inputs WHERE decision_id = ?", (decision_id,))
    assert set(inputs.feed) == set(snapshots.TABLES)
    assert inputs.missing.eq(0).all()
    stats = inputs[inputs.season.eq(SEASON) & inputs.feed.eq("player_stats")].iloc[0]
    assert stats.observed_at == snapshots.timestamp(POST_THU)
    stored = conn.execute(
        "SELECT content_hash FROM input_observations WHERE observation_id = ?",
        (int(stats.observation_id),),
    ).fetchone()
    assert stored["content_hash"] == stats.content_hash


def test_a_decision_survives_a_later_feed_correction(tmp_path):
    """The reason outcomes are joined at read time and never stored on an event."""
    conn, unplayed = _staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    _publish(conn, unplayed, "thursday")
    decision_id, _, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))
    before, advice_before, recorded, _ = capture.reconstruct(conn, decision_id)

    _publish(conn, unplayed, "rest")
    with conn:
        conn.execute("UPDATE player_weeks SET rec_td = rec_td + 5 WHERE season = ?", (SEASON,))
    for feed in ("player_stats", "touchdowns"):
        snapshots.archive(conn, SEASON, feed, observed_at=POST_SUN)

    after, advice_after, recorded_after, _ = capture.reconstruct(conn, decision_id)
    pd.testing.assert_frame_equal(before, after)
    assert recorded == recorded_after
    assert [a.recommended.player_id for a in advice_before] == [
        a.recommended.player_id for a in advice_after
    ]


def test_a_pick_and_its_history_change_together_or_not_at_all(tmp_path, monkeypatch):
    """`my_picks` is overwritten in place, so a replacement that commits without its
    correction event destroys the identity it replaced with no way back. Capture must
    not be able to fail after the pick has already moved."""
    path = tmp_path / "pool.db"
    _seed(db.connect(path)).close()
    runner = CliRunner()
    args = ["--season", str(SEASON), "--db", str(path)]
    assert runner.invoke(app, ["record", "--week", "1", "--rb", "AAA RB1", *args]).exit_code == 0

    # Whatever makes recording history impossible -- here, no source fingerprint.
    monkeypatch.setattr(
        capture, "_code_identity", lambda: (_ for _ in ()).throw(OSError("git unavailable"))
    )
    result = runner.invoke(app, ["record", "--week", "1", "--rb", "BBB RB1", *args])
    assert result.exit_code != 0

    conn = db.connect(path)
    picks = state.picks(conn, SEASON)
    assert list(picks.player_id) == ["AAA-RB1"], "the replacement must not have landed"
    log = capture.events(conn, SEASON, 1)
    assert list(zip(log.kind, log.player_id, strict=True)) == [("submitted", "AAA-RB1")]


def test_removing_a_pick_and_recording_the_correction_are_one_write(tmp_path, monkeypatch):
    path = tmp_path / "pool.db"
    _seed(db.connect(path)).close()
    runner = CliRunner()
    args = ["--season", str(SEASON), "--db", str(path)]
    assert runner.invoke(app, ["record", "--week", "1", "--rb", "AAA RB1", *args]).exit_code == 0
    monkeypatch.setattr(
        capture, "_code_identity", lambda: (_ for _ in ()).throw(OSError("git unavailable"))
    )
    assert runner.invoke(app, ["unrecord", "1", "RB", *args]).exit_code != 0

    conn = db.connect(path)
    assert list(state.picks(conn, SEASON).player_id) == ["AAA-RB1"]
    assert set(capture.events(conn, SEASON, 1).kind) == {"submitted"}


def test_two_decisions_under_different_settings_get_different_identities(tmp_path):
    """The identity cache keyed on model and calibrator while caching the constants,
    so two decisions made under different premiums recorded one premium between them --
    and it was the first one's, whichever decision you asked about."""
    conn, unplayed = _staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    _publish(conn, unplayed, "thursday")
    before_kickoff = datetime.fromisoformat("2024-09-19T18:30:00+00:00")

    identities, holds = {}, {}
    for premium in (0.10, 0.0001):
        with config.override(INFO_PREMIUM_TD=premium):
            decision_id, _, advice = _decide(conn, 3, before_kickoff)
        identities[premium] = capture.recorded_identity(conn, decision_id)
        holds[premium] = {a.slot: a.hold for a in advice}

    # The cheapest later alternative costs 0.00036 TDs, so these straddle it.
    assert holds[0.10] != holds[0.0001], "fixture must actually flip a hold decision"
    assert identities[0.10]["constants"]["INFO_PREMIUM_TD"] == 0.10
    assert identities[0.0001]["constants"]["INFO_PREMIUM_TD"] == 0.0001
    assert identities[0.10] != identities[0.0001]


def test_a_captured_hold_stays_a_hold_when_the_premium_moves(tmp_path):
    """Reconstruction re-derives advice, and `advise_slot` reads the premium when it is
    called. Under today's configuration a captured hold came back as a commit from
    byte-identical stored data."""
    conn, unplayed = _staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    _publish(conn, unplayed, "thursday")
    before_kickoff = datetime.fromisoformat("2024-09-19T18:30:00+00:00")
    now = state.eastern_now(before_kickoff)
    with config.override(INFO_PREMIUM_TD=0.10):
        decision_id, _, advice = _decide(conn, 3, before_kickoff)
    captured = {a.slot: a.hold for a in advice}
    assert any(captured.values()), "fixture must capture a hold"

    with config.override(INFO_PREMIUM_TD=0.0001):
        # Under the new premium this decision would commit; the recorded one held.
        live = advise_week(
            P.projections_for(conn, SEASON, from_week=3),
            3,
            set(),
            {slot: {} for slot in config.SLOTS},
            now=now,
        )
        assert not any(a.hold for a in live)
        _, redone, _, drift = capture.reconstruct(conn, decision_id)
    assert {a.slot: a.hold for a in redone} == captured
    assert drift["constants_changed"]["INFO_PREMIUM_TD"]["recorded"] == 0.10


def test_reconstruction_refuses_a_source_tree_it_was_not_captured_under(tmp_path):
    """The recommender's behaviour is not carried by the recorded constants alone, so a
    silent substitution of current code would answer a different question."""
    conn, unplayed = _staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    _publish(conn, unplayed, "thursday")
    decision_id, _, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))

    capture._code_identity.cache_clear()
    try:
        with pytest.raises(ValueError, match="captured under source fingerprint"):
            with _pretend_source_moved():
                capture.reconstruct(conn, decision_id)
        with _pretend_source_moved():
            _, _, _, drift = capture.reconstruct(conn, decision_id, allow_code_drift=True)
        assert drift["code_hash_changed"]
        assert drift["current_code_hash"] == "moved"
    finally:
        capture._code_identity.cache_clear()


@contextlib.contextmanager
def _pretend_source_moved():
    original = capture._code_identity
    capture._code_identity = lambda: ("moved", "moved", False)
    try:
        yield
    finally:
        capture._code_identity = original


def test_a_setting_this_version_cannot_rebuild_is_refused_not_guessed(tmp_path):
    """`DEPTH_MULT` keys come back from the log as strings. Installing that form would
    demote a backup quarterback from 0.15 to 0.05 -- a worse reconstruction than none."""
    recorded = capture._canonical(capture.current_constants())
    recorded["DEPTH_MULT"] = {"QB": {"1": 1.0, "2": 0.9}}
    with pytest.raises(ValueError, match="DEPTH_MULT"):
        with capture.recorded_settings(recorded):
            pass
    # A scalar that moved is restored rather than refused.
    recorded = capture._canonical(capture.current_constants())
    recorded["INFO_PREMIUM_TD"] = 0.42
    with capture.recorded_settings(recorded):
        assert config.INFO_PREMIUM_TD == 0.42
    assert config.INFO_PREMIUM_TD != 0.42


@pytest.mark.parametrize("table", ["decision_events", "decision_inputs"])
def test_captured_events_cannot_be_edited_or_deleted(tmp_path, table):
    conn, unplayed = _staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    _publish(conn, unplayed, "thursday")
    _decide(conn, 3, datetime.fromisoformat(DECISION))
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        with conn:
            conn.execute(f"UPDATE {table} SET season = 1999")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        with conn:
            conn.execute(f"DELETE FROM {table}")


def test_repeating_a_decision_on_unchanged_inputs_stores_one_surface(tmp_path):
    """Content-addressed like the feed archive: re-running `recommend` should cost an
    event, not another copy of the whole surface."""
    conn, unplayed = _staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    _publish(conn, unplayed, "thursday")
    first, _, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))
    second, _, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))

    assert first != second
    hashes = {
        r[0]
        for r in conn.execute("SELECT surface_hash FROM decision_events WHERE kind = 'surface'")
    }
    assert len(hashes) == 1
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM input_payloads WHERE codec = ?", (capture.CODEC,)
        ).fetchone()[0]
        == 1
    )


def test_outcomes_are_joined_separately_and_absence_is_not_a_zero(tmp_path):
    conn, unplayed = _staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    _publish(conn, unplayed, "thursday")
    _decide(conn, 3, datetime.fromisoformat(DECISION))

    joined = capture.outcomes(conn, SEASON)
    assert "actual_tds" in joined and "outcome_complete" in joined
    # Only the Thursday game has been played, so week 3 is not fully scored and week 4
    # has not started; neither may be read as a zero.
    assert not joined.outcome_complete.any()
    assert joined.actual_tds.isna().all()

    _publish(conn, unplayed, "rest")
    complete = capture.outcomes(conn, SEASON)
    assert complete.outcome_complete.all()
    assert complete.actual_tds.notna().all()


def test_the_cli_records_submissions_and_corrections_without_editing_history(tmp_path):
    """`my_picks` holds the current answer and is overwritten in place. The capture log
    has to hold every answer, or a corrected pick erases the one it replaced."""
    path = tmp_path / "pool.db"
    conn = _seed(db.connect(path))
    conn.close()
    runner = CliRunner()
    for args in (
        ["record", "--week", "1", "--rb", "AAA RB1", "--season", str(SEASON), "--db", str(path)],
        ["record", "--week", "1", "--rb", "BBB RB1", "--season", str(SEASON), "--db", str(path)],
        ["unrecord", "1", "RB", "--season", str(SEASON), "--db", str(path)],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, (result.output, result.exception)

    conn = db.connect(path)
    log = capture.events(conn, SEASON, 1)
    assert list(zip(log.kind, log.player_id, strict=True)) == [
        ("submitted", "AAA-RB1"),
        ("correction", "AAA-RB1"),
        ("submitted", "BBB-RB1"),
        ("correction", "BBB-RB1"),
    ]
    assert conn.execute("SELECT COUNT(*) FROM my_picks").fetchone()[0] == 0


def test_the_cli_captures_a_recommendation_and_can_be_asked_not_to(tmp_path):
    path = tmp_path / "pool.db"
    conn = _seed(db.connect(path))
    conn.close()
    runner = CliRunner()
    args = ["recommend", "--week", "1", "--season", str(SEASON), "--db", str(path)]
    assert runner.invoke(app, args).exit_code == 0
    conn = db.connect(path)
    assert len(capture.events(conn, SEASON, 1)) == 4  # one surface, three slots
    conn.close()

    assert runner.invoke(app, [*args, "--no-capture"]).exit_code == 0
    conn = db.connect(path)
    assert len(capture.events(conn, SEASON, 1)) == 4


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
    conn = _seed(db.connect(tmp_path / "diag-source.db"))
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
        len(set(diagnostics._cluster(current))) == current.groupby(["player_id", "season"]).ngroups
    )
    assert (
        len(set(diagnostics._cluster(future)))
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
    assert (table.excluded_from_fit == table.zero_lam_n).all()

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
        app, ["diagnose", "--run", str(run), "--out", str(tmp_path / "x"), "--model", "no-such"]
    )
    assert result.exit_code == 1
    assert "No saved surface" in result.output
