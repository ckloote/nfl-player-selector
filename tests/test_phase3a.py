"""Phase 3A: evidence, capture and readiness repairs.

Each test states the failure it prevents. The guards here exist because the
unguarded fit returned plausible numbers for populations that identify nothing,
and a plausible number survives into a published table.
"""

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

from pool import benchmark, config, db, snapshots
from pool import evaluate as ev
from pool import projections as P
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
    monkeypatch.setattr(benchmark, "code_identity", lambda: dict(code_hash="t", revision="t", dirty=False))
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
    monkeypatch.setattr(benchmark, "code_identity", lambda: dict(code_hash="t", revision="t", dirty=False))
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
    monkeypatch.setattr(benchmark, "code_identity", lambda: dict(code_hash="t", revision="t", dirty=False))
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
