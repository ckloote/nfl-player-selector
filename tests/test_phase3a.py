"""Phase 3A: evidence, capture and readiness repairs.

Each test states the failure it prevents. The guards here exist because the
unguarded fit returned plausible numbers for populations that identify nothing,
and a plausible number survives into a published table.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import optimize

from pool import config
from pool import evaluate as ev

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
