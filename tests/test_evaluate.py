"""Tests for the player-week forecast evaluation layer."""

import numpy as np
import pandas as pd
import pytest

from pool import evaluate as ev
from pool import models
from pool import projections as P
from tests.test_backtest import SEASON, WEEKS


# --- the forecast set -------------------------------------------------------
def test_forecast_set_keeps_only_the_week_being_forecast(seeded):
    """A frozen frame carries every remaining week, but only week W is a
    forecast anyone acts on — the rest will be rebuilt before they are picked."""
    df = ev.forecasts(seeded, SEASON, model="shipped", weeks=WEEKS)
    assert list(df.columns) == ev.FORECAST_COLUMNS
    assert sorted(df.week.unique()) == WEEKS
    per_week = df.groupby("week").size()
    assert (per_week == per_week.iloc[0]).all()


def test_actuals_and_played_are_distinct_facts(seeded):
    """Scoring zero and never taking the field look identical in the total and
    have opposite implications for the model."""
    df = ev.forecasts(seeded, SEASON, model="shipped", weeks=WEEKS)
    assert df.played.all()  # every seeded player has a stat line
    starters = df[df.player_id.str.endswith("1")]
    assert starters.actual_tds.gt(0).any()
    assert df[df.player_id.str.endswith("2")].actual_tds.eq(0).all()


def test_rank_available_excludes_players_already_spent(seeded):
    """The pool may use each player once, so by week 2 the top of the ranking
    is gone. rank_in_slot ignores that; rank_available is the real choice set."""
    df = ev.forecasts(seeded, SEASON, model="shipped", weeks=WEEKS)
    first = df[(df.week == WEEKS[0]) & (df.rank_available == 1)]
    spent = set(first.player_id)
    later = df[df.week > WEEKS[0]]
    assert later[later.player_id.isin(spent)].rank_available.isna().all()
    # Everyone still available is ranked, contiguously from 1.
    for (_, _), g in later.groupby(["week", "slot"]):
        ranks = g.rank_available.dropna().sort_values().to_numpy()
        assert (ranks == np.arange(1, len(ranks) + 1)).all()


def test_a_common_pool_is_the_one_every_model_faces(seeded):
    """A model that spends its players badly leaves better ones behind, which
    inflates its own later top-k — a mechanical reward for being worse. Passing
    one depleted pool is what removes that; this checks it is actually used."""
    spent = {w: set() for w in WEEKS}
    spent[WEEKS[-1]] = {"AAA-QB1", "AAA-RB1"}
    df = ev.forecasts(seeded, SEASON, model="shipped", weeks=WEEKS, available_from=spent)
    last = df[df.week == WEEKS[-1]]
    assert last[last.player_id.isin(spent[WEEKS[-1]])].rank_available.isna().all()
    assert last[~last.player_id.isin(spent[WEEKS[-1]])].rank_available.notna().all()
    # Earlier weeks were given an empty spent set, so nobody is excluded there.
    first = df[df.week == WEEKS[0]]
    assert first.rank_available.notna().all()


def test_the_forecast_set_is_reproducible(seeded):
    """A benchmark that does not repeat cannot support a holdout claim."""
    a = ev.forecasts(seeded, SEASON, model="shipped", weeks=WEEKS)
    b = ev.forecasts(seeded, SEASON, model="shipped", weeks=WEEKS)
    pd.testing.assert_frame_equal(a, b)


# --- calibration ------------------------------------------------------------
def test_poisson_glm_recovers_a_known_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(-1.0, 0.8, 40000)
    y = rng.poisson(np.exp(0.2 + 0.75 * x))
    fit = ev.poisson_glm(y, x, np.arange(len(y)))
    assert fit["slope"] == pytest.approx(0.75, abs=0.03)
    assert fit["intercept"] == pytest.approx(0.2, abs=0.03)


def test_clustering_widens_the_interval():
    """A player's error repeats across every week of their season, so the rows
    carry far less information than their count suggests. Naive errors would
    make every comparison significant."""
    rng = np.random.default_rng(1)
    n_g, per = 200, 100
    g = np.repeat(np.arange(n_g), per)
    x = np.repeat(rng.normal(-1.0, 0.8, n_g), per) + rng.normal(0, 0.05, n_g * per)
    y = rng.poisson(np.exp(0.2 + 0.75 * x + np.repeat(rng.normal(0, 0.5, n_g), per)))
    clustered = ev.poisson_glm(y, x, g)["se_slope"]
    naive = ev.poisson_glm(y, x, np.arange(len(y)))["se_slope"]
    assert clustered > 3 * naive


def test_a_perfect_forecast_has_slope_one():
    rng = np.random.default_rng(2)
    lam = rng.uniform(0.05, 2.0, 30000)
    y = rng.poisson(lam)
    fit = ev.poisson_glm(y, np.log(lam), np.arange(len(y)))
    assert fit["slope_lo"] <= 1.0 <= fit["slope_hi"]


def test_poisson_deviance_is_minimised_by_the_truth():
    rng = np.random.default_rng(3)
    lam = rng.uniform(0.1, 1.5, 20000)
    y = rng.poisson(lam)
    truth = ev.poisson_deviance(y, lam)
    assert truth < ev.poisson_deviance(y, lam * 1.4)
    assert truth < ev.poisson_deviance(y, lam * 0.7)


def test_the_calibration_correction_cannot_reorder_anybody(seeded):
    """lambda' = exp(a) * lambda**b is monotone, so it moves the level and never
    the ranking. Every pick, and therefore every season total, is unchanged —
    which is exactly why it is a layer-A fix and not a pool improvement."""
    df = ev.forecasts(seeded, SEASON, model="shipped", weeks=WEEKS)
    fixed = df.copy()
    fixed["lam"] = np.exp(-0.14) * fixed.lam.to_numpy(dtype=float) ** 0.89
    key = ["season", "week", "slot"]
    before = df.sort_values([*key, "lam", "player_id"], ascending=[True, True, True, False, True])
    after = fixed.sort_values([*key, "lam", "player_id"], ascending=[True, True, True, False, True])
    assert list(before.player_id) == list(after.player_id)


def test_reliability_splits_a_pooled_ratio_that_hides_the_tilt():
    """The whole point of the bin table: mean projected can equal mean actual
    while the bottom is under-projected and the top over-projected."""
    df = pd.DataFrame(
        {
            "model": "m",
            "lam": [0.05] * 100 + [1.0] * 100,
            "actual_tds": [0.15] * 100 + [0.9] * 100,
            "played": True,
        }
    )
    assert df.lam.mean() == pytest.approx(df.actual_tds.mean(), abs=0.02)
    rel = ev.reliability(df)
    ratios = rel.ratio.dropna()
    assert ratios.min() < 0.5 and ratios.max() > 1.0


# --- models -----------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(models.BUILDERS))
def test_every_model_returns_the_frame_contract(seeded, name):
    """Downstream code — the optimizer, the harness, this module — depends only
    on the projection frame, so a model that breaks the contract breaks all of it."""
    builder = models.get(name)
    frames = P.load_frames(seeded, SEASON, as_of_week=2)
    build = P.build_projections if builder is None else builder
    out = build(frames, 2, role_source="usage")
    assert list(out.columns) == P.PROJECTION_COLUMNS
    assert np.isfinite(out.lam.to_numpy(dtype=float)).all()
    assert (out.lam >= 0).all()


def test_the_nulls_preserve_the_multiset_of_projections(seeded):
    """A shuffle must move information, not create or destroy it: same values,
    different rows. Anything else would make the null a different model."""
    frames = P.load_frames(seeded, SEASON, as_of_week=2)
    shipped = P.build_projections(frames, 2, role_source="usage")
    for name in ("random", "within-player"):
        out = models.get(name)(frames, 2, role_source="usage")
        assert sorted(out.lam.round(9)) == sorted(shipped.lam.round(9))


def test_within_player_shuffle_keeps_each_player_their_own_values(seeded):
    """Model 0b destroys only the week-to-week signal. If it also moved values
    between players it would be Model 0a with extra steps."""
    frames = P.load_frames(seeded, SEASON, as_of_week=1)
    shipped = P.build_projections(frames, 1, role_source="usage")
    out = models.get("within-player")(frames, 1, role_source="usage")
    for pid, g in out.groupby("player_id"):
        original = shipped[shipped.player_id == pid].lam
        assert sorted(g.lam.round(9)) == sorted(original.round(9))


def test_seeded_nulls_are_reproducible(seeded):
    frames = P.load_frames(seeded, SEASON, as_of_week=2)
    from pool.models.baselines import shuffle_within_slot_week

    a = shuffle_within_slot_week(seed=5)(frames, 2, role_source="usage")
    b = shuffle_within_slot_week(seed=5)(frames, 2, role_source="usage")
    c = shuffle_within_slot_week(seed=6)(frames, 2, role_source="usage")
    pd.testing.assert_frame_equal(a, b)

    # The frame is re-sorted by lambda, so the column itself is identical
    # across seeds by construction. What a different seed must change is which
    # player holds which value.
    def mapping(df):
        return df.set_index(["week", "player_id"]).lam.to_dict()

    assert mapping(a) == mapping(b)
    assert mapping(a) != mapping(c)
