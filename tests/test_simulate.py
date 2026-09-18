"""Joint outcomes: shared credit where it is real, a shared game where it is real, and
every player's expectation left exactly where the shipped forecast put it."""

import numpy as np
import pandas as pd
import pytest

from pool import simulate
from tests.conftest import proj_row
from tests.test_rivals import _imports

SIMS = 120_000


def cell(pid, slot, lam, team, game, position=None, week=1):
    row = proj_row(pid, pid.upper(), slot, week, lam, team=team, position=position)
    return row | {"game_id": game}


@pytest.fixture
def frame():
    """One game with two teams on it, plus an unrelated game elsewhere."""
    return pd.DataFrame(
        [
            cell("qa", "QB", 1.0, "A", "g1"),
            cell("wa", "FLEX", 0.6, "A", "g1"),
            cell("ta", "FLEX", 0.4, "A", "g1", position="TE"),
            cell("ra", "RB", 0.5, "A", "g1"),
            cell("qb", "QB", 0.8, "B", "g1"),
            cell("wb", "FLEX", 0.4, "B", "g1"),
            cell("qc", "QB", 0.7, "C", "g2"),
            cell("wc", "FLEX", 0.6, "C", "g2"),
        ]
    )


@pytest.fixture
def drawn(frame):
    return frame, simulate.sample(frame, [1], sims=SIMS, seed=11)


def _series(frame, draws, pid):
    return draws.values[draws.index[(1, pid)]]


def test_every_expectation_returns_the_shipped_rate(drawn):
    """The identity the residual term exists to preserve. A simulator that moved the point
    forecast would be a calibration change, which Phase 3B's outcome forbids -- this adds a
    joint distribution around the shipped rates and nothing else."""
    frame, draws = drawn
    for row in frame.itertuples():
        mean = _series(frame, draws, row.player_id).mean()
        assert mean == pytest.approx(row.lam, rel=0.03), f"{row.player_id}: {mean} vs {row.lam}"


def test_a_quarterback_moves_with_his_own_receivers_and_barely_with_his_back(drawn):
    """+0.465 against +0.013 in 15 seasons. The first is shared credit -- one passing
    touchdown pays the passer and the catcher -- and the second is nearly nothing."""
    frame, draws = drawn
    qa = _series(frame, draws, "qa")
    with_receiver = np.corrcoef(qa, _series(frame, draws, "wa"))[0, 1]
    with_tight_end = np.corrcoef(qa, _series(frame, draws, "ta"))[0, 1]
    with_back = np.corrcoef(qa, _series(frame, draws, "ra"))[0, 1]
    assert with_receiver > 0.2
    assert with_tight_end > 0.2
    assert with_back < with_receiver / 2


def test_opposing_teams_in_one_game_move_together_and_separate_games_do_not(drawn):
    """The shared game factor, which the data supports, and the team factor, which it
    does not: a running back and a receiver on one team are linked only by the game."""
    frame, draws = drawn
    same_game = np.corrcoef(_series(frame, draws, "qa"), _series(frame, draws, "qb"))[0, 1]
    other_game = np.corrcoef(_series(frame, draws, "qa"), _series(frame, draws, "qc"))[0, 1]
    assert same_game > 0.01
    assert abs(other_game) < 0.01
    assert same_game > other_game


def test_outcomes_carry_the_fitted_spread_rather_than_poisson(drawn):
    """Poisson has var == mean. Mixing over the fitted game factor adds `lam^2 / k_game`,
    and the calibration run shows that reproduces the observed player-level spread to
    within three percent. Asserted against the model's own formula, not a round number,
    so re-fitting `k_game` moves the expectation rather than breaking the test."""
    frame, draws = drawn
    for pid, lam in (("wa", 0.6), ("ta", 0.4), ("ra", 0.5)):
        values = _series(frame, draws, pid)
        assert values.var() == pytest.approx(lam + lam**2 / simulate.K_GAME, rel=0.06)
        assert values.var() > values.mean()


def test_one_player_week_is_one_row_shared_by_everyone_who_picked_him(drawn):
    """Week 1 already produced the case: two entrants both held Joe Burrow."""
    frame, draws = drawn
    brad = draws.totals([(1, "qa"), (1, "wa")])
    keith = draws.totals([(1, "qa"), (1, "wb")])
    assert np.array_equal(brad - _series(frame, draws, "wa"), keith - _series(frame, draws, "wb"))


def test_an_unknown_pick_contributes_nothing_rather_than_failing(drawn):
    _, draws = drawn
    assert np.array_equal(draws.totals([(1, "nobody")]), np.zeros(SIMS, dtype=np.int32))
    assert np.array_equal(draws.totals([]), np.zeros(SIMS, dtype=np.int32))


def test_receivers_that_out_project_their_quarterback_rescale_rather_than_clamp(frame):
    """Scaling receiving down without scaling the rest up would silently under-credit the
    highest-scoring offences. The guard preserves both expectations and is counted."""
    thin = pd.DataFrame([cell("qa", "QB", 0.1, "A", "g1"), cell("wa", "FLEX", 2.0, "A", "g1")])
    draws = simulate.sample(thin, [1], sims=SIMS, seed=5)
    assert draws.rescaled == 1
    assert draws.values[draws.index[(1, "wa")]].mean() == pytest.approx(2.0, rel=0.03)
    assert draws.values[draws.index[(1, "qa")]].mean() == pytest.approx(0.1, rel=0.08)


def test_a_team_with_no_quarterback_in_the_pool_still_scores_its_own_rate():
    orphan = pd.DataFrame([cell("wa", "FLEX", 0.6, "A", "g1")])
    draws = simulate.sample(orphan, [1], sims=SIMS, seed=3)
    assert draws.values[draws.index[(1, "wa")]].mean() == pytest.approx(0.6, rel=0.03)
    assert draws.rescaled == 0


def test_a_backup_quarterback_does_not_re_credit_his_starter_s_receivers(frame):
    """One team's receiving touchdowns were thrown once. Crediting two passers for them
    would invent touchdowns that never happened."""
    two = pd.concat([frame, pd.DataFrame([cell("qa2", "QB", 0.15, "A", "g1")])], ignore_index=True)
    draws = simulate.sample(two, [1], sims=SIMS, seed=11)
    assert draws.values[draws.index[(1, "qa2")]].mean() == pytest.approx(0.15, rel=0.06)
    assert draws.values[draws.index[(1, "qa")]].mean() == pytest.approx(1.0, rel=0.03)


@pytest.mark.parametrize(
    "mine,rivals,expected",
    [
        (6, [5, 4], 1.0),
        (5, [5, 4], 0.5),
        (5, [5, 5], 1 / 3),
        (4, [5, 4], 0.0),
        (0, [0, 0], 1 / 3),
    ],
)
def test_a_tie_for_first_splits_the_pot(mine, rivals, expected):
    """Season totals are small integers, so ties are common and a rule that ignored them
    would undervalue every position that reliably draws level."""
    got = simulate.shares(np.array([mine]), np.array([rivals]))
    assert got[0] == pytest.approx(expected)


def test_shares_over_all_entrants_sum_to_one_in_every_draw():
    rng = np.random.default_rng(0)
    totals = rng.integers(0, 6, size=(500, 4))
    each = [
        simulate.shares(totals[:, i], np.delete(totals, i, axis=1)) for i in range(totals.shape[1])
    ]
    assert np.allclose(np.sum(each, axis=0), 1.0)


def test_with_no_rivals_the_pot_is_mine():
    assert np.array_equal(simulate.shares(np.array([0, 3]), np.empty((2, 0))), np.ones(2))


def test_the_same_seed_gives_the_same_season(frame):
    """A non-deterministic policy cannot be reconstructed, and `capture.reconstruct` would
    have nothing to restore."""
    a = simulate.sample(frame, [1], sims=500, seed=99)
    b = simulate.sample(frame, [1], sims=500, seed=99)
    c = simulate.sample(frame, [1], sims=500, seed=100)
    assert np.array_equal(a.values, b.values)
    assert not np.array_equal(a.values, c.values)


def test_pairing_beats_independent_draws_on_the_difference(frame):
    """Shared draws leave the rivals' maximum identical in both arms, so the two candidate
    shares move together and their difference is estimated far more precisely than either.
    Both candidates sit in a game neither rival touches, so the only thing linking them is
    the rivals' maximum -- which is exactly the mechanism being asserted."""
    contest = pd.concat(
        [
            frame,
            pd.DataFrame([cell("wd", "FLEX", 0.9, "D", "g3"), cell("we", "FLEX", 0.8, "D", "g3")]),
        ],
        ignore_index=True,
    )
    draws = simulate.sample(contest, [1], sims=40_000, seed=7)
    rivals = np.column_stack([draws.totals([(1, "qa")]), draws.totals([(1, "qc")])])
    first = simulate.shares(draws.totals([(1, "wd")]), rivals)
    second = simulate.shares(draws.totals([(1, "we")]), rivals)
    _, paired_se = simulate.paired(first, second)
    independent = np.sqrt(first.var(ddof=1) / len(first) + second.var(ddof=1) / len(second))
    assert paired_se < independent, f"{paired_se} !< {independent}"
    assert np.corrcoef(first, second)[0, 1] > 0.3, "the shared rival max is what pairs them"


def test_weeks_outside_the_horizon_are_not_sampled(frame):
    later = frame.assign(week=2)
    draws = simulate.sample(pd.concat([frame, later], ignore_index=True), [1], sims=50, seed=1)
    assert all(week == 1 for week, _ in draws.index)


def test_simulate_reads_nothing_from_the_world():
    assert not _imports(simulate) & {
        "sqlite3",
        "db",
        "entrants",
        "standings",
        "predictions",
        "os",
        "pathlib",
        "io",
    }
