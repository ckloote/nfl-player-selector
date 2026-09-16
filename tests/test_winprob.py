"""The win-probability view: a one-step lookahead that sits beside the expected-TD advice
and never edits it."""

import copy
from datetime import UTC, datetime

import pandas as pd
import pytest

from pool import benchmark, config, rivals, simulate
from pool.recommend import advise_slot, advise_week

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

# Before the fixture's kickoffs, so nothing is deadline-blocked out of the candidate pool.
NOW = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)


def cell(pid, slot, lam, team, game, position=None, week=1):
    from tests.conftest import proj_row

    row = proj_row(pid, pid.upper(), slot, week, lam, team=team, position=position)
    return row | {"game_id": game}


def frame(qb_a=1.00, qb_b=0.99, weeks=(1,)):
    """Four quarterbacks in unrelated games, plus a filler back and receiver.

    `qa` edges `qb`, so both the solver and a greedy rival take `qa` -- which makes that pair
    a clean test of differentiation: near-identical in expected touchdowns, differing only in
    who else is likely to be holding them. `qz` and `qy` are exactly equal and both beneath
    what any rival would take, so they are exchangeable and nothing should order them.
    """
    rows = []
    for week in weeks:
        rows += [
            cell("qa", "QB", qb_a, "A", f"g{week}a", week=week),
            cell("qb", "QB", qb_b, "B", f"g{week}b", week=week),
            cell("qz", "QB", 0.90, "G", f"g{week}g", week=week),
            cell("qy", "QB", 0.90, "H", f"g{week}h", week=week),
            cell("q5", "QB", 0.80, "I", f"g{week}i", week=week),
            cell("q6", "QB", 0.70, "J", f"g{week}j", week=week),
            cell("ra", "RB", 0.5, "C", f"g{week}c", week=week),
            cell("rb", "RB", 0.4, "D", f"g{week}d", week=week),
            cell("fa", "FLEX", 0.5, "E", f"g{week}e", week=week),
            cell("fb", "FLEX", 0.4, "F", f"g{week}f", week=week),
        ]
    return pd.DataFrame(rows)


def rival(name, tds, used=()):
    return rivals.RivalState(name, name.title(), frozenset(used), 0, tds)


def shares_for(proj, pool, week=1, sims=6000):
    with config.override(WINPROB_SIMS=sims, RIVAL_NOISE_TOP_N=1):
        advice = advise_week(proj, week, set(), {}, now=NOW, pool=pool)
    return {a.slot: {s.player_id: s for s in a.shares} for a in advice}, advice


# --- the acceptance test: the second objective never edits the first ---------------


def test_advice_without_pool_state_is_exactly_what_it_always_was():
    """What fails if the win-probability work ever starts quietly changing the EV advice."""
    proj = frame()
    plain = advise_slot(proj, "QB", 1, set(), {}, now=NOW)
    withstate = advise_week(
        proj, 1, set(), {}, now=NOW, pool=rivals.PoolState((rival("x", 0),), 0)
    )
    qb = next(a for a in withstate if a.slot == "QB")
    assert qb.recommended.player_id == plain.recommended.player_id
    assert [c.player_id for c in qb.alternatives] == [c.player_id for c in plain.alternatives]
    assert (qb.hold, qb.recommended.cost, qb.recommended.lam) == (
        plain.hold, plain.recommended.cost, plain.recommended.lam,
    )


def test_an_empty_pool_produces_no_second_view_rather_than_failing():
    proj = frame()
    for pool in (None, rivals.PoolState()):
        advice = advise_week(proj, 1, set(), {}, now=NOW, pool=pool)
        assert all(not a.shares for a in advice)
        assert all(not a.divergent for a in advice)


# --- what the policy is supposed to do ---------------------------------------------


def test_level_standings_still_favour_differentiation_when_rivals_converge():
    """Corrects a claim this repo made before it was measured.

    `docs/DESIGN.md` says win probability and expected touchdowns agree early, when everyone
    is level. That is only true when my candidates are equally unrelated to what rivals hold.
    When every rival is about to take the same player I would take, mirroring them buys a
    guaranteed four-way split and differentiating buys a chance of the whole pot -- 0.25
    against 0.44 here, which is not a rounding difference. Level standings do not by
    themselves make the objectives agree.
    """
    proj = frame()
    pool = rivals.PoolState((rival("one", 0), rival("two", 0), rival("three", 0)), 0)
    shares, advice = shares_for(proj, pool)
    qb_advice = next(a for a in advice if a.slot == "QB")
    assert qb_advice.recommended.player_id == "qa", "expected touchdowns still says qa"
    assert shares["QB"]["qa"].share == pytest.approx(0.25, abs=0.01), "a guaranteed 4-way tie"
    assert shares["QB"]["qb"].share > shares["QB"]["qa"].share
    assert qb_advice.divergent, "and the tool must say the two disagree"


def test_the_objectives_agree_when_neither_candidate_is_one_a_rival_would_hold():
    """The condition under which the two really do converge: my choice does not interact
    with theirs, so the higher rate simply wins."""
    proj = frame()
    # Their whole quarterback pool is disjoint from mine: they are certain to take `q5`,
    # which is not a candidate I am weighing. Nothing less than this really decouples us --
    # in a pool where everyone picks off one board, my choice usually does touch theirs.
    spent = ("qa", "qb", "qz", "qy")
    pool = rivals.PoolState((rival("one", 0, spent), rival("two", 0, spent)), 0)
    with config.override(WINPROB_SIMS=6000, RIVAL_NOISE_TOP_N=1):
        advice = advise_week(proj, 1, set(), {}, now=NOW, pool=pool)
    qb_advice = next(a for a in advice if a.slot == "QB")
    # Stated as "no preference beyond noise" rather than "the same name", because the two
    # survivors here have identical rates: which one a tie-break lands on is not a finding.
    assert not qb_advice.divergent
    assert qb_advice.best_share.tied


def test_behind_against_one_leader_the_policy_refuses_the_leader_s_likely_pick():
    """Mirroring the leader freezes the gap, and a frozen gap is a loss."""
    proj = frame()
    pool = rivals.PoolState((rival("leader", 3),), 0)
    shares, advice = shares_for(proj, pool)
    qa, qb = shares["QB"]["qa"], shares["QB"]["qb"]
    assert qb.share > qa.share, f"differentiate: {qb.share:.4f} vs {qa.share:.4f}"
    assert next(a for a in advice if a.slot == "QB").recommended.player_id == "qa"


def test_ahead_against_one_chaser_the_preference_reverses():
    """Sharing their player matches their gains and runs out the clock."""
    proj = frame()
    pool = rivals.PoolState((rival("chaser", 0),), 3)
    shares, _ = shares_for(proj, pool)
    qa, qb = shares["QB"]["qa"], shares["QB"]["qb"]
    assert qa.share > qb.share, f"mirror: {qa.share:.4f} vs {qb.share:.4f}"


def test_the_sign_flips_at_level_and_not_somewhere_else():
    """The one internal-consistency check that says the arithmetic is right."""
    proj = frame()
    behind = shares_for(proj, rivals.PoolState((rival("r", 3),), 0))[0]["QB"]
    ahead = shares_for(proj, rivals.PoolState((rival("r", 0),), 3))[0]["QB"]
    assert (behind["qb"].share - behind["qa"].share) > 0
    assert (ahead["qb"].share - ahead["qa"].share) < 0


def test_mirroring_covers_one_threat_and_not_several():
    """Sharing a player matches that opponent's gains and nobody else's, so the case for it
    weakens as soon as more than one rival is live. Here the second rival has already spent
    `qa`, so mirroring it does nothing about them."""
    proj = frame()
    alone = shares_for(proj, rivals.PoolState((rival("r", 0),), 3))[0]["QB"]
    plus_other = shares_for(
        proj, rivals.PoolState((rival("r", 0), rival("s", 0, ("qa",))), 3)
    )[0]["QB"]
    assert (alone["qa"].share - alone["qb"].share) > (
        plus_other["qa"].share - plus_other["qb"].share
    )


# --- the guards ---------------------------------------------------------------------


def test_two_exchangeable_candidates_are_not_ordered():
    """`qz` and `qy` have identical rates, sit in unrelated games and are beneath anything a
    rival would take, so nothing distinguishes them. Without the significance guard the tool
    would still rank one above the other, every week, on Monte Carlo noise."""
    proj = frame()
    pool = rivals.PoolState((rival("r", 0), rival("s", 0)), 0)
    shares, _ = shares_for(proj, pool, sims=4000)
    qz, qy = shares["QB"]["qz"], shares["QB"]["qy"]
    spread = (qz.se**2 + qy.se**2) ** 0.5
    assert abs(qz.share - qy.share) <= config.WINPROB_SIGNIFICANCE * spread


def test_the_recommended_pick_is_its_own_reference_and_ties_itself():
    proj = frame()
    shares, advice = shares_for(proj, rivals.PoolState((rival("r", 1),), 0))
    qb = next(a for a in advice if a.slot == "QB")
    own = shares["QB"][qb.recommended.player_id]
    assert own.delta == 0.0 and own.tied


def test_the_future_plan_excludes_the_player_this_week_spends():
    """Spending a player now changes the rest of the season, and the share must be scored
    against the season that actually follows from the pick."""
    proj = frame(weeks=(1, 2))
    advice = advise_slot(proj, "QB", 1, set(), {}, now=NOW)
    for candidate in [advice.recommended, *advice.alternatives]:
        rows = {r for w, r in candidate.future.items() if w != 1}
        assert candidate.future[1] not in rows, "a player cannot be used in two weeks"


def test_the_same_seed_gives_the_same_advice_twice():
    proj = frame()
    pool = rivals.PoolState((rival("r", 2),), 0)
    first = shares_for(copy.deepcopy(proj), pool)[0]["QB"]
    second = shares_for(copy.deepcopy(proj), pool)[0]["QB"]
    assert {k: v.share for k, v in first.items()} == {k: v.share for k, v in second.items()}


# --- the closure ---------------------------------------------------------------------


def test_the_closure_gained_exactly_the_two_deciding_modules():
    modules = benchmark.decision_modules()
    assert "src/pool/rivals.py" in modules
    assert "src/pool/simulate.py" in modules
    for outside in ("predictions", "standings", "entrants", "cli"):
        assert f"src/pool/{outside}.py" not in modules


def test_a_pot_share_is_a_share_not_a_win_probability():
    """A tie at the top splits the pot, so the quantity maximised counts a k-way tie as
    1/k. Asserted here because the distinction is the whole objective."""
    assert simulate.shares(pd.array([5]).to_numpy(), pd.DataFrame([[5, 5]]).to_numpy())[0] == 1 / 3
