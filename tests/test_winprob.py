"""The win-probability view: a one-step lookahead that sits beside the expected-TD advice
and never edits it."""

import copy
from datetime import UTC, datetime

import numpy as np
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

    `docs/DESIGN.md` said the two objectives agree early, when everyone is level, and
    `docs/PHASE4_PLAN.md` said it twice. Both are struck: it is only true when my candidates
    are equally unrelated to what rivals hold. When every rival is about to take the same
    player I would take, mirroring them buys a guaranteed four-way split and differentiating
    buys a chance of the whole pot -- 0.25 against 0.44 here, which is not a rounding
    difference. Level standings do not by themselves make the objectives agree, and this
    test is what keeps the corrected wording honest.
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


# --- correlating with my own quarterback, and when it is worth it ---------------------


def pairing_frame(weeks=(1,)):
    """My quarterback, a receiver who shares his credit, and one who shares nothing.

    `fa` is on `qa`'s team and in his game, so a passing touchdown can pay them both; `fb`
    is unrelated to everything I hold. They project identically, so expected touchdowns is
    exactly indifferent between them and the only thing left is correlation. The rival's
    three players project identically to mine and share no game with them, so the standings
    are the single asymmetry in the whole frame.
    """
    rows = []
    for week in weeks:
        rows += [
            cell("qa", "QB", 4.0, "A", f"g{week}a", week=week),
            cell("fa", "FLEX", 0.8, "A", f"g{week}a", week=week),
            cell("fb", "FLEX", 0.8, "C", f"g{week}c", week=week),
            cell("ra", "RB", 0.5, "D", f"g{week}d", week=week),
            cell("qz", "QB", 4.0, "E", f"g{week}e", week=week),
            cell("fz", "FLEX", 0.8, "F", f"g{week}f", week=week),
            cell("rz", "RB", 0.5, "G", f"g{week}g", week=week),
        ]
    return pd.DataFrame(rows)


def pairing(deficit, weeks=(1,), sims=20000):
    """FLEX shares with me `deficit` touchdowns behind one rival who holds nothing of mine."""
    mine, theirs = max(0, -deficit), max(0, deficit)
    state = rivals.RivalState("r1", "Rival", frozenset(("qa", "fa", "fb", "ra")), 0, theirs)
    pool = rivals.PoolState((state,), mine)
    with config.override(WINPROB_SIMS=sims, RIVAL_NOISE_TOP_N=1):
        advice = advise_week(pairing_frame(weeks), 1, set(), {}, now=NOW, pool=pool)
    flex = next(a for a in advice if a.slot == "FLEX")
    shares = {s.player_id: s for s in flex.shares}
    return shares["fa"], shares["fb"]


def test_behind_the_policy_correlates_with_its_own_quarterback():
    """Shared credit reaching the decision, not just the simulator.

    At equal expected touchdowns the only difference between these two receivers is that
    one is paid by the same passing touchdowns my quarterback is. Behind, that correlation
    is worth having: it fattens the tail I need to reach.
    """
    pair, split = pairing(+6)
    assert pair.share > split.share and not split.tied
    assert split.delta < 0, "the unrelated receiver is the worse pick when behind"


def test_ahead_the_policy_decorrelates_instead():
    pair, split = pairing(-3)
    assert split.share > pair.share and not split.tied


def test_level_sits_on_the_decorrelating_side_rather_than_at_the_pivot():
    """Corrects this document's claim 16b, which had the preference reversing around level.

    It does not. At level, with my total more variable than the rival's and both counts
    skewed, more variance costs me: a count distribution stretched to the right puts more
    mass below its own mean, so `P(mine > theirs)` falls. The pivot is strictly on the
    behind side of level, and level belongs with ahead.
    """
    pair, split = pairing(0)
    assert split.share > pair.share and not split.tied


def test_behind_is_measured_against_what_is_left_to_play_not_the_scoreboard():
    """The same deficit, and the opposite answer, because the schedule changed.

    Three touchdowns down with one week to play is behind. Three down with three weeks to
    play is not: there is room to win on the mean, and buying variance only widens a
    distribution that is already on the right side. A rule keyed to the scoreboard alone
    would get one of these two wrong.
    """
    near_pair, near_split = pairing(+3, weeks=(1,))
    assert near_pair.share > near_split.share, "one week left: behind enough to correlate"
    far_pair, far_split = pairing(+3, weeks=(1, 2, 3))
    assert far_split.share > far_pair.share, "three weeks left: the same deficit is not behind"


# --- the explanation describes the distribution the policy drew from -------------------


def test_the_sampling_set_shown_is_the_one_the_rollouts_walk():
    """`contested` is only worth printing if it names the same candidates a rollout could
    actually take. Two copies of that rule would be two chances to drift apart."""
    proj = frame(weeks=(1, 2))
    players, values = rivals.remaining_matrix(proj, "QB", 1, [1, 2], frozenset())
    rows = rivals.options(values, 0, top_n=3)
    taken = {
        rivals.rollout(players, values, [1, 2], rng=np.random.default_rng(seed), top_n=3)[1]
        for seed in range(60)
    }
    assert taken == {str(players.iloc[r].player_id) for r in rows}


def test_contested_names_the_rivals_who_may_take_each_candidate():
    proj = frame()
    pool = rivals.PoolState((rival("pat", 0), rival("jamie", 0, used=("qa",))), 0)
    with config.override(WINPROB_SIMS=200, RIVAL_NOISE_TOP_N=1):
        advice = advise_week(proj, 1, set(), {}, now=NOW, pool=pool)
    qb = next(a for a in advice if a.slot == "QB")
    # Pat's best remaining quarterback is `qa`; Jamie has spent him and falls to `qb`.
    assert qb.contested["qa"] == (("Pat", 1.0),)
    assert qb.contested["qb"] == (("Jamie", 1.0),)


def test_a_rival_whose_week_never_arrived_is_not_a_complete_pool():
    """Three more players are spent per missing week and none of them can be named, so the
    simulated rival is free to spend a player they have in fact already used. That is the
    same defect as an unresolved name, and it used to report itself as complete."""
    assert rivals.RivalState("a", "A", frozenset({"qa"}), 0, 3).pool_complete
    assert not rivals.RivalState("a", "A", frozenset({"qa"}), 1, 3).pool_complete
    gap = rivals.RivalState("a", "A", frozenset({"qa"}), 0, 3, missing_weeks=(2,))
    assert not gap.pool_complete


# --- a pick already on record is not a pick to predict ---------------------------------


def test_a_reported_pick_is_used_instead_of_a_prediction_of_it():
    """Claim 20. The rollout would otherwise guess at a week it has been told the answer to.

    `qz` is the best quarterback this rival has left, so greedy predicts him with certainty.
    Their report says they took `q6`, the worst. The simulated rival must score `q6`.
    """
    proj = frame()
    reported = rivals.RivalState(
        "r1", "Rival", frozenset(("qa", "qb")), 0, 0, known=(("QB", 1, "q6"),)
    )
    players, values = rivals.remaining_matrix(proj, "QB", 1, [1], reported.used_ids)
    guessed = rivals.rollout(players, values, [1], rng=np.random.default_rng(0), top_n=1)
    # `qz` and `qy` are exchangeable at 0.90; which of the two it lands on is a tie-break,
    # and the point is only that it reaches for the top of their list.
    assert guessed[1] in ("qz", "qy"), "left to itself the rollout takes one of the best"
    pinned = rivals.rollout(
        players, values, [1], rng=np.random.default_rng(0), top_n=1,
        known=reported.pinned("QB", [1]),
    )
    assert pinned[1] == "q6", "told the answer, it uses the answer"


def test_a_pinned_player_cannot_be_spent_again_later():
    """A reported pick depletes the pool it came from, or the rival gets him twice."""
    proj = frame(weeks=(1, 2))
    state = rivals.RivalState("r1", "Rival", frozenset(), 0, 0, known=(("QB", 1, "qz"),))
    players, values = rivals.remaining_matrix(proj, "QB", 1, [1, 2], state.used_ids)
    path = rivals.rollout(
        players, values, [1, 2], rng=np.random.default_rng(0), top_n=1,
        known=state.pinned("QB", [1, 2]),
    )
    assert path[1] == "qz" and path[2] != "qz"


def test_a_reported_pick_enters_the_explanation_at_certainty():
    """It is the strongest form of "this rival holds this player", so the divergence
    sentence has to read it as such rather than as one of three things they might do."""
    proj = frame()
    pool = rivals.PoolState(
        (rivals.RivalState("r1", "Rival", frozenset(), 0, 0, known=(("QB", 1, "qa"),)),), 0
    )
    with config.override(WINPROB_SIMS=200, RIVAL_NOISE_TOP_N=3):
        advice = advise_week(proj, 1, set(), {}, now=NOW, pool=pool)
    qb = next(a for a in advice if a.slot == "QB")
    assert qb.contested["qa"] == (("Rival", 1.0),)
    assert list(qb.contested) == ["qa"], "and nothing else is a candidate for that slot"


# --- how it ships ---------------------------------------------------------------------


@pytest.fixture
def local(tmp_path):
    from tests import test_workflow as workflow

    yield from workflow.local.__wrapped__(tmp_path)


@pytest.fixture
def wide(monkeypatch):
    """Rich hard-wraps to the terminal, and a sentence split across two lines is still the
    sentence. Widen the console rather than assert on fragments of one."""
    monkeypatch.setenv("COLUMNS", "200")


@pytest.fixture
def seeded(local):
    """A real database with week 1's report imported, so `pool recommend` has rivals."""
    from pool import scoring
    from tests import test_predictions as log
    from tests import test_workflow as workflow

    conn, path = local
    scoring.import_touchdowns(
        conn, 2026, pd.DataFrame([workflow.play(), workflow.end(), workflow.end("g2", 0, 0)])
    )
    log._history(conn)
    log._report(conn, 1, log.WEEK1, "2026-09-14T00:00:00+00:00")
    return conn, path


def test_recommend_prints_a_share_beside_every_expected_touchdown(seeded, wide):
    from typer.testing import CliRunner

    from pool.cli import app

    conn, path = seeded
    conn.close()
    with config.override(WINPROB_SIMS=200):
        result = CliRunner().invoke(app, ["recommend", "--week", "2", "--db", str(path)])
    assert result.exit_code == 0, result.output
    assert "Pot share" in result.output and "vs EV" in result.output
    assert "pot share" in result.output, "the recommended pick carries its own share"
    assert "Pot share vs 2 rivals" in result.output


def test_recommend_without_reports_says_so_rather_than_printing_zeroes(local, wide):
    from typer.testing import CliRunner

    from pool.cli import app
    from tests import test_predictions as log

    conn, path = local
    log._history(conn)
    conn.close()
    result = CliRunner().invoke(app, ["recommend", "--week", "2", "--db", str(path)])
    assert result.exit_code == 0, result.output
    assert "no pot-share view" in result.output and "pool report import" in result.output
    assert "Pot share" not in result.output, "no column of nothing"


def test_the_nudge_stops_once_the_week_can_be_seen(seeded):
    """A reminder to predict a week that has kicked off is an invitation to file a record
    `predictions.score` will refuse. It has to fall silent exactly when the window shuts."""
    from datetime import timedelta

    from pool import predictions
    from pool.cli import _prediction_nudge

    conn, _ = seeded
    kickoff = predictions.first_kickoff(conn, 2026, 2)
    before = _prediction_nudge(conn, 2026, 2, kickoff - timedelta(hours=1))
    assert before and "pool predict record --week 2" in before
    assert _prediction_nudge(conn, 2026, 2, kickoff) is None
    predictions.archive(conn, 2026, 2, predictions.predict(conn, 2026, 2, _proj_for(conn)))
    assert _prediction_nudge(conn, 2026, 2, kickoff - timedelta(hours=1)) is None


def _proj_for(conn):
    from pool import projections

    return projections.projections_for(conn, 2026, from_week=2)


def test_a_decision_made_against_rivals_reconstructs_from_its_own_record(seeded):
    """The capture is sufficient or it is decoration. Re-deriving the advice from the
    stored surface has to reproduce the shares too, which it can only do if the rival
    state behind them was written down beside them."""
    from pool import capture, predictions, prospective, state

    conn, _ = seeded
    now = state.eastern_now(datetime(2026, 9, 19, 12, 0, tzinfo=UTC))
    proj = _proj_for(conn)
    used, locked = state.used_ids(conn, 2026), state.locked_by_slot(conn, 2026)
    pool = predictions.pool_state(conn, 2026, 2)
    assert pool.rivals, "the fixture is meant to have opposition"
    with config.override(WINPROB_SIMS=200):
        advice = advise_week(proj, 2, used, locked, now=now, pool=pool)
        assert any(a.shares for a in advice), "the view has to have run for this to prove it"
        decision_id = capture.record_decision(
            conn, 2026, 2, proj, advice, used, locked,
            decision_at=state.decision_instant(now), pool=pool,
        )
        rebuilt = prospective.reconstruction(conn, decision_id)
    assert rebuilt["ok"], rebuilt


def test_a_capture_that_forgets_the_opposition_cannot_re_derive_its_own_shares(seeded):
    """The negative of the test above: without the pool on the surface, reconstruction
    reports the decision as differing from itself. This is what pins the storage."""
    from pool import capture, predictions, prospective, state

    conn, _ = seeded
    now = state.eastern_now(datetime(2026, 9, 19, 12, 0, tzinfo=UTC))
    proj = _proj_for(conn)
    used, locked = state.used_ids(conn, 2026), state.locked_by_slot(conn, 2026)
    pool = predictions.pool_state(conn, 2026, 2)
    with config.override(WINPROB_SIMS=200):
        advice = advise_week(proj, 2, used, locked, now=now, pool=pool)
        decision_id = capture.record_decision(
            conn, 2026, 2, proj, advice, used, locked,
            decision_at=state.decision_instant(now), pool=None,
        )
        rebuilt = prospective.reconstruction(conn, decision_id)
    assert not rebuilt["ok"] and rebuilt["reason"] == "re-derived advice differs"


def test_a_divergence_states_its_reason_in_the_terms_the_policy_used():
    """A divergence nobody can check is a bug report, not advice. The sentence has to name
    the rivals whose likely picks caused it, and it must come from the sampling sets the
    simulation drew from rather than from a second guess at them."""
    from pool.cli import _divergence

    proj = frame()
    pool = rivals.PoolState((rival("one", 0), rival("two", 0), rival("three", 0)), 0)
    _shares, advice = shares_for(proj, pool)
    qb = next(a for a in advice if a.slot == "QB")
    assert qb.divergent
    reason = _divergence(qb, pool)
    assert reason is not None, "a divergence with no statable reason is a defect"
    assert reason.startswith("Differentiating:")
    assert "and 1 more may take QA this week" in reason, "it names who is converging"
    assert "QB is further down their lists" in reason
    assert "level with the leader" in reason, "and the standing the policy read"
    assert "share for" in reason, "and what the swap costs in expected touchdowns"


def test_the_reason_is_withheld_rather_than_invented_when_nothing_explains_it():
    """The guard that keeps the sentence honest: no overlap either way and level standings
    leave the policy with nothing to say, and it must say nothing rather than guess."""
    from pool.cli import _divergence

    proj = frame()
    pool = rivals.PoolState((rival("one", 0), rival("two", 0), rival("three", 0)), 0)
    _shares, advice = shares_for(proj, pool)
    qb = next(a for a in advice if a.slot == "QB")
    assert _divergence(qb, pool), "the explicable case, for contrast"
    # Take the overlap away and leave everything else: level standings and a real
    # divergence with nothing left to attribute it to.
    qb.contested = {}
    assert _divergence(qb, pool) is None


def test_a_record_is_checked_on_what_it_claims_not_on_what_it_never_stored(seeded):
    """The case every 2026 capture hit the moment the view shipped.

    Those five decisions were recorded before `shares` existed. Demanding that they
    re-derive a field they never stored reported all five as insufficient, which is the
    opposite of what they are. What the record claims must still re-derive exactly; about
    a key it never had it makes no claim.
    """
    from pool import predictions, state
    from pool.prospective import _advice_details, _claims_hold

    conn, _ = seeded
    now = state.eastern_now(datetime(2026, 9, 19, 12, 0, tzinfo=UTC))
    proj = _proj_for(conn)
    used, locked = state.used_ids(conn, 2026), state.locked_by_slot(conn, 2026)
    with config.override(WINPROB_SIMS=200):
        advice = advise_week(
            proj, 2, used, locked, now=now, pool=predictions.pool_state(conn, 2026, 2)
        )
    derived = _advice_details(advice)["QB"]
    older = {k: v for k, v in derived.items() if k not in ("shares", "sims", "contested")}
    assert older != derived, "the fixture must exercise fields the older record lacked"
    assert _claims_hold(derived, older), "a record from before the field still verifies"
    assert not _claims_hold(derived, older | {"plan_total": -1.0}), "a claim it made still binds"
    assert not _claims_hold({"plan_total": older["plan_total"]}, older), "a dropped key fails"


def test_a_week_nobody_has_played_is_not_a_week_nobody_reported(seeded):
    """The distinction the first version of this missed.

    Asking for advice one week ahead counts spending through the week in between. If that
    week has not kicked off, nobody has picked in it and no report can exist, so treating
    its absence as missing evidence invents three spent players per rival out of a week
    that has not happened. Once it has been played and still has no report, the same
    absence is exactly the gap it was mistaken for.
    """
    from datetime import timedelta

    from pool import predictions

    conn, _ = seeded
    kickoff = predictions.first_kickoff(conn, 2026, 2)
    ahead = predictions.pool_state(conn, 2026, 3, at=kickoff - timedelta(hours=1))
    assert ahead.rivals and all(not r.missing_weeks for r in ahead.rivals)
    assert all(r.pool_complete for r in ahead.rivals), "looking ahead is not a data gap"
    after = predictions.pool_state(conn, 2026, 3, at=kickoff + timedelta(hours=1))
    assert all(r.missing_weeks == (2,) for r in after.rivals), "played and never reported"
    assert all(not r.pool_complete for r in after.rivals)


def test_the_policy_reads_a_report_that_arrived_before_the_decision_and_not_one_after(seeded):
    """The mid-week report, which the pool's own cadence makes normal.

    Week 2's report can land before week 2 locks. From that moment the policy knows what
    every rival holds this week and should stop guessing -- but only from that moment. A
    decision made an hour earlier must reconstruct against what it actually had, or a
    capture replays with intelligence the decision never saw.
    """
    from datetime import timedelta

    from pool import predictions
    from tests import test_predictions as log

    conn, _ = seeded
    arrival = datetime.fromisoformat(log.REPORT_AT)
    log._report(conn, 2, log.WEEK2, log.REPORT_AT)
    before = predictions.pool_state(conn, 2026, 2, at=arrival - timedelta(hours=1))
    assert all(not r.known for r in before.rivals), "not yet in hand"
    after = predictions.pool_state(conn, 2026, 2, at=arrival + timedelta(hours=1))
    assert after.rivals and all(r.known for r in after.rivals)
    pat = next(r for r in after.rivals if r.display_name == "Pat")
    assert pat.pinned("QB", [2]) == {2: "q2"}, "Pat's reported week-2 quarterback"
    assert all(week >= 2 for _slot, week, _pid in pat.known), "week 1 is spending, not a pick"


# --- does the answer need the numbers we had to fit? ----------------------------------


def _sweep(proj, pool, week=1, sims=1500):
    from pool.cli import _sensitivity

    with config.override(WINPROB_SIMS=sims, RIVAL_NOISE_TOP_N=1):
        advice = advise_week(proj, week, set(), {}, now=NOW, pool=pool)
        return _sensitivity(proj, week, advice, pool, {})


def test_the_sweep_finds_the_knob_the_divergence_actually_rests_on():
    """The sweep earning its cost, on the case the suite already pins.

    `k_game` turns out not to matter here: at every value, the policy still declines the
    player the whole field is about to take. The rival-noise parameter does matter, and it
    is the one declared provisional in `config` -- with a deterministic opponent the
    divergence is sharp, and with a noisy one it falls inside the band. Never, at any
    setting, does the expected-TD pick come back as the best share. That is the shape worth
    reporting: the direction is settled, the strength is not.
    """
    proj = frame()
    pool = rivals.PoolState((rival("one", 0), rival("two", 0), rival("three", 0)), 0)
    sweep = _sweep(proj, pool)
    qb = sweep[sweep.slot.eq("QB")]
    assert len(qb) == len(config.SENSITIVITY_K_GAME) * len(config.SENSITIVITY_NOISE)
    assert "qa" not in set(qb.player_id), "the expected-TD pick never wins on share"
    sharp = qb[~qb.tied]
    assert set(sharp.player_id) == {"qb"}, "where it separates it always names the same one"
    assert set(qb[qb.noise.eq(1)].player_id) == {"qb"} and not qb[qb.noise.eq(1)].tied.any()
    assert qb.tied.any(), "and with a noisier opponent it stops separating"


def test_a_slot_where_nothing_separates_reports_as_tied_not_as_knob_sensitive():
    """The distinction the first version of this table got wrong.

    When every candidate is inside the noise band, the argmax wanders between settings --
    but what is moving is the coin, not the knob. Reading the raw winner would report such
    a slot as wildly parameter-sensitive and send the reader looking for a calibration
    problem that is not there.
    """
    proj = frame()
    pool = rivals.PoolState((rival("one", 0, used=("qz", "qy", "q5", "q6")),), 0)
    sweep = _sweep(proj, pool)
    flex = sweep[sweep.slot.eq("FLEX")]
    assert len(flex) and flex.tied.all(), "nothing in FLEX separates at any setting"


def test_the_sweep_does_not_touch_the_advice_it_was_given():
    """It is a diagnostic. If it mutated the advice, the printed pick would depend on
    whether the flag was passed."""
    from pool.cli import _sensitivity

    proj = frame()
    pool = rivals.PoolState((rival("one", 0), rival("two", 0), rival("three", 0)), 0)
    with config.override(WINPROB_SIMS=1500, RIVAL_NOISE_TOP_N=1):
        advice = advise_week(proj, 1, set(), {}, now=NOW, pool=pool)
        before = [(a.slot, [s.player_id for s in a.shares], a.sims) for a in advice]
        _sensitivity(proj, 1, advice, pool, {})
        after = [(a.slot, [s.player_id for s in a.shares], a.sims) for a in advice]
    assert before == after
