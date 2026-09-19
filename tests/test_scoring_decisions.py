"""Regressions for prospective scoring and conditional pot-share decisions."""

import numpy as np
import pandas as pd
import pytest

from pool import config, recommend, rivals, simulate
from pool.research import backtest
from tests.test_winprob import NOW, cell, frame, rival


def test_clustered_uncertainty_uses_block_sums_and_unequal_sizes():
    values = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    edges = np.array([0, 2, 5])
    expected = np.sqrt(2 * ((-1.2) ** 2 + 1.2**2) / 25)
    assert simulate.clustered_se(values, edges) == pytest.approx(expected)
    assert simulate.paired(values, np.zeros(5), edges)[1] == pytest.approx(expected)
    assert np.isinf(simulate.clustered_se(values, [0, 5]))


def test_fixed_outcomes_replace_draws_include_missing_and_count_once():
    draws = simulate.Draws(np.array([[1, 2, 3]]), {(1, "a"): 0}, 3)
    fixed = ((1, "a", 0), (1, "missing", 4))
    assert np.array_equal(
        draws.totals([(1, "a"), (1, "missing"), (1, "missing")], fixed), [4, 4, 4]
    )


def test_shares_evaluate_hidden_eighth_qb_independent_of_display_limit():
    proj = pd.DataFrame([cell(f"q{i}", "QB", 2 - i / 10, f"T{i}", f"g{i}") for i in range(8)])
    pool = rivals.PoolState((rivals.RivalState("r", "Rival", known=(("QB", 1, "q7"),)),), 1)
    results = []
    for limit in (0, 20):
        with config.override(ALTERNATIVES_SHOWN=limit, WINPROB_SIMS=2000):
            qb = recommend.advise_week(proj, 1, set(), {}, now=NOW, pool=pool)[0]
        assert len(qb.shares) == 8
        assert qb.best_share.player_id == "q7"
        assert qb.best_share.share == 1
        assert "q7" in {c.player_id for c in qb.alternatives}
        results.append(qb.shares)
    assert results[0] == results[1]


def test_pot_shares_sample_full_frame_with_spent_and_capped_teammates(monkeypatch):
    proj = frame()
    proj = pd.concat(
        [
            proj,
            pd.DataFrame(
                [
                    cell("teammate", "FLEX", 0.01, "A", "g1a"),
                ]
            ),
        ],
        ignore_index=True,
    )
    real = simulate.sample
    seen = []

    def sample(sub, weeks, **kw):
        draws = real(sub, weeks, **kw)
        full = real(proj, weeks, **kw)
        for key, row in draws.index.items():
            assert np.array_equal(draws.values[row], full.values[full.index[key]])
        seen.append(set(sub.player_id))
        return draws

    monkeypatch.setattr(simulate, "sample", sample)
    with config.override(WINPROB_SIMS=100, CANDIDATES_PER_SLOT=2):
        recommend.advise_week(
            proj,
            1,
            {"teammate"},
            {},
            now=NOW,
            pool=rivals.PoolState((rival("r", 0, ("teammate",)),)),
        )
    assert seen == [set(proj.player_id)]


def test_naive_invented_rivals_fill_eighteen_weeks_despite_candidate_cap():
    proj = pd.DataFrame(
        [
            cell(f"q{i}", "QB", 20 - i, f"T{i}", f"g{w}-{i}", week=w)
            for w in range(1, 19)
            for i in range(20)
        ]
    )
    field = backtest.Field(count=1, behaviour="naive", top_n=1).start()
    with config.override(CANDIDATES_PER_SLOT=4):
        for week in range(1, 19):
            field.advance(proj, week, {})
    picks = [pid for _, slot, pid in field.entrants[0].picks if slot == "QB"]
    assert None not in picks
    assert len(set(picks)) == 18


@pytest.fixture
def local(tmp_path):
    from tests import test_workflow as workflow

    yield from workflow.local.__wrapped__(tmp_path)


def test_live_pool_keeps_finalized_current_results_separate_from_banked(local):
    from datetime import UTC, datetime

    from pool import predictions, scoring
    from tests import test_predictions as log
    from tests.test_workflow import end, play

    conn, _ = local
    log._report(conn, 1, log.WEEK1, "2026-09-09T12:00:00+00:00")
    scoring.import_touchdowns(
        conn,
        2026,
        pd.DataFrame(
            [
                *[play(pid="q1", play_id=i) for i in range(1, 5)],
                end(),
                end("g2", 0, 0),
            ]
        ),
    )
    with conn:
        conn.execute("UPDATE game_results SET imported_at = '2026-09-11T12:00:00+00:00'")
        conn.execute(
            "UPDATE game_results SET imported_at = '2026-09-14T12:00:00+00:00' WHERE game_id = 'g2'"
        )
        conn.execute(
            "INSERT INTO my_picks(season, week, slot, player_id, player_name, "
            "recorded_at, game_id) VALUES (2026, 1, 'RB', 'r1', 'Runner One', "
            "'2026-09-09T12:00:00+00:00', 'g1')"
        )
    pool = predictions.pool_state(conn, 2026, 1, at=datetime(2026, 9, 12, tzinfo=UTC))
    assert pool.my_tds == 0
    assert all(r.season_tds == 0 for r in pool.rivals)
    assert (1, "q1", 4) in pool.finalized
    assert (1, "r1", 0) in pool.finalized
    assert (1, "q2", 0) in pool.finalized
    assert not any(pid == "f1" for _, pid, _ in pool.finalized), "Sunday not imported yet"
    assert len(pool.finalized) == len(set(pool.finalized)), "shared players appear once"
    later = predictions.pool_state(conn, 2026, 2, at=datetime(2026, 9, 15, tzinfo=UTC))
    assert later.my_tds == 4
    assert next(r for r in later.rivals if r.display_name == "Pat").season_tds == 4
    assert not later.finalized


def test_backtest_conditions_later_slots_on_accepted_deviation(monkeypatch):
    from pool.recommend import PotShare

    proj = frame()
    real = recommend.advise_week
    calls = []

    def conditional(proj, week, used, locked, **kw):
        assert kw["discount"] == 0
        calls.append({s: dict(p) for s, p in locked.items()})
        result = real(proj, week, used, locked, **kw)
        for one in result:
            if one.recommended is None or one.slot == "RB":
                continue
            # Both deviations beat the original lineup, but their combination loses.
            if one.slot == "FLEX" and locked.get("QB") == {1: "qb"}:
                one.shares = []
                continue
            alt = next(
                c for c in one.candidates if c.player_id == ("qb" if one.slot == "QB" else "fb")
            )
            one.shares = [PotShare(alt.player_id, alt.player_name, 0.9, 0.01, 0.4, 0.01)]
        return result

    monkeypatch.setattr(recommend, "advise_week", conditional)
    memo = {}
    against = backtest.Field(count=1).start()
    with config.override(WINPROB_SIMS=100):
        choices = [
            backtest.pick_winprob(proj, slot, 1, set(), against=against, memo=memo, discount=0)
            for slot in config.SLOTS
        ]
        again = backtest.pick_winprob(proj, "QB", 1, set(), against=against, memo=memo)
    assert [c.player_id for c in choices] == ["qb", "ra", "fa"]
    assert again == choices[0]
    assert calls == [{}, {"QB": {1: "qb"}}]
    assert memo[1] == dict(zip(config.SLOTS, choices, strict=True))


def test_discount_zero_fallback_matches_optimizer(monkeypatch):
    proj = frame(weeks=(1, 2))
    real = recommend.advise_week
    seen = []

    def tied(*args, **kw):
        seen.append(kw.get("discount"))
        result = real(*args, **kw)
        for one in result:
            one.shares = []
        return result

    monkeypatch.setattr(recommend, "advise_week", tied)
    memo = {}
    against = backtest.Field(count=1).start()
    with config.override(WINPROB_SIMS=50):
        for slot in config.SLOTS:
            chosen = backtest.pick_winprob(
                proj, slot, 1, set(), against=against, memo=memo, discount=0
            )
            assert chosen == backtest.pick_optimizer(proj, slot, 1, set(), discount=0)
    assert seen == [0]


@pytest.mark.parametrize("sims", [1, 2, 5, 33])
def test_rival_scenarios_are_nonempty_and_bound_uncertainty(sims):
    proj = frame()
    pool = rivals.PoolState((rival("r", 0),))
    draws = simulate.sample(proj, [1], sims=sims, seed=4)
    totals, _, edges = recommend._rival_totals(proj, 1, [1], pool, draws, sims, 4)
    assert totals.shape == (sims, 1)
    assert len(edges) - 1 == min(config.RIVAL_SCENARIOS, sims)
    assert (np.diff(edges) > 0).all()
    with config.override(WINPROB_SIMS=sims):
        advice = recommend.advise_week(proj, 1, set(), {}, now=NOW, pool=pool)
    for one in advice:
        for share in one.shares:
            assert np.isinf(share.se) if sims == 1 else np.isfinite(share.se)
            if sims == 1:
                assert share.tied and np.isinf(share.delta_se)


def test_exchangeable_qbs_seed_four_improvement_is_within_scenario_uncertainty():
    proj = pd.DataFrame([cell(f"q{i}", "QB", 1, f"T{i}", f"g{i}") for i in range(2)])
    pool = rivals.PoolState((rival("r", 0),), my_tds=1)
    with config.override(WINPROB_SIMS=2000, WINPROB_SEED=4, RIVAL_NOISE_TOP_N=2):
        qb = recommend.advise_week(proj, 1, set(), {}, now=NOW, pool=pool)[0]
        draws = simulate.sample(proj, [1], sims=2000, seed=4)
        opposition, _, _ = recommend._rival_totals(proj, 1, [1], pool, draws, 2000, 4)
    reference = simulate.shares(
        pool.my_tds + draws.totals([(1, qb.recommended.player_id)]), opposition
    )
    candidate = simulate.shares(
        pool.my_tds + draws.totals([(1, qb.best_share.player_id)]), opposition
    )
    delta, iid_se = simulate.paired(candidate, reference)
    assert qb.best_share.player_id == "q1"
    assert delta > config.WINPROB_SIGNIFICANCE * iid_se, "the old estimator claimed an edge"
    assert qb.best_share.delta == pytest.approx(0.0675)
    assert qb.best_share.tied
    assert not qb.divergent


@pytest.mark.parametrize("owner", ["mine", "rival", "shared"])
@pytest.mark.parametrize("missing_projection", [False, True])
def test_final_thursday_four_tds_change_sunday_shares(owner, missing_projection):
    from dataclasses import replace

    proj = frame()
    proj.loc[proj.player_id.eq("qa"), "kickoff"] = "2026-09-10T20:15"
    locked = {"QB": {1: "qa"}} if owner in ("mine", "shared") else {}
    known = (("QB", 1, "qa"),) if owner in ("rival", "shared") else (("QB", 1, "qb"),)
    pool = rivals.PoolState((rivals.RivalState("r", "Rival", known=known),))
    if missing_projection:
        proj = proj[proj.player_id != "qa"]
    with config.override(WINPROB_SIMS=2000, RIVAL_NOISE_TOP_N=1):
        before = recommend.advise_week(proj, 1, set(), locked, now=NOW, pool=pool)
        after = recommend.advise_week(
            proj, 1, set(), locked, now=NOW, pool=replace(pool, finalized=((1, "qa", 4),))
        )
    b = next(a for a in before if a.slot == "FLEX").shares
    a = next(a for a in after if a.slot == "FLEX").shares
    b = {s.player_id: s.share for s in b}
    a = {s.player_id: s.share for s in a}
    if owner == "shared":
        assert a == b, "the same result on both sides cancels exactly"
    elif owner == "mine":
        assert a["fa"] > b["fa"]
    else:
        assert a["fa"] < b["fa"]


def test_capture_stores_final_results_and_survives_later_corrections(local):
    from dataclasses import replace
    from datetime import UTC, datetime

    from pool import capture, predictions, scoring
    from pool.research import verify
    from tests import test_predictions as log
    from tests.test_workflow import end, play

    conn, _ = local
    log._report(conn, 1, log.WEEK1, "2026-09-09T12:00:00+00:00")
    scoring.import_touchdowns(
        conn,
        2026,
        pd.DataFrame(
            [
                *[play(pid="q1", play_id=i) for i in range(1, 5)],
                end(),
            ]
        ),
    )
    with conn:
        conn.execute("UPDATE game_results SET imported_at = '2026-09-11T12:00:00+00:00'")
    now = datetime(2026, 9, 12, tzinfo=UTC)
    pool = predictions.pool_state(conn, 2026, 1, at=now)
    assert (1, "q1", 4) in pool.finalized
    detail = capture._pool_detail(pool)
    assert capture.pool_from_detail(detail) == pool
    assert detail["state_hash"] != capture._pool_detail(replace(pool, finalized=()))["state_hash"]
    old = {k: v for k, v in detail.items() if k != "finalized"}
    assert capture.pool_from_detail(old).finalized == ()
    proj = frame()
    proj["player_id"] = proj.player_id.replace({"qa": "q1", "qb": "q2"})
    proj.loc[proj.player_id.eq("q1"), "kickoff"] = "2026-09-10T20:15"
    with config.override(WINPROB_SIMS=100):
        advice = recommend.advise_week(proj, 1, set(), {}, now=now, pool=pool)
        unfixed = recommend.advise_week(
            proj, 1, set(), {}, now=now, pool=replace(pool, finalized=())
        )
        assert advice[0].shares != unfixed[0].shares
        decision = capture.record_decision(
            conn, 2026, 1, proj, advice, set(), {}, decision_at=now, pool=pool
        )
    scoring.import_touchdowns(conn, 2026, pd.DataFrame([end(), end("g2", 0, 0)]))
    result = verify.reconstruction(conn, decision)
    assert result["ok"], result


def test_pending_and_not_yet_reported_results_are_excluded(local):
    from datetime import UTC, datetime

    from pool import predictions, scoring
    from tests import test_predictions as log
    from tests.test_workflow import end, play

    conn, _ = local
    scoring.import_touchdowns(conn, 2026, pd.DataFrame([play(pid="q1"), end()]))
    with conn:
        conn.execute("UPDATE game_results SET imported_at = '2026-09-11T12:00:00+00:00'")
    log._report(conn, 1, log.WEEK1, "2026-09-13T12:00:00+00:00")
    before = predictions.pool_state(conn, 2026, 1, at=datetime(2026, 9, 12, tzinfo=UTC))
    assert not before.finalized
    assert all(not r.known for r in before.rivals)
    with conn:
        conn.execute("UPDATE game_results SET complete = 0, reason = 'incomplete'")
    after = predictions.pool_state(conn, 2026, 1, at=datetime(2026, 9, 14, tzinfo=UTC))
    assert all(r.known for r in after.rivals)
    assert not after.finalized


def test_individually_beneficial_qb_and_flex_deviations_cannot_be_combined(monkeypatch):
    """Each swap wins 100%, but combining them drops share from 70% to 65%."""
    from types import SimpleNamespace

    proj = pd.DataFrame(
        [
            cell("qa", "QB", 2.7, "A", "a"),
            cell("qb", "QB", 2.6, "B", "b"),
            cell("fa", "FLEX", 3.4, "C", "c"),
            cell("fb", "FLEX", 3.3, "D", "d"),
        ]
    )
    values = np.tile(
        np.array(
            [
                [3] * 7 + [2] * 3,
                [2] * 7 + [4] * 3,
                [4] * 7 + [2] * 3,
                [3] * 7 + [4] * 3,
            ]
        ),
        (1, 200),
    )
    draws = simulate.Draws(values, {(1, pid): i for i, pid in enumerate(proj.player_id)}, 2000)
    monkeypatch.setattr(simulate, "sample", lambda *a, **kw: draws)
    pool = rivals.PoolState(
        (
            rivals.RivalState(
                "r",
                "Rival",
                known=(("QB", 1, "rq"), ("FLEX", 1, "rf")),
            ),
        ),
        finalized=((1, "rq", 5), (1, "rf", 0)),
    )
    advice = recommend.advise_week(proj, 1, set(), {}, now=NOW, pool=pool)
    assert advice[0].divergent and advice[2].divergent
    assert advice[0].best_share.player_id == "qb"
    assert advice[2].best_share.player_id == "fb"
    memo = {}
    chosen = [
        backtest.pick_winprob(
            proj, s, 1, set(), memo=memo, against=SimpleNamespace(state=lambda banked: pool)
        )
        for s in config.SLOTS
    ]
    assert [c.player_id for c in chosen] == ["qb", None, "fa"]
    total = draws.totals([(1, c.player_id) for c in chosen if c.player_id])
    rivals_total = np.full((draws.sims, 1), 5)
    assert simulate.shares(total, rivals_total).mean() == 1
    incompatible = draws.totals([(1, "qb"), (1, "fb")])
    assert simulate.shares(incompatible, rivals_total).mean() == 0.65
