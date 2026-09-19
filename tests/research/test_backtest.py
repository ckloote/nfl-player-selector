"""Season replays: every strategy spends each player once and fills every slot,
hindsight bounds them all, and a replay against invented rivals says so everywhere."""

import pandas as pd
import pytest

from pool import config, db
from pool import projections as P
from pool.research import backtest as B
from pool.research import models
from tests.support.frames import proj_row
from tests.support.season import SEASON, WEEKS, seed_season


@pytest.mark.parametrize("week", [1, 2, 3])
@pytest.mark.parametrize("model", sorted(models.BUILDERS))
def test_projections_are_identical_whether_or_not_the_future_exists(tmp_path, week, model):
    """The frozen frame must be a pure function of the visible slice.

    Stated this way the test catches any future leak, including through a
    column nobody thought about — which is how the depth-chart snapshot (a
    single end-of-season row, no week) slipped past review in the first place.

    Every benchmark model is checked, not just the shipped one: a baseline that
    quietly reads the whole season would flatter itself in the comparison, and
    the comparison is the entire point of having baselines.
    """
    full = seed_season(db.connect(tmp_path / "full.db"))
    trimmed = seed_season(db.connect(tmp_path / "trimmed.db"))
    with trimmed:
        trimmed.execute("DELETE FROM player_weeks WHERE season=? AND week>=?", (SEASON, week))
        trimmed.execute("DELETE FROM rosters WHERE season=? AND week>?", (SEASON, week))
        trimmed.execute(
            "UPDATE games SET spread_line=NULL, total_line=NULL WHERE season=? AND week>=?",
            (SEASON, week),
        )
    builder = models.get(model)
    build = P.build_projections if builder is None else builder
    a = build(P.load_frames(full, SEASON, as_of_week=week), week, role_source="usage")
    b = build(P.load_frames(trimmed, SEASON, as_of_week=week), week, role_source="usage")
    pd.testing.assert_frame_equal(a, b)


# --- scoring ----------------------------------------------------------------
def test_scoring_counts_every_touchdown_a_pick_throws_or_scores(seeded):
    actuals = B.actual_tds(seeded, SEASON)
    assert actuals[(1, "AAA-QB1")] == 1.0
    assert actuals[(1, "AAA-QB2")] == 0.0
    assert actuals.get((1, "nobody"), 0.0) == 0.0  # missing row is zero, not NaN


# --- replay invariants ------------------------------------------------------
@pytest.mark.parametrize("strategy", ["optimizer", "greedy", "random"])
def test_no_player_is_used_twice_and_every_slot_week_is_filled(seeded, strategy):
    import numpy as np

    frames = B.weekly_projections(seeded, SEASON, WEEKS)
    out = B.replay(
        frames,
        B.actual_tds(seeded, SEASON),
        SEASON,
        strategy,
        weeks=WEEKS,
        rng=np.random.default_rng(0),
    )
    assert len(out.picks) == len(WEEKS) * len(config.SLOTS)
    assert out.empty_slots == 0
    assert out.players_used == len(out.picks)  # the once-per-season rule


def test_hindsight_bounds_every_strategy(seeded):
    rows = {
        r.strategy: r
        for r in B.run_season(
            seeded, SEASON, ["optimizer", "greedy", "random", "hindsight"], weeks=WEEKS, trials=3
        )
    }
    ceiling = rows["hindsight"].total
    for name in ("optimizer", "greedy", "random"):
        assert rows[name].total <= ceiling


def test_the_random_baseline_is_reproducible_from_its_seed(seeded):
    kw = dict(weeks=WEEKS, trials=5, strategies=["random"])
    a = B.run_season(seeded, SEASON, seed=1, **kw)[0]
    b = B.run_season(seeded, SEASON, seed=1, **kw)[0]
    c = B.run_season(seeded, SEASON, seed=2, **kw)[0]
    assert [p.player_id for p in a.picks] == [p.player_id for p in b.picks]
    assert a.total != c.total or [p.player_id for p in a.picks] != [p.player_id for p in c.picks]


def test_a_backtest_never_touches_recorded_picks(seeded):
    """The harness keeps its state in memory; a replay must not be able to
    corrupt the real pool state it happens to share a database with."""
    with seeded:
        seeded.execute(
            "INSERT INTO my_picks(season, week, slot, player_id, player_name, recorded_at)"
            " VALUES (?,?,?,?,?,?)",
            (SEASON, 1, "QB", "AAA-QB1", "AAA QB1", "now"),
        )
    B.run_season(seeded, SEASON, ["optimizer", "greedy"], weeks=WEEKS)
    rows = seeded.execute("SELECT player_id FROM my_picks").fetchall()
    assert [r[0] for r in rows] == ["AAA-QB1"]


def test_the_optimizer_spends_the_star_in_his_best_week_and_greedy_does_not(make_proj):
    """The whole premise of the assignment approach, as the design doc puts it:
    greedy 'burns Josh Allen in week 1 against a top defense when week 9 offers
    him against the league's worst.'"""
    rows = []
    for wk, star, role in [(1, 1.0, 0.9), (2, 2.0, 0.2)]:
        rows += [proj_row("star", "Star", "QB", wk, star), proj_row("role", "Role", "QB", wk, role)]
    proj = make_proj(rows)
    frames = {1: proj, 2: proj}
    actuals = {(1, "star"): 1.0, (2, "star"): 5.0, (1, "role"): 1.0, (2, "role"): 0.0}
    greedy = B.replay(frames, actuals, 2024, "greedy", weeks=[1, 2])
    optimal = B.replay(frames, actuals, 2024, "optimizer", weeks=[1, 2], discount=1.0)
    assert [p.player_id for p in greedy.picks if p.slot == "QB"] == ["star", "role"]
    assert [p.player_id for p in optimal.picks if p.slot == "QB"] == ["role", "star"]
    assert optimal.total > greedy.total


def test_an_alternative_projection_model_can_be_swapped_in(seeded):
    """Candidate models are compared against the shipped one on identical frozen
    data; that seam is what the experiment branches hang off."""

    def half_speed(frames, from_week=1, role_source=None):
        out = P.build_projections(frames, from_week, role_source=role_source)
        return out.assign(lam=out.lam.astype(float) / 2)

    frames = B.weekly_projections(seeded, SEASON, WEEKS, builder=half_speed)
    assert all(
        (
            f.lam
            <= P.build_projections(
                P.load_frames(seeded, SEASON, as_of_week=w), w, role_source="usage"
            ).lam.max()
        ).all()
        for w, f in frames.items()
    )
    picks = B.replay(frames, B.actual_tds(seeded, SEASON), SEASON, "greedy", weeks=WEEKS)
    assert picks.players_used == len(picks.picks)  # still a valid replay


# --- the win-probability seam -----------------------------------------------
@pytest.fixture
def replayed(seeded):
    """Frozen frames and finished results for the shared fixture season."""
    return B.weekly_projections(seeded, SEASON, WEEKS), B.actual_tds(seeded, SEASON)


def _run(frames, actuals, field):
    """Walk a field through the season on its own, so its picks can be inspected."""
    run = field.start()
    for week in WEEKS:
        run.advance(frames[week], week, actuals)
    return run


def test_winprob_replays_a_season_without_spending_a_player_twice(replayed):
    frames, actuals = replayed
    out = B.replay(frames, actuals, SEASON, "winprob", weeks=WEEKS, against=B.Field(count=4))
    assert len(out.picks) == len(WEEKS) * len(config.SLOTS)
    assert out.empty_slots == 0
    assert out.players_used == len(out.picks)  # the once-per-season rule


def test_a_winprob_replay_is_labelled_as_run_against_invented_rivals(replayed):
    """Claim 25. The label travels on the `Replay` and on the `Summary` built from it,
    because the row is what ends up in a message to somebody."""
    frames, actuals = replayed
    out = B.replay(frames, actuals, SEASON, "winprob", weeks=WEEKS, against=B.Field(count=4))
    assert "4 invented rivals" in out.against
    assert "not of the idea" in B.INVENTED
    assert B.Summary.of(out).against == out.against
    assert B.Summary.of(out).finish == out.finish


def test_a_replay_against_nobody_claims_no_opposition(replayed):
    """The absence has to be legible too: an ordinary replay is played against nothing,
    and a blank `against` is what stops the caveat printing where it would be false."""
    frames, actuals = replayed
    out = B.replay(frames, actuals, SEASON, "greedy", weeks=WEEKS)
    assert out.against is None and out.standing == () and out.finish is None


def test_winprob_refuses_to_run_against_nobody(replayed):
    frames, actuals = replayed
    with pytest.raises(ValueError, match="invented rivals"):
        B.replay(frames, actuals, SEASON, "winprob", weeks=WEEKS)


def test_run_season_refuses_winprob_before_it_builds_a_single_frame(seeded, monkeypatch):
    """Named at the top rather than raised from three slots inside the week loop, after
    the expensive part has already been paid for."""
    monkeypatch.setattr(
        B, "weekly_projections", lambda *a, **kw: pytest.fail("built projections anyway")
    )
    with pytest.raises(ValueError, match="winprob"):
        B.run_season(seeded, SEASON, ["greedy", "winprob"], weeks=WEEKS)


def test_the_invented_field_plays_the_same_season_under_every_strategy(replayed):
    """They never see my picks -- this pool lets two entrants hold the same player -- so
    one specification gives every strategy the identical opposition, and the finishes in
    a single table are comparable rather than three separate seasons."""
    frames, actuals = replayed
    field = B.Field(count=4, seed=7)
    runs = [
        B.replay(frames, actuals, SEASON, name, weeks=WEEKS, against=field)
        for name in ("winprob", "optimizer", "greedy")
    ]
    assert len({r.standing for r in runs}) == 1
    assert all(r.standing for r in runs)


@pytest.mark.parametrize("behaviour", ["greedy", "naive", "optimizer"])
def test_invented_rivals_obey_the_one_player_per_season_rule(replayed, behaviour):
    """`naive` ignores it when it ranks, which is the whole of that hypothesis. The rule
    is still the pool's: a rival submitting a spent player would simply be rejected, so
    the invented one takes his next choice and cannot score the same man twice."""
    frames, actuals = replayed
    run = _run(frames, actuals, B.Field(count=2, behaviour=behaviour))
    for opponent in run.entrants:
        taken = [pid for _, _, pid in opponent.picks if pid]
        assert len(taken) == len(set(taken))
        assert len(opponent.picks) == len(WEEKS) * len(config.SLOTS)


def test_a_field_is_reproducible_from_its_seed_and_moves_with_it(replayed):
    frames, actuals = replayed

    def picks(seed):
        return [o.picks for o in _run(frames, actuals, B.Field(count=4, seed=seed)).entrants]

    assert picks(3) == picks(3)
    assert picks(3) != picks(4)


def test_an_unknown_rival_behaviour_is_refused_where_it_is_named(replayed):
    with pytest.raises(ValueError, match="telepathic"):
        B.Field(behaviour="telepathic")
    with pytest.raises(ValueError, match="at least one"):
        B.Field(count=0)


def test_cached_weekly_lineup_agrees_with_every_conditional_lock(replayed, monkeypatch):
    frames, actuals = replayed
    seen = []
    real = B.recommend.advise_week

    def spy(proj, week, used, locked, **kw):
        seen.append((week, {s: dict(p) for s, p in locked.items()}))
        return real(proj, week, used, locked, **kw)

    monkeypatch.setattr(B.recommend, "advise_week", spy)
    result = B.replay(frames, actuals, SEASON, "winprob", weeks=WEEKS, against=B.Field(count=2))
    final = {(p.week, p.slot): p.player_id for p in result.picks}
    for week in WEEKS:
        calls = [locks for w, locks in seen if w == week]
        assert calls[0] == {}
        assert len(calls) <= len(config.SLOTS)
        for locks in calls[1:]:
            assert list(locks) == list(config.SLOTS)[: len(locks)]
            assert all(final[week, slot] == picks[week] for slot, picks in locks.items())


def test_the_replay_clock_forbids_nothing_the_other_strategies_kept(replayed):
    """`advise_slot` needs a clock and the other strategies have none -- they go through
    `build_matrix`, which consults none. A live reading would drop every candidate whose
    game had kicked off, and the gap between two strategies would stop being a gap
    between two decision rules."""
    from pool import state

    frames, _ = replayed
    for week in WEEKS:
        assert not state.unavailable_cells(frames[week], week, B.decision_time(frames[week], week))


def test_a_shared_first_place_is_reported_as_shared_and_not_as_a_win():
    """A tie at the top splits the pot. A finish that called it a win would be the same
    mistake `simulate.shares` exists to avoid, printed instead of simulated."""
    mine = [B.Pick(1, "QB", "x", "X", "AAA", 0.0, 7.0)]
    assert B.Replay(SEASON, "winprob", mine, standing=(("A", 9.0), ("B", 3.0))).finish == "2nd of 3"
    shared = B.Replay(SEASON, "winprob", mine, standing=(("A", 7.0), ("B", 3.0)))
    assert shared.rank == 1 and shared.level == 1
    assert shared.finish == "1st of 3, sharing with 1"


@pytest.mark.parametrize(("delta", "deviates"), [(0.5, True), (0.0005, False)])
def test_winprob_deviates_only_where_the_pot_share_separates(
    replayed, monkeypatch, delta, deviates
):
    """The deviation rule is `SlotAdvice.divergent` and not the raw pot-share argmax:
    inside simulation noise it keeps the expected-TD pick. Asserted in both directions,
    because the seam passing every other test here is also what a synonym for `optimizer`
    would do."""
    from pool.recommend import PotShare

    frames, actuals = replayed
    week = WEEKS[0]
    real = B.recommend.advise_week

    def separated(*a, **kw):
        advice = real(*a, **kw)
        for one in advice:
            if one.recommended is None:
                continue
            alt, best = one.alternatives[0], one.recommended
            one.shares = [
                PotShare(alt.player_id, alt.player_name, 0.9, 0.001, delta, 0.001),
                PotShare(best.player_id, best.player_name, 0.9 - delta, 0.001, 0.0, 0.001),
            ]
        return advice

    monkeypatch.setattr(B.recommend, "advise_week", separated)
    out = B.replay(frames, actuals, SEASON, "winprob", weeks=[week], against=B.Field(count=2))
    plain = B.replay(frames, actuals, SEASON, "optimizer", weeks=[week])
    moved = [p.player_id for p in out.picks] != [p.player_id for p in plain.picks]
    assert moved is deviates
