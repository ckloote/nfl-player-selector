"""Captured decisions: the whole surface a pick was advised from, written down once,
re-derivable from that record alone, and linked to the picks that came from it.

Each test states the failure it prevents. The same-week tests come first: a decision made
mid-week has to read the same inputs live as a replay of that instant reads from the
archive, or nothing captured about it can be checked.
"""

import contextlib
import json
import sqlite3
from datetime import datetime

import pandas as pd
import pytest
from typer.testing import CliRunner

from pool import capture, config, db, snapshots, state
from pool import projections as P
from pool.cli import app
from pool.recommend import advise_week
from tests.support.decisions import (
    DECISION,
    POST_SUN,
    POST_THU,
    PRE_WEEK3,
    archived,
    decide,
    publish,
    staged,
)
from tests.support.season import SEASON, archive_all, seed_season


# --- one same-week stats contract -------------------------------------------
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
    conn, unplayed = staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    publish(conn, unplayed, "thursday")
    for feed in ("player_stats", "touchdowns"):
        snapshots.archive(conn, SEASON, feed, observed_at=POST_THU)

    now = datetime.fromisoformat(DECISION)
    live = P.build_projections(P.load_frames(conn, SEASON, input_policy="live"), 3, "usage")
    live_advice = _advice(live, now)

    # Everything after the decision arrives before the replay is run.
    publish(conn, unplayed, "rest")
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
    conn, unplayed = staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    publish(conn, unplayed, "thursday")
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
    conn, unplayed = staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    publish(conn, unplayed, "thursday")
    for feed in ("player_stats", "touchdowns"):
        snapshots.archive(conn, SEASON, feed, observed_at=POST_THU)
    before = P.build_projections(
        P.load_frames(conn, SEASON, 3, input_policy="snapshots", decision_at=DECISION), 3, "usage"
    )

    publish(conn, unplayed, "rest")
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
    conn, _ = staged(tmp_path)
    frames = P.load_frames(conn, SEASON, 3)
    assert sorted(frames.pw_cur.week.unique()) == [1, 2]
    assert P._stats_through("historical", 3) == 2
    assert P._stats_through("legacy-closing", 3) == 2
    assert P._stats_through("snapshots", 3) == 3
    assert P._stats_through("live", None) is None


# --- append-only decision capture -------------------------------------------
def test_the_capture_stores_the_whole_surface_not_the_shortlist(tmp_path):
    """The optimizer prunes to eighty candidates and the recommender shows six. A
    diagnostic asking about eligible zero rates, or about a player at a four-week
    horizon, has to find them here, because nothing downstream keeps them."""
    conn, unplayed = staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    publish(conn, unplayed, "thursday")
    decision_id, proj, _ = decide(conn, 3, datetime.fromisoformat(DECISION))

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
    conn, unplayed = staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    publish(conn, unplayed, "thursday")
    decision_id, _, advice = decide(conn, 3, datetime.fromisoformat(DECISION))

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
    conn, unplayed = staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    publish(conn, unplayed, "thursday")
    # Before the Thursday kickoff, so its players are still live and early.
    decide(conn, 3, datetime.fromisoformat("2024-09-19T18:30:00+00:00"))
    kinds = set(capture.events(conn, SEASON).kind)
    assert {"surface", "advice"} <= kinds
    assert kinds & {"hold", "commit"}


def test_captured_inputs_name_the_observations_the_decision_could_see(tmp_path):
    conn, unplayed = staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    publish(conn, unplayed, "thursday")
    for feed in ("player_stats", "touchdowns"):
        snapshots.archive(conn, SEASON, feed, observed_at=POST_THU)
    decision_id, _, _ = decide(conn, 3, datetime.fromisoformat(DECISION))

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


def test_a_snapshot_transaction_holds_its_reads_before_the_body_runs(tmp_path):
    """A SAVEPOINT is deferred: SQLite fixes what a connection sees at its first read,
    not when the savepoint opens. So a transaction that has not read yet is no defence
    at all, and a timestamp taken beside one describes data nobody has looked at."""
    path = tmp_path / "pool.db"
    seed_season(db.connect(path)).close()
    row = ("g-late", SEASON, 9, "REG", "2024-11-03T13:00", "Sun", "AAA", "BBB")
    insert = (
        "INSERT OR REPLACE INTO games (game_id, season, week, game_type, kickoff, weekday,"
        " home_team, away_team) VALUES (?,?,?,?,?,?,?,?)"
    )

    def writes_through(snapshot):
        holder, writer = db.connect(path), db.connect(path)
        writer.execute("PRAGMA busy_timeout = 50")
        try:
            with db.transaction(holder, snapshot=snapshot):
                try:
                    writer.execute(insert, row)
                    writer.commit()
                    return True
                except sqlite3.OperationalError:
                    return False
        finally:
            holder.close()
            writer.close()

    assert writes_through(snapshot=False), "the savepoint alone was expected to hold nothing"
    assert not writes_through(snapshot=True)


def test_a_refresh_cannot_land_between_the_decision_clock_and_the_data_it_names(
    tmp_path, monkeypatch
):
    """The half of the one-transaction fix that was missing.

    `recommend` timestamped the decision, then opened a transaction whose first read
    came later. A refresh committing in that window fed the forecast an injury that
    `observed_inputs` -- everything at or before the timestamp -- did not contain, and
    the command reported success. Here the refresh tries to commit at exactly the
    instant the clock is read, which is the worst case the ordering has to survive.
    """
    path = tmp_path / "pool.db"
    conn = seed_season(db.connect(path))
    archive_all(conn, PRE_WEEK3)
    conn.commit()
    conn.close()

    refresher = db.connect(path)  # a concurrent `pool refresh`
    refresher.execute("PRAGMA busy_timeout = 50")
    landed = []

    def refresh():
        """Rule BBB WR1 out and archive the feed, as `pool refresh` would."""
        with db.transaction(refresher):
            refresher.execute(
                "INSERT OR REPLACE INTO injuries VALUES (?,?,?,?,?,?,?,?)",
                (SEASON, 1, "BBB-WR1", "BBB WR1", "BBB", "WR", "Out", "DNP"),
            )
            snapshots.archive(refresher, SEASON, "injuries")  # observed now, after the clock
        refresher.commit()

    real_clock = state.eastern_now

    def clock(value=None):
        if not landed:
            landed.append("committed")
            try:
                refresh()
            except sqlite3.OperationalError:
                landed[0] = "blocked"
                refresher.rollback()  # the refresh gives up, as a real one would
        return real_clock(value)

    monkeypatch.setattr(state, "eastern_now", clock)
    result = CliRunner().invoke(
        app, ["recommend", "--week", "1", "--season", str(SEASON), "--db", str(path)]
    )
    monkeypatch.undo()
    refresher.close()
    assert result.exit_code == 0, (result.output, result.exception)
    assert landed == ["blocked"], "a refresh reached the window between the clock and the reads"

    conn = db.connect(path)
    head = conn.execute(
        "SELECT decision_id, surface_hash FROM decision_events WHERE kind = 'surface'"
    ).fetchone()
    surface = capture.load_surface(conn, head["surface_hash"])
    ruled_out = surface[surface.player_id.eq("BBB-WR1") & surface.week.eq(1)]
    assert len(ruled_out) and ruled_out.avail_mult.eq(1.0).all(), (
        "the forecast used an injury report the capture does not reference"
    )
    inputs = db.read_df(
        conn, "SELECT * FROM decision_inputs WHERE decision_id = ?", (head["decision_id"],)
    )
    injuries = inputs[inputs.feed.eq("injuries") & inputs.season.eq(SEASON)].iloc[0]
    assert injuries.observed_at == snapshots.timestamp(PRE_WEEK3)
    conn.close()


def test_a_decision_survives_a_later_feed_correction(tmp_path):
    """The reason outcomes are joined at read time and never stored on an event."""
    conn, unplayed = staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    publish(conn, unplayed, "thursday")
    decision_id, _, _ = decide(conn, 3, datetime.fromisoformat(DECISION))
    before, advice_before, recorded, _ = capture.reconstruct(conn, decision_id)

    publish(conn, unplayed, "rest")
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
    seed_season(db.connect(path)).close()
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
    seed_season(db.connect(path)).close()
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
    conn, unplayed = staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    publish(conn, unplayed, "thursday")
    before_kickoff = datetime.fromisoformat("2024-09-19T18:30:00+00:00")

    identities, holds = {}, {}
    for premium in (0.10, 0.0001):
        with config.override(INFO_PREMIUM_TD=premium):
            decision_id, _, advice = decide(conn, 3, before_kickoff)
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
    conn, unplayed = staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    publish(conn, unplayed, "thursday")
    before_kickoff = datetime.fromisoformat("2024-09-19T18:30:00+00:00")
    now = state.eastern_now(before_kickoff)
    with config.override(INFO_PREMIUM_TD=0.10):
        decision_id, _, advice = decide(conn, 3, before_kickoff)
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


def test_a_reconstructed_deadline_is_the_one_the_decision_was_made_under(tmp_path):
    """Restoring the recorded settings is not enough if the answer is read afterwards.

    `Candidate.deadline` was a property over `config.PICK_DEADLINE_MINUTES`, and advice
    is returned after the restoring override has ended, so every reconstructed deadline
    silently re-derived itself under today's policy -- 19:15 where 18:15 was recorded.
    The stored event was right the whole time, which is what makes it checkable."""
    conn, unplayed = staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    publish(conn, unplayed, "thursday")
    with config.override(PICK_DEADLINE_MINUTES=120):
        decision_id, _, advice = decide(conn, 3, datetime.fromisoformat(DECISION))
    assert config.PICK_DEADLINE_MINUTES == 60, "the fixture must reconstruct under a moved setting"

    _, redone, recorded, _ = capture.reconstruct(conn, decision_id)
    seen = 0
    for again in redone:
        detail = recorded[again.slot]
        for candidate, stored in [
            (again.recommended, detail["recommended"]),
            (again.hold_alternative, detail["hold_alternative"]),
            *zip(again.alternatives, detail["alternatives"], strict=True),
        ]:
            if candidate is None:
                assert stored is None
                continue
            assert candidate.player_id == stored["player_id"]
            assert candidate.deadline.isoformat() == stored["deadline"]
            seen += 1
    assert seen, "the fixture must recommend something to compare"


def test_reconstruction_refuses_a_source_tree_it_was_not_captured_under(tmp_path):
    """The recommender's behaviour is not carried by the recorded constants alone, so a
    silent substitution of current code would answer a different question."""
    conn, unplayed = staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    publish(conn, unplayed, "thursday")
    decision_id, _, _ = decide(conn, 3, datetime.fromisoformat(DECISION))

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


def test_a_module_the_decision_does_not_depend_on_does_not_refuse_it(tmp_path):
    """The fingerprint the checks enforce covers what a decision is a function of. A
    leaderboard or an ingestion command changes the tree and changes nothing the
    recommender did, and refusing to reconstruct on that would fail a capture nothing
    had touched -- which is what makes an unrelated feature cost six weeks of evidence."""
    conn, unplayed = staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    publish(conn, unplayed, "thursday")
    decision_id, _, _ = decide(conn, 3, datetime.fromisoformat(DECISION))
    _code, decision_hash, revision, dirty = capture._code_identity()

    capture._code_identity.cache_clear()
    original = capture._code_identity
    capture._code_identity = lambda: (
        "a module outside the closure moved",
        decision_hash,
        revision,
        dirty,
    )
    try:
        _, _, _, drift = capture.reconstruct(conn, decision_id)  # no allow_code_drift
        assert not drift["code_hash_changed"]
        assert drift["whole_tree_changed"]  # recorded, and visible, but not enforced
    finally:
        capture._code_identity = original
        capture._code_identity.cache_clear()


@contextlib.contextmanager
def _pretend_source_moved():
    original = capture._code_identity
    # Whole tree and decision closure both moved: this is the case that must be refused.
    capture._code_identity = lambda: ("moved", "moved", "moved", False)
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
    conn, unplayed = staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    publish(conn, unplayed, "thursday")
    decide(conn, 3, datetime.fromisoformat(DECISION))
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        with conn:
            conn.execute(f"UPDATE {table} SET season = 1999")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        with conn:
            conn.execute(f"DELETE FROM {table}")


def test_repeating_a_decision_on_unchanged_inputs_stores_one_surface(tmp_path):
    """Content-addressed like the feed archive: re-running `recommend` should cost an
    event, not another copy of the whole surface."""
    conn, unplayed = staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    publish(conn, unplayed, "thursday")
    first, _, _ = decide(conn, 3, datetime.fromisoformat(DECISION))
    second, _, _ = decide(conn, 3, datetime.fromisoformat(DECISION))

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
    conn, unplayed = staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    publish(conn, unplayed, "thursday")
    decide(conn, 3, datetime.fromisoformat(DECISION))

    joined = capture.outcomes(conn, SEASON)
    assert "actual_tds" in joined and "outcome_complete" in joined
    # Only the Thursday game has been played, so week 3 is not fully scored and week 4
    # has not started; neither may be read as a zero.
    assert not joined.outcome_complete.any()
    assert joined.actual_tds.isna().all()

    publish(conn, unplayed, "rest")
    complete = capture.outcomes(conn, SEASON)
    assert complete.outcome_complete.all()
    assert complete.actual_tds.notna().all()


def test_the_cli_records_submissions_and_corrections_without_editing_history(tmp_path):
    """`my_picks` holds the current answer and is overwritten in place. The capture log
    has to hold every answer, or a corrected pick erases the one it replaced."""
    path = tmp_path / "pool.db"
    conn = seed_season(db.connect(path))
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
    conn = seed_season(db.connect(path))
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


def test_a_submission_links_to_the_decision_it_names_not_the_latest(tmp_path):
    """With a Thursday and a Sunday decision in one week, "the most recent advice for
    this slot" is whichever happened last, which is not the same thing as the one the
    pick came from."""
    conn, _ = archived(tmp_path)
    early, proj, _ = decide(conn, 3, datetime.fromisoformat(PRE_WEEK3))
    late, _, _ = decide(conn, 3, datetime.fromisoformat(DECISION))
    player = proj[proj.week.eq(3) & proj.slot.eq("QB")].player_id.iloc[0]
    capture.record_action(
        conn, SEASON, 3, "QB", "submitted", player, {"player_name": "x"}, decision_id=early
    )
    events = capture.events(conn, SEASON)
    detail = json.loads(events[events.kind.eq("submitted")].detail.iloc[0])
    assert detail["linked_decision"] == early and detail["link_source"] == "named"
    assert early != late


def test_a_named_decision_that_does_not_exist_is_refused(tmp_path):
    """Minting a fresh id for a typo would write an event that reconstructs against
    nothing, and it would look exactly like a pick submitted without advice."""
    conn, _ = archived(tmp_path)
    decide(conn, 3, datetime.fromisoformat(DECISION))
    with pytest.raises(ValueError, match="No captured decision"):
        capture.record_action(
            conn, SEASON, 3, "QB", "submitted", "AAA-QB1", {}, decision_id="not-an-id"
        )


def test_an_unnamed_submission_still_records_how_it_was_linked(tmp_path):
    """The fallback is allowed -- a pick can be entered without running `recommend` --
    but a later reader must not have to guess which of the two links this was."""
    conn, _ = archived(tmp_path)
    decide(conn, 3, datetime.fromisoformat(DECISION))
    capture.record_action(conn, SEASON, 3, "QB", "submitted", "AAA-QB1", {})
    events = capture.events(conn, SEASON)
    detail = json.loads(events[events.kind.eq("submitted")].detail.iloc[0])
    assert detail["link_source"] == "latest advice"


def test_a_named_link_and_the_fallback_stay_distinguishable(tmp_path):
    """A deliberate `--decision` link and the most-recent-advice fallback are different
    claims about which decision a pick came from, and with two decisions in a week the
    fallback is whichever happened last."""
    conn, _ = archived(tmp_path)
    early, proj, _ = decide(conn, 3, datetime.fromisoformat(PRE_WEEK3))
    late, later, _ = decide(conn, 3, datetime.fromisoformat(DECISION))
    named = proj[proj.week.eq(3) & proj.slot.eq("QB")].player_id.iloc[0]
    inferred = later[later.week.eq(3) & later.slot.eq("RB")].player_id.iloc[0]
    capture.record_action(
        conn, SEASON, 3, "QB", "submitted", named, {"player_name": "x"}, decision_id=early
    )
    capture.record_action(conn, SEASON, 3, "RB", "submitted", inferred, {"player_name": "y"})

    events = capture.events(conn, SEASON)
    submitted = events[events.kind.eq("submitted")].set_index("slot")
    links = {slot: json.loads(detail) for slot, detail in submitted.detail.items()}
    assert links["QB"]["link_source"] == "named"
    assert links["QB"]["linked_decision"] == early
    assert links["RB"]["link_source"] == "latest advice"
    assert links["RB"]["linked_decision"] == late
