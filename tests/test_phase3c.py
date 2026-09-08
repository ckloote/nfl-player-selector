"""Phase 3C: the prospective identity baseline and the checks that make it evidence.

Each test states the failure it prevents. Nothing here evaluates a candidate: none
qualified in Phase 3B, so the only open question is whether a captured live decision is
faithful enough to describe at all.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from pool import capture, config, prospective
from pool import projections as P
from pool.cli import app
from tests.test_backtest import SEASON
from tests.test_phase2 import archive_all
from tests.test_phase3a import DECISION, POST_SUN, POST_THU, PRE_WEEK3, _decide, _publish, _staged

PROTOCOL = Path("experiments/phase3c-baseline.toml")


def _protocol(tmp_path, *, weeks="[3]", review=3, season=SEASON, drop=(), replace=()):
    """The shipped protocol, re-pointed at the fixture season.

    Built from the real file rather than a hand-written stub so a test that passes is
    evidence about the protocol that will actually be signed.
    """
    text = PROTOCOL.read_text()
    for old, new in (
        ("season = 2026", f"season = {season}"),
        ("collection_weeks = [1, 2, 3, 4, 5, 6]", f"collection_weeks = {weeks}"),
        ("review_after_week = 6", f"review_after_week = {review}"),
        *replace,
    ):
        assert old in text, old
        text = text.replace(old, new, 1)
    if drop:
        text = "\n".join(line for line in text.splitlines() if not line.startswith(tuple(drop)))
    path = tmp_path / "protocol.toml"
    path.write_text(text)
    return path


def _archived(tmp_path):
    """A week-3 decision on Friday, with the Thursday game played and archived."""
    conn, unplayed = _staged(tmp_path)
    archive_all(conn, PRE_WEEK3)
    _publish(conn, unplayed, "thursday")
    for feed in ("player_stats", "touchdowns"):
        from pool import snapshots

        snapshots.archive(conn, SEASON, feed, observed_at=POST_THU)
    return conn, unplayed


# --- the protocol is a preregistration or it is nothing ---------------------
@pytest.mark.parametrize("floor", prospective.REQUIRED_FLOORS)
def test_a_protocol_missing_a_floor_is_refused_before_anything_is_read(tmp_path, floor):
    """A threshold chosen once the captures are on screen is not a threshold. The
    collection has to refuse the protocol, not fall back to a module default."""
    path = _protocol(tmp_path, drop=[floor])
    with pytest.raises(ValueError, match="a missing floor blocks the collection"):
        prospective.resolve(path)


@pytest.mark.parametrize("value", ["true", '"1.0"', "nan"])
def test_a_floor_that_is_not_a_finite_number_is_not_a_floor(tmp_path, value):
    """`isinstance(True, int)` is true and so is `isinstance(nan, float)`; neither can
    be compared against a rate."""
    path = _protocol(tmp_path, replace=[("min_parity_rate = 1.0", f"min_parity_rate = {value}")])
    with pytest.raises(ValueError, match="missing|must lie in"):
        prospective.resolve(path)


def test_a_floor_outside_zero_to_one_is_refused(tmp_path):
    """A rate floor above one is unreachable and below zero is vacuous. Either freezes
    a condition that cannot do its job."""
    path = _protocol(tmp_path, replace=[("min_parity_rate = 1.0", "min_parity_rate = 1.5")])
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        prospective.resolve(path)


def test_editing_the_protocols_prose_starts_a_new_window(tmp_path):
    """The file carries the reasoning as well as the keys. Hashing the text puts the
    reasoning inside the window identity, so an edit cannot reinterpret a finished
    collection as though it had always said something else."""
    first = prospective.resolve(_protocol(tmp_path))
    (tmp_path / "amended").mkdir()
    second = prospective.resolve(
        _protocol(
            tmp_path / "amended",
            replace=[("# Phase 3C: prospective", "# Amended. Phase 3C: prospective")],
        )
    )

    assert first["floors"] == second["floors"]
    assert first["specification_sha256"] != second["specification_sha256"]


def test_a_review_cannot_widen_its_own_scope(tmp_path):
    """No candidate qualified in 3B. A protocol that quietly authorises a policy claim
    would make this window the promotion path the outcome note refused to open."""
    path = _protocol(
        tmp_path, replace=[('policy_claims = "none"', 'policy_claims = "achieved_tds"')]
    )
    with pytest.raises(ValueError, match="review declares unsupported"):
        prospective.resolve(path)


def test_the_declared_method_must_name_what_this_module_does(tmp_path):
    """3B refused a specification naming an estimand it did not compute. A protocol
    naming a calibrated model would describe a collection this module does not run."""
    path = _protocol(
        tmp_path, replace=[('calibrator = "identity"', 'calibrator = "cal-logaffine-position"')]
    )
    with pytest.raises(ValueError, match="calibrator must be 'identity'"):
        prospective.resolve(path)


def test_the_review_point_is_the_last_collection_week(tmp_path):
    """A review before the window closes reads a window that is still filling."""
    path = _protocol(tmp_path, weeks="[1, 2, 3]", review=2)
    with pytest.raises(ValueError, match="review_after_week must be"):
        prospective.resolve(path)


# --- the decision-event schedule -------------------------------------------
def test_the_weekly_events_are_the_deadlines_they_have_to_beat(tmp_path):
    """Two events in a week with a Thursday game; one in a week that starts on Sunday.
    Demanding a second event of a week that cannot hold one would count an impossible
    event as a miss."""
    conn, _ = _staged(tmp_path)
    both = prospective.week_events(conn, SEASON, 3)
    assert list(both) == ["thursday_deadline", "sunday_slate"]
    assert both["thursday_deadline"] < both["sunday_slate"]
    assert list(prospective.week_events(conn, SEASON, 4)) == ["sunday_slate"]


def test_a_decision_is_the_event_whose_deadline_it_beat(tmp_path):
    """Classification is by the deadline, not by the day of the week: a Friday decision
    is the Sunday event because Sunday's deadline is the next one it can still beat."""
    conn, _ = _staged(tmp_path)
    events = prospective.week_events(conn, SEASON, 3)
    assert prospective.classify_event(PRE_WEEK3, events) == "thursday_deadline"
    assert prospective.classify_event(DECISION, events) == "sunday_slate"
    # After the slate has locked, a decision is not one of the declared events at all.
    assert prospective.classify_event(POST_SUN, events) == "after_deadline"


def _wave(conn, game_id, kickoff, home, away):
    """Add a kickoff wave to the fixture week without touching its rosters.

    Teams nobody is rostered on, so the schedule gains a wave and the projection surface
    gains nothing. Four real waves would need eight team-slots and the fixture has four
    teams; what these tests are about is which deadline a decision beat.
    """
    with conn:
        conn.execute(
            "INSERT INTO games (game_id, season, week, game_type, kickoff, home_team, "
            "away_team, spread_line, total_line, kickoff_known) "
            "VALUES (?, ?, 3, 'REG', ?, ?, ?, -3.0, 45.0, 1)",
            (game_id, SEASON, kickoff, home, away),
        )


def _four_waves(conn):
    """The shape week 1 of the window actually has: Wed, Thu, Sun, Mon.

    Week 3 ships as Thursday plus Sunday. Every week of the 2026 window has a Monday
    game, and week 1 opens on a Wednesday and plays again on Thursday, so a two-wave
    schedule cannot show what classification does with the others.
    """
    _wave(conn, "g2024-3-EEE", "2024-09-18T20:15", "EEE", "FFF")  # Wednesday opener
    _wave(conn, "g2024-3-GGG", "2024-09-23T20:15", "GGG", "HHH")  # Monday closer


def test_every_kickoff_wave_is_a_decision_point_but_only_two_are_required(tmp_path):
    """A week offers a pick before each kickoff day. Declaring two events is a statement
    about what must be captured, not about how many decisions the week contains, and the
    two must stay exactly what they were."""
    conn, _ = _staged(tmp_path)
    _four_waves(conn)
    waves = prospective.week_waves(conn, SEASON, 3)
    assert list(waves) == ["thursday_deadline", "thursday_wave", "sunday_slate", "monday_wave"]
    assert sorted(waves.values()) == list(waves.values())  # in the order they fall
    # The required schedule is unchanged by the waves around it.
    assert list(prospective.week_events(conn, SEASON, 3)) == list(prospective.EVENTS)
    assert prospective.week_events(conn, SEASON, 3)["sunday_slate"] == waves["sunday_slate"]


def test_a_decision_is_classified_by_the_wave_it_beat_not_the_next_declared_event(tmp_path):
    """Filing a Thursday decision as the Sunday one describes a cadence nobody worked to,
    and filing a Monday-game decision as late describes a pick made before its own
    kickoff as one made after the slate."""
    conn, _ = _staged(tmp_path)
    _four_waves(conn)
    waves = prospective.week_waves(conn, SEASON, 3)
    assert prospective.classify_event("2024-09-18T18:00:00+00:00", waves) == "thursday_deadline"
    assert prospective.classify_event(PRE_WEEK3, waves) == "thursday_wave"
    assert prospective.classify_event(DECISION, waves) == "sunday_slate"
    assert prospective.classify_event(POST_SUN, waves) == "monday_wave"
    # Beating no deadline at all still has its own name.
    assert prospective.classify_event("2024-09-24T12:00:00+00:00", waves) == "after_deadline"


def test_an_undeclared_wave_is_reported_but_not_demanded(tmp_path):
    """A Monday decision is real evidence and belongs in the record, verified like any
    other. Requiring it would put a floor of 1.0 behind an event the protocol never
    scheduled, so one missed Monday would fail a window whose declared events were all
    captured."""
    conn, unplayed = _archived(tmp_path)
    _wave(conn, "g2024-3-GGG", "2024-09-23T20:15", "GGG", "HHH")  # Monday closer
    _decide(conn, 3, datetime.fromisoformat(PRE_WEEK3))  # thursday_deadline
    _decide(conn, 3, datetime.fromisoformat(DECISION))  # sunday_slate
    _publish(conn, unplayed, "rest")
    _decide(conn, 3, datetime.fromisoformat(POST_SUN))  # monday_wave, undeclared
    spec = prospective.resolve(_protocol(tmp_path))
    events = prospective.event_coverage(conn, spec)

    declared = events[events.scheduled.eq(1)].set_index("event")
    assert set(declared.index) == set(prospective.EVENTS)
    assert list(declared.captured) == [1, 1]
    extra = events[events.scheduled.eq(0)].set_index("event")
    assert list(extra.index) == ["monday_wave"]
    assert int(extra.loc["monday_wave", "captured"]) == 1
    assert extra.loc["monday_wave", "deadline"]  # it beat a real deadline of its own

    _table, _events, rates = prospective.fidelity(conn, spec)
    capture = rates[rates.measure.eq("event_capture")].iloc[0]
    assert capture.numerator == 2 and capture.denominator == 2 and capture.rate == 1.0
    # Every captured decision is still held to reconstruction and parity, declared or not.
    assert int(rates[rates.measure.eq("parity")].iloc[0].denominator) == 3


def test_the_protocol_must_declare_how_it_classifies_events(tmp_path):
    """The method travels in the file. A reader who could not tell whether a Monday
    decision was a wave or a late pick could not tell what the event table means."""
    path = _protocol(
        tmp_path,
        replace=[
            (
                "events_classify_by_kickoff_wave = true",
                "events_classify_by_kickoff_wave = false",
            )
        ],
    )
    with pytest.raises(ValueError, match="events_classify_by_kickoff_wave must be declared true"):
        prospective.resolve(path)


def test_a_missed_event_is_a_miss_not_a_smaller_denominator(tmp_path):
    """Two events were scheduled and one was captured. Reporting a rate of one over the
    events that happened to be captured would make a missed week look like a clean one."""
    conn, _ = _archived(tmp_path)
    _decide(conn, 3, datetime.fromisoformat(DECISION))
    spec = prospective.resolve(_protocol(tmp_path))
    events = prospective.event_coverage(conn, spec)
    assert set(events.event) == {"thursday_deadline", "sunday_slate"}
    assert int(events.scheduled.sum()) == 2 and int(events.captured.sum()) == 1
    _table, _events, rates = prospective.fidelity(conn, spec)
    capture_rate = rates[rates.measure.eq("event_capture")].iloc[0]
    assert capture_rate.rate == 0.5
    assert not capture_rate.meets_floor


def test_two_decisions_in_one_week_are_two_decisions(tmp_path):
    """One decision per week is the replay's convention, not the pool's. Both events
    have to survive as separate decisions or the Thursday-to-Sunday news change is not
    in the record at all."""
    conn, _ = _archived(tmp_path)
    early, _, _ = _decide(conn, 3, datetime.fromisoformat(PRE_WEEK3))
    late, _, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))
    spec = prospective.resolve(_protocol(tmp_path))
    found = prospective.decisions(conn, spec)
    assert list(found.decision_id) == [early, late]
    assert list(found.event) == ["thursday_deadline", "sunday_slate"]
    for decision_id in (early, late):
        assert prospective.reconstruction(conn, decision_id)["ok"]


# --- live and the archive are the same function or replay means nothing ----
def test_a_captured_decision_matches_a_snapshot_replay_of_the_same_instant(tmp_path):
    """Every replay assumes the live path and the archive are the same function of the
    same inputs. Until now that was checked only inside one process on a staged
    database, never against a decision that had actually been captured."""
    conn, unplayed = _archived(tmp_path)
    decision_id, _, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))
    # Everything after the decision arrives before the check is run.
    _publish(conn, unplayed, "rest")
    from pool import snapshots

    for feed in ("player_stats", "touchdowns"):
        snapshots.archive(conn, SEASON, feed, observed_at=POST_SUN)

    spec = prospective.resolve(_protocol(tmp_path))
    result = prospective.parity(conn, decision_id, spec)
    assert result["ok"], result
    assert result["rows_missing_in_replay"] == 0 and result["rows_only_in_replay"] == 0
    assert result["differing_columns"] == []
    assert result["observations_match"] and result["advice_matches"]


def test_the_deadline_is_reconciled_rather_than_compared(tmp_path):
    """Live loading leaves the deadline out of `hard_eligible` and applies it in the
    recommender; snapshot replay folds it in. Comparing the two columns would report the
    contract as a defect, so the captured status is reconciled against the replay's
    exclusion instead -- which is the same fact, stated where it is true."""
    conn, _ = _archived(tmp_path)
    decision_id, proj, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))
    spec = prospective.resolve(_protocol(tmp_path))

    stored = capture.load_surface(conn, capture.events(conn, SEASON).surface_hash.iloc[0])
    assert stored.hard_eligible.all()  # the live column, deadline and all
    assert (stored.decision_status == "deadline passed").any()
    frames = P.load_frames(conn, SEASON, 3, input_policy="snapshots", decision_at=DECISION)
    replayed = P.build_projections(frames, 3, "depth")
    assert not replayed.hard_eligible.all()  # the replay's column, deadline folded in

    result = prospective.parity(conn, decision_id, spec)
    assert result["deadline_reconciled"] and result["ok"]
    assert len(proj) == len(replayed)


def test_parity_names_the_column_that_moved(tmp_path, monkeypatch):
    """A silent divergence is the failure this exists to catch, so the report has to say
    which column moved and by how much rather than returning a bare false."""
    conn, _ = _archived(tmp_path)
    decision_id, _, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))
    spec = prospective.resolve(_protocol(tmp_path))
    real = P.build_projections

    def nudged(frames, from_week=1, role_source=None):
        out = real(frames, from_week, role_source)
        out.loc[out.index[0], "lam"] = float(out.lam.iloc[0]) + 0.25
        return out

    monkeypatch.setattr(prospective.projections, "build_projections", nudged)
    result = prospective.parity(conn, decision_id, spec)
    assert not result["ok"]
    assert result["differing_columns"] == ["lam"]
    assert result["largest_difference"] == pytest.approx(0.25)


def test_a_decision_whose_archive_was_never_written_cannot_be_verified(tmp_path):
    """An unverifiable decision is not a verified one. Dropping it from the denominator
    would let a window with no archive at all report a perfect parity rate."""
    conn, _ = _staged(tmp_path)  # nothing archived
    decision_id, _, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))
    spec = prospective.resolve(_protocol(tmp_path))
    result = prospective.parity(conn, decision_id, spec)
    assert not result["ok"] and not result["checkable"]
    assert "archive cannot be resolved" in result["reason"]
    _table, _events, rates = prospective.fidelity(conn, spec)
    parity_rate = rates[rates.measure.eq("parity")].iloc[0]
    assert parity_rate.denominator == 1 and parity_rate.rate == 0.0


def test_reconstruction_refuses_a_source_tree_it_was_not_captured_under(tmp_path, monkeypatch):
    """The recommender's behaviour is not carried by the recorded constants alone, so a
    reconstruction under different code is reported as unverified rather than as a
    matching decision."""
    conn, _ = _archived(tmp_path)
    decision_id, _, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))
    assert prospective.reconstruction(conn, decision_id)["ok"]
    capture._code_identity.cache_clear()
    monkeypatch.setattr(capture, "_code_identity", lambda: ("different", "different", "rev", False))
    result = prospective.reconstruction(conn, decision_id)
    assert not result["ok"] and "fingerprint" in result["reason"]


def test_drift_outside_the_decision_path_is_reported_and_still_verifies(tmp_path, monkeypatch):
    """Tolerating that drift is the point of narrowing the fingerprint; saying nothing
    about it is not. A verified row indistinguishable from a clean one hides the tolerance
    -- and hides it in the CLI and the exports, which are themselves outside the closure,
    so their silence is not evidence of anything."""
    conn, unplayed = _archived(tmp_path)
    _decide(conn, 3, datetime.fromisoformat(DECISION))
    _publish(conn, unplayed, "rest")
    spec = prospective.resolve(_protocol(tmp_path))

    decision_hash = _moved(monkeypatch, decision_hash=None)  # only the whole tree moved

    checks, _events, rates = prospective.fidelity(conn, spec)
    row = checks.iloc[0]
    assert row.reconstructs and row.parity  # the decision is a function of what did not move
    assert row.whole_tree_changed and row.fingerprint_scope == "decision path"
    assert row.recorded_fingerprint == row.current_fingerprint == decision_hash
    assert not row.fingerprint_changed  # the enforced fingerprint held
    assert rates[rates.measure.eq("reconstruction")].iloc[0].rate == 1.0

    out = prospective.export(conn, spec, tmp_path / "drift", log=lambda _x: None)
    assert "moved outside the decision path" in (out / "BASELINE.md").read_text()
    exported = pd.read_csv(out / "decisions.csv").iloc[0]
    assert bool(exported.whole_tree_changed) and exported.fingerprint_scope == "decision path"
    identity = json.loads((out / "identities.json").read_text())
    identity = identity["collection_provenance"]["decision_identities"][0]
    assert identity["decision_hash"] == decision_hash
    assert identity["enforced_fingerprint"] == "decision path"
    # What the decision recorded, not what is running now.
    assert identity["code_hash"] != capture._code_identity()[0]

    conn.commit()
    conn.close()
    result = CliRunner().invoke(
        app, ["verify-capture", "--season", str(SEASON), "--db", str(tmp_path / "parity.db")]
    )
    assert result.exit_code == 0, result.output
    assert "outside the decision path" in result.output


def _moved(monkeypatch, decision_hash):
    """Pretend the source moved, holding the enforced fingerprint at `decision_hash`."""
    _code, real, revision, dirty = capture._code_identity()
    capture._code_identity.cache_clear()
    fake = ("a module outside the closure moved", decision_hash or real, revision, dirty)
    monkeypatch.setattr(capture, "_code_identity", lambda: fake)
    return real


def test_an_overridden_fingerprint_is_not_reported_as_harmless_drift(tmp_path, monkeypatch):
    """`--allow-code-drift` lets a decision pass with the enforced fingerprint moved. That
    is an override, not the narrow fingerprint doing its job, and reporting it as code
    moving outside the decision path asserts the opposite of what happened."""
    conn, unplayed = _archived(tmp_path)
    _decide(conn, 3, datetime.fromisoformat(DECISION))
    _publish(conn, unplayed, "rest")
    spec = prospective.resolve(_protocol(tmp_path))
    _moved(monkeypatch, decision_hash="the decision path moved too")

    checks, _events, _rates = prospective.fidelity(conn, spec, allow_code_drift=True)
    row = checks.iloc[0]
    assert row.reconstructs and row.parity  # accepted, but only by the override
    assert row.fingerprint_changed and row.whole_tree_changed
    out = prospective.export(
        conn, spec, tmp_path / "override", allow_code_drift=True, log=lambda _x: None
    )
    note = (out / "BASELINE.md").read_text()
    assert "fingerprint check was overridden" in note
    assert "moved outside the decision path" not in note

    conn.commit()
    conn.close()
    result = CliRunner().invoke(
        app,
        [
            "verify-capture",
            "--season",
            str(SEASON),
            "--db",
            str(tmp_path / "parity.db"),
            "--allow-code-drift",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "overridden" in result.output
    assert "outside the decision path" not in result.output


def test_a_decision_that_failed_parity_is_not_called_verified(tmp_path, monkeypatch):
    """Reconstruction alone is not verification. A decision whose archive was never
    written re-derives its advice from its own stored surface and still fails parity, and
    counting it as verified would contradict the table printed directly above it."""
    conn, _ = _staged(tmp_path)  # nothing archived, so there is nothing to replay against
    _decide(conn, 3, datetime.fromisoformat(DECISION))
    spec = prospective.resolve(_protocol(tmp_path))
    _moved(monkeypatch, decision_hash=None)  # only the whole tree moved

    checks, _events, _rates = prospective.fidelity(conn, spec)
    row = checks.iloc[0]
    assert row.reconstructs and not row.parity and row.whole_tree_changed
    out = prospective.export(conn, spec, tmp_path / "unverified", log=lambda _x: None)
    assert "moved outside the decision path" not in (out / "BASELINE.md").read_text()
    exported = pd.read_csv(out / "decisions.csv").iloc[0]
    assert bool(exported.whole_tree_changed) and not bool(exported.fingerprint_changed)


def test_the_reconstruction_wrapper_keeps_the_constants_diagnostic(tmp_path):
    """The drift record said which settings had moved before the fingerprint work wrapped
    it. Returning only the fingerprint comparison drops that silently, and a diagnostic
    nothing reads yet is exactly the kind that rots unnoticed."""
    conn, _ = _archived(tmp_path)
    decision_id, _, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))
    assert prospective.reconstruction(conn, decision_id)["drift"]["constants_changed"] == {}

    with config.override(HOME_MULT=config.HOME_MULT + 0.5):
        drift = prospective.reconstruction(conn, decision_id)["drift"]
    assert "HOME_MULT" in drift["constants_changed"]
    assert not drift["code_hash_changed"]  # a constant moving is not the source moving


# --- submissions attach to the decision they came from ---------------------
def test_a_submission_links_to_the_decision_it_names_not_the_latest(tmp_path):
    """With a Thursday and a Sunday decision in one week, "the most recent advice for
    this slot" is whichever happened last, which is not the same thing as the one the
    pick came from."""
    conn, _ = _archived(tmp_path)
    early, proj, _ = _decide(conn, 3, datetime.fromisoformat(PRE_WEEK3))
    late, _, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))
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
    conn, _ = _archived(tmp_path)
    _decide(conn, 3, datetime.fromisoformat(DECISION))
    with pytest.raises(ValueError, match="No captured decision"):
        capture.record_action(
            conn, SEASON, 3, "QB", "submitted", "AAA-QB1", {}, decision_id="not-an-id"
        )


def test_an_unnamed_submission_still_records_how_it_was_linked(tmp_path):
    """The fallback is allowed -- a pick can be entered without running `recommend` --
    but a later reader must not have to guess which of the two links this was."""
    conn, _ = _archived(tmp_path)
    _decide(conn, 3, datetime.fromisoformat(DECISION))
    capture.record_action(conn, SEASON, 3, "QB", "submitted", "AAA-QB1", {})
    events = capture.events(conn, SEASON)
    detail = json.loads(events[events.kind.eq("submitted")].detail.iloc[0])
    assert detail["link_source"] == "latest advice"


# --- the described surface -------------------------------------------------
def _described(conn, tmp_path, **kwargs):
    spec = prospective.resolve(_protocol(tmp_path, **kwargs))
    return spec, prospective.frame(conn, spec)


def test_eligibility_reconciles_the_deadline_before_describing_a_population(tmp_path):
    """A cell whose kickoff has passed is not a candidate you could pick, so it is not
    an eligible population. It keeps its positive rate and is counted as an exclusion,
    the way Phase 3A counts an undecidable cell."""
    conn, _ = _archived(tmp_path)
    _decide(conn, 3, datetime.fromisoformat(DECISION))
    _spec, rows = _described(conn, tmp_path)
    elapsed = rows[rows.decision_status.eq("deadline passed")]
    assert len(elapsed)
    assert not elapsed.hard_eligible.any()
    assert (elapsed.lam > 0).any()  # the forecast was usable; the cell was not
    zeros = prospective.diagnostics.zero_accounting(rows, prospective.population_masks(rows))
    undecidable = zeros[zeros.population.eq("excluded_undecidable")]
    assert int(undecidable.n.sum()) == len(elapsed)


def test_the_declared_populations_are_built_from_events_not_outcomes(tmp_path):
    """`recommended` and `submitted` are decision-week facts read off the log. A
    population defined by what happened afterwards would select on the outcome."""
    conn, _ = _archived(tmp_path)
    decision_id, proj, advice = _decide(conn, 3, datetime.fromisoformat(DECISION))
    picked = advice[0].recommended.player_id
    capture.record_action(
        conn, SEASON, 3, "QB", "submitted", picked, {"player_name": "x"}, decision_id=decision_id
    )
    _spec, rows = _described(conn, tmp_path)
    masks = prospective.population_masks(rows)
    assert list(masks) == list(prospective.POPULATIONS)
    # Decision-week facts, so nothing beyond horizon 0 carries them.
    for name in ("recommended", "submitted"):
        assert rows.loc[rows.lead_horizon.gt(0), name].isna().all()
        assert masks[name].any()
    # Depletion is a decision-week fact, so the two partition the current-week eligible
    # rows and say nothing about the future surface -- which is the point of blanking it.
    current = rows.lead_horizon.eq(0)
    assert not (masks["available"] & masks["depleted"]).any()
    assert (masks["available"] | masks["depleted"]).sum() == (masks["all_eligible"] & current).sum()
    assert (masks["all_eligible"] & ~current).any()
    assert masks["common_top1"].sum() <= masks["common_top3"].sum()


def test_an_outcome_missing_because_the_week_is_unplayed_is_missing_not_zero(tmp_path):
    """A week still being played has no outcomes, and reading absence as zero there
    would score the feed rather than the model."""
    conn, unplayed = _archived(tmp_path)
    _decide(conn, 3, datetime.fromisoformat(DECISION))
    spec, rows = _described(conn, tmp_path)
    before = prospective.outcome_coverage(rows, spec)
    assert not rows.outcome_known.any()
    assert float(before[before.population.eq("horizon_0_in_window")].coverage.iloc[0]) == 0.0

    _publish(conn, unplayed, "rest")
    _spec, after_rows = _described(conn, tmp_path)
    after = prospective.outcome_coverage(after_rows, spec)
    assert float(after[after.population.eq("horizon_0_in_window")].coverage.iloc[0]) == 1.0


def test_a_target_week_outside_the_window_is_unsettled_not_missing(tmp_path):
    """A week-3 decision forecasts week 4 too. At a week-3 review that week has not been
    played, and counting it as missing coverage would measure the calendar."""
    conn, _ = _archived(tmp_path)
    _decide(conn, 3, datetime.fromisoformat(DECISION))
    spec, rows = _described(conn, tmp_path)
    coverage = prospective.outcome_coverage(rows, spec)
    outside = coverage[coverage.population.eq("future_target_week_outside_window")].iloc[0]
    assert outside.rows == int((rows.lead_horizon.gt(0) & rows.hard_eligible).sum())
    assert pd.isna(outside.floor) and pd.isna(outside.meets_floor)


# --- the export ------------------------------------------------------------
def test_the_export_refuses_a_window_with_nothing_in_it(tmp_path):
    """Before the first decision there is nothing to describe, and an empty export that
    reported floors as met would be the worst possible answer."""
    conn, _ = _archived(tmp_path)
    spec = prospective.resolve(_protocol(tmp_path))
    with pytest.raises(ValueError, match="No captured decisions"):
        prospective.export(conn, spec, tmp_path / "out", log=lambda _x: None)


def test_the_export_reads_its_floors_from_the_protocol(tmp_path):
    """The floors travel in the dated file. An export that fell back to a module
    constant would report a threshold nobody signed."""
    conn, unplayed = _archived(tmp_path)
    _decide(conn, 3, datetime.fromisoformat(PRE_WEEK3))
    late, _later, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))
    _spec, surface = _described(conn, tmp_path)
    # One pick the population describes and one the deadline reconciled out of it.
    picked = _cell(surface, "QB", decision=late)
    elapsed = _cell(surface, "RB", eligible=False, decision=late)
    for slot, player in (("QB", picked), ("RB", elapsed)):
        capture.record_action(
            conn, SEASON, 3, slot, "submitted", player, {"player_name": slot}, decision_id=late
        )
    _publish(conn, unplayed, "rest")
    path = _protocol(
        tmp_path,
        replace=[("min_outcome_coverage_future = 0.95", "min_outcome_coverage_future = 0.5")],
    )
    spec = prospective.resolve(path)
    out = prospective.export(conn, spec, tmp_path / "out", log=lambda _x: None)

    rates = pd.read_csv(out / "fidelity.csv")
    assert set(rates.measure) == {"event_capture", "reconstruction", "parity"}
    assert float(rates[rates.measure.eq("event_capture")].floor.iloc[0]) == 1.0
    coverage = pd.read_csv(out / "outcome-coverage.csv")
    future = coverage[coverage.population.eq("target_week_in_window")].iloc[0]
    assert float(future.floor) == 0.5
    note = (out / "BASELINE.md").read_text()
    assert "Identity only" in note and "not touchdowns gained or lost by waiting" in note
    submissions = pd.read_csv(out / "submissions.csv")
    assert list(submissions.columns)[1:] == list(prospective.SUBMISSION_COLUMNS)
    submissions = submissions.set_index("slot")
    assert submissions.loc["QB", "status"] == "matched"
    assert submissions.loc["QB", "link_source"] == "named"
    assert submissions.loc["RB", "status"] == "not eligible"
    assert submissions.loc["RB", "exclusion"] == "deadline passed"
    assert "### Submissions" in note and "1 of 2 recorded submissions" in note
    assert "1 deadline passed" in note and "outside the eligible population" in note
    identities = json.loads((out / "identities.json").read_text())
    assert (
        identities["collection_provenance"]["specification_sha256"] == spec["specification_sha256"]
    )
    assert identities["collection_provenance"]["future_discounts"] == [config.FUTURE_DISCOUNT]
    assert identities["diagnostic_provenance"]["populations"] == list(prospective.POPULATIONS)


def test_the_cli_lists_and_verifies_what_it_captured(tmp_path):
    """The read side existed only as a library API, so nothing a person could run said
    whether the evidence being collected was any good."""
    conn, unplayed = _archived(tmp_path)
    _decide(conn, 3, datetime.fromisoformat(DECISION))
    _publish(conn, unplayed, "rest")
    conn.commit()
    conn.close()
    runner = CliRunner()
    listed = runner.invoke(
        app, ["captures", "--season", str(SEASON), "--db", str(tmp_path / "parity.db")]
    )
    assert listed.exit_code == 0, listed.output
    assert "sunday_slate" in listed.output
    verified = runner.invoke(
        app, ["verify-capture", "--season", str(SEASON), "--db", str(tmp_path / "parity.db")]
    )
    assert verified.exit_code == 0, verified.output
    assert "reconstruct and match replay" in verified.output


def test_the_cli_fails_loudly_when_a_decision_cannot_be_verified(tmp_path):
    """A verification command that exits zero on an unverifiable decision is worse than
    no command at all."""
    conn, _ = _staged(tmp_path)  # nothing archived, so nothing to replay against
    _decide(conn, 3, datetime.fromisoformat(DECISION))
    conn.commit()
    conn.close()
    result = CliRunner().invoke(
        app, ["verify-capture", "--season", str(SEASON), "--db", str(tmp_path / "parity.db")]
    )
    assert result.exit_code == 1
    assert "did not verify" in result.output


def test_the_shipped_protocol_resolves_as_written():
    """The file that will be signed has to be the file the collection can run."""
    spec = prospective.resolve(PROTOCOL)
    assert spec["season"] == 2026
    assert spec["collection_weeks"] == [1, 2, 3, 4, 5, 6]
    assert spec["review_after_week"] == 6
    assert spec["floors"]["min_reconstruction_rate"] == 1.0
    assert list(spec["populations"]) == list(prospective.POPULATIONS)
    assert config.ROLE_SOURCE == spec["role_source"]


# --- the review's findings, each as the regression it prevents ---------------
BACKDATED = "2024-09-20T14:00:00+00:00"  # after every archived feed, before the decision


def test_an_observation_archived_afterwards_is_not_agreement(tmp_path):
    """Resolving the captured side at check time compares the archive against itself.
    Both sides then move together, so an observation stamped before the decision but
    written after it changes what the replay reads while the comparison goes on
    reporting agreement -- and identical bytes are still two different readings."""
    conn, _ = _archived(tmp_path)
    decision_id, _, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))
    spec = prospective.resolve(_protocol(tmp_path))
    assert prospective.parity(conn, decision_id, spec)["ok"]

    archive_all(conn, BACKDATED)  # same content, later stamp: it wins the replay's rule
    result = prospective.parity(conn, decision_id, spec)
    assert not result["observations_match"] and not result["ok"]
    assert result["differing_observations"]
    assert result["differing_columns"] == []  # nothing about the forecast moved
    assert "other observations" in result["reason"]


def test_a_pick_replaced_then_removed_is_submitted_by_nobody(tmp_path):
    """`record` writes a correction naming the player it replaced and `unrecord` writes
    one that removes it. Reading only the submissions leaves every player ever entered in
    the population, so a slot corrected once and then emptied contributes two forecast
    rows for a pick that is in nobody's lineup."""
    conn, _ = _archived(tmp_path)
    decision_id, proj, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))
    qbs = list(proj[proj.week.eq(3) & proj.slot.eq("QB")].player_id)
    first, second = qbs[0], qbs[1]
    for slot, kind, player, detail in (
        ("QB", "submitted", first, {"player_name": "first"}),
        ("QB", "correction", first, {"replaced_by": second}),
        ("QB", "submitted", second, {"player_name": "second"}),
    ):
        capture.record_action(conn, SEASON, 3, slot, kind, player, detail, decision_id=decision_id)
    # `unrecord` links through the fallback, so the withdrawal can arrive under another id.
    capture.record_action(conn, SEASON, 3, "QB", "correction", second, {"removed": True})

    _spec, rows = _described(conn, tmp_path)
    assert not prospective.population_masks(rows)["submitted"].any()
    links = rows.attrs["submission_links"]
    assert dict(zip(links.player_id, links.status, strict=True)) == {
        first: "superseded",
        second: "withdrawn",
    }


def test_a_submission_the_decision_never_forecast_is_reported_not_attributed(tmp_path):
    """A link that names an existing decision is not yet a link to a described row. A
    player absent from that capture produces an apparently successful attribution that
    contributes no forecast, and an audit without the player identity cannot find it."""
    conn, _ = _archived(tmp_path)
    decision_id, _, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))
    capture.record_action(
        conn, SEASON, 3, "QB", "submitted", "NOT-ON-THE-SURFACE", {}, decision_id=decision_id
    )
    _spec, rows = _described(conn, tmp_path)
    assert not prospective.population_masks(rows)["submitted"].any()
    link = rows.attrs["submission_links"].iloc[0]
    assert link.player_id == "NOT-ON-THE-SURFACE"
    assert link.status == "unmatched surface" and link.attributed == decision_id


def test_parity_survives_a_constant_that_moved_after_the_decision(tmp_path):
    """`build_projections` reads its multipliers when it is called. Replaying under
    today's would report a configuration change as a live/archive divergence -- and would
    report it while reconstruction passed, because the stored surface already has the old
    multiplier in it."""
    conn, _ = _archived(tmp_path)
    decision_id, _, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))
    spec = prospective.resolve(_protocol(tmp_path))
    assert prospective.parity(conn, decision_id, spec)["ok"]

    with config.override(HOME_MULT=config.HOME_MULT + 0.5):
        result = prospective.parity(conn, decision_id, spec)
    assert result["ok"], result
    assert result["differing_columns"] == []


def test_the_planning_value_uses_the_discount_the_decision_was_made_under(tmp_path):
    """The planning quantity is what the optimizer compares. Recomputing it under a
    discount that moved after the capture reports a different planning value for a
    decision that never changed."""
    conn, _ = _archived(tmp_path)
    _decide(conn, 3, datetime.fromisoformat(DECISION))
    _spec, rows = _described(conn, tmp_path)
    future = rows.lead_horizon.gt(0)
    assert (rows.planning_lam[future] < rows.lam[future]).any()  # the discount is doing work

    with config.override(FUTURE_DISCOUNT=0.5):
        _moved_spec, moved = _described(conn, tmp_path)
    pd.testing.assert_series_equal(rows.planning_lam, moved.planning_lam)


def test_a_named_link_and_the_fallback_stay_distinguishable(tmp_path):
    """A deliberate `--decision` link and the most-recent-advice fallback are different
    claims about which decision a pick came from, and with two decisions in a week the
    fallback is whichever happened last."""
    conn, _ = _archived(tmp_path)
    early, proj, _ = _decide(conn, 3, datetime.fromisoformat(PRE_WEEK3))
    late, later, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))
    named = proj[proj.week.eq(3) & proj.slot.eq("QB")].player_id.iloc[0]
    inferred = later[later.week.eq(3) & later.slot.eq("RB")].player_id.iloc[0]
    capture.record_action(
        conn, SEASON, 3, "QB", "submitted", named, {"player_name": "x"}, decision_id=early
    )
    capture.record_action(conn, SEASON, 3, "RB", "submitted", inferred, {"player_name": "y"})

    _spec, rows = _described(conn, tmp_path)
    links = rows.attrs["submission_links"].set_index("slot")
    assert links.loc["QB", "link_source"] == "named"
    assert links.loc["QB", "attributed"] == early
    assert links.loc["RB", "link_source"] == "latest advice"
    assert links.loc["RB", "attributed"] == late
    assert links.reader_fallback.isna().all()


def _cell(rows, slot, *, eligible=True, n=0, decision=None):
    """A player the decision forecast in that slot, on the eligible or the elapsed side.

    Derived from the described frame rather than named, so the fixture's kickoff times
    stay the fixture's business and the test says which side of the deadline it wants.
    Which decision matters: a player eligible on Wednesday's surface has had his kickoff
    by Friday, so a frame holding two decisions has him on both sides at once.
    """
    at = rows[rows.lead_horizon.eq(0) & rows.slot.eq(slot)]
    if decision is not None:
        at = at[at.decision_id.eq(decision)]
    side = at[at.hard_eligible] if eligible else at[at.decision_status.eq("deadline passed")]
    assert len(side) > n, (slot, eligible)
    return side.player_id.iloc[n]


def test_verifying_an_empty_window_is_not_a_verification(tmp_path):
    """Completeness is checked by the protocol's capture floor, which a caller running
    this command on its own never reaches. Exiting zero here would tell that caller the
    window is sound when the window is empty."""
    conn, _ = _archived(tmp_path)
    conn.commit()
    conn.close()
    result = CliRunner().invoke(
        app, ["verify-capture", "--season", str(SEASON), "--db", str(tmp_path / "parity.db")]
    )
    assert result.exit_code == 1
    assert "nothing was verified" in result.output


# --- the follow-up review's findings ----------------------------------------
def test_a_submission_the_deadline_ruled_out_is_not_described(tmp_path):
    """Being on the credited decision's surface is not being described. The live path
    leaves the deadline out of eligibility and the population folds it in, so a pick made
    against an earlier decision matches a later decision's surface with its kickoff
    already gone -- and the audit would report it described while the population that is
    supposed to describe it holds no row at all."""
    conn, _ = _archived(tmp_path)
    decision_id, _, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))
    _spec, surface = _described(conn, tmp_path)
    elapsed = _cell(surface, "QB", eligible=False)
    capture.record_action(
        conn, SEASON, 3, "QB", "submitted", elapsed, {"player_name": "x"}, decision_id=decision_id
    )

    _spec, rows = _described(conn, tmp_path)
    assert not prospective.population_masks(rows)["submitted"].any()
    link = rows.attrs["submission_links"].iloc[0]
    assert link.player_id == elapsed and link.attributed == decision_id
    assert link.status == "not eligible" and link.exclusion == "deadline passed"


def test_the_described_count_is_population_membership_not_a_claim_beside_it(tmp_path):
    """The note counts `matched` as described, so `matched` has to be read off the
    population itself. Two submissions the decision forecast, one of them past its
    kickoff: any count that does not come from the mask reports two."""
    conn, _ = _archived(tmp_path)
    decision_id, _, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))
    _spec, surface = _described(conn, tmp_path)
    standing, elapsed = _cell(surface, "QB"), _cell(surface, "RB", eligible=False)
    for slot, player in (("QB", standing), ("RB", elapsed)):
        capture.record_action(
            conn,
            SEASON,
            3,
            slot,
            "submitted",
            player,
            {"player_name": slot},
            decision_id=decision_id,
        )

    _spec, rows = _described(conn, tmp_path)
    links = rows.attrs["submission_links"]
    described = prospective.population_masks(rows)["submitted"]
    assert int(links.status.eq("matched").sum()) == int(described.sum()) == 1
    assert set(rows.player_id[described]) == {standing}
    assert sorted(links.status) == ["matched", "not eligible"]


def test_an_action_outside_the_window_is_not_read_into_it(tmp_path):
    """The protocol declares the collection weeks and the captures are restricted to
    them, so reading the whole season reports a correctly linked pick from another week
    as this window's attribution failure -- and lets a submission made after the window
    closed move the totals of a baseline regenerated afterwards."""
    conn, unplayed = _archived(tmp_path)
    inside, _, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))
    _publish(conn, unplayed, "rest")
    outside, later, _ = _decide(conn, 4, datetime.fromisoformat(POST_SUN))
    _spec, surface = _described(conn, tmp_path)
    capture.record_action(
        conn,
        SEASON,
        3,
        "QB",
        "submitted",
        _cell(surface, "QB"),
        {"player_name": "x"},
        decision_id=inside,
    )
    # Correctly linked, and none of this window's business.
    capture.record_action(
        conn,
        SEASON,
        4,
        "QB",
        "submitted",
        later[later.week.eq(4) & later.slot.eq("QB")].player_id.iloc[0],
        {"player_name": "y"},
        decision_id=outside,
    )

    _spec, rows = _described(conn, tmp_path, weeks="[3]")
    links = rows.attrs["submission_links"]
    assert set(links.week) == {3}
    assert list(links.status) == ["matched"] and links.attributed.iloc[0] == inside


def test_recording_the_same_player_again_keeps_the_first_attribution(tmp_path):
    """`record` writes a correction only when the player changes but writes the
    submission either way, so re-entering an unchanged lineup against the Sunday decision
    overwrites the Thursday entry. The deliberate link disappears and the audit reports a
    pick that was only ever attributed by fallback."""
    conn, _ = _archived(tmp_path)
    early, _, _ = _decide(conn, 3, datetime.fromisoformat(PRE_WEEK3))
    late, _, _ = _decide(conn, 3, datetime.fromisoformat(DECISION))
    _spec, surface = _described(conn, tmp_path)
    standing = _cell(surface, "QB", decision=late)
    for decision_id in (early, late):
        capture.record_action(
            conn,
            SEASON,
            3,
            "QB",
            "submitted",
            standing,
            {"player_name": "x"},
            decision_id=decision_id,
        )

    _spec, rows = _described(conn, tmp_path)
    links = rows.attrs["submission_links"]
    assert len(links) == 2 and set(links.player_id) == {standing}
    by_status = dict(zip(links.status, links.attributed, strict=True))
    assert by_status == {"resubmitted": early, "matched": late}
    # One pick still stands, so re-entering it does not double the population.
    assert int(prospective.population_masks(rows)["submitted"].sum()) == 1
