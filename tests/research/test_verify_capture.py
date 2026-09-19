"""Captured decisions checked after the fact: re-derived from their own record, and
replayed from the archive, by `pool research verify-capture`.

Each test states the failure it prevents. These checks began as Phase 3C's evidence
gates; the protocol closed without collecting and its window, floors and export were
retired, but a decision `pool week` writes down is still only worth keeping if it
reconstructs, and every replay still assumes the live path and the archive agree.
"""

from datetime import datetime

import pytest
from typer.testing import CliRunner

from pool import capture, config, snapshots
from pool import projections as P
from pool.cli import app
from pool.research import verify
from tests.support.decisions import (
    DECISION,
    POST_SUN,
    PRE_WEEK3,
    archived,
    decide,
    publish,
    staged,
)
from tests.support.season import SEASON, archive_all

BACKDATED = "2024-09-20T14:00:00+00:00"  # after every archived feed, before the decision


def test_two_decisions_in_one_week_are_two_decisions(tmp_path):
    """One decision per week is the replay's convention, not the pool's. Both events
    have to survive as separate decisions or the Thursday-to-Sunday news change is not
    in the record at all."""
    conn, _ = archived(tmp_path)
    early, _, _ = decide(conn, 3, datetime.fromisoformat(PRE_WEEK3))
    late, _, _ = decide(conn, 3, datetime.fromisoformat(DECISION))
    found = verify.decisions(conn, SEASON)
    assert list(found.decision_id) == [early, late]
    assert list(verify.decisions(conn, SEASON, week=4).decision_id) == []
    for decision_id in (early, late):
        assert verify.reconstruction(conn, decision_id)["ok"]


def test_a_captured_decision_matches_a_snapshot_replay_of_the_same_instant(tmp_path):
    """Every replay assumes the live path and the archive are the same function of the
    same inputs. Until now that was checked only inside one process on a staged
    database, never against a decision that had actually been captured."""
    conn, unplayed = archived(tmp_path)
    decision_id, _, _ = decide(conn, 3, datetime.fromisoformat(DECISION))
    # Everything after the decision arrives before the check is run.
    publish(conn, unplayed, "rest")
    for feed in ("player_stats", "touchdowns"):
        snapshots.archive(conn, SEASON, feed, observed_at=POST_SUN)

    result = verify.parity(conn, decision_id)
    assert result["ok"], result
    assert result["rows_missing_in_replay"] == 0 and result["rows_only_in_replay"] == 0
    assert result["differing_columns"] == []
    assert result["observations_match"] and result["advice_matches"]


def test_the_deadline_is_reconciled_rather_than_compared(tmp_path):
    """Live loading leaves the deadline out of `hard_eligible` and applies it in the
    recommender; snapshot replay folds it in. Comparing the two columns would report the
    contract as a defect, so the captured status is reconciled against the replay's
    exclusion instead -- which is the same fact, stated where it is true."""
    conn, _ = archived(tmp_path)
    decision_id, proj, _ = decide(conn, 3, datetime.fromisoformat(DECISION))

    stored = capture.load_surface(conn, capture.events(conn, SEASON).surface_hash.iloc[0])
    assert stored.hard_eligible.all()  # the live column, deadline and all
    assert (stored.decision_status == "deadline passed").any()
    frames = P.load_frames(conn, SEASON, 3, input_policy="snapshots", decision_at=DECISION)
    replayed = P.build_projections(frames, 3, "depth")
    assert not replayed.hard_eligible.all()  # the replay's column, deadline folded in

    result = verify.parity(conn, decision_id)
    assert result["deadline_reconciled"] and result["ok"]
    assert len(proj) == len(replayed)


def test_parity_names_the_column_that_moved(tmp_path, monkeypatch):
    """A silent divergence is the failure this exists to catch, so the report has to say
    which column moved and by how much rather than returning a bare false."""
    conn, _ = archived(tmp_path)
    decision_id, _, _ = decide(conn, 3, datetime.fromisoformat(DECISION))
    real = P.build_projections

    def nudged(frames, from_week=1, role_source=None):
        out = real(frames, from_week, role_source)
        out.loc[out.index[0], "lam"] = float(out.lam.iloc[0]) + 0.25
        return out

    monkeypatch.setattr(verify.projections, "build_projections", nudged)
    result = verify.parity(conn, decision_id)
    assert not result["ok"]
    assert result["differing_columns"] == ["lam"]
    assert result["largest_difference"] == pytest.approx(0.25)


def test_a_decision_made_while_an_earlier_week_was_unfinished_reaches_parity(tmp_path):
    """Finding 7 of the 2026-09-17 review, reproduced. A week-4 decision made on the
    Friday of week 3 read week 3 as it stood: the Thursday game final, the Sunday game
    unplayed. Replay demanded complete coverage of every earlier week, which the live path
    never did, so a faithfully archived decision could not be checked -- and because the
    archive is immutable, no later refresh could ever change that. Replay now rebuilds
    what the live path read. Research runs keep the stricter gate by asking for it."""
    conn, unplayed = archived(tmp_path)
    decision_id, _, _ = decide(conn, 4, datetime.fromisoformat(DECISION))
    # Everything after the decision arrives before the check is run.
    publish(conn, unplayed, "rest")
    for feed in ("player_stats", "touchdowns"):
        snapshots.archive(conn, SEASON, feed, observed_at=POST_SUN)
    result = verify.parity(conn, decision_id)
    assert result["ok"], result
    with pytest.raises(ValueError, match="Incomplete touchdown coverage"):
        P.load_frames(
            conn,
            SEASON,
            4,
            input_policy="snapshots",
            decision_at=DECISION,
            require_complete=True,
        )


def test_a_decision_whose_archive_was_never_written_cannot_be_verified(tmp_path):
    """An unverifiable decision is not a verified one: a season with no archive at all
    must not read as a season whose decisions all match replay."""
    conn, _ = staged(tmp_path)  # nothing archived
    decision_id, _, _ = decide(conn, 3, datetime.fromisoformat(DECISION))
    result = verify.parity(conn, decision_id)
    assert not result["ok"] and not result["checkable"]
    assert "archive cannot be resolved" in result["reason"]


def test_reconstruction_refuses_a_source_tree_it_was_not_captured_under(tmp_path, monkeypatch):
    """The recommender's behaviour is not carried by the recorded constants alone, so a
    reconstruction under different code is reported as unverified rather than as a
    matching decision."""
    conn, _ = archived(tmp_path)
    decision_id, _, _ = decide(conn, 3, datetime.fromisoformat(DECISION))
    assert verify.reconstruction(conn, decision_id)["ok"]
    capture._code_identity.cache_clear()
    monkeypatch.setattr(capture, "_code_identity", lambda: ("different", "different", "rev", False))
    result = verify.reconstruction(conn, decision_id)
    assert not result["ok"] and "fingerprint" in result["reason"]


def test_drift_outside_the_decision_path_is_reported_and_still_verifies(tmp_path, monkeypatch):
    """Tolerating that drift is the point of narrowing the fingerprint; saying nothing
    about it is not. A verified row indistinguishable from a clean one hides the tolerance
    -- and hides it in the CLI, which is itself outside the closure, so its silence is not
    evidence of anything."""
    conn, unplayed = archived(tmp_path)
    decision_id, _, _ = decide(conn, 3, datetime.fromisoformat(DECISION))
    publish(conn, unplayed, "rest")

    decision_hash = _moved(monkeypatch, decision_hash=None)  # only the whole tree moved

    rebuilt = verify.reconstruction(conn, decision_id)
    assert rebuilt["ok"] and verify.parity(conn, decision_id)["ok"]  # what did not move
    drift = rebuilt["drift"]
    assert drift["whole_tree_changed"] and drift["fingerprint_scope"] == "decision path"
    assert drift["recorded_code_hash"] == drift["current_code_hash"] == decision_hash
    assert not drift["code_hash_changed"]  # the enforced fingerprint held
    # What the decision recorded, not what is running now.
    recorded = capture.recorded_identity(conn, decision_id)
    assert recorded["decision_hash"] == decision_hash
    assert recorded["code_hash"] != capture._code_identity()[0]

    conn.commit()
    conn.close()
    result = CliRunner().invoke(
        app,
        [
            "research",
            "verify-capture",
            "--season",
            str(SEASON),
            "--db",
            str(tmp_path / "parity.db"),
        ],
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
    conn, unplayed = archived(tmp_path)
    decision_id, _, _ = decide(conn, 3, datetime.fromisoformat(DECISION))
    publish(conn, unplayed, "rest")
    _moved(monkeypatch, decision_hash="the decision path moved too")

    assert not verify.reconstruction(conn, decision_id)["ok"]  # refused without the override
    rebuilt = verify.reconstruction(conn, decision_id, allow_code_drift=True)
    matched = verify.parity(conn, decision_id, allow_code_drift=True)
    assert rebuilt["ok"] and matched["ok"]  # accepted, but only by the override
    assert rebuilt["drift"]["code_hash_changed"] and rebuilt["drift"]["whole_tree_changed"]

    conn.commit()
    conn.close()
    result = CliRunner().invoke(
        app,
        [
            "research",
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
    reporting its tolerated drift as a verified decision's would contradict the table."""
    conn, _ = staged(tmp_path)  # nothing archived, so there is nothing to replay against
    decision_id, _, _ = decide(conn, 3, datetime.fromisoformat(DECISION))
    _moved(monkeypatch, decision_hash=None)  # only the whole tree moved

    rebuilt = verify.reconstruction(conn, decision_id)
    assert rebuilt["ok"] and rebuilt["drift"]["whole_tree_changed"]
    assert not verify.parity(conn, decision_id)["ok"]
    conn.commit()
    conn.close()
    result = CliRunner().invoke(
        app,
        [
            "research",
            "verify-capture",
            "--season",
            str(SEASON),
            "--db",
            str(tmp_path / "parity.db"),
        ],
    )
    assert result.exit_code == 1 and "did not verify" in result.output
    assert "outside the decision path" not in result.output


def test_the_reconstruction_wrapper_keeps_the_constants_diagnostic(tmp_path):
    """The drift record said which settings had moved before the fingerprint work wrapped
    it. Returning only the fingerprint comparison drops that silently, and a diagnostic
    nothing reads yet is exactly the kind that rots unnoticed."""
    conn, _ = archived(tmp_path)
    decision_id, _, _ = decide(conn, 3, datetime.fromisoformat(DECISION))
    assert verify.reconstruction(conn, decision_id)["drift"]["constants_changed"] == {}

    with config.override(HOME_MULT=config.HOME_MULT + 0.5):
        drift = verify.reconstruction(conn, decision_id)["drift"]
    assert "HOME_MULT" in drift["constants_changed"]
    assert not drift["code_hash_changed"]  # a constant moving is not the source moving


def test_an_observation_archived_afterwards_is_not_agreement(tmp_path):
    """Resolving the captured side at check time compares the archive against itself.
    Both sides then move together, so an observation stamped before the decision but
    written after it changes what the replay reads while the comparison goes on
    reporting agreement -- and identical bytes are still two different readings."""
    conn, _ = archived(tmp_path)
    decision_id, _, _ = decide(conn, 3, datetime.fromisoformat(DECISION))
    assert verify.parity(conn, decision_id)["ok"]

    archive_all(conn, BACKDATED)  # same content, later stamp: it wins the replay's rule
    result = verify.parity(conn, decision_id)
    assert not result["observations_match"] and not result["ok"]
    assert result["differing_observations"]
    assert result["differing_columns"] == []  # nothing about the forecast moved
    assert "other observations" in result["reason"]


def test_parity_survives_a_constant_that_moved_after_the_decision(tmp_path):
    """`build_projections` reads its multipliers when it is called. Replaying under
    today's would report a configuration change as a live/archive divergence -- and would
    report it while reconstruction passed, because the stored surface already has the old
    multiplier in it."""
    conn, _ = archived(tmp_path)
    decision_id, _, _ = decide(conn, 3, datetime.fromisoformat(DECISION))
    assert verify.parity(conn, decision_id)["ok"]

    with config.override(HOME_MULT=config.HOME_MULT + 0.5):
        result = verify.parity(conn, decision_id)
    assert result["ok"], result
    assert result["differing_columns"] == []


def test_the_cli_lists_and_verifies_what_it_captured(tmp_path):
    """The read side existed only as a library API, so nothing a person could run said
    whether a captured decision was any good."""
    conn, unplayed = archived(tmp_path)
    decide(conn, 3, datetime.fromisoformat(DECISION))
    publish(conn, unplayed, "rest")
    conn.commit()
    conn.close()
    runner = CliRunner()
    listed = runner.invoke(
        app, ["research", "captures", "--season", str(SEASON), "--db", str(tmp_path / "parity.db")]
    )
    assert listed.exit_code == 0, listed.output
    assert "Fri 9/20 12:00PM ET" in listed.output  # when it was made, on the Eastern clock
    verified = runner.invoke(
        app,
        [
            "research",
            "verify-capture",
            "--season",
            str(SEASON),
            "--db",
            str(tmp_path / "parity.db"),
        ],
    )
    assert verified.exit_code == 0, verified.output
    assert "reconstruct and match replay" in verified.output


def test_the_cli_fails_loudly_when_a_decision_cannot_be_verified(tmp_path):
    """A verification command that exits zero on an unverifiable decision is worse than
    no command at all."""
    conn, _ = staged(tmp_path)  # nothing archived, so nothing to replay against
    decide(conn, 3, datetime.fromisoformat(DECISION))
    conn.commit()
    conn.close()
    result = CliRunner().invoke(
        app,
        [
            "research",
            "verify-capture",
            "--season",
            str(SEASON),
            "--db",
            str(tmp_path / "parity.db"),
        ],
    )
    assert result.exit_code == 1
    assert "did not verify" in result.output


def test_verifying_a_season_with_no_decisions_is_not_a_verification(tmp_path):
    """Exiting zero with nothing checked would tell a caller the season's decisions verify
    when there are none."""
    conn, _ = archived(tmp_path)
    conn.commit()
    conn.close()
    result = CliRunner().invoke(
        app,
        [
            "research",
            "verify-capture",
            "--season",
            str(SEASON),
            "--db",
            str(tmp_path / "parity.db"),
        ],
    )
    assert result.exit_code == 1
    assert "nothing was verified" in result.output


def test_the_listed_id_is_enough_to_name_a_decision(tmp_path):
    """`captures` prints twelve characters of each id, and asking for one of them back used
    to fail because only the full id matched."""
    import typer

    from pool.research import cli as research_cli

    conn, _ = archived(tmp_path)
    early, _, _ = decide(conn, 3, datetime.fromisoformat(PRE_WEEK3))
    late, _, _ = decide(conn, 3, datetime.fromisoformat(DECISION))
    assert research_cli._decision_id(conn, early[:12]) == early
    with pytest.raises(typer.Exit):
        research_cli._decision_id(conn, "")  # starts both
    with pytest.raises(typer.Exit):
        research_cli._decision_id(conn, "not-a-decision")
    conn.commit()
    conn.close()
    shown = CliRunner().invoke(
        app,
        ["research", "captures", "--decision", early[:12], "--db", str(tmp_path / "parity.db")],
    )
    assert shown.exit_code == 0, shown.output
    assert early in shown.output
    checked = CliRunner().invoke(
        app,
        [
            "research",
            "verify-capture",
            "--season",
            str(SEASON),
            "--decision",
            late[:12],
            "--db",
            str(tmp_path / "parity.db"),
        ],
    )
    assert checked.exit_code == 0, checked.output
    assert "All 1 captured decisions" in checked.output
