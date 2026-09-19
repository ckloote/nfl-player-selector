"""Rival-pick predictions: archived before the answer, scored only if they beat both deadlines."""

from datetime import UTC, datetime

import pandas as pd
import pytest
from typer.testing import CliRunner

from pool import capture, db, freshness, predictions, scoring, snapshots
from pool.cli import app
from tests.support.frames import proj_row
from tests.support.local import end, play
from tests.support.reports import (
    AFTER_KICKOFF,
    AFTER_REPORT,
    BEFORE_ALL,
    KICKOFF,
    REPORT_AT,
    WEEK1,
    WEEK2,
    history,
    report,
)

runner = CliRunner()


def _proj():
    return pd.DataFrame(
        [
            proj_row(pid, name, slot, 2, lam, kickoff="2026-09-20T13:00")
            for pid, name, slot, lam in (
                ("q1", "Quarter One", "QB", 5.0),
                ("q2", "Quarter Two", "QB", 4.0),
                ("r1", "Runner One", "RB", 3.0),
                ("f1", "Flex One", "FLEX", 2.0),
                ("f2", "Flex Two", "FLEX", 1.0),
            )
        ]
    )


@pytest.fixture
def pool(local):
    conn, path = local
    scoring.import_touchdowns(conn, 2026, pd.DataFrame([play(), end(), end("g2", 0, 0)]))
    history(conn)
    report(conn, 1, WEEK1, "2026-09-14T00:00:00+00:00")
    return conn, path


def _record(conn, at=BEFORE_ALL, week=2):
    payload = predictions.predict(conn, 2026, week, _proj())
    return predictions.archive(conn, 2026, week, payload, observed_at=at), payload


def test_archive_keeps_every_predictor_and_its_ranked_list(pool):
    """The ranked list, not its winner, is what can calibrate a distribution later --
    and it is the one thing that cannot be reconstructed after the fact."""
    conn, _ = pool
    observation, _ = _record(conn)
    back = predictions.archived(conn, 2026)
    assert [r["observation_id"] for r in back] == [observation]
    block = back[0]["payload"]["rivals"]
    assert set(block) == {"pat", "jamie"}, "my own row is not a rival"
    for rival in block.values():
        for byname in rival["slots"].values():
            assert set(byname) == set(predictions.rivals.PREDICTORS)
            assert all(len(ranked) >= 1 for ranked in byname.values())
            assert all("lam" in c and "player_name" in c for c in byname["greedy"])


def test_the_remaining_pool_at_prediction_time_travels_with_it(pool):
    """A prediction made against 41 quarterbacks is a different prediction from one made
    against 8, and by week 14 nobody will remember which it was."""
    conn, _ = pool
    _, payload = _record(conn)
    assert payload["rivals"]["pat"]["remaining"].keys() == {"QB", "RB", "FLEX"}
    assert payload["rivals"]["pat"]["used"] == ["f2", "q1"]
    assert payload["rivals"]["jamie"]["used"] == ["f1", "q2"]


def test_spending_is_counted_through_the_week_before_not_every_imported_week(pool):
    """Reports routinely arrive before their week is played. Counting week 2's own report
    here would let a week-2 prediction be built against a pool that already knows week 2 --
    the leak stage 2 closed in `remaining_counts`, arriving through a different door."""
    conn, _ = pool
    before = predictions.rival_states(conn, 2026, 2)
    report(conn, 2, WEEK2, REPORT_AT)
    assert predictions.rival_states(conn, 2026, 2) == before
    later = {s.entrant_id: s.used_ids for s in predictions.rival_states(conn, 2026, 3)}
    assert later["pat"] == frozenset({"q1", "f2", "q2", "r1", "f1"})


def test_greedy_respects_their_used_pool_and_naive_does_not(pool):
    conn, _ = pool
    _, payload = _record(conn)
    first = {
        (eid, slot, name): ranked[0]["player_id"]
        for eid, rival in payload["rivals"].items()
        for slot, byname in rival["slots"].items()
        for name, ranked in byname.items()
    }
    assert first[("pat", "QB", "greedy")] == "q2", "Pat spent Quarter One in week 1"
    assert first[("jamie", "QB", "greedy")] == "q1", "Jamie spent Quarter Two in week 1"
    assert first[("pat", "QB", "naive")] == first[("jamie", "QB", "naive")] == "q1"


def test_the_prediction_feed_is_invisible_to_capture_restore_and_freshness(pool):
    """The load-bearing claim of the storage design: a guess about an opponent is not a
    projection input, so it must not reach decision capture, freshness or replay."""
    conn, _ = pool
    stamp = "2026-09-15T00:00:00+00:00"
    for year in (2025, 2026):
        for feed in snapshots.TABLES:
            snapshots.archive(conn, year, feed, observed_at=stamp)
    before = capture.observed_inputs(conn, 2026, "2026-09-21T00:00:00+00:00")
    _record(conn)
    assert predictions.FEED not in snapshots.TABLES
    assert predictions.FEED not in freshness.FEEDS
    assert capture.observed_inputs(conn, 2026, "2026-09-21T00:00:00+00:00") == before
    _, warnings = freshness.report(conn, 2026, 2, datetime(2026, 9, 19, tzinfo=UTC))
    assert not any(predictions.FEED in w for w in warnings)
    restored, provenance = snapshots.restore(conn, 2026, stamp, 1)
    try:
        assert all(row["feed"] != predictions.FEED for row in provenance)
    finally:
        restored.close()


def test_first_kickoff_is_the_moment_the_week_stops_being_unobservable(pool):
    conn, _ = pool
    assert predictions.first_kickoff(conn, 2026, 2) == KICKOFF


@pytest.mark.parametrize(
    "at,reason",
    [(AFTER_KICKOFF, "after first kickoff"), (AFTER_REPORT, "after the report arrived")],
)
def test_a_prediction_that_missed_either_deadline_is_kept_but_not_scored(pool, at, reason):
    """Both deadlines, because a prediction made after the week's games ran on projections
    rebuilt from that week's own statistics even though the report had not landed."""
    conn, _ = pool
    report(conn, 2, WEEK2, REPORT_AT)
    _record(conn, at)
    scored, notes = predictions.score(conn, 2026)
    assert scored.empty
    assert any(reason in note["reason"] for note in notes)
    assert len(predictions.archived(conn, 2026)) == 1, "kept, not discarded"


def test_the_last_prediction_that_beat_both_deadlines_is_the_one_scored(pool):
    """Latest information state that could not have seen the answer."""
    conn, _ = pool
    _record(conn, "2026-09-18T00:00:00+00:00")
    _record(conn, BEFORE_ALL)
    _record(conn, AFTER_KICKOFF)
    report(conn, 2, WEEK2, REPORT_AT)
    chosen, _ = predictions.eligible(predictions.archived(conn, 2026), KICKOFF, REPORT_AT)
    assert datetime.fromisoformat(chosen["observed_at"]) == datetime.fromisoformat(BEFORE_ALL)


def test_hit_rates_separate_the_predictors(pool):
    conn, _ = pool
    _record(conn)
    report(conn, 2, WEEK2, REPORT_AT)
    scored, _ = predictions.score(conn, 2026)
    assert set(scored.entrant_id) == {"pat", "jamie"}, "only rivals are scored"
    rates = predictions.hit_rates(scored).set_index("predictor")
    assert rates.loc["greedy", "n"] == 6, "two rivals, three slots"
    assert rates.loc["greedy", "hits"] == 6
    assert rates.loc["naive", "hits"] == 4, "naive re-picks players they already spent"
    assert rates.loc["greedy", "hit_rate"] == 1.0


def test_an_unresolved_reported_name_is_unscorable_rather_than_a_miss(pool):
    """Scoring it as a miss would make an import defect look like a model defect, and
    those two rates are not the same number."""
    conn, _ = pool
    _record(conn)
    report(conn, 2, WEEK2, REPORT_AT)
    with conn:
        conn.execute(
            "UPDATE pool_picks SET player_id = NULL WHERE week = 2 AND entrant_id = 'pat' "
            "AND slot = 'QB'"
        )
    scored, notes = predictions.score(conn, 2026)
    assert not len(scored[scored.entrant_id.eq("pat") & scored.slot.eq("QB")])
    assert any("never resolved" in note["reason"] for note in notes)
    assert predictions.hit_rates(scored).set_index("predictor").loc["greedy", "n"] == 5


def test_a_reported_no_pick_is_not_called_an_unresolved_name(pool):
    """Neither is scorable, for different reasons: a no-pick is an answer no predictor
    offered, and an unresolved name is an answer nobody can read yet."""
    conn, _ = pool
    _record(conn)
    report(conn, 2, dict(WEEK2, Pat=("Quarter Two", "", "Flex One")), REPORT_AT)
    scored, notes = predictions.score(conn, 2026)
    assert not len(scored[scored.entrant_id.eq("pat") & scored.slot.eq("RB")])
    reasons = [note["reason"] for note in notes]
    assert "pat RB: no pick reported; nothing to score" in reasons
    assert not any("never resolved" in reason for reason in reasons)


def test_a_week_with_no_report_yet_is_reported_as_standing_unscored(pool):
    conn, _ = pool
    _record(conn)
    scored, notes = predictions.score(conn, 2026)
    assert scored.empty
    assert any("no report yet" in note["reason"] for note in notes)


def test_cli_records_and_scores(pool):
    conn, path = pool
    conn.close()
    dry = runner.invoke(
        app, ["research", "predict", "record", "--week", "2", "--dry-run", "--db", str(path)]
    )
    assert dry.exit_code == 0, dry.output
    assert "Dry run" in dry.output and "Quarter Two" in dry.output
    conn = db.connect(path)
    assert not predictions.archived(conn, 2026), "a dry run archives nothing"
    _record(conn)
    report(conn, 2, WEEK2, REPORT_AT)
    conn.close()
    scored = runner.invoke(app, ["research", "predict", "score", "--db", str(path)])
    assert scored.exit_code == 0, scored.output
    assert "greedy" in scored.output and "naive" in scored.output


def test_cli_says_so_when_the_prediction_can_no_longer_be_scored(pool):
    """The archive still happens -- a record that cannot be scored is still evidence --
    but a scripted run has to be told the observation was lost."""
    conn, path = pool
    report(conn, 2, WEEK2, REPORT_AT)
    conn.close()
    result = runner.invoke(app, ["research", "predict", "record", "--week", "2", "--db", str(path)])
    assert result.exit_code == 1
    assert "will not be scorable" in result.output
    conn = db.connect(path)
    assert len(predictions.archived(conn, 2026)) == 1, "archived anyway"
    conn.close()


def test_cli_refuses_to_predict_without_an_identity_and_archives_nothing(local):
    """Without an identity I am one of the rivals, and a prediction of my own picks is not
    evidence about anybody. The command says what sets it rather than archiving that."""
    conn, path = local
    scoring.import_touchdowns(conn, 2026, pd.DataFrame([play(), end(), end("g2", 0, 0)]))
    history(conn)
    report(conn, 1, WEEK1, "2026-09-14T00:00:00+00:00", me=None)
    conn.close()
    result = runner.invoke(app, ["research", "predict", "record", "--week", "2", "--db", str(path)])
    assert result.exit_code == 1
    assert "--me" in result.output
    conn = db.connect(path)
    assert not predictions.archived(conn, 2026) and not predictions.archived(conn, 2026, "pit")
    conn.close()


def _legacy_check(conn, week, observed_at):
    """A receipt as `report import --check` wrote them, before checks stopped archiving."""
    import json
    from dataclasses import asdict

    from pool import entrants

    observation = entrants.archive_report(
        conn,
        2026,
        week,
        f"draft {week} {observed_at}".encode(),
        source="draft.csv",
        observed_at=observed_at,
        coverage=dict(parsed=False, format="csv"),
    )
    errors = [] if week else ["Give --week or include a week column in the report"]
    receipt = entrants.ImportResult(observation, 2026, week, check=True, errors=errors)
    db.set_meta(conn, f"{entrants.META_PREFIX}{observation}", json.dumps(asdict(receipt)))
    return observation


def test_a_check_receipt_is_not_a_delivery(pool):
    """Old `--check` runs archived drafts. One naming a week closed that week's window
    before the real report arrived, and one naming none was reported, on every score, as
    a delivery nobody could place."""
    conn, _ = pool
    _legacy_check(conn, None, "2026-09-18T12:00:00+00:00")
    _legacy_check(conn, 2, "2026-09-18T13:00:00+00:00")
    arrivals, unattributed = predictions.report_arrivals(conn, 2026)
    assert unattributed == 0 and 2 not in arrivals
    _record(conn)  # after both drafts, before the real report
    report(conn, 2, WEEK2, REPORT_AT)
    arrivals, _ = predictions.report_arrivals(conn, 2026)
    assert datetime.fromisoformat(arrivals[2]) == datetime.fromisoformat(REPORT_AT)
    scored, notes = predictions.score(conn, 2026)
    assert len(scored), "a draft did not close the window on the prediction"
    assert not any("name no week" in note["reason"] for note in notes)


def test_cli_prints_notes_as_text_and_names_only_a_real_undated_delivery(pool):
    """The notes were printed with markup off and the markup still in them, so every
    one arrived wrapped in literal `[yellow]` or `[dim]`, and the PIT section called an
    unplaced report "Week None"."""
    from pool import entrants

    conn, path = pool
    _legacy_check(conn, None, "2026-09-18T12:00:00+00:00")
    undated = entrants.import_report(
        conn, 2026, None, b"entrant,slot,player_name\nPat,QB,Quarter One\n", source="undated.csv"
    )
    assert undated.observation_id and undated.errors, "a real delivery whose week is unknown"
    conn.close()
    result = runner.invoke(app, ["research", "predict", "score", "--db", str(path)])
    assert result.exit_code == 0, result.output
    assert "[yellow]" not in result.output and "[dim]" not in result.output
    assert "Week None" not in result.output
    assert result.output.count("1 archived report(s) name no week") == 2, "once per section"
