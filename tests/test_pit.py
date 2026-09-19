"""The prospective PIT: a distribution pinned before the week, scored after it."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from pool import capture, config, db, pit, predictions, simulate
from pool.cli import app
from tests.support import reports as log

runner = CliRunner()


def _proj(conn, week=1):
    from pool import projections

    return projections.projections_for(conn, 2026, from_week=week)


def test_the_commitment_stores_its_outcomes_and_reads_them_back(reported):
    """What makes this evidence rather than decoration: the outcomes themselves, drawn
    before kickoff, beside the seed and frame that produced them. Reading them back gives
    exactly what was drawn, however many times it is read."""
    conn, _ = reported
    proj = _proj(conn)
    pit.commit(conn, 2026, 1, proj, observed_at="2026-09-09T12:00:00+00:00")
    record = pit.archived(conn, 2026)[0]
    payload = record["payload"]
    assert payload["schema_version"] == 2 and payload["draws_hash"]
    assert set(payload) >= {"seed", "sims", "surface_hash", "params", "week"}
    drawn = simulate.sample(
        proj[proj.week.eq(1)].reset_index(drop=True),
        [1],
        sims=payload["sims"],
        seed=payload["seed"],
    )
    stored = pit._draws(conn, record)
    assert np.array_equal(stored.values, drawn.values) and stored.index == drawn.index
    assert np.array_equal(pit._draws(conn, record).values, stored.values)


def test_the_frame_is_pinned_by_hash_so_a_later_refresh_cannot_move_it(reported):
    """A distribution defined by 'whatever the projections say' is not a commitment. The
    surface is content-addressed, so a re-forecast next week rebuilds the old one exactly."""
    conn, _ = reported
    pit.commit(conn, 2026, 1, _proj(conn), observed_at="2026-09-09T12:00:00+00:00")
    record = pit.archived(conn, 2026)[0]
    before = pit._draws(conn, record).values.sum()
    conn.execute("UPDATE rosters SET status = 'ACT' WHERE season = 2026")
    assert pit._draws(conn, record).values.sum() == before


def test_the_two_claims_on_one_feed_do_not_read_each_other(reported):
    """Both are things the model said before the week; neither scorer may see the other."""
    conn, _ = reported
    predictions.archive(conn, 2026, 2, predictions.predict(conn, 2026, 2, _proj(conn, 2)))
    pit.commit(conn, 2026, 2, _proj(conn, 2), observed_at=log.BEFORE_ALL)
    assert len(predictions.archived(conn, 2026)) == 1
    assert len(pit.archived(conn, 2026)) == 1
    assert predictions.archived(conn, 2026)[0]["payload"]["rivals"]
    assert "rivals" not in pit.archived(conn, 2026)[0]["payload"]


def test_an_unfinished_week_is_not_scored_as_a_small_one(reported):
    """A missing or pending slot makes the week unobserved, not low-scoring. Folding a
    partial total in would pile mass at the bottom of the histogram from nothing."""
    conn, _ = reported
    pit.commit(conn, 2026, 2, _proj(conn, 2), observed_at=log.BEFORE_ALL)
    log.report(conn, 2, log.WEEK2, log.REPORT_AT)
    scored, notes = pit.score(conn, 2026)
    assert scored.empty
    assert any("not fully scored" in note["reason"] for note in notes)


def test_a_finished_week_gives_one_draw_per_entrant_inside_the_unit_interval(reported):
    conn, _ = reported
    pit.commit(conn, 2026, 1, _proj(conn), observed_at="2026-09-09T12:00:00+00:00")
    scored, _notes = pit.score(conn, 2026)
    assert len(scored) == 3, "one per entrant in the fixture"
    assert scored.pit.between(0, 1).all()
    assert (scored.below <= scored.pit).all()
    assert (scored.pit <= scored.below + scored.mass).all()


def test_scoring_the_same_week_twice_gives_the_same_answer(reported):
    """The randomisation inside the observed value is what makes a PIT over counts
    uniform, and it has to be reported, or the histogram moves on every run."""
    conn, _ = reported
    pit.commit(conn, 2026, 1, _proj(conn), observed_at="2026-09-09T12:00:00+00:00")
    first, _ = pit.score(conn, 2026)
    second, _ = pit.score(conn, 2026)
    pd.testing.assert_frame_equal(first, second)


def test_the_histogram_reports_a_shape_and_not_a_verdict(reported):
    conn, _ = reported
    pit.commit(conn, 2026, 1, _proj(conn), observed_at="2026-09-09T12:00:00+00:00")
    scored, _ = pit.score(conn, 2026)
    table = pit.uniformity(scored, bins=5)
    assert list(table.bin) == [1, 2, 3, 4, 5]
    assert table["count"].sum() == len(scored)
    assert set(table.columns) == {"bin", "low", "high", "count", "expected"}
    assert "pass" not in table.columns and "ok" not in table.columns


def test_recording_a_prediction_commits_the_distribution_with_it(reported, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    conn, path = reported
    conn.close()
    result = runner.invoke(app, ["research", "predict", "record", "--week", "2", "--db", str(path)])
    assert result.exit_code == 0, result.output
    assert "Committed week 2's implied distribution" in result.output
    conn = db.connect(path)
    assert len(pit.archived(conn, 2026)) == 1
    assert len(predictions.archived(conn, 2026)) == 1
    conn.close()


def test_a_dry_run_commits_nothing(reported):
    conn, path = reported
    conn.close()
    result = runner.invoke(
        app, ["research", "predict", "record", "--week", "2", "--dry-run", "--db", str(path)]
    )
    assert result.exit_code == 0, result.output
    conn = db.connect(path)
    assert not pit.archived(conn, 2026)
    conn.close()


def test_a_week_with_no_projections_refuses_rather_than_committing_nothing(reported):
    conn, _ = reported
    with pytest.raises(ValueError, match="No projection rows"):
        pit.commit(conn, 2026, 17, _proj(conn))


def test_the_jitter_seed_does_not_depend_on_this_process():
    """`hash()` on a string is salted per interpreter. Seeding the PIT's randomisation with
    it would give a different histogram on every run while the module claimed the scoring
    was reproducible -- so this asks a differently-salted interpreter for the same number."""
    import subprocess
    import sys

    assert pit._jitter_seed(2026, 3, "pat") != pit._jitter_seed(2026, 3, "jamie")
    script = "from pool import pit; print(pit._jitter_seed(2026, 3, 'pat'))"
    elsewhere = [
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": salt, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for salt in ("1", "12345")
    ]
    assert elsewhere == [str(pit._jitter_seed(2026, 3, "pat"))] * 2, elsewhere


def test_a_commitment_cannot_be_made_inside_an_open_transaction(reported):
    conn, _ = reported
    conn.execute("BEGIN")
    with pytest.raises(ValueError, match="no open transaction"):
        pit.commit(conn, 2026, 1, _proj(conn), observed_at="2026-09-09T12:00:00+00:00")
    conn.rollback()


UNUSED = datetime(2026, 9, 12, tzinfo=UTC)


@pytest.mark.parametrize("deadline", ["kickoff", "report"])
@pytest.mark.parametrize("offset", [0, 1])
def test_commitment_at_or_after_either_deadline_is_rejected(reported, deadline, offset):
    from datetime import timedelta

    conn, _ = reported
    if deadline == "report":
        log.report(conn, 1, log.WEEK1, "2026-09-09T15:00:00+00:00")
        instant = datetime(2026, 9, 9, 15, tzinfo=UTC)
    else:
        instant = predictions.first_kickoff(conn, 2026, 1)
    pit.commit(conn, 2026, 1, _proj(conn), observed_at=instant + timedelta(seconds=offset))
    scored, notes = pit.score(conn, 2026)
    assert scored.empty
    assert notes
    assert len(pit.archived(conn, 2026)) == 1


def test_only_latest_eligible_commitment_is_rebuilt(reported, monkeypatch):
    conn, _ = reported
    for stamp, sims in [
        ("2026-09-08T12:00:00+00:00", 10),
        ("2026-09-09T12:00:00+00:00", 20),
        ("2026-09-14T12:00:00+00:00", 30),
    ]:
        pit.commit(conn, 2026, 1, _proj(conn), observed_at=stamp, sims=sims)
    real = pit._draws
    seen = []

    def rebuild(conn, record):
        seen.append(record["payload"]["sims"])
        return real(conn, record)

    monkeypatch.setattr(pit, "_draws", rebuild)
    scored, notes = pit.score(conn, 2026)
    assert len(scored) == 3
    assert seen == [20]
    assert notes
    assert len(pit.archived(conn, 2026)) == 3


@pytest.mark.parametrize("missing", ["kickoff", "report"])
def test_pit_requires_both_deadlines(reported, missing):
    conn, _ = reported
    if missing == "kickoff":
        with conn:
            conn.execute("UPDATE games SET kickoff_known = 0 WHERE season = 2026")
    else:
        log.report(conn, 2, log.WEEK2, log.REPORT_AT)
        # Keep entrant picks but remove the archive arrival seam for this test.
    pit.commit(conn, 2026, 1, _proj(conn), observed_at="2026-09-09T12:00:00+00:00")
    if missing == "report":
        from unittest.mock import patch

        with patch.object(predictions, "report_arrivals", return_value=({}, 0)):
            scored, notes = pit.score(conn, 2026)
    else:
        scored, notes = pit.score(conn, 2026)
    assert scored.empty and notes


def test_pit_requires_every_slot_to_be_final():
    from types import SimpleNamespace

    complete = [
        SimpleNamespace(slot=slot, status="final", player_id=slot, tds=0)
        for slot in ("QB", "RB", "FLEX")
    ]
    assert pit._realised(complete) == (0, ["QB", "RB", "FLEX"])
    assert pit._realised(complete[:2]) is None
    complete[-1].status = "pending"
    assert pit._realised(complete) is None


def _changed_sampler(monkeypatch):
    """A later sampler: the same inputs, different draws."""
    real = simulate.sample

    def later(proj, weeks, *, sims, seed, params=None):
        return real(proj, weeks, sims=sims, seed=seed + 1, params=params)

    monkeypatch.setattr(simulate, "sample", later)


def test_a_later_sampler_cannot_restate_a_commitment(reported, monkeypatch):
    """Finding 6 of the 2026-09-17 review, reproduced. The commitment used to keep only
    its seed, so scoring re-ran whatever `simulate.sample` had become and accepted
    different draws without a word. Stored outcomes are read, never re-drawn."""
    conn, _ = reported
    pit.commit(conn, 2026, 1, _proj(conn), observed_at="2026-09-09T12:00:00+00:00")
    before, _ = pit.score(conn, 2026)
    frame = _proj(conn)
    original = simulate.sample(frame, [1], sims=50, seed=1).values
    _changed_sampler(monkeypatch)
    assert not np.array_equal(simulate.sample(frame, [1], sims=50, seed=1).values, original)
    after, _ = pit.score(conn, 2026)
    pd.testing.assert_frame_equal(before, after)


def _schema_one(conn, week=1, observed_at="2026-09-09T12:00:00+00:00"):
    """A commitment as schema 1 wrote it: the seed and the frame, no outcomes."""
    from dataclasses import asdict

    proj = _proj(conn, week)
    payload = dict(
        schema_version=1,
        season=2026,
        week=week,
        sims=config.WINPROB_SIMS,
        seed=config.WINPROB_SEED,
        params=asdict(simulate.Params()),
        surface_hash=None,
    )
    with db.transaction(conn):
        payload["surface_hash"] = capture.store_surface(
            conn, proj[proj.week.eq(week)].reset_index(drop=True)
        )
    return predictions.archive(conn, 2026, week, payload, observed_at=observed_at, kind=pit.KIND)


def test_a_commitment_made_before_outcomes_were_stored_is_frozen_on_first_read(
    reported, monkeypatch
):
    """Drawn once from its seed, stored, and never drawn again. Scoring says so, and a
    sampler change after the freeze no longer reaches it."""
    conn, _ = reported
    _schema_one(conn)
    record = pit.archived(conn, 2026)[0]
    assert pit.frozen(conn, record) is None
    first, notes = pit.score(conn, 2026)
    receipt = pit.frozen(conn, record)
    assert receipt and receipt["numpy"] == np.__version__
    assert any("frozen at" in note["reason"] for note in notes)
    assert len(pit.archived(conn, 2026)) == 1, "a copy of a claim, not a new one"
    _changed_sampler(monkeypatch)
    again, _ = pit.score(conn, 2026)
    pd.testing.assert_frame_equal(first, again)
    assert pit.frozen(conn, record) == receipt, "frozen once, never redrawn"


def test_an_old_commitment_is_frozen_before_its_week_can_be_scored(reported):
    """The freeze is faithful only while the sampler is unchanged, so it cannot wait for
    the week's report and final scores. The first score of the season freezes it."""
    conn, _ = reported
    _schema_one(conn, week=2, observed_at=log.BEFORE_ALL)
    record = pit.archived(conn, 2026)[0]
    scored, notes = pit.score(conn, 2026)
    assert scored.empty and any("no report yet" in note["reason"] for note in notes)
    assert pit.frozen(conn, record), "frozen although nothing could be scored yet"


def test_stored_outcomes_that_do_not_match_their_hash_are_refused(reported):
    conn, _ = reported
    pit.commit(conn, 2026, 1, _proj(conn), observed_at="2026-09-09T12:00:00+00:00")
    record = pit.archived(conn, 2026)[0]
    import zlib

    with conn:
        conn.execute(
            "UPDATE input_payloads SET payload = ? WHERE content_hash = ?",
            (zlib.compress(b"not the outcomes"), record["payload"]["draws_hash"]),
        )
    with pytest.raises(ValueError, match="hash mismatch"):
        pit._draws(conn, record)


# The sampler's exact output on a frame that reaches every branch of it: a starter and a
# backup quarterback, catchers linked to the starter, a running back, a receiver whose
# team has no quarterback, and a team whose catchers out-project their passer.
SAMPLER_DIGEST = "b24c3c81b0f2ec1754d037ba34a240f238314c2f6af69831ba54e049cd98f63e"


def _every_branch():
    from tests.support.frames import proj_row

    rows = [
        ("qa1", "QB", 1.2, "A", "gA", "QB"),
        ("qa2", "QB", 0.3, "A", "gA", "QB"),
        ("wa", "FLEX", 0.6, "A", "gA", "WR"),
        ("ta", "FLEX", 0.3, "A", "gA", "TE"),
        ("ra", "RB", 0.5, "A", "gA", "RB"),
        ("wb", "FLEX", 0.5, "B", "gA", "WR"),
        ("qc", "QB", 0.4, "C", "gC", "QB"),
        ("wc", "FLEX", 1.5, "C", "gC", "WR"),
        ("rd", "RB", 0.7, "D", "gC", "RB"),
    ]
    return pd.DataFrame(
        [
            proj_row(pid, pid, slot, 1, lam, team=team, position=position) | {"game_id": game}
            for pid, slot, lam, team, game, position in rows
        ]
    )


def test_the_sampler_output_is_pinned():
    """Stored outcomes no longer depend on the sampler. Two things still do: a schema-1
    commitment that has not been frozen yet, and every captured pot-share decision, which
    re-derives its shares by sampling. If this fails, `simulate.sample` or numpy changed.
    Freeze any old commitments first (`pool predict score` on the live database), then
    update SAMPLER_DIGEST in the same commit as the change that moved it."""
    import hashlib

    draws = simulate.sample(_every_branch(), [1], sims=400, seed=7)
    assert draws.rescaled == 1, "the frame must reach the rescale guard"
    key = sorted(draws.index.items())
    digest = hashlib.sha256(
        draws.values.astype(np.int32).tobytes() + repr(key).encode()
    ).hexdigest()
    assert digest == SAMPLER_DIGEST, "simulate.sample's output changed; read the docstring"
