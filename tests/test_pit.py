"""The prospective PIT: a distribution pinned before the week, scored after it."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from pool import db, pit, predictions, scoring
from pool.cli import app
from tests import test_predictions as log
from tests import test_workflow as workflow
from tests.test_workflow import end, play

runner = CliRunner()


@pytest.fixture
def local(tmp_path):
    yield from workflow.local.__wrapped__(tmp_path)


@pytest.fixture
def seeded(local):
    conn, path = local
    scoring.import_touchdowns(conn, 2026, pd.DataFrame([play(), end(), end("g2", 0, 0)]))
    log._history(conn)
    log._report(conn, 1, log.WEEK1, "2026-09-14T00:00:00+00:00")
    return conn, path


def _proj(conn, week=1):
    from pool import projections

    return projections.projections_for(conn, 2026, from_week=week)


def test_the_commitment_stores_no_numbers_and_still_fixes_the_distribution(seeded):
    """What makes this evidence rather than decoration.

    The payload is a seed, a size and a content hash. Rebuilding from it twice has to give
    the identical distribution, or the week's claim was never pinned to anything and could
    be quietly restated once the answer was in.
    """
    conn, _ = seeded
    pit.commit(conn, 2026, 1, _proj(conn))
    record = pit.archived(conn, 2026)[0]["payload"]
    assert set(record) >= {"seed", "sims", "surface_hash", "params", "week"}
    assert "draws" not in record and "values" not in record
    first = pit._draws(conn, record)
    second = pit._draws(conn, record)
    assert np.array_equal(first.values, second.values)


def test_the_frame_is_pinned_by_hash_so_a_later_refresh_cannot_move_it(seeded):
    """A distribution defined by 'whatever the projections say' is not a commitment. The
    surface is content-addressed, so a re-forecast next week rebuilds the old one exactly."""
    conn, _ = seeded
    pit.commit(conn, 2026, 1, _proj(conn))
    record = pit.archived(conn, 2026)[0]["payload"]
    before = pit._draws(conn, record).values.sum()
    conn.execute("UPDATE rosters SET status = 'ACT' WHERE season = 2026")
    assert pit._draws(conn, record).values.sum() == before


def test_the_two_claims_on_one_feed_do_not_read_each_other(seeded):
    """Both are things the model said before the week; neither scorer may see the other."""
    conn, _ = seeded
    predictions.archive(conn, 2026, 2, predictions.predict(conn, 2026, 2, _proj(conn, 2)))
    pit.commit(conn, 2026, 2, _proj(conn, 2))
    assert len(predictions.archived(conn, 2026)) == 1
    assert len(pit.archived(conn, 2026)) == 1
    assert predictions.archived(conn, 2026)[0]["payload"]["rivals"]
    assert "rivals" not in pit.archived(conn, 2026)[0]["payload"]


def test_an_unfinished_week_is_not_scored_as_a_small_one(seeded):
    """A missing or pending slot makes the week unobserved, not low-scoring. Folding a
    partial total in would pile mass at the bottom of the histogram from nothing."""
    conn, _ = seeded
    pit.commit(conn, 2026, 2, _proj(conn, 2))
    scored, notes = pit.score(conn, 2026)
    assert scored.empty
    assert any("not fully scored" in note["reason"] for note in notes)


def test_a_finished_week_gives_one_draw_per_entrant_inside_the_unit_interval(seeded):
    conn, _ = seeded
    pit.commit(conn, 2026, 1, _proj(conn))
    scored, _notes = pit.score(conn, 2026)
    assert len(scored) == 3, "one per entrant in the fixture"
    assert scored.pit.between(0, 1).all()
    assert (scored.below <= scored.pit).all()
    assert (scored.pit <= scored.below + scored.mass).all()


def test_scoring_the_same_week_twice_gives_the_same_answer(seeded):
    """The randomisation inside the observed value is what makes a PIT over counts
    uniform, and it has to be seeded, or the histogram moves on every run."""
    conn, _ = seeded
    pit.commit(conn, 2026, 1, _proj(conn))
    first, _ = pit.score(conn, 2026)
    second, _ = pit.score(conn, 2026)
    pd.testing.assert_frame_equal(first, second)


def test_the_histogram_reports_a_shape_and_not_a_verdict(seeded):
    conn, _ = seeded
    pit.commit(conn, 2026, 1, _proj(conn))
    scored, _ = pit.score(conn, 2026)
    table = pit.uniformity(scored, bins=5)
    assert list(table.bin) == [1, 2, 3, 4, 5]
    assert table["count"].sum() == len(scored)
    assert set(table.columns) == {"bin", "low", "high", "count", "expected"}
    assert "pass" not in table.columns and "ok" not in table.columns


def test_recording_a_prediction_commits_the_distribution_with_it(seeded, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    conn, path = seeded
    conn.close()
    result = runner.invoke(app, ["predict", "record", "--week", "2", "--db", str(path)])
    assert result.exit_code == 0, result.output
    assert "Committed week 2's implied distribution" in result.output
    conn = db.connect(path)
    assert len(pit.archived(conn, 2026)) == 1
    assert len(predictions.archived(conn, 2026)) == 1
    conn.close()


def test_a_dry_run_commits_nothing(seeded):
    conn, path = seeded
    conn.close()
    result = runner.invoke(
        app, ["predict", "record", "--week", "2", "--dry-run", "--db", str(path)]
    )
    assert result.exit_code == 0, result.output
    conn = db.connect(path)
    assert not pit.archived(conn, 2026)
    conn.close()


def test_a_week_with_no_projections_refuses_rather_than_committing_nothing(seeded):
    conn, _ = seeded
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
            capture_output=True, text=True, check=True,
            env={"PYTHONHASHSEED": salt, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for salt in ("1", "12345")
    ]
    assert elsewhere == [str(pit._jitter_seed(2026, 3, "pat"))] * 2, elsewhere


def test_a_commitment_cannot_be_made_inside_an_open_transaction(seeded):
    conn, _ = seeded
    conn.execute("BEGIN")
    with pytest.raises(ValueError, match="no open transaction"):
        pit.commit(conn, 2026, 1, _proj(conn))
    conn.rollback()


UNUSED = datetime(2026, 9, 12, tzinfo=UTC)
