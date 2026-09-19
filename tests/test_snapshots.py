"""The input archive: what a feed looked like at an instant, kept once, and read
back exactly as it was.
"""

import json
import sqlite3

import pandas as pd
import pytest

from pool import db, snapshots
from pool import projections as P
from tests.support.season import SEASON, archive_all


def test_snapshots_boundaries_timezones_corrections_and_depth(seeded):
    with seeded:
        seeded.execute(
            "INSERT INTO depth_charts VALUES (?, 'AAA-QB1', 'AAA', 'QB', 2, 'old')", (SEASON,)
        )
    archive_all(seeded)
    decision = "2024-09-01T00:00:00Z"
    a = P.load_frames(seeded, SEASON, 1, input_policy="snapshots", decision_at=decision)
    before = P.build_projections(a, 1)
    with seeded:
        seeded.execute("UPDATE depth_charts SET rank=1")
        seeded.execute("UPDATE games SET total_line=90")
        seeded.execute("DELETE FROM touchdown_credits")
    archive_all(seeded, "2024-09-02T00:00:00Z")
    b = P.load_frames(
        seeded, SEASON, 1, input_policy="snapshots", decision_at="2024-08-31T20:00:00-04:00"
    )
    pd.testing.assert_frame_equal(before, P.build_projections(b, 1))
    assert b.depth.iloc[0]["rank"] == 2
    assert before[before.player_id.eq("AAA-QB1")].role_mult.eq(0.15).all()
    with pytest.raises(ValueError, match="Missing snapshot"):
        P.load_frames(
            seeded, SEASON, 1, input_policy="snapshots", decision_at="2024-08-31T23:59:59.999999Z"
        )


def test_snapshot_optional_absence_staleness_and_deadline(seeded):
    archive_all(seeded, optional=False)
    stamp = "2024-09-11T12:00:00-04:00"  # exact first deadline in the fixture
    loaded = P.load_frames(seeded, SEASON, 1, input_policy="snapshots", decision_at=stamp)
    assert any(p["missing"] for p in loaded.provenance if p["feed"] == "depth_charts")
    assert all(p["stale"] for p in loaded.provenance if not p["missing"])
    frame = P.build_projections(loaded, 1)
    assert not frame[frame.week.eq(1)].hard_eligible.any()
    assert frame[frame.week.gt(1)].hard_eligible.all()


def test_snapshot_dedup_empty_and_atomic_rollback(seeded):
    snapshots.archive(seeded, SEASON, "injuries", observed_at="2024-01-01T00:00:00Z")
    snapshots.archive(seeded, SEASON, "injuries", observed_at="2024-01-02T00:00:00Z")
    assert seeded.execute("SELECT COUNT(*) FROM input_payloads").fetchone()[0] == 1
    assert seeded.execute("SELECT COUNT(*) FROM input_observations").fetchone()[0] == 2
    assert (
        json.loads(seeded.execute("SELECT coverage FROM input_observations LIMIT 1").fetchone()[0])[
            "rows"
        ]
        == 0
    )
    count = seeded.execute("SELECT COUNT(*) FROM rosters").fetchone()[0]
    with pytest.raises(RuntimeError), db.transaction(seeded):
        db.replace_season(
            seeded, "rosters", SEASON, db.read_df(seeded, "SELECT * FROM rosters LIMIT 0")
        )
        snapshots.archive(seeded, SEASON, "rosters")
        raise RuntimeError("abort")
    assert seeded.execute("SELECT COUNT(*) FROM rosters").fetchone()[0] == count
    assert seeded.execute("SELECT COUNT(*) FROM input_observations").fetchone()[0] == 2
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        seeded.execute("DELETE FROM input_observations")


def test_decision_csv_rejects_naive_and_duplicate_times(tmp_path):
    path = tmp_path / "times.csv"
    path.write_text("season,week,decision_at\n2024,1,2024-09-01T00:00:00\n")
    with pytest.raises(ValueError, match="timezone"):
        snapshots.decision_times(path)
    path.write_text(
        "season,week,decision_at\n2024,1,2024-09-01T00:00:00Z\n2024,1,2024-09-02T00:00:00Z\n"
    )
    with pytest.raises(ValueError, match="Duplicate"):
        snapshots.decision_times(path)
