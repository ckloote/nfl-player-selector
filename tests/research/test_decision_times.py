"""A benchmark's decision times are frozen inputs: read once, fingerprinted, and
handed to workers as timestamps rather than as a path to re-read.
"""

import hashlib
import sqlite3
from pathlib import Path

import pytest

from pool import db, snapshots
from pool.research import benchmark
from tests.support.season import PRIOR, SEASON, WEEKS, archive_all, seed_season


# --- frozen decision-time inputs --------------------------------------------
def _times_csv(path, weeks=WEEKS, stamp="2024-09-01T00:00:00Z"):
    path.write_text("season,week,decision_at\n" + "".join(f"{SEASON},{w},{stamp}\n" for w in weeks))
    return path


def _snapshot_spec(csv_path, **overrides):
    spec = benchmark.resolve(Path("experiments/phase2-validation.toml"))
    spec.update(
        seasons=[SEASON],
        history_start=PRIOR,
        models=["shipped", "random"],
        baseline="shipped",
        seeds=[0, 1],
        random_trials=2,
        workers=1,
        eras={"test retrospective": [SEASON, SEASON]},
        expected_games={str(PRIOR): 8, str(SEASON): 8},
        input_policy="snapshots",
        role_source="depth",
        decision_times=str(csv_path),
    )
    spec.update(overrides)
    spec["resolved_decision_times"] = benchmark.decision_time_records(spec)
    return spec


def _research_db(tmp_path, name):
    conn = seed_season(db.connect(tmp_path / f"{name}-source.db"))
    archive_all(conn, "2024-09-01T00:00:00Z")
    out = tmp_path / name
    out.mkdir()
    destination = sqlite3.connect(out / "research.db")
    conn.backup(destination)
    destination.close()
    return out


def test_editing_the_decision_csv_in_place_changes_the_run_identity(tmp_path):
    """The identity covered the path, so the same path with different timestamps was
    the same run. The contents are the input; the path is where it happened to live."""
    csv = _times_csv(tmp_path / "times.csv")
    before = _snapshot_spec(csv)
    _times_csv(csv, stamp="2024-09-02T00:00:00Z")
    after = _snapshot_spec(csv)

    assert before["decision_times"] == after["decision_times"]
    assert before["resolved_decision_times"] != after["resolved_decision_times"]
    digest = lambda spec: hashlib.sha256(benchmark.json_text(spec).encode()).hexdigest()  # noqa: E731
    assert digest(before) != digest(after)


@pytest.mark.parametrize(
    ("contents", "match"),
    [
        ("season,week,decision_at\n2024,1,2024-09-01T00:00:00\n", "timezone"),
        (
            "season,week,decision_at\n2024,1,2024-09-01T00:00:00Z\n2024,1,2024-09-02T00:00:00Z\n",
            "Duplicate",
        ),
        ("season,week\n2024,1\n", "decision_at"),
    ],
)
def test_an_unusable_decision_csv_fails_while_resolving_the_configuration(
    tmp_path, contents, match
):
    """These already failed -- inside a worker, after the manifest was written."""
    csv = tmp_path / "times.csv"
    csv.write_text(contents)
    with pytest.raises(ValueError, match=match):
        _snapshot_spec(csv)


def test_a_missing_week_fails_before_any_worker_starts(tmp_path, monkeypatch):
    """A fifteen-season run should not discover a gap in season eleven."""
    spec = _snapshot_spec(_times_csv(tmp_path / "times.csv", weeks=[1, 2]))
    monkeypatch.setattr(benchmark, "resolve", lambda path: spec)
    monkeypatch.setattr(
        benchmark, "code_identity", lambda: dict(code_hash="t", revision="t", dirty=False)
    )
    monkeypatch.setattr(
        benchmark, "evaluate_season", lambda *a: pytest.fail("started a worker anyway")
    )
    out = _research_db(tmp_path, "gap")
    with pytest.raises(ValueError, match="missing timestamps"):
        benchmark.run("unused", out, log=lambda x: None)
    assert not (out / "manifest.json").exists()


@pytest.mark.parametrize(
    ("policy", "path", "match"),
    [
        ("snapshots", None, "requires decision_times"),
        ("historical", "times.csv", "requires input_policy"),
    ],
)
def test_a_snapshot_policy_and_a_decision_csv_require_each_other(tmp_path, policy, path, match):
    _times_csv(tmp_path / "times.csv")
    spec = benchmark.resolve(Path("experiments/phase2-validation.toml"))
    spec["input_policy"] = policy
    spec["decision_times"] = None if path is None else str(tmp_path / path)
    with pytest.raises(ValueError, match=match):
        benchmark.decision_time_records(spec)


def test_workers_read_the_frozen_timestamps_not_the_path(tmp_path, monkeypatch):
    """Each worker is a spawned process with its own working directory, and re-read the
    path there. The frozen records travel in the pickled specification instead."""
    spec = _snapshot_spec(_times_csv(tmp_path / "times.csv"))
    monkeypatch.setattr(benchmark, "resolve", lambda path: spec)
    monkeypatch.setattr(
        benchmark, "code_identity", lambda: dict(code_hash="t", revision="t", dirty=False)
    )
    monkeypatch.setattr(
        snapshots, "decision_times", lambda path: pytest.fail("re-read the mutable path")
    )
    compact = benchmark.run("unused", _research_db(tmp_path, "frozen"), log=lambda x: None)
    assert (compact.parent / "decision-times.csv").exists()
    assert (compact / "EVALUATION.md").exists()


def test_resume_rejects_an_edited_decision_csv_at_the_same_path(tmp_path, monkeypatch):
    """The whole point: a run continued after its inputs changed under it."""
    csv = _times_csv(tmp_path / "times.csv")
    spec = _snapshot_spec(csv)
    monkeypatch.setattr(benchmark, "resolve", lambda path: spec)
    monkeypatch.setattr(
        benchmark, "code_identity", lambda: dict(code_hash="t", revision="t", dirty=False)
    )
    out = _research_db(tmp_path, "resume")
    benchmark.run("unused", out, log=lambda x: None)

    unchanged = _snapshot_spec(csv)
    monkeypatch.setattr(benchmark, "resolve", lambda path: unchanged)
    monkeypatch.setattr(
        benchmark, "evaluate_season", lambda *a: pytest.fail("recomputed a saved season")
    )
    benchmark.run("unused", out, resume=True, log=lambda x: None)

    _times_csv(csv, stamp="2024-09-02T00:00:00Z")
    edited = _snapshot_spec(csv)
    monkeypatch.setattr(benchmark, "resolve", lambda path: edited)
    with pytest.raises(ValueError, match="fingerprint"):
        benchmark.run("unused", out, resume=True, log=lambda x: None)
