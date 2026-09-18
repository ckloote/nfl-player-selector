"""Runtime code identity: what a capture records, wherever the package runs from."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pool import benchmark, capture, db, identity
from tests.test_backtest import SEASON, _seed

SOURCE = Path(identity.__file__).resolve().parent


@pytest.fixture
def fresh():
    """`runtime` is cached per process; a test that changes what it reads must not leak."""
    identity.runtime.cache_clear()
    yield
    identity.runtime.cache_clear()


def test_captures_and_research_fingerprint_the_same_closure():
    """Two identities, one closure. Research adds its lockfile; a capture records the
    dependency versions it actually ran with instead."""
    assert benchmark.decision_modules() == sorted([*identity.decision_modules(), "uv.lock"])
    assert identity.runtime()["decision_sources"] == identity.decision_modules()


def test_git_is_provenance_not_a_requirement(fresh, monkeypatch):
    """Without git the fingerprint is unchanged and only the revision is unknown, which
    is exactly what an installed tool looks like."""
    with_git = identity.runtime()
    identity.runtime.cache_clear()

    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(identity.subprocess, "run", missing)
    without = identity.runtime()
    assert without["revision"] is None and without["dirty"] is None
    assert without["decision_hash"] == with_git["decision_hash"]
    assert without["code_hash"] == with_git["code_hash"]


def test_a_dependency_upgrade_moves_the_enforced_fingerprint(fresh, monkeypatch):
    """The arithmetic a decision performs belongs to the numpy it ran on as much as to
    this source, so a different numpy is a different decision function."""
    before = identity.runtime()["decision_hash"]
    identity.runtime.cache_clear()
    real = identity.dependencies
    monkeypatch.setattr(identity, "dependencies", lambda: dict(real(), numpy="99.0"))
    assert identity.runtime()["decision_hash"] != before


def test_an_installed_copy_records_a_pick_outside_any_checkout(tmp_path):
    """Finding 5 of the 2026-09-17 review. `uv tool install .` puts the package in a site
    directory with no git, no pyproject and no lockfile, and recording a pick there failed
    on the missing `pyproject.toml`. A copy of the package, run from outside the
    repository, is that layout."""
    site = tmp_path / "site"
    shutil.copytree(SOURCE, site / "pool", ignore=shutil.ignore_patterns("__pycache__"))
    path = tmp_path / "pool.db"
    _seed(db.connect(path)).close()
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "PYTHONPATH": str(site)}

    def run(*args):
        return subprocess.run(
            [sys.executable, *args], cwd=tmp_path, env=env, capture_output=True, text=True
        )

    where = run("-c", "import pool; print(pool.__file__)")
    assert where.returncode == 0, where.stderr
    assert Path(where.stdout.strip()).parent == site / "pool", "the copy, not the checkout"
    result = run(
        "-m", "pool.cli", "record", "--week", "1", "--rb", "AAA RB1",
        "--season", str(SEASON), "--db", str(path),
    )
    assert result.returncode == 0, result.stdout + result.stderr

    conn = db.connect(path)
    events = capture.events(conn, SEASON, 1)
    assert list(events.kind) == ["submitted"]
    raw = conn.execute(
        "SELECT payload FROM decision_identities WHERE identity_hash = ?",
        (events.identity_hash.iloc[0],),
    ).fetchone()[0]
    conn.close()
    recorded = json.loads(raw)
    assert recorded["revision"] is None and recorded["dirty"] is None
    running = identity.runtime()
    assert recorded["decision_hash"] == running["decision_hash"], "same code, same fingerprint"
    assert recorded["dependencies"] == running["dependencies"]
