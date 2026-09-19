"""Runtime code identity: what a capture records, wherever the package runs from."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pool import capture, db, identity
from pool.research import benchmark
from tests.support.season import SEASON, seed_season

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
    seed_season(db.connect(path)).close()
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path), "PYTHONPATH": str(site)}

    def run(*args):
        return subprocess.run(
            [sys.executable, *args], cwd=tmp_path, env=env, capture_output=True, text=True
        )

    where = run("-c", "import pool; print(pool.__file__)")
    assert where.returncode == 0, where.stderr
    assert Path(where.stdout.strip()).parent == site / "pool", "the copy, not the checkout"
    result = run(
        "-m",
        "pool.cli",
        "record",
        "--week",
        "1",
        "--rb",
        "AAA RB1",
        "--season",
        str(SEASON),
        "--db",
        str(path),
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


# The closure is computed, so this test is what makes widening it a decision rather than
# an accident: an import added to the decision path fails here and has to be looked at.
DECISION_SOURCES = [
    "src/pool/__init__.py",
    "src/pool/config.py",
    "src/pool/db.py",
    "src/pool/freshness.py",
    "src/pool/optimizer.py",
    "src/pool/projections.py",
    "src/pool/recommend.py",
    "src/pool/rivals.py",
    "src/pool/scoring.py",
    "src/pool/simulate.py",
    "src/pool/snapshots.py",
    "src/pool/state.py",
    "uv.lock",
]


def test_the_enforced_fingerprint_covers_the_decision_path_and_only_that():
    """What a decision is a function of, pinned. A module that slipped out of this list
    would leave the fingerprint matching while the recommender moved underneath it, and
    one that slipped in would make an unrelated feature invalidate real captures."""
    assert benchmark.decision_modules() == DECISION_SOURCES
    identity = benchmark.code_identity()
    assert identity["decision_sources"] == DECISION_SOURCES
    assert identity["decision_hash"] != identity["code_hash"]
    # The readers, the CLI and the research harness describe decisions; they do not make
    # them, and the record keeps their hashes without enforcing them.
    outside = {
        "standings",
        "predictions",
        "cli",
        "ingest",
        "entrants",
        "diagnostics",
        "verify",
        "capture",
        "benchmark",
    }
    assert not outside & {Path(p).stem for p in DECISION_SOURCES}
    assert set(identity["source_hashes"]) > set(DECISION_SOURCES) - {"uv.lock"}


def _nested_tree(tmp_path, recommend_import, strategy_init):
    """A synthetic package whose decision path reaches into a subpackage.

    `src/pool` is flat today, so the real tree cannot exercise nested resolution at all --
    and the day someone moves decision logic into a subpackage is the day a closure that
    resolves it wrongly starts fingerprinting less than it claims to.
    """
    pool = tmp_path / "src" / "pool"
    (pool / "strategy").mkdir(parents=True)
    for name, body in (
        ("__init__.py", ""),
        ("recommend.py", recommend_import),
        ("projections.py", "from . import config\n"),
        ("snapshots.py", "from . import db\n"),
        ("config.py", ""),
        ("db.py", ""),
    ):
        (pool / name).write_text(body)
    (pool / "strategy" / "__init__.py").write_text(strategy_init)
    (pool / "strategy" / "model.py").write_text("def blend():\n    return 1\n")
    (pool / "strategy" / "weights.py").write_text("DEPTH = 0.15\n")
    return benchmark.decision_modules(root=tmp_path)


def test_a_subpackage_initializer_is_part_of_the_decision(tmp_path):
    """`from .strategy.model import blend` runs `strategy/__init__.py` on the way in.
    Fingerprinting the leaf alone leaves that file able to change the decision without
    changing the hash that is supposed to certify it."""
    found = _nested_tree(
        tmp_path,
        recommend_import="from .strategy.model import blend\n",
        strategy_init="",
    )
    assert "src/pool/strategy/model.py" in found
    assert "src/pool/strategy/__init__.py" in found


def test_an_initializer_resolves_its_own_relative_imports(tmp_path):
    """Inside `strategy/__init__.py`, `from . import weights` means `strategy.weights`.
    Resolving it the way a plain module's relative import resolves points at the package
    root, finds nothing, and silently drops executable code from the closure."""
    found = _nested_tree(
        tmp_path,
        recommend_import="from . import strategy\n",
        strategy_init="from . import weights\n",
    )
    assert "src/pool/strategy/__init__.py" in found
    assert "src/pool/strategy/weights.py" in found


def test_a_subpackage_the_decision_path_never_reaches_stays_out(tmp_path):
    """The closure is only worth narrowing if it still excludes what it should."""
    found = _nested_tree(
        tmp_path,
        recommend_import="from . import config\n",
        strategy_init="from . import weights\n",
    )
    assert not [p for p in found if "strategy" in p]
