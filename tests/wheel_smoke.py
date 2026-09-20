"""Standalone installed-wheel check; intentionally outside pytest collection.

Run with Python from .python-version and pass exactly one freshly built wheel.
Only the synthetic season helper is copied into the isolated working directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Executed by the installed interpreter, outside the checkout. Project metadata is
# passed as data; no checkout modules are put on this process's import path.
VERIFY = """
import importlib.metadata as metadata
import json
import re
import sys
from pathlib import Path

import pool
from pool import capture, db, identity, state

project = json.loads(sys.argv[1])
environment = Path(sys.prefix).resolve()
assert Path(pool.__file__).resolve().is_relative_to(environment), pool.__file__
dist = metadata.distribution(project["name"])
assert dist.metadata["Name"] == project["name"]
assert dist.version == project["version"]
normalize = lambda value: re.sub(r"[-_.]+", "-", value).lower().replace(" ", "")
assert {normalize(r) for r in dist.requires} == {
    normalize(r) for r in project["dependencies"]
}, dist.requires
entry, = [e for e in dist.entry_points if e.group == "console_scripts" and e.name == "pool"]
assert entry.value == project["scripts"]["pool"]
assert callable(entry.load())
runtime = identity.runtime()
expected = {
    re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", r).group(0).lower()
    for r in project["dependencies"]
}
assert set(runtime["dependencies"]) == expected
assert all(runtime["dependencies"].values()), runtime["dependencies"]
for name, version in runtime["dependencies"].items():
    assert metadata.version(name) == version
assert runtime["revision"] is None and runtime["dirty"] is None

if sys.argv[2] == "seed":
    from season import seed_season
    conn = db.connect("fixture.db")
    seed_season(conn)
else:
    from season import SEASON
    conn = db.connect("fixture.db")
    picks = state.picks(conn, SEASON)
    assert len(picks) == 1
    pick = picks.iloc[0]
    assert (pick.week, pick.slot, pick.player_id, pick.player_name) == (
        1, "RB", "AAA-RB1", "AAA RB1"
    )
    events = capture.events(conn, SEASON, 1)
    assert list(events.kind) == ["submitted"]
    event = events.iloc[0]
    assert event.slot == "RB" and event.player_id == "AAA-RB1"
    raw = conn.execute(
        "SELECT payload FROM decision_identities WHERE identity_hash = ?",
        (event.identity_hash,),
    ).fetchone()[0]
    recorded = json.loads(raw)
    for key in ("code_hash", "decision_hash", "dependencies", "python", "revision", "dirty"):
        assert recorded[key] == runtime[key], (key, recorded[key], runtime[key])
    assert recorded["constants"] == json.loads(identity.json_text(identity.constants()))
conn.close()
print("Installed package, metadata and " + sys.argv[2] + " checks passed:", pool.__file__)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    if wheel.suffix != ".whl":
        parser.error("expected a built .whl file")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("PYTHON")
        and key not in {"VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "UV_RUN_RECURSION_DEPTH"}
    }
    env.update(
        PYTHONNOUSERSITE="1",
        _TYPER_FORCE_DISABLE_TERMINAL="1",
        TTY_COMPATIBLE="0",
        COLUMNS="120",
    )
    # /tmp also avoids a caller's TMPDIR pointing inside the checkout.
    with tempfile.TemporaryDirectory(prefix="pool-wheel-", dir="/tmp") as directory:
        work = Path(directory)

        def run(*command: str | Path) -> str:
            print("+", " ".join(map(str, command)), flush=True)
            result = subprocess.run(
                list(map(str, command)),
                cwd=work,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            print(result.stdout, end="", flush=True)
            result.check_returncode()
            return result.stdout

        venv = work / "venv"
        run("uv", "venv", "--python", sys.executable, venv)
        python = venv / "bin" / "python"
        pool = venv / "bin" / "pool"
        run("uv", "pip", "install", "--python", python, wheel)
        shutil.copyfile(ROOT / "tests" / "support" / "season.py", work / "season.py")
        probe = work / "verify_install.py"
        probe.write_text(VERIFY)
        run(python, probe, json.dumps(project), "seed")
        help_text = run(pool, "--help")
        assert "record" in help_text and "picks" in help_text
        run(
            pool,
            "record",
            "--week",
            "1",
            "--rb",
            "AAA RB1",
            "--season",
            "2024",
            "--db",
            work / "fixture.db",
        )
        picks = run(pool, "picks", "--season", "2024", "--db", work / "fixture.db")
        assert "AAA RB1" in picks
        run(python, probe, json.dumps(project), "capture")
    print("Wheel smoke check passed.")


if __name__ == "__main__":
    main()
