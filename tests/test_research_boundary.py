"""The weekly commands never load research code.

Replay, benchmarks, calibration and the capture checks sit in `pool.research`, behind
`pool research`. A weekly module that imported one of them would couple the everyday path
to a study again and pay for its imports on every run, and nothing else would notice.
"""

import subprocess
import sys

from pool import identity

# The main CLI mounts the research command group, and that module alone.
MOUNT = {"cli": {"research", "research.cli"}}


def _research(name: str) -> bool:
    return name == "research" or name.startswith("research.")


def test_nothing_outside_research_imports_it():
    modules = identity._package_modules()
    reached = {
        name: {m for m in identity._imported(path, name, modules) if _research(m)}
        for name, path in modules.items()
        if not _research(name)
    }
    leaks = {
        name: sorted(found - MOUNT.get(name, set()))
        for name, found in reached.items()
        if found - MOUNT.get(name, set())
    }
    assert not leaks


def test_starting_the_cli_loads_no_research_module():
    """Only the command group itself: each research command imports what it uses when it
    runs. A fresh interpreter, because this one has imported everything by now."""
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, pool.cli; "
            "print(' '.join(sorted(m for m in sys.modules if m.startswith('pool.research'))))",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert loaded == ["pool.research", "pool.research.cli"]
