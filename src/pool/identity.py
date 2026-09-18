"""Which code made a decision, read from the package wherever it is installed.

A captured decision records the source it was made with, so that reconstruction can refuse
to substitute different code. That record used to come from the research harness, which
hashes a git checkout, `pyproject.toml` and `uv.lock` -- none of which exist once the tool
is installed with `uv tool install`, so recording a pick failed there. Everything here reads
the package's own files and the installed versions of what it depends on, and treats git as
provenance to attach when present rather than a requirement.

Research runs keep their stricter identity in `benchmark.code_identity`, which still
demands a checkout: a study is reproduced from a revision, and a decision is not.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from functools import cache
from pathlib import Path

from . import config

PACKAGE = Path(__file__).resolve().parent
DISTRIBUTION = "nfl-pool"
# Source is keyed by where it sits in the repository, whatever directory it runs from, so
# the same code fingerprints the same installed or checked out.
PREFIX = "src/pool"

# What a captured decision is a function of: the advice `capture.reconstruct` re-derives
# from a stored surface, and the surface `prospective.parity` rebuilds from the archive.
# Everything reachable from these by import is part of that function; nothing else is.
DECISION_ROOTS = ("recommend", "projections", "snapshots")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"


def constants():
    return {
        k: sorted(v) if isinstance(v, (set, frozenset)) else v
        for k, v in vars(config).items()
        if k.isupper() and k not in ("DB_PATH", "DEFAULT_SEASON", "FRESHNESS_HOURS")
    }


def _key(path: Path, package: Path) -> str:
    return f"{PREFIX}/{path.relative_to(package).as_posix()}"


def _package_modules(package: Path = PACKAGE) -> dict[str, Path]:
    """Every module of `pool`, keyed by its dotted name below the package root."""
    out = {}
    for path in sorted(package.rglob("*.py")):
        parts = path.relative_to(package).with_suffix("").parts
        name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        out[name] = path
    return out


def _with_parents(name: str, modules: dict[str, Path]) -> set[str]:
    """A module and every package above it that exists.

    Importing `pool.strategy.model` runs `pool/strategy/__init__.py` on the way, so an
    initializer is executable code the decision depends on even when nothing imports it
    by name. Recording the leaf alone would leave that file able to change the decision
    without changing its fingerprint.
    """
    parts = name.split(".") if name else []
    return {p for i in range(len(parts) + 1) if (p := ".".join(parts[:i])) in modules}


def _imported(path: Path, name: str, modules: dict[str, Path]) -> set[str]:
    """Sibling modules this one imports, however it spells the import.

    Read from the source rather than by importing it: resolving the closure must not
    depend on which modules a process happens to have loaded, and a function-local
    `from . import x` counts exactly as much as a top-level one.

    A package initializer's own package is itself, not its parent. `from . import weights`
    inside `strategy/__init__.py` means `strategy.weights`; resolving it the way a plain
    module's relative import resolves would point at the package root and quietly find
    nothing.
    """
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    parts = package.split(".") if package else []
    found: set[str] = set()

    def add(target: str, children):
        found.update(_with_parents(target, modules))
        for child in children:
            found.update(_with_parents(f"{target}.{child}".strip("."), modules))

    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            if node.level:  # from . import x  /  from .x import y  /  from ..x import y
                kept = parts[: len(parts) - (node.level - 1)]
                base = ".".join([*kept, node.module] if node.module else kept)
            elif node.module and node.module.split(".")[0] == "pool":
                base = node.module.split(".", 1)[1] if "." in node.module else ""
            else:
                continue
            add(base, [a.name for a in node.names])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "pool":
                    add(alias.name.split(".", 1)[1] if "." in alias.name else "", [])
    return found


def decision_modules(package: Path = PACKAGE) -> list[str]:
    """Source keys of the decision path's import closure.

    Computed rather than listed. A hand-maintained list that quietly loses a module is
    worse than hashing the whole tree: the fingerprint would go on matching while the
    function it certifies had moved underneath it. `tests/test_phase3a.py` pins the
    result, so widening the closure is a review decision rather than a silent one.

    The package root is always in it -- every import of a submodule runs `__init__`.
    """
    modules = _package_modules(package)
    seen: set[str] = set()
    stack = ["", *DECISION_ROOTS]
    while stack:
        name = stack.pop()
        if name in seen or name not in modules:
            continue
        seen.add(name)
        stack += sorted(_imported(modules[name], name, modules) - seen)
    return sorted(_key(modules[name], package) for name in seen)


def dependencies() -> dict[str, str | None]:
    """The installed version of every runtime dependency the package declares.

    What actually runs, rather than what a lockfile asked for: an installed tool has no
    lockfile, and the arithmetic a decision performs belongs to the numpy and pandas it
    imported. Empty when the package's own metadata cannot be found, which is recorded
    as exactly that rather than guessed at.
    """
    try:
        required = importlib.metadata.requires(DISTRIBUTION) or []
    except importlib.metadata.PackageNotFoundError:
        return {}
    out: dict[str, str | None] = {}
    for requirement in required:
        if "extra ==" in requirement:
            continue
        name = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement).group(0).lower()
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return dict(sorted(out.items()))


def git_state() -> tuple[str | None, bool | None]:
    """The checkout this package runs from, when it runs from one.

    Provenance, never a requirement. The checkout has to be the one this package lives
    in: an installed copy sitting under some unrelated repository must not borrow that
    repository's revision.
    """

    def git(*args, cwd):
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout

    try:
        top = Path(git("rev-parse", "--show-toplevel", cwd=PACKAGE).strip())
        if (top / PREFIX).resolve() != PACKAGE:
            return None, None
        revision = git("rev-parse", "HEAD", cwd=top).strip()
        status = git("status", "--porcelain", cwd=top)
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return revision, bool(status.strip())


@cache
def runtime() -> dict:
    """Fingerprint the running package once per process, whole and decision-scoped.

    Source cannot change under a running process in any way this tool would survive, and
    hashing it shells out to git. Constants *can* change -- `config.override` exists -- so
    they are deliberately not part of this and are read by the caller on every use.
    """
    hashes = {_key(path, PACKAGE): digest(path) for path in sorted(PACKAGE.rglob("*.py"))}
    deps = dependencies()
    closure = decision_modules()
    decision = {path: hashes[path] for path in closure}
    revision, dirty = git_state()
    return dict(
        source_hashes=hashes,
        code_hash=hashlib.sha256(json_text(dict(hashes, dependencies=deps)).encode()).hexdigest(),
        decision_sources=closure,
        decision_hash=hashlib.sha256(
            json_text(dict(decision, dependencies=deps)).encode()
        ).hexdigest(),
        dependencies=deps,
        python=platform.python_version(),
        revision=revision,
        dirty=dirty,
    )
