"""Backups and exports: a copy that reads back row for row, and picks that read back as
`pool picks` shows them."""

import csv
import errno
import os
import re
import sqlite3

import pandas as pd
import pytest
from typer.testing import CliRunner

from pool import keeping, results, scoring
from pool.cli import app
from tests.support.local import end, play, record

runner = CliRunner()


def _rows(path):
    """Every table's rows, in a form two databases can be compared by."""
    conn = sqlite3.connect(path)
    try:
        tables = [
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY 1")
        ]
        return {t: sorted(map(repr, conn.execute(f"SELECT * FROM {t}"))) for t in tables}
    finally:
        conn.close()


@pytest.fixture
def scored(local):
    """Three picks for week 1: two in the finished game, one in a game still being played."""
    conn, path = local
    record(conn, QB="q1", RB="r1", FLEX="f1")
    scoring.import_touchdowns(
        conn, 2026, pd.DataFrame([play(), end(), end("g2", 0, 0, desc="in progress")])
    )
    return conn, path


def test_a_backup_reads_back_row_for_row(scored, tmp_path):
    _, path = scored
    target = tmp_path / "copy.db"
    out = runner.invoke(app, ["backup", "--to", str(target), "--db", str(path)])
    assert out.exit_code == 0, out.output
    copied = _rows(target)
    assert copied == _rows(path)
    assert len(copied["my_picks"]) == 3 and copied["touchdown_credits"]


def test_a_backup_is_named_by_the_minute_and_never_overwritten(scored):
    _, path = scored
    first = runner.invoke(app, ["backup", "--db", str(path)])
    assert first.exit_code == 0, first.output
    [made] = (path.parent / "backups").glob("*")
    # The suite's clock is 18 September 2026, noon Eastern.
    assert re.fullmatch(r"pool-20260918-12\d\d\.db", made.name)
    before = made.read_bytes()
    again = runner.invoke(app, ["backup", "--to", str(made), "--db", str(path)])
    assert again.exit_code == 1 and "already exists" in again.output
    assert made.read_bytes() == before
    assert [p.name for p in made.parent.iterdir()] == [made.name]  # no partial copy left


def test_backing_up_a_database_that_is_not_there_writes_nothing(tmp_path):
    source, target = tmp_path / "missing.db", tmp_path / "copy.db"
    out = runner.invoke(app, ["backup", "--to", str(target), "--db", str(source)])
    assert out.exit_code == 1 and "No database" in out.output
    assert not target.exists() and not source.exists()


def test_backup_preserves_an_unrelated_partial_file(scored, tmp_path):
    _, source = scored
    target = tmp_path / "copy.db"
    partial = tmp_path / "copy.db.partial"
    partial.write_bytes(b"unrelated records")
    keeping.backup(source, target)
    assert _rows(target) == _rows(source)
    assert partial.read_bytes() == b"unrelated records"


def test_backup_can_copy_a_source_named_like_the_old_partial(scored, tmp_path):
    _, original = scored
    source = tmp_path / "copy.db.partial"
    target = tmp_path / "copy.db"
    keeping.backup(original, source)
    before = source.read_bytes()
    keeping.backup(source, target)
    assert source.read_bytes() == before
    assert _rows(target) == _rows(original)


@pytest.mark.parametrize("alias", ["file", "source", "symlink", "hardlink", "dangling"])
def test_backup_refuses_existing_destinations(scored, tmp_path, alias):
    _, source = scored
    target = tmp_path / "copy.db"
    if alias == "file":
        target.write_bytes(b"existing backup")
    elif alias == "source":
        target = source
    elif alias == "symlink":
        target.symlink_to(source)
    elif alias == "hardlink":
        target.hardlink_to(source)
    else:
        target.symlink_to(tmp_path / "missing.db")
    before = source.read_bytes()
    contents = target.read_bytes() if target.exists() else None
    with pytest.raises(FileExistsError):
        keeping.backup(source, target)
    assert source.read_bytes() == before
    assert os.path.lexists(target)
    assert (target.read_bytes() if target.exists() else None) == contents


def test_backup_refuses_a_destination_created_during_the_copy(scored, tmp_path, monkeypatch):
    _, source = scored
    target = tmp_path / "copy.db"
    before = set(tmp_path.iterdir())
    link = os.link

    def raced(temporary, destination):
        target.write_bytes(b"another writer's backup")
        link(temporary, destination)

    monkeypatch.setattr(keeping.os, "link", raced)
    with pytest.raises(FileExistsError):
        keeping.backup(source, target)
    assert target.read_bytes() == b"another writer's backup"
    assert set(tmp_path.iterdir()) == before | {target}


@pytest.mark.parametrize("failure", ["source", "destination", "copy", "check", "integrity", "link"])
def test_backup_failures_close_connections_and_remove_only_their_temp_file(
    scored, tmp_path, monkeypatch, failure
):
    _, source = scored
    target = tmp_path / "copy.db"
    unrelated = tmp_path / "copy.db.partial"
    unrelated.write_bytes(b"keep me")
    before = {p: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    connections = []
    connect = sqlite3.connect

    class Connection(sqlite3.Connection):
        def backup(self, destination):
            if failure == "copy":
                raise sqlite3.OperationalError("injected copy failure")
            return super().backup(destination)

        def execute(self, sql, *args):
            if sql == "PRAGMA integrity_check":
                if failure == "check":
                    raise sqlite3.OperationalError("injected check failure")
                if failure == "integrity":
                    return super().execute("SELECT 'injected integrity failure'")
            return super().execute(sql, *args)

    def tracked(*args, **kwargs):
        if failure == "source" or (failure == "destination" and connections):
            raise sqlite3.OperationalError(f"injected {failure} failure")
        conn = connect(*args, **kwargs, factory=Connection)
        connections.append(conn)
        return conn

    def unsupported(*args):
        raise OSError(errno.EOPNOTSUPP, "injected hard links unsupported")

    monkeypatch.setattr(keeping.sqlite3, "connect", tracked)
    if failure == "link":
        monkeypatch.setattr(keeping.os, "link", unsupported)
    out = runner.invoke(app, ["backup", "--to", str(target), "--db", str(source)])
    assert out.exit_code == 1 and "injected" in out.output, out.output
    assert {p: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()} == before
    for conn in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("SELECT 1")


@pytest.mark.parametrize("alias", ["csv", "source", "alternate", "symlink", "hardlink", "dangling"])
def test_export_refuses_every_existing_destination(scored, tmp_path, alias):
    _, source = scored
    target = tmp_path / "picks.csv"
    if alias == "csv":
        target.write_bytes(b"existing CSV")
    elif alias == "source":
        target = source
    elif alias == "alternate":
        child = tmp_path / "child"
        child.mkdir()
        target = child / ".." / source.name
    elif alias == "symlink":
        target.symlink_to(source)
    elif alias == "hardlink":
        target.hardlink_to(source)
    else:
        target.symlink_to(tmp_path / "missing.db")
    before = source.read_bytes()
    rows = _rows(source)
    contents = target.read_bytes() if target.exists() else None
    out = runner.invoke(app, ["export", "picks", "--csv", str(target), "--db", str(source)])
    assert out.exit_code == 1 and "File exists" in out.output, out.output
    assert source.read_bytes() == before and _rows(source) == rows
    assert os.path.lexists(target)
    assert (target.read_bytes() if target.exists() else None) == contents


def test_export_reports_other_filesystem_errors(scored, tmp_path):
    _, source = scored
    target = tmp_path / "missing" / "picks.csv"
    out = runner.invoke(app, ["export", "picks", "--csv", str(target), "--db", str(source)])
    assert out.exit_code == 1 and "No such file or directory" in out.output, out.output


def test_the_export_reads_back_as_pool_picks_shows_it(scored, tmp_path):
    conn, path = scored
    target = tmp_path / "picks.csv"
    out = runner.invoke(app, ["export", "picks", "--csv", str(target), "--db", str(path)])
    assert out.exit_code == 0 and "Wrote 3 picks for 2026" in out.output, out.output
    with target.open(newline="") as stream:
        back = {row["slot"]: row for row in csv.DictReader(stream)}
    shown = results.pick_results(conn, 2026).set_index("slot")
    for slot, row in back.items():
        tds = shown.loc[slot, "tds"]
        assert row["week"] == "1" and row["player_name"] == shown.loc[slot, "player_name"]
        assert row["tds"] == ("" if pd.isna(tds) else str(int(tds)))
        assert row["pending"] == shown.loc[slot, "pending_reason"]
    assert (back["QB"]["tds"], back["RB"]["tds"], back["FLEX"]["tds"]) == ("0", "1", "")
    assert back["FLEX"]["pending"] == "no end-of-game marker"
    assert {s: (r["team"], r["position"]) for s, r in back.items()} == {
        "QB": ("A", "QB"),
        "RB": ("A", "RB"),
        "FLEX": ("C", "WR"),
    }
    # Without --csv the same file is printed, unwrapped, for a pipe or a redirect.
    printed = runner.invoke(app, ["export", "picks", "--db", str(path)])
    assert printed.exit_code == 0 and printed.stdout == target.read_text()
