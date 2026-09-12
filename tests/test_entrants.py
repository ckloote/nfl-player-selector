import hashlib
import json
import sqlite3
import zlib
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from pool import capture, config, db, entrants, freshness, snapshots
from pool.cli import app

REFERENCE = Path(__file__).parent / "fixtures" / "pool_report.csv"


@pytest.fixture
def report_db(tmp_path):
    conn = db.connect(tmp_path / "reports.db")
    with conn:
        conn.executemany(
            "INSERT INTO games(game_id, season, week, game_type, kickoff, home_team, away_team, "
            "home_score, away_score, kickoff_known) VALUES (?, 2026, ?, 'REG', ?, ?, ?, 0, 0, 1)",
            [
                ("g1", 1, "2026-09-10T20:15", "A", "B"),
                ("g2", 1, "2026-09-13T13:00", "C", "D"),
                ("g3", 2, "2026-09-20T13:00", "A", "B"),
            ],
        )
        conn.executemany(
            "INSERT INTO rosters(season, week, player_id, player_name, position, team, status) "
            "VALUES (2026, 1, ?, ?, ?, ?, 'ACT')",
            [
                ("q1", "Quarter One", "QB", "A"),
                ("q2", "Quarter Two", "QB", "A"),
                ("r1", "Runner One", "RB", "A"),
                ("f1", "Flex One", "WR", "C"),
                ("f2", "Flex Two", "TE", "A"),
            ],
        )
        conn.executemany(
            "INSERT INTO game_results VALUES (?, 2026, 1, 1, 'complete', 0, 0, ?)",
            [("g1", "2026-09-15T00:00:00+00:00"), ("g2", "2026-09-15T00:00:00+00:00")],
        )
    yield conn
    conn.close()


def import_reference(conn, raw=None, **kwargs):
    return entrants.import_report(
        conn,
        2026,
        kwargs.pop("week", 1),
        REFERENCE.read_bytes() if raw is None else raw,
        source="week1.csv",
        **kwargs,
    )


def count(conn, table):
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_transform_separates_grains_and_preserves_reported_names():
    frame = entrants.parse_csv(REFERENCE.read_bytes())
    identities, picks, totals = entrants.transform_report(frame, 2026, 1)
    assert identities.to_dict("records") == [
        dict(season=2026, entrant_id="chris k", display_name="Chris K."),
        dict(season=2026, entrant_id="pat", display_name="Pat"),
    ]
    assert len(picks) == 6 and set(picks.slot) == set(config.SLOTS)
    assert set(picks.player_name) == {
        "Quarter One",
        "Quarter Two",
        "Runner One",
        "Flex One",
        "Flex Two",
    }
    assert totals.to_dict("records") == [
        dict(
            season=2026,
            week=1,
            entrant_id="chris k",
            reported_week=3,
            reported_total=3,
            reported_rank=1,
        ),
        dict(
            season=2026,
            week=1,
            entrant_id="pat",
            reported_week=2,
            reported_total=2,
            reported_rank=2,
        ),
    ]
    pd.testing.assert_frame_equal(frame, entrants.parse_csv(REFERENCE.read_bytes()))


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda f: f.assign(slot="K"), "Unknown slots"),
        (lambda f: f.iloc[:-1], "missing slots"),
        (lambda f: pd.concat([f, f.iloc[:1]]), "Duplicate entrant/slot"),
        (lambda f: f.assign(entrant="!!!"), "letters or numbers"),
        (lambda f: f.assign(player_name=" "), "empty player_name"),
        (lambda f: f.assign(reported_total="1.5"), "integer"),
        (lambda f: f.assign(reported_total="-1"), "integer"),
        (lambda f: f.assign(reported_rank="0"), "positive"),
        (lambda f: f.assign(reported_week=["1", "2", "", "2", "", ""]), "conflicting"),
        (lambda f: f.assign(week="2"), "disagrees"),
        (lambda f: f.assign(season="2025"), "season disagrees"),
        (lambda f: f.assign(total="3"), "Unknown report columns"),
    ],
)
def test_transform_rejects_silent_data_loss(change, message):
    with pytest.raises(ValueError, match=message):
        entrants.transform_report(change(entrants.parse_csv(REFERENCE.read_bytes())), 2026, 1)


def test_import_and_readback_keep_provenance_and_reported_totals(report_db):
    result = import_reference(report_db)
    assert result.ok and result.written and result.parsed
    assert (result.entrants, result.picks, result.totals) == (2, 6, 2)
    assert not result.warnings
    rows = entrants.entrant_picks(report_db, 2026, 1)
    assert len(rows) == 6 and set(rows.observation_id) == {result.observation_id}
    assert rows.observed_at.notna().all() and rows.content_hash.notna().all()
    assert set(rows.game_id) == {"g1", "g2"}
    assert set(rows.player_id) == {"q1", "q2", "r1", "f1", "f2"}
    assert set(entrants.reported_totals(report_db, 2026).reported_total) == {2, 3}
    receipt = report_db.execute("SELECT * FROM input_observations").fetchone()
    payload = report_db.execute("SELECT * FROM input_payloads").fetchone()
    raw = REFERENCE.read_bytes()
    assert payload["codec"] == "zlib-bytes" and zlib.decompress(payload["payload"]) == raw
    assert receipt["content_hash"] == hashlib.sha256(raw).hexdigest()
    assert receipt["source_timestamp"] is None
    assert json.loads(receipt["coverage"]) == dict(
        week=1, source="week1.csv", parsed=False, format="csv"
    )
    report = entrants.reports(report_db, 2026).iloc[0]
    assert report.parsed and report.written and report.picks == 6 and report.unresolved == 0
    assert report.source == "week1.csv"


def test_reimport_deduplicates_bytes_updates_provenance_preserves_first_seen(report_db):
    first = import_reference(report_db, observed_at="2026-09-15T00:00:00Z")
    second = import_reference(report_db, observed_at="2026-09-16T00:00:00Z")
    assert second.ok and second.observation_id != first.observation_id
    assert [
        count(report_db, t)
        for t in (
            "pool_entrants",
            "pool_picks",
            "pool_report_totals",
            "input_observations",
            "input_payloads",
        )
    ] == [2, 6, 2, 2, 1]
    assert set(entrants.entrant_picks(report_db, 2026).observation_id) == {second.observation_id}
    assert report_db.execute("SELECT first_seen FROM pool_entrants LIMIT 1").fetchone()[0] == (
        "2026-09-15T00:00:00.000000+00:00"
    )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        with report_db:
            report_db.execute("UPDATE input_observations SET coverage = '{}'")


@pytest.mark.parametrize(
    "raw, kwargs, error",
    [
        (b"", {}, "empty"),
        (b"%PDF-unrecognized", {}, "--week"),
        (b"\xff\xfe", {}, "Cannot parse CSV"),
        (b"a,a\n1,2", {}, "unique"),
        (b"a,b\n1,2,3", {}, "expected 2 fields"),
        (b'a,b\n"unfinished', {}, "Cannot parse CSV"),
        (REFERENCE.read_bytes(), {"fmt": "pdf"}, "Unknown report format"),
        (REFERENCE.read_bytes(), {"week": 2}, "disagrees"),
        (REFERENCE.read_bytes().replace(b"1,", b"19,"), {}, "outside"),
    ],
)
def test_invalid_attempt_still_leaves_recoverable_receipt(report_db, raw, kwargs, error):
    result = import_reference(report_db, raw, week=kwargs.pop("week", None), **kwargs)
    assert not result.ok and any(error in e for e in result.errors)
    assert not result.written and count(report_db, "pool_picks") == 0
    assert count(report_db, "input_observations") == count(report_db, "input_payloads") == 1
    assert (
        zlib.decompress(report_db.execute("SELECT payload FROM input_payloads").fetchone()[0])
        == raw
    )


def test_archive_is_committed_before_parser_runs_and_survives_unexpected_failure(
    report_db, monkeypatch
):
    path = report_db.execute("PRAGMA database_list").fetchone()[2]

    def crash(raw):
        other = db.connect(path)
        try:
            assert count(other, "input_observations") == 1
            assert count(other, "pool_picks") == 0
            assert (
                zlib.decompress(other.execute("SELECT payload FROM input_payloads").fetchone()[0])
                == raw
            )
        finally:
            other.close()
        raise RuntimeError("parser crashed")

    monkeypatch.setitem(entrants.PARSERS, "csv", crash)
    with pytest.raises(RuntimeError, match="parser crashed"):
        import_reference(report_db)
    assert count(report_db, "input_observations") == 1
    assert not entrants.reports(report_db, 2026).iloc[0].parsed


def test_second_transaction_rolls_back_rows_but_keeps_archive(report_db, monkeypatch):
    write = entrants._write_report

    def fail(*args):
        write(*args)
        raise sqlite3.IntegrityError("write failed")

    monkeypatch.setattr(entrants, "_write_report", fail)
    with pytest.raises(sqlite3.IntegrityError, match="write failed"):
        import_reference(report_db)
    assert count(report_db, "input_observations") == 1
    assert all(
        count(report_db, t) == 0 for t in ("pool_entrants", "pool_picks", "pool_report_totals")
    )


def test_unknown_name_is_retained_and_reimport_resolves_it(report_db):
    raw = REFERENCE.read_bytes().replace(b"Quarter One", b"Mystery Player")
    first = import_reference(report_db, raw)
    assert not first.ok and first.written
    assert first.unresolved == [
        dict(entrant_id="chris k", slot="QB", player_name="Mystery Player", candidates=[])
    ]
    unresolved = (
        entrants.entrant_picks(report_db, 2026)
        .query("slot == 'QB' and entrant_id == 'chris k'")
        .iloc[0]
    )
    assert unresolved.player_name == "Mystery Player" and pd.isna(unresolved.player_id)
    assert (
        report_db.execute(
            "SELECT player_id FROM pool_picks WHERE entrant_id = 'chris k' AND slot = 'QB'"
        ).fetchone()[0]
        is None
    )
    with report_db:
        report_db.execute(
            "INSERT INTO rosters(season, week, player_id, player_name, position, team) "
            "VALUES (2026, 1, 'm1', 'Mystery Player', 'QB', 'A')"
        )
    second = import_reference(report_db, raw)
    assert second.ok and not second.unresolved and count(report_db, "pool_picks") == 6
    assert "m1" in set(entrants.entrant_picks(report_db, 2026).player_id)
    assert count(report_db, "my_picks") == 0


def test_ambiguity_lists_candidates_and_slot_restriction_resolves_names(report_db):
    ambiguous = import_reference(
        report_db, REFERENCE.read_bytes().replace(b"Quarter One", b"Quarter")
    )
    assert not ambiguous.ok
    assert {c["player_id"] for c in ambiguous.unresolved[0]["candidates"]} == {"q1", "q2"}
    with report_db:
        report_db.execute(
            "INSERT INTO rosters(season, week, player_id, player_name, position, team) "
            "VALUES (2026, 1, 'same', 'Quarter One', 'WR', 'B')"
        )
    assert import_reference(report_db).ok


@pytest.mark.parametrize("change", ["rename", "addition", "removal"])
def test_roster_changes_are_written_then_flagged_and_can_be_acknowledged(report_db, change):
    import_reference(report_db)
    frame = entrants.parse_csv(REFERENCE.read_bytes()).assign(week="2")
    if change == "rename":
        frame.loc[frame.entrant.eq("Pat"), "entrant"] = "Robin"
    elif change == "addition":
        frame = pd.concat([frame, frame.iloc[:3].assign(entrant="Robin")])
    else:
        frame = frame[frame.entrant.ne("Pat")]
    raw = frame.to_csv(index=False).encode()
    rejected = import_reference(report_db, raw, week=2)
    assert not rejected.ok and rejected.written and rejected.previous_week == 1
    assert bool(rejected.added) == (change != "removal")
    assert bool(rejected.removed) == (change != "addition")
    assert len(entrants.entrant_picks(report_db, 2026, 2)) == len(frame)
    assert import_reference(report_db, raw, week=2, allow_roster_change=True).ok


def test_corrected_week_removes_obsolete_rows_and_check_preserves_them(report_db):
    import_reference(report_db)
    frame = entrants.parse_csv(REFERENCE.read_bytes()).query("entrant != 'Pat'")
    raw = frame.to_csv(index=False).encode()
    assert not import_reference(report_db, raw, check=True).ok
    assert len(entrants.entrant_picks(report_db, 2026, 1)) == 6
    result = import_reference(report_db, raw)
    assert not result.ok and result.removed == ["pat"]
    assert len(entrants.entrant_picks(report_db, 2026, 1)) == 3
    assert len(entrants.reported_totals(report_db, 2026, 1)) == 1
    assert count(report_db, "pool_entrants") == 2
    assert import_reference(report_db, raw, allow_roster_change=True).ok


def test_normalized_entrant_name_remains_one_identity_across_weeks(report_db):
    import_reference(report_db)
    raw = REFERENCE.read_bytes().replace(b"Chris K.", b"chris k")
    raw = entrants.parse_csv(raw).assign(week=2).to_csv(index=False).encode()
    assert import_reference(report_db, raw, week=2).ok
    assert count(report_db, "pool_entrants") == 2


def test_me_comparison_persists_and_never_edits_my_picks(report_db):
    with report_db:
        report_db.executemany(
            "INSERT INTO my_picks(season, week, slot, player_id, player_name, recorded_at, tds) "
            "VALUES (2026, 1, ?, ?, ?, '2026-09-10T00:00:00Z', 7)",
            [("QB", "q2", "Quarter Two"), ("RB", "r1", "Runner One"), ("FLEX", "f1", "Flex One")],
        )
    before = db.read_df(report_db, "SELECT * FROM my_picks")
    first = import_reference(report_db, me="chris k")
    assert first.comparison == [dict(slot="QB", recorded="Quarter Two", reported="Quarter One")]
    assert not first.ok and first.written
    assert import_reference(report_db).comparison == first.comparison
    assert not import_reference(report_db, me="Nobody").ok
    assert not import_reference(report_db, me="Pat").ok
    pd.testing.assert_frame_equal(before, db.read_df(report_db, "SELECT * FROM my_picks"))
    assert (
        report_db.execute("SELECT entrant_id FROM pool_entrants WHERE is_me = 1").fetchone()[0]
        == "chris k"
    )


def test_check_archives_and_parses_without_any_entrant_rows(report_db):
    result = import_reference(report_db, check=True, me="Chris K.")
    assert result.check and result.parsed and not result.written and result.picks == 6
    assert count(report_db, "input_observations") == 1
    assert all(
        count(report_db, t) == 0 for t in ("pool_entrants", "pool_picks", "pool_report_totals")
    )
    assert entrants.reports(report_db, 2026).iloc[0].check


def test_week_inference_and_incomplete_scoring_warning(report_db):
    assert import_reference(report_db, week=None).week == 1
    raw = entrants.parse_csv(REFERENCE.read_bytes()).assign(week=2).to_csv(index=False).encode()
    result = import_reference(report_db, raw, week=None)
    assert result.week == 2 and result.ok and result.written
    assert "0/1 games complete" in result.warnings[0]
    no_week = (
        entrants.parse_csv(REFERENCE.read_bytes()).drop(columns="week").to_csv(index=False).encode()
    )
    assert import_reference(report_db, no_week).ok
    assert not import_reference(report_db, no_week, week=None).ok


def test_week_stats_game_takes_precedence_over_roster_team_and_unresolved_game_is_null(report_db):
    with report_db:
        report_db.execute(
            "INSERT INTO player_weeks(season, week, season_type, player_id, player_name, position, "
            "team, game_id) VALUES (2026, 1, 'REG', 'q1', 'Quarter One', 'QB', 'C', 'g1')"
        )
        report_db.execute("UPDATE rosters SET team = 'Z' WHERE player_id = 'q2'")
    assert import_reference(report_db).ok
    rows = entrants.entrant_picks(report_db, 2026).set_index("player_id")
    assert rows.loc["q1", "game_id"] == "g1" and pd.isna(rows.loc["q2", "game_id"])


def test_report_is_invisible_to_capture_restore_and_freshness(report_db):
    stamp = "2026-09-15T00:00:00+00:00"
    for year in (2025, 2026):
        for feed in snapshots.TABLES:
            snapshots.archive(report_db, year, feed, observed_at=stamp)
    before = capture.observed_inputs(report_db, 2026, "2026-09-16T00:00:00+00:00")
    import_reference(report_db, observed_at=stamp)
    assert "pool_report" not in snapshots.TABLES and "pool_report" not in freshness.FEEDS
    assert capture.observed_inputs(report_db, 2026, "2026-09-16T00:00:00+00:00") == before
    restored, provenance = snapshots.restore(report_db, 2026, stamp, 1)
    try:
        assert len(db.read_df(restored, "SELECT * FROM games")) == 3
        assert all(
            count(restored, t) == 0 for t in ("pool_entrants", "pool_picks", "pool_report_totals")
        )
        assert all(row["feed"] != "pool_report" for row in provenance)
    finally:
        restored.close()
    assert "pool_report" in set(snapshots.coverage(report_db).feed)


def test_missing_totals_stay_missing_and_literal_na_names_survive():
    raw = b'entrant,slot,player_name\nNA,QB,NA\nNA,RB,"Runner, One"\nNA,FLEX,Flex One\n'
    identities, picks, totals = entrants.transform_report(entrants.parse_csv(raw), 2026, 1)
    assert identities.display_name.tolist() == ["NA"]
    assert picks.player_name.tolist() == ["NA", "Runner, One", "Flex One"]
    assert totals[entrants.TOTAL_COLUMNS].isna().all().all()


def test_transform_accepts_integer_valued_pandas_numeric_columns():
    frame = pd.read_csv(REFERENCE)
    _, _, totals = entrants.transform_report(frame, 2026, 1)
    assert list(totals.reported_week) == [3, 2]


def test_missing_schedule_is_archived_before_validation(tmp_path):
    conn = db.connect(tmp_path / "empty.db")
    try:
        result = import_reference(conn)
        assert not result.ok and "schedule" in result.errors[0]
        assert count(conn, "input_observations") == 1 and count(conn, "pool_picks") == 0
    finally:
        conn.close()


@pytest.fixture
def report_cli(report_db, monkeypatch):
    monkeypatch.setenv("COLUMNS", "240")
    path = report_db.execute("PRAGMA database_list").fetchone()[2]

    def invoke(*args):
        return CliRunner().invoke(app, [*args, "--season", "2026", "--db", path])

    return invoke


def test_cli_import_list_and_reported_standings(report_cli, report_db, tmp_path):
    result = report_cli("report", "import", str(REFERENCE))
    assert result.exit_code == 0, result.output
    assert "observation 1" in result.output and "2 entrants, 6 picks" in result.output
    listing = report_cli("report", "list")
    assert listing.exit_code == 0 and "imported" in listing.output
    assert "pool_report.csv" in listing.output
    standings = report_cli("standings")
    assert standings.exit_code == 0 and "as of week 1" in standings.output
    assert "Reported total TDs" in standings.output and "Quarter One" in standings.output
    assert "Chris K." in standings.output and "Pat" in standings.output
    assert "computed" not in standings.output.lower()
    frame = entrants.parse_csv(REFERENCE.read_bytes()).assign(week=2)
    frame.loc[frame.entrant.eq("Pat"), "reported_total"] = "12"
    frame.loc[frame.entrant.eq("Chris K."), "reported_total"] = "13"
    path = tmp_path / "week2.csv"
    path.write_text(frame.to_csv(index=False))
    imported = report_cli("report", "import", str(path))
    assert imported.exit_code == 0 and "games complete" in imported.output
    assert "as of week 2" in report_cli("standings").output
    assert "as of week 1" in report_cli("standings", "--week", "1").output
    assert len(entrants.reported_totals(report_db, 2026)) == 4


@pytest.mark.parametrize(
    "raw, extra, message, written",
    [
        (b"not a report", [], "missing required columns", False),
        (REFERENCE.read_bytes(), ["--format", "pdf"], "Unknown report format", False),
        (REFERENCE.read_bytes(), ["--week", "2"], "disagrees", False),
        (REFERENCE.read_bytes().replace(b"1,", b"19,"), ["--week", "19"], "outside", False),
        (REFERENCE.read_bytes().replace(b"Quarter One", b"Mystery Player"), [], "no match", True),
        (REFERENCE.read_bytes().replace(b"Quarter One", b"Quarter"), [], "ambiguous", True),
    ],
)
def test_cli_nonzero_failures_keep_archive(
    report_cli, report_db, tmp_path, raw, extra, message, written
):
    path = tmp_path / "delivered.csv"
    path.write_bytes(raw)
    result = report_cli("report", "import", str(path), "--week", "1", *extra)
    assert result.exit_code == 1 and message in result.output
    assert "observation 1" in result.output
    assert count(report_db, "input_observations") == 1
    assert count(report_db, "pool_picks") == (6 if written else 0)
    assert "review needed" in report_cli("report", "list").output
    if message == "ambiguous":
        assert "Quarter One" in result.output and "Quarter Two" in result.output
    if message == "no match":
        assert "Mystery Player" in result.output
        assert "(unresolved)" in report_cli("standings").output


def test_cli_check_does_not_change_standings_or_entrant_identity(report_cli, report_db, tmp_path):
    assert report_cli("report", "import", str(REFERENCE)).exit_code == 0
    path = tmp_path / "week2.csv"
    path.write_text(entrants.parse_csv(REFERENCE.read_bytes()).assign(week=2).to_csv(index=False))
    checked = report_cli("report", "import", str(path), "--check", "--me", "Chris K.")
    assert checked.exit_code == 1 and "Check only" in checked.output
    assert "my_picks mismatch" in checked.output
    assert count(report_db, "input_observations") == 2
    assert count(report_db, "pool_picks") == 6
    assert report_db.execute("SELECT SUM(is_me) FROM pool_entrants").fetchone()[0] == 0
    assert "as of week 1" in report_cli("standings").output
    assert "check" in report_cli("report", "list").output


def test_cli_roster_acknowledgement_and_my_pick_comparison(report_cli, report_db, tmp_path):
    result = report_cli("report", "import", str(REFERENCE), "--me", "Chris K.")
    assert result.exit_code == 1 and "my_picks mismatch" in result.output
    assert "(me)" in report_cli("standings").output
    assert count(report_db, "my_picks") == 0
    path = tmp_path / "renamed.csv"
    path.write_bytes(REFERENCE.read_bytes().replace(b"Pat", b"Robin"))
    rejected = report_cli("report", "import", str(path))
    assert rejected.exit_code == 1 and "Added: robin" in rejected.output
    assert "Removed: pat" in rejected.output and "--allow-roster-change" in rejected.output
    # Matching recorded picks remove the independent self-comparison failure.
    with report_db:
        report_db.execute(
            "INSERT INTO my_picks(season, week, slot, player_id, player_name, recorded_at) "
            "SELECT season, week, slot, player_id, player_name, '2026-09-10T00:00:00Z' "
            "FROM pool_picks WHERE entrant_id = 'chris k'"
        )
    accepted = report_cli("report", "import", str(path), "--allow-roster-change")
    assert accepted.exit_code == 0, accepted.output


def test_cli_empty_reports_and_missing_path_are_clear(report_cli):
    assert "No archived pool reports" in report_cli("report", "list").output
    assert "No imported pool standings" in report_cli("standings").output
    result = report_cli("report", "import", "/no/such/report.csv")
    assert result.exit_code == 1 and "Cannot read report" in result.output


def test_cli_check_success_and_missing_totals_remain_unknown(report_cli, report_db, tmp_path):
    checked = report_cli("report", "import", str(REFERENCE), "--check")
    assert checked.exit_code == 0 and "Check only" in checked.output
    assert count(report_db, "pool_picks") == 0
    raw = entrants.parse_csv(REFERENCE.read_bytes()).drop(columns=entrants.TOTAL_COLUMNS)
    path = tmp_path / "no_totals.csv"
    path.write_text(raw.to_csv(index=False))
    assert report_cli("report", "import", str(path)).exit_code == 0
    shown = report_cli("standings")
    assert shown.exit_code == 0 and "— means not supplied" in shown.output
    assert entrants.reported_totals(report_db, 2026)[entrants.TOTAL_COLUMNS].isna().all().all()
