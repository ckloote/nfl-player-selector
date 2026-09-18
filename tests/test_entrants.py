import hashlib
import json
import sqlite3
import zlib
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from pool import capture, config, db, entrants, freshness, predictions, snapshots, standings
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
        (lambda f: f.assign(slot=" "), "empty slot"),
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
    assert result.warnings == [entrants.IDENTITY_PROMPT], "imported without --me"
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


def test_a_name_matched_only_by_partial_or_close_spelling_is_noted_not_failed(report_db):
    """`find_player` quietly accepts one partial or close-spelling match. That is usually a
    typo finding the right player, so it imports, but it is listed so a wrong one is seen."""
    raw = (
        REFERENCE.read_bytes()
        .replace(b"1,Chris K.,RB,Runner One", b"1,Chris K.,RB,Runner")
        .replace(b"1,Pat,RB,Runner One", b"1,Pat,RB,Runer One")
        .replace(b"Quarter Two", b"QUARTER TWO")
    )
    result = import_reference(report_db, raw)
    assert result.ok and result.written and not result.unresolved
    matched = dict(matched="Runner One", player_id="r1", team="A", position="RB")
    assert result.inexact == [
        dict(entrant_id="chris k", slot="RB", player_name="Runner", **matched),
        dict(entrant_id="pat", slot="RB", player_name="Runer One", **matched),
    ]
    rows = entrants.entrant_picks(report_db, 2026, 1).set_index(["entrant_id", "slot"])
    assert rows.loc[("pat", "RB"), "player_name"] == "Runer One"
    assert rows.loc[("pat", "RB"), "player_id"] == "r1"
    assert not import_reference(report_db).inexact


@pytest.mark.parametrize("change", ["rename", "addition", "removal"])
def test_roster_changes_are_flagged_before_any_rows_change(report_db, change):
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
    assert not rejected.ok and not rejected.written and rejected.previous_week == 1
    assert bool(rejected.added) == (change != "removal")
    assert bool(rejected.removed) == (change != "addition")
    assert entrants.entrant_picks(report_db, 2026, 2).empty
    assert import_reference(report_db, raw, week=2, allow_roster_change=True).ok
    assert len(entrants.entrant_picks(report_db, 2026, 2)) == len(frame)


def test_corrected_week_removes_obsolete_rows_only_once_acknowledged(report_db):
    import_reference(report_db)
    frame = entrants.parse_csv(REFERENCE.read_bytes()).query("entrant != 'Pat'")
    raw = frame.to_csv(index=False).encode()
    assert not import_reference(report_db, raw, check=True).ok
    assert len(entrants.entrant_picks(report_db, 2026, 1)) == 6
    # A file missing an entrant is a correction or a truncation, and nothing distinguishes
    # them here. Until that is acknowledged the week keeps every pick it already had.
    rejected = import_reference(report_db, raw)
    assert not rejected.ok and not rejected.written and rejected.removed == ["pat"]
    assert len(entrants.entrant_picks(report_db, 2026, 1)) == 6
    assert len(entrants.reported_totals(report_db, 2026, 1)) == 2
    accepted = import_reference(report_db, raw, allow_roster_change=True)
    assert accepted.ok and accepted.written
    assert len(entrants.entrant_picks(report_db, 2026, 1)) == 3
    assert len(entrants.reported_totals(report_db, 2026, 1)) == 1
    assert count(report_db, "pool_entrants") == 2
    # The identity row is kept; the entrant is not. Pat has no picks left to stand on.
    assert [r.entrant_id for r in standings.board(report_db, 2026).rows] == ["chris k"]


def test_normalized_entrant_name_remains_one_identity_across_weeks(report_db):
    import_reference(report_db)
    raw = REFERENCE.read_bytes().replace(b"Chris K.", b"chris k")
    raw = entrants.parse_csv(raw).assign(week=2).to_csv(index=False).encode()
    assert import_reference(report_db, raw, week=2).ok
    assert count(report_db, "pool_entrants") == 2


def test_a_corrected_entrant_name_leaves_no_phantom_opponent(report_db):
    """The review's reproduction. A misspelt rival, corrected by re-importing the week, keeps
    an identity row with no picks behind it. That is a spelling, not an entrant: it must not
    reach the standings, the used pools, or the field the pot share is simulated against."""
    typo = REFERENCE.read_bytes().replace(b"Pat", b"Rivla")
    assert import_reference(report_db, typo, me="Chris K.").ok
    corrected = import_reference(report_db, allow_roster_change=True)
    assert corrected.ok and corrected.written
    assert corrected.added == ["pat"] and corrected.removed == ["rivla"]
    assert {r.entrant_id for r in standings.board(report_db, 2026).rows} == {"chris k", "pat"}
    assert set(standings.used_pools(report_db, 2026)) == {"chris k", "pat"}
    field = predictions.pool_state(report_db, 2026, 1).rivals
    assert [r.entrant_id for r in field] == ["pat"], "one rival, not the misspelling as well"


def test_my_own_misspelt_name_can_be_corrected_in_one_import(report_db):
    """The same correction applied to my row. The misspelling is about to have no picks, so
    it is no identity to disagree with, and `--me` moves to the corrected spelling rather
    than refusing it forever."""
    typo = REFERENCE.read_bytes().replace(b"Chris K.", b"Chris Kk")
    assert import_reference(report_db, typo, me="Chris Kk").ok
    corrected = import_reference(report_db, me="Chris K.", allow_roster_change=True)
    assert corrected.ok and corrected.written, corrected.errors
    assert [r.entrant_id for r in standings.board(report_db, 2026).rows if r.is_me] == ["chris k"]
    mine = report_db.execute("SELECT entrant_id FROM pool_entrants WHERE is_me = 1").fetchall()
    assert [r[0] for r in mine] == ["chris k"], "the corrected-away spelling is not me as well"
    # An identity with picks in another week is still established: moving it is a
    # disagreement, even in a file that leaves me out.
    frame = entrants.parse_csv(REFERENCE.read_bytes()).assign(week="2")
    raw = frame[frame.entrant.ne("Chris K.")].to_csv(index=False).encode()
    moved = import_reference(report_db, raw, week=2, me="Pat", allow_roster_change=True)
    assert not moved.ok and "disagrees with the existing identity" in moved.errors[0]


def test_an_import_without_me_is_standings_only_until_an_identity_is_set(report_db):
    """The review's reproduction. Every entrant without the flag reads as a rival, so a
    report imported without --me made me one of my own opponents. Standings need no
    identity and keep working; the pot share and the rival predictions refuse, and say
    what one import sets it."""
    result = import_reference(report_db)
    assert result.ok and result.written
    assert result.warnings == [entrants.IDENTITY_PROMPT]
    assert len(standings.board(report_db, 2026).rows) == 2
    pool = predictions.pool_state(report_db, 2026, 1)
    assert not pool.rivals and pool.withheld == (entrants.IDENTITY_PROMPT,)
    assert not pool.ready
    with pytest.raises(ValueError, match="--me"):
        predictions.rival_states(report_db, 2026, 1)
    assert import_reference(report_db, me="Chris K.").warnings == [], "set once"
    assert import_reference(report_db).warnings == [], "and kept without repeating it"
    pool = predictions.pool_state(report_db, 2026, 1)
    assert pool.ready and [r.entrant_id for r in pool.rivals] == ["pat"]


def test_me_comparison_persists_and_never_edits_my_picks(report_db):
    with report_db:
        report_db.executemany(
            "INSERT INTO my_picks(season, week, slot, player_id, player_name, recorded_at, tds) "
            "VALUES (2026, 1, ?, ?, ?, '2026-09-10T00:00:00Z', 7)",
            [("QB", "q2", "Quarter Two"), ("RB", "r1", "Runner One"), ("FLEX", "f1", "Flex One")],
        )
    before = db.read_df(report_db, "SELECT * FROM my_picks")
    first = import_reference(report_db, me="chris k")
    assert first.conflicts == [dict(slot="QB", recorded="Quarter Two", reported="Quarter One")]
    assert not first.ok and first.written and not first.unrecorded
    assert import_reference(report_db).conflicts == first.conflicts
    assert not import_reference(report_db, me="Nobody").ok
    assert not import_reference(report_db, me="Pat").ok
    pd.testing.assert_frame_equal(before, db.read_df(report_db, "SELECT * FROM my_picks"))
    assert (
        report_db.execute("SELECT entrant_id FROM pool_entrants WHERE is_me = 1").fetchone()[0]
        == "chris k"
    )


def test_check_parses_without_archiving_or_writing_anything(report_db):
    """A dry run leaves no trace. Archiving before parsing is the import's guarantee, and
    the import keeps it; a check run over a file still being typed would archive drafts."""
    result = import_reference(report_db, check=True, me="Chris K.")
    assert result.check and result.parsed and not result.written and result.picks == 6
    assert result.observation_id is None
    tables = ("input_observations", "input_payloads", "pool_entrants", "pool_picks")
    assert all(count(report_db, t) == 0 for t in (*tables, "pool_report_totals"))
    receipts = report_db.execute("SELECT COUNT(*) FROM meta WHERE key LIKE 'pool_report:%'")
    assert receipts.fetchone()[0] == 0
    assert entrants.reports(report_db, 2026).empty


def test_failed_check_leaves_nothing_but_importing_the_same_bytes_still_archives(report_db):
    raw = b"%PDF-unrecognized"
    checked = import_reference(report_db, raw, check=True)
    assert not checked.ok and checked.observation_id is None
    assert count(report_db, "input_observations") == count(report_db, "input_payloads") == 0
    assert db.get_meta(report_db, "pool_report:None") is None
    imported = import_reference(report_db, raw)
    assert not imported.ok and imported.observation_id is not None
    assert count(report_db, "input_observations") == count(report_db, "input_payloads") == 1


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


BLANK_RB = (b"1,Chris K.,RB,Runner One,,,", b"1,Chris K.,RB,,,,")


def test_blank_pick_is_stored_as_a_no_pick_but_an_absent_row_is_still_rejected():
    """A report that says an entrant picked nothing is evidence; a row that simply is not
    there is indistinguishable from a truncated file, and only the first is storable."""
    frame = entrants.parse_csv(REFERENCE.read_bytes())
    blank = frame.copy()
    blank.loc[blank.slot.eq("RB") & blank.entrant.eq("Chris K."), "player_name"] = ""
    _, picks, _ = entrants.transform_report(blank, 2026, 1)
    assert picks.set_index(["entrant_id", "slot"]).loc[("chris k", "RB"), "player_name"] is None
    absent = frame[~(frame.slot.eq("RB") & frame.entrant.eq("Chris K."))]
    with pytest.raises(ValueError, match="missing slots"):
        entrants.transform_report(absent, 2026, 1)


def test_reported_no_pick_is_not_an_unresolved_name(report_db):
    result = import_reference(report_db, REFERENCE.read_bytes().replace(*BLANK_RB))
    assert result.ok and result.written and not result.unresolved
    assert result.blanks == [dict(entrant_id="chris k", slot="RB")]
    assert "No pick reported for chris k RB" in result.warnings[0]
    row = report_db.execute(
        "SELECT player_name, player_id, game_id FROM pool_picks "
        "WHERE entrant_id = 'chris k' AND slot = 'RB'"
    ).fetchone()
    assert row["player_name"] is None and row["player_id"] is None and row["game_id"] is None
    # Distinguishable from a week nobody imported, which has no row at all.
    assert len(entrants.entrant_picks(report_db, 2026, 1)) == 6
    assert entrants.entrant_picks(report_db, 2026, 2).empty


def test_unrecorded_slots_note_while_a_contradicted_slot_fails(report_db):
    assert import_reference(report_db, me="Chris K.").ok
    noted = import_reference(report_db)
    assert len(noted.unrecorded) == 3 and not noted.conflicts and noted.ok
    with report_db:
        report_db.execute(
            "INSERT INTO my_picks(season, week, slot, player_id, player_name, recorded_at) "
            "VALUES (2026, 1, 'QB', 'q2', 'Quarter Two', '2026-09-10T00:00:00Z')"
        )
    conflicted = import_reference(report_db)
    assert conflicted.conflicts == [dict(slot="QB", recorded="Quarter Two", reported="Quarter One")]
    assert len(conflicted.unrecorded) == 2 and not conflicted.ok


def test_recorded_pick_against_a_reported_no_pick_is_a_contradiction(report_db):
    with report_db:
        report_db.execute(
            "INSERT INTO my_picks(season, week, slot, player_id, player_name, recorded_at) "
            "VALUES (2026, 1, 'RB', 'r1', 'Runner One', '2026-09-10T00:00:00Z')"
        )
    result = import_reference(report_db, REFERENCE.read_bytes().replace(*BLANK_RB), me="Chris K.")
    assert result.conflicts == [dict(slot="RB", recorded="Runner One", reported=None)]
    assert not result.ok


def test_an_unresolved_report_name_is_not_also_called_a_contradiction(report_db):
    with report_db:
        report_db.execute(
            "INSERT INTO my_picks(season, week, slot, player_id, player_name, recorded_at) "
            "VALUES (2026, 1, 'QB', 'q1', 'Quarter One', '2026-09-10T00:00:00Z')"
        )
    raw = REFERENCE.read_bytes().replace(b"Quarter One", b"Mystery Player")
    result = import_reference(report_db, raw, me="Chris K.")
    assert result.unresolved and not result.conflicts and not result.ok


def test_a_report_that_leaves_me_out_is_not_a_contradiction(report_db):
    """A present row with a blank name is the report saying I picked nobody. A file with no
    row for me says nothing about my picks at all; my absence is a roster change and is
    reported as one, so comparing slots against it would invent three contradictions."""
    assert import_reference(report_db, me="Chris K.").ok
    with report_db:
        report_db.executemany(
            "INSERT INTO my_picks(season, week, slot, player_id, player_name, recorded_at) "
            "VALUES (2026, 2, ?, ?, ?, '2026-09-19T00:00:00Z')",
            [("QB", "q1", "Quarter One"), ("RB", "r1", "Runner One"), ("FLEX", "f1", "Flex One")],
        )
    frame = entrants.parse_csv(REFERENCE.read_bytes()).assign(week="2")
    raw = frame[frame.entrant.ne("Chris K.")].to_csv(index=False).encode()
    rejected = import_reference(report_db, raw, week=2)
    assert rejected.removed == ["chris k"] and not rejected.written
    assert not rejected.conflicts and not rejected.unrecorded
    accepted = import_reference(report_db, raw, week=2, allow_roster_change=True)
    assert accepted.ok and accepted.written and not accepted.conflicts


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


def test_cli_import_list_and_reported_standings(report_cli, report_db, tmp_path, monkeypatch):
    # Imported by a relative path, so the listing's source column is the same short name
    # wherever the repository is checked out. By its absolute path, a deep enough checkout
    # pushed the table past the console width and truncated the name this test looks for.
    monkeypatch.chdir(REFERENCE.parent)
    result = report_cli("report", "import", REFERENCE.name)
    assert result.exit_code == 0, result.output
    assert "observation 1" in result.output and "2 entrants, 6 picks" in result.output
    listing = report_cli("report", "list")
    assert listing.exit_code == 0 and "imported" in listing.output
    assert "pool_report.csv" in listing.output
    standings = report_cli("standings")
    assert standings.exit_code == 0 and "as of week 1" in standings.output
    assert "Reported total TDs" in standings.output and "Quarter One" in standings.output
    assert "Chris K." in standings.output and "Pat" in standings.output
    assert "Computed standings" in standings.output
    assert "Week TDs" in standings.output and "Season TDs" in standings.output
    frame = entrants.parse_csv(REFERENCE.read_bytes()).assign(week=2)
    frame.loc[frame.entrant.eq("Pat"), "reported_total"] = "12"
    frame.loc[frame.entrant.eq("Chris K."), "reported_total"] = "13"
    path = tmp_path / "week2.csv"
    path.write_text(frame.to_csv(index=False))
    imported = report_cli("report", "import", str(path))
    assert imported.exit_code == 0 and "games complete" in imported.output
    assert "week 2 picks" in report_cli("standings").output
    assert "ranked on season totals through week 1" in report_cli("standings").output
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
    assert checked.exit_code == 0 and "Check only: nothing archived" in checked.output
    assert "Archived report as observation" not in checked.output
    assert "no recorded pick for QB" in checked.output
    assert count(report_db, "input_observations") == 1
    assert count(report_db, "pool_picks") == 6
    assert report_db.execute("SELECT SUM(is_me) FROM pool_entrants").fetchone()[0] == 0
    assert "as of week 1" in report_cli("standings").output
    assert len(entrants.reports(report_db, 2026)) == 1


def test_cli_roster_acknowledgement_and_my_pick_comparison(report_cli, report_db, tmp_path):
    # Nothing recorded yet is a note, not a failure: an import must not be red because
    # the operator has not also typed their own picks in.
    result = report_cli("report", "import", str(REFERENCE), "--me", "Chris K.")
    assert result.exit_code == 0 and "no recorded pick for QB" in result.output
    assert "my_picks mismatch" not in result.output
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


def test_cli_standings_distinguishes_a_no_pick_from_an_unresolved_name(report_cli, tmp_path):
    path = tmp_path / "blank.csv"
    path.write_bytes(REFERENCE.read_bytes().replace(*BLANK_RB))
    imported = report_cli("report", "import", str(path))
    assert imported.exit_code == 0 and "No pick reported for chris k RB" in imported.output
    shown = report_cli("standings")
    assert shown.exit_code == 0 and "(no pick)" in shown.output
    assert "(unresolved)" not in shown.output


def test_cli_notes_an_inexact_name_match_in_check_and_import(report_cli, tmp_path):
    path = tmp_path / "typo.csv"
    path.write_bytes(REFERENCE.read_bytes().replace(b"1,Pat,RB,Runner One", b"1,Pat,RB,Runer One"))
    expected = "Note: pat RB: typed 'Runer One', matched Runner One (A RB)."
    checked = report_cli("report", "import", str(path), "--check")
    assert checked.exit_code == 0 and expected in checked.output
    imported = report_cli("report", "import", str(path))
    assert imported.exit_code == 0 and expected in imported.output
    assert "Note:" not in report_cli("report", "import", str(REFERENCE)).output


def test_cli_empty_reports_and_missing_path_are_clear(report_cli):
    assert "No archived pool reports" in report_cli("report", "list").output
    assert "No imported pool standings" in report_cli("standings").output
    result = report_cli("report", "import", "/no/such/report.csv")
    assert result.exit_code == 1 and "Cannot read report" in result.output


def test_cli_check_success_and_missing_totals_remain_unknown(report_cli, report_db, tmp_path):
    checked = report_cli("report", "import", str(REFERENCE), "--check")
    assert checked.exit_code == 0 and "Check only" in checked.output
    assert count(report_db, "pool_picks") == count(report_db, "input_observations") == 0
    raw = entrants.parse_csv(REFERENCE.read_bytes()).drop(columns=entrants.TOTAL_COLUMNS)
    path = tmp_path / "no_totals.csv"
    path.write_text(raw.to_csv(index=False))
    assert report_cli("report", "import", str(path)).exit_code == 0
    shown = report_cli("standings")
    assert shown.exit_code == 0 and "— means not supplied" in shown.output
    assert entrants.reported_totals(report_db, 2026)[entrants.TOTAL_COLUMNS].isna().all().all()
