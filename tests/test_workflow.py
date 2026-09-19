"""Acceptance tests for live eligibility, recording and complete pool scoring."""

import sqlite3
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from typer.testing import CliRunner

from pool import db, ingest, projections, results, scoring, state
from pool.cli import app
from pool.optimizer import plan_slot
from pool.recommend import advise_slot
from pool.research import backtest
from tests.support.frames import proj_row
from tests.support.local import end, play, record

runner = CliRunner()


@pytest.mark.parametrize("offset, expected", [(-1, "thu"), (0, "sun"), (1, "sun")])
def test_deadline_boundary_converts_aware_time_and_preserves_future(offset, expected):
    proj = pd.DataFrame(
        [
            proj_row("thu", "Thursday", "RB", 1, 3, kickoff="2026-09-10T20:15"),
            proj_row("sun", "Sunday", "RB", 1, 1, kickoff="2026-09-13T13:00"),
            proj_row("thu", "Thursday", "RB", 2, 2, kickoff="2026-09-20T13:00"),
            proj_row("sun", "Sunday", "RB", 2, 0.1, kickoff="2026-09-20T13:00"),
        ]
    )
    decision = datetime(2026, 9, 10, 23, 15, tzinfo=UTC) + timedelta(seconds=offset)
    advice = advise_slot(proj, "RB", 1, set(), {}, now=decision)
    assert advice.recommended.player_id == expected
    if offset >= 0:
        assert advice.plan.pick_for(2).player_id == "thu"
        assert all(p.player_id != "thu" for p in advice.alternatives)
        assert not advice.hold
    # The optimizer itself never reads the clock.
    assert plan_slot(proj, "RB", 1, set()).pick_for(1).player_id == "thu"


def test_unknown_kickoffs_and_all_expired_still_keep_future_and_locks():
    proj = pd.DataFrame(
        [
            proj_row("a", "A", "QB", 1, 10) | {"kickoff_known": 0},
            proj_row("b", "B", "QB", 1, 2),
            proj_row("a", "A", "QB", 2, 10, kickoff="2026-09-20T13:00") | {"kickoff_known": 0},
        ]
    )
    now = datetime(2026, 9, 13, 12)
    advice = advise_slot(proj, "QB", 1, set(), {}, now=now)
    assert advice.recommended is None and not advice.alternatives
    assert advice.plan.pick_for(2).player_id == "a"
    locked = advise_slot(proj, "QB", 1, {"b"}, {1: "b"}, now=now)
    assert locked.locked_player == "B"
    assert locked.plan.pick_for(2).player_id == "a"


@pytest.mark.parametrize("week", [0, 99, 18])
def test_schedule_validates_actual_17_week_season(week):
    conn = db.connect(":memory:")
    with conn:
        conn.executemany(
            "INSERT INTO games(game_id, season, week, game_type, kickoff, home_team, "
            "away_team) VALUES (?, 2020, ?, 'REG', '2020-09-01T13:00', 'A', 'B')",
            [(f"g{w}", w) for w in range(1, 18)],
        )
    with pytest.raises(state.PickError, match="weeks 1-17"):
        state.record_pick(conn, 2020, week, "QB", "q", "Quarter", "QB")
    assert state.picks(conn, 2020).empty
    assert state.validate_week(conn, 2020, 17) == 17


def test_historical_inactive_lookup_and_atomic_cli_errors(local):
    conn, path = local
    with conn:
        conn.execute(
            "INSERT INTO rosters(season,week,player_id,player_name,position,team,status) "
            "VALUES (2026, 2, 'r1', 'Runner One', 'RB', 'Z', 'RET')"
        )
    assert state.historical_pool(conn, 2026, 1).set_index("player_id").loc["r1", "team"] == "A"
    for args in [
        ["--qb", "Quarter One", "--rb", "Nobody"],
        ["--qb", "Quarter One", "--rb", "Flex One"],
        ["--qb", "Quarter"],
    ]:
        out = runner.invoke(app, ["record", "--week", "1", *args, "--db", str(path)])
        assert out.exit_code == 1, out.stdout
        assert state.picks(conn, 2026).empty
    good = runner.invoke(app, ["record", "--week", "1", "--rb", "Runner One", "--db", str(path)])
    assert good.exit_code == 0, good.stdout
    assert "apparent unavailability" in good.stdout
    assert state.picks(conn, 2026).iloc[0].game_id == "g1"
    warnings = record(conn, QB="q1")
    assert any("deadline has passed" in w for w in warnings)


def test_replacement_resets_score_same_player_preserves_and_remove_subtracts(local):
    conn, _ = local
    record(conn, QB="q1")
    with conn:
        conn.execute("UPDATE my_picks SET tds = 3")
    record(conn, QB="q1")
    assert state.picks(conn, 2026).tds.iloc[0] == 3
    record(conn, QB="q2")
    assert pd.isna(state.picks(conn, 2026).tds.iloc[0])
    assert state.used_ids(conn, 2026) == {"q2"}
    state.remove_pick(conn, 2026, 1, "QB")
    assert state.picks(conn, 2026).tds.sum() == 0


def test_atomic_reuse_rejection_preserves_existing_slots(local):
    conn, _ = local
    record(conn, QB="q1")
    before = state.picks(conn, 2026)
    with pytest.raises(state.PickError, match="already used"):
        state.record_picks(
            conn,
            2026,
            2,
            [
                dict(slot="RB", player_id="r1", player_name="Runner", position="RB"),
                dict(slot="QB", player_id="q1", player_name="Quarter", position="QB"),
            ],
        )
    pd.testing.assert_frame_equal(state.picks(conn, 2026), before)


def test_all_touchdown_kinds_deduplicate_ignore_overlaps_conversions_and_negation(local):
    conn, _ = local
    rows = [
        play(pid="receiver", pass_touchdown=1, passer_player_id="q1", play_type="pass"),
        play(pid="rusher", play_id=2),
        play(pid="kickreturn", play_id=3, play_type="kickoff", special_teams_tds=1),
        play(pid="puntreturn", play_id=4, play_type="punt"),
        play(pid="offrecover", play_id=5, fumble_recovery_1_player_id="offrecover"),
        play(pid="defrecover", play_id=6, return_touchdown=1, td_team="B"),
        play(
            pid="lateral",
            play_id=7,
            receiver_player_id="wrong_receiver",
            pass_touchdown=1,
            passer_player_id="q1",
            play_type="pass",
        ),
        play(
            pid="conversion",
            play_id=8,
            two_point_attempt=1,
            pass_touchdown=1,
            passer_player_id="q1",
        ),
        play(pid="pat", play_id=9, extra_point_attempt=1),
        play(pid="negated", play_id=10, play_type="no_play"),
        play(pid="negated2", play_id=11, no_play=1),
        end(),
    ]
    rows.append(rows[2].copy())
    count = scoring.import_touchdowns(conn, 2026, pd.DataFrame(rows))
    assert count == 9  # seven scorers + two throwing credits
    totals = scoring.touchdown_totals(conn, 2026).set_index("player_id").pool_td.to_dict()
    assert totals == dict(
        q1=2,
        receiver=1,
        rusher=1,
        kickreturn=1,
        puntreturn=1,
        offrecover=1,
        defrecover=1,
        lateral=1,
    )
    assert scoring.coverage(conn, 2026).set_index("game_id").loc["g1", "complete"] == 1
    assert scoring.import_touchdowns(conn, 2026, pd.DataFrame(rows)) == count


@pytest.mark.parametrize(
    "rows, reason",
    [
        ([play()], "end-of-game"),
        ([play(), end(home=27)], "terminal scores"),
        ([play(pid=None), end()], "identities unresolved"),
        ([play(pass_touchdown=1), end()], "identities unresolved"),
        ([play(), play(pid="other"), end()], "conflicting duplicate"),
    ],
)
def test_incomplete_game_does_not_claim_zero(local, rows, reason):
    conn, _ = local
    record(conn, RB="r1")
    scoring.import_touchdowns(conn, 2026, pd.DataFrame(rows))
    result = results.pick_results(conn, 2026).iloc[0]
    assert reason in result.pending_reason
    assert pd.isna(result.tds)


def test_score_partial_week_zero_inactive_and_unresolved(local):
    conn, path = local
    record(conn, QB="q1", RB="r1", FLEX="f1")
    scoring.import_touchdowns(
        conn, 2026, pd.DataFrame([end(), end("g2", 0, 0, desc="in progress")])
    )
    result = results.pick_results(conn, 2026).set_index("slot")
    assert result.loc["QB", "tds"] == result.loc["RB", "tds"] == 0
    assert result.loc["FLEX", "pending_reason"] == "no end-of-game marker"
    out = runner.invoke(app, ["picks", "--db", str(path)])
    assert out.exit_code == 0 and "1 pending" in out.stdout and "(incomplete)" in out.stdout
    with conn:
        conn.execute("UPDATE my_picks SET game_id = NULL WHERE slot = 'FLEX'")
    # g2 has no end-of-game marker, so week 1 is not final and we still do not know.
    assert (
        results.pick_results(conn, 2026).set_index("slot").loc["FLEX", "pending_reason"]
        == "game unresolved"
    )
    # Once the week is final, a pick with no game is the pool's own zero, not a wait.
    scoring.import_touchdowns(conn, 2026, pd.DataFrame([play(), end(), end("g2", 0, 0)]))
    flex = results.pick_results(conn, 2026).set_index("slot").loc["FLEX"]
    assert flex.pending_reason == "" and flex.tds == 0


def test_picks_are_scored_when_read_and_follow_a_correction(local, monkeypatch):
    """No scoring step to run: a finished game counts on the next read, and so does a
    result corrected upstream. Nothing reaches for the network to do it."""
    conn, path = local
    record(conn, RB="r1")

    def no_network(*args, **kw):
        pytest.fail("reading picks attempted a refresh")

    monkeypatch.setattr(ingest, "refresh", no_network)
    scoring.import_touchdowns(conn, 2026, pd.DataFrame([play(), end()]))
    for command in ("picks", "score"):
        out = runner.invoke(app, [command, "--db", str(path)])
        assert out.exit_code == 0, out.stdout
        assert "Season 2026 subtotal: 1 TDs; 0 pending" in out.stdout
    scoring.import_touchdowns(conn, 2026, pd.DataFrame([end()]))
    assert results.pick_results(conn, 2026).tds.iloc[0] == 0
    # A result that goes missing is pending again, not the last value that was seen.
    with conn:
        conn.execute("DELETE FROM game_results")
    pending = results.pick_results(conn, 2026).iloc[0]
    assert pd.isna(pending.tds) and pending.pending_reason == "scoring feed missing"


def test_actual_stat_game_overrides_recorded_team_and_schedule_correction_invalidates(local):
    conn, _ = local
    record(conn, FLEX="f1")  # roster associates g2
    with conn:
        conn.execute(
            "INSERT INTO player_weeks(season,week,season_type,player_id,player_name,"
            "position,team,game_id) VALUES (2026,1,'REG','f1','Flex One','WR','A','g1')"
        )
    scoring.import_touchdowns(conn, 2026, pd.DataFrame([play(pid="f1"), end()]))
    result = results.pick_results(conn, 2026)
    assert result.tds.iloc[0] == 1
    assert result.game_id.iloc[0] == "g1"
    with conn:
        conn.execute("UPDATE games SET home_score = 29 WHERE game_id = 'g1'")
    result = results.pick_results(conn, 2026).iloc[0]
    assert pd.isna(result.tds) and "terminal scores" in result.pending_reason


def test_shared_aggregation_includes_roster_only_scorers_and_legacy_stays_unknown(local):
    conn, _ = local
    with conn:
        conn.execute(
            "INSERT INTO player_weeks(season,week,season_type,player_id,player_name,"
            "position,team,pass_td) VALUES (2026,1,'REG','q1','Quarter One','QB','A',2)"
        )
    legacy = scoring.pool_history(conn, 2026)
    assert legacy.pool_td.isna().all()
    assert projections.per_player_totals(legacy).loc["q1", "tds"] == 2
    assert backtest.scored_weeks(conn, 2026) == []
    with pytest.raises(ValueError, match="Incomplete touchdown coverage"):
        backtest.run_season(conn, 2026, ["hindsight"], weeks=[1])
    scoring.import_touchdowns(conn, 2026, pd.DataFrame([play(pid="f2"), end(), end("g2", 0, 0)]))
    history = scoring.pool_history(conn, 2026).set_index("player_id")
    assert history.loc["f2", "pool_td"] == 1 and history.loc["q1", "pool_td"] == 0
    assert projections.per_player_totals(history.reset_index()).loc["f2", "tds"] == 1
    assert backtest.actual_tds(conn, 2026)[(1, "f2")] == 1
    assert backtest.hindsight(conn, 2026, [1]).total == 1


def test_populated_legacy_migration_is_idempotent_and_leaves_ambiguous_games_unresolved(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(db.SCHEMA)
    with conn:
        conn.executemany(
            "INSERT INTO games(game_id,season,week,game_type,kickoff,home_team,away_team) "
            "VALUES (?,2025,?,'REG','2025-09-01T13:00','A','B')",
            [("unique", 1), ("ambiguous1", 2), ("ambiguous2", 2)],
        )
        conn.executemany(
            "INSERT INTO player_weeks(season,week,season_type,player_id,player_name,"
            "position,team,pass_td) VALUES (2025,?,'REG',?,'Quarter','QB','A',3)",
            [(1, "q1"), (2, "q2")],
        )
        conn.executemany(
            "INSERT INTO my_picks VALUES (2025,?,'QB',?,'Quarter','legacy stamp',3)",
            [(1, "q1"), (2, "q2")],
        )
    conn.close()
    for _ in range(2):
        conn = db.connect(path)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == max(db.MIGRATIONS)
        picks = state.picks(conn, 2025).set_index("week")
        assert picks.tds.sum() == 6
        assert picks.loc[1, "game_id"] == "unique" and pd.isna(picks.loc[2, "game_id"])
        assert conn.execute("SELECT COUNT(*) FROM touchdown_credits").fetchone()[0] == 0
        assert not scoring.coverage(conn, 2025).complete.any()
        assert not conn.execute("SELECT MAX(kickoff_known) FROM games").fetchone()[0]
        conn.close()


def test_failed_migration_rolls_back_ddl(tmp_path, monkeypatch):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(db.SCHEMA)
    monkeypatch.setattr(db, "MIGRATIONS", {1: [*db.MIGRATIONS[1], "INVALID SQL"]})
    with pytest.raises(sqlite3.OperationalError):
        db.migrate(conn)
    assert "game_id" not in [r[1] for r in conn.execute("PRAGMA table_info(my_picks)")]
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 0


@pytest.mark.parametrize(
    "command",
    [
        ["record", "--qb", "Quarter One"],
        ["score"],
        ["recommend"],
        ["plan"],
        ["players", "--pos", "QB"],
    ],
)
@pytest.mark.parametrize("week", ["0", "99"])
def test_cli_week_validation_leaves_picks_unchanged(local, command, week):
    conn, path = local
    record(conn, QB="q1")
    before = state.picks(conn, 2026)
    out = runner.invoke(app, [*command, "--week", week, "--db", str(path)])
    assert out.exit_code == 1 and "outside the 2026 season" in out.stdout
    pd.testing.assert_frame_equal(state.picks(conn, 2026), before)


def test_reading_scores_writes_nothing_and_ignores_the_old_cache(local):
    """`my_picks.tds` was the cache `pool score` filled. It is still in the schema, and a
    value left in it is ignored rather than shown."""
    conn, path = local
    record(conn, QB="q1", RB="r1")
    state.record_pick(conn, 2026, 2, "QB", "q2", "Quarter Two", "QB")
    scoring.import_touchdowns(conn, 2026, pd.DataFrame([play(), end()]))
    with conn:
        conn.execute("UPDATE my_picks SET tds = 9")
        conn.execute(
            "CREATE TRIGGER no_writes BEFORE UPDATE ON my_picks "
            "BEGIN SELECT RAISE(ABORT, 'picks were written'); END"
        )
    before = state.picks(conn, 2026)
    week = results.pick_results(conn, 2026, week=1).set_index("slot")
    assert list(week.tds) == [0, 1] and set(week.week) == {1}
    for command in (["picks"], ["score", "--week", "1"]):
        out = runner.invoke(app, [*command, "--db", str(path)])
        assert out.exit_code == 0, out.stdout
        assert "Week 1 subtotal: 1 TDs; 0 pending" in out.stdout
    pd.testing.assert_frame_equal(state.picks(conn, 2026), before)


def test_score_is_kept_as_a_hidden_name_for_picks():
    import typer

    commands = typer.main.get_command(app).commands
    assert commands["score"].hidden and not commands["picks"].hidden


def test_candidate_limit_cannot_be_consumed_by_expired_cells():
    proj = pd.DataFrame(
        [
            proj_row("old", "Old", "RB", 1, 10, kickoff="2026-09-10T20:15"),
            proj_row("open", "Open", "RB", 1, 1),
        ]
    )
    unavailable = state.unavailable_cells(proj, 1, datetime(2026, 9, 11))
    plan = plan_slot(proj, "RB", 1, set(), unavailable=unavailable, max_players=1)
    assert plan.pick_for(1).player_id == "open"


def test_all_expired_cli_displays_future_plan_and_used_status(local, monkeypatch):
    conn, path = local
    with conn:
        conn.executemany(
            "INSERT INTO player_weeks(season,week,season_type,player_id,player_name,"
            "position,team,pass_td) VALUES (2026,1,'REG',?,?,'QB','A',1)",
            [("q1", "Quarter One"), ("q2", "Quarter Two")],
        )
    monkeypatch.setenv("COLUMNS", "200")
    original_now = state.eastern_now
    monkeypatch.setattr(
        state,
        "eastern_now",
        lambda now=None: datetime(2026, 9, 14) if now is None else original_now(now),
    )
    out = runner.invoke(app, ["recommend", "--week", "1", "--db", str(path)])
    assert out.exit_code == 0, (out.stdout, out.exception)
    assert "deadlines have passed" in out.stdout and "Future plan: week 2" in out.stdout
    plan = runner.invoke(app, ["plan", "--week", "1", "--db", str(path)])
    assert plan.exit_code == 0 and "Quarter" in plan.stdout
    record(conn, QB="q1")
    shown = runner.invoke(app, ["players", "--pos", "QB", "--week", "1", "--db", str(path)])
    assert shown.exit_code == 0
    assert "already used" in shown.stdout and "deadline passed" in shown.stdout


def test_final_marker_is_authoritative_over_late_inserted_play_id(local):
    conn, _ = local
    timeout = play(
        play_id=1001, touchdown=0, play_type="no_play", desc="Timeout #3", total_home_score=7
    )
    scoring.import_touchdowns(conn, 2026, pd.DataFrame([timeout, play(), end()]))
    assert scoring.coverage(conn, 2026).set_index("game_id").loc["g1", "complete"] == 1
    assert scoring.touchdown_totals(conn, 2026).pool_td.sum() == 1
