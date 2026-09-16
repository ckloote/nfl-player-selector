"""Computed pool standings share personal scoring and preserve incomplete evidence."""

import pandas as pd
import pytest

from pool import scoring
from tests import test_workflow as workflow
from tests.test_workflow import end, play, record


@pytest.fixture
def local(tmp_path):
    yield from workflow.local.__wrapped__(tmp_path)


@pytest.mark.parametrize(
    "rows,pid,slot,expected",
    [([play(), end()], "r1", "RB", 1), ([end()], "q1", "QB", 0), ([play()], "r1", "RB", None)],
)
def test_shared_score_matches_personal_pick(local, rows, pid, slot, expected):
    conn, _ = local
    record(conn, **{slot: pid})
    scoring.import_touchdowns(conn, 2026, pd.DataFrame(rows))
    board = scoring.score_board(conn, 2026)
    game = scoring.resolve_pick_game(conn, 2026, 1, pid, "g1")
    shared = scoring.score_pick(board, 1, pid, game)
    mine = scoring.pick_results(conn, 2026, recompute=True).iloc[0]
    assert shared.tds == expected
    assert shared.pending == mine.pending_reason
    if expected is None:
        assert pd.isna(mine.tds)
    else:
        assert type(shared.tds) is int
        assert shared.tds == mine.tds


DEFAULT_PICKS = {
    "Chris": ("Quarter One", "Runner One", "Flex One"),
    "Pat": ("Quarter One", "", "Flex Two"),
    "Jamie": ("Quarter Two", "", "Flex One"),
}


def report(conn, week=1, picks=None):
    from pool import entrants

    frame = pd.DataFrame(
        [
            dict(
                week=week,
                entrant=name,
                slot=slot,
                player_name=player,
                reported_week=99,
                reported_total=123,
                reported_rank=7,
            )
            for name, players in (DEFAULT_PICKS if picks is None else picks).items()
            for slot, player in zip(("QB", "RB", "FLEX"), players, strict=True)
        ]
    )
    result = entrants.import_report(
        conn,
        2026,
        week,
        frame.to_csv(index=False).encode(),
        me="Chris",
        allow_roster_change=True,
    )
    assert result.written, result.errors
    return result


@pytest.fixture
def pool(local):
    conn, path = local
    scoring.import_touchdowns(
        conn,
        2026,
        pd.DataFrame(
            [
                play(pass_touchdown=1, passer_player_id="q1"),
                play(pid="f2", play_id=2, play_type="punt", desc="punt return TOUCHDOWN"),
                play(pid="f2", play_id=3, two_point_attempt=1),
                end(),
                end("g2", 0, 0),
            ]
        ),
    )
    report(conn)
    return conn, path


def test_entrant_scores_share_counts_reasons_and_actual_game(pool):
    from pool import standings

    conn, _ = pool
    record(conn, QB="q1", RB="r1", FLEX="f1")
    personal = scoring.pick_results(conn, 2026, recompute=True).set_index("slot")
    reported = standings.entrant_scores(conn, 2026, weeks=[1])
    for pick in reported:
        if pick.entrant_id == "chris":
            assert pick.tds == personal.loc[pick.slot, "tds"]
            assert pick.pending == personal.loc[pick.slot, "pending_reason"]
    assert [p.tds for p in reported if p.player_id == "q1"] == [1, 1]
    assert next(p for p in reported if p.player_id == "f2").tds == 1
    assert next(p for p in reported if p.player_id == "f1").tds == 0
    with conn:
        conn.execute(
            "INSERT INTO player_weeks(season,week,season_type,player_id,player_name,"
            "position,team,game_id) VALUES (2026,1,'REG','f1','Flex One','WR','A','g1')"
        )
    assert all(
        p.game_id == "g1" for p in standings.entrant_scores(conn, 2026) if p.player_id == "f1"
    )
    with conn:
        conn.execute("UPDATE games SET home_score = 29 WHERE game_id = 'g1'")
    personal = scoring.pick_results(conn, 2026).set_index("slot")
    for pick in standings.entrant_scores(conn, 2026):
        if pick.entrant_id == "chris":
            assert pick.tds is None
            assert pick.pending == personal.loc[pick.slot, "pending_reason"]


def test_final_ties_reported_passthrough_and_independent_used_pools(pool):
    from pool import standings

    conn, _ = pool
    board = standings.board(conn, 2026)
    assert board.as_of == board.week == 1 and board.final
    assert [r.rank for r in board.rows] == [1, 1, 3]
    assert [r.share for r in board.rows] == [0.5, 0.5, 0]
    assert [r.display_name for r in board.leaders] == ["Chris", "Pat"]
    assert [r.season_total.tds for r in board.rows] == [2, 2, 0]
    assert all(
        (r.reported_week, r.reported_total, r.reported_rank) == (99, 123, 7) for r in board.rows
    )
    used = standings.used_pools(conn, 2026)
    assert used["chris"].player_ids == {"q1", "r1", "f1"}
    assert used["pat"].player_ids == {"q1", "f2"}
    assert all(u.complete for u in used.values())
    assert standings.remaining_counts(conn, 2026, 1, "pat") == {"QB": 1, "RB": 1, "FLEX": 1}
    no_pick = next(
        p for p in standings.entrant_scores(conn, 2026) if p.entrant_id == "pat" and p.slot == "RB"
    )
    assert no_pick.tds == 0 and no_pick.status == "final"


def test_imported_unfinished_week_spends_players_but_does_not_change_ranks(pool):
    from pool import standings

    conn, _ = pool
    report(
        conn,
        2,
        {"Chris": ("Quarter Two", "", ""), "Pat": ("Quarter One", "", ""), "Jamie": ("", "", "")},
    )
    result = standings.board(conn, 2026)
    assert result.week == 2 and result.as_of == 1 and result.used_through == 2
    assert result.final and [r.season_total.tds for r in result.rows] == [2, 2, 0]
    assert len(result.in_progress) == 9
    pending = [p for p in result.in_progress if p.status == "pending"]
    assert len(pending) == 2 and all(p.tds is None for p in pending)
    assert result.rows[0].week_total.incomplete
    assert result.rows[0].week_total.pending == 1
    used = standings.used_pools(conn, 2026)
    assert used["chris"].player_ids == {"q1", "q2", "r1", "f1"}
    assert used["pat"].repeats == {"q1": (1, 2)}
    assert len(used["pat"].player_ids) == 2
    assert standings.used_pools(conn, 2026, through=1)["pat"].repeats == {}
    assert "q2" not in standings.used_pools(conn, 2026, through=1)["chris"].player_ids
    old_display = standings.board(conn, 2026, week=1)
    assert old_display.week == old_display.as_of == 1 and old_display.used_through == 2


def test_unknown_name_is_not_a_zero_or_a_pending_game(pool):
    from pool import standings

    conn, _ = pool
    report(conn, picks=DEFAULT_PICKS | {"Pat": ("Quarter One", "Unknown Runner", "Flex Two")})
    result = standings.board(conn, 2026)
    assert not result.final
    pat = next(r for r in result.rows if r.entrant_id == "pat")
    assert pat.season_total.unresolved == 1 and pat.season_total.pending == 0
    assert pat.week_total.incomplete and pat.season_total.incomplete
    assert pat.used.unknown == 1 and not pat.used.complete
    assert pat.used.player_ids == {"q1", "f2"}
    pick = next(p for p in pat.picks if p.slot == "RB")
    assert pick.status == "unresolved" and pick.tds is None


def test_missing_entrant_week_and_unresolved_game_remain_incomplete(pool):
    from pool import standings

    conn, _ = pool
    report(conn, picks={k: v for k, v in DEFAULT_PICKS.items() if k != "Jamie"})
    with conn:
        conn.execute(
            "UPDATE pool_picks SET game_id = NULL WHERE entrant_id = 'pat' AND slot = 'QB'"
        )
    result = standings.board(conn, 2026)
    assert result.as_of == 1 and not result.final
    jamie = next(r for r in result.rows if r.entrant_id == "jamie")
    assert jamie.season_total.missing == 3 and jamie.week_total.missing == 3
    assert jamie.used.missing_weeks == (1,) and not jamie.used.complete
    pat = next(r for r in result.rows if r.entrant_id == "pat")
    assert pat.season_total.pending == 1
    pick = next(p for p in pat.picks if p.slot == "QB")
    assert pick.tds is None and pick.pending == "game unresolved"


def test_resolved_prefix_cannot_skip_a_week(pool):
    from pool import standings

    conn, _ = pool
    with conn:
        conn.execute(
            "INSERT INTO games(game_id,season,week,game_type,home_team,away_team,"
            "home_score,away_score,kickoff) VALUES "
            "('g4',2026,3,'REG','A','B',0,0,'2026-09-27T13:00')"
        )
        conn.execute("INSERT INTO game_results VALUES ('g4',2026,3,1,'complete',0,0,'now')")
    assert scoring.complete_weeks(conn, 2026) == [1, 3]
    assert scoring.resolved_through(conn, 2026) == 1
    report(conn, 3)
    assert standings.board(conn, 2026).as_of == 1
    with conn:
        conn.execute("UPDATE games SET home_score=0,away_score=0 WHERE game_id='g3'")
        conn.execute("INSERT INTO game_results VALUES ('g3',2026,2,1,'complete',0,0,'now')")
    result = standings.board(conn, 2026)
    assert result.as_of == 3 and not result.final
    assert all(r.season_total.missing == 3 for r in result.rows)
    with conn:
        conn.execute("DELETE FROM game_results WHERE week=1")
    assert scoring.resolved_through(conn, 2026) is None
    result = standings.board(conn, 2026)
    assert result.as_of is None and not result.leaders and not result.final
    assert all(r.rank is None and r.share == 0 for r in result.rows)


def test_recorded_future_picks_and_cache_nudge_without_writing(pool):
    from datetime import datetime

    from pool import standings, state

    conn, _ = pool
    record(conn, QB="q1", RB="r1", FLEX="f1")
    state.record_picks(
        conn,
        2026,
        2,
        [dict(slot="QB", player_id="q2", player_name="Quarter Two", position="QB")],
        now=datetime(2026, 9, 15),
    )
    before = state.picks(conn, 2026).copy()
    result = standings.board(conn, 2026)
    assert result.cache_stale and len(result.in_progress) == 1
    pick = result.in_progress[0]
    assert pick.source == "recorded" and pick.player_id == "q2" and pick.week == 2
    pd.testing.assert_frame_equal(state.picks(conn, 2026), before)
    personal = scoring.pick_results(conn, 2026, week=1, recompute=True)
    result = standings.board(conn, 2026)
    assert not result.cache_stale
    assert result.rows[0].season_total.tds == int(personal.tds.sum())
    with conn:
        conn.execute("UPDATE my_picks SET tds=9 WHERE week=1 AND slot='QB'")
    assert standings.board(conn, 2026).cache_stale
    report(conn, 2)
    result = standings.board(conn, 2026)
    assert result.report_conflicts
    assert all(p.source == "reported" for p in result.in_progress)
    assert (
        next(p for p in result.in_progress if p.entrant_id == "chris" and p.slot == "QB").player_id
        == "q1"
    )


def test_ties_below_first_do_not_split_pot(pool):
    from pool import standings

    conn, _ = pool
    report(
        conn, picks={"Chris": DEFAULT_PICKS["Chris"], "Jamie": ("", "", ""), "Pat": ("", "", "")}
    )
    result = standings.board(conn, 2026)
    assert [r.rank for r in result.rows] == [1, 2, 2]
    assert [r.share for r in result.rows] == [1, 0, 0]
    assert len(result.leaders) == 1
