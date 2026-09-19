"""Computed pool standings share personal scoring and preserve incomplete evidence."""

import pandas as pd
import pytest

from pool import results, scoring
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
    board = results.score_board(conn, 2026)
    game = results.resolve_pick_game(conn, 2026, 1, pid, "g1")
    shared = results.score_pick(board, 1, pid, game)
    mine = results.pick_results(conn, 2026).iloc[0]
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
    personal = results.pick_results(conn, 2026).set_index("slot")
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
    personal = results.pick_results(conn, 2026).set_index("slot")
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
    # Week 1's own report is excluded: Pat's q1 and f2 were spent in week 1, not before it.
    assert standings.remaining_counts(conn, 2026, 1, "pat") == {"QB": 2, "RB": 1, "FLEX": 2}
    assert standings.remaining_counts(conn, 2026, 2, "pat") == {"QB": 1, "RB": 1, "FLEX": 1}
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
    # Jamie is reported in week 2, so she is still an entrant when week 1 drops her. Without
    # that she would have no picks at all, which is a corrected-away spelling, not a gap.
    report(conn, week=2)
    report(conn, picks={k: v for k, v in DEFAULT_PICKS.items() if k != "Jamie"})
    with conn:
        conn.execute(
            "UPDATE pool_picks SET game_id = NULL WHERE entrant_id = 'pat' AND slot = 'QB'"
        )
    result = standings.board(conn, 2026, week=1)
    assert result.as_of == 1 and not result.final
    jamie = next(r for r in result.rows if r.entrant_id == "jamie")
    assert jamie.season_total.missing == 3 and jamie.week_total.missing == 3
    assert jamie.used.missing_weeks == (1,) and not jamie.used.complete
    # Week 1 is final, so a pick with no game is the pool's zero rather than a wait --
    # but it carries a note, because a team the schedule cannot match looks identical.
    pat = next(r for r in result.rows if r.entrant_id == "pat")
    assert pat.season_total.pending == 0
    pick = next(p for p in pat.picks if p.slot == "QB")
    assert pick.tds == 0 and pick.pending == "" and pick.status == "final"
    assert pick.note == "no game that week; scored zero"


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
    assert results.resolved_through(conn, 2026) == 1
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
    assert results.resolved_through(conn, 2026) is None
    result = standings.board(conn, 2026)
    assert result.as_of is None and not result.leaders and not result.final
    assert all(r.rank is None and r.share == 0 for r in result.rows)


def test_recorded_future_picks_show_in_progress_without_writing(pool):
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
    assert len(result.in_progress) == 1
    pick = result.in_progress[0]
    assert pick.source == "recorded" and pick.player_id == "q2" and pick.week == 2
    pd.testing.assert_frame_equal(state.picks(conn, 2026), before)
    personal = results.pick_results(conn, 2026, week=1)
    assert result.rows[0].season_total.tds == int(personal.tds.sum())
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


@pytest.fixture
def invoke(pool, monkeypatch):
    from typer.testing import CliRunner

    from pool.cli import app

    monkeypatch.setenv("COLUMNS", "240")
    _, path = pool

    def run(*args):
        result = CliRunner().invoke(app, [*args, "--season", "2026", "--db", str(path)])
        assert result.exit_code == 0, result.output
        return " ".join(result.output.split())

    return run


def test_cli_computed_and_reported_columns_final_tie_and_score_parity(pool, invoke):
    conn, _ = pool
    record(conn, QB="q1", RB="r1", FLEX="f1")
    shown = invoke("standings")
    assert "week 1 picks; ranked on season totals through week 1" in shown
    assert "Chris and Pat are tied for first with 2 TDs." in shown
    # Week 2 is still unplayed, so the tie is real but the split is not settled.
    assert "A tie at the end splits the winnings 1/2 each." in shown
    assert "each takes" not in shown and "leader" not in shown
    assert "Week TDs" in shown and "Season TDs" in shown
    assert "Reported week TDs" in shown and "Reported total TDs" in shown
    assert "99" in shown and "123" in shown
    assert "Totals are touchdown counts, not points." in shown
    # Your picks are scored when they are read, so both views agree with no step between.
    assert "pool score" not in shown
    assert "Season 2026 subtotal: 2 TDs; 0 pending" in invoke("picks")
    assert "Week 1 subtotal: 2 TDs; 0 pending" in invoke("score")


def test_cli_provisional_tie_distinguishes_unresolved_names(pool, invoke):
    conn, _ = pool
    report(conn, picks=DEFAULT_PICKS | {"Pat": ("Quarter One", "Unknown Runner", "Flex Two")})
    shown = invoke("standings")
    assert "as of week 1 (provisional)" in shown
    assert (
        "As it stands Chris and Pat are tied for first with 2 TDs, with 1 unresolved name." in shown
    )
    assert "A tie at the end splits the winnings 1/2 each." in shown
    assert "each takes" not in shown and "leader" not in shown
    assert "Unknown Runner (unresolved)" in shown and "1 unknown" in shown
    assert "incomplete" in shown and "pool report import" in shown


def test_cli_single_lead_and_lower_tie_then_provisional_lead(pool, invoke):
    """A ranked week has no pending picks left, so a lead is provisional for other reasons.

    Every game in a ranked week is final by construction, and a pick with no game of its
    own now scores zero, so nothing inside the prefix is still waiting. What can still
    hold a lead open is a name that never resolved or a week nobody reported.
    """
    conn, _ = pool
    report(
        conn, picks={"Chris": DEFAULT_PICKS["Chris"], "Jamie": ("", "", ""), "Pat": ("", "", "")}
    )
    shown = invoke("standings")
    assert "Chris leads with 2 TDs." in shown and "splits" not in shown
    report(
        conn,
        picks={
            "Chris": ("Quarter One", "Runner One", "Unknown Flex"),
            "Jamie": ("", "", ""),
            "Pat": ("", "", ""),
        },
    )
    shown = invoke("standings")
    assert "As it stands Chris leads with 2 TDs, with 1 unresolved name." in shown
    assert "provisional" in shown and "splits" not in shown
    assert "pick pending" not in shown


def test_cli_in_progress_spending_repeats_and_selected_week(pool, invoke):
    conn, _ = pool
    report(conn, 2)
    shown = invoke("standings")
    assert "week 2 picks; ranked on season totals through week 1" in shown
    assert "In progress" in shown and "reported" in shown
    assert "pending: scoring feed missing" in shown
    assert "Used includes all imported picks through week 2" in shown
    assert "repeated player q1 in weeks 1, 2; counted once" in shown
    selected = invoke("standings", "--week", "1")
    assert "week 1 picks; ranked on season totals through week 1" in selected
    assert "In progress" in selected


def test_cli_recorded_future_picks_without_report(pool, invoke):
    from datetime import datetime

    from pool import state

    conn, _ = pool
    state.record_picks(
        conn,
        2026,
        2,
        [dict(slot="QB", player_id="q2", player_name="Quarter Two", position="QB")],
        now=datetime(2026, 9, 15),
    )
    shown = invoke("standings")
    assert "In progress" in shown and "recorded" in shown and "Quarter Two" in shown
    report(conn, 2)
    shown = invoke("standings")
    assert "Reported picks differ from your recorded picks" in shown
    assert "use pool report import with --me" in shown


def test_cli_no_completed_week_and_no_imports(pool, invoke):
    conn, _ = pool
    with conn:
        conn.execute("DELETE FROM game_results")
    shown = invoke("standings")
    assert "season cannot be ranked yet" in shown
    assert "In progress" in shown
    assert "leads" not in shown and "tied for first" not in shown
    with conn:
        conn.execute("DELETE FROM pool_picks")
        conn.execute("DELETE FROM pool_report_totals")
        conn.execute("DELETE FROM pool_entrants")
    shown = invoke("standings")
    assert "No imported pool standings" in shown and "Run pool report import" in shown


def test_standings_refuses_a_week_outside_the_season_rather_than_inventing_one(pool):
    """A table for a week that cannot exist reads as evidence that its report is missing.

    Every other week-taking command validates first, and stage 1's standings said plainly
    that there was nothing imported. Synthesizing three missing slots per entrant for
    week 99 invents the very gap the operator would then go looking for.
    """
    from typer.testing import CliRunner

    from pool.cli import app

    _, path = pool
    for week in ("99", "0", "-3"):
        result = CliRunner().invoke(
            app, ["standings", "--week", week, "--season", "2026", "--db", str(path)]
        )
        assert result.exit_code == 1, result.output
        assert "outside the 2026 season" in result.output
        assert "missing" not in result.output


def test_a_tie_for_first_splits_the_pot_only_once_the_season_runs_out_of_weeks(pool, invoke):
    """`final` means the ranked weeks have no gaps, which is true after the first one.

    Saying the pot is shared is a claim about money, and it is only true when no week is
    left to change the standing. The fixture's week 2 is unplayed, so the same tie is
    reported two different ways either side of that fact.
    """
    conn, _ = pool
    from pool import standings

    assert standings.board(conn, 2026).final
    assert not standings.board(conn, 2026).season_complete
    assert "A tie at the end splits the winnings 1/2 each." in invoke("standings")
    with conn:
        conn.execute("DELETE FROM games WHERE season = 2026 AND week > 1")
    assert standings.board(conn, 2026).season_complete
    shown = invoke("standings")
    assert "A tie for first splits the winnings: each takes 1/2 of the pot." in shown
    assert "leader" not in shown


def test_a_pick_with_no_game_waits_while_the_week_is_unfinished_then_scores_zero(pool, invoke):
    """Nobody picks a player who is not playing, and the pool scores it zero if they do.

    The zero waits for the week to finish, because until then "no game found" and "no game
    yet" are the same silence, and one pick left waiting forever held the whole season at
    provisional. The note exists because the other route here is a team abbreviation the
    schedule does not know, which would otherwise zero every pick on that team silently.
    """
    from pool import standings

    conn, _ = pool
    with conn:
        conn.execute("UPDATE pool_picks SET game_id = NULL WHERE entrant_id = 'pat'")
        conn.execute(
            "UPDATE game_results SET complete = 0, reason = 'no end-of-game marker' "
            "WHERE game_id = 'g2'"
        )
    pat = next(p for p in standings.entrant_scores(conn, 2026) if p.entrant_id == "pat")
    assert pat.tds is None and pat.pending == "game unresolved" and pat.note == ""

    with conn:
        conn.execute(
            "UPDATE game_results SET complete = 1, reason = 'complete' WHERE game_id = 'g2'"
        )
    scored = [p for p in standings.entrant_scores(conn, 2026) if p.entrant_id == "pat"]
    assert all(p.tds == 0 and p.pending == "" for p in scored)
    assert [p.note for p in scored if p.player_name] == ["no game that week; scored zero"] * 2
    board = standings.board(conn, 2026)
    assert next(r for r in board.rows if r.entrant_id == "pat").season_total.incomplete is False
    shown = invoke("standings")
    assert "had no week 1 game; scored 0" in shown
    assert "A bye or a team the schedule does not match." in shown
