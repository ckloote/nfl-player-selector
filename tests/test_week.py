"""`pool week`: what the weekly screen asks for, what it saves, and the line it has you paste."""

import io
import shlex
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import time_machine
from rich.console import Console
from typer.testing import CliRunner

from pool import cli, db, freshness, ingest, scoring, weekly
from pool.cli import app
from pool.recommend import advise_week
from tests.conftest import proj_row
from tests.test_workflow import end, play

runner = CliRunner()
EASTERN = ZoneInfo("America/New_York")

# Every kind of day week 12 has -- Wednesday, two Thursday kickoffs, Friday -- and then
# Sunday and Monday, set in week 2 so the rest of the fixture stays small. Kickoffs are
# Eastern wall clock, as the schedule stores them.
GAMES = [
    ("w1", 1, "2026-09-13T13:00", "XA", "XB", 21, 14),
    ("wed", 2, "2026-09-16T20:00", "WA", "WB", None, None),
    ("thu1", 2, "2026-09-17T13:00", "TA", "TB", None, None),
    ("thu2", 2, "2026-09-17T20:20", "UA", "UB", None, None),
    ("fri", 2, "2026-09-18T15:00", "FA", "FB", None, None),
    ("sun", 2, "2026-09-20T13:00", "SA", "SB", None, None),
    ("mon", 2, "2026-09-21T20:15", "MA", "MB", None, None),
    ("w3", 3, "2026-09-27T13:00", "XA", "XB", None, None),
]
KICKOFF = {gid: kickoff for gid, _, kickoff, *_ in GAMES}

# (id, name, slot, position, team, game, week-2 rate). Every week-3 rate is 0.3, so spending
# a player now costs what he would have been worth there.
#   QB: the Wednesday passer is 2.0 TD clear of any later option -- commit, Wednesday.
#   RB: the Thursday runner is 1.2 clear of the best Sunday option -- commit, Thursday.
#   FLEX: the Friday catcher is only 0.05 ahead of the Monday one -- hold.
HOLD_WEEK = [
    ("wq", "Wednesday Passer", "QB", "QB", "WA", "wed", 3.0),
    ("sq", "Sunday Passer", "QB", "QB", "SA", "sun", 1.0),
    ("tr", "Thursday Runner", "RB", "RB", "TA", "thu1", 2.0),
    ("ur", "Late Runner", "RB", "RB", "UA", "thu2", 1.9),
    ("sr", "Sunday Runner", "RB", "RB", "SA", "sun", 0.8),
    ("fc", "Friday Catcher", "FLEX", "WR", "FA", "fri", 0.95),
    ("mc", "Monday Catcher", "FLEX", "TE", "MA", "mon", 0.90),
]


def frame(players, weeks=(2, 3)):
    rows = []
    for pid, name, slot, position, team, game, lam in players:
        for week in weeks:
            rows.append(
                proj_row(
                    pid,
                    name,
                    slot,
                    week,
                    lam if week == 2 else 0.3,
                    kickoff=KICKOFF[game] if week == 2 else KICKOFF["w3"],
                    team=team,
                    position=position,
                )
            )
    return pd.DataFrame(rows)


def at(text):
    """An Eastern wall-clock instant, the way a deadline is read."""
    return datetime.fromisoformat(text).replace(tzinfo=EASTERN)


@pytest.fixture
def week_db(tmp_path, monkeypatch):
    """A schedule shaped like week 12, a roster to record against, and projections that
    stand in for the model: `_projections` is the one seam the week reads them through."""
    path = tmp_path / "week.db"
    conn = db.connect(path)
    with conn:
        conn.executemany(
            "INSERT INTO games(game_id, season, week, game_type, kickoff, home_team, "
            "away_team, home_score, away_score, kickoff_known) "
            "VALUES (?, 2026, ?, 'REG', ?, ?, ?, ?, ?, 1)",
            GAMES,
        )
        conn.executemany(
            "INSERT INTO rosters(season, week, player_id, player_name, position, team, status) "
            "VALUES (2026, 2, ?, ?, ?, ?, 'ACT')",
            [(pid, name, pos, team) for pid, name, _, pos, team, _, _ in HOLD_WEEK],
        )
    conn.close()
    proj = frame(HOLD_WEEK)
    monkeypatch.setattr(cli, "_projections", lambda conn, season, wk: proj[proj.week >= wk])
    return path


def run_week(path, when, *args):
    with time_machine.travel(at(when), tick=False):
        result = runner.invoke(app, ["week", "--no-refresh", "--db", str(path), *args])
    assert result.exit_code == 0, (result.output, result.exception)
    return result.output


def pasted(output):
    """The printed `record` line, split the way a shell would."""
    (line,) = [row for row in output.splitlines() if " record --week " in row]
    words = shlex.split(line)
    return words[words.index("record") :]


def paste(path, output, when):
    with time_machine.travel(at(when), tick=False):
        result = runner.invoke(app, pasted(output))
    assert result.exit_code == 0, (result.output, result.exception)
    return result.output


def decisions(path):
    conn = db.connect(path)
    try:
        return [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT decision_id FROM decision_events WHERE kind = 'surface' "
                "ORDER BY event_id"
            )
        ]
    finally:
        conn.close()


def links(path):
    conn = db.connect(path)
    try:
        return {
            r["slot"]: (r["decision_id"], r["link"])
            for r in conn.execute(
                "SELECT slot, decision_id, json_extract(detail, '$.link_source') AS link "
                "FROM decision_events WHERE kind = 'submitted'"
            )
        }
    finally:
        conn.close()


# --- the view, without a database -----------------------------------------------------


def test_each_day_before_sunday_is_its_own_decision_and_the_slate_is_one():
    """Week 12's shape. From Tuesday the Wednesday commit is due and the Thursday one
    waits; once Wednesday is spent, both Thursday kickoffs are one decision; a Monday
    player goes in with Sunday's."""
    thursday_pair = [
        ("wq", "Wednesday Passer", "QB", "QB", "WA", "wed", 3.0),
        ("sq", "Sunday Passer", "QB", "QB", "SA", "sun", 1.0),
        ("tr", "Thursday Runner", "RB", "RB", "TA", "thu1", 2.0),
        ("sr", "Sunday Runner", "RB", "RB", "SA", "sun", 0.8),
        ("uc", "Late Catcher", "FLEX", "WR", "UA", "thu2", 2.0),
        ("mc", "Monday Catcher", "FLEX", "TE", "MA", "mon", 0.8),
    ]
    proj = frame(thursday_pair)

    def split(when, locked=None):
        locked = locked or {}
        views = weekly.slot_views(
            advise_week(
                proj, 2, set(locked.values()), {s: {2: p} for s, p in locked.items()}, now=at(when)
            )
        )
        now, later = weekly.due(views)
        return [v.slot for v in now], [v.slot for v in later]

    assert split("2026-09-15T12:00") == (["QB"], ["RB", "FLEX"])
    assert split("2026-09-16T21:00", {"QB": "wq"}) == (["RB", "FLEX"], [])

    slate = [
        ("sq", "Sunday Passer", "QB", "QB", "SA", "sun", 1.0),
        ("sr", "Sunday Runner", "RB", "RB", "SA", "sun", 0.8),
        ("mc", "Monday Catcher", "FLEX", "TE", "MA", "mon", 0.8),
    ]
    proj = frame(slate)
    assert split("2026-09-19T12:00") == (["QB", "RB", "FLEX"], [])


def test_a_hold_is_never_due_and_names_what_it_waits_for():
    proj = frame(HOLD_WEEK)
    views = weekly.slot_views(advise_week(proj, 2, set(), {}, now=at("2026-09-15T12:00")))
    flex = next(v for v in views if v.slot == "FLEX")
    assert flex.status == "hold" and flex.player.player_id == "fc"
    assert flex.wait_for.player_id == "mc"
    now, later = weekly.due(views)
    assert "FLEX" not in {v.slot for v in now + later}


def test_a_name_record_cannot_single_out_is_left_off_the_line():
    """Two players with one name: the line would record whichever `record` chose, or
    refuse. It leaves him out and says so, rather than printing a line that fails."""
    views = weekly.slot_views(
        advise_week(frame(HOLD_WEEK), 2, set(), {}, now=at("2026-09-15T12:00"))
    )
    qb = [v for v in views if v.slot == "QB"]
    players = pd.DataFrame(
        [
            dict(player_id="wq", player_name="Wednesday Passer", position="QB"),
            dict(player_id="xx", player_name="Wednesday Passer", position="QB"),
        ]
    )
    line, notes = weekly.record_command(2026, 2, qb, players, "d1")
    assert line is None and "cannot pick out Wednesday Passer" in notes[0]


def test_a_name_with_an_apostrophe_survives_the_shell():
    views = weekly.slot_views(
        advise_week(frame(HOLD_WEEK), 2, set(), {}, now=at("2026-09-15T12:00"))
    )
    qb = [v for v in views if v.slot == "QB"]
    object.__setattr__(qb[0].player, "player_name", "Ja'Marr Chase")
    players = pd.DataFrame([dict(player_id="wq", player_name="Ja'Marr Chase", position="QB")])
    line, _ = weekly.record_command(2026, 2, qb, players, "d1")
    words = shlex.split(line)
    assert words[words.index("--qb") + 1] == "Ja'Marr Chase"
    assert words[-2:] == ["--decision", "d1"]


# --- refresh --------------------------------------------------------------------------


@pytest.fixture
def refreshes(monkeypatch):
    calls = []

    def fake(conn, season, log=print):
        calls.append(season)
        result = ingest.RefreshResult()
        for feed in freshness.FEEDS:
            freshness.record_status(conn, season, feed, "success", "2026-09-15T16:00:00+00:00")
        return result

    monkeypatch.setattr(ingest, "refresh", fake)
    return calls


def test_stale_data_is_refreshed_first_and_fresh_data_is_not(week_db, refreshes):
    with time_machine.travel(at("2026-09-15T12:00"), tick=False):
        first = runner.invoke(app, ["week", "--db", str(week_db)])
        again = runner.invoke(app, ["week", "--db", str(week_db)])
        forced = runner.invoke(app, ["week", "--refresh", "--db", str(week_db)])
    assert first.exit_code == again.exit_code == forced.exit_code == 0, first.output
    assert refreshes == [2026, 2026], "never fetched, then fresh, then asked for"
    assert "refreshed just now" in first.output
    assert "refreshed just now" not in again.output and "data under a minute old" in again.output


def test_an_hour_later_the_schedule_is_stale_again(week_db, refreshes):
    conn = db.connect(week_db)
    with time_machine.travel(at("2026-09-15T12:00"), tick=False):
        refreshes.clear()
        ingest.refresh(conn, 2026)
        assert not weekly.needs_refresh(conn, 2026)
    with time_machine.travel(at("2026-09-15T13:01"), tick=False):
        assert weekly.needs_refresh(conn, 2026), "schedule and lines go stale after an hour"
    conn.close()


def test_a_failed_refresh_is_one_line_and_the_advice_still_prints(week_db, monkeypatch):
    def failing(conn, season, log=print):
        result = ingest.RefreshResult()
        result.failures.append("injuries 2026: HTTPError: 503")
        return result

    monkeypatch.setattr(ingest, "refresh", failing)
    with time_machine.travel(at("2026-09-15T12:00"), tick=False):
        result = runner.invoke(app, ["week", "--db", str(week_db)])
    assert result.exit_code == 0, result.output
    assert "Refresh failed for injuries 2026: HTTPError: 503" in result.output
    assert "PICK Wednesday Passer" in result.output


def test_a_finished_season_is_not_refreshed(week_db):
    conn = db.connect(week_db)
    with conn:
        conn.execute("UPDATE games SET home_score = 7, away_score = 3")
    assert not weekly.needs_refresh(conn, 2026)
    conn.close()


# --- the screen -----------------------------------------------------------------------


def test_the_screen_fits_eighty_columns_without_cutting_a_name(week_db):
    output = run_week(week_db, "2026-09-15T12:00")
    assert "…" not in output
    for _, name, *_ in HOLD_WEEK:
        assert name in output
    assert max(len(line) for line in output.splitlines() if " record " not in line) <= 80
    alternatives = [line for line in output.splitlines() if line.startswith("         ")]
    assert len(alternatives) <= 3 * weekly.ALTERNATIVES


def test_tuesday_prints_only_wednesday_s_pick_and_says_when_to_come_back(week_db):
    output = run_week(week_db, "2026-09-15T12:00")
    assert "PICK Wednesday Passer" in output and "PICK Thursday Runner" in output
    assert "HOLD Friday Catcher" in output
    assert "Wait for injury news" in output and "Monday Catcher" in output
    assert "Submit next: QB (first deadline Wed 7:00PM)" in output
    words = pasted(output)
    assert "--qb" in words and "--rb" not in words and "--flex" not in words
    assert "Waiting: RB, FLEX. Run `" in output and " week` again before Thu 12:00PM" in output


def test_thursday_then_sunday(week_db):
    """The rhythm the tool is used in: decide what plays early, submit only that, and come
    back once those games are in."""
    tuesday = run_week(week_db, "2026-09-15T12:00")
    paste(week_db, tuesday, "2026-09-15T12:05")
    tuesday_id = decisions(week_db)[-1]
    assert links(week_db)["QB"] == (tuesday_id, "named")

    thursday = run_week(week_db, "2026-09-17T10:00")
    assert "locked Wednesday Passer · pending" in thursday
    assert "Submit next: RB (first deadline Thu 12:00PM)" in thursday
    words = pasted(thursday)
    assert words[words.index("--rb") + 1] == "Thursday Runner" and "--flex" not in words
    assert "HOLD Friday Catcher" in thursday
    # Held, so decided before the Friday catcher is gone rather than the Monday one.
    assert "Waiting: FLEX." in thursday and " week` again before Fri 2:00PM" in thursday
    paste(week_db, thursday, "2026-09-17T10:05")
    thursday_id = decisions(week_db)[-1]
    assert links(week_db)["RB"] == (thursday_id, "named")

    # Thursday's game ends with a touchdown for the pick.
    conn = db.connect(week_db)
    with conn:
        conn.execute("UPDATE games SET home_score = 7, away_score = 0 WHERE game_id = 'thu1'")
    scoring.import_touchdowns(
        conn,
        2026,
        pd.DataFrame(
            [
                play(
                    "thu1", "tr", td_team="TA", posteam="TA", total_home_score=7, total_away_score=0
                ),
                end("thu1", 7, 0),
            ]
        ),
    )
    conn.close()

    saturday = run_week(week_db, "2026-09-19T12:00")
    assert "locked Thursday Runner · 1 TD, final" in saturday
    # Friday's catcher is past his deadline, so the held slot now takes the Monday one.
    assert "PICK Monday Catcher" in saturday
    assert "Changed since the Thu 10:00AM run, which said Friday Catcher" in saturday
    assert "Nothing needs submitting before Sunday." in saturday
    words = pasted(saturday)
    assert words[words.index("--flex") + 1] == "Monday Catcher"
    assert words[words.index("--decision") + 1] == decisions(week_db)[-1] != thursday_id


def test_a_fully_recorded_week_saves_no_decision_and_points_at_the_report(week_db):
    paste(week_db, run_week(week_db, "2026-09-15T12:00"), "2026-09-15T12:05")
    paste(week_db, run_week(week_db, "2026-09-17T10:00"), "2026-09-17T10:05")
    paste(week_db, run_week(week_db, "2026-09-19T12:00"), "2026-09-19T12:05")
    before = decisions(week_db)
    output = run_week(week_db, "2026-09-19T12:10")
    assert decisions(week_db) == before
    assert "Week 2 is fully recorded." in output
    assert "report import <file>" in output
    assert " record --week " not in output


def test_every_run_with_an_open_slot_saves_one_decision(week_db):
    run_week(week_db, "2026-09-15T12:00")
    run_week(week_db, "2026-09-15T12:30")
    assert len(decisions(week_db)) == 2


def test_the_pasted_line_starts_the_way_the_command_was_started(monkeypatch):
    monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)
    assert weekly.program() == "pool"
    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")
    assert weekly.program() == "uv run pool"


# --- the pot share --------------------------------------------------------------------


@pytest.fixture
def local(tmp_path):
    from tests import test_workflow as workflow

    yield from workflow.local.__wrapped__(tmp_path)


@pytest.fixture
def seeded(local):
    """Week 1's report imported over real projections, so week 2 has rivals."""
    from tests import test_winprob

    return test_winprob.seeded.__wrapped__(local)


def test_the_pot_share_is_one_line_and_one_column(seeded):
    from pool import config

    conn, path = seeded
    conn.close()
    with config.override(WINPROB_SIMS=200):
        result = runner.invoke(app, ["week", "--no-refresh", "--db", str(path)])
    assert result.exit_code == 0, result.output
    assert "Pot share (experimental) vs 2 rivals" in result.output
    assert " Share " in result.output, "a column beside expected TDs"
    assert "vs EV" not in result.output and "paired simulations" not in result.output


def test_without_an_identity_the_share_is_withheld_with_its_fix(local, monkeypatch):
    from tests import test_predictions as log

    conn, path = local
    scoring.import_touchdowns(conn, 2026, pd.DataFrame([play(), end(), end("g2", 0, 0)]))
    log._history(conn)
    log._report(conn, 1, log.WEEK1, "2026-09-14T00:00:00+00:00", me=None)
    conn.close()
    monkeypatch.setenv("COLUMNS", "200")
    result = runner.invoke(app, ["week", "--no-refresh", "--db", str(path)])
    assert result.exit_code == 0, result.output
    assert "Pot share (experimental) withheld" in result.output
    assert '--me "Your Name"' in result.output
    assert "PICK " in result.output, "the expected-TD advice is unaffected"


# --- rival predictions ----------------------------------------------------------------

# g3, week 2's only game, kicks off Sunday 2026-09-20 13:00 Eastern.
BEFORE_KICKOFF = "2026-09-19T12:00"
AFTER_KICKOFF = "2026-09-20T14:00"


def predicted(path):
    from pool import pit, predictions

    conn = db.connect(path)
    try:
        return (
            [r["week"] for r in predictions.archived(conn, 2026)],
            [r["week"] for r in pit.archived(conn, 2026)],
        )
    finally:
        conn.close()


def weekly_run(path, when, *args):
    from pool import config

    with config.override(WINPROB_SIMS=200):
        return run_week(path, when, *args)


def test_a_prediction_is_saved_before_kickoff_and_not_again_until_it_changes(seeded):
    conn, path = seeded
    conn.close()
    first = weekly_run(path, BEFORE_KICKOFF)
    assert predicted(path) == ([2], [2])
    assert "Rival predictions for week 2 saved; they count until first kickoff" in first
    assert "Sun 1:00PM" in first
    again = weekly_run(path, "2026-09-19T12:30")
    assert predicted(path) == ([2], [2]), "the same prediction is not archived twice"
    assert "unchanged since the last save" in again


def test_a_changed_prediction_supersedes_the_earlier_one(seeded):
    from pool import predictions

    conn, path = seeded
    weekly_run(path, BEFORE_KICKOFF)
    # Pat's week-1 report is corrected, which changes what Pat has left to spend.
    from tests import test_predictions as log

    corrected = dict(log.WEEK1, Pat=("Quarter Two", "", "Flex Two"))
    log._report(conn, 1, corrected, "2026-09-19T16:10:00+00:00")
    conn.close()
    weekly_run(path, "2026-09-19T12:30")
    assert predicted(path) == ([2, 2], [2, 2])
    conn = db.connect(path)
    with time_machine.travel(at(AFTER_KICKOFF), tick=False):
        chosen = predictions.scorable(conn, 2026, 2)
    assert chosen["observation_id"] == predictions.archived(conn, 2026)[-1]["observation_id"]
    conn.close()


def test_after_kickoff_a_week_with_no_prediction_says_it_will_not_be_scored(seeded):
    conn, path = seeded
    conn.close()
    missed = weekly_run(path, AFTER_KICKOFF)
    assert predicted(path) == ([], [])
    assert "No rival prediction was saved for week 2 before it could be seen" in missed


def test_after_kickoff_a_saved_prediction_is_reported_and_not_added_to(seeded):
    conn, path = seeded
    conn.close()
    weekly_run(path, BEFORE_KICKOFF)
    kept = weekly_run(path, AFTER_KICKOFF)
    assert predicted(path) == ([2], [2])
    assert "on record from before the week could be seen" in kept


def test_nothing_is_predicted_for_a_week_that_is_not_current(seeded):
    conn, path = seeded
    conn.close()
    output = weekly_run(path, BEFORE_KICKOFF, "--week", "1")
    assert predicted(path) == ([], [])
    assert "Rival predictions" not in output


def test_nothing_is_predicted_without_an_identity(local):
    from tests import test_predictions as log

    conn, path = local
    scoring.import_touchdowns(conn, 2026, pd.DataFrame([play(), end(), end("g2", 0, 0)]))
    log._history(conn)
    log._report(conn, 1, log.WEEK1, "2026-09-14T00:00:00+00:00", me=None)
    conn.close()
    output = weekly_run(path, BEFORE_KICKOFF)
    assert predicted(path) == ([], [])
    assert "Rival predictions" not in output


# --- recommend, the detailed view -----------------------------------------------------


def test_recommend_keeps_names_whole_at_eighty_columns_even_with_the_pot_share(monkeypatch):
    """The review's smoke check found "Alternati…" and "Jaxon Smith-Nji…" at 80 columns.
    With the two pot-share columns there is the least room there will ever be."""
    from pool import config, rivals
    from tests import test_winprob as winprob

    long = {
        "qb": "Jaxon Smith-Njigba Junior",
        "qz": "Jacory Croskey-Merritt",
        "qy": "Amon-Ra St. Brown",
        "q5": "Christian McCaffrey",
    }
    proj = winprob.frame(weeks=(1, 2))
    proj["player_name"] = proj.player_id.map(long).fillna(proj.player_name)
    pool = rivals.PoolState((winprob.rival("pat", 2, {"qa"}), winprob.rival("jo", 1)), 1)
    with config.override(WINPROB_SIMS=300):
        advice = advise_week(proj, 1, set(), {}, now=winprob.NOW, pool=pool)
    qb = next(a for a in advice if a.slot == "QB")
    assert qb.shares, "the widest form of the table"
    # A console of its own: setting `width` on the shared one would pin it for every test
    # after this, and those that widen it through COLUMNS would stop being able to.
    narrow = Console(width=80, record=True, file=io.StringIO())
    monkeypatch.setattr(cli, "console", narrow)
    cli._render_slot(qb, pool)
    output = narrow.export_text()
    assert "…" not in output
    for name in long.values():
        assert name in output
