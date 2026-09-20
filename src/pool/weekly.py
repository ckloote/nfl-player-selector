"""What to do this week, sorted out of the advice: submit, wait, or already done.

`pool week` shows the same decision `recommend` does -- both come from `cli._decide` -- in
the few lines needed to act on it. Nothing here decides anything. It sorts the advice into
what is due by the next deadline and what can wait for a later run, and writes the `record`
line to paste once the picks are in. It sits outside the enforced decision closure, so the
screen can change without moving a captured decision's fingerprint.
"""

from __future__ import annotations

import json
import os
import shlex
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from . import config, freshness, ingest, pit, predictions, results, state
from .recommend import Candidate, SlotAdvice

ALTERNATIVES = 3
MAIN_SLATE = "main slate"


def needs_refresh(conn: sqlite3.Connection, season: int, now: datetime | None = None) -> bool:
    """A feed older than the age `status` warns about, or never fetched.

    The same limits, so `pool week` refreshes exactly when the data would otherwise be
    reported stale. Automatic refresh stops only once the season's scores, touchdown
    coverage, player stats and successful feeds satisfy `ingest.settled`.
    """
    instant = freshness.utc_now(now)
    if ingest.settled(conn, season):
        return False
    fetched = {
        row["feed"]: row["last_success"]
        for row in conn.execute(
            "SELECT feed, last_success FROM feed_status WHERE season = ?", (season,)
        )
    }
    return any(
        not fetched.get(feed)
        or instant - datetime.fromisoformat(fetched[feed]) > timedelta(hours=hours)
        for feed, hours in config.FRESHNESS_HOURS.items()
    )


@dataclass(frozen=True)
class Previous:
    """What an earlier run this week recommended for one slot."""

    player_id: str | None
    player_name: str | None
    decided_at: datetime  # Eastern wall clock, like every kickoff


@dataclass(frozen=True)
class SlotView:
    slot: str
    status: str  # locked, pick, hold, none
    player: Candidate | None = None  # the recommendation; for a hold, the early player
    alternatives: tuple[Candidate, ...] = ()
    wait_for: Candidate | None = None  # a hold's later player
    locked: str | None = None  # the recorded player's name
    result: str | None = None  # the recorded pick's score so far
    was: Previous | None = None  # an earlier run's different recommendation


def slot_views(
    advice: list[SlotAdvice],
    results: dict[str, tuple[str, str]] | None = None,
    previous: dict[str, Previous] | None = None,
) -> list[SlotView]:
    """One view per slot, in the order the advice gives them.

    `results` maps a recorded slot to its player's name and score; `previous` maps a slot
    to what the last earlier run recommended, which is shown only where it differs.
    """
    results, previous = results or {}, previous or {}
    views = []
    for a in advice:
        if a.locked_player is not None:
            name, result = results.get(a.slot, (a.locked_player, None))
            views.append(SlotView(a.slot, "locked", locked=name, result=result))
            continue
        before = previous.get(a.slot)
        if a.recommended is None:
            views.append(SlotView(a.slot, "none"))
            continue
        # An earlier run with nothing to recommend here, or with this slot locked, said
        # nothing this one could differ from.
        changed = before and before.player_id and before.player_id != a.recommended.player_id
        was = before if changed else None
        views.append(
            SlotView(
                a.slot,
                "hold" if a.hold else "pick",
                a.recommended,
                tuple(a.alternatives[:ALTERNATIVES]),
                a.hold_alternative if a.hold else None,
                was=was,
            )
        )
    return views


def decision_day(c: Candidate) -> date | str:
    """Each kickoff date before the main slate is a decision of its own; the slate is one.

    A Wednesday game and a Friday game are two chances to act, days apart, so one line
    recording both would log a pick that may not have been submitted yet. Sunday and Monday
    go in together, because that is how they are submitted. This groups what is printed and
    nothing else: every player keeps his own deadline, and a later valid pick is never
    refused because of the day it falls on.
    """
    return c.kickoff.date() if c.early else MAIN_SLATE


def due(views: list[SlotView]) -> tuple[list[SlotView], list[SlotView]]:
    """The PICK slots due on the next decision day, and the PICK slots that can wait."""
    picks = [v for v in views if v.status == "pick"]
    if not picks:
        return [], []
    day = decision_day(min(picks, key=lambda v: v.player.deadline).player)
    return (
        [v for v in picks if decision_day(v.player) == day],
        [v for v in picks if decision_day(v.player) != day],
    )


def program() -> str:
    """How this run was started, so the pasted line starts the same way.

    `uv run` marks the processes it starts with this variable; a globally installed `pool`
    has no such parent.
    """
    return "uv run pool" if os.environ.get("UV_RUN_RECURSION_DEPTH") else "pool"


def command(
    args: Sequence[str],
    *,
    season: int = config.DEFAULT_SEASON,
    db_path: str | Path | None = None,
) -> str:
    """A pasteable shell command retaining the selected launcher and data context."""
    tokens = [*shlex.split(program()), *args]
    if season != config.DEFAULT_SEASON:
        tokens += ["--season", str(season)]
    if db_path is not None:
        tokens += ["--db", str(db_path)]
    return shlex.join(tokens)


def record_command(
    season: int,
    week: int,
    slots: list[SlotView],
    pool: pd.DataFrame,
    decision_id: str | None,
    db_path: str | None = None,
) -> tuple[str | None, list[str]]:
    """The `record` line for `slots`, and a note for any slot it had to leave out.

    Each name is checked with `record`'s own lookup against the pool `record` will read,
    so a pasted line records the player it names or refuses; it can never quietly resolve
    to somebody else. The decision is named because the fallback link -- the latest advice
    for the slot -- is whichever run happened last, and with a Thursday and a Sunday run in
    one week that is not always the one the pick came from.
    """
    args, notes = [], []
    for view in slots:
        name = view.player.player_name
        matches = state.find_player(pool, name)
        if len(matches) != 1 or matches.iloc[0].player_id != view.player.player_id:
            notes.append(
                f"{view.slot}: `pool record` cannot pick out {name} by name alone, so this "
                "line leaves him out; `pool record` lists who it matched."
            )
            continue
        args += [f"--{view.slot.lower()}", name]
    if not args:
        return None, notes
    args = ["record", "--week", str(week), *args]
    if decision_id:
        args += ["--decision", decision_id]
    return command(args, season=season, db_path=db_path), notes


def locked_results(conn: sqlite3.Connection, season: int, week: int) -> dict[str, tuple[str, str]]:
    """Each recorded pick this week, by slot: the player's name and his score so far.

    Scored the way `pool picks` and standings score it, from the results as they stand, so
    a Thursday pick shows its touchdowns on Sunday.
    """
    picks = state.picks(conn, season)
    picks = picks[picks.week.eq(week)]
    if picks.empty:
        return {}
    board = results.score_board(conn, season)
    out = {}
    for pick in picks.itertuples():
        game = results.resolve_pick_game(conn, season, week, pick.player_id, pick.game_id)
        score = results.score_pick(board, week, pick.player_id, game)
        out[pick.slot] = (
            pick.player_name,
            "pending" if score.pending else f"{score.tds} TD, final",
        )
    return out


def previous_advice(
    conn: sqlite3.Connection, season: int, week: int, exclude: str | None = None
) -> dict[str, Previous]:
    """The last captured decision this week other than `exclude`, as its advice by slot."""
    row = conn.execute(
        "SELECT decision_id FROM decision_events WHERE season = ? AND week = ? "
        "AND kind = 'advice' AND decision_id IS NOT ? ORDER BY event_id DESC LIMIT 1",
        (season, week, exclude),
    ).fetchone()
    if row is None:
        return {}
    out = {}
    for event in conn.execute(
        "SELECT slot, player_id, detail, decision_at FROM decision_events "
        "WHERE decision_id = ? AND kind = 'advice'",
        (row["decision_id"],),
    ):
        recommended = json.loads(event["detail"]).get("recommended")
        out[event["slot"]] = Previous(
            event["player_id"],
            recommended["player_name"] if recommended else None,
            state.eastern_now(datetime.fromisoformat(event["decision_at"])),
        )
    return out


def save_predictions(
    conn: sqlite3.Connection, season: int, week: int, proj: pd.DataFrame, at: datetime
) -> tuple[str, datetime] | None:
    """Archive rival rankings and the PIT distribution independently, once per change.

    A prediction is scored only if it was archived before the week's first kickoff and
    before its report, so it is worth making on every run until then and worthless after.
    Saving again only when it changed keeps a run that learned nothing from adding a record
    that says nothing; the scorer reads the last one that beat both deadlines, so a changed
    prediction supersedes the earlier one. The PIT commitment compares its full projection
    surface and sampling settings separately, so either archive can retry a failed save.

    Returns what happened -- saved, unchanged, on record, or missed -- with the week's first
    kickoff, or None when there is nothing to say: not the current week, no rivals, no
    kickoff yet, or no identity, whose fix the withheld pot share already names.

    Needs a connection with no open transaction, like the archive it writes.
    """
    if week != state.current_week(conn, season, at) or not predictions.identified(conn, season):
        return None
    kickoff = predictions.first_kickoff(conn, season, week)
    if kickoff is None:
        return None
    if at >= kickoff or predictions.report_arrivals(conn, season)[0].get(week):
        found = predictions.scorable(conn, season, week)
        return ("on record" if found else "missed"), kickoff
    payload = predictions.predict(conn, season, week, proj)
    if not payload["rivals"]:
        return None
    earlier = [r for r in predictions.archived(conn, season) if r["week"] == week]
    same = bool(earlier) and json.dumps(earlier[-1]["payload"], sort_keys=True) == json.dumps(
        payload, sort_keys=True
    )
    if not same:
        predictions.archive(conn, season, week, payload)
    commitment = pit.commit(conn, season, week, proj, skip_unchanged=True)
    return ("saved" if not same or commitment is not None else "unchanged"), kickoff
