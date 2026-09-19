"""Rival-pick predictions, archived before they could have seen the answer.

Outside the enforced decision closure on purpose: this module reads the database, writes the
archive and scores the record, while `rivals` decides. Nothing in the closure imports it and
nothing ever should -- that is what keeps a report parser out of the decision fingerprint.

The deadline is the whole reason it exists. The phase plan gates a prediction on the
report's arrival, which is necessary and not sufficient: a prediction made after a week's
games but before its report lands has run on projections rebuilt from that week's own
statistics, and would be scored against an answer it had partly seen. So a prediction counts
only when it was archived **before the week's first kickoff** and **before the week's report
observation**. Both are timestamps already in `input_observations`, which is exactly why
predictions go there too: the comparison is then between two rows of one append-only table,
and neither can be edited out from under the other.

No migration. A prediction is an observation of what the model said at a time, which is what
`input_observations` already is. The feed is deliberately absent from `snapshots.TABLES` and
`freshness.FEEDS`, so it stays invisible to decision capture, freshness and replay -- for the
same reason `pool_report` is: a guess about an opponent is not a projection input.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import zlib
from dataclasses import asdict, replace
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from . import config, db, entrants, results, rivals, standings
from .state import picks as personal_picks

SCHEMA_VERSION = 1
FEED = "pool_prediction"


def _utc(value: str | datetime) -> str:
    instant = datetime.fromisoformat(value) if isinstance(value, str) else value
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError(f"Timestamp {value!r} must include a timezone")
    return instant.astimezone(UTC).isoformat(timespec="microseconds")


def _kickoff_utc(value: str) -> datetime:
    """Kickoffs are stored as Eastern wall-clock; the archive is UTC. Compare in UTC."""
    return datetime.fromisoformat(value).replace(tzinfo=ZoneInfo(config.TIMEZONE)).astimezone(UTC)


def first_kickoff(conn: sqlite3.Connection, season: int, week: int) -> datetime | None:
    """When the week stops being unobservable. None when no kickoff is confirmed yet."""
    row = conn.execute(
        "SELECT MIN(kickoff) FROM games WHERE season = ? AND week = ? "
        "AND game_type = 'REG' AND kickoff_known = 1",
        (season, week),
    ).fetchone()
    return _kickoff_utc(row[0]) if row and row[0] else None


def report_arrivals(conn: sqlite3.Connection, season: int) -> tuple[dict[int, str], int]:
    """Earliest archived report observation per week, and how many could not be attributed.

    Parse outcome is deliberately ignored. A report whose parser failed still delivered the
    answer -- the bytes were in hand -- so it closes the window just as a clean one does.

    A `--check` receipt is not a delivery. Checks archived until they stopped, and what
    they archived were drafts of a file being put together; `entrants.import_report` says
    as much. Counted here, a draft checked before the real import would close a window
    early, and one with no week would be reported as a delivery nobody can place.
    """
    reports = entrants.reports(conn, season)
    if not reports.empty:
        reports = reports[~reports.check.astype(bool)]
    if reports.empty:
        return {}, 0
    known = reports[reports.week.notna()]
    arrivals = known.groupby(known.week.astype(int)).observed_at.min().to_dict()
    return arrivals, int(len(reports) - len(known))


def identified(conn: sqlite3.Connection, season: int) -> bool:
    """False when there are entrants and none of them is marked as me.

    Every entrant without the flag is read as a rival, so without it I am one of my own
    opponents: predicted against, and simulated beside the entry that is actually mine.
    No entrants at all is not a missing identity, only an empty pool.
    """
    found = entrants.members(conn, season)
    return found.empty or bool(found.is_me.any())


def _standing(row, spent, played: set[int], known=None) -> rivals.RivalState:
    pool = spent[row.entrant_id]
    return rivals.RivalState(
        row.entrant_id,
        row.display_name,
        pool.player_ids,
        pool.unknown,
        row.season_total.tds,
        tuple(w for w in pool.missing_weeks if w in played),
        (known or {}).get(row.entrant_id, ()),
    )


def known_picks(
    conn: sqlite3.Connection, season: int, week: int, at: datetime
) -> dict[str, tuple[tuple[str, int, str | None], ...]]:
    """Each entrant's picks from `week` on that were already on record at `at`.

    Not the leak `_as_of` guards against, and worth being clear about the difference. That
    one is about *spending*: counting a week's report as depletion before that week is
    decided shrinks the pool a prediction of that same week is made against, and makes the
    prediction look better than it was. This is the opposite operation -- using a reported
    pick as itself, in the week it belongs to. It replaces a guess with the answer.

    `at` is what keeps it honest. A report that landed after the decision was made is not
    something the decision could have used, and including it would let a captured decision
    reconstruct against intelligence it never had.

    The latest observation of a cell wins, so a corrected report supersedes the one it
    corrects rather than contributing a second pick for the same slot.

    A reported no-pick is kept, as None: the slot is held empty rather than predicted. A
    name that never resolved is not -- who they took is unknown, so the week stays a
    prediction, and `unresolved_ahead` says so.
    """
    rows = _reported_ahead(conn, season, week, at)
    rows = rows[rows.player_id.notna() | rows.player_name.isna()]
    out: dict[str, list[tuple[str, int, str | None]]] = {}
    for row in rows.itertuples():
        pid = None if pd.isna(row.player_id) else str(row.player_id)
        out.setdefault(row.entrant_id, []).append((str(row.slot), int(row.week), pid))
    return {entrant: tuple(sorted(picks)) for entrant, picks in out.items()}


def unresolved_ahead(
    conn: sqlite3.Connection, season: int, week: int, at: datetime
) -> list[tuple[str, int, str, str]]:
    """Reported picks from `week` on whose names never resolved: (entrant, week, slot, name).

    The pot share simulates these as unknown choices, which is no worse than the week
    before its report arrives -- but it is a guess where the report gave an answer, and has
    to be visible as one.
    """
    rows = _reported_ahead(conn, season, week, at)
    rows = rows[rows.player_id.isna() & rows.player_name.notna()]
    return [
        (str(row.display_name), int(row.week), str(row.slot), str(row.player_name))
        for row in rows.itertuples()
    ]


def _reported_ahead(conn, season, week, at) -> pd.DataFrame:
    """Every reported cell from `week` on that was on record at `at`, latest observation."""
    rows = entrants.entrant_picks(conn, season)
    rows = rows[rows.week.ge(week)]
    # Compared as instants, not as text, for the reason `eligible` is: the two writers
    # agree on a format today and nothing enforces that they keep agreeing.
    arrived = [datetime.fromisoformat(value) <= at for value in rows.observed_at]
    rows = rows[pd.Series(arrived, index=rows.index, dtype=bool)]
    return rows.sort_values(["observed_at", "week", "entrant_id", "slot"]).drop_duplicates(
        ["entrant_id", "week", "slot"], keep="last"
    )


def played_weeks(conn: sqlite3.Connection, season: int, through: int, at: datetime) -> set[int]:
    """Weeks at or before `through` that have kicked off by `at`.

    A week nobody has played is not a week nobody reported. Picks for it may not have been
    made yet, so an absent report is the expected state rather than missing evidence.
    Without this, asking the tool to look one week ahead invents three spent players per
    rival out of a week that has not happened.

    Kickoff is the line for the same reason it is elsewhere in this module: it is when the
    week stops being unobservable. Before it, absence says nothing.
    """
    rows = conn.execute(
        "SELECT week, MIN(kickoff) AS first FROM games WHERE season = ? AND game_type = 'REG' "
        "AND kickoff_known = 1 AND week <= ? GROUP BY week",
        (season, through),
    ).fetchall()
    return {int(row["week"]) for row in rows if _kickoff_utc(row["first"]) <= at}


def _as_of(conn, season, week, at):
    """The board, every entrant's pool going into `week`, and which of those weeks was played.

    Spending is counted through `week - 1`, never through every imported week. A report can
    arrive before its own week is played, so counting it here would let a prediction for
    week N be built against a pool that already knows week N -- the same leak stage 2 closed
    in `standings.remaining_counts`, arriving here through a different door.
    """
    with db.transaction(conn):
        return (
            standings.board(conn, season),
            standings.used_pools(conn, season, through=week - 1),
            played_weeks(conn, season, week - 1, at),
            known_picks(conn, season, week, at),
        )


def rival_states(
    conn: sqlite3.Connection, season: int, week: int, *, at: datetime | None = None
) -> list[rivals.RivalState]:
    """Every entrant but me, with what they had spent going into `week`."""
    at = at or datetime.now(UTC)
    if not identified(conn, season):
        raise ValueError(entrants.IDENTITY_PROMPT)
    board, spent, played, _known = _as_of(conn, season, week, at)
    # Deliberately without `known`: this builds the state a *prediction* is made from, and
    # handing it the answer would make every hit rate meaningless. `pool_state` takes it,
    # because the policy is entitled to intelligence the prediction log is not.
    return [_standing(row, spent, played) for row in board.rows if not row.is_me]


def pool_state(
    conn: sqlite3.Connection, season: int, week: int, *, at: datetime | None = None
) -> rivals.PoolState:
    """Where the pool stands going into `week`, in the form the policy takes it.

    This is the seam `rivals.PoolState` describes. The database is read here, on this side
    of the enforced decision closure, and what crosses into it is a frozen dataclass of
    names, identifiers and integers -- never a connection and never a report parser.

    `at` is the caller's decision instant where it has one, so a command that has already
    fixed its own clock does not acquire a second one here.
    """
    at = at or datetime.now(UTC)
    with db.transaction(conn):
        if not identified(conn, season):
            return rivals.PoolState(withheld=(entrants.IDENTITY_PROMPT,))
        board, spent, played, known = _as_of(conn, season, week, at)
        banked, my_tds, finalized, gaps = _decision_outcomes(conn, season, week, at, board.rows)
    return rivals.PoolState(
        tuple(
            replace(_standing(row, spent, played, known), season_tds=banked.get(row.entrant_id, 0))
            for row in board.rows
            if not row.is_me
        ),
        my_tds,
        finalized,
        _withheld(gaps, played),
    )


def _decision_outcomes(conn, season, week, at, members):
    """Authoritative scores available now: past totals, fixed future cells, and the gaps.

    Scoring tables hold the latest import, not a history. A later import cannot be used
    for an earlier decision; captures carry these resolved values for subsequent replay.

    A past cell that is not final -- pending, an unresolved name, or nothing on record --
    is a gap, never a zero. The simulation starts at `week`, so nothing else would ever
    account for it, and a season total missing a score is indistinguishable from one that
    scored nothing. Gaps are (who, week, slot, status), with me as "you".
    """
    scores = results.score_board(conn, season)
    imports = {
        row["game_id"]: datetime.fromisoformat(row["imported_at"])
        for row in conn.execute(
            "SELECT game_id, imported_at FROM game_results WHERE season = ?", (season,)
        )
    }
    games = scores.games.copy()
    unseen = [gid for gid in games.index if gid not in imports or imports[gid] > at]
    games.loc[unseen, "complete"] = 0
    games.loc[unseen, "reason"] = "results not available at decision"
    scores = replace(scores, games=games)
    cells, fixed = {}, {}
    reported = entrants.entrant_picks(conn, season)
    if not reported.empty:
        reported = reported[[datetime.fromisoformat(t) <= at for t in reported.observed_at]]
        for pick in standings.score_rows(conn, season, reported, scores):
            if pick.week < week:
                cells[(pick.entrant_id, pick.week, pick.slot)] = (pick.status, pick.tds)
            elif not pick.is_me and pick.status == "final" and pick.player_id:
                fixed[(pick.week, pick.player_id)] = pick.tds
    me = next((row.entrant_id for row in members if row.is_me), None)
    for pick in personal_picks(conn, season).itertuples():
        if datetime.fromisoformat(pick.recorded_at) > at:
            continue
        gid = results.resolve_pick_game(conn, season, pick.week, pick.player_id, pick.game_id)
        result = results.score_pick(scores, pick.week, pick.player_id, gid)
        if pick.week < week:
            # What I recorded is what I submitted, so it stands in for the report's copy.
            cells[(me, pick.week, pick.slot)] = (
                "pending" if result.pending else "final",
                result.tds,
            )
        elif not result.pending:
            fixed[(pick.week, pick.player_id)] = result.tds
    banked, gaps = {}, []
    for row in members:
        who = "you" if row.is_me else row.display_name
        for past in range(1, week):
            for slot in config.SLOTS:
                status, tds = cells.get((row.entrant_id, past, slot), ("missing", None))
                if status == "final":
                    banked[row.entrant_id] = banked.get(row.entrant_id, 0) + int(tds)
                else:
                    gaps.append((who, past, slot, status))
    finalized = tuple((w, pid, int(tds)) for (w, pid), tds in sorted(fixed.items()))
    return banked, banked.pop(me, 0), finalized, gaps


def _names(names) -> str:
    names = list(dict.fromkeys(names))
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _withheld(gaps, played: set[int]) -> tuple[str, ...]:
    """Why the pot share cannot be computed, one sentence per week and kind of gap.

    Each ends with what fixes it, because the reader is deciding a pick and the useful
    question is what to do next, not which table is short.
    """
    reasons = []
    for past in sorted({gap[1] for gap in gaps}):
        here = [gap for gap in gaps if gap[1] == past]
        if past not in played:
            reasons.append(
                f"Week {past} has not been played yet, so nobody's season before this week is "
                "known. The pot share is only for the next week to be decided."
            )
            continue
        missing = [who for who, _, _, status in here if status == "missing"]
        if missing:
            reasons.append(
                f"Week {past}: nothing on record for {_names(missing)}. "
                "Import that week's report with `pool report import`."
            )
        pending = sum(status == "pending" for *_, status in here)
        if pending:
            reasons.append(
                f"Week {past}: {pending} pick{'s are' if pending > 1 else ' is'} not final "
                "yet. Run `pool refresh` once the games are over."
            )
        unresolved = [f"{who} {slot}" for who, _, slot, status in here if status == "unresolved"]
        if unresolved:
            reasons.append(
                f"Week {past}: the name for {_names(unresolved)} did not resolve to a player. "
                "Re-import that week's report once it does."
            )
    return tuple(reasons)


def predict(conn: sqlite3.Connection, season: int, week: int, proj: pd.DataFrame) -> dict:
    """Build the payload. Writes nothing, so it can be inspected before it is committed.

    `predicted_at` is deliberately absent: the observation carries the time, and repeating
    it here would make two identical predictions differ in content and defeat the payload
    de-duplication that `input_payloads` exists to provide.
    """
    states = rival_states(conn, season, week)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "season": season,
        "week": week,
        "top_n": rivals.TOP_N,
        "as_of_week": results.resolved_through(conn, season),
        "rivals": {},
    }
    for state in states:
        payload["rivals"][state.entrant_id] = {
            "display_name": state.display_name,
            "season_tds": state.season_tds,
            "used": sorted(state.used_ids),
            "unknown": state.unknown,
            "remaining": standings.remaining_counts(conn, season, week, state.entrant_id),
            "slots": {
                slot: {
                    name: [asdict(c) for c in ranked]
                    for name, ranked in rivals.predict_all(proj, slot, week, state).items()
                }
                for slot in config.SLOTS
            },
        }
    return payload


def archive(
    conn: sqlite3.Connection,
    season: int,
    week: int,
    payload: dict,
    *,
    observed_at: str | datetime | None = None,
    kind: str = "picks",
) -> int:
    """Commit the prediction. Mirrors `entrants.archive_report`, for the same reasons.

    `kind` separates the claims sharing this feed -- predicted rival picks, and the PIT
    commitment in `pit.py`. Both are statements made before the week could be seen, which
    is why they belong to one feed; neither scorer has any business reading the other's
    rows, which is why they are labelled.
    """
    if conn.in_transaction:
        raise ValueError("Archiving a prediction requires a connection with no open transaction")
    stamp = _utc(observed_at or datetime.now(UTC))
    raw = json.dumps(payload, sort_keys=True).encode()
    digest = hashlib.sha256(raw).hexdigest()
    with db.transaction(conn):
        conn.execute(
            "INSERT OR IGNORE INTO input_payloads VALUES (?, ?, 'zlib-bytes', ?)",
            (digest, SCHEMA_VERSION, zlib.compress(raw)),
        )
        cursor = conn.execute(
            "INSERT INTO input_observations "
            "(season, feed, observed_at, source_timestamp, content_hash, coverage, schema_version)"
            " VALUES (?, ?, ?, NULL, ?, ?, ?)",
            (
                season,
                FEED,
                stamp,
                digest,
                json.dumps(
                    dict(kind=kind, week=week, rivals=sorted(payload.get("rivals", {}))),
                    sort_keys=True,
                ),
                SCHEMA_VERSION,
            ),
        )
    return int(cursor.lastrowid)


def archived(conn: sqlite3.Connection, season: int, kind: str = "picks") -> list[dict]:
    """Every archived claim of one kind, in observation order, payload decoded.

    Rows written before the feed carried two kinds have no label and are rival-pick
    predictions, which is what the default reads them as.
    """
    rows = conn.execute(
        "SELECT o.observation_id, o.observed_at, o.coverage, p.payload "
        "FROM input_observations o JOIN input_payloads p ON p.content_hash = o.content_hash "
        "WHERE o.season = ? AND o.feed = ? ORDER BY o.observation_id",
        (season, FEED),
    ).fetchall()
    out = []
    for row in rows:
        coverage = json.loads(row["coverage"])
        if coverage.get("kind", "picks") != kind:
            continue
        out.append(
            dict(
                observation_id=int(row["observation_id"]),
                observed_at=row["observed_at"],
                week=coverage["week"],
                payload=json.loads(zlib.decompress(row["payload"]).decode()),
            )
        )
    return out


def eligible(records, kickoff, arrival):
    """The last prediction that beat both deadlines, and why the others did not.

    Compared as instants rather than as text. Both timestamps are written UTC with
    microseconds today, so string order happens to be time order, but that is a property of
    two independent writers agreeing on a format -- and the day one of them stops, a
    lexicographic comparison would silently start scoring predictions that saw the answer.
    """
    arrived = datetime.fromisoformat(arrival) if arrival else None
    reasons, survivors = [], []
    for record in records:
        observed = datetime.fromisoformat(record["observed_at"])
        if kickoff is None:
            reasons.append("no confirmed kickoff for the week")
        elif observed >= kickoff:
            reasons.append(f"archived {record['observed_at']} after first kickoff {kickoff}")
        elif arrived is not None and observed >= arrived:
            reasons.append(f"archived {record['observed_at']} after the report arrived {arrival}")
        else:
            survivors.append(record)
    return (survivors[-1] if survivors else None), reasons


def scorable(conn: sqlite3.Connection, season: int, week: int) -> dict | None:
    """The archived prediction `score` would read for `week`: the last one that beat both
    its first kickoff and its report. None when none did."""
    records = [r for r in archived(conn, season) if r["week"] == week]
    arrival = report_arrivals(conn, season)[0].get(week)
    return eligible(records, first_kickoff(conn, season, week), arrival)[0]


SCORED_COLUMNS = [
    "week",
    "entrant_id",
    "display_name",
    "slot",
    "predictor",
    "actual_player_id",
    "actual_player_name",
    "rank",
    "hit",
]


def score(conn: sqlite3.Connection, season: int) -> tuple[pd.DataFrame, list[dict]]:
    """Score every archived prediction that beat both deadlines against what happened.

    An unresolved name is unscorable rather than a miss. Scoring it as a miss would make an
    import defect look like a model defect, and those two rates are not the same number.
    """
    records = archived(conn, season)
    arrivals, unattributed = report_arrivals(conn, season)
    actual = entrants.entrant_picks(conn, season)
    rows, notes = [], []
    if unattributed:
        notes.append(dict(week=None, reason=f"{unattributed} archived report(s) name no week"))
    by_week: dict[int, list[dict]] = {}
    for record in records:
        by_week.setdefault(record["week"], []).append(record)
    for week in sorted(by_week):
        arrival = arrivals.get(week)
        chosen, reasons = eligible(by_week[week], first_kickoff(conn, season, week), arrival)
        if chosen is None:
            notes.append(dict(week=week, reason="; ".join(reasons) or "no prediction archived"))
            continue
        if arrival is None:
            notes.append(dict(week=week, reason="no report yet; prediction stands unscored"))
            continue
        truth = actual[actual.week.eq(week)]
        for entrant_id, block in chosen["payload"]["rivals"].items():
            for slot, predictors in block["slots"].items():
                found = truth[truth.entrant_id.eq(entrant_id) & truth.slot.eq(slot)]
                if not len(found):
                    notes.append(
                        dict(week=week, reason=f"{entrant_id} {slot}: no reported pick to score")
                    )
                    continue
                pick = found.iloc[0]
                pid = None if pd.isna(pick.player_id) else str(pick.player_id)
                if pid is None:
                    # Two different facts. A no-pick is an answer no predictor offered, and
                    # an unresolved name is an answer nobody can read yet.
                    what = (
                        "no pick reported; nothing to score"
                        if pd.isna(pick.player_name)
                        else "reported name never resolved; unscorable"
                    )
                    notes.append(dict(week=week, reason=f"{entrant_id} {slot}: {what}"))
                    continue
                for name, ranked in predictors.items():
                    rank = rivals.rank_of([rivals.Ranked(**c) for c in ranked], pid)
                    rows.append(
                        dict(
                            week=week,
                            entrant_id=entrant_id,
                            display_name=block["display_name"],
                            slot=slot,
                            predictor=name,
                            actual_player_id=pid,
                            actual_player_name=pick.player_name,
                            rank=rank,
                            hit=rank == 1,
                        )
                    )
    return pd.DataFrame(rows, columns=SCORED_COLUMNS), notes


def hit_rates(scored: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Hit rate and top-N recall per predictor, optionally split by another column."""
    if scored.empty:
        return pd.DataFrame(columns=["predictor", "n", "hits", "hit_rate", "in_top_n"])
    keys = ["predictor"] + ([by] if by else [])
    grouped = scored.groupby(keys, sort=True)
    out = grouped.agg(
        n=("hit", "size"), hits=("hit", "sum"), in_top_n=("rank", lambda s: s.notna().sum())
    ).reset_index()
    out["hit_rate"] = out.hits / out.n
    out["in_top_n"] = out.in_top_n / out.n
    return out
