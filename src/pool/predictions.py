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
from dataclasses import asdict
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from . import config, db, entrants, rivals, scoring, standings

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
    """
    reports = entrants.reports(conn, season)
    if reports.empty:
        return {}, 0
    known = reports[reports.week.notna()]
    arrivals = known.groupby(known.week.astype(int)).observed_at.min().to_dict()
    return arrivals, int(len(reports) - len(known))


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
) -> dict[str, tuple[tuple[str, int, str], ...]]:
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
    """
    rows = entrants.entrant_picks(conn, season)
    if rows.empty:
        return {}
    rows = rows[rows.week.ge(week) & rows.player_id.notna()]
    if rows.empty:
        return {}
    # Compared as instants, not as text, for the reason `_eligible` is: the two writers
    # agree on a format today and nothing enforces that they keep agreeing.
    arrived = [datetime.fromisoformat(value) <= at for value in rows.observed_at]
    rows = rows[pd.Series(arrived, index=rows.index)]
    if rows.empty:
        return {}
    rows = rows.sort_values("observed_at").drop_duplicates(
        ["entrant_id", "week", "slot"], keep="last"
    )
    out: dict[str, list[tuple[str, int, str]]] = {}
    for row in rows.itertuples():
        out.setdefault(row.entrant_id, []).append(
            (str(row.slot), int(row.week), str(row.player_id))
        )
    return {entrant: tuple(sorted(picks)) for entrant, picks in out.items()}


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
    board, spent, played, known = _as_of(conn, season, week, at)
    mine = next((row for row in board.rows if row.is_me), None)
    return rivals.PoolState(
        tuple(_standing(row, spent, played, known) for row in board.rows if not row.is_me),
        mine.season_total.tds if mine else 0,
    )


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
        "as_of_week": scoring.resolved_through(conn, season),
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


def _eligible(records, kickoff, arrival):
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


SCORED_COLUMNS = [
    "week", "entrant_id", "display_name", "slot", "predictor",
    "actual_player_id", "actual_player_name", "rank", "hit",
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
        chosen, reasons = _eligible(by_week[week], first_kickoff(conn, season, week), arrival)
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
                    notes.append(
                        dict(
                            week=week,
                            reason=f"{entrant_id} {slot}: reported name never resolved; unscorable",
                        )
                    )
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
