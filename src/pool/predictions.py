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


def rival_states(conn: sqlite3.Connection, season: int, week: int) -> list[rivals.RivalState]:
    """Every entrant but me, with what they had spent *going into* `week`.

    Spending is counted through `week - 1`, never through every imported week. A report can
    arrive before its own week is played, so counting it here would let a prediction for
    week N be built against a pool that already knows week N -- the same leak stage 2 closed
    in `standings.remaining_counts`, arriving here through a different door.
    """
    with db.transaction(conn):
        board = standings.board(conn, season)
        spent = standings.used_pools(conn, season, through=week - 1)
    return [
        rivals.RivalState(
            row.entrant_id,
            row.display_name,
            spent[row.entrant_id].player_ids,
            spent[row.entrant_id].unknown,
            row.season_total.tds,
        )
        for row in board.rows
        if not row.is_me
    ]


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
) -> int:
    """Commit the prediction. Mirrors `entrants.archive_report`, for the same reasons."""
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
                    dict(week=week, rivals=sorted(payload["rivals"])), sort_keys=True
                ),
                SCHEMA_VERSION,
            ),
        )
    return int(cursor.lastrowid)


def archived(conn: sqlite3.Connection, season: int) -> list[dict]:
    """Every archived prediction in observation order, payload decoded."""
    rows = conn.execute(
        "SELECT o.observation_id, o.observed_at, o.coverage, p.payload "
        "FROM input_observations o JOIN input_payloads p ON p.content_hash = o.content_hash "
        "WHERE o.season = ? AND o.feed = ? ORDER BY o.observation_id",
        (season, FEED),
    ).fetchall()
    return [
        dict(
            observation_id=int(row["observation_id"]),
            observed_at=row["observed_at"],
            week=json.loads(row["coverage"])["week"],
            payload=json.loads(zlib.decompress(row["payload"]).decode()),
        )
        for row in rows
    ]


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
