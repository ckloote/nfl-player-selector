"""Append-only record of what was known at a decision and what was decided.

A projection frame is rebuilt from scratch every time it is asked for, so by the
following week the inputs behind a pick no longer exist anywhere. Feed snapshots
recover the inputs but not the decision: not the surface the optimizer actually saw
before pruning, not which players were already spent, not whether the advice was to
hold or to commit, and not which of it was submitted. Phase 3B needs forecast and
outcome pairs it can attribute to a decision, and Phase 3C needs to replay real
events; both need that evidence written down when the decision is made.

Events are append-only. A correction is a new event, because a record that can be
edited afterwards cannot evidence what was known at the time. Outcomes are never
stored here -- they are joined from finalized scoring at read time, the same way a
pick's touchdowns are -- so a later stat correction cannot rewrite a decision.
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import zlib
from datetime import UTC, datetime
from functools import cache
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from . import config, db, snapshots, state
from .recommend import SlotAdvice, advise_week

SCHEMA_VERSION = 1
CODEC = "zlib-parquet"
# The whole surface, not the rows that survived pruning or made it onto the screen.
SURFACE_EXTRAS = [
    "decision_week",
    "lead_horizon",
    "used",
    "locked",
    "decision_status",
    "original_lam",
    "mapped_lam",
]


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


@cache
def _identity_payload(model: str, calibrator: str, artifact_hash: str | None) -> tuple[str, str]:
    """Hashing the source tree shells out to git; a command that records three picks
    should not do it three times. Source cannot change under a running process in any
    way this tool would survive anyway."""
    from . import benchmark  # imports the feed layer; not needed to read a capture

    code = benchmark.code_identity()
    raw = _json(
        {
            "model": model,
            "calibrator": calibrator,
            "calibrator_artifact_hash": artifact_hash,
            "code_hash": code["code_hash"],
            "revision": code.get("revision"),
            "dirty": code.get("dirty"),
            "constants": benchmark.constants(),
        }
    )
    return hashlib.sha256(raw.encode()).hexdigest(), raw


def identity(conn: sqlite3.Connection, model: str, calibrator: str, artifact_hash=None) -> str:
    """Record which code, constants, model and calibrator produced a decision.

    Without this a captured forecast cannot be attributed: two runs of "the shipped
    model" a month apart are different functions if a constant moved between them.
    """
    digest, raw = _identity_payload(model, calibrator, artifact_hash)
    with db.transaction(conn):
        conn.execute(
            "INSERT OR IGNORE INTO decision_identities VALUES (?, ?, ?)",
            (digest, SCHEMA_VERSION, raw),
        )
    return digest


def observed_inputs(conn: sqlite3.Connection, season: int, decision_at: str) -> list[dict]:
    """The latest observation of each feed at or before the decision.

    Snapshot replay resolves inputs exactly this way, so a live capture and a replay
    of the same instant name the same observations.
    """
    out = []
    for year in (season - 1, season):
        for feed in snapshots.TABLES:
            row = conn.execute(
                "SELECT observation_id, observed_at, content_hash FROM input_observations "
                "WHERE season = ? AND feed = ? AND observed_at <= ? "
                "ORDER BY observed_at DESC, observation_id DESC LIMIT 1",
                (year, feed, decision_at),
            ).fetchone()
            if row is None:
                out.append(
                    dict(
                        season=year,
                        feed=feed,
                        observation_id=None,
                        content_hash=None,
                        observed_at=None,
                        missing=1,
                        age_hours=None,
                        stale=None,
                    )
                )
                continue
            age = (
                datetime.fromisoformat(decision_at) - datetime.fromisoformat(row["observed_at"])
            ).total_seconds() / 3600
            out.append(
                dict(
                    season=year,
                    feed=feed,
                    observation_id=int(row["observation_id"]),
                    content_hash=row["content_hash"],
                    observed_at=row["observed_at"],
                    missing=0,
                    age_hours=age,
                    stale=int(age > config.FRESHNESS_HOURS[feed]),
                )
            )
    return out


def surface(
    proj: pd.DataFrame,
    week: int,
    used_ids: set[str],
    locked_by_slot: dict[str, dict[int, str]],
    now: datetime,
    mapped: pd.Series | None = None,
) -> pd.DataFrame:
    """The full pre-pruning surface, with the state that made it a decision.

    `optimizer.build_matrix` drops used and hard-ineligible players and keeps only the
    top `CANDIDATES_PER_SLOT`, and the recommender then shows a handful. All of that is
    downstream of this frame; storing anything narrower would record the answer rather
    than the choice, and would make a stratum such as "eligible zero rates" impossible
    to diagnose after the fact.
    """
    out = proj.copy()
    locked_cells = {(w, pid) for slot in locked_by_slot for w, pid in locked_by_slot[slot].items()}
    out["decision_week"] = week
    out["lead_horizon"] = out.week.astype(int) - week
    out["used"] = out.player_id.isin(used_ids)
    out["locked"] = [
        (int(w), pid) in locked_cells for w, pid in zip(out.week, out.player_id, strict=True)
    ]
    # Live loading leaves the deadline out of `hard_eligible` and applies it in the
    # recommender, so the frame alone does not say why a cell was unusable. Resolve it
    # here: a diagnostic should not have to re-derive an elapsed kickoff from a clock.
    out["decision_status"] = [state.availability(row, week, now) for _, row in out.iterrows()]
    out["original_lam"] = out.lam.astype(float)
    out["mapped_lam"] = out.original_lam if mapped is None else pd.Series(mapped).astype(float)
    return out.reset_index(drop=True)


def _store_surface(conn: sqlite3.Connection, frame: pd.DataFrame) -> str:
    """Content-address the surface beside the archived feeds.

    Parquet rather than the feeds' JSON: a surface is a typed numeric frame, and a
    reconstruction that has to guess dtypes back is not a reconstruction.
    """
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    raw = buffer.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    conn.execute(
        "INSERT OR IGNORE INTO input_payloads VALUES (?, ?, ?, ?)",
        (digest, SCHEMA_VERSION, CODEC, zlib.compress(raw)),
    )
    return digest


def load_surface(conn: sqlite3.Connection, content_hash: str) -> pd.DataFrame:
    row = conn.execute(
        "SELECT payload, codec FROM input_payloads WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    if row is None:
        raise ValueError(f"No stored surface for {content_hash}")
    if row["codec"] != CODEC:
        raise ValueError(f"Unsupported surface codec {row['codec']!r}")
    raw = zlib.decompress(row["payload"])
    if hashlib.sha256(raw).hexdigest() != content_hash:
        raise ValueError("Stored surface hash mismatch")
    return pd.read_parquet(io.BytesIO(raw))


def _advice_detail(a: SlotAdvice) -> dict:
    def candidate(c):
        if c is None:
            return None
        return dict(
            player_id=c.player_id,
            player_name=c.player_name,
            team=c.team,
            position=c.position,
            lam=float(c.lam),
            cost=float(c.cost),
            early=bool(c.early),
            deadline=c.deadline.isoformat(),
            planned_week=c.planned_week,
        )

    return dict(
        locked_player=a.locked_player,
        recommended=candidate(a.recommended),
        alternatives=[candidate(c) for c in a.alternatives],
        hold=bool(a.hold),
        hold_alternative=candidate(a.hold_alternative),
        plan_total=float(a.plan.total),
        plan={int(w): str(a.plan.players.iloc[r].player_id) for w, r in a.plan.assignment.items()},
    )


def _append(conn, decision_id, recorded_at, decision_at, season, week, identity_hash, rows):
    conn.executemany(
        "INSERT INTO decision_events (decision_id, recorded_at, decision_at, season, week,"
        " slot, kind, player_id, detail, surface_hash, identity_hash, schema_version)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                decision_id,
                recorded_at,
                decision_at,
                season,
                week,
                slot,
                kind,
                player_id,
                _json(detail),
                surface_hash,
                identity_hash,
                SCHEMA_VERSION,
            )
            for slot, kind, player_id, detail, surface_hash in rows
        ],
    )


def record_decision(
    conn: sqlite3.Connection,
    season: int,
    week: int,
    proj: pd.DataFrame,
    advice: list[SlotAdvice],
    used_ids: set[str],
    locked_by_slot: dict[str, dict[int, str]],
    *,
    decision_at: str | datetime,
    model: str = "shipped",
    calibrator: str = "identity",
    artifact_hash: str | None = None,
) -> str:
    """Write one decision: its surface, its advice, and what it was looking at."""
    stamp = snapshots.timestamp(decision_at)
    stamp_naive = datetime.fromisoformat(stamp)
    decision_id = uuid4().hex
    identity_hash = identity(conn, model, calibrator, artifact_hash)
    frame = surface(proj, week, used_ids, locked_by_slot, state.eastern_now(stamp_naive))
    with db.transaction(conn):
        surface_hash = _store_surface(conn, frame)
        rows = [
            (
                None,
                "surface",
                None,
                dict(
                    rows=int(len(frame)),
                    weeks=sorted({int(w) for w in frame.week}),
                    slots=sorted({str(s) for s in frame.slot}),
                    used=sorted(used_ids),
                    locked={
                        slot: {int(w): pid for w, pid in locked_by_slot.get(slot, {}).items()}
                        for slot in config.SLOTS
                    },
                ),
                surface_hash,
            )
        ]
        for a in advice:
            rows.append((a.slot, "advice", _recommended_id(a), _advice_detail(a), surface_hash))
            if a.recommended is not None and a.recommended.early:
                # Hold or commit is the decision the early deadline forces, so it is
                # its own event rather than a field on the advice.
                rows.append(
                    (
                        a.slot,
                        "hold" if a.hold else "commit",
                        _recommended_id(a),
                        dict(
                            premium=config.INFO_PREMIUM_TD,
                            alternative=None
                            if a.hold_alternative is None
                            else a.hold_alternative.player_id,
                            alternative_cost=None
                            if a.hold_alternative is None
                            else float(a.hold_alternative.cost),
                        ),
                        surface_hash,
                    )
                )
        _append(
            conn,
            decision_id,
            snapshots.timestamp(datetime.now(UTC)),
            stamp,
            season,
            week,
            identity_hash,
            rows,
        )
        conn.executemany(
            "INSERT INTO decision_inputs (decision_id, season, feed, observation_id,"
            " content_hash, observed_at, missing, age_hours, stale) VALUES (?,?,?,?,?,?,?,?,?)",
            [
                (
                    decision_id,
                    r["season"],
                    r["feed"],
                    r["observation_id"],
                    r["content_hash"],
                    r["observed_at"],
                    r["missing"],
                    r["age_hours"],
                    r["stale"],
                )
                for r in observed_inputs(conn, season, stamp)
            ],
        )
    return decision_id


def _recommended_id(a: SlotAdvice) -> str | None:
    return None if a.recommended is None else a.recommended.player_id


def record_action(
    conn: sqlite3.Connection,
    season: int,
    week: int,
    slot: str,
    kind: str,
    player_id: str | None,
    detail: dict,
    *,
    decision_at: str | datetime | None = None,
    model: str = "shipped",
    calibrator: str = "identity",
) -> str:
    """Append a submitted pick or a later correction, linked to its decision.

    Submissions happen outside this tool -- the pool's own site -- so the link is the
    most recent decision for the same season, week and slot, and is null when a pick
    was recorded without one.
    """
    stamp = snapshots.timestamp(decision_at or datetime.now(UTC))
    row = conn.execute(
        "SELECT decision_id FROM decision_events WHERE season = ? AND week = ? AND kind = 'advice'"
        " AND (slot = ? OR ? IS NULL) ORDER BY event_id DESC LIMIT 1",
        (season, week, slot, slot),
    ).fetchone()
    linked = row["decision_id"] if row else None
    identity_hash = identity(conn, model, calibrator)
    decision_id = linked or uuid4().hex
    with db.transaction(conn):
        _append(
            conn,
            decision_id,
            snapshots.timestamp(datetime.now(UTC)),
            stamp,
            season,
            week,
            identity_hash,
            [(slot, kind, player_id, dict(detail, linked_decision=linked), None)],
        )
    return decision_id


def events(conn: sqlite3.Connection, season: int, week: int | None = None) -> pd.DataFrame:
    sql = "SELECT * FROM decision_events WHERE season = ?"
    params: tuple = (season,)
    if week is not None:
        sql += " AND week = ?"
        params += (week,)
    return db.read_df(conn, sql + " ORDER BY event_id", params)


def reconstruct(conn: sqlite3.Connection, decision_id: str):
    """Rebuild a decision's surface and re-derive its advice from that surface alone.

    Nothing here reads a current table, so a feed corrected after the decision cannot
    change what comes back. This is the check that the capture is sufficient: if the
    re-derived advice differs from what was recorded, something the decision depended
    on was never written down.
    """
    rows = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM decision_events WHERE decision_id = ? ORDER BY event_id", (decision_id,)
        )
    ]
    if not rows:
        raise ValueError(f"No captured decision {decision_id}")
    head = next(r for r in rows if r["kind"] == "surface")
    frame = load_surface(conn, head["surface_hash"])
    state = json.loads(head["detail"])
    advice = advise_week(
        frame,
        int(head["week"]),
        set(state["used"]),
        {
            slot: {int(w): pid for w, pid in cells.items()}
            for slot, cells in state["locked"].items()
        },
        now=datetime.fromisoformat(head["decision_at"]),
    )
    recorded = {r["slot"]: json.loads(r["detail"]) for r in rows if r["kind"] == "advice"}
    return frame, advice, recorded


def outcomes(conn: sqlite3.Connection, season: int) -> pd.DataFrame:
    """Captured forecasts joined to finalized scoring -- separately, and at read time.

    Outcomes are not part of the decision. Storing them beside it would mean a later
    correction rewrote history, and the decision is precisely the thing that cannot
    change after the fact.
    """
    from . import scoring

    frames = []
    for decision_id, content_hash in conn.execute(
        "SELECT decision_id, surface_hash FROM decision_events "
        "WHERE season = ? AND kind = 'surface' ORDER BY event_id",
        (season,),
    ):
        frame = load_surface(conn, content_hash)
        frames.append(frame.assign(decision_id=decision_id))
    if not frames:
        return pd.DataFrame()
    surfaces = pd.concat(frames, ignore_index=True)
    totals = scoring.touchdown_totals(conn, season)[["week", "player_id", "pool_td"]]
    complete = set(scoring.complete_weeks(conn, season))
    out = surfaces.merge(totals, on=["week", "player_id"], how="left")
    # Absence is zero only where every game in the week is scored. Anywhere else it is
    # unknown, and a diagnostic that reads it as zero is scoring the feed, not a model.
    out["outcome_complete"] = out.week.isin(complete)
    out["actual_tds"] = np.where(out.outcome_complete, out.pool_td.fillna(0.0), np.nan)
    return out.drop(columns=["pool_td"])
