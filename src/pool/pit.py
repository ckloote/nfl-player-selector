"""Prospective probability-integral transform: what the joint model said a week would look
like, committed before the week could be seen.

The simulator was checked retrospectively in `experiments/`, over seasons it was fitted on
and over every player rather than the handful five people actually pick. This is the
forward check, on the only population that decides the pool. Each week, before kickoff, the
model implies a distribution for every entrant's weekly total; after the week, where the
realised total fell in that distribution is one draw. Five entrants over seventeen weeks is
eighty-five draws, and a flat rank histogram is the pass. It tests `lam`, the game factor
and the shared-credit mechanism together, which is how they are used and not how any of
them was fitted.

What is archived is the **outcomes**: every player-week the frame could be scored on, drawn
once at commitment and stored content-addressed, beside the seed, parameters and frame hash
that produced them. A seed alone fixes a distribution only under the same sampler and the
same numpy, and rebuilding from it at scoring time let a later improvement to either
quietly restate a claim made before kickoff. Stored, the claim survives both changing. It
costs about 130 KB a week. Which cells get summed is decided afterwards by the reports --
that is the part the model is not allowed to know -- and it cannot reach back into
outcomes already pinned by a hash.

Commitments made before outcomes were stored are frozen the first time they are read:
drawn once from their seed and stored the same way, then never drawn again.

Same feed as the rival-pick predictions, because it is the same kind of claim: something
the model said before it could see the answer. The two are told apart by `kind` in the
observation's coverage, so neither scorer reads the other's rows.
"""

from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import zlib
from dataclasses import asdict
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from . import capture, config, db, identity, predictions, scoring, simulate, standings

# 2 stores the outcomes; 1 stored only the seed and frame they were drawn from.
SCHEMA_VERSION = 2
FEED = predictions.FEED
KIND = "pit"
DRAWS_CODEC = "zlib-npz"
# A frozen schema-1 commitment's outcomes, keyed by its observation. `meta`, like a report's
# parse receipt, and never a new observation: `_eligible` reads observation times, and a
# freeze written after kickoff is a copy of an earlier claim, not a late one.
FROZEN_PREFIX = "pit_draws:"

SCORED_COLUMNS = [
    "week",
    "entrant_id",
    "display_name",
    "is_me",
    "actual",
    "below",
    "mass",
    "pit",
    "mean",
    "sims",
]


def commit(
    conn: sqlite3.Connection,
    season: int,
    week: int,
    proj: pd.DataFrame,
    *,
    sims: int | None = None,
    seed: int | None = None,
    observed_at: str | datetime | None = None,
) -> int:
    """Pin this week's implied distribution. Nothing about the week is read here."""
    if conn.in_transaction:
        raise ValueError("Committing a PIT requires a connection with no open transaction")
    sims = config.WINPROB_SIMS if sims is None else sims
    seed = config.WINPROB_SEED if seed is None else seed
    frame = proj[proj.week.eq(week)].reset_index(drop=True)
    if not len(frame):
        raise ValueError(f"No projection rows for {season} week {week}")
    params = simulate.Params()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "season": season,
        "week": week,
        "sims": int(sims),
        "seed": int(seed),
        "params": asdict(params),
        "surface_hash": None,
        "draws_hash": None,
    }
    draws = simulate.sample(frame, [week], sims=int(sims), seed=int(seed), params=params)
    with db.transaction(conn):
        payload["surface_hash"] = capture._store_surface(conn, frame)
        payload["draws_hash"] = _store_draws(conn, draws)
    return predictions.archive(conn, season, week, payload, observed_at=observed_at, kind=KIND)


def archived(conn: sqlite3.Connection, season: int) -> list[dict]:
    return predictions.archived(conn, season, kind=KIND)


def _store_draws(conn: sqlite3.Connection, draws: simulate.Draws) -> str:
    """Content-address sampled outcomes, in the smallest type that holds them. Needs a
    transaction held by the caller."""
    keys = sorted(draws.index, key=draws.index.get)
    values = draws.values
    kind = np.min_scalar_type(int(values.max()) if values.size else 0)
    buffer = io.BytesIO()
    np.savez(
        buffer,
        values=values.astype(kind),
        week=np.array([week for week, _ in keys], dtype=np.int16),
        player_id=np.array([pid for _, pid in keys], dtype=str),
    )
    raw = buffer.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    conn.execute(
        "INSERT OR IGNORE INTO input_payloads VALUES (?, ?, ?, ?)",
        (digest, SCHEMA_VERSION, DRAWS_CODEC, zlib.compress(raw)),
    )
    return digest


def _load_draws(conn: sqlite3.Connection, digest: str) -> simulate.Draws:
    row = conn.execute(
        "SELECT payload, codec FROM input_payloads WHERE content_hash = ?", (digest,)
    ).fetchone()
    if row is None:
        raise ValueError(f"No stored PIT outcomes {digest}")
    if row["codec"] != DRAWS_CODEC:
        raise ValueError(f"Unsupported PIT outcome codec {row['codec']!r}")
    raw = zlib.decompress(row["payload"])
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Stored PIT outcomes hash mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        values = data["values"].astype(np.int32)
        index = {
            (int(week), str(pid)): position
            for position, (week, pid) in enumerate(
                zip(data["week"], data["player_id"], strict=True)
            )
        }
    return simulate.Draws(values, index, int(values.shape[1]))


def _draws(conn: sqlite3.Connection, record: dict) -> simulate.Draws:
    """The committed outcomes, read back and never re-drawn once stored."""
    payload = record["payload"]
    if payload.get("draws_hash"):
        return _load_draws(conn, payload["draws_hash"])
    receipt = frozen(conn, record) or _freeze(conn, record)
    return _load_draws(conn, receipt["draws_hash"])


def frozen(conn: sqlite3.Connection, record: dict) -> dict | None:
    """When and how a schema-1 commitment's outcomes were drawn, once they have been."""
    value = db.get_meta(conn, f"{FROZEN_PREFIX}{record['observation_id']}")
    return json.loads(value) if value else None


def _freeze(conn: sqlite3.Connection, record: dict) -> dict:
    """Draw a schema-1 commitment's outcomes once, store them, and never draw them again.

    Such a commitment kept only its seed and frame. Drawing it afresh at every score let a
    later sampler or numpy rewrite it; drawing it once pins it at what the current sampler
    gives -- which is what was committed only while `simulate.sample` is unchanged since
    the commitment. `tests/test_pit.py` pins the sampler's output so that a change to it
    fails the suite before it can reach an unfrozen commitment.
    """
    payload = record["payload"]
    draws = simulate.sample(
        capture.load_surface(conn, payload["surface_hash"]),
        [int(payload["week"])],
        sims=int(payload["sims"]),
        seed=int(payload["seed"]),
        params=simulate.Params(**payload["params"]),
    )
    key = f"{FROZEN_PREFIX}{record['observation_id']}"
    with db.transaction(conn):
        receipt = dict(
            draws_hash=_store_draws(conn, draws),
            frozen_at=datetime.now(UTC).isoformat(timespec="seconds"),
            numpy=np.__version__,
            revision=identity.runtime()["revision"],
        )
        # First writer wins. Two freezes of one commitment draw the same outcomes anyway,
        # and a receipt is never overwritten once something may have been scored from it.
        conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
            (key, json.dumps(receipt, sort_keys=True)),
        )
    return frozen(conn, record)


def _jitter_seed(season: int, week: int, entrant_id: str) -> int:
    """A stable seed for the PIT's randomisation.

    `hash()` on a string is salted per process, so using it here would give a different
    histogram on every run and quietly break the reproducibility this module claims.
    """
    key = f"{season}:{week}:{entrant_id}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def _realised(picks) -> tuple[int, list[str]] | None:
    """One entrant's finished week: the total, and the cells behind it.

    None when any slot is unscored. A week with a missing or pending pick is not a smaller
    week, it is an unobserved one, and folding it in as a partial total would put mass at
    the bottom of the histogram that the model never claimed.
    """
    if {p.slot for p in picks} != set(config.SLOTS) or any(p.status != "final" for p in picks):
        return None
    cells = [p.player_id for p in picks if p.player_id]
    return sum(p.tds or 0 for p in picks), cells


def score(conn: sqlite3.Connection, season: int) -> tuple[pd.DataFrame, list[dict]]:
    """Where each entrant's realised weekly total fell in the distribution committed for it.

    Randomised within the observed value, which is what makes a PIT over counts uniform
    under a correct model: a discrete CDF evaluated at the realisation alone is not, and a
    histogram built from it slopes even when nothing is wrong. The jitter is drawn from a
    seed fixed by season, week and entrant, so scoring the same week twice gives the same
    answer.
    """
    rows, notes = [], []
    complete = set(scoring.complete_weeks(conn, season))
    arrivals, unattributed = predictions.report_arrivals(conn, season)
    if unattributed:
        notes.append(dict(week=None, reason=f"{unattributed} archived report(s) name no week"))
    by_week: dict[int, list[dict]] = {}
    for record in archived(conn, season):
        # Frozen as soon as anything reads them, not once their week becomes scorable: the
        # sampler can change in between, and a freeze is faithful only before it does.
        if not record["payload"].get("draws_hash") and not frozen(conn, record):
            _freeze(conn, record)
        by_week.setdefault(int(record["week"]), []).append(record)
    for week, records in sorted(by_week.items()):
        arrival = arrivals.get(week)
        chosen, reasons = predictions._eligible(
            records, predictions.first_kickoff(conn, season, week), arrival
        )
        notes.extend(dict(week=week, reason=reason) for reason in reasons)
        if chosen is None:
            continue
        superseded = len(records) - len(reasons) - 1
        if superseded:
            notes.append(
                dict(
                    week=week,
                    reason=f"{superseded} earlier eligible commitment(s) "
                    "superseded by the last eligible archive",
                )
            )
        if arrival is None:
            notes.append(dict(week=week, reason="no report yet; commitment stands unscored"))
            continue
        payload = chosen["payload"]
        if week not in complete:
            notes.append(dict(week=week, reason="week is not fully scored yet"))
            continue
        scored = standings.entrant_scores(conn, season, weeks=[week])
        by_entrant: dict[str, list] = {}
        for pick in scored:
            by_entrant.setdefault(pick.entrant_id, []).append(pick)
        draws = _draws(conn, chosen)
        receipt = None if payload.get("draws_hash") else frozen(conn, chosen)
        if receipt:
            notes.append(
                dict(
                    week=week,
                    reason=f"committed before outcomes were stored; drawn once from its seed "
                    f"and frozen at {receipt['frozen_at']}",
                )
            )
        for entrant_id, picks in sorted(by_entrant.items()):
            outcome = _realised(picks)
            if outcome is None:
                notes.append(dict(week=week, entrant_id=entrant_id, reason="week not fully scored"))
                continue
            actual, cells = outcome
            sampled = draws.totals([(week, pid) for pid in cells])
            below = float((sampled < actual).mean())
            # Named `mass`, not `at`: a column called `at` is reachable only as
            # `frame["at"]`, because `frame.at` is pandas' scalar indexer.
            mass = float((sampled == actual).mean())
            jitter = np.random.default_rng(_jitter_seed(season, week, entrant_id)).random()
            rows.append(
                dict(
                    week=week,
                    entrant_id=entrant_id,
                    display_name=picks[0].display_name,
                    is_me=picks[0].is_me,
                    actual=int(actual),
                    below=below,
                    mass=mass,
                    pit=below + jitter * mass,
                    mean=float(sampled.mean()),
                    sims=int(payload["sims"]),
                )
            )
    return pd.DataFrame(rows, columns=SCORED_COLUMNS), notes


def uniformity(scored: pd.DataFrame, bins: int = 5) -> pd.DataFrame:
    """The rank histogram, with the count a flat model would give for comparison.

    No pass or fail attached. Eighty-five draws is not enough to test uniformity with any
    power, and printing a verdict off it would dress a coin flip as a conclusion; the
    shape is for reading, and the direction of a lean is the useful part -- mass at the top
    means the model is under-predicting these picks, mass at the bottom over-predicting.
    """
    if scored.empty:
        return pd.DataFrame(columns=["bin", "low", "high", "count", "expected"])
    edges = np.linspace(0, 1, bins + 1)
    counts, _ = np.histogram(scored.pit.to_numpy(dtype=float), bins=edges)
    return pd.DataFrame(
        dict(
            bin=range(1, bins + 1),
            low=edges[:-1],
            high=edges[1:],
            count=counts.astype(int),
            expected=len(scored) / bins,
        )
    )
