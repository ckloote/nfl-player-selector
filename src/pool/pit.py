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

What is archived is the **commitment**, not the numbers: the seed, the simulation size, the
parameters, and the content hash of the frame. Those fix the distribution exactly, because
`simulate.sample` is a deterministic function of them, and they fix it in a fraction of the
space the draws would take. Which cells get summed is decided afterwards by the reports --
that is the part the model is not allowed to know -- and it cannot reach back into a
distribution already pinned by a hash.

Same feed as the rival-pick predictions, because it is the same kind of claim: something
the model said before it could see the answer. The two are told apart by `kind` in the
observation's coverage, so neither scorer reads the other's rows.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import asdict
from datetime import datetime

import numpy as np
import pandas as pd

from . import capture, config, db, predictions, scoring, simulate, standings

SCHEMA_VERSION = 1
FEED = predictions.FEED
KIND = "pit"

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
    payload = {
        "schema_version": SCHEMA_VERSION,
        "season": season,
        "week": week,
        "sims": int(sims),
        "seed": int(seed),
        "params": asdict(simulate.Params()),
        "surface_hash": None,
    }
    with db.transaction(conn):
        payload["surface_hash"] = capture._store_surface(conn, frame)
    return predictions.archive(
        conn, season, week, payload, observed_at=observed_at, kind=KIND
    )


def archived(conn: sqlite3.Connection, season: int) -> list[dict]:
    return predictions.archived(conn, season, kind=KIND)


def _draws(conn: sqlite3.Connection, payload: dict) -> simulate.Draws:
    """Rebuild the committed distribution. Deterministic, or the commitment means nothing."""
    frame = capture.load_surface(conn, payload["surface_hash"])
    params = simulate.Params(**payload["params"])
    return simulate.sample(
        frame,
        [int(payload["week"])],
        sims=int(payload["sims"]),
        seed=int(payload["seed"]),
        params=params,
    )


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
            notes.append(dict(week=week, reason=f"{superseded} earlier eligible commitment(s) "
                              "superseded by the last eligible archive"))
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
        draws = _draws(conn, payload)
        for entrant_id, picks in sorted(by_entrant.items()):
            outcome = _realised(picks)
            if outcome is None:
                notes.append(
                    dict(week=week, entrant_id=entrant_id, reason="week not fully scored")
                )
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
