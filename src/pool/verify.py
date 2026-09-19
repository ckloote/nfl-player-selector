"""Check a captured decision: re-derive it from its own record, and replay its instant.

Two checks, each answering one question about a decision `recommend` or `pool week` wrote
down. Reconstruction asks whether the record is sufficient: re-deriving the advice from
the stored surface alone must give what was recorded beside it, or something the decision
depended on was never written down. Parity asks whether the live path and the archive are
the same function: rebuilding the same instant from the archived feeds must give the same
surface and the same advice, which is what every replay of a past week assumes.

These outlived the Phase 3C collection protocol they were written for, which closed on
2026-09-07 without collecting. Its window, floors, populations and baseline export were
retired; the protocol, its signed note and its results stay in `docs/` and `experiments/`.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import ExitStack
from datetime import datetime

import numpy as np
import pandas as pd

from . import capture, db, projections, state
from .recommend import advise_week

# The live path's role source, which every captured decision was made with.
ROLE_SOURCE = "depth"

# Compared exactly between the captured surface and a snapshot replay of the same
# instant. `hard_eligible` is not among them on purpose: live loading leaves the deadline
# out of it and applies it in the recommender, while snapshot replay folds it in, so the
# two differ by construction and comparing them would report a difference that is the
# contract rather than a defect. The deadline is reconciled separately instead.
PARITY_COLUMNS = (
    "base_rate",
    "def_mult",
    "vegas_mult",
    "home_mult",
    "avail_mult",
    "role_mult",
    "lam",
)
PARITY_KEYS = ("week", "slot", "player_id")

# `state.availability` values that mean the cell could not be acted on. The forecast was
# usable; the cell was not.
BLOCKED_STATUS = ("deadline passed", "kickoff unconfirmed")


def _advice_details(advice) -> dict:
    """Re-derived advice in the form the log stores it, so the comparison is like for
    like. A round trip through JSON turns the plan's integer week keys into strings and
    numpy scalars into plain numbers; comparing the live objects against the stored
    record reports every decision as differing in the encoding rather than the answer.
    """
    return json.loads(capture._json({a.slot: capture._advice_detail(a) for a in advice}))


def _claims_hold(derived: dict | None, recorded: dict | None) -> bool:
    """Does everything the record actually wrote down still re-derive?

    Compared on the recorded detail's own keys rather than by whole-dictionary equality.
    A decision captured before a field existed cannot re-derive that field, and demanding
    it would report every decision taken before the newest one as insufficient -- which is
    the opposite of what those decisions are. Adding the win-probability view retroactively
    failed all five of the 2026 captures this way before this narrowing.

    It is still a strict check of what was claimed: a recorded key whose value no longer
    re-derives fails, and so does a recorded key the current code no longer produces at
    all. Only keys the record never had are exempt, and about those it has no claim to
    make.
    """
    if derived is None or recorded is None:
        return derived == recorded
    return all(derived.get(key) == value for key, value in recorded.items())


def _advice_matches(derived: dict, recorded: dict) -> dict[str, bool]:
    return {
        slot: _claims_hold(derived.get(slot), recorded.get(slot))
        for slot in set(derived) | set(recorded)
    }


def reconstruction(
    conn: sqlite3.Connection, decision_id: str, *, allow_code_drift: bool = False
) -> dict:
    """Re-derive a captured decision from its stored surface and diff it against record.

    This is the check that the capture is sufficient. If the re-derived advice differs
    from what was recorded beside it, something the decision depended on was never
    written down, and no later diagnostic over that surface describes the decision that
    was actually made.
    """
    # Read before rebuilding, so a decision the fingerprint refuses still reports what
    # moved. Taking it from the rebuild's own return would leave the failing case -- the
    # one an operator most needs described -- with no drift record at all.
    try:
        payload = capture.recorded_identity(conn, decision_id)
        drift = dict(
            capture.fingerprint_drift(payload),
            constants_changed=capture.constants_drift(payload.get("constants", {})),
        )
    except (ValueError, KeyError):
        drift = {}
    try:
        _frame, advice, recorded, rebuilt_drift = capture.reconstruct(
            conn, decision_id, allow_code_drift=allow_code_drift
        )
    except ValueError as exc:
        return dict(decision_id=decision_id, ok=False, reason=str(exc), slots={}, drift=drift)
    # The rebuild's own record where there is one; the pre-computed stands in only on the
    # failure path, and carries the same keys so a reader never has to ask which produced it.
    drift = rebuilt_drift
    slots = _advice_matches(_advice_details(advice), recorded)
    return dict(
        decision_id=decision_id,
        ok=all(slots.values()) and bool(slots),
        reason=None if all(slots.values()) and slots else "re-derived advice differs",
        slots=slots,
        drift=drift,
        differing=sorted(slot for slot, same in slots.items() if not same),
    )


def _keyed(frame: pd.DataFrame, columns) -> pd.DataFrame:
    return frame.set_index(list(PARITY_KEYS))[list(columns)].sort_index()


def _captured_observations(conn: sqlite3.Connection, decision_id: str) -> dict:
    """The observations the decision recorded for itself, from the append-only table.

    Not `capture.observed_inputs`: that re-resolves `input_observations` at the time it
    is asked, which is the same table and the same rule the replay uses. Both sides would
    then move together, and a backdated observation -- one archived afterwards but stamped
    before the decision -- would change what the replay reads while the comparison went on
    reporting agreement.
    """
    return {
        (int(r["season"]), r["feed"]): (
            r["observation_id"],
            r["content_hash"],
            r["observed_at"],
            bool(r["missing"]),
        )
        for r in conn.execute(
            "SELECT season, feed, observation_id, content_hash, observed_at, missing "
            "FROM decision_inputs WHERE decision_id = ?",
            (decision_id,),
        )
    }


def _replayed_observations(provenance) -> dict:
    """The same identity, from what the replay actually resolved.

    Identity, not content: two observations of a feed can carry identical bytes and still
    be different readings, and a comparison on the content hash alone would call them the
    same observation. Missingness travels with it for the same reason.
    """
    return {
        (int(r["season"]), r["feed"]): (
            r.get("observation_id"),
            r.get("content_hash"),
            r.get("observed_at"),
            bool(r.get("missing")),
        )
        for r in provenance
    }


def parity(conn: sqlite3.Connection, decision_id: str, *, allow_code_drift: bool = False) -> dict:
    """Compare a captured live decision against a snapshot replay of the same instant.

    Phase 2 replay assumes the live path and the archive are the same function of the
    same inputs. Nothing had ever checked that on a real decision: the fixture that
    checks it builds both sides in one process from a staged database. This resolves the
    archive at the captured timestamp and rebuilds, so a divergence shows up as a
    divergence rather than as a quietly different forecast.

    The rebuild runs under the settings the decision recorded rather than today's.
    `build_projections` reads its multipliers when it is called, so a moved constant would
    otherwise be reported as a live/archive divergence -- and reported while
    reconstruction still passed, because the stored surface already has the old multiplier
    baked into it.

    A decision whose archive cannot be resolved is a failure, not an exemption. The
    archive is written by `refresh`, so an unresolvable decision means the feeds were
    never snapshotted before the pick -- which is a gap in the evidence, and the operator
    is the only one who can close it.
    """
    head = conn.execute(
        "SELECT * FROM decision_events WHERE decision_id = ? AND kind = 'surface' "
        "ORDER BY event_id LIMIT 1",
        (decision_id,),
    ).fetchone()
    if head is None:
        return dict(
            decision_id=decision_id, ok=False, reason="no captured surface", checkable=False
        )
    season, week, decision_at = int(head["season"]), int(head["week"]), head["decision_at"]
    stored = capture.load_surface(conn, head["surface_hash"])
    detail = json.loads(head["detail"])
    unchecked = dict(
        decision_id=decision_id, week=week, decision_at=decision_at, ok=False, checkable=False
    )

    try:
        identity_payload = capture.recorded_identity(conn, decision_id)
    except ValueError as exc:
        return dict(unchecked, reason=str(exc))
    # The decision-scoped fingerprint, not the whole tree: parity rebuilds the surface
    # from `projections` and `snapshots`, and a module outside that closure cannot have
    # moved it.
    code_ok = (
        allow_code_drift or not capture.fingerprint_drift(identity_payload)["code_hash_changed"]
    )
    advice_reason = (
        None if code_ok else "source tree differs from the one the decision was captured under"
    )

    # One settings block around the whole rebuild: loading the frames, building the
    # projections and re-deriving the advice each read their constants when called. A
    # structured setting the log cannot restore leaves the decision unverifiable rather
    # than checkable and wrong.
    restored = ExitStack()
    try:
        restored.enter_context(capture.recorded_settings(identity_payload.get("constants", {})))
    except ValueError as exc:
        return dict(unchecked, reason=f"cannot rebuild under the decision's own settings: {exc}")

    with restored:
        try:
            frames = projections.load_frames(
                conn, season, week, input_policy="snapshots", decision_at=decision_at
            )
            replayed = projections.build_projections(frames, week, ROLE_SOURCE)
        except (ValueError, KeyError) as exc:
            return dict(unchecked, reason=f"archive cannot be resolved at the decision: {exc}")

        left, right = _keyed(stored, PARITY_COLUMNS), _keyed(replayed, PARITY_COLUMNS)
        missing = left.index.difference(right.index)
        extra = right.index.difference(left.index)
        shared = left.index.intersection(right.index)
        a, b = left.loc[shared], right.loc[shared]
        differing, worst = [], 0.0
        for column in PARITY_COLUMNS:
            x, y = a[column].to_numpy(dtype=float), b[column].to_numpy(dtype=float)
            if not np.array_equal(x, y, equal_nan=True):
                differing.append(column)
                gap = np.nanmax(np.abs(x - y)) if len(x) else np.nan
                worst = max(worst, float(0.0 if np.isnan(gap) else gap))

        # The deadline, reconciled rather than compared. Live `hard_eligible` omits it and
        # the captured `decision_status` carries it; the replay folds it in. Equality of
        # the two sides after that reconciliation is the same fact the columns cannot
        # state directly.
        decidable = ~stored.decision_status.isin(BLOCKED_STATUS)
        reconciled = _keyed(
            stored.assign(eligible=stored.hard_eligible.fillna(False).astype(bool) & decidable),
            ["eligible"],
        ).loc[shared]
        replayed_eligible = _keyed(
            replayed.assign(eligible=replayed.hard_eligible.fillna(False).astype(bool)),
            ["eligible"],
        ).loc[shared]
        deadline_ok = bool(
            np.array_equal(reconciled.eligible.to_numpy(), replayed_eligible.eligible.to_numpy())
        )

        # The same instant must name the same observations, or the two sides agreed about
        # inputs neither of them read.
        captured_inputs = _captured_observations(conn, decision_id)
        replay_inputs = _replayed_observations(frames.provenance)
        differing_observations = sorted(
            f"{key[0]}:{key[1]}"
            for key in set(captured_inputs) | set(replay_inputs)
            if captured_inputs.get(key) != replay_inputs.get(key)
        )
        observations_ok = bool(captured_inputs) and not differing_observations

        advice_ok = False
        if code_ok:
            try:
                derived = advise_week(
                    replayed,
                    week,
                    set(detail["used"]),
                    {
                        slot: {int(w): pid for w, pid in cells.items()}
                        for slot, cells in detail["locked"].items()
                    },
                    now=state.eastern_now(datetime.fromisoformat(decision_at)),
                    # Restored from the record for the same reason `reconstruct` restores
                    # it: rebuilding the opposition from today's standings would replay the
                    # decision against a pool that has since spent more weeks.
                    pool=capture._pool_from_detail(detail.get("pool")),
                )
                recorded = {
                    r["slot"]: json.loads(r["detail"])
                    for r in conn.execute(
                        "SELECT slot, detail FROM decision_events "
                        "WHERE decision_id = ? AND kind = 'advice' ORDER BY event_id",
                        (decision_id,),
                    )
                }
                advice_ok = all(_advice_matches(_advice_details(derived), recorded).values())
                advice_reason = None if advice_ok else "replayed advice differs from the record"
            except (ValueError, KeyError) as exc:
                advice_reason = str(exc)

    reasons = []
    if len(missing) or len(extra) or differing:
        reasons.append("surface differs")
    if not deadline_ok:
        reasons.append("deadline reconciliation differs")
    if not captured_inputs:
        reasons.append("the decision recorded no inputs of its own")
    elif differing_observations:
        reasons.append(f"replay read other observations: {', '.join(differing_observations)}")
    if advice_reason:
        reasons.append(advice_reason)
    return dict(
        decision_id=decision_id,
        week=week,
        decision_at=decision_at,
        ok=not reasons,
        checkable=True,
        rows_captured=int(len(left)),
        rows_replayed=int(len(right)),
        rows_missing_in_replay=int(len(missing)),
        rows_only_in_replay=int(len(extra)),
        differing_columns=differing,
        largest_difference=worst,
        deadline_reconciled=deadline_ok,
        observations_match=observations_ok,
        differing_observations=differing_observations,
        advice_matches=advice_ok,
        reason="; ".join(reasons) or None,
    )


def decisions(conn: sqlite3.Connection, season: int, week: int | None = None) -> pd.DataFrame:
    """Every captured decision of the season, or of one week, in the order it was made."""
    rows = db.read_df(
        conn,
        "SELECT decision_id, week, decision_at, recorded_at, surface_hash, identity_hash "
        "FROM decision_events WHERE season = ? AND kind = 'surface' ORDER BY event_id",
        (season,),
    )
    if week is not None:
        rows = rows[rows.week.eq(week)].reset_index(drop=True)
    return rows
