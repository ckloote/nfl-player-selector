"""Phase 3C: the prospective identity baseline, and the checks that make it evidence.

Phase 3B ended with no candidate promoted, so there is nothing to shadow and nothing to
compare against. What is left to establish is narrower and has to come first anyway:
that a live decision is captured faithfully, that replaying the same instant from the
archive reproduces it, and that the forward surface can be described by the same frozen
diagnostics the retrospective study used. None of that is a claim about the model.

The protocol is a dated file rather than a set of defaults, for the reason Phase 3B's
margins were: a window, a population or a floor chosen once the captures are on screen
is not a preregistration. `resolve` refuses a protocol that omits any of them, and hashes
the file so an edit to the prose starts a new window rather than reinterpreting a
finished one.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tomllib
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import capture, config, db, diagnostics, projections, state
from . import evaluate as ev
from .recommend import advise_week

SCHEMA_VERSION = 1

# The live path, unchanged. A protocol naming anything else is describing a run this
# module does not perform, which is exactly what Phase 3B's declared-method checks exist
# to refuse.
MODEL = "shipped"
CALIBRATOR = "identity"
INPUT_POLICY = "live"
ROLE_SOURCE = "depth"
CONSTANTS = "shipped"

# The two decision events per week, in the order they occur.
EVENTS = ("thursday_deadline", "sunday_slate")

# Populations of the prospective log. A bake-off has two replayed policies; a live week
# has one recommendation and one submission, and calling either of them a policy's
# selection would name a comparison that was never run.
POPULATIONS = (
    "all_eligible",
    "available",
    "depleted",
    "recommended",
    "submitted",
    "common_top1",
    "common_top3",
    "common_top5",
    "common_top10",
)

# Decision-week facts of this log, for `diagnostics.prepare` to blank on future rows.
DECISION_WEEK_FLAGS = ("baseline_spent", "recommended", "submitted")

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

# Every recorded submission, and what became of it. A pick that was corrected, removed,
# attributed to no captured decision, or credited to a decision that never forecast it
# is a fact about the collection; leaving any of them out of the audit would report a
# tidier history than the one the log holds.
SUBMISSION_COLUMNS = (
    "week",
    "slot",
    "player_id",
    "attributed",
    "link_source",
    "reader_fallback",
    "status",
)
SUBMISSION_STATUS = ("matched", "unmatched surface", "unattributed", "superseded", "withdrawn")

# `state.availability` values that mean the cell could not be acted on. The forecast was
# usable; the cell was not.
BLOCKED_STATUS = ("deadline passed", "kickoff unconfirmed")

# Coverage populations. A week-1 decision forecasts through week 18, and at a week-6
# review those target weeks have no outcomes and cannot have any. Counting them as
# missing coverage would measure the calendar; counting them as zeros would be worse.
CURRENT_COVERAGE_POPULATION = "horizon_0_in_window"
FUTURE_COVERAGE_POPULATION = "target_week_in_window"

RATE_FLOORS = ("min_event_capture_rate", "min_reconstruction_rate", "min_parity_rate")
COVERAGE_FLOORS = ("min_outcome_coverage_current", "min_outcome_coverage_future")
REQUIRED_FLOORS = RATE_FLOORS + COVERAGE_FLOORS

# The review's declared scope, carried as data for the same reason Phase 3B's promotion
# rule was: prose cannot be checked, and a scope that widens to fit an interesting number
# is not a scope.
REVIEW = {
    "reports": ("baseline_readiness",),
    "candidates": ("none",),
    "max_promoted": (0,),
    "policy_claims": ("none",),
    "production_change": ("none",),
}

# Declared method fields, each of which must name what this module actually does.
DECLARED = (
    ("model", MODEL),
    ("calibrator", CALIBRATOR),
    ("input_policy", INPUT_POLICY),
    ("role_source", ROLE_SOURCE),
    ("constants", CONSTANTS),
    ("current_coverage_population", CURRENT_COVERAGE_POPULATION),
    ("future_coverage_population", FUTURE_COVERAGE_POPULATION),
)


# --- protocol ---------------------------------------------------------------
def resolve(path: str | Path) -> dict:
    """Validate and freeze the collection protocol, before a single decision is read.

    Everything checked here ends up in the protocol dict, and the protocol is what the
    window identity hashes, so a window, an event schedule, a population or a floor
    cannot be changed after the fact. The floors especially: a missing floor blocks
    execution, because a threshold chosen once the numbers exist is not a threshold.
    """
    path = Path(path)
    spec = tomllib.loads(path.read_text())
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Protocol schema_version must be {SCHEMA_VERSION}")
    if not spec.get("specification_date"):
        raise ValueError("A collection protocol must be dated before anything is captured")
    if not isinstance(spec.get("season"), int):
        raise ValueError("A collection protocol must name one season")
    weeks = sorted({int(w) for w in spec.get("collection_weeks", [])})
    if not weeks:
        raise ValueError("A collection protocol must declare its collection weeks")
    if weeks != list(range(weeks[0], weeks[-1] + 1)):
        raise ValueError(f"Collection weeks {weeks} must be contiguous")
    spec["collection_weeks"] = weeks
    if spec.get("review_after_week") != weeks[-1]:
        raise ValueError(
            f"review_after_week must be the last collection week ({weeks[-1]}); "
            f"this protocol declares {spec.get('review_after_week')!r}"
        )
    if list(spec.get("decision_events", ())) != list(EVENTS):
        raise ValueError(f"decision_events must be {list(EVENTS)}")
    for name, expected in DECLARED:
        if spec.get(name) != expected:
            raise ValueError(
                f"{name} must be {expected!r}; this protocol declares {spec.get(name)!r}"
            )
    if list(spec.get("populations", ())) != list(POPULATIONS):
        raise ValueError(f"populations must be {list(POPULATIONS)}")
    if list(spec.get("parity_columns", ())) != list(PARITY_COLUMNS):
        raise ValueError(f"parity_columns must be {list(PARITY_COLUMNS)}")
    for name in (
        "parity_reconciles_deadline",
        "parity_requires_same_observations",
        "parity_uses_recorded_settings",
        "reconstruction_checks_advice",
        "submissions_reduce_corrections",
    ):
        if spec.get(name) is not True:
            raise ValueError(f"{name} must be declared true; this module always checks it")
    floors = spec.get("floors") or {}
    # `isinstance(True, int)` and `isinstance(nan, float)` are both true, and neither is a
    # floor. A floor has to be a finite real number to be compared against anything.
    absent = [
        k
        for k in REQUIRED_FLOORS
        if isinstance(floors.get(k), bool)
        or not isinstance(floors.get(k), (int, float))
        or not math.isfinite(floors[k])
    ]
    if absent:
        raise ValueError(f"floors is missing {absent}; a missing floor blocks the collection")
    outside = [k for k in REQUIRED_FLOORS if not 0.0 <= floors[k] <= 1.0]
    if outside:
        raise ValueError(f"floors {outside} are rates and must lie in [0, 1]")
    spec["floors"] = {k: float(floors[k]) for k in REQUIRED_FLOORS}
    review = spec.get("review") or {}
    wrong = [k for k, allowed in REVIEW.items() if review.get(k) not in allowed]
    if wrong:
        raise ValueError(
            f"review declares unsupported {wrong}; no candidate qualified in Phase 3B, so "
            "the scope of this window is frozen with the protocol rather than chosen later"
        )
    # The parsed keys above are what the collection executes; the file also carries the
    # prose that says what they mean. Hashing the text puts that reasoning inside the
    # window identity, so an edit to it starts a new window.
    spec["specification_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    spec["protocol_path"] = str(path)
    return spec


# --- the decision-event schedule --------------------------------------------
def _instant(kickoff: pd.Timestamp) -> datetime:
    """The aware instant behind an Eastern wall-clock kickoff, minus the pick window."""
    naive = kickoff.to_pydatetime() - pd.Timedelta(minutes=config.PICK_DEADLINE_MINUTES)
    return naive.replace(tzinfo=ZoneInfo(config.TIMEZONE)).astimezone(UTC)


def week_events(conn: sqlite3.Connection, season: int, week: int) -> dict[str, datetime]:
    """The deadlines the week's declared decision events must precede.

    `thursday_deadline` is the week's first confirmed kickoff; `sunday_slate` is the main
    slate, resolved the way `recommend.main_slate_start` resolves it. A week whose first
    game is already in the main slate schedules one event rather than two: demanding a
    second would count an impossible event as a miss.
    """
    games = db.read_df(
        conn,
        "SELECT kickoff, kickoff_known FROM games "
        "WHERE season = ? AND week = ? AND game_type = 'REG'",
        (season, week),
    )
    known = games[games.kickoff_known.eq(1)] if len(games) else games
    if not len(known):
        return {}
    kicks = pd.to_datetime(known.kickoff)
    sundays = kicks[kicks.dt.dayofweek == 6]
    slate = sundays.min() if len(sundays) else kicks.max()
    first = kicks.min()
    out = {}
    if first < slate:
        out["thursday_deadline"] = _instant(first)
    out["sunday_slate"] = _instant(slate)
    return {name: out[name] for name in EVENTS if name in out}


def classify_event(decision_at: str, events: dict[str, datetime]) -> str:
    """Which declared event a decision is, by the deadline it beat.

    A decision after the last deadline is not one of the declared events. It is recorded
    under its own name rather than folded into the nearest one, because a pick made after
    the slate has started is a different thing from a pick made before it.
    """
    at = datetime.fromisoformat(decision_at)
    for name in EVENTS:
        if name in events and at <= events[name]:
            return name
    return "after_deadline"


def decisions(conn: sqlite3.Connection, spec: dict) -> pd.DataFrame:
    """Every captured decision whose week is inside the collection window."""
    season, weeks = spec["season"], set(spec["collection_weeks"])
    rows = db.read_df(
        conn,
        "SELECT decision_id, week, decision_at, recorded_at, surface_hash, identity_hash "
        "FROM decision_events WHERE season = ? AND kind = 'surface' ORDER BY event_id",
        (season,),
    )
    if not len(rows):
        return rows.assign(event=pd.Series(dtype=str))
    rows = rows[rows.week.isin(weeks)].reset_index(drop=True)
    schedule = {w: week_events(conn, season, w) for w in sorted(weeks)}
    rows["event"] = [
        classify_event(at, schedule.get(int(w), {}))
        for at, w in zip(rows.decision_at, rows.week, strict=True)
    ]
    return rows


def event_coverage(conn: sqlite3.Connection, spec: dict) -> pd.DataFrame:
    """Scheduled decision events against captured ones, week by week.

    A missed event is evidence that does not exist and cannot be recreated, so it is
    reported as a miss rather than as a smaller denominator.
    """
    season = spec["season"]
    captured = decisions(conn, spec)
    out = []
    for week in spec["collection_weeks"]:
        scheduled = week_events(conn, season, week)
        for name in EVENTS:
            if name not in scheduled:
                continue
            got = captured[captured.week.eq(week) & captured.event.eq(name)]
            out.append(
                dict(
                    season=season,
                    week=week,
                    event=name,
                    deadline=scheduled[name].isoformat(),
                    scheduled=1,
                    captured=int(len(got)),
                    decision_id=got.decision_id.iloc[0] if len(got) else None,
                )
            )
    extra = captured[captured.event.eq("after_deadline")]
    for row in extra.itertuples():
        out.append(
            dict(
                season=season,
                week=int(row.week),
                event="after_deadline",
                deadline=None,
                scheduled=0,
                captured=1,
                decision_id=row.decision_id,
            )
        )
    return pd.DataFrame(out)


# --- fidelity checks --------------------------------------------------------
def _advice_details(advice) -> dict:
    """Re-derived advice in the form the log stores it, so the comparison is like for
    like. A round trip through JSON turns the plan's integer week keys into strings and
    numpy scalars into plain numbers; comparing the live objects against the stored
    record reports every decision as differing in the encoding rather than the answer.
    """
    return json.loads(capture._json({a.slot: capture._advice_detail(a) for a in advice}))


def reconstruction(
    conn: sqlite3.Connection, decision_id: str, *, allow_code_drift: bool = False
) -> dict:
    """Re-derive a captured decision from its stored surface and diff it against record.

    This is the check that the capture is sufficient. If the re-derived advice differs
    from what was recorded beside it, something the decision depended on was never
    written down, and no later diagnostic over that surface describes the decision that
    was actually made.
    """
    try:
        _frame, advice, recorded, drift = capture.reconstruct(
            conn, decision_id, allow_code_drift=allow_code_drift
        )
    except ValueError as exc:
        return dict(decision_id=decision_id, ok=False, reason=str(exc), slots={})
    derived = _advice_details(advice)
    slots = {slot: derived.get(slot) == recorded.get(slot) for slot in set(derived) | set(recorded)}
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


def parity(
    conn: sqlite3.Connection, decision_id: str, spec: dict, *, allow_code_drift: bool = False
) -> dict:
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
    season = spec["season"]
    head = conn.execute(
        "SELECT * FROM decision_events WHERE decision_id = ? AND kind = 'surface' "
        "ORDER BY event_id LIMIT 1",
        (decision_id,),
    ).fetchone()
    if head is None:
        return dict(
            decision_id=decision_id, ok=False, reason="no captured surface", checkable=False
        )
    week, decision_at = int(head["week"]), head["decision_at"]
    stored = capture.load_surface(conn, head["surface_hash"])
    detail = json.loads(head["detail"])
    unchecked = dict(
        decision_id=decision_id, week=week, decision_at=decision_at, ok=False, checkable=False
    )

    try:
        identity_payload = capture.recorded_identity(conn, decision_id)
    except ValueError as exc:
        return dict(unchecked, reason=str(exc))
    code_hash, _, _ = capture._code_identity()
    code_ok = allow_code_drift or identity_payload.get("code_hash") == code_hash
    advice_reason = (
        None if code_ok else "source tree differs from the one the decision was captured under"
    )

    # One settings block around the whole rebuild: loading the frames, building the
    # projections and re-deriving the advice each read their constants when called. A
    # structured setting the log cannot restore leaves the decision unverifiable rather
    # than checkable and wrong -- and unverifiable already counts against the floor.
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
            replayed = projections.build_projections(frames, week, spec["role_source"])
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
                )
                recorded = {
                    r["slot"]: json.loads(r["detail"])
                    for r in conn.execute(
                        "SELECT slot, detail FROM decision_events "
                        "WHERE decision_id = ? AND kind = 'advice' ORDER BY event_id",
                        (decision_id,),
                    )
                }
                advice_ok = _advice_details(derived) == recorded
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


# --- the described surface --------------------------------------------------
def _ranks(rows: pd.DataFrame) -> pd.Series:
    """Rank within slot among the eligible unused candidates of each decision.

    The decision's own pool, not the target week's: a rank is a statement about what was
    on the board when the choice was made.
    """
    out = pd.Series(np.nan, index=rows.index, dtype=float)
    at = rows.lead_horizon.eq(0) & rows.hard_eligible & ~rows.baseline_spent
    sub = rows[at].sort_values(
        ["decision_id", "slot", "lam", "player_id"], ascending=[True, True, False, True]
    )
    out.loc[sub.index] = sub.groupby(["decision_id", "slot"]).cumcount().to_numpy() + 1.0
    return out


def _actions(conn: sqlite3.Connection, season: int) -> pd.DataFrame:
    """Every advice, submission and correction of the season, in the order they happened."""
    return db.read_df(
        conn,
        "SELECT decision_id, week, slot, kind, player_id, detail FROM decision_events "
        "WHERE season = ? AND kind IN ('advice', 'submitted', 'correction') ORDER BY event_id",
        (season,),
    )


def _player(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


def _surviving(events: pd.DataFrame) -> tuple[dict, list[dict]]:
    """Reduce the ordered action history to the pick that still stands in each slot.

    `record` appends a correction naming the player it replaced and then the submission
    that replaced him; `unrecord` appends a correction that removes one. Reading only the
    submissions leaves every player ever entered marked as submitted, so a slot corrected
    twice contributes three forecast rows to a population that describes one -- and a slot
    later emptied contributes rows for a pick that is in nobody's lineup.

    The fold runs across decision ids deliberately. `unrecord` links its correction through
    the latest-advice fallback, so the withdrawal of a Thursday pick can arrive carrying a
    Sunday decision's id; keying the reduction on the slot is what makes it find the pick
    it actually replaced.
    """
    live: dict[tuple[int, str], dict] = {}
    history: list[dict] = []
    for e in events.itertuples():
        if e.kind == "advice":
            continue
        key = (int(e.week), str(e.slot))
        detail = json.loads(e.detail) if e.detail else {}
        if e.kind == "correction":
            gone = live.pop(key, None)
            if gone is not None:
                history.append(
                    dict(gone, status="withdrawn" if detail.get("removed") else "superseded")
                )
            continue
        player_id = _player(e.player_id)
        if player_id is None:
            continue
        live[key] = dict(
            week=key[0],
            slot=key[1],
            player_id=player_id,
            attributed=detail.get("linked_decision") or e.decision_id,
            # How the link was made travels with it. A deliberate `--decision` link and the
            # latest-advice fallback are different claims about which decision the pick came
            # from, and with two decisions in a week the fallback is whichever happened last.
            link_source=detail.get("link_source") or "unrecorded",
            reader_fallback=None,
        )
    return live, history


def _flags(conn: sqlite3.Connection, rows: pd.DataFrame, captured: pd.DataFrame, season: int):
    """Which surface row each slot's advice recommended, and which one is still submitted.

    A submission happens on the pool's own site, so the link back to a decision is
    recorded rather than inferred. Where the recorded link points outside the window the
    submission is attributed to the last decision captured that week, and that reader-side
    substitution is reported separately from the link the operator made -- a submission
    silently dropped is a population that quietly shrinks, and one silently re-attributed
    is a population that quietly lies.
    """
    known = set(captured.decision_id)
    # `decisions` reads in event order, so the last row of a week is the last decision of
    # that week.
    last_of_week = {int(row.week): row.decision_id for row in captured.itertuples()}
    keys = list(
        zip(
            rows.decision_id,
            rows.week.astype(int),
            rows.slot.astype(str),
            rows.player_id.astype(str),
            strict=True,
        )
    )
    surface = set(keys)

    events = _actions(conn, season)
    recommended = set()
    for e in events[events.kind.eq("advice")].itertuples():
        player_id = _player(e.player_id)
        if player_id is not None and e.decision_id in known:
            recommended.add((e.decision_id, int(e.week), str(e.slot), player_id))

    live, history = _surviving(events)
    submitted, links = set(), list(history)
    for entry in live.values():
        record = dict(entry)
        if record["attributed"] not in known:
            record["attributed"] = last_of_week.get(record["week"])
            record["reader_fallback"] = "latest in week"
        if record["attributed"] is None:
            links.append(dict(record, status="unattributed"))
            continue
        cell = (record["attributed"], record["week"], record["slot"], record["player_id"])
        # The decision has to have forecast the player it was credited with picking.
        # Without this a submission naming somebody absent from that capture reads as a
        # successful attribution while contributing no forecast row to describe.
        if cell not in surface:
            links.append(dict(record, status="unmatched surface"))
            continue
        submitted.add(cell)
        links.append(dict(record, status="matched"))

    at_decision = rows.lead_horizon.eq(0).to_numpy()
    return (
        pd.Series([k in recommended for k in keys], index=rows.index) & at_decision,
        pd.Series([k in submitted for k in keys], index=rows.index) & at_decision,
        pd.DataFrame(links, columns=list(SUBMISSION_COLUMNS)).sort_values(
            ["week", "slot", "status"], ignore_index=True
        ),
    )


def recorded_discounts(conn: sqlite3.Connection, captured: pd.DataFrame) -> dict[str, float]:
    """The future discount each decision was actually made under.

    A decision records its constants, so this is a fact about the capture rather than a
    setting of the reader. Where a payload does not carry one -- nothing shipped has ever
    omitted it, but a reader that assumed would be asserting something it had not checked
    -- the current value stands in, and `provenance` says which decisions that applied to.
    """
    out = {}
    for decision_id in captured.decision_id if len(captured) else []:
        try:
            constants = capture.recorded_identity(conn, decision_id).get("constants", {})
        except ValueError:
            constants = {}
        out[decision_id] = float(constants.get("FUTURE_DISCOUNT", config.FUTURE_DISCOUNT))
    return out


def frame(conn: sqlite3.Connection, spec: dict) -> pd.DataFrame:
    """Every captured surface in the window, joined to outcomes and ready to describe.

    Eligibility is reconciled with the deadline here rather than left as the live path
    leaves it. On the live surface `hard_eligible` is availability alone, and a cell
    whose deadline had passed is still marked eligible; describing that as an eligible
    population would mix cells the tool would let you pick with cells it would not.
    After reconciliation the two exclusion classes mean what they mean in Phase 3A: a
    ruled-out player is masked to zero, and an undecidable one keeps his positive rate.
    """
    season = spec["season"]
    captured = decisions(conn, spec)
    known = set(captured.decision_id)
    rows = capture.outcomes(conn, season)
    if not len(rows):
        raise ValueError(f"No captured decisions for {season}; nothing to describe")
    rows = rows[rows.decision_id.isin(known)].reset_index(drop=True)
    if not len(rows):
        raise ValueError(
            f"No captured decisions inside weeks {spec['collection_weeks']} of {season}"
        )
    rows["season"] = season
    # Participation, on the same terms the study exports it: a data-presence proxy for
    # the target week, not an observed active status. The reliability table reads it.
    played = ev.played_pairs(conn, season)
    rows["played"] = [
        (int(w), pid) in played for w, pid in zip(rows.week, rows.player_id, strict=True)
    ]
    decidable = ~rows.decision_status.isin(BLOCKED_STATUS)
    rows["hard_eligible"] = rows.hard_eligible.fillna(False).astype(bool) & decidable
    rows["baseline_spent"] = rows.used.fillna(False).astype(bool)
    rows["rank_available"] = _ranks(rows)
    rows["recommended"], rows["submitted"], links = _flags(conn, rows, captured, season)
    # Each decision's own discount, not today's. The planning quantity is what the
    # optimizer compares, and recomputing it under a setting that moved after the capture
    # would report a different planning value for a decision that never changed.
    out = diagnostics.prepare(
        rows,
        discount=rows.decision_id.map(recorded_discounts(conn, captured)),
        decision_week_flags=DECISION_WEEK_FLAGS,
    )
    out.attrs["submission_links"] = links
    return out


def population_masks(rows: pd.DataFrame) -> dict[str, pd.Series]:
    """The declared populations, defined without reference to any outcome."""

    def flag(column):
        return rows[column].fillna(False).astype(bool)

    eligible = flag("hard_eligible")
    known_depletion = rows.baseline_spent.notna()
    masks = {
        "all_eligible": eligible,
        "available": eligible & known_depletion & ~flag("baseline_spent"),
        "depleted": eligible & known_depletion & flag("baseline_spent"),
        "recommended": eligible & flag("recommended"),
        "submitted": eligible & flag("submitted"),
    }
    for k in (1, 3, 5, 10):
        masks[f"common_top{k}"] = eligible & rows.rank_available.le(k).fillna(False).astype(bool)
    return {name: masks[name] for name in POPULATIONS}


def outcome_coverage(rows: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Coverage over the populations the floors are declared against.

    Future rows whose target week falls outside the window are reported as unsettled and
    excluded from the floor. They are not missing coverage: the week has not been played.
    """
    weeks = set(spec["collection_weeks"])
    eligible = rows.hard_eligible.fillna(False).astype(bool)
    current = rows[eligible & rows.lead_horizon.eq(0) & rows.decision_week.isin(weeks)]
    future_all = rows[eligible & rows.lead_horizon.gt(0)]
    future = future_all[future_all.week.isin(weeks)]
    unsettled = future_all[~future_all.week.isin(weeks)]
    floors = spec["floors"]
    out = [
        dict(
            population=CURRENT_COVERAGE_POPULATION,
            rows=int(len(current)),
            outcomes_known=int(current.outcome_known.sum()),
            coverage=float(current.outcome_known.mean()) if len(current) else np.nan,
            floor=floors["min_outcome_coverage_current"],
        ),
        dict(
            population=FUTURE_COVERAGE_POPULATION,
            rows=int(len(future)),
            outcomes_known=int(future.outcome_known.sum()),
            coverage=float(future.outcome_known.mean()) if len(future) else np.nan,
            floor=floors["min_outcome_coverage_future"],
        ),
        dict(
            population="future_target_week_outside_window",
            rows=int(len(unsettled)),
            outcomes_known=int(unsettled.outcome_known.sum()),
            coverage=np.nan,
            floor=np.nan,
        ),
    ]
    table = pd.DataFrame(out)
    table["meets_floor"] = [
        bool(r.coverage >= r.floor) if not (pd.isna(r.floor) or pd.isna(r.coverage)) else pd.NA
        for r in table.itertuples()
    ]
    return table


def fidelity(conn: sqlite3.Connection, spec: dict, *, allow_code_drift: bool = False):
    """Per-decision reconstruction and parity, and the rates the floors are read on.

    Both rates divide by every captured decision rather than by the ones that could be
    checked. A decision whose archive was never written cannot be verified, and an
    unverifiable decision is not a verified one; putting it in the denominator is what
    keeps the floor from passing by shrinking.
    """
    captured = decisions(conn, spec)
    rows = []
    for row in captured.itertuples():
        rebuilt = reconstruction(conn, row.decision_id, allow_code_drift=allow_code_drift)
        matched = parity(conn, row.decision_id, spec, allow_code_drift=allow_code_drift)
        rows.append(
            dict(
                decision_id=row.decision_id,
                week=int(row.week),
                event=row.event,
                decision_at=row.decision_at,
                reconstructs=bool(rebuilt["ok"]),
                reconstruction_reason=rebuilt.get("reason"),
                parity=bool(matched["ok"]),
                parity_checkable=bool(matched.get("checkable")),
                parity_reason=matched.get("reason"),
                differing_columns=",".join(matched.get("differing_columns") or ()),
                largest_difference=matched.get("largest_difference"),
                deadline_reconciled=matched.get("deadline_reconciled"),
                observations_match=matched.get("observations_match"),
                advice_matches=matched.get("advice_matches"),
            )
        )
    table = pd.DataFrame(rows)
    events = event_coverage(conn, spec)
    scheduled = int(events.scheduled.sum()) if len(events) else 0
    caught = int(events[events.scheduled.eq(1)].captured.clip(upper=1).sum()) if len(events) else 0
    floors = spec["floors"]
    rates = pd.DataFrame(
        [
            dict(
                measure="event_capture",
                numerator=caught,
                denominator=scheduled,
                rate=float(caught / scheduled) if scheduled else np.nan,
                floor=floors["min_event_capture_rate"],
            ),
            dict(
                measure="reconstruction",
                numerator=int(table.reconstructs.sum()) if len(table) else 0,
                denominator=int(len(table)),
                rate=float(table.reconstructs.mean()) if len(table) else np.nan,
                floor=floors["min_reconstruction_rate"],
            ),
            dict(
                measure="parity",
                numerator=int(table.parity.sum()) if len(table) else 0,
                denominator=int(len(table)),
                rate=float(table.parity.mean()) if len(table) else np.nan,
                floor=floors["min_parity_rate"],
            ),
        ]
    )
    rates["meets_floor"] = [
        bool(r.rate >= r.floor) if not pd.isna(r.rate) else pd.NA for r in rates.itertuples()
    ]
    return table, events, rates


# --- provenance and the note ------------------------------------------------
def provenance(conn: sqlite3.Connection, spec: dict) -> dict:
    """Two identities, because two different pieces of code decide these numbers.

    The decisions' own identities say which forecasts are being described -- and there
    can be more than one, because a constant moving mid-window makes two functions of
    the same name. The module hashes say what described them.
    """
    captured = decisions(conn, spec)
    identities = []
    for digest in sorted(set(captured.identity_hash)) if len(captured) else []:
        row = conn.execute(
            "SELECT payload FROM decision_identities WHERE identity_hash = ?", (digest,)
        ).fetchone()
        payload = json.loads(row["payload"]) if row else {}
        constants = payload.get("constants", {})
        identities.append(
            dict(
                identity_hash=digest,
                model=payload.get("model"),
                calibrator=payload.get("calibrator"),
                code_hash=payload.get("code_hash"),
                revision=payload.get("revision"),
                dirty=payload.get("dirty"),
                # The planning quantity is derived under this, so it is part of what
                # describes these decisions rather than a detail of the reader.
                future_discount=float(constants.get("FUTURE_DISCOUNT", config.FUTURE_DISCOUNT)),
                future_discount_source=(
                    "decision" if "FUTURE_DISCOUNT" in constants else "current configuration"
                ),
                decisions=int(captured.identity_hash.eq(digest).sum()),
            )
        )
    drift = {}
    if len(identities) > 1:
        first = json.loads(
            conn.execute(
                "SELECT payload FROM decision_identities WHERE identity_hash = ?",
                (identities[0]["identity_hash"],),
            ).fetchone()["payload"]
        ).get("constants", {})
        for other in identities[1:]:
            payload = json.loads(
                conn.execute(
                    "SELECT payload FROM decision_identities WHERE identity_hash = ?",
                    (other["identity_hash"],),
                ).fetchone()["payload"]
            )
            drift[other["identity_hash"]] = capture.constants_drift(
                first, payload.get("constants", {})
            )
    return {
        "collection_provenance": {
            "protocol": Path(spec["protocol_path"]).name,
            "specification_date": spec["specification_date"],
            "specification_sha256": spec["specification_sha256"],
            "season": spec["season"],
            "collection_weeks": spec["collection_weeks"],
            "decision_events": list(EVENTS),
            "decision_identities": identities,
            "identity_drift": drift,
            # More than one value here means the window holds two functions of the same
            # name. It is reported rather than pooled over.
            "future_discounts": sorted({i["future_discount"] for i in identities}),
        },
        "diagnostic_provenance": {
            "prospective_module": diagnostics.module_hash("prospective"),
            "capture_module": diagnostics.module_hash("capture"),
            "diagnostics_module": diagnostics.module_hash("diagnostics"),
            "evaluate_module": diagnostics.module_hash("evaluate"),
            "min_inference_clusters": config.MIN_INFERENCE_CLUSTERS,
            "populations": list(POPULATIONS),
            "positions": list(diagnostics.POSITIONS),
            "horizon_buckets": [label for _, _, label in diagnostics.HORIZON_BUCKETS],
            "artifact_schema": SCHEMA_VERSION,
        },
    }


def baseline_note(
    spec, rows, fits, checks, events, rates, coverage, submissions, identities
) -> str:
    def table(frame, columns):
        head = "| " + " | ".join(columns) + " |"
        rule = "|" + "|".join("---" for _ in columns) + "|"
        body = [
            "| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in columns) + " |"
            for _, r in frame.iterrows()
        ]
        return [head, rule, *body, ""]

    weeks = spec["collection_weeks"]
    lines = [
        "# Phase 3C Baseline Note",
        "",
        f"Generated {datetime.now(UTC).date().isoformat()} from the captured decisions of "
        f"{spec['season']} weeks {weeks[0]}-{weeks[-1]}, under the protocol dated "
        f"{spec['specification_date']}. Measurements and identities only; interpretation "
        "belongs in a separately authored, dated assessment.",
        "",
        "Identity only. No calibrated candidate qualified in Phase 3B, so nothing is under "
        "evaluation here and nothing is promoted by it. Production constants and submitted "
        "picks are unchanged.",
        "",
        "## Identities",
        "",
    ]
    for block, label in (
        ("collection_provenance", "Decisions described"),
        ("diagnostic_provenance", "Implementation describing them"),
    ):
        lines += [f"**{label}**", ""]
        for k, v in identities[block].items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")
    lines += [
        "## Collection",
        "",
        f"- Captured decisions: {len(checks):,} over weeks "
        f"{sorted(set(checks.week)) if len(checks) else []}.",
        f"- Surface rows: {len(rows):,}, {rows.player_id.nunique():,} players, lead horizons "
        f"{int(rows.lead_horizon.min())}-{int(rows.lead_horizon.max())}.",
        "",
        "### Scheduled events against captured ones",
        "",
    ]
    lines += table(events, ["week", "event", "deadline", "scheduled", "captured"])
    lines += ["### Fidelity", ""]
    lines += table(rates, ["measure", "numerator", "denominator", "rate", "floor", "meets_floor"])
    lines += [
        "Both fidelity rates divide by every captured decision, not by the ones that could "
        "be checked: a decision whose archive was never written cannot be verified, and an "
        "unverifiable decision is not a verified one.",
        "",
        "### Outcome coverage",
        "",
    ]
    lines += table(
        coverage, ["population", "rows", "outcomes_known", "coverage", "floor", "meets_floor"]
    )
    lines += [
        "Future rows whose target week falls outside the window are unsettled rather than "
        "missing: the week has not been played, and neither a missing-coverage count nor a "
        "zero would describe them.",
        "",
    ]
    if len(submissions):
        counts = submissions.groupby("status").size().rename("submissions").reset_index()
        described = int(submissions.status.eq("matched").sum())
        lines += ["### Submissions", ""]
        lines += table(counts, ["status", "submissions"])
        lines += [
            f"{described:,} of {len(submissions):,} recorded submissions are described by "
            "the population. Corrections and removals are folded before counting, so a slot "
            "carries the pick that still stands rather than every pick ever entered, and a "
            "submission credited to a decision that never forecast that player is reported "
            "as unmatched rather than counted as attributed.",
            "",
        ]
    lines += [
        "## Group definitions",
        "",
        f"- Populations: {', '.join(POPULATIONS)}. `recommended` is the row each slot's "
        "advice named; `submitted` is the pick that still stands there after corrections "
        "and removals. Neither is a replayed policy's selection, and no achieved-score "
        "comparison is made here.",
        f"- Positions: {', '.join(diagnostics.POSITIONS)}, WR and TE separately throughout.",
        f"- Lead horizons: "
        f"{', '.join(label for _, _, label in diagnostics.HORIZON_BUCKETS)}, never pooled.",
        "- Eligibility reconciles the deadline: a cell whose kickoff had passed or whose "
        "kickoff was unconfirmed is excluded and counted, not described as eligible.",
        "- Every definition was fixed in the dated protocol before any decision was "
        "captured; none selects on outcomes.",
        "",
        "## Limits",
        "",
        f"- {len(checks):,} decisions over {len(weeks)} weeks is a small sample, sized to "
        "validate capture, parity and coverage. It cannot detect a forecast or policy "
        "difference and no such difference is reported.",
        "- A shadow advice log is not a counterfactual achieved season score. The static "
        "hold and commit events recorded here are the decisions the early deadline forced, "
        "not touchdowns gained or lost by waiting.",
        "- Measuring the value of waiting for news requires a multi-event replay that "
        "processes timestamped observations in order. It does not exist, and this window "
        "does not substitute for it.",
        "- Nothing here authorizes a production change. Production constants are unchanged.",
        "",
    ]
    unsupported = fits[fits.fit_status.eq("unsupported")] if len(fits) else fits
    if len(fits):
        lines += [
            "## Fits",
            "",
            f"- Strata fitted: {len(fits):,}; unsupported: {len(unsupported):,}. A stratum "
            "too small to identify a fit reports its reason rather than a coefficient.",
            "",
        ]
    return "\n".join(lines)


def export(
    conn: sqlite3.Connection,
    spec: dict,
    out: Path,
    *,
    allow_code_drift: bool = False,
    log=print,
) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = frame(conn, spec)
    log(f"Describing {len(rows):,} captured surface rows")
    masks = population_masks(rows)
    described, fitted = diagnostics.strata(rows, masks)
    checks, events, rates = fidelity(conn, spec, allow_code_drift=allow_code_drift)
    coverage = outcome_coverage(rows, spec)
    submissions = rows.attrs.get("submission_links", pd.DataFrame())
    tables = {
        "decisions.csv": checks,
        "events.csv": events,
        "fidelity.csv": rates,
        "outcome-coverage.csv": coverage,
        "submissions.csv": submissions,
        "strata.csv": described,
        "fits.csv": fitted,
        "reliability.csv": diagnostics.reliability(rows, masks),
        "zero-accounting.csv": diagnostics.zero_accounting(rows, masks),
        "coverage.csv": diagnostics.coverage(rows),
    }
    for name, table in tables.items():
        table = table.copy()
        table.insert(0, "schema_version", SCHEMA_VERSION)
        table.to_csv(out / name, index=False, float_format="%.12g")
        log(f"  {name}: {len(table):,} rows")
    identities = provenance(conn, spec)
    (out / "identities.json").write_text(json.dumps(identities, indent=2, sort_keys=True) + "\n")
    (out / "BASELINE.md").write_text(
        baseline_note(spec, rows, fitted, checks, events, rates, coverage, submissions, identities)
    )
    log(f"Wrote {out}")
    return out
