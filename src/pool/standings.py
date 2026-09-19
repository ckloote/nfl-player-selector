"""Cache-free entrant scores, competition ranks, and observed player depletion."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from . import config, db, entrants, results, state


@dataclass(frozen=True)
class EntrantPick:
    entrant_id: str
    display_name: str
    is_me: bool
    week: int
    slot: str
    player_name: str | None
    player_id: str | None
    game_id: str | None
    tds: int | None
    status: str  # final, pending, unresolved, missing
    pending: str = ""
    source: str = "reported"
    note: str = ""


@dataclass(frozen=True)
class Total:
    tds: int = 0
    pending: int = 0
    unresolved: int = 0
    missing: int = 0

    @property
    def incomplete(self) -> bool:
        return bool(self.pending or self.unresolved or self.missing)


@dataclass(frozen=True)
class UsedPool:
    player_ids: frozenset[str] = frozenset()
    unknown: int = 0
    repeats: dict[str, tuple[int, ...]] = field(default_factory=dict)
    through: int | None = None
    missing_weeks: tuple[int, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.unknown and not self.missing_weeks


@dataclass(frozen=True)
class Standing:
    entrant_id: str
    display_name: str
    is_me: bool
    rank: int | None
    share: float
    picks: tuple[EntrantPick, ...]
    week_total: Total
    season_total: Total
    used: UsedPool
    reported_week: int | None
    reported_total: int | None
    reported_rank: int | None


@dataclass(frozen=True)
class Board:
    season: int
    week: int | None
    as_of: int | None
    rows: tuple[Standing, ...]
    in_progress: tuple[EntrantPick, ...]
    leaders: tuple[Standing, ...]
    report_conflicts: bool
    used_through: int | None
    season_weeks: int | None = None

    @property
    def season_complete(self) -> bool:
        """Every regular-season week is ranked, so a tie for first really does split.

        `final` says only that the ranked prefix has no gaps, which is true after one
        scored week of eighteen. Whether money is shared is decided when the season runs
        out of weeks, not when the weeks played so far happen to be fully scored.
        """
        return (
            self.final and self.season_weeks is not None and (self.as_of or 0) >= self.season_weeks
        )

    @property
    def final(self) -> bool:
        return (
            self.as_of is not None
            and bool(self.rows)
            and all(not row.season_total.incomplete for row in self.rows)
        )


def _season_weeks(conn, season) -> int | None:
    last = conn.execute(
        "SELECT MAX(week) FROM games WHERE season = ? AND game_type = 'REG'", (season,)
    ).fetchone()[0]
    return int(last) if last is not None else None


def _optional(value):
    return None if pd.isna(value) else value


def score_rows(conn, season, rows, scores) -> list[EntrantPick]:
    result = []
    for row in rows.itertuples():
        name, pid = _optional(row.player_name), _optional(row.player_id)
        gid = _optional(row.game_id)
        note = ""
        if name is None:
            value, status, reason = 0, "final", ""
        elif pid is None:
            value, status, reason = None, "unresolved", "name unresolved; re-import pool report"
        else:
            gid = results.resolve_pick_game(conn, season, int(row.week), pid, gid)
            score = results.score_pick(scores, int(row.week), pid, gid)
            value, reason, note = score.tds, score.pending, score.note
            status = "pending" if reason else "final"
        result.append(
            EntrantPick(
                row.entrant_id,
                row.display_name,
                bool(row.is_me),
                int(row.week),
                row.slot,
                name,
                pid,
                gid,
                value,
                status,
                reason,
                note=note,
            )
        )
    return result


def entrant_scores(conn: sqlite3.Connection, season: int, *, weeks=None) -> list[EntrantPick]:
    """Score reported picks without reading or writing a personal-score cache."""
    with db.transaction(conn):
        rows = entrants.entrant_picks(conn, season)
        if weeks is not None:
            rows = rows[rows.week.isin(weeks)]
        return score_rows(conn, season, rows, results.score_board(conn, season))


def used_pools(
    conn: sqlite3.Connection, season: int, *, through: int | None = None
) -> dict[str, UsedPool]:
    """Observe all imports, including unfinished games; repeats never spend a player twice."""
    with db.transaction(conn):
        rows = entrants.entrant_picks(conn, season)
        identities = entrants.members(conn, season)
    if through is not None:
        rows = rows[rows.week.le(through)]
    last = through if through is not None else (int(rows.week.max()) if len(rows) else None)
    result = {}
    for eid in identities.entrant_id:
        picks = rows[rows.entrant_id.eq(eid)]
        spent = defaultdict(list)
        for pick in picks[picks.player_id.notna()].itertuples():
            spent[pick.player_id].append(int(pick.week))
        result[eid] = UsedPool(
            frozenset(spent),
            int((picks.player_name.notna() & picks.player_id.isna()).sum()),
            {pid: tuple(sorted(weeks)) for pid, weeks in spent.items() if len(weeks) > 1},
            last,
            tuple(sorted(set(range(1, (last or 0) + 1)) - set(picks.week))),
        )
    return result


def remaining_counts(
    conn: sqlite3.Connection, season: int, week: int, entrant_id: str
) -> dict[str, int]:
    """Count what was still available going into `week`, by slot.

    Spending is counted through `week - 1`, not through every imported week. A report can
    arrive before that week's games finish, so counting it here would answer "what could
    they pick in week N" differently depending on whether week N's report had landed --
    which is exactly the leak the reporting cadence makes easy. Consult `used_pools` for
    names that never resolved, which are spent by somebody this count cannot exclude.
    """
    with db.transaction(conn):
        used = used_pools(conn, season, through=week - 1)[entrant_id]
        pool = state.historical_pool(conn, season, week)
    remaining = pool[~pool.player_id.isin(used.player_ids)]
    return {
        slot: int(remaining.position.isin(positions).sum())
        for slot, positions in config.SLOTS.items()
    }


def _total(picks) -> Total:
    return Total(
        sum(p.tds for p in picks if p.tds is not None),
        sum(p.status == "pending" for p in picks),
        sum(p.status == "unresolved" for p in picks),
        sum(p.status == "missing" for p in picks),
    )


def board(conn: sqlite3.Connection, season: int, *, week: int | None = None) -> Board:
    """Rank one consistent snapshot through its last unbroken complete week.

    The displayed week and used-player cutoff are independent of the ranking cutoff.
    A missing report is represented by three missing slots, never three final zeroes.
    """
    with db.transaction(conn):
        return _board(conn, season, week)


def _board(conn, season, week) -> Board:
    raw = entrants.entrant_picks(conn, season)
    latest = int(raw.week.max()) if len(raw) else None
    shown = week if week is not None else latest
    scores = results.score_board(conn, season)
    as_of = results.resolved_through(conn, season)
    picks = score_rows(conn, season, raw, scores)
    used = used_pools(conn, season)
    identities = entrants.members(conn, season)
    reported = {r.entrant_id: r for r in entrants.reported_totals(conn, season, shown).itertuples()}
    indexed = {(p.entrant_id, p.week, p.slot): p for p in picks}

    def slots(entrant, wk):
        return tuple(
            indexed.get((entrant.entrant_id, wk, slot))
            or EntrantPick(
                entrant.entrant_id,
                entrant.display_name,
                bool(entrant.is_me),
                wk,
                slot,
                None,
                None,
                None,
                None,
                "missing",
                "report missing; run pool report import",
            )
            for slot in config.SLOTS
        )

    totals = {
        r.entrant_id: _total([p for wk in range(1, (as_of or 0) + 1) for p in slots(r, wk)])
        for r in identities.itertuples()
    }
    ordered = sorted(
        identities.itertuples(),
        key=lambda r: (
            -totals[r.entrant_id].tds,
            r.display_name.casefold(),
            r.entrant_id,
        ),
    )
    top = max((t.tds for t in totals.values()), default=0)
    leaders_count = sum(t.tds == top for t in totals.values())
    rows, previous, rank = [], None, None
    for index, entrant in enumerate(ordered, 1):
        total = totals[entrant.entrant_id]
        if as_of is not None and total.tds != previous:
            rank = index
        previous = total.tds
        selected = slots(entrant, shown) if shown is not None else ()
        supplied = reported.get(entrant.entrant_id)
        rows.append(
            Standing(
                entrant.entrant_id,
                entrant.display_name,
                bool(entrant.is_me),
                rank,
                1 / leaders_count if as_of is not None and total.tds == top else 0.0,
                selected,
                _total(selected),
                total,
                used[entrant.entrant_id],
                *(
                    _optional(getattr(supplied, name)) if supplied else None
                    for name in ("reported_week", "reported_total", "reported_rank")
                ),
            )
        )
    in_progress = [p for p in picks if p.week > (as_of or 0)]
    imported_weeks = set(raw.week)
    mine = state.picks(conn, season)
    me = next((r for r in rows if r.is_me), None)
    conflicts = False
    for pick in mine.itertuples():
        gid = results.resolve_pick_game(conn, season, int(pick.week), pick.player_id, pick.game_id)
        score = results.score_pick(scores, int(pick.week), pick.player_id, gid)
        supplied = indexed.get((me.entrant_id, pick.week, pick.slot)) if me else None
        if supplied and supplied.status != "unresolved" and supplied.player_id != pick.player_id:
            conflicts = True
        if pick.week > (as_of or 0) and pick.week not in imported_weeks:
            in_progress.append(
                EntrantPick(
                    me.entrant_id if me else "",
                    me.display_name if me else "Me",
                    True,
                    int(pick.week),
                    pick.slot,
                    pick.player_name,
                    pick.player_id,
                    gid,
                    score.tds,
                    "pending" if score.pending else "final",
                    score.pending,
                    "recorded",
                )
            )
    return Board(
        season,
        shown,
        as_of,
        tuple(rows),
        tuple(sorted(in_progress, key=lambda p: (p.week, p.display_name.casefold(), p.slot))),
        tuple(r for r in rows if r.share),
        conflicts,
        latest,
        _season_weeks(conn, season),
    )
