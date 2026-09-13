"""Delivered pool reports: archive first, then derive attributable entrant records."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
import zlib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from numbers import Real

import pandas as pd

from . import config, db, scoring, state

SCHEMA_VERSION = 1
TOTAL_COLUMNS = ["reported_week", "reported_total", "reported_rank"]
META_PREFIX = "pool_report:"


@dataclass
class ImportResult:
    observation_id: int
    season: int
    week: int | None
    check: bool = False
    parsed: bool = False
    written: bool = False
    entrants: int = 0
    picks: int = 0
    totals: int = 0
    unresolved: list[dict] = field(default_factory=list)
    blanks: list[dict] = field(default_factory=list)
    previous_week: int | None = None
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    unrecorded: list[dict] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Findings that mean the import cannot be trusted as it stands.

        `unrecorded` and `blanks` are deliberately absent. A week I never recorded is
        incomplete bookkeeping of my own, and a slot the report says went unpicked is a
        fact about the week; neither says the ingested data is wrong. Failing on them
        would make the routine import red and teach the exit code to be ignored.
        """
        return not (self.errors or self.unresolved or self.conflicts)


def parse_csv(raw: bytes) -> pd.DataFrame:
    """Read UTF-8 CSV without treating literal names such as 'NA' as missing."""
    try:
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True))
    except (UnicodeError, csv.Error) as exc:
        raise ValueError(f"Cannot parse CSV: {exc}") from exc
    if not rows:
        raise ValueError("Report is empty")
    columns = [c.strip().lower() for c in rows[0]]
    if not all(columns) or len(set(columns)) != len(columns):
        raise ValueError("CSV headers must be nonempty and unique")
    data = []
    for line, row in enumerate(rows[1:], 2):
        if not row or not any(c.strip() for c in row):
            continue
        if len(row) != len(columns):
            raise ValueError(f"CSV row {line}: expected {len(columns)} fields, got {len(row)}")
        data.append(row)
    return pd.DataFrame(data, columns=columns)


PARSERS = {"csv": parse_csv}


def _blank(value) -> bool:
    """True when the report supplied no player for a slot."""
    return value is None or pd.isna(value)


def _integer(value, column: str) -> int:
    if isinstance(value, Real) and pd.notna(value) and value >= 0 and value <= 2**63 - 1:
        if int(value) == value:
            return int(value)
    text = str(value).strip()
    if not text.isascii() or not text.isdecimal() or int(text) > 2**63 - 1:
        raise ValueError(f"{column} must be a nonnegative integer; got {value!r}")
    return int(text)


def _report_week(frame: pd.DataFrame, week: int | None) -> int:
    if "week" in frame:
        weeks = {_integer(value, "week") for value in frame.week}
        if len(weeks) != 1:
            raise ValueError("Report must contain exactly one week")
        reported = weeks.pop()
        if week is not None and week != reported:
            raise ValueError(f"Report week {reported} disagrees with --week {week}")
        return reported
    if week is None:
        raise ValueError("Give --week or include a week column in the report")
    return week


def transform_report(
    frame: pd.DataFrame, season: int, week: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate one complete week, separating pick and standing grains."""
    required = {"entrant", "slot", "player_name"}
    allowed = required | {"season", "week", *TOTAL_COLUMNS}
    if missing := required - set(frame.columns):
        raise ValueError(f"Report missing required columns: {sorted(missing)}")
    if extra := set(frame.columns) - allowed:
        raise ValueError(f"Unknown report columns: {sorted(extra)}")
    if frame.empty:
        raise ValueError("Report has no picks")
    _report_week(frame, week)
    if "season" in frame and {_integer(v, "season") for v in frame.season} != {season}:
        raise ValueError(f"Report season disagrees with --season {season}")
    work = frame.copy()
    for column in ("entrant", "slot"):
        if work[column].isna().any() or work[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Report contains an empty {column}")
    # A blank player_name is the report saying this entrant submitted nothing for the
    # slot. That is a fact about the week, so it is stored; a slot row that is absent
    # entirely is still rejected below, because it cannot be told from a truncated file.
    work["player_name"] = pd.Series(
        [None if pd.isna(v) or not str(v).strip() else v for v in work.player_name],
        index=work.index,
        dtype=object,
    )
    # Names remain exactly as delivered; only identifiers and slots are normalized.
    work["entrant_id"] = work.entrant.map(state._norm)
    if work.entrant_id.eq("").any():
        raise ValueError("Entrant names must contain letters or numbers")
    aliases = {"WR": "FLEX", "TE": "FLEX", "WR/TE": "FLEX"}
    work["slot"] = work.slot.str.strip().str.upper().replace(aliases)
    if unknown := set(work.slot) - set(config.SLOTS):
        raise ValueError(f"Unknown slots: {sorted(unknown)}; expected {list(config.SLOTS)}")
    if work.duplicated(["entrant_id", "slot"]).any():
        raise ValueError("Duplicate entrant/slot (including normalized name or slot collisions)")
    for entrant, rows in work.groupby("entrant_id", sort=False):
        if missing := set(config.SLOTS) - set(rows.slot):
            raise ValueError(f"{entrant}: missing slots {sorted(missing)}")
    for column in TOTAL_COLUMNS:
        values = work[column] if column in work else pd.Series(None, index=work.index)
        work[column] = pd.Series(
            [None if pd.isna(v) or not str(v).strip() else _integer(v, column) for v in values],
            index=work.index,
            dtype=object,
        )
        if column == "reported_rank" and work[column].eq(0).any():
            raise ValueError("reported_rank must be positive")
    entrant_rows, total_rows = [], []
    for entrant, rows in work.groupby("entrant_id", sort=False):
        entrant_rows.append(
            dict(season=season, entrant_id=entrant, display_name=rows.entrant.iloc[0])
        )
        total = dict(season=season, week=week, entrant_id=entrant)
        for column in TOTAL_COLUMNS:
            values = rows[column].dropna().unique()
            if len(values) > 1:
                raise ValueError(f"{entrant}: conflicting {column} values")
            total[column] = int(values[0]) if len(values) else None
        total_rows.append(total)
    picks = work[["entrant_id", "slot", "player_name"]].copy()
    picks.insert(0, "week", week)
    picks.insert(0, "season", season)
    return pd.DataFrame(entrant_rows), picks, pd.DataFrame(total_rows, dtype=object)


def resolve_players(
    conn: sqlite3.Connection, season: int, week: int, picks: pd.DataFrame
) -> tuple[pd.DataFrame, list[dict]]:
    """Resolve by slot against week-specific history; retain every unresolved name."""
    history = state.historical_pool(conn, season, week)
    resolved = picks.copy()
    resolved["player_id"] = None
    resolved["game_id"] = None
    unresolved = []
    for index, pick in picks.iterrows():
        if _blank(pick.player_name):
            continue  # the report named no player; there is nothing to resolve
        matches = state.find_player(history, pick.player_name, positions=config.SLOTS[pick.slot])
        if len(matches) != 1:
            unresolved.append(
                dict(
                    entrant_id=pick.entrant_id,
                    slot=pick.slot,
                    player_name=pick.player_name,
                    candidates=matches[["player_id", "player_name", "team", "position"]]
                    .fillna("")
                    .to_dict("records"),
                )
            )
            continue
        player = matches.iloc[0]
        stat_game = conn.execute(
            "SELECT g.* FROM player_weeks p JOIN games g ON g.game_id = p.game_id "
            "WHERE p.season = ? AND p.week = ? AND p.player_id = ? "
            "AND g.season = p.season AND g.week = p.week AND g.game_type = 'REG'",
            (season, week, player.player_id),
        ).fetchone()
        game = stat_game or db.resolve_game(conn, season, week, player.team)
        resolved.at[index, "player_id"] = player.player_id
        resolved.at[index, "game_id"] = game["game_id"] if game else None
    return resolved, unresolved


def archive_report(
    conn: sqlite3.Connection,
    season: int,
    week: int | None,
    raw: bytes,
    *,
    source: str,
    observed_at: str | datetime | None = None,
    coverage: dict,
) -> int:
    """Commit delivered bytes before parsing. Requires a connection with no open write."""
    if conn.in_transaction:
        raise ValueError("Report archiving requires a connection without an open transaction")
    instant = observed_at or datetime.now(UTC)
    if isinstance(instant, str):
        instant = datetime.fromisoformat(instant)
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("observed_at must include a timezone")
    stamp = instant.astimezone(UTC).isoformat(timespec="microseconds")
    digest = hashlib.sha256(raw).hexdigest()
    with db.transaction(conn):
        conn.execute(
            "INSERT OR IGNORE INTO input_payloads VALUES (?, ?, 'zlib-bytes', ?)",
            (digest, SCHEMA_VERSION, zlib.compress(raw)),
        )
        cursor = conn.execute(
            "INSERT INTO input_observations "
            "(season, feed, observed_at, source_timestamp, content_hash, coverage, schema_version) "
            "VALUES (?, 'pool_report', ?, NULL, ?, ?, ?)",
            (
                season,
                stamp,
                digest,
                json.dumps(coverage | dict(week=week, source=source), sort_keys=True),
                SCHEMA_VERSION,
            ),
        )
    return cursor.lastrowid


def _compare_me(conn, season, week, me_id, picks) -> tuple[list[dict], list[dict]]:
    """My recorded picks against the report's own row for me, split by what it means.

    A slot I never recorded is a gap in my bookkeeping; a slot where the report
    contradicts what I recorded means one of the two is wrong about what I submitted.
    Only the second is a reason to distrust the import, so they are returned separately.
    """
    if me_id is None:
        return [], []
    mine = {
        r["slot"]: dict(r)
        for r in conn.execute(
            "SELECT slot, player_id, player_name FROM my_picks WHERE season = ? AND week = ?",
            (season, week),
        )
    }
    reported = {r.slot: r for r in picks[picks.entrant_id.eq(me_id)].itertuples()}
    unrecorded, conflicts = [], []
    for slot in config.SLOTS:
        own, report = mine.get(slot), reported.get(slot)
        name = None if report is None or _blank(report.player_name) else report.player_name
        if own is None and name is None:
            continue
        entry = dict(slot=slot, recorded=own["player_name"] if own else None, reported=name)
        if own is None:
            unrecorded.append(entry)
        elif name is None or own["player_id"] != report.player_id:
            # An unresolved reported name is already reported as unresolved, and its
            # identity is unknown rather than different, so it is not a contradiction.
            if name is None or report.player_id is not None:
                conflicts.append(entry)
    return unrecorded, conflicts


def _write_report(conn, result, entrant_rows, picks, totals, me_id):
    stamp = conn.execute(
        "SELECT observed_at FROM input_observations WHERE observation_id = ?",
        (result.observation_id,),
    ).fetchone()[0]
    for entrant in entrant_rows.itertuples():
        conn.execute(
            "INSERT INTO pool_entrants VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(season, entrant_id) DO UPDATE SET display_name = excluded.display_name, "
            "is_me = excluded.is_me",
            (
                result.season,
                entrant.entrant_id,
                entrant.display_name,
                int(entrant.entrant_id == me_id),
                stamp,
            ),
        )
    # A report is the complete week's set. Corrected files must remove obsolete rows,
    # while retaining historical entrant identities and all original observations.
    ids = list(entrant_rows.entrant_id)
    for table in ("pool_picks", "pool_report_totals"):
        conn.execute(
            f"DELETE FROM {table} WHERE season = ? AND week = ? "
            f"AND entrant_id NOT IN ({','.join('?' for _ in ids)})",
            (result.season, result.week, *ids),
        )
    for table, frame, keys in (
        ("pool_picks", picks, ("season", "week", "entrant_id", "slot")),
        ("pool_report_totals", totals, ("season", "week", "entrant_id")),
    ):
        frame = frame.assign(observation_id=result.observation_id)
        columns = list(frame.columns)
        updates = ", ".join(f"{c} = excluded.{c}" for c in columns if c not in keys)
        conn.executemany(
            f"INSERT INTO {table} ({','.join(columns)}) "
            f"VALUES ({','.join('?' for _ in columns)}) "
            f"ON CONFLICT({','.join(keys)}) DO UPDATE SET {updates}",
            db._rows(frame),
        )


def import_report(
    conn: sqlite3.Connection,
    season: int,
    week: int | None,
    raw: bytes,
    *,
    fmt: str = "csv",
    me: str | None = None,
    check: bool = False,
    allow_roster_change: bool = False,
    source: str = "<bytes>",
    observed_at: str | datetime | None = None,
) -> ImportResult:
    observation_id = archive_report(
        conn,
        season,
        week,
        raw,
        source=source,
        observed_at=observed_at,
        coverage=dict(parsed=False, format=fmt),
    )
    result = ImportResult(observation_id, season, week, check=check)
    try:
        if fmt not in PARSERS:
            raise ValueError(f"Unknown report format {fmt!r}; expected {list(PARSERS)}")
        frame = PARSERS[fmt](raw)
        result.week = week = _report_week(frame, week)
        entrant_rows, picks, totals = transform_report(frame, season, week)
        result.parsed = True
        result.entrants, result.picks, result.totals = len(entrant_rows), len(picks), len(totals)
        state.validate_week(conn, season, week)
        picks, result.unresolved = resolve_players(conn, season, week, picks)
        result.blanks = [
            dict(entrant_id=r.entrant_id, slot=r.slot)
            for r in picks.itertuples()
            if _blank(r.player_name)
        ]
        if result.blanks:
            result.warnings.append(
                "No pick reported for "
                + ", ".join(f"{b['entrant_id']} {b['slot']}" for b in result.blanks)
                + "; stored as reported."
            )
        games = scoring.coverage(conn, season)
        games = games[games.week.eq(week)]
        if not games.complete.eq(1).all():
            result.warnings.append(
                f"Week {week}: {int(games.complete.eq(1).sum())}/{len(games)} games complete; "
                "report retained despite incomplete scoring coverage."
            )
        previous = conn.execute(
            "SELECT DISTINCT week FROM pool_picks WHERE season = ? "
            "ORDER BY CASE WHEN week < ? THEN 0 WHEN week = ? THEN 1 ELSE 2 END, "
            "CASE WHEN week < ? THEN -week ELSE week END LIMIT 1",
            (season, week, week, week),
        ).fetchone()
        if previous:
            result.previous_week = int(previous[0])
            old = {
                r[0]
                for r in conn.execute(
                    "SELECT DISTINCT entrant_id FROM pool_picks WHERE season = ? AND week = ?",
                    (season, result.previous_week),
                )
            }
            incoming = set(entrant_rows.entrant_id)
            result.added, result.removed = sorted(incoming - old), sorted(old - incoming)
            if (result.added or result.removed) and not allow_roster_change:
                result.errors.append(
                    "Entrant roster changed; review the difference and re-run with "
                    "--allow-roster-change to acknowledge it."
                )
        own = conn.execute(
            "SELECT entrant_id FROM pool_entrants WHERE season = ? AND is_me = 1", (season,)
        ).fetchone()
        me_id = own[0] if own else None
        if me is not None:
            requested = state._norm(me)
            if requested not in set(entrant_rows.entrant_id):
                raise ValueError(f"--me {me!r} does not identify an entrant in this report")
            if me_id is not None and me_id != requested:
                raise ValueError(f"--me disagrees with the existing identity {me_id!r}")
            me_id = requested
        result.unrecorded, result.conflicts = _compare_me(conn, season, week, me_id, picks)
        # An unacknowledged roster change writes nothing. `_write_report` drops the rows
        # of any entrant absent from the file, so writing first and complaining after
        # would let a truncated delivery destroy the week the tripwire exists to protect.
        with db.transaction(conn):
            if not check and not result.errors:
                _write_report(conn, result, entrant_rows, picks, totals, me_id)
                result.written = True
            # The immutable receipt predates parsing. Its derived outcome belongs in
            # separate metadata, committed with the rows it describes, even for --check.
            conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                (f"{META_PREFIX}{observation_id}", json.dumps(asdict(result), sort_keys=True)),
            )
    except ValueError as exc:
        result.errors.append(str(exc))
        db.set_meta(
            conn, f"{META_PREFIX}{observation_id}", json.dumps(asdict(result), sort_keys=True)
        )
    return result


def reports(conn: sqlite3.Connection, season: int) -> pd.DataFrame:
    """List every receipt, including failures/checks, joined to its parse outcome."""
    rows = []
    for row in conn.execute(
        "SELECT o.*, m.value AS parse_result FROM input_observations o "
        "LEFT JOIN meta m ON m.key = ? || o.observation_id "
        "WHERE o.season = ? AND o.feed = 'pool_report' ORDER BY o.observation_id",
        (META_PREFIX, season),
    ):
        coverage = json.loads(row["coverage"])
        outcome = json.loads(row["parse_result"]) if row["parse_result"] else {}
        rows.append(
            dict(
                observation_id=row["observation_id"],
                observed_at=row["observed_at"],
                content_hash=row["content_hash"],
                source=coverage["source"],
                week=outcome.get("week", coverage["week"]),
                parsed=outcome.get("parsed", False),
                written=outcome.get("written", False),
                check=outcome.get("check", False),
                entrants=outcome.get("entrants"),
                picks=outcome.get("picks"),
                unresolved=len(outcome["unresolved"]) if outcome else None,
                errors=outcome.get("errors", []),
                needs_review=bool(
                    outcome.get("errors") or outcome.get("unresolved") or outcome.get("conflicts")
                ),
            )
        )
    return pd.DataFrame(rows)


def _read_report_rows(conn, season, week, table, order):
    return db.read_df(
        conn,
        f"SELECT p.*, e.display_name, e.is_me, o.observed_at, o.content_hash, o.coverage "
        f"FROM {table} p JOIN pool_entrants e USING(season, entrant_id) "
        "JOIN input_observations o USING(observation_id) "
        "WHERE p.season = ?"
        + (" AND p.week = ?" if week is not None else "")
        + f" ORDER BY {order}",
        (season, week) if week is not None else (season,),
    )


def entrant_picks(conn: sqlite3.Connection, season: int, week: int | None = None) -> pd.DataFrame:
    return _read_report_rows(conn, season, week, "pool_picks", "p.week, p.entrant_id, p.slot")


def reported_totals(conn: sqlite3.Connection, season: int, week: int | None = None) -> pd.DataFrame:
    return _read_report_rows(conn, season, week, "pool_report_totals", "p.week, p.entrant_id")
