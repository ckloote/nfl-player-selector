"""Recording picks and reading them back: `record`, `unrecord`, `picks`, `score`, and the
`status` and `refresh` of the data they are scored from.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from rich.table import Table

from .. import capture, db, ingest, projections, scoring, snapshots, state
from .common import DbOpt, SeasonOpt, WeekOpt, _conn, _status, _week, console


def refresh(
    season: int = SeasonOpt,
    full: bool = typer.Option(
        False, "--full", help="Also re-download a finished prior season, for stat corrections"
    ),
    db_path: Path | None = DbOpt,
):
    """Pull latest stats, schedule, rosters, injuries from nflverse."""
    conn = _conn(db_path)
    console.print(f"Refreshing {season} (prior season {season - 1})...")
    counts = ingest.refresh(conn, season, log=console.print, full=full)
    for k, v in counts.items():
        console.print(f"  {k}: {v} rows")
    if counts.failures:
        raise typer.Exit(1)


def record(
    week: int | None = WeekOpt,
    qb: str | None = typer.Option(None, help="QB pick (name)"),
    rb: str | None = typer.Option(None, help="RB pick (name)"),
    flex: str | None = typer.Option(None, help="WR/TE pick (name)"),
    season: int = SeasonOpt,
    db_path: Path | None = DbOpt,
    decision: str | None = typer.Option(
        None, "--decision", help="Captured decision these picks came from (see `recommend`)"
    ),
):
    """Log submitted picks, including historical entries and corrections."""
    conn = _conn(db_path)
    now = state.eastern_now()
    wk = _week(conn, season, week, now)
    pool = state.historical_pool(conn, season, wk)
    given = {"QB": qb, "RB": rb, "FLEX": flex}
    if not any(given.values()):
        console.print("[red]Give at least one of --qb, --rb, --flex.[/red]")
        raise typer.Exit(1)
    entries = []
    for slot, name in given.items():
        if not name:
            continue
        matches = state.find_player(pool, name)
        if len(matches) != 1:
            console.print(
                f"[red]{slot}: {'no match' if not len(matches) else 'ambiguous'} for {name!r}[/red]"
            )
            for _, m in matches.head(8).iterrows():
                console.print(f"    {m.player_name} ({m.team} {m.position})")
            raise typer.Exit(1)
        m = matches.iloc[0]
        entries.append(
            dict(
                slot=slot,
                player_id=m.player_id,
                player_name=m.player_name,
                position=m.position,
                team=m.team,
            )
        )
    # `my_picks` holds the current answer; the capture log holds every answer. Both in
    # one transaction: a pick that changed without its history leaves no way back to
    # the identity it overwrote, and retrying cannot recover it.
    try:
        with db.transaction(conn):
            replaced = {
                r["slot"]: r["player_id"]
                for r in conn.execute(
                    "SELECT slot, player_id FROM my_picks WHERE season = ? AND week = ?",
                    (season, wk),
                )
            }
            warnings = state.record_picks(conn, season, wk, entries, now=now)
            for e in entries:
                prior = replaced.get(e["slot"])
                if prior is not None and prior != e["player_id"]:
                    capture.record_action(
                        conn,
                        season,
                        wk,
                        e["slot"],
                        "correction",
                        prior,
                        dict(replaced_by=e["player_id"]),
                        decision_id=decision,
                    )
                capture.record_action(
                    conn,
                    season,
                    wk,
                    e["slot"],
                    "submitted",
                    e["player_id"],
                    dict(player_name=e["player_name"], team=e.get("team"), replaced=prior),
                    decision_id=decision,
                )
    except (state.PickError, ValueError) as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    for warning in warnings:
        console.print(f"[yellow]Warning: {warning}[/yellow]")
    for e in entries:
        console.print(f"[green]Week {wk} {e['slot']}: {e['player_name']} ({e['team']})[/green]")


def unrecord(week: int, slot: str, season: int = SeasonOpt, db_path: Path | None = DbOpt):
    """Remove a recorded pick (slot: QB, RB, FLEX)."""
    conn = _conn(db_path)
    _week(conn, season, week)
    try:
        with db.transaction(conn):
            removed = conn.execute(
                "SELECT player_id FROM my_picks WHERE season = ? AND week = ? AND slot = ?",
                (season, week, slot.upper()),
            ).fetchone()
            ok = state.remove_pick(conn, season, week, slot.upper())
            if ok:
                capture.record_action(
                    conn,
                    season,
                    week,
                    slot.upper(),
                    "correction",
                    removed["player_id"],
                    dict(removed=True),
                )
    except state.PickError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    console.print("removed" if ok else "[yellow]nothing to remove[/yellow]")


def picks(season: int = SeasonOpt, db_path: Path | None = DbOpt):
    """Show recorded picks."""
    conn = _conn(db_path)
    _render_scores(conn, season)


def _pick_results(
    conn,
    season: int,
    week: int | None = None,
    recompute: bool = False,
    preserve_existing: bool = False,
):
    try:
        return scoring.pick_results(conn, season, week, recompute, preserve_existing)
    except state.PickError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e


def _render_scores(conn, season: int, week: int | None = None):
    all_picks = _pick_results(conn, season)
    shown = all_picks if week is None else all_picks[all_picks.week == week]
    t = Table("Week", "Slot", "Player", "TDs / pending reason", "Recorded")
    for r in shown.itertuples():
        value = str(int(r.tds)) if pd.notna(r.tds) else ""
        if r.pending_reason:
            value = f"pending: {r.pending_reason}" + (f" (last scored {value})" if value else "")
        recorded = r.recorded_at
        if "+" not in recorded and not recorded.endswith("Z"):
            recorded += " (legacy; timezone unknown)"
        t.add_row(str(r.week), r.slot, r.player_name, value, recorded)
    console.print(t)

    def subtotal(label, frame):
        pending = int(frame.pending_reason.ne("").sum())
        suffix = " (incomplete)" if pending else ""
        console.print(f"{label} subtotal: {int(frame.tds.sum())} TDs{suffix}; {pending} pending")

    for wk, rows in shown.groupby("week"):
        subtotal(f"Week {wk}", rows)
    subtotal(f"Season {season}", all_picks)


def score(
    week: int | None = typer.Option(None, "--week", "-w", help="Week (default: all picks)"),
    season: int = SeasonOpt,
    refresh: bool = typer.Option(False, "--refresh", help="Refresh all feeds before scoring"),
    db_path: Path | None = DbOpt,
):
    """Recompute recorded picks with complete game results; defaults to local data."""
    conn = _conn(db_path)
    if week is not None:
        _week(conn, season, week)
    failed = False
    if refresh:
        result = ingest.refresh(conn, season, log=console.print)
        failed = bool(result.failures)
    if failed:
        console.print(
            "[yellow]Refresh partially failed; stored scores retained. "
            "Scoring completed unscored picks from available local results. "
            "Run pool score without --refresh to recompute existing scores.[/yellow]"
        )
    _pick_results(conn, season, week, recompute=True, preserve_existing=failed)
    _render_scores(conn, season, week)
    if failed:
        raise typer.Exit(1)


def status(week: int | None = WeekOpt, season: int = SeasonOpt, db_path: Path | None = DbOpt):
    """Show feed attempts, successful imports, coverage and model fallbacks."""
    conn = _conn(db_path)
    now = state.eastern_now()
    # Status remains useful even with no schedule (unlike advice or recording).
    has_schedule = projections.available_weeks(conn, season)
    wk = _week(conn, season, week, now) if has_schedule else (week if week is not None else 1)
    console.print(f"[bold]Data status — {season}, week {wk}[/bold]")
    _status(conn, season, wk, now, detail=True)
    archived = snapshots.coverage(conn)
    archived = archived[archived.season == season]
    console.print("Snapshot coverage (UTC):")
    console.print(archived.to_string(index=False) if len(archived) else "No archived observations.")
