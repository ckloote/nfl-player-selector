"""What every `pool` command shares: the console, the common options, and the helpers
that open the database, resolve the week and build its projections.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .. import config, db, freshness, projections, state

console = Console()

SeasonOpt = typer.Option(config.DEFAULT_SEASON, "--season", "-s", help="Season year")
DbOpt = typer.Option(None, "--db", help="SQLite path (default data/pool.db)")
WeekOpt = typer.Option(None, "--week", "-w", help="Week (default: current)")


def print_command(command: str) -> None:
    """Keep shell syntax literal and copyable even on a narrow terminal."""
    console.print(f"  {command}", soft_wrap=True, markup=False, highlight=False)


def _conn(path: Path | None):
    return db.connect(path)


def _week(conn, season: int, week: int | None, now: datetime | None = None) -> int:
    try:
        wk = week if week is not None else state.current_week(conn, season, now)
        return state.validate_week(conn, season, wk)
    except state.PickError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


def _projections(conn, season: int, wk: int):
    _week(conn, season, wk)
    history = conn.execute(
        "SELECT COUNT(*) FROM player_weeks WHERE season IN (?, ?)", (season - 1, season)
    ).fetchone()[0]
    if not history:
        console.print(
            f"[red]No projections for week {wk}: player history is missing; "
            f"run `pool refresh --season {season}` first.[/red]"
        )
        raise typer.Exit(1)
    proj = projections.projections_for(conn, season, from_week=wk)
    if len(proj):
        return proj
    console.print(f"[red]No projections for week {wk}; check rosters and run `pool refresh`.[/red]")
    raise typer.Exit(1)


def _status(conn, season: int, week: int, now: datetime, detail: bool = False):
    _print_freshness(*freshness.report(conn, season, week, now), detail=detail)


def _print_freshness(rows, warnings, detail: bool = False):
    """Render a freshness report. Split from reading it so a caller holding a snapshot
    can gather inside it and print outside, rather than re-reading after it is gone."""
    if detail:
        t = Table(
            "Feed",
            "Outcome",
            "Last attempt (UTC)",
            "Last success (UTC)",
            "Age",
            "Coverage",
            "Source timestamp",
        )
        for row in rows:
            cov = row["coverage"]
            coverage = f"{cov['rows']} rows"
            if "weeks" in cov:
                coverage += f"; weeks {','.join(map(str, cov['weeks'])) or 'none'}"
            if "complete_games" in cov:
                coverage += f"; {cov['complete_games']}/{cov['scheduled_games']} games complete"
            if cov.get("as_of"):
                coverage += f"; as of {cov['as_of']}"
            t.add_row(
                row["feed"],
                row["outcome"],
                row["last_attempt"] or "unknown",
                row["last_success"] or "unknown",
                row["age"],
                coverage,
                row["source_timestamp"] or "unavailable",
            )
        console.print(t)
    else:
        summary = "; ".join(f"{r['feed']}: {r['age']} ({r['outcome']})" for r in rows)
        console.print(f"[dim]Local data — {summary}[/dim]")
    for warning in warnings:
        console.print(f"[yellow]Warning: {warning}[/yellow]")


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%a %m/%d %I:%M%p ET").replace(" 0", " ")


def _short(dt: datetime) -> str:
    """A deadline inside one week: the day names it, so the date would only take room."""
    return dt.strftime("%a %I:%M%p").replace(" 0", " ")
