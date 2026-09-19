"""`pool backup` and `pool export`: copies of what only this database holds."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Annotated

import typer

from .. import config, keeping, state
from .common import DbOpt, SeasonOpt, _conn, console

export_app = typer.Typer(help="Write your records out as plain files.", no_args_is_help=True)


def backup(
    to: Annotated[
        Path | None,
        typer.Option(
            "--to", help="Where to write the copy (default: backups/ beside the database)"
        ),
    ] = None,
    db_path: Path | None = DbOpt,
):
    """Copy the database: your picks, the pool's reports and the decisions you saved."""
    source = db_path or config.DB_PATH
    target = to or source.parent / "backups" / f"pool-{state.eastern_now():%Y%m%d-%H%M}.db"
    try:
        keeping.backup(source, target)
    except (OSError, RuntimeError, sqlite3.Error) as exc:
        console.print(str(exc), style="red", markup=False)
        raise typer.Exit(1) from exc
    size = target.stat().st_size / 1_000_000
    console.print(f"Backed up {source} to {target} ({size:.1f} MB).", markup=False)
    console.print(
        f"To restore it, copy it over {source} while no pool command is running.", markup=False
    )


@export_app.command("picks")
def export_picks(
    csv_path: Annotated[
        Path | None, typer.Option("--csv", help="File to write (default: print it)")
    ] = None,
    season: int = SeasonOpt,
    db_path: Path | None = DbOpt,
):
    """Your picks for the season and what each has scored, as CSV."""
    conn = _conn(db_path)
    try:
        frame = keeping.picks(conn, season)
    except state.PickError as exc:
        console.print(str(exc), style="red", markup=False)
        raise typer.Exit(1) from exc
    finally:
        conn.close()
    text = frame.to_csv(index=False)
    if csv_path is None:
        # Straight to stdout rather than through Rich, which would wrap long rows.
        typer.echo(text, nl=False)
        return
    csv_path.write_text(text, encoding="utf-8")
    console.print(f"Wrote {len(frame)} picks for {season} to {csv_path}.", markup=False)
