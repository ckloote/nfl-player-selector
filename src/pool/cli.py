"""`pool` command-line interface."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import config, db, ingest, projections, state
from .optimizer import plan_slot
from .recommend import Candidate, SlotAdvice, advise_week

app = typer.Typer(help="NFL touchdown pool decision support.", no_args_is_help=True)
console = Console()

SeasonOpt = typer.Option(config.DEFAULT_SEASON, "--season", "-s", help="Season year")
DbOpt = typer.Option(None, "--db", help="SQLite path (default data/pool.db)")
WeekOpt = typer.Option(None, "--week", "-w", help="Week (default: current)")


def _conn(path: Path | None):
    return db.connect(path)


def _week(conn, season: int, week: int | None) -> int:
    return week if week is not None else state.current_week(conn, season)


def _projections(conn, season: int, wk: int):
    """Projections from `wk` onward, or a message saying why there are none.

    The week is checked against the loaded schedule first: an out-of-range week
    still yields rows when it is *below* the season (everything is "onward"),
    so emptiness alone doesn't identify the problem.
    """
    lo, hi = conn.execute(
        "SELECT MIN(week), MAX(week) FROM games WHERE season = ? AND game_type = 'REG'", (season,)
    ).fetchone()
    if lo is None:
        msg = f"No {season} schedule loaded; run `pool refresh` first."
    elif not lo <= wk <= hi:
        msg = f"Week {wk} is outside the {season} season (weeks {lo}-{hi})."
    else:
        proj = projections.projections_for(conn, season, from_week=wk)
        if len(proj):
            return proj
        msg = f"No projections for week {wk}; run `pool refresh` first."
    console.print(f"[red]{msg}[/red]")
    raise typer.Exit(1)


def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%a %m/%d %I:%M%p ET").replace(" 0", " ")


# --- commands ---------------------------------------------------------------
@app.command()
def refresh(season: int = SeasonOpt, db_path: Path | None = DbOpt):
    """Pull latest stats, schedule, rosters, injuries from nflverse."""
    conn = _conn(db_path)
    console.print(f"Refreshing {season} (prior season {season - 1})...")
    counts = ingest.refresh(conn, season, log=console.print)
    for k, v in counts.items():
        console.print(f"  {k}: {v} rows")


@app.command()
def recommend(week: int | None = WeekOpt, season: int = SeasonOpt, db_path: Path | None = DbOpt):
    """Recommend picks for every open slot this week."""
    conn = _conn(db_path)
    wk = _week(conn, season, week)
    proj = _projections(conn, season, wk)
    advice = advise_week(proj, wk, state.used_ids(conn, season), state.locked_by_slot(conn, season))
    console.print(
        f"[bold]Week {wk} — {season}[/bold]  (data refreshed {db.get_meta(conn, 'last_refresh')})"
    )
    for a in advice:
        _render_slot(a)


def _render_slot(a: SlotAdvice) -> None:
    console.rule(f"[bold]{a.slot}[/bold]")
    if a.locked_player:
        console.print(f"  Locked: [green]{a.locked_player}[/green]")
        return
    if a.recommended is None:
        console.print("  [red]No playable candidates.[/red]")
        return
    r = a.recommended
    console.print(
        f"  [bold green]PICK: {r.player_name}[/bold green] ({r.team} {r.position}) — {_matchup(r)}"
    )
    console.print(f"  Expected TDs {r.lam:.2f} · deadline {_fmt_dt(r.deadline)}")
    if r.early:
        if a.hold and a.hold_alternative:
            h = a.hold_alternative
            console.print(
                f"  [yellow]HOLD:[/yellow] {r.player_name} plays early and is worth only "
                f"{h.cost:.2f} TD more than {h.player_name} ({_matchup(h)}). "
                f"Wait for injury news unless the gap grows."
            )
        elif a.hold_alternative:
            h = a.hold_alternative
            console.print(
                f"  [cyan]COMMIT:[/cyan] plays early, but the edge over {h.player_name} "
                f"({h.cost:.2f} TD) beats the information premium ({config.INFO_PREMIUM_TD:.2f})."
            )
    t = Table(show_header=True, header_style="dim", box=None, padding=(0, 1))
    for col in ("Alternative", "Matchup", "xTD", "Season cost", "Plan uses in", "Deadline", "Note"):
        t.add_column(col)
    for c in a.alternatives:
        t.add_row(
            f"{c.player_name} ({c.team})",
            _matchup(c),
            f"{c.lam:.2f}",
            f"-{c.cost:.2f}",
            f"wk {c.planned_week}" if c.planned_week else "—",
            _fmt_dt(c.deadline),
            _note(c),
        )
    console.print(t)


def _matchup(c: Candidate) -> str:
    ha = "vs" if c.home else "@"
    return f"{ha} {c.opponent}, Vegas x{c.vegas_mult:.2f}, opp-D x{c.def_mult:.2f}"


def _note(c: Candidate) -> str:
    notes = []
    if c.early:
        notes.append("early game")
    if c.report_status:
        notes.append(c.report_status)
    return ", ".join(notes)


@app.command()
def record(
    week: int | None = WeekOpt,
    qb: str | None = typer.Option(None, help="QB pick (name)"),
    rb: str | None = typer.Option(None, help="RB pick (name)"),
    flex: str | None = typer.Option(None, help="WR/TE pick (name)"),
    season: int = SeasonOpt,
    db_path: Path | None = DbOpt,
):
    """Lock one or more slots for a week. Slots can be recorded separately."""
    conn = _conn(db_path)
    wk = _week(conn, season, week)
    frames = projections.load_frames(conn, season)
    pool = projections.player_pool(frames.rosters, frames.pw_prior, frames.pw_cur)
    given = {"QB": qb, "RB": rb, "FLEX": flex}
    if not any(given.values()):
        console.print("[red]Give at least one of --qb, --rb, --flex.[/red]")
        raise typer.Exit(1)
    for slot, name in given.items():
        if not name:
            continue
        matches = state.find_player(pool, name, config.SLOTS[slot])
        if len(matches) != 1:
            console.print(
                f"[red]{slot}: {'no match' if not len(matches) else 'ambiguous'} for {name!r}[/red]"
            )
            for _, m in matches.head(8).iterrows():
                console.print(f"    {m.player_name} ({m.team} {m.position})")
            raise typer.Exit(1)
        m = matches.iloc[0]
        try:
            state.record_pick(conn, season, wk, slot, m.player_id, m.player_name, m.position)
        except state.PickError as e:
            console.print(f"[red]{slot}: {e}[/red]")
            raise typer.Exit(1) from e
        console.print(f"[green]Week {wk} {slot}: {m.player_name} ({m.team})[/green]")


@app.command()
def unrecord(week: int, slot: str, season: int = SeasonOpt, db_path: Path | None = DbOpt):
    """Remove a recorded pick (slot: QB, RB, FLEX)."""
    conn = _conn(db_path)
    ok = state.remove_pick(conn, season, week, slot.upper())
    console.print("removed" if ok else "[yellow]nothing to remove[/yellow]")


@app.command()
def picks(season: int = SeasonOpt, db_path: Path | None = DbOpt):
    """Show recorded picks."""
    conn = _conn(db_path)
    df = state.picks(conn, season)
    t = Table("Week", "Slot", "Player", "TDs", "Recorded")
    for _, r in df.iterrows():
        t.add_row(
            str(r.week), r.slot, r.player_name, "" if r.tds is None else str(r.tds), r.recorded_at
        )
    console.print(t)


@app.command()
def plan(week: int | None = WeekOpt, season: int = SeasonOpt, db_path: Path | None = DbOpt):
    """Show the current rest-of-season assignment per slot."""
    conn = _conn(db_path)
    wk = _week(conn, season, week)
    proj = _projections(conn, season, wk)
    used = state.used_ids(conn, season)
    locked = state.locked_by_slot(conn, season)
    names = proj.drop_duplicates("player_id").set_index("player_id").player_name
    plans = {slot: plan_slot(proj, slot, wk, used, locked[slot]) for slot in config.SLOTS}
    t = Table("Week", *config.SLOTS, title=f"Remaining-season plan from week {wk}")
    for w in range(wk, int(proj.week.max()) + 1):
        cells = []
        for slot, p in plans.items():
            if w in locked[slot]:
                cells.append(f"[green]{names.get(locked[slot][w], '?')} (locked)[/green]")
            elif (pick := p.pick_for(w)) is not None:
                cells.append(f"{pick.player_name} {p.lam(p.assignment[w], w):.2f}")
            else:
                cells.append("[red]—[/red]")
        t.add_row(str(w), *cells)
    console.print(t)
    totals = ", ".join(f"{s}: {p.total:.1f}" for s, p in plans.items())
    console.print(f"Projected remaining TDs (discounted) — {totals}")


@app.command()
def players(
    pos: str = typer.Option(..., "--pos", "-p", help="QB, RB, FLEX (WR/TE)"),
    week: int | None = WeekOpt,
    top: int = typer.Option(25, help="Rows to show"),
    season: int = SeasonOpt,
    db_path: Path | None = DbOpt,
):
    """Projected TDs for a slot in a given week."""
    conn = _conn(db_path)
    wk = _week(conn, season, week)
    proj = _projections(conn, season, wk)
    sub = proj[(proj.week == wk) & (proj.slot == pos.upper())].head(top)
    t = Table(
        "Player",
        "Team",
        "Opp",
        "Kickoff",
        "Base",
        "Opp-D",
        "Vegas",
        "Role",
        "Avail",
        "xTD",
        title=f"{pos.upper()} week {wk}",
    )
    for _, r in sub.iterrows():
        t.add_row(
            r.player_name,
            r.team,
            ("vs " if r.home else "@ ") + r.opponent,
            _fmt_dt(datetime.fromisoformat(r.kickoff)),
            f"{r.base_rate:.2f}",
            f"{r.def_mult:.2f}",
            f"{r.vegas_mult:.2f}",
            f"{r.role_mult:.2f}",
            f"{r.avail_mult:.2f}",
            f"{r.lam:.2f}",
        )
    console.print(t)


if __name__ == "__main__":
    app()
