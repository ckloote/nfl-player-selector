"""`pool week`: this week's decisions on one screen, and the line to paste once the picks
are in.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import typer
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from .. import config, freshness, ingest, predictions, state, weekly
from ..recommend import Candidate, SlotAdvice, main_slate_start
from .advice import Decided, _decide, _divergence
from .common import DbOpt, SeasonOpt, WeekOpt, _conn, _short, console, print_command


@dataclass
class _WeekReads:
    """What the weekly screen reads beyond the decision, from the same snapshot."""

    results: dict  # slot -> (recorded player's name, score so far)
    previous: dict  # slot -> weekly.Previous
    players: pd.DataFrame  # the pool `record` resolves typed names against
    reported: bool  # this week's pool report has arrived


def week_command(
    week: int | None = WeekOpt,
    season: int = SeasonOpt,
    db_path: Path | None = DbOpt,
    refresh_first: bool | None = typer.Option(
        None,
        "--refresh/--no-refresh",
        help="Refresh the data first (default: only when it is stale)",
    ),
):
    """This week at a glance: what to submit, what can wait, and what is done."""
    conn = _conn(db_path)
    refreshed = None
    if refresh_first or (refresh_first is None and weekly.needs_refresh(conn, season)):
        # Per-feed progress is `pool refresh`'s to print. Here only the outcome matters,
        # and a failure is reported below without stopping the advice.
        refreshed = ingest.refresh(conn, season, log=lambda line: None).failures

    def reads(conn, wk, decision_id):
        arrivals, _ = predictions.report_arrivals(conn, season)
        return _WeekReads(
            weekly.locked_results(conn, season, wk),
            weekly.previous_advice(conn, season, wk, exclude=decision_id),
            state.historical_pool(conn, season, wk),
            wk in arrivals,
        )

    d = _decide(conn, season, week, capture_when="open", also=reads)
    # After the snapshot closes: the archive refuses to write inside a transaction, and
    # the prediction is made on the very frame the advice above was derived from.
    try:
        predicted = weekly.save_predictions(conn, season, d.week, d.proj, d.decided)
    except Exception as exc:  # the advice above must print whatever fails here
        predicted = ("failed", f"{type(exc).__name__}: {exc}")
    _render_week(d, season, refreshed, db_path, predicted)


def _age(rows, now: datetime) -> str:
    """How old the oldest feed is, which is the one a stale warning would name."""
    instant = freshness.utc_now(now)
    stamps = [datetime.fromisoformat(r["last_success"]) for r in rows if r["last_success"]]
    if not stamps:
        return "never refreshed"
    minutes = max(0.0, (instant - min(stamps)).total_seconds() / 60)
    if minutes < 1:
        return "data under a minute old"
    if minutes < 90:
        return f"data {minutes:.0f} min old"
    if minutes < 48 * 60:
        return f"data {minutes / 60:.0f} h old"
    return f"data {minutes / 1440:.0f} days old"


def _slots(views) -> str:
    return ", ".join(v.slot for v in views)


def _render_week(d: Decided, season: int, refreshed, db_path, predicted=None) -> None:
    reads: _WeekReads = d.extra
    pool, wk = d.pool, d.week
    views = weekly.slot_views(d.advice, reads.results, reads.previous)
    header = "refreshed just now" if refreshed is not None else _age(d.freshness_rows, d.now)
    console.print(f"[bold]Week {wk} · {season}[/bold] · {header} · times ET")
    if refreshed:
        console.print(
            f"Refresh failed for {'; '.join(refreshed)}. Using the data from before it.",
            style="yellow",
            markup=False,
        )
    for warning in d.freshness_warnings:
        console.print(f"Warning: {warning}", style="yellow", markup=False)
    _print_week_pool(pool, season, d.uncertain, wk, db_path)
    console.print()
    _print_week_slots(d.advice, views, pool)
    console.print()
    _print_next(d, season, views, reads, db_path)
    if predicted:
        _print_predicted(d.week, *predicted, season=season, db_path=db_path)
    console.print("Full detail:", style="dim")
    print_command(weekly.command(["recommend", "--week", str(wk)], season=season, db_path=db_path))


def _print_predicted(wk: int, what: str, kickoff, *, season: int, db_path) -> None:
    """The save status of the rival rankings and implied distribution."""
    if what == "failed":
        console.print(
            f"Could not save rival rankings and distribution for week {wk}: {kickoff}. "
            "Save them by hand before first kickoff:",
            style="yellow",
            markup=False,
        )
        print_command(
            weekly.command(
                ["research", "predict", "record", "--week", str(wk)],
                season=season,
                db_path=db_path,
            )
        )
        return
    when = _short(state.eastern_now(kickoff))
    console.print(
        {
            "saved": f"Rival rankings and distribution for week {wk} saved; they count until first "
            f"kickoff, {when}.",
            "unchanged": f"Rival rankings and distribution for week {wk} unchanged since "
            f"the last save; they count until first kickoff, {when}.",
            "on record": f"Rival predictions for week {wk} are on record from before the "
            "week could be seen.",
            "missed": f"No rival prediction was saved for week {wk} before it could be seen, "
            "so this week will not be scored.",
        }[what],
        style="yellow" if what == "missed" else "dim",
        markup=False,
    )


def _week_rows(a: SlotAdvice, view, pool, shares: bool):
    """One slot's rows, as plain cells with a style each, and the notes printed under them."""
    rows, notes = [], []
    blank = [""] * (3 + shares)
    if view.status == "locked":
        text = Text.assemble(("locked ", "green"), view.locked)
        if view.result:
            text.append(f" · {view.result}")
        return [([view.slot, text, *blank], None)], notes
    if view.status == "none":
        return [([view.slot, Text("no playable candidate", "red"), *blank], None)], notes
    c = view.player
    share = {s.player_id: s for s in a.shares}
    label = ("PICK ", "bold green") if view.status == "pick" else ("HOLD ", "yellow")
    cells = [view.slot, Text.assemble(label, f"{c.player_name} ({c.team}) {_versus(c)}")]
    cells += [f"{c.lam:.2f}", ""]
    if shares:
        own = share.get(c.player_id)
        cells.append(f"{own.share:.1%}" if own else "—")
    rows.append(([*cells, _short(c.deadline)], None))
    for alt in view.alternatives:
        cells = ["", f"  {alt.player_name} ({alt.team})", f"{alt.lam:.2f}", f"-{alt.cost:.2f}"]
        if shares:
            own = share.get(alt.player_id)
            cells.append(f"{own.share:.1%}" if own else "—")
        rows.append(([*cells, _short(alt.deadline)], "dim"))
    if view.status == "hold":
        h = view.wait_for
        notes.append(
            (
                f"Hold: {c.player_name} plays {c.kickoff:%A} and is only {h.cost:.2f} TD ahead "
                f"of {h.player_name} ({h.team}, {_short(h.deadline)}). Wait for injury news, "
                f"and take him before {_short(c.deadline)} only if the gap grows.",
                "yellow",
            )
        )
    elif c.early and a.hold_alternative is not None:
        h = a.hold_alternative
        notes.append(
            (
                f"Plays {c.kickoff:%A}: commit. He is {h.cost:.2f} TD ahead of the best later "
                f"option ({h.player_name}), more than the {config.INFO_PREMIUM_TD:.2f} TD that "
                "waiting for news is worth.",
                "cyan",
            )
        )
    if view.was is not None:
        notes.append(
            (
                f"Changed since the {_short(view.was.decided_at)} run, which said "
                f"{view.was.player_name}.",
                "cyan",
            )
        )
    if pool and a.divergent:
        reason = _divergence(a, pool)
        notes.append(
            (f"Share: {reason}", "magenta")
            if reason
            else (
                f"Pot share prefers {a.best_share.player_name} and this tool cannot say why. "
                "Treat that as a bug report, not as advice, and pick on expected TDs.",
                "red",
            )
        )
    return rows, notes


def _print_week_slots(advice, views, pool) -> None:
    """Each slot as its own table, all sharing one set of column widths.

    One table would need its notes as rows, where the no-wrap name column either cut them
    off or forced every name to wrap with them. Separate tables at the same widths keep the
    columns aligned and let a note wrap as long as it needs to.
    """
    shares = bool(pool.ready and any(a.shares for a in advice))
    blocks = [_week_rows(a, v, pool, shares) for a, v in zip(advice, views, strict=True)]
    names = ["Slot", "Player", "xTD", "Cost", *(["Share"] if shares else []), "Deadline"]
    widths = [len(name) for name in names]
    for rows, _ in blocks:
        for cells, _ in rows:
            for i, cell in enumerate(cells):
                widths[i] = max(widths[i], len(cell.plain if isinstance(cell, Text) else cell))
    indent = widths[0] + 3
    for index, (rows, notes) in enumerate(blocks):
        table = Table(box=None, padding=(0, 1), show_header=index == 0, header_style="dim")
        for name, size in zip(names, widths, strict=True):
            right = name in ("xTD", "Cost", "Share")
            table.add_column(name, width=size, no_wrap=True, justify="right" if right else "left")
        for cells, style in rows:
            table.add_row(*cells, style=style)
        console.print(table)
        for text, style in notes:
            console.print(Padding(Text(text, style), (0, 0, 0, indent)))


def _versus(c: Candidate) -> str:
    return f"{'vs' if c.home else '@'} {c.opponent}"


def _print_week_pool(pool, season: int, uncertain, wk: int, db_path) -> None:
    """The pot share in a line or two: its state, and what fixes it when it is withheld."""
    if pool.withheld:
        console.print(
            "[yellow]Pot share (experimental) withheld; the picks below are the "
            "expected-TD advice.[/yellow]"
        )
        for reason in pool.withheld:
            console.print(Padding(Text(reason, "yellow"), (0, 0, 0, 2)))
    elif not pool:
        console.print(
            f"[dim]No rivals on record for {season}, so no pot-share view; "
            "import a pool report to turn it on.[/dim]"
        )
        _print_report_import(season, wk, db_path)
    else:
        leader = max(pool.rivals, key=lambda r: r.season_tds)
        console.print(
            f"[dim]Pot share (experimental) vs {len(pool.rivals)} rivals: you {pool.my_tds} "
            f"TD, best rival {leader.display_name} {leader.season_tds}.[/dim]"
        )
        for name, week, slot, reported in uncertain:
            console.print(
                f"Simulated as unknown: {name} {slot} week {week} ({reported!r} did not "
                "resolve). Re-import that report once it resolves to use the pick.",
                style="yellow",
                markup=False,
            )


def _print_report_import(season: int, wk: int, db_path) -> None:
    console.print("Replace the quoted '<file>' placeholder with the report filename.", markup=False)
    print_command(
        weekly.command(
            ["report", "import", "<file>", "--week", str(wk)], season=season, db_path=db_path
        )
    )


def _print_next(d: Decided, season: int, views, reads: _WeekReads, db_path) -> None:
    """What to do next, and the line to paste once it is done."""
    wk = d.week
    now_views, later = weekly.due(views)
    holds = [v for v in views if v.status == "hold"]
    if all(v.status == "locked" for v in views):
        console.print(f"Week {wk} is fully recorded.")
        if not reads.reported:
            console.print(f"When the pool's week {wk} report arrives:")
            _print_report_import(season, wk, db_path)
    if now_views:
        first = min(now_views, key=lambda v: v.player.deadline).player
        slate = main_slate_start(d.proj, wk)
        if (
            weekly.decision_day(first) == weekly.MAIN_SLATE
            and slate is not None
            and d.now.date() < slate.date()
        ):
            console.print(f"Nothing needs submitting before {slate:%A}.")
        # The first deadline, not a shared one: the slots due together can close hours or a
        # day apart -- Sunday's with Monday night's -- and each keeps its own above.
        console.print(
            f"Submit next: {_slots(now_views)} (first deadline {_short(first.deadline)}). "
            "Once submitted, record with:"
        )
        command, notes = weekly.record_command(
            season,
            wk,
            now_views,
            reads.players,
            d.decision_id,
            str(db_path) if db_path is not None else None,
        )
        if command:
            print_command(command)
        for note in notes:
            console.print(note, style="yellow", markup=False)
    waiting = later + holds
    if waiting:
        # A hold is decided by its own early deadline, not its fallback's: waiting for news
        # means looking again before the early player is gone, in case the gap has grown.
        until = min(v.player.deadline for v in waiting)
        console.print(
            f"Waiting: {_slots(waiting)}. Run again before {_short(until)}:",
            markup=False,
        )
        print_command(weekly.command(["week", "--week", str(wk)], season=season, db_path=db_path))
