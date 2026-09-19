"""`pool` command-line interface."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from . import (
    capture,
    config,
    db,
    entrants,
    freshness,
    ingest,
    predictions,
    projections,
    scoring,
    simulate,
    snapshots,
    state,
    weekly,
)
from . import standings as st
from .optimizer import plan_slot
from .recommend import Candidate, SlotAdvice, advise_week, main_slate_start, pot_shares
from .research.cli import app as research_app
from .rivals import PoolState

app = typer.Typer(help="NFL touchdown pool decision support.", no_args_is_help=True)
report_app = typer.Typer(
    help="Archive and read official weekly pool reports.", no_args_is_help=True
)
app.add_typer(report_app, name="report")
app.add_typer(research_app, name="research")
console = Console()

SeasonOpt = typer.Option(config.DEFAULT_SEASON, "--season", "-s", help="Season year")
DbOpt = typer.Option(None, "--db", help="SQLite path (default data/pool.db)")
WeekOpt = typer.Option(None, "--week", "-w", help="Week (default: current)")


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


# --- commands ---------------------------------------------------------------
@report_app.command("import")
def report_import(
    path: Annotated[Path, typer.Argument(help="Delivered report file")],
    week: int | None = typer.Option(None, "--week", "-w", help="Week (default: report column)"),
    season: int = SeasonOpt,
    fmt: str = typer.Option("csv", "--format", help="Report format (csv)"),
    me: str | None = typer.Option(None, "--me", help="Your entrant display name"),
    allow_roster_change: bool = typer.Option(
        False, "--allow-roster-change", help="Acknowledge additions or removals of entrants"
    ),
    check: bool = typer.Option(
        False, "--check", help="Validate and report issues; archives and writes nothing"
    ),
    db_path: Path | None = DbOpt,
):
    """Archive the original bytes, then import a complete week of entrant picks."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        console.print(f"Cannot read report: {exc}", style="red", markup=False)
        raise typer.Exit(1) from exc
    conn = _conn(db_path)
    try:
        result = entrants.import_report(
            conn,
            season,
            week,
            raw,
            fmt=fmt,
            me=me,
            check=check,
            allow_roster_change=allow_roster_change,
            source=str(path),
        )
    finally:
        conn.close()
    if not check:
        console.print(
            f"Archived report as observation {result.observation_id}: {path}", markup=False
        )
    if result.parsed:
        action = "Imported" if result.written else "Parsed"
        console.print(
            f"{action} {season} week {result.week}: {result.entrants} entrants, "
            f"{result.picks} picks, {result.totals} reported totals."
        )
    if check:
        console.print("Check only: nothing archived, and no entrant, pick, or total rows written.")
    for warning in result.warnings:
        console.print(f"Warning: {warning}", style="yellow", markup=False)
    for item in result.unresolved:
        reason = "ambiguous" if item["candidates"] else "no match"
        console.print(
            f"Unresolved {item['entrant_id']} {item['slot']}: {item['player_name']!r} ({reason})",
            style="yellow",
            markup=False,
        )
        for candidate in item["candidates"]:
            console.print(
                f"  {candidate['player_name']} ({candidate['team']} {candidate['position']}; "
                f"{candidate['player_id']})",
                markup=False,
            )
    for item in result.inexact:
        console.print(
            f"Note: {item['entrant_id']} {item['slot']}: typed {item['player_name']!r}, "
            f"matched {item['matched']} ({item['team']} {item['position']}). "
            "Not an exact name match; confirm it is the intended player.",
            style="yellow",
            markup=False,
        )
    if result.added or result.removed:
        console.print(f"Entrant changes compared with week {result.previous_week}:")
        if result.added:
            console.print("  Added: " + ", ".join(result.added), markup=False)
        if result.removed:
            console.print("  Removed: " + ", ".join(result.removed), markup=False)
    for item in result.unrecorded:
        console.print(
            f"Note: no recorded pick for {item['slot']}; the report has "
            f"{item['reported']}. Run `pool record` if that is an omission.",
            style="yellow",
            markup=False,
        )
    for item in result.conflicts:
        console.print(
            f"my_picks mismatch for {item['slot']}: "
            f"recorded {item['recorded'] or '(missing)'}; "
            f"reported {item['reported'] or '(no pick)'}. "
            "my_picks was not changed.",
            style="red",
            markup=False,
        )
    for error in result.errors:
        console.print(error, style="red", markup=False)
    if not result.ok:
        raise typer.Exit(1)


def _reported_number(value) -> str:
    return str(int(value)) if pd.notna(value) else "—"


@report_app.command("list")
def report_list(season: int = SeasonOpt, db_path: Path | None = DbOpt):
    """List every archived import attempt, including failed imports."""
    conn = _conn(db_path)
    try:
        reports = entrants.reports(conn, season)
    finally:
        conn.close()
    if reports.empty:
        console.print(f"No archived pool reports for {season}.")
        return
    table = Table(
        "Week",
        "Observation",
        "Observed (UTC)",
        "Status",
        "Entrants",
        "Picks",
        "Unresolved",
        "Source",
        title=f"Archived pool reports — {season}",
    )
    for row in reports.itertuples():
        status = "imported" if row.written else "check" if row.check else "not imported"
        if row.needs_review:
            status += "; review needed"
        table.add_row(
            _reported_number(row.week),
            str(row.observation_id),
            row.observed_at,
            status,
            _reported_number(row.entrants) if row.parsed else "—",
            _reported_number(row.picks) if row.parsed else "—",
            _reported_number(row.unresolved) if row.parsed else "—",
            row.source,
        )
    console.print(table)


@app.command()
def standings(
    week: int | None = typer.Option(None, "--week", "-w", help="Week (default: latest imported)"),
    season: int = SeasonOpt,
    db_path: Path | None = DbOpt,
):
    """Show computed season standings beside the pool's reported weekly picks and totals."""
    conn = _conn(db_path)
    try:
        # Every other week-taking command validates first. Without this a typo renders a
        # full table for a week that cannot exist, reporting each entrant's three slots as
        # a missing report rather than saying there is no such week.
        if week is not None:
            _week(conn, season, week)
        result = st.board(conn, season, week=week)
    finally:
        conn.close()
    _render_standings(result)


def _standing_pick(pick: st.EntrantPick) -> str:
    if pick.status == "missing":
        return "—"
    if pick.player_name is None:
        return "(no pick)"
    return pick.player_name + (" (unresolved)" if pick.status == "unresolved" else "")


def _standing_total(total: st.Total) -> str:
    issues = []
    for count, label in (
        (total.pending, "pending"),
        (total.unresolved, "unresolved"),
        (total.missing, "missing"),
    ):
        if count:
            issues.append(f"{count} {label}")
    return str(total.tds) + (f" (incomplete; {', '.join(issues)})" if issues else "")


def _standing_lead(result: st.Board) -> str:
    if not result.leaders:
        return ""
    names = [r.display_name for r in result.leaders]
    name = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
    total = result.leaders[0].season_total.tds
    verb = "leads" if len(names) == 1 else "are tied for first"
    sentence = ("" if result.final else "As it stands ") + f"{name} {verb} with {total} TDs"
    if not result.final:
        issues = []
        for field, singular, plural in (
            ("pending", "pick pending", "picks pending"),
            ("unresolved", "unresolved name", "unresolved names"),
            ("missing", "missing pick", "missing picks"),
        ):
            count = sum(getattr(r.season_total, field) for r in result.rows)
            if count:
                issues.append(f"{count} {singular if count == 1 else plural}")
        sentence += ", with " + ", ".join(issues)
    sentence += "."
    if len(names) > 1:
        # Settled language needs the season to be over, not merely the ranked weeks to be
        # fully scored -- `final` is true after one clean week of eighteen, and the pot is
        # not shared until there are no weeks left to change it.
        if result.season_complete:
            sentence += (
                f" A tie for first splits the winnings: each takes 1/{len(names)} of the pot."
            )
        else:
            sentence += f" A tie at the end splits the winnings 1/{len(names)} each."
    return sentence


def _render_standings(result: st.Board) -> None:
    if not result.rows:
        console.print(f"No imported pool standings for {result.season}. Run pool report import.")
    else:
        ranking = (
            f"ranked on season totals through week {result.as_of}"
            if result.as_of is not None
            else "season unranked"
        )
        table = Table(
            "Rank",
            "Entrant",
            *config.SLOTS,
            "Week TDs",
            "Season TDs",
            "Used",
            "Reported week TDs",
            "Reported total TDs",
            "Reported rank",
            title=f"Computed standings — {result.season}, week {result.week} picks; {ranking}",
        )
        for row in result.rows:
            used = str(len(row.used.player_ids))
            if row.used.unknown:
                used += f" + {row.used.unknown} unknown"
            if not row.used.complete:
                used += " (incomplete)"
            table.add_row(
                str(row.rank) if row.rank is not None else "—",
                row.display_name + (" (me)" if row.is_me else ""),
                *(_standing_pick(p) for p in row.picks),
                _standing_total(row.week_total),
                _standing_total(row.season_total) if result.as_of is not None else "—",
                used,
                _reported_number(row.reported_week),
                _reported_number(row.reported_total),
                _reported_number(row.reported_rank),
            )
        console.print(table)
    if result.as_of is None:
        console.print("No consecutive completed weeks from week 1; season cannot be ranked yet.")
    elif result.rows:
        qualifier = " (provisional)" if not result.final else ""
        console.print(f"Season standings as of week {result.as_of}{qualifier}.")
        console.print(_standing_lead(result), markup=False)
    for row in result.rows:
        if row.season_total.incomplete:
            console.print(
                f"{row.display_name}: season total {_standing_total(row.season_total)} TDs.",
                markup=False,
            )
        if row.used.unknown:
            console.print(
                f"{row.display_name}: {row.used.unknown} unresolved names in the used pool; "
                "re-import with pool report import.",
                markup=False,
            )
        if row.used.missing_weeks:
            console.print(
                f"{row.display_name}: used pool missing reports for weeks "
                + ", ".join(map(str, row.used.missing_weeks))
                + ".",
                markup=False,
            )
        for pid, weeks in row.used.repeats.items():
            console.print(
                f"{row.display_name}: repeated player {pid} in weeks "
                + ", ".join(map(str, weeks))
                + "; counted once in Used.",
                markup=False,
            )
    if result.rows:
        console.print(
            f"Used includes all imported picks through week {result.used_through}. "
            f"Reported columns describe week {result.week}; — means not supplied."
        )
    noted = [(row, pick) for row in result.rows for pick in row.picks if pick.note]
    for row, pick in noted:
        console.print(
            f"{row.display_name} {pick.slot}: {pick.player_name} had no week {pick.week} game; "
            "scored 0. A bye or a team the schedule does not match.",
            markup=False,
        )
    if result.report_conflicts:
        console.print(
            "Reported picks differ from your recorded picks. The report is shown; "
            "use pool report import with --me to review the comparison."
        )
    if result.in_progress:
        table = Table(
            "Week",
            "Entrant",
            "Slot",
            "Player",
            "Source",
            "TDs / status",
            title="In progress — excluded from season ranks",
        )
        for pick in result.in_progress:
            value = str(pick.tds) if pick.status == "final" else f"{pick.status}: {pick.pending}"
            table.add_row(
                str(pick.week),
                pick.display_name,
                pick.slot,
                _standing_pick(pick),
                pick.source,
                value,
            )
        console.print(table)
    console.print("Totals are touchdown counts, not points.")
    if result.cache_stale:
        console.print(
            "Your recorded score cache is empty or stale for a finished game. "
            "Run pool score so pool picks and standings use the same current counts."
        )


@app.command()
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


@dataclass
class Decided:
    """One decision and everything printed about it, all read from a single snapshot."""

    week: int
    now: datetime
    decided: datetime
    proj: pd.DataFrame
    advice: list[SlotAdvice]
    locked: dict[str, dict[int, str]]
    pool: PoolState
    decision_id: str | None
    freshness_rows: list
    freshness_warnings: list
    picks: pd.DataFrame
    nudge: str | None
    uncertain: list
    extra: object = None

    @property
    def recorded_names(self) -> dict[str, str]:
        return self.picks.set_index("player_id").player_name.to_dict()


def _decide(conn, season: int, week: int | None, *, capture_when: str, also=None) -> Decided:
    """Advise every open slot, and capture the decision `always`, `never`, or only while a
    slot is still `open`. Shared by `recommend` and `week`, so the two cannot disagree
    about what was decided or when.

    `also(conn, week, decision_id)` reads anything else a caller prints, inside the same
    snapshot, for the reason the freshness report is read there.
    """
    # One snapshot over the whole decision, taken before the clock. The forecast reads
    # mutable feed tables and the capture then names the observations behind them by
    # asking for everything at or before this instant; a refresh landing between the
    # timestamp and the reads would feed the forecast data the reference excludes.
    # `snapshot=True` fixes what this connection sees first, because a SAVEPOINT on its
    # own does not -- so the clock below is now read against data already held. A
    # concurrent refresh waits, or fails loudly, rather than interleaving.
    with db.transaction(conn, snapshot=True):
        # One instant for the whole decision: the Eastern form compares against
        # kickoffs, the aware form identifies which feed observations were available.
        now = state.eastern_now()
        decided = state.decision_instant(now)
        wk = _week(conn, season, week, now)
        proj = _projections(conn, season, wk)
        used, locked = state.used_ids(conn, season), state.locked_by_slot(conn, season)
        # Read here, on this side of the closure, and handed across as plain data. With no
        # reports imported this is empty and `advise_week` gives exactly the advice it
        # always has -- the second objective is additive and never edits the first.
        pool = predictions.pool_state(conn, season, wk, at=decided)
        advice = advise_week(proj, wk, used, locked, now=now, pool=pool)
        decision_id = None
        if capture_when == "always" or (
            capture_when == "open" and any(a.locked_player is None for a in advice)
        ):
            decision_id = capture.record_decision(
                conn, season, wk, proj, advice, used, locked, decision_at=decided, pool=pool
            )
        # Read inside the snapshot too: a data-age line or a pick name drawn from a
        # feed the advice never saw would describe a decision that was not made.
        freshness_rows, freshness_warnings = freshness.report(conn, season, wk, now)
        picks = state.picks(conn, season)
        # Only with rivals to predict. Without them the command it names would refuse.
        nudge = _prediction_nudge(conn, season, wk, decided) if pool.rivals else None
        uncertain = predictions.unresolved_ahead(conn, season, wk, decided) if pool.ready else []
        extra = also(conn, wk, decision_id) if also else None
    return Decided(
        wk,
        now,
        decided,
        proj,
        advice,
        locked,
        pool,
        decision_id,
        freshness_rows,
        freshness_warnings,
        picks,
        nudge,
        uncertain,
        extra,
    )


@app.command()
def recommend(
    week: int | None = WeekOpt,
    season: int = SeasonOpt,
    db_path: Path | None = DbOpt,
    capture_decision: bool = typer.Option(
        True, "--capture/--no-capture", help="Record this decision's inputs, surface and advice"
    ),
    sensitivity: bool = typer.Option(
        False, "--sensitivity", help="Re-rank across the stress range of the fitted knobs"
    ),
):
    """Recommend picks for every open slot this week, in full detail."""
    conn = _conn(db_path)
    d = _decide(conn, season, week, capture_when="always" if capture_decision else "never")
    wk, pool = d.week, d.pool
    # On the same frame the advice was derived from, which the sweep reads in memory: it
    # touches no table, so it describes the stability of this decision and no later one.
    sweep = _sensitivity(d.proj, wk, d.advice, pool, d.locked) if sensitivity and pool else None
    console.print(f"[bold]Week {wk} — {season}[/bold]")
    _print_freshness(d.freshness_rows, d.freshness_warnings)
    _print_pool(pool, season, d.nudge, d.uncertain)
    if d.decision_id:
        # Printed so the submission can name it. With two decisions in a week the
        # fallback link -- the most recent advice for the slot -- is whichever happened
        # last, which is not the same thing as the one the pick came from.
        console.print(f"[dim]Captured decision {d.decision_id}[/dim]")
    recorded_names = d.recorded_names
    for a in d.advice:
        if a.locked_player in recorded_names:
            a.locked_player = recorded_names[a.locked_player]
        _render_slot(a, pool)
    if sweep is not None:
        _print_sensitivity(sweep)


@dataclass
class _WeekReads:
    """What the weekly screen reads beyond the decision, from the same snapshot."""

    results: dict  # slot -> (recorded player's name, score so far)
    previous: dict  # slot -> weekly.Previous
    players: pd.DataFrame  # the pool `record` resolves typed names against
    reported: bool  # this week's pool report has arrived


@app.command("week")
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


def _short(dt: datetime) -> str:
    """A deadline inside one week: the day names it, so the date would only take room."""
    return dt.strftime("%a %I:%M%p").replace(" 0", " ")


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
    _print_week_pool(pool, season, d.uncertain)
    console.print()
    _print_week_slots(d.advice, views, pool)
    console.print()
    _print_next(d, season, views, reads, db_path)
    if predicted:
        _print_predicted(d.week, *predicted)
    console.print(
        f"Full detail: {weekly.program()} recommend --week {d.week}"
        + (f" --season {season}" if season != config.DEFAULT_SEASON else ""),
        style="dim",
    )


def _print_predicted(wk: int, what: str, kickoff) -> None:
    """One line on the rival prediction, which only counts if saved before first kickoff."""
    if what == "failed":
        console.print(
            f"Could not save rival predictions for week {wk}: {kickoff}. Run pool research "
            f"predict record --week {wk} before first kickoff to save them by hand.",
            style="yellow",
            markup=False,
        )
        return
    when = _short(state.eastern_now(kickoff))
    console.print(
        {
            "saved": f"Rival predictions for week {wk} saved; they count until first "
            f"kickoff, {when}.",
            "unchanged": f"Rival predictions for week {wk} unchanged since the last save; "
            f"they count until first kickoff, {when}.",
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


def _print_week_pool(pool, season: int, uncertain) -> None:
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
            "`pool report import` turns it on.[/dim]"
        )
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


def _print_next(d: Decided, season: int, views, reads: _WeekReads, db_path) -> None:
    """What to do next, and the line to paste once it is done."""
    wk = d.week
    now_views, later = weekly.due(views)
    holds = [v for v in views if v.status == "hold"]
    prog = weekly.program()
    season_arg = f" --season {season}" if season != config.DEFAULT_SEASON else ""
    if all(v.status == "locked" for v in views):
        console.print(f"Week {wk} is fully recorded.")
        if not reads.reported:
            console.print(
                f"When the pool's week {wk} report arrives: {prog} report import "
                f"<file>{season_arg}",
                markup=False,
            )
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
            # One line whatever the width: a command Rich wrapped would paste as two.
            console.print(f"  {command}", soft_wrap=True, markup=False, highlight=False)
        for note in notes:
            console.print(note, style="yellow", markup=False)
    waiting = later + holds
    if waiting:
        # A hold is decided by its own early deadline, not its fallback's: waiting for news
        # means looking again before the early player is gone, in case the gap has grown.
        until = min(v.player.deadline for v in waiting)
        console.print(
            f"Waiting: {_slots(waiting)}. Run `{prog} week` again before {_short(until)}.",
            markup=False,
        )


def _render_slot(a: SlotAdvice, pool=None) -> None:
    console.rule(f"[bold]{a.slot}[/bold]")
    if a.locked_player:
        console.print(f"  Locked: [green]{a.locked_player}[/green]")
        return
    if a.recommended is None:
        console.print("  [red]No playable candidates.[/red]")
        for week in a.plan.weeks:
            pick = a.plan.pick_for(week)
            if week > a.week and pick is not None:
                console.print(f"  Future plan: week {week} — {pick.player_name}")
        return
    r = a.recommended
    console.print(
        f"  [bold green]PICK: {r.player_name}[/bold green] ({r.team} {r.position}) — {_matchup(r)}"
    )
    shares = {s.player_id: s for s in a.shares}
    own = shares.get(r.player_id)
    share = f" · pot share {own.share:.1%} (±{own.se:.1%} simulation noise)" if own else ""
    console.print(f"  Expected TDs {r.lam:.2f}{share} · deadline {_fmt_dt(r.deadline)}")
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
    # Two lines an alternative: the matchup under the name, any note under the deadline.
    # Nine columns in a row do not fit 80 once the pot share is on, and whatever narrowed
    # got cut -- a name, which is what you type into `record`, or a number. Widths are set
    # here rather than left to Rich, whose collapse shrinks the name column first whatever
    # its minimum: each column fits its longest entry, and the name column takes what is
    # left, never less than the longest name, so only the matchup under it wraps.
    rows = []
    for c in a.alternatives:
        cells = [f"{c.lam:.2f}", f"-{c.cost:.2f}"]
        if a.shares:
            cells += _share_cells(shares.get(c.player_id))
        cells += [f"wk {c.planned_week}" if c.planned_week else "—"]
        rows.append(
            (f"{c.player_name} ({c.team})", _matchup(c), cells, _short(c.deadline), _note(c))
        )
    headers = ["xTD", "Season cost", *(["Pot share", "vs EV"] if a.shares else []), "Plan uses in"]
    widths = [
        max([len(w) for w in header.split()] + [len(Text.from_markup(r[2][i]).plain) for r in rows])
        for i, header in enumerate(headers)
    ]
    deadline = max([len(r[3]) for r in rows] + [len(w) for r in rows for w in r[4].split()] + [8])
    room = console.width - sum(widths) - deadline - 2 * (len(headers) + 2)
    longest = max((len(r[0]) for r in rows), default=0)
    name = max(longest, min(max((len(r[1]) for r in rows), default=0), room))
    t = Table(show_header=True, header_style="dim", box=None, padding=(0, 1))
    t.add_column("Alternative", width=name)
    for header, width in zip(headers, widths, strict=True):
        t.add_column(header, width=width)
    t.add_column("Deadline", width=deadline)
    for who, matchup, cells, when, note in rows:
        label = Text(who)
        label.append(f"\n{matchup}", style="dim")
        stamp = Text(when)
        if note:
            stamp.append(f"\n{note}", style="dim")
        t.add_row(label, *cells, stamp)
    console.print(t)
    _share_notes(a, pool)


def _sensitivity(proj, wk: int, advice, pool, locked) -> pd.DataFrame:
    """Re-rank this week's candidates across the stress range of the two fitted knobs.

    `k_game` sets how much one game moves both its teams together and was fitted on a
    cross-team covariance; the rival-noise parameter is frankly provisional and says so
    where it is defined. Neither is measured well enough to be load-bearing, so the useful
    question is not what they are but whether the answer needs them.

    Only the pot-share view is re-run. The expected-TD advice does not read either knob, so
    re-deriving it would cost the same again to produce the same table.
    """
    rows = []
    for k in config.SENSITIVITY_K_GAME:
        for noise in config.SENSITIVITY_NOISE:
            trial = [copy.copy(a) for a in advice]
            with config.override(RIVAL_NOISE_TOP_N=noise):
                pot_shares(proj, wk, trial, pool, locked, params=simulate.Params(k_game=k))
            for a in trial:
                best = a.best_share
                if best is None:
                    continue
                rows.append(
                    dict(
                        slot=a.slot,
                        k_game=k,
                        noise=noise,
                        player_id=best.player_id,
                        player_name=best.player_name,
                        share=best.share,
                        tied=best.tied,
                    )
                )
    return pd.DataFrame(rows)


def _print_sensitivity(frame: pd.DataFrame) -> None:
    if frame.empty:
        console.print("[dim]Nothing to sweep: no pot-share view for this week.[/dim]")
        return
    table = Table(
        "Slot",
        "Where the two objectives part",
        "Cells",
        "Verdict",
        title="Sensitivity to k_game and rival noise",
    )
    for slot, block in frame.groupby("slot", sort=False):
        # Only a cell where the pot share *clears the noise band* is a cell where the two
        # objectives actually disagree. Reading the raw argmax instead would report a slot
        # whose candidates are all statistically tied as wildly parameter-sensitive, when
        # what is moving between settings is the coin, not the knob.
        parting = block[~block.tied]
        winners = sorted(set(parting.player_name))
        if parting.empty:
            where, verdict = "nowhere", "[green]agrees with expected TDs at every setting[/green]"
        elif len(winners) > 1:
            # Different settings name different players. Nothing here picks between them,
            # and quoting whichever the fitted cell happened to give would present a choice
            # of constant as a finding.
            where = ", ".join(winners)
            verdict = "[yellow]undetermined: follow expected TDs[/yellow]"
        elif len(parting) == len(block):
            where, verdict = winners[0], "[green]stable: separates at every setting[/green]"
        else:
            # One name wherever it separates, inside the noise band elsewhere. That is a
            # consistent answer of uncertain strength, which is not the same thing as a
            # contradictory one, and collapsing the two would lose the distinction that
            # matters: whether to act, or which way.
            where = winners[0]
            verdict = (
                f"[cyan]consistent, weak: separates in {len(parting)} of {len(block)} "
                "settings and never the other way[/cyan]"
            )
        table.add_row(slot, where, str(len(block)), verdict)
    console.print(table)
    console.print(
        f"k_game over {list(config.SENSITIVITY_K_GAME)} and rival noise over "
        f"{list(config.SENSITIVITY_NOISE)}. A declared stress range, not a fitted interval: "
        "the calibration gives k_game a point estimate and no uncertainty, so this brackets "
        "it rather than quoting one."
    )


def _share_cells(s) -> list[str]:
    """A candidate's share, and its difference from the expected-TD pick's.

    A difference inside the noise band prints as "tied" rather than as a number with a
    sign. Printing ±0.1% there would invite reading an ordering into it, and the whole
    point of the paired draws is to know when there is not one.
    """
    if s is None:
        return ["—", "—"]
    if s.tied:
        return [f"{s.share:.1%}", "[dim]tied[/dim]"]
    colour = "green" if s.delta > 0 else "dim"
    return [f"{s.share:.1%}", f"[{colour}]{s.delta:+.1%}[/{colour}]"]


def _prediction_nudge(conn, season: int, wk: int, at: datetime) -> str | None:
    """The prediction deadline, as a reminder, at the only moment it can still be met.

    Silent once the week is observable. A nudge to record a prediction for a week that has
    kicked off or whose report has landed is not a reminder, it is an invitation to file a
    record that `predictions.score` will correctly refuse to score.

    `at` is the decision's own instant rather than a fresh clock read, for the reason
    `state.decision_instant` exists: one command that reads the clock twice can print a
    deadline reminder that contradicts the advice printed above it.
    """
    kickoff = predictions.first_kickoff(conn, season, wk)
    if kickoff is None or at >= kickoff:
        return None
    arrivals, _ = predictions.report_arrivals(conn, season)
    if arrivals.get(wk) or any(r["week"] == wk for r in predictions.archived(conn, season)):
        return None
    return (
        f"No rival-pick prediction on record for week {wk}. Run `pool week` before "
        f"{kickoff.isoformat(timespec='minutes')}, which saves it; after kickoff the "
        "week can be seen and the record can no longer be made."
    )


def _print_pool(pool, season: int, nudge: str | None, uncertain=()) -> None:
    if pool.withheld:
        console.print(
            "[yellow]Pot share withheld; the expected-TD advice below is unaffected.[/yellow]"
        )
        for reason in pool.withheld:
            console.print(f"  {reason}", style="yellow", markup=False)
    elif not pool:
        console.print(
            f"[dim]No rivals on record for {season}, so there is no pot-share view. "
            "Run `pool report import` to turn it on.[/dim]"
        )
    else:
        # An unreported played week and an unresolved past name used to be warnings here.
        # Both leave a season total unknown, so both now withhold the view instead.
        leader = max(pool.rivals, key=lambda r: r.season_tds)
        console.print(
            f"[dim]Pot share (model estimate) vs {len(pool.rivals)} rivals: you {pool.my_tds} TD, "
            f"best rival {leader.display_name} {leader.season_tds}. "
            f"{config.WINPROB_SIMS} paired simulations, seed {config.WINPROB_SEED}.[/dim]"
        )
        # Reported, but unreadable: simulated as though the report had not arrived.
        for name, week, slot, reported in uncertain:
            console.print(
                f"Simulated as unknown: {name} {slot} week {week} ({reported!r} did not "
                "resolve). Re-import that report once it resolves to use the pick.",
                style="yellow",
                markup=False,
            )
    if nudge:
        console.print(f"[yellow]{nudge}[/yellow]")


def _who(pairs, limit: int = 2) -> str:
    names = [name for name, _ in pairs]
    if len(names) <= limit:
        return " and ".join(names)
    return f"{', '.join(names[:limit])} and {len(names) - limit} more"


def _standing(pool) -> str:
    lead = pool.my_tds - max(r.season_tds for r in pool.rivals)
    if lead > 0:
        return f"You lead by {lead}"
    if lead < 0:
        return f"You are {-lead} behind the leader"
    return "You are level with the leader"


def _divergence(a: SlotAdvice, pool) -> str | None:
    """Why the two objectives named different players, in the terms the policy used.

    None when nothing here can say why. That is reported as a defect rather than dressed
    up: a divergence with no statable reason is advice nobody can check, and the failure
    is in the explanation or the policy, never in the reader.
    """
    ev, best = a.recommended, a.best_share
    alt = next((c for c in (a.candidates or a.alternatives) if c.player_id == best.player_id), None)
    trade = f"{best.delta:+.1%} share for {alt.cost:.2f} expected TDs" if alt else "no season cost"
    mine = a.contested.get(ev.player_id, ())
    theirs = a.contested.get(best.player_id, ())
    weight, other = sum(p for _, p in mine), sum(p for _, p in theirs)
    if weight > other:
        return (
            f"Differentiating: {_who(mine)} may take {ev.player_name} this week; "
            f"{best.player_name} is further down their lists. Holding a player a rival holds "
            f"ties your season to theirs, and a tie for first splits the pot. "
            f"{_standing(pool)}. Buys {trade}."
        )
    if other > weight:
        return (
            f"Mirroring: {_who(theirs)} may take {best.player_name} this week; "
            f"{ev.player_name} is further down their lists. Moving with the field narrows "
            f"the gap it can open on you. {_standing(pool)}. Buys {trade}."
        )
    if pool.my_tds != max(r.season_tds for r in pool.rivals):
        return (
            f"{_standing(pool)}, and no rival is more likely to take one of these two than "
            f"the other. What is left is the shape of the seasons rather than their means: "
            f"finishing first is not the same target as scoring most. Buys {trade}."
        )
    return None


def _share_notes(a: SlotAdvice, pool) -> None:
    best = a.best_share
    # Without the opposition there is no second objective to comment on, and every
    # sentence below reads the standings. `_render_slot` defaults `pool` to None so it
    # stays callable on its own, so this cannot assume the two arrived together.
    if best is None or a.recommended is None or not pool:
        return
    if best.se == 0 and all(s.share == best.share for s in a.shares):
        console.print(
            f"  [dim]Every remaining line gives the same {best.share:.0%} share, with no "
            f"variation across {a.sims} simulated seasons: nothing in this slot moves it.[/dim]"
        )
        return
    if a.divergent:
        reason = _divergence(a, pool)
        if reason:
            console.print(f"  [magenta]SHARE:[/magenta] {reason}")
        else:
            console.print(
                f"  [red]Pot share prefers {best.player_name} over "
                f"{a.recommended.player_name} and this tool cannot say why. Treat that as a "
                "bug report, not as advice, and pick on expected TDs.[/red]"
            )
        return
    if all(s.tied or s.player_id == a.recommended.player_id for s in a.shares):
        band = config.WINPROB_SIGNIFICANCE * max((s.delta_se for s in a.shares), default=0.0)
        console.print(
            f"  [dim]No alternative separates from {a.recommended.player_name} by more than "
            f"simulation noise (±{band:.1%} over {a.sims} draws); the two objectives agree "
            "here.[/dim]"
        )


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


@app.command()
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


@app.command()
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


@app.command()
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


@app.command()
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


@app.command()
def plan(week: int | None = WeekOpt, season: int = SeasonOpt, db_path: Path | None = DbOpt):
    """Show the current rest-of-season assignment per slot."""
    conn = _conn(db_path)
    now = state.eastern_now()
    wk = _week(conn, season, week, now)
    proj = _projections(conn, season, wk)
    used = state.used_ids(conn, season)
    locked = state.locked_by_slot(conn, season)
    names = proj.drop_duplicates("player_id").set_index("player_id").player_name.to_dict()
    names.update(state.picks(conn, season).set_index("player_id").player_name.to_dict())
    _status(conn, season, wk, now)
    unavailable = state.unavailable_cells(proj, wk, now)
    plans = {
        slot: plan_slot(
            proj,
            slot,
            wk,
            used,
            locked[slot],
            unavailable=unavailable,
            last_week=max(projections.available_weeks(conn, season)),
        )
        for slot in config.SLOTS
    }
    t = Table("Week", *config.SLOTS, title=f"Remaining-season plan from week {wk}")
    for w in [w for w in projections.available_weeks(conn, season) if w >= wk]:
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
    now = state.eastern_now()
    wk = _week(conn, season, week, now)
    proj = _projections(conn, season, wk)
    if pos.upper() not in config.SLOTS:
        console.print("[red]Position must be QB, RB, or FLEX.[/red]")
        raise typer.Exit(1)
    used = state.used_ids(conn, season)
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
        "Status",
        title=f"{pos.upper()} week {wk}",
    )
    for _, r in sub.iterrows():
        t.add_row(
            r.player_name,
            r.team,
            ("vs " if r.home else "@ ") + r.opponent,
            _fmt_dt(datetime.fromisoformat(r.kickoff)) if r.kickoff_known else "unconfirmed",
            f"{r.base_rate:.2f}",
            f"{r.def_mult:.2f}",
            f"{r.vegas_mult:.2f}",
            f"{r.role_mult:.2f}",
            f"{r.avail_mult:.2f}",
            f"{r.lam:.2f}",
            state.availability(r, wk, now, used),
        )
    console.print(t)


if __name__ == "__main__":
    app()
