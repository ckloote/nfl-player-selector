"""`pool report` and `pool standings`: the pool's weekly reports, and the standings
computed from them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.table import Table

from .. import config, entrants
from .. import standings as st
from .common import DbOpt, SeasonOpt, _conn, _week, console

report_app = typer.Typer(
    help="Archive and read official weekly pool reports.", no_args_is_help=True
)


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
