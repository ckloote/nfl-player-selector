"""`pool` command-line interface."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from . import backtest as bt
from . import (
    capture,
    config,
    db,
    diagnostics,
    freshness,
    ingest,
    models,
    projections,
    scoring,
    snapshots,
    state,
)
from . import evaluate as ev
from .optimizer import plan_slot
from .recommend import Candidate, SlotAdvice, advise_week

app = typer.Typer(help="NFL touchdown pool decision support.", no_args_is_help=True)
console = Console()

SeasonOpt = typer.Option(config.DEFAULT_SEASON, "--season", "-s", help="Season year")
DbOpt = typer.Option(None, "--db", help="SQLite path (default data/pool.db)")
WeekOpt = typer.Option(None, "--week", "-w", help="Week (default: current)")
SeasonsOpt = typer.Option(
    "2025", "--season", "-s", help="Season, list, or range: 2025 | 2024,2025 | 2017-2025"
)
RoleSourceOpt = typer.Option(None, "--role-source", help="usage, depth, or none")
HorizonOpt = typer.Option(
    None, "--vegas-horizon", help="Closing-line horizon (legacy-closing policy only)"
)
PolicyOpt = typer.Option(
    "historical", "--input-policy", help="historical, snapshots, legacy-closing"
)
DecisionOpt = typer.Option(
    None, "--decision-times", help="CSV: season,week,decision_at (with timezone)"
)
CsvOpt = typer.Option(None, "--csv", help="Write every cell to a CSV")
ProjectionOpt = typer.Option(
    "shipped", "--projection", help="Projection model; see `pool models` for the list"
)


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
    rows, warnings = freshness.report(conn, season, week, now)
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
@app.command()
def refresh(season: int = SeasonOpt, db_path: Path | None = DbOpt):
    """Pull latest stats, schedule, rosters, injuries from nflverse."""
    conn = _conn(db_path)
    console.print(f"Refreshing {season} (prior season {season - 1})...")
    counts = ingest.refresh(conn, season, log=console.print)
    for k, v in counts.items():
        console.print(f"  {k}: {v} rows")
    if counts.failures:
        raise typer.Exit(1)


@app.command()
def recommend(
    week: int | None = WeekOpt,
    season: int = SeasonOpt,
    db_path: Path | None = DbOpt,
    capture_decision: bool = typer.Option(
        True, "--capture/--no-capture", help="Record this decision's inputs, surface and advice"
    ),
):
    """Recommend picks for every open slot this week."""
    conn = _conn(db_path)
    # One instant for the whole decision: the Eastern form compares against kickoffs,
    # the aware form identifies which feed observations were available.
    now = state.eastern_now()
    decided = state.decision_instant(now)
    wk = _week(conn, season, week, now)
    proj = _projections(conn, season, wk)
    used, locked = state.used_ids(conn, season), state.locked_by_slot(conn, season)
    advice = advise_week(proj, wk, used, locked, now=now)
    if capture_decision:
        capture.record_decision(conn, season, wk, proj, advice, used, locked, decision_at=decided)
    console.print(f"[bold]Week {wk} — {season}[/bold]")
    _status(conn, season, wk, now)
    recorded_names = state.picks(conn, season).set_index("player_id").player_name.to_dict()
    for a in advice:
        if a.locked_player in recorded_names:
            a.locked_player = recorded_names[a.locked_player]
        _render_slot(a)


def _render_slot(a: SlotAdvice) -> None:
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
    replaced = {
        r["slot"]: r["player_id"]
        for r in conn.execute(
            "SELECT slot, player_id FROM my_picks WHERE season = ? AND week = ?", (season, wk)
        )
    }
    try:
        warnings = state.record_picks(conn, season, wk, entries, now=now)
    except state.PickError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    # `my_picks` holds the current answer; the capture log holds every answer. A slot
    # that replaces a different player records the correction as well as the entry.
    for e in entries:
        prior = replaced.get(e["slot"])
        if prior is not None and prior != e["player_id"]:
            capture.record_action(
                conn, season, wk, e["slot"], "correction", prior, dict(replaced_by=e["player_id"])
            )
        capture.record_action(
            conn,
            season,
            wk,
            e["slot"],
            "submitted",
            e["player_id"],
            dict(player_name=e["player_name"], team=e.get("team"), replaced=prior),
        )
    for warning in warnings:
        console.print(f"[yellow]Warning: {warning}[/yellow]")
    for e in entries:
        console.print(f"[green]Week {wk} {e['slot']}: {e['player_name']} ({e['team']})[/green]")


@app.command()
def unrecord(week: int, slot: str, season: int = SeasonOpt, db_path: Path | None = DbOpt):
    """Remove a recorded pick (slot: QB, RB, FLEX)."""
    conn = _conn(db_path)
    _week(conn, season, week)
    removed = conn.execute(
        "SELECT player_id FROM my_picks WHERE season = ? AND week = ? AND slot = ?",
        (season, week, slot.upper()),
    ).fetchone()
    try:
        ok = state.remove_pick(conn, season, week, slot.upper())
    except state.PickError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    if ok:
        capture.record_action(
            conn, season, week, slot.upper(), "correction", removed["player_id"], dict(removed=True)
        )
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


# --- backtesting ------------------------------------------------------------
def _seasons(spec: str) -> list[int]:
    """Parse "2025", "2024,2025", or "2017-2025"."""
    out: list[int] = []
    try:
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-", 1))
                out.extend(range(lo, hi + 1))
            else:
                out.append(int(part))
    except ValueError:
        console.print(f"[red]Cannot read {spec!r} as a season, list, or range.[/red]")
        raise typer.Exit(1) from None
    return out


def _floats(spec: str, name: str) -> list[float]:
    try:
        return [float(x) for x in spec.split(",") if x.strip()]
    except ValueError:
        console.print(f"[red]Cannot read {spec!r} as a list of {name} values.[/red]")
        raise typer.Exit(1) from None


def _builder(name: str):
    """Resolve a projection-model name, or exit naming the valid ones."""
    try:
        return models.get(name)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None


def _backtest_ready(conn, season: int) -> list[int]:
    """Weeks available to replay, or a message naming the refresh that fixes it."""
    weeks = projections.available_weeks(conn, season)
    prior = conn.execute(
        "SELECT COUNT(*) FROM player_weeks WHERE season = ?", (season - 1,)
    ).fetchone()[0]
    if not projections.available_weeks(conn, season):
        msg = f"No {season} schedule loaded; run `pool refresh --season {season}` first."
    elif not prior:
        msg = (
            f"No {season - 1} stats loaded — the prior season is the model's starting "
            f"prior. Run `pool refresh --season {season}`, which imports both."
        )
    elif not weeks:
        msg = (
            f"No complete {season} touchdown coverage; run `pool refresh --season {season}` "
            "before replay comparisons."
        )
    else:
        try:
            scoring.require_complete(
                conn, season - 1, projections.available_weeks(conn, season - 1)
            )
            scoring.require_complete(conn, season, weeks)
            return weeks
        except ValueError as exc:
            msg = str(exc)
    console.print(f"[red]{msg}[/red]")
    raise typer.Exit(1)


@app.command()
def backtest(
    season: str = SeasonsOpt,
    strategy: str = typer.Option(
        "optimizer,greedy,random,hindsight", "--strategy", help="Comma-separated strategies"
    ),
    trials: int = typer.Option(config.RANDOM_TRIALS, help="Trials for the random baseline"),
    seed: int = typer.Option(0, help="Seed for the random baseline"),
    discount: float | None = typer.Option(None, help="Override FUTURE_DISCOUNT"),
    prior_weight: float | None = typer.Option(None, help="Override PRIOR_WEIGHT_GAMES"),
    role_source: str | None = RoleSourceOpt,
    vegas_horizon: int | None = HorizonOpt,
    input_policy: str = PolicyOpt,
    decision_times: Path | None = DecisionOpt,
    projection: str = ProjectionOpt,
    detail: bool = typer.Option(False, "--detail", help="Show every week's picks"),
    db_path: Path | None = DbOpt,
):
    """Replay finished seasons with data frozen at each pick deadline."""
    builder = _builder(projection)
    role_source, times = _input_options(input_policy, role_source, vegas_horizon, decision_times)
    conn = _conn(db_path)
    console.print(
        f"Projection: {projection}; random strategy trials: {trials}; "
        f"trial seeds: {seed}–{seed + trials - 1}; "
        f"discount: {config.FUTURE_DISCOUNT if discount is None else discount}; "
        f"prior weight: {config.PRIOR_WEIGHT_GAMES if prior_weight is None else prior_weight}"
    )
    names = [s.strip() for s in strategy.split(",") if s.strip()]
    unknown = [n for n in names if n not in bt.STRATEGIES and n != "hindsight"]
    if unknown:
        known = [*bt.STRATEGIES, "hindsight"]
        console.print(f"[red]Unknown strategy {unknown[0]!r}; choose from {known}.[/red]")
        raise typer.Exit(1)

    deltas: list[float] = []
    overrides = {} if prior_weight is None else {"PRIOR_WEIGHT_GAMES": prior_weight}
    for yr in _seasons(season):
        weeks = _backtest_ready(conn, yr)
        with config.override(**overrides):
            rows = bt.run_season(
                conn,
                yr,
                names,
                weeks=weeks,
                trials=trials,
                seed=seed,
                discount=discount,
                role_source=role_source,
                vegas_horizon=vegas_horizon,
                input_policy=input_policy,
                decision_times=times,
                builder=builder,
            )
        _render_input_provenance(rows[0].input_provenance)
        by_name = {r.strategy: r for r in rows}
        ceiling = by_name["hindsight"].total if "hindsight" in by_name else 0.0
        base = by_name["greedy"].total if "greedy" in by_name else None
        if base is not None and "optimizer" in by_name:
            deltas.append(by_name["optimizer"].total - base)
        _render_backtest(yr, weeks, rows, base, ceiling)
        if detail:
            _render_picks(by_name.get("optimizer") or rows[0])

    if len(deltas) > 1:
        mean = sum(deltas) / len(deltas)
        sd = (sum((d - mean) ** 2 for d in deltas) / (len(deltas) - 1)) ** 0.5
        wins = sum(1 for d in deltas if d > 0)
        console.print(
            f"\n[bold]optimizer - greedy:[/bold] mean {mean:+.2f} TD/season over "
            f"{len(deltas)} seasons (SD {sd:.2f}, won {wins}/{len(deltas)}). "
            f"Standard error {sd / len(deltas) ** 0.5:.2f} — treat anything smaller as noise."
        )


def _render_backtest(season: int, weeks: list[int], rows, base: float | None, ceiling: float):
    t = Table(
        "Strategy",
        "TDs",
        "vs greedy",
        *config.SLOTS,
        "Projected",
        "Proj/act",
        "Zero picks",
        "% ceiling",
        title=f"Backtest {season} (weeks {weeks[0]}-{weeks[-1]})",
    )
    for r in rows:
        total = f"{r.total:.1f}" + (f" ± {r.sd:.1f}" if r.sd else "")
        delta = "—" if base is None or r.strategy == "greedy" else f"{r.total - base:+.1f}"
        ratio = f"{r.projected / r.total:.2f}" if r.projected and r.total else "—"
        t.add_row(
            r.strategy,
            total,
            delta,
            *(f"{r.by_slot[s]:.0f}" for s in config.SLOTS),
            f"{r.projected:.1f}" if r.projected else "—",
            ratio,
            f"{r.zero_picks}/{len(r.picks)}",
            f"{100 * r.total / ceiling:.0f}%" if ceiling else "—",
        )
    console.print(t)


def _render_picks(summary) -> None:
    t = Table(
        "Week",
        "Slot",
        "Player",
        "xTD",
        "Actual",
        "Running",
        title=f"{summary.strategy} picks, {summary.season}",
    )
    running = 0.0
    for p in summary.picks:
        running += p.actual
        t.add_row(
            str(p.week),
            p.slot,
            f"{p.player_name} ({p.team})" if p.player_name else "—",
            f"{p.projected:.2f}",
            f"{p.actual:.0f}",
            f"{running:.0f}",
        )
    console.print(t)


@app.command()
def sweep(
    season: str = SeasonsOpt,
    discount: str = typer.Option("0.9,0.95,0.985,1.0", help="FUTURE_DISCOUNT grid"),
    prior_weight: str = typer.Option("5,7,10", help="PRIOR_WEIGHT_GAMES grid"),
    role_source: str | None = RoleSourceOpt,
    vegas_horizon: int | None = HorizonOpt,
    input_policy: str = PolicyOpt,
    decision_times: Path | None = DecisionOpt,
    csv_out: Path | None = CsvOpt,
    db_path: Path | None = DbOpt,
):
    """Grid-search the future discount and prior weight against the greedy baseline."""
    role_source, times = _input_options(input_policy, role_source, vegas_horizon, decision_times)
    conn = _conn(db_path)
    seasons = _seasons(season)
    for yr in seasons:
        _backtest_ready(conn, yr)
    discounts = _floats(discount, "discount")
    weights = _floats(prior_weight, "prior weight")
    console.print(
        f"Sweeping {len(weights)}x{len(discounts)} cells over {len(seasons)} season(s)..."
    )
    df = bt.sweep(
        conn,
        seasons,
        discounts,
        weights,
        role_source=role_source,
        vegas_horizon=vegas_horizon,
        input_policy=input_policy,
        decision_times=times,
    )
    _render_input_provenance(df.attrs["input_provenance"])
    if csv_out:
        df.to_csv(csv_out, index=False)
        from . import benchmark as bench

        bench.write_json(
            csv_out.with_suffix(".metadata.json"),
            dict(
                schema_version=bench.SCHEMA_VERSION,
                scoring_version=scoring.SCORING_VERSION,
                seasons=seasons,
                discounts=discounts,
                prior_weights=weights,
                input_policy=input_policy,
                role_source=role_source,
                vegas_horizon=vegas_horizon,
                decision_times=[
                    dict(season=s, week=w, decision_at=t) for (s, w), t in (times or {}).items()
                ],
                input_provenance=df.attrs["input_provenance"],
                constants=bench.constants(),
                code=bench.code_identity(),
                assumptions=bench.ASSUMPTIONS,
                input_metadata=bench.input_metadata(conn, {}),
            ),
        )
        console.print(f"Wrote {csv_out} and reproducibility metadata")

    grid = df.pivot_table(index="prior_weight", columns="discount", values="delta", aggfunc="mean")
    sd = df.groupby(["prior_weight", "discount"]).delta.std().mean()
    t = Table(
        "prior weight",
        *(f"{d:g}" for d in grid.columns),
        title="Mean optimizer-minus-greedy TDs per season",
    )
    best = grid.stack().idxmax()
    for w, row in grid.iterrows():
        cells = []
        for d, v in row.items():
            cell = f"{v:+.2f}"
            cells.append(f"[bold green]{cell}[/bold green]" if (w, d) == best else cell)
        marker = " *" if w == config.PRIOR_WEIGHT_GAMES else ""
        t.add_row(f"{w:g}{marker}", *cells)
    console.print(t)
    console.print(
        f"Best cell: prior weight {best[0]:g}, discount {best[1]:g}. "
        f"Current config: prior weight {config.PRIOR_WEIGHT_GAMES:g}, "
        f"discount {config.FUTURE_DISCOUNT:g} (*)."
    )
    console.print(
        f"[yellow]Per-season SD of the delta is {sd:.1f} TD.[/yellow] Differences smaller than "
        "that are noise; prefer a cell that wins in most seasons over the maximum."
    )


# --- projection benchmark ---------------------------------------------------
@app.command("models")
def list_models():
    """List the projection models the benchmark can run."""
    t = Table("Model", "Role", title="Projection models")
    roles = {
        "random": "null — lambda shuffled within slot-week",
        "within-player": "null — each player's lambda shuffled across their weeks",
        "historical-rate": "baseline — prior-season TD/game, unregressed",
        "regressed-rate": "baseline — prior season regressed to the positional mean",
        "current-season-rate": "baseline — current season only, shrunk",
        "vegas-environment": "baseline — team scoring environment, no player TD history",
        "player-vegas": "baseline — base rate x Vegas; the principal challenger",
        "shipped": "the production model, frozen",
        "no-vegas": "ablation — shipped without the Vegas multiplier",
        "base-rate-only": "ablation — shipped base rate, no matchup context",
    }
    for name in models.BUILDERS:
        t.add_row(name, roles.get(name, ""))
    console.print(t)


def _model_set(spec: str) -> dict:
    """Resolve a comma-separated model list, or `bakeoff` / `all`."""
    if spec == "bakeoff":
        names = list(models.BAKEOFF)
    elif spec == "all":
        names = list(models.BUILDERS)
    else:
        names = [n.strip() for n in spec.split(",") if n.strip()]
    return {n: _builder(n) for n in names}


@app.command()
def evaluate(
    season: str = SeasonsOpt,
    model: str = typer.Option("bakeoff", "--model", "-m", help="Models, or bakeoff / all"),
    baseline: str = typer.Option("shipped", help="Model every comparison is paired against"),
    seeds: str = typer.Option("0-19", help="Shuffled model seeds: integers or ranges"),
    k: int = typer.Option(ev.PRIMARY_K, "--k", help="Primary top-k for paired comparison"),
    role_source: str | None = RoleSourceOpt,
    vegas_horizon: int | None = HorizonOpt,
    input_policy: str = PolicyOpt,
    decision_times: Path | None = DecisionOpt,
    csv_out: Path | None = CsvOpt,
    db_path: Path | None = DbOpt,
):
    """Report common-pool ranking diagnostics and separate achieved season scores."""
    chosen = _model_set(model)
    try:
        ev.validate_models(chosen, baseline)
        seed_values = models.parse_seeds(seeds)
        if k < 1:
            raise ValueError("k must be positive")
    except (ValueError, KeyError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    role_source, times = _input_options(input_policy, role_source, vegas_horizon, decision_times)
    conn = _conn(db_path)
    seasons = _seasons(season)
    for yr in seasons:
        _backtest_ready(conn, yr)
    console.print(f"Evaluating {len(chosen)} model(s) over {len(seasons)} season(s)...")
    df = ev.forecast_set(
        conn,
        seasons,
        chosen,
        role_source=role_source,
        vegas_horizon=vegas_horizon,
        baseline=baseline,
        seeds=seed_values,
        input_policy=input_policy,
        decision_times=times,
        log=None,
    )
    console.print(
        f"{len(df):,} forecast rows; shuffled seeds {seed_values}; "
        "deterministic models run once. Ranking uses all hard-eligible candidates "
        "before assignment pruning."
    )
    if csv_out:
        from . import benchmark as bench

        df.to_csv(csv_out, index=False)
        bench.save_frame(csv_out.with_suffix(".replays.csv"), pd.DataFrame(df.attrs["replays"]))
        bench.save_frame(csv_out.with_suffix(".picks.csv"), pd.DataFrame(df.attrs["picks"]))
        bench.write_json(
            csv_out.with_suffix(".metadata.json"),
            dict(
                schema_version=bench.SCHEMA_VERSION,
                scoring_version=scoring.SCORING_VERSION,
                baseline=baseline,
                models=list(chosen),
                seeds=seed_values,
                seasons=seasons,
                input_policy=input_policy,
                role_source=role_source,
                vegas_horizon=vegas_horizon,
                decision_times=[
                    dict(season=s, week=w, decision_at=t) for (s, w), t in (times or {}).items()
                ],
                input_provenance=df.attrs["input_provenance"],
                constants=bench.constants(),
                code=bench.code_identity(),
                assumptions=bench.ASSUMPTIONS,
                input_metadata=bench.input_metadata(conn, {}),
            ),
        )
        console.print(f"Wrote {csv_out}, replay picks, scores and reproducibility metadata")
    _render_evaluation(df, baseline, k)


def _input_options(policy, role, horizon, path):
    try:
        role = projections.validate_policy(policy, role, horizon)
        if policy == "snapshots" and path is None:
            raise ValueError("--input-policy snapshots requires --decision-times PATH")
        if path is not None and policy != "snapshots":
            raise ValueError("--decision-times requires --input-policy snapshots")
        times = snapshots.decision_times(path) if path is not None else None
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(
        f"Input policy: {policy}; roles: {role}; scoring: all thrown/scored TD credits. "
        "Historical schedules, weekly reports and corrected stats are approximations."
    )
    from . import benchmark as bench

    console.print(
        "Candidates: active-roster QB/RB/WR/TE, with latest stat-team fallback; "
        "hard exclusions before pruning/ranking, eligible zeros retained, player-ID ties. "
        f"Code fingerprint: {bench.code_identity()['code_hash']}"
    )
    return role, times


def _render_input_provenance(records):
    observed = [r for r in records if r["inputs"]]
    if not observed:
        return
    table = Table("Decision", "Input season", "Feed", "Observed (UTC)", "Age hours", "Status")
    for record in observed:
        for feed in record["inputs"]:
            table.add_row(
                f"{record['season']} W{record['week']}: {record['decision_at']}",
                str(feed["season"]),
                feed["feed"],
                feed["observed_at"] or "none",
                "—" if feed["age_hours"] is None else f"{feed['age_hours']:.2f}",
                "missing / fallback"
                if feed["missing"]
                else "stale"
                if feed["stale"]
                else "observed",
            )
    console.print(table)


def _render_evaluation(df, baseline: str, k: int) -> None:
    _render_input_provenance(df.attrs["input_provenance"])
    console.print(
        "Ranking diagnostics: TDs per ranked candidate; shared baseline greedy depletion."
    )
    console.print(ev.top_k(df).to_string(index=False))
    if df.model.nunique() > 1:
        console.print(f"Paired ranking differences vs {baseline}; uncertainty across seasons:")
        console.print(ev.paired_top_k(df, baseline, k).to_string(index=False))
        console.print("Paired deviance on baseline lambda > 0.30 (diagnostic):")
        console.print(ev.paired_deviance(df, baseline).to_string(index=False))
    console.print("Calibration computed separately for each seed (slope, intercept):")
    for (model, seed), group in df.groupby(["model", "seed"]):
        fit = ev.calibration(group)
        console.print(
            f"{model} seed={seed}: slope {fit['slope']:.3f}, intercept {fit['intercept']:.3f}"
        )
    console.print("Achieved season TDs: each strategy uses its own player history.")
    console.print(
        ev.replay_summary(pd.DataFrame(df.attrs["replays"]), baseline).to_string(index=False)
    )


@app.command()
def diagnose(
    run: Annotated[Path, typer.Option("--run", help="A completed benchmark output directory")],
    out: Annotated[Path, typer.Option("--out", help="Where to write the diagnostic export")],
    model: str = typer.Option("shipped", help="Model whose saved surface to diagnose"),
    seed: int = typer.Option(-1, help="Seed; -1 is the deterministic sentinel"),
    seasons: str | None = typer.Option(None, "--season", help="Season or range, e.g. 2019-2025"),
):
    """Describe a saved study's rate errors by population, position, rate, availability
    and forecast horizon, and write a dated readiness note.

    Descriptive only: it fits no correction and selects no model. Group definitions are
    frozen before any outcome is read.
    """
    try:
        diagnostics.export(
            run,
            out,
            model=model,
            seed=seed,
            seasons=_seasons(seasons) if seasons else None,
            log=console.print,
        )
    except (ValueError, OSError, KeyError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


@app.command()
def benchmark(
    config_path: Annotated[Path, typer.Option("--config", help="Experiment TOML")],
    output: Annotated[Path, typer.Option(help="Ignored research dataset and experiment artifacts")],
    resume: bool = typer.Option(False, help="Continue compatible season checkpoints"),
):
    """Run a specified benchmark against a frozen, fully audited research database."""
    from . import benchmark as bench

    try:
        bench.run(config_path, output, resume=resume, log=console.print)
    except (ValueError, OSError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


if __name__ == "__main__":
    app()
