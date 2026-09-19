"""The decision in full: `recommend`, `plan` and `players`, and the one decision both
`recommend` and `week` take and render.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd
import typer
from rich.table import Table
from rich.text import Text

from .. import capture, config, db, freshness, predictions, projections, simulate, state
from ..optimizer import plan_slot
from ..recommend import Candidate, SlotAdvice, advise_week, pot_shares
from ..rivals import PoolState
from .common import (
    DbOpt,
    SeasonOpt,
    WeekOpt,
    _conn,
    _fmt_dt,
    _print_freshness,
    _projections,
    _short,
    _status,
    _week,
    console,
)


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
