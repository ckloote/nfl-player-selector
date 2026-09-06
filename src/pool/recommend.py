"""Turn slot plans into this week's advice: pick, alternatives, hold-or-commit."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from . import config, state
from .optimizer import SlotPlan, forced_total, plan_slot


@dataclass
class Candidate:
    player_id: str
    player_name: str
    team: str
    position: str
    opponent: str
    home: bool
    kickoff: datetime
    lam: float
    def_mult: float
    vegas_mult: float
    plan_total: float  # season value if this player is picked now
    cost: float  # optimal_total - plan_total (0 for the recommended pick)
    early: bool  # kicks off before the main (Sunday) slate
    report_status: str | None
    planned_week: int | None  # where the optimal plan would otherwise use them

    @property
    def deadline(self) -> datetime:
        return self.kickoff - timedelta(minutes=config.PICK_DEADLINE_MINUTES)


@dataclass
class SlotAdvice:
    slot: str
    week: int
    locked_player: str | None
    recommended: Candidate | None
    alternatives: list[Candidate]
    hold: bool  # True if the recommended pick is early and the edge is too small
    hold_alternative: Candidate | None
    plan: SlotPlan


def main_slate_start(proj: pd.DataFrame, week: int) -> datetime | None:
    """Kickoff of the first Sunday game (the point after which most injury news is in)."""
    wk = proj[proj.week == week]
    if not len(wk):
        return None
    kicks = pd.to_datetime(wk.kickoff.unique())
    sundays = [k for k in kicks if k.dayofweek == 6]
    return min(sundays) if sundays else max(kicks)


def _candidate(
    plan: SlotPlan,
    proj_week: pd.DataFrame,
    row: int,
    week: int,
    optimal: float,
    slate_start: datetime | None,
) -> Candidate:
    p = plan.players.iloc[row]
    info = proj_week[proj_week.player_id == p.player_id].iloc[0]
    kickoff = datetime.fromisoformat(info.kickoff)
    total = forced_total(plan, row, week)
    planned = next((w for w, r in plan.assignment.items() if r == row), None)
    return Candidate(
        player_id=p.player_id,
        player_name=p.player_name,
        team=p.team,
        position=p.position,
        opponent=info.opponent,
        home=bool(info.home),
        kickoff=kickoff,
        lam=plan.lam(row, week),
        def_mult=float(info.def_mult),
        vegas_mult=float(info.vegas_mult),
        plan_total=total,
        cost=optimal - total,
        early=bool(slate_start is not None and kickoff < slate_start),
        report_status=info.report_status if isinstance(info.report_status, str) else None,
        planned_week=planned,
    )


def _row_id(plan: SlotPlan, row: int | None) -> str | None:
    return None if row is None else str(plan.players.iloc[row].player_id)


def advise_slot(
    proj: pd.DataFrame,
    slot: str,
    week: int,
    used_ids: set[str],
    locked: dict[int, str],
    n_alternatives: int | None = None,
    now: datetime | None = None,
) -> SlotAdvice:
    n_alternatives = config.ALTERNATIVES_SHOWN if n_alternatives is None else n_alternatives
    if n_alternatives < 0:
        raise ValueError("n_alternatives must be zero or more")
    now = state.eastern_now(now)
    plan = plan_slot(
        proj, slot, week, used_ids, locked, unavailable=state.unavailable_cells(proj, week, now)
    )
    if week in locked:
        name = proj.loc[proj.player_id == locked[week], "player_name"]
        return SlotAdvice(
            slot, week, name.iloc[0] if len(name) else locked[week], None, [], False, None, plan
        )

    proj_week = proj[(proj.week == week) & (proj.slot == slot)]
    slate = main_slate_start(proj, week)
    if week not in plan.weeks:
        return SlotAdvice(slot, week, None, None, [], False, None, plan)
    col = plan.weeks.index(week)
    # Every candidate the solver itself considered, not a display-sized slice of them.
    # Sizing the evaluated set by `n_alternatives` made the advice a function of how
    # much of it we intended to print: a cheap Sunday alternative ranked outside the
    # shown rows was invisible to the hold comparison, so asking for a longer list
    # could turn a commit into a hold. `optimizer.build_matrix` already caps this pool
    # at `config.CANDIDATES_PER_SLOT`, so this is one re-solve per surviving candidate.
    playable = [r for r in range(len(plan.players)) if plan.values[r, col] > -1e5]
    cands = [_candidate(plan, proj_week, r, week, plan.total, slate) for r in playable]
    # The solver's own pick keeps precedence at equal cost -- an alternative that ties
    # it describes an equally good plan, not a better one -- and player ID breaks the
    # rest, so a wider evaluated pool cannot reorder the answer by arrival order.
    planned = plan.assignment.get(week)
    cands.sort(key=lambda c: (c.cost, c.player_id != _row_id(plan, planned), -c.lam, c.player_id))
    recommended = cands[0] if cands else None
    # Cheapest deviations first, but always show the top raw-xTD options so the
    # "obvious" pick and its season cost are visible.
    rest = cands[1:]
    top_lam = {c.player_id for c in sorted(rest, key=lambda c: -c.lam)[:3]}
    alternatives = [c for c in rest[:n_alternatives]]
    alternatives += [c for c in rest[n_alternatives:] if c.player_id in top_lam]
    alternatives.sort(key=lambda c: c.cost)

    # The hold decision reads the whole evaluated pool, so it cannot change with how
    # many alternatives are displayed.
    hold, hold_alt = False, None
    if recommended and recommended.early:
        later = [c for c in cands if not c.early]
        if later:
            hold_alt = later[0]
            hold = hold_alt.cost < config.INFO_PREMIUM_TD
    return SlotAdvice(slot, week, None, recommended, alternatives, hold, hold_alt, plan)


def advise_week(
    proj: pd.DataFrame,
    week: int,
    used_ids: set[str],
    locked_by_slot: dict[str, dict[int, str]],
    now: datetime | None = None,
) -> list[SlotAdvice]:
    now = state.eastern_now(now)
    return [
        advise_slot(proj, slot, week, used_ids, locked_by_slot.get(slot, {}), now=now)
        for slot in config.SLOTS
    ]
