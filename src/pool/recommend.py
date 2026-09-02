"""Turn slot plans into this week's advice: pick, alternatives, hold-or-commit."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from . import config
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


def advise_slot(
    proj: pd.DataFrame,
    slot: str,
    week: int,
    used_ids: set[str],
    locked: dict[int, str],
    n_alternatives: int = config.ALTERNATIVES_SHOWN,
) -> SlotAdvice:
    plan = plan_slot(proj, slot, week, used_ids, locked)
    if week in locked:
        name = proj.loc[proj.player_id == locked[week], "player_name"]
        return SlotAdvice(
            slot, week, name.iloc[0] if len(name) else locked[week], None, [], False, None, plan
        )

    proj_week = proj[(proj.week == week) & (proj.slot == slot)]
    slate = main_slate_start(proj, week)
    col = plan.weeks.index(week)
    playable = [r for r in range(len(plan.players)) if plan.values[r, col] > -1e5]
    # Evaluate the plan's pick plus the top-N by this-week lambda.
    by_lam = sorted(playable, key=lambda r: -plan.raw[r, col])
    rows = list(
        dict.fromkeys(
            ([plan.assignment[week]] if week in plan.assignment else [])
            + by_lam[: n_alternatives + 4]
        )
    )
    cands = [_candidate(plan, proj_week, r, week, plan.total, slate) for r in rows]
    cands.sort(key=lambda c: c.cost)
    recommended = cands[0] if cands else None
    # Cheapest deviations first, but always show the top raw-xTD options so the
    # "obvious" pick and its season cost are visible.
    rest = cands[1:]
    top_lam = {c.player_id for c in sorted(rest, key=lambda c: -c.lam)[:3]}
    alternatives = [c for c in rest[:n_alternatives]]
    alternatives += [c for c in rest[n_alternatives:] if c.player_id in top_lam]
    alternatives.sort(key=lambda c: c.cost)

    hold, hold_alt = False, None
    if recommended and recommended.early:
        later = [c for c in cands if not c.early]
        if later:
            hold_alt = later[0]
            hold = hold_alt.cost < config.INFO_PREMIUM_TD
    return SlotAdvice(slot, week, None, recommended, alternatives, hold, hold_alt, plan)


def advise_week(
    proj: pd.DataFrame, week: int, used_ids: set[str], locked_by_slot: dict[str, dict[int, str]]
) -> list[SlotAdvice]:
    return [
        advise_slot(proj, slot, week, used_ids, locked_by_slot.get(slot, {}))
        for slot in config.SLOTS
    ]
