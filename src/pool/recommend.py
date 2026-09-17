"""Turn slot plans into this week's advice: pick, alternatives, hold-or-commit.

Two objectives, side by side, never one replacing the other. Expected touchdowns is what
the assignment solver maximises and what this module has always given. Expected share of
the pot is the second opinion: a tie for first splits the winnings, so a pot share counts a
win as 1 and a k-way tie as 1/k, and that is a function of the joint distribution of five
season totals rather than a separable sum any matrix can express.

They agree when my candidates are equally unrelated to what rivals hold -- not, as this
module used to say, when everyone is level. Levelness is neither necessary nor sufficient:
with the standings dead level and every rival about to take the player I would take,
mirroring them buys a guaranteed k-way split of the pot and differentiating buys a chance
at all of it. Their disagreement is the interesting output, which is why both are shown and
neither is silently swapped for the other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import config, rivals, simulate, state
from .optimizer import SlotPlan, forced_plan, plan_slot


@dataclass
class Candidate:
    player_id: str
    player_name: str
    team: str
    position: str
    opponent: str
    home: bool
    kickoff: datetime
    # Decided when the candidate is built, not read back later. As a property this
    # answered under whatever policy was current at access time, so advice derived
    # under a captured decision's settings returned today's deadline once the
    # restoring override had ended.
    deadline: datetime
    lam: float
    def_mult: float
    vegas_mult: float
    plan_total: float  # season value if this player is picked now
    cost: float  # optimal_total - plan_total (0 for the recommended pick)
    early: bool  # kicks off before the main (Sunday) slate
    report_status: str | None
    planned_week: int | None  # where the optimal plan would otherwise use them
    # Which player each remaining week gets if this candidate is spent now. Carried rather
    # than recomputed because spending a player changes the whole rest of the season, and
    # a pot share is a function of the season, not of one week.
    future: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PotShare:
    """One candidate's simulated expected share of the pot, against the expected-TD pick."""

    player_id: str
    player_name: str
    share: float
    se: float
    delta: float  # minus the expected-TD recommendation's share, on shared draws
    delta_se: float

    @property
    def tied(self) -> bool:
        """Inside simulation noise of the expected-TD pick, so not ordered against it."""
        return abs(self.delta) <= config.WINPROB_SIGNIFICANCE * self.delta_se


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
    shares: list[PotShare] = field(default_factory=list)
    sims: int = 0
    # player_id -> ((rival display name, probability they take them this week), ...). Taken
    # from the same sampling sets the rollouts drew from, so a printed explanation of a
    # divergence describes the distribution the policy actually used rather than a second
    # guess at it. Empty when no pool state was supplied.
    contested: dict[str, tuple[tuple[str, float], ...]] = field(default_factory=dict)

    @property
    def best_share(self) -> PotShare | None:
        return self.shares[0] if self.shares else None

    @property
    def divergent(self) -> bool:
        """The two objectives name different players, and the difference clears the noise."""
        best = self.best_share
        return bool(
            best
            and self.recommended
            and best.player_id != self.recommended.player_id
            and not best.tied
        )


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
    total, future = forced_plan(plan, row, week)
    planned = next((w for w, r in plan.assignment.items() if r == row), None)
    return Candidate(
        player_id=p.player_id,
        player_name=p.player_name,
        team=p.team,
        position=p.position,
        opponent=info.opponent,
        home=bool(info.home),
        kickoff=kickoff,
        deadline=kickoff - timedelta(minutes=config.PICK_DEADLINE_MINUTES),
        lam=plan.lam(row, week),
        def_mult=float(info.def_mult),
        vegas_mult=float(info.vegas_mult),
        plan_total=total,
        cost=optimal - total,
        early=bool(slate_start is not None and kickoff < slate_start),
        report_status=info.report_status if isinstance(info.report_status, str) else None,
        planned_week=planned,
        future=future,
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


def _locked_picks(locked_by_slot, week) -> list[tuple[int, str]]:
    """Picks already recorded for this week or later. They score, whatever I choose now."""
    return [
        (int(w), str(pid))
        for locks in locked_by_slot.values()
        for w, pid in locks.items()
        if w >= week
    ]


def _baseline(advice: list[SlotAdvice]) -> dict[str, list[tuple[int, str]]]:
    """Each slot's remaining-season plan as (week, player) pairs, before anything is forced."""
    return {
        a.slot: [
            (int(w), str(a.plan.players.iloc[r].player_id)) for w, r in a.plan.assignment.items()
        ]
        for a in advice
    }


def _rival_totals(proj, week, weeks, pool, draws, sims, seed):
    """Each rival's simulated season total as (sims, rivals), and who they may take now.

    Their pick paths do not depend on what I choose -- this pool lets two entrants hold the
    same player, so nothing I take is denied to them -- which is what lets these be drawn
    once and reused across every candidate I am weighing.

    Uncertainty about their choices enters as a finite set of scenarios rather than one path
    per simulation, because a path costs a greedy rollout and an outcome costs a lookup.

    The second return value is this week's sampling set per slot, read off the same grids
    the rollouts walk. It explains a divergence; it never changes one.
    """
    rng = np.random.default_rng(seed)
    totals = np.zeros((sims, len(pool.rivals)))
    contested: dict[str, dict[str, list[tuple[str, float]]]] = {slot: {} for slot in config.SLOTS}
    edges = np.linspace(0, sims, config.RIVAL_SCENARIOS + 1).astype(int)
    for index, rival in enumerate(pool.rivals):
        grids = {
            slot: rivals.remaining_matrix(proj, slot, week, weeks, rival.used_ids)
            for slot in config.SLOTS
        }
        pinned = {slot: rival.pinned(slot, weeks) for slot in config.SLOTS}
        for slot, (players, values) in grids.items():
            # A reported pick is not a guess at this week, so it enters the explanation at
            # certainty rather than as one of three things they might do.
            reported = pinned[slot].get(week)
            if reported is not None:
                contested[slot].setdefault(reported, []).append((rival.display_name, 1.0))
                continue
            rows = rivals.options(values, 0, top_n=config.RIVAL_NOISE_TOP_N)
            for row in rows:
                pid = str(players.iloc[row].player_id)
                contested[slot].setdefault(pid, []).append((rival.display_name, 1 / len(rows)))
        column = np.full(sims, float(rival.season_tds))
        for scenario in range(config.RIVAL_SCENARIOS):
            picks: list[tuple[int, str]] = []
            for slot, (players, values) in grids.items():
                path = rivals.rollout(
                    players,
                    values,
                    weeks,
                    rng=rng,
                    top_n=config.RIVAL_NOISE_TOP_N,
                    known=pinned[slot],
                )
                picks += list(path.items())
            low, high = edges[scenario], edges[scenario + 1]
            column[low:high] += draws.totals(picks)[low:high]
        totals[:, index] = column
    frozen = {
        slot: {
            pid: tuple(sorted(who, key=lambda pair: (-pair[1], pair[0])))
            for pid, who in by_player.items()
        }
        for slot, by_player in contested.items()
    }
    return totals, frozen


def pot_shares(
    proj: pd.DataFrame,
    week: int,
    advice: list[SlotAdvice],
    pool: rivals.PoolState,
    locked_by_slot: dict[str, dict[int, str]],
    *,
    sims: int | None = None,
    seed: int | None = None,
    params: simulate.Params | None = None,
) -> None:
    """Score every candidate by its simulated expected share of the pot, in place.

    A one-step lookahead, and it should be called nothing grander: hold the expected-TD plan
    for the rest of the season, vary only this week's pick, and simulate. The search space
    over whole seasons is far too large to enumerate and this does not pretend to.

    Every candidate is evaluated on the *same* draws and the same rival paths, so the
    difference between two of them is estimated far more precisely than either level. That
    is the quantity the decision turns on, and `PotShare.tied` refuses to order two
    candidates whose difference does not clear it.
    """
    sims = config.WINPROB_SIMS if sims is None else sims
    seed = config.WINPROB_SEED if seed is None else seed
    weeks = sorted({int(w) for w in proj.week.unique() if w >= week})
    if not weeks or not pool.rivals:
        return
    keep = {pid for a in advice for pid in a.plan.players.player_id}
    keep |= {pid for _, pid in _locked_picks(locked_by_slot, week)}
    # A reported pick has to be sampled even when the candidate matrix never kept him.
    # `Draws.totals` scores an absent cell as zero, so omitting one here would credit a
    # rival with nothing for a week they have already told us they filled.
    keep |= {pid for rival in pool.rivals for _, _, pid in rival.known}
    for slot in config.SLOTS:
        for rival in pool.rivals:
            players, _ = rivals.remaining_matrix(proj, slot, week, weeks, rival.used_ids)
            keep |= set(players.player_id)
    sub = proj[proj.player_id.isin(keep) & proj.week.isin(weeks)]
    draws = simulate.sample(sub, weeks, sims=sims, seed=seed, params=params)
    opposition, contested = _rival_totals(proj, week, weeks, pool, draws, sims, seed)

    locked = _locked_picks(locked_by_slot, week)
    baseline = _baseline(advice)
    for slot_advice in advice:
        if not slot_advice.recommended:
            continue
        others = [
            pick
            for slot, picks in baseline.items()
            if slot != slot_advice.slot
            for pick in picks
        ]
        evaluated = {}
        for candidate in [slot_advice.recommended, *slot_advice.alternatives]:
            mine = (
                locked
                + others
                + [
                    (int(w), str(slot_advice.plan.players.iloc[r].player_id))
                    for w, r in candidate.future.items()
                ]
            )
            total = pool.my_tds + draws.totals(mine)
            evaluated[candidate.player_id] = (candidate, simulate.shares(total, opposition))
        reference = evaluated[slot_advice.recommended.player_id][1]
        shares = []
        for candidate, drawn in evaluated.values():
            delta, delta_se = simulate.paired(drawn, reference)
            shares.append(
                PotShare(
                    candidate.player_id,
                    candidate.player_name,
                    float(drawn.mean()),
                    float(drawn.std(ddof=1) / np.sqrt(sims)),
                    delta,
                    delta_se,
                )
            )
        # The expected-TD pick keeps precedence at equal share, for the reason it keeps it
        # at equal cost: a candidate that ties it describes an equally good plan, not a
        # better one.
        shares.sort(
            key=lambda s: (-s.share, s.player_id != slot_advice.recommended.player_id, s.player_id)
        )
        slot_advice.shares = shares
        slot_advice.sims = sims
        slot_advice.contested = contested.get(slot_advice.slot, {})


def advise_week(
    proj: pd.DataFrame,
    week: int,
    used_ids: set[str],
    locked_by_slot: dict[str, dict[int, str]],
    now: datetime | None = None,
    pool: rivals.PoolState | None = None,
) -> list[SlotAdvice]:
    """Expected-TD advice, plus the win-probability view when the pool's state is known.

    With no `pool` this returns exactly what it always has, candidate for candidate. The
    second objective is additive; it never edits the first.
    """
    now = state.eastern_now(now)
    advice = [
        advise_slot(proj, slot, week, used_ids, locked_by_slot.get(slot, {}), now=now)
        for slot in config.SLOTS
    ]
    if pool:
        pot_shares(proj, week, advice, pool, locked_by_slot)
    return advice
