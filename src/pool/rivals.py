"""What a rival will pick from what they have left. Pure: no database, no clock, no I/O.

Three hypotheses about a person, not one. The phase plan names greedy-from-remaining as the
defensible baseline and it probably is, but which of these describes these five people is a
fact about them rather than a choice to make in advance -- and it is only ever measurable if
all three were written down before the week they describe. Recording the other two costs a
dictionary lookup now and cannot be done retroactively at any price.

The ranked list matters more than its winner. A simulator needs a distribution over a
rival's pick, and a hit rate on the argmax alone cannot calibrate one; where the actual pick
landed in each ranking is exactly what can.

Nothing here touches the database, because this module is bound for the enforced decision
closure and the closure has no business acquiring a report parser. The caller assembles
`RivalState` and passes it in, which is the seam `recommend.advise_slot` already uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .optimizer import FORBIDDEN, build_matrix, forced_total, plan_slot

# How much of each ranking is kept. Five is enough to locate almost every actual pick and
# small enough that the archived payload stays readable by eye, which matters for a record
# whose whole purpose is being checked later.
TOP_N = 5


@dataclass(frozen=True)
class RivalState:
    """One opponent, as of a decision: who they are, what they have spent, where they stand."""

    entrant_id: str
    display_name: str
    used_ids: frozenset[str] = frozenset()
    unknown: int = 0
    season_tds: int = 0
    # Weeks before this one that were *played* and never reported. Three more players are
    # spent per such week and none of them can be named, which overstates the remaining
    # pool the same way `unknown` does -- the same defect through a different door, and the
    # reason this is a second field rather than folded into that count. A week that has not
    # kicked off does not belong here: nobody has spent anything in it, and the caller is
    # simply looking further ahead than the pool has been played.
    missing_weeks: tuple[int, ...] = ()
    # (slot, week, player_id) for picks from this week on that are already on record. A
    # pick that has been reported is not a pick to predict, and simulating a guess at one
    # I have been told is strictly worse than using it.
    known: tuple[tuple[str, int, str], ...] = ()

    def pinned(self, slot: str, weeks) -> dict[int, str]:
        """This rival's already-reported picks for one slot, within `weeks`."""
        allowed = {int(w) for w in weeks}
        return {
            int(week): str(pid)
            for name, week, pid in self.known
            if name == slot and int(week) in allowed
        }

    @property
    def pool_complete(self) -> bool:
        """False when anything they have spent cannot be named.

        Either a name their report carried never resolved to a player, or a week's report
        never arrived at all. Either way `used_ids` is short by an unknown amount, every
        prediction below is made against a pool that is too large, and the simulated rival
        is free to spend a player they have in fact already used. Recorded rather than
        hidden: a prediction made under an incomplete pool is weaker evidence than one made
        under a known one, and the difference has to be visible when the hit rate is read.
        """
        return not self.unknown and not self.missing_weeks


@dataclass(frozen=True)
class PoolState:
    """Where the pool stands, as the policy needs it: everyone else, and what I have banked.

    Assembled by the caller from `standings`, never read from the database here. That is
    the seam that keeps a report parser out of the enforced decision closure.
    """

    rivals: tuple[RivalState, ...] = ()
    my_tds: int = 0
    # Final player-week results known at the decision instant, including zeroes.
    finalized: tuple[tuple[int, str, int], ...] = ()

    def __bool__(self) -> bool:
        return bool(self.rivals)


@dataclass(frozen=True)
class Ranked:
    player_id: str
    player_name: str
    lam: float
    score: float  # the predictor's own ordering value; equals lam for greedy and naive


def _column(proj: pd.DataFrame, slot: str, week: int, used_ids):
    """This week's candidates through the optimizer's own matrix builder.

    Going through `build_matrix` rather than filtering by hand is what makes a predictor
    comparable to the tool's own advice: same candidate pool, same eligibility, same cap.
    """
    players, values, _ = build_matrix(proj, slot, [week], week, set(used_ids))
    if not len(players):
        return None, None, None
    column = values[:, 0]
    rows = np.flatnonzero(column > FORBIDDEN / 2)
    return (players, column, rows) if rows.size else (None, None, None)


def _ranked(players: pd.DataFrame, order, scores, lams, top_n: int) -> list[Ranked]:
    out = []
    for row in order[:top_n]:
        r = int(row)
        player = players.iloc[r]
        out.append(
            Ranked(str(player.player_id), str(player.player_name), float(lams[r]), float(scores[r]))
        )
    return out


def predict_greedy(proj, slot, week, state: RivalState, *, top_n: int = TOP_N) -> list[Ranked]:
    """Takes the best player available this week, respecting what they have spent."""
    players, column, rows = _column(proj, slot, week, state.used_ids)
    if players is None:
        return []
    return _ranked(players, rows[np.argsort(-column[rows], kind="stable")], column, column, top_n)


def predict_naive(proj, slot, week, state: RivalState, *, top_n: int = TOP_N) -> list[Ranked]:
    """Takes the best player in the slot, ignoring their own used pool entirely.

    Here to answer a real question that nothing else can: do these people track the
    one-player-per-season constraint at all? The gap between this hit rate and greedy's is
    the measurement, it is a fact about them rather than about the model, and it changes
    what the simulator should do with their remaining pool.
    """
    players, column, rows = _column(proj, slot, week, ())
    if players is None:
        return []
    return _ranked(players, rows[np.argsort(-column[rows], kind="stable")], column, column, top_n)


def predict_optimizer(proj, slot, week, state: RivalState, *, top_n: int = TOP_N) -> list[Ranked]:
    """Plans the rest of the season the way this tool does.

    Ranked by the season value of forcing each candidate into this week, which is the
    quantity `recommend` already prints as "season cost". A rival modelled this way is
    modelled as running this program, which is the most sophisticated opponent worth
    positing and a useful ceiling even if nobody is actually doing it.
    """
    plan = plan_slot(proj, slot, week, set(state.used_ids))
    if not len(plan.players) or week not in plan.weeks:
        return []
    col = plan.weeks.index(week)
    rows = np.flatnonzero(plan.values[:, col] > FORBIDDEN / 2)
    if not rows.size:
        return []
    totals = np.full(plan.values.shape[0], -np.inf)
    lams = np.zeros(plan.values.shape[0])
    for row in rows:
        totals[int(row)] = forced_total(plan, int(row), week)
        lams[int(row)] = plan.lam(int(row), week)
    order = rows[np.argsort(-totals[rows], kind="stable")]
    return _ranked(plan.players, order, totals, lams, top_n)


def remaining_matrix(proj, slot: str, week: int, weeks, used_ids):
    """The candidate grid a rival still has, built once and walked many times.

    Rolling a season out scenario by scenario would otherwise rebuild this matrix for every
    draw; it is the expensive part and it does not change between them.
    """
    players, values, _ = build_matrix(proj, slot, list(weeks), week, set(used_ids))
    return players, values


def options(values, col: int, taken=(), *, top_n: int = 1) -> list[int]:
    """The rows `rollout` may take in one week: the best `top_n` still playable.

    Shared with `rollout` rather than written out again beside it. Telling somebody which
    rivals might take a player is only worth saying if it names the same distribution the
    simulation drew from, and two copies of this rule would drift apart silently.
    """
    if not values.shape[0] or col >= values.shape[1]:
        return []
    column = values[:, col].copy()
    if len(taken):
        column[list(taken)] = FORBIDDEN
    playable = np.flatnonzero(column > FORBIDDEN / 2)
    if not playable.size:
        return []
    return [int(r) for r in playable[np.argsort(-column[playable], kind="stable")][:top_n]]


def rollout(players, values, weeks, *, rng, top_n: int = 1, known=None) -> dict[int, str]:
    """One plausible path of a rival's remaining picks: greedy, with noise, no reuse.

    `top_n` above 1 is where uncertainty about what an opponent does enters. A perfectly
    predictable rival would let the policy block them exactly, which is the more dangerous
    error -- it would claim an edge that depends on knowing something unknowable.

    `known` pins weeks whose pick has already been reported. Those weeks are not predicted
    and the players are spent, so the rest of the path cannot spend them again. A pinned
    player who is not in this matrix -- capped out of the candidate pool, or ineligible --
    still holds his week; he simply contributes whatever his cell is worth, which is the
    honest answer for a player the forecast does not rate.
    """
    picks: dict[int, str] = {int(w): str(p) for w, p in (known or {}).items()}
    if not len(players):
        return picks
    rows = {str(pid): row for row, pid in enumerate(players.player_id)}
    taken: list[int] = [rows[pid] for pid in picks.values() if pid in rows]
    for col, week in enumerate(weeks):
        if int(week) in picks:
            continue
        best = options(values, col, taken, top_n=top_n)
        if not best:
            continue
        row = best[0] if len(best) == 1 else int(rng.choice(best))
        taken.append(row)
        picks[int(week)] = str(players.iloc[row].player_id)
    return picks


PREDICTORS = {
    "greedy": predict_greedy,
    "optimizer": predict_optimizer,
    "naive": predict_naive,
}


def predict_all(proj, slot, week, state: RivalState, *, top_n: int = TOP_N) -> dict:
    """Every hypothesis at once. The archive keeps all of them; nothing here picks a winner."""
    return {name: fn(proj, slot, week, state, top_n=top_n) for name, fn in PREDICTORS.items()}


def rank_of(ranked: list[Ranked], player_id: str | None) -> int | None:
    """Where the actual pick landed, 1-based, or None if it was outside the kept list."""
    if player_id is None:
        return None
    for position, candidate in enumerate(ranked, 1):
        if candidate.player_id == player_id:
            return position
    return None
