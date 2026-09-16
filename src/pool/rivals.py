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

    @property
    def pool_complete(self) -> bool:
        """False when a name their report carried never resolved to a player.

        That player is spent and cannot be named, so `used_ids` is short by an unknown
        amount and every prediction below is made against a pool that is slightly too
        large. Recorded rather than hidden: a prediction made under an incomplete pool is
        weaker evidence than one made under a known one, and the difference has to be
        visible when the hit rate is read, not buried in it.
        """
        return not self.unknown


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
