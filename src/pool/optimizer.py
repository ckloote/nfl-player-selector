"""Season-long assignment: which player to spend in which week, per slot.

Each slot (QB, RB, FLEX) is an independent linear assignment problem:
rows = candidate players, cols = remaining weeks, value = discounted expected
TDs. Unavailable cells (bye, injured, already used) are forbidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from . import config

FORBIDDEN = -1e6


@dataclass
class SlotPlan:
    slot: str
    weeks: list[int]
    players: pd.DataFrame  # candidate table: player_id, player_name, team, position
    values: np.ndarray  # players x weeks, discounted; FORBIDDEN where unavailable
    raw: np.ndarray  # players x weeks, undiscounted lambda; nan where unavailable
    assignment: dict[int, int] = field(default_factory=dict)  # week -> row index
    total: float = 0.0

    def pick_for(self, week: int) -> pd.Series | None:
        idx = self.assignment.get(week)
        return None if idx is None else self.players.iloc[idx]

    def lam(self, row: int, week: int) -> float:
        return float(self.raw[row, self.weeks.index(week)])


def build_matrix(
    proj: pd.DataFrame,
    slot: str,
    weeks: list[int],
    current_week: int,
    used_ids: set[str],
    discount: float | None = None,
    max_players: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # Resolved here, not in the signature: a default argument binds at import,
    # so `config.override(FUTURE_DISCOUNT=...)` would never reach this and a
    # parameter sweep would silently report the same number for every value.
    discount = config.FUTURE_DISCOUNT if discount is None else discount
    max_players = config.CANDIDATES_PER_SLOT if max_players is None else max_players
    sub = proj[(proj.slot == slot) & proj.week.isin(weeks) & ~proj.player_id.isin(used_ids)]
    if not len(sub):
        empty = pd.DataFrame(columns=["player_id", "player_name", "team", "position"])
        return empty, np.zeros((0, len(weeks))), np.zeros((0, len(weeks)))
    # Rank candidates by their best available week (keeps matchup plays in the pool).
    rank = sub.groupby("player_id").lam.max().sort_values(ascending=False)
    keep = list(rank.index[:max_players])
    sub = sub[sub.player_id.isin(keep)]
    players = (
        sub.drop_duplicates("player_id")[["player_id", "player_name", "team", "position"]]
        .set_index("player_id")
        .loc[keep]
        .reset_index()
    )
    raw = sub.pivot(index="player_id", columns="week", values="lam").reindex(
        index=keep, columns=weeks
    )
    raw_arr = raw.to_numpy(dtype=float)
    disc = np.array([discount ** max(w - current_week, 0) for w in weeks])
    values = np.where(np.isnan(raw_arr), FORBIDDEN, raw_arr * disc)
    values = np.where(raw_arr <= 0, FORBIDDEN, values)  # ruled out this week
    return players, values, raw_arr


def solve(values: np.ndarray, weeks: list[int]) -> tuple[dict[int, int], float]:
    if values.shape[0] == 0 or values.shape[1] == 0:
        return {}, 0.0
    rows, cols = linear_sum_assignment(values, maximize=True)
    assignment: dict[int, int] = {}
    total = 0.0
    for r, c in zip(rows, cols, strict=True):
        if values[r, c] > FORBIDDEN / 2:
            assignment[weeks[c]] = int(r)
            total += float(values[r, c])
    return assignment, total


def plan_slot(
    proj: pd.DataFrame,
    slot: str,
    current_week: int,
    used_ids: set[str],
    locked: dict[int, str] | None = None,
    last_week: int | None = None,
    **kwargs,
) -> SlotPlan:
    """Optimal remaining-season plan for one slot.

    `locked` maps week -> player_id for picks already recorded this season; those
    weeks are removed from the problem and those players count as used.
    """
    locked = locked or {}
    last_week = last_week or int(proj.week.max())
    weeks = [w for w in range(current_week, last_week + 1) if w not in locked]
    used = set(used_ids) | set(locked.values())
    players, values, raw = build_matrix(proj, slot, weeks, current_week, used, **kwargs)
    assignment, total = solve(values, weeks)
    return SlotPlan(slot, weeks, players, values, raw, assignment, total)


def forced_total(plan: SlotPlan, row: int, week: int) -> float:
    """Plan value if `row` is forced into `week` and the rest re-optimized."""
    col = plan.weeks.index(week)
    v = plan.values[row, col]
    if v <= FORBIDDEN / 2:
        return float("-inf")
    keep_rows = [r for r in range(plan.values.shape[0]) if r != row]
    keep_cols = [c for c in range(len(plan.weeks)) if c != col]
    sub = plan.values[np.ix_(keep_rows, keep_cols)]
    _, rest = solve(sub, [plan.weeks[c] for c in keep_cols])
    return float(v) + rest
