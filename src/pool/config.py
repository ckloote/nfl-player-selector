"""Tunable parameters and paths. Everything here is a modeling choice, not a fact."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

# --- Season / storage -------------------------------------------------------
DEFAULT_SEASON = int(os.environ.get("POOL_SEASON", "2026"))
DB_PATH = Path(os.environ.get("POOL_DB", "data/pool.db"))
TIMEZONE = "America/New_York"  # nflverse gametime is US Eastern
PICK_DEADLINE_MINUTES = 60  # picks lock this long before kickoff

# --- Slots ------------------------------------------------------------------
# Slot name -> positions eligible for that slot.
SLOTS: dict[str, tuple[str, ...]] = {
    "QB": ("QB",),
    "RB": ("RB",),
    "FLEX": ("WR", "TE"),
}
POSITIONS = tuple(p for ps in SLOTS.values() for p in ps)


def slot_for_position(position: str) -> str | None:
    for slot, positions in SLOTS.items():
        if position in positions:
            return slot
    return None


# --- Projection model -------------------------------------------------------
# Regress a player's prior-season TD rate toward the positional mean with this
# many pseudo-games. TD rates are noisy; regress hard.
PRIOR_SEASON_SHRINK_GAMES = 6.0
# Weight (in games) of the regressed prior when blending with current-season
# observations. By ~7 games this season, this year's data has equal say.
PRIOR_WEIGHT_GAMES = 7.0
# Players with no prior-season history (rookies, returns from injury) get this
# fraction of the positional "regular" mean as their prior.
NO_HISTORY_FACTOR = 0.6
# "Regulars" used to compute positional mean rates: min games and min usage
# per game (attempts for QB, carries+targets for RB, targets for WR/TE).
REGULAR_MIN_GAMES = 8
REGULAR_MIN_USAGE = {"QB": 15.0, "RB": 8.0, "WR": 4.0, "TE": 3.0}

# Opponent-defense multiplier: TDs allowed to a position group per game,
# regressed toward league average with this many pseudo-games. Prior-season
# defense data counts at this fraction of a current-season game.
DEF_SHRINK_GAMES = 8.0
DEF_PRIOR_SEASON_WEIGHT = 0.5

# Vegas: league-average implied team total used when no line is available.
FALLBACK_TEAM_TOTAL = 22.5
HOME_MULT = 1.03
AWAY_MULT = 0.97

# Availability multipliers by injury report status (current week only).
INJURY_MULT = {"Out": 0.0, "Doubtful": 0.0, "Questionable": 0.85}
# Roster statuses that count as available to play.
ACTIVE_ROSTER_STATUSES = {"ACT"}

# Role multiplier from depth-chart rank (1 = starter). Backups rarely see the
# field at QB; committee RBs and WR2/WR3 still score. Beyond the listed ranks
# the last value applies; players missing from the depth chart get DEPTH_DEFAULT.
DEPTH_MULT = {
    "QB": {1: 1.0, 2: 0.15, 3: 0.05},
    "RB": {1: 1.0, 2: 0.8, 3: 0.45, 4: 0.25},
    "WR": {1: 1.0, 2: 0.85, 3: 0.5},
    "TE": {1: 1.0, 2: 0.6, 3: 0.3},
}
DEPTH_DEFAULT_MULT = 0.5

# --- Optimizer --------------------------------------------------------------
# Per-week decay on future projected value (injury risk, role changes).
FUTURE_DISCOUNT = 0.985
# Only the top-N players per slot by baseline rate are considered; keeps the
# matrices small and output readable without affecting the answer in practice.
CANDIDATES_PER_SLOT = 80

# --- Recommendation layer ---------------------------------------------------
# Expected-TD edge an early-game (pre-Sunday) pick must have over the best
# later-game alternative to justify locking before injury news arrives.
INFO_PREMIUM_TD = 0.10
ALTERNATIVES_SHOWN = 6

# --- Role source ------------------------------------------------------------
# Where the role multiplier comes from: the depth-chart snapshot ("depth"),
# usage share to date ("usage"), or nothing at all ("none"). Live picks use the
# depth chart; backtests use usage, because `depth_charts` stores a single
# end-of-season snapshot that cannot be rewound to a past week.
ROLE_SOURCE = "depth"

# --- Diagnostics and inference ---------------------------------------------
# Predeclared minimum number of clusters for cluster-robust inference. Below it the
# sandwich estimator is not a usable inferential object — with one cluster the score
# sum is the gradient at the MLE, which is zero, so the "interval" has no width. Point
# estimates are still reported; the uncertainty is withheld. Chosen before looking at
# any stratified fit, so a sparse stratum cannot be promoted by choosing a lower bar.
MIN_INFERENCE_CLUSTERS = 30

# --- Backtesting ------------------------------------------------------------
# Closing-line horizon retained only for explicitly labeled legacy-closing sensitivity
# comparisons. Historical replay masks all current/future closing lines; live defaults
# continue to use the available feed. This constant is not an availability guarantee.
VEGAS_HORIZON_WEEKS = 6
# The random baseline picks uniformly among this many top players per slot.
RANDOM_TOP_N = 10
RANDOM_TRIALS = 20


@contextlib.contextmanager
def override(**values: object):
    """Temporarily rebind tunables in this module. Used by the backtest sweep.

    Unknown names raise: a typo'd parameter that silently changed nothing would
    make a sweep report a flat surface and look like a finding.
    """
    missing = [k for k in values if k not in globals()]
    if missing:
        raise KeyError(f"unknown config parameter(s): {sorted(missing)}")
    previous = {k: globals()[k] for k in values}
    globals().update(values)
    try:
        yield
    finally:
        globals().update(previous)


# Maximum live-feed ages in hours; historical completed seasons are exempt.
FRESHNESS_HOURS = {
    feed: float(os.environ.get(f"POOL_FRESHNESS_{feed.upper()}_HOURS", default))
    for feed, default in {
        "schedule": "1",
        "player_stats": "24",
        "rosters": "24",
        "injuries": "24",
        "depth_charts": "24",
        "touchdowns": "24",
    }.items()
}
