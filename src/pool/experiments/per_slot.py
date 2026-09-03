"""A different base rate per slot: opportunity for WR/TE, touchdown history for QB.

Touchdown rate is a noisy estimator, and how noisy depends on the position. A
quarterback throws two a game, so his history says something. A receiver catches
six or eight a season, so ranking receivers by prior touchdown rate largely
selects last season's lucky finishers, who then regress. Targets per game are
far more stable. This model therefore uses:

    QB    touchdown history      (the shipped rate)
    RB    geometric blend of the two
    FLEX  opportunity only       (touches/game x league TDs-per-touch)

everything else - opponent defence, Vegas, home, availability, role - unchanged.

RESULT: this is a cautionary tale, and worth reading before trusting any small
win from the backtest harness.

Searching ~12 variants against 2017-2025 produced what looked like a strong
finding. The FLEX slot scored worse than random (-1.09 TD/season, 2/9 seasons),
its top-ranked pick lost to a coin flip among its own top 3, and this model was
worth +3.67 TD/season (SE 1.59, 7/9 seasons). It came with a plausible
mechanism, and the anti-correlation at FLEX rank 1 predicted it before it was
tested.

It did not replicate. Backfilling 2011-2016 as a genuine holdout, with the spec
frozen beforehand:

                             2017-2025 (searched)   2011-2016 (holdout)   all 15
    per-slot - current       +3.67 (SE 1.59) 7/9    +0.17 (SE 3.03) 3/6   +2.27
    ... at FLEX              +3.11 (SE 1.24) 7/9    +0.67 (SE 1.69) 3/6   +2.13

The underlying defect did not replicate either: on the holdout the shipped
model's FLEX beat random by +1.77. The unbiased estimate is the holdout column,
and it is about zero. Not merged.

What is arguably still interesting: the opportunity-based FLEX score is stable
across eras (10.11 on the searched seasons, 10.50 on the holdout) while the
shipped TD-history version swings (7.00 then 9.83). That is a variance claim
rather than a mean-improvement claim, and 15 seasons cannot settle it - see
docs/BACKTEST.md section 6 on what this harness can resolve.

    uv run pool backtest --season 2011-2025 --projection per-slot
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from .. import projections as P

CONTEXT = ["def_mult", "vegas_mult", "home_mult", "avail_mult", "role_mult"]


def opportunity_rate(frames: P.Frames, pool: pd.DataFrame) -> pd.Series:
    """Per-player TD rate implied by usage alone.

    Touches per game, blended prior-to-current with the model's own shrinkage,
    times the league's touchdowns per touch for that position.
    """
    prior = P.per_player_totals(frames.pw_prior)
    cur = P.per_player_totals(frames.pw_cur)
    if not len(prior):
        return pd.Series(0.0, index=pool.player_id.to_numpy())

    per_touch, mean_pg = {}, {}
    for pos in config.POSITIONS:
        sub = prior[prior.position == pos]
        per_touch[pos] = float(sub.tds.sum() / sub.usage.sum()) if sub.usage.sum() else 0.0
        mean_pg[pos] = float((sub.usage / sub.games).mean()) if len(sub) else 0.0

    zero = pd.Series(0.0, index=pool.index)
    pg = pool.player_id.map(prior.games).fillna(0.0)
    pu = pool.player_id.map(prior.usage).fillna(0.0)
    cg = pool.player_id.map(cur.games).fillna(0.0) if len(cur) else zero
    cu = pool.player_id.map(cur.usage).fillna(0.0) if len(cur) else zero

    k, w = config.PRIOR_SEASON_SHRINK_GAMES, config.PRIOR_WEIGHT_GAMES
    regressed = (pu + k * pool.position.map(mean_pg).fillna(0.0)) / (pg + k)
    blended = (w * regressed + cu) / (w + cg)
    rate = blended * pool.position.map(per_touch).fillna(0.0)
    return pd.Series(rate.to_numpy(), index=pool.player_id.to_numpy())


def build(frames: P.Frames, from_week: int = 1, role_source: str | None = None) -> pd.DataFrame:
    """The shipped frame, with `lam` recomputed from a per-slot base rate."""
    shipped = P.build_projections(frames, from_week, role_source=role_source)
    pool = P.player_pool(frames.rosters, frames.pw_prior, frames.pw_cur)
    pool = pool[pool.position.isin(config.POSITIONS)]

    context = np.prod([shipped[c].astype(float).to_numpy() for c in CONTEXT], axis=0)
    by_touchdowns = shipped.lam.astype(float).to_numpy()
    by_opportunity = (
        shipped.player_id.map(opportunity_rate(frames, pool)).fillna(0.0).to_numpy(float) * context
    )
    blend = np.sqrt(np.clip(by_touchdowns, 0, None) * np.clip(by_opportunity, 0, None))

    slot = shipped.slot.to_numpy()
    lam = np.where(slot == "QB", by_touchdowns, np.where(slot == "RB", blend, by_opportunity))
    return shipped.assign(lam=lam)
