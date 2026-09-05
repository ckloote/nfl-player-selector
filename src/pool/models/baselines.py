"""Benchmark projection models: Models 0-6 of the projection benchmark plan.

Every model is built by taking the shipped projection frame for its scaffolding
— candidate pool, schedule, bye weeks, injury availability — and replacing only
`lam`. Sharing the scaffolding is the point: the benchmark compares forecasts,
so a difference in who is eligible would confound it.

`hard_eligible` excludes ruled-out cells independently of lambda. `avail_mult`
retains soft Questionable adjustments; shuffled models permute availability-free
forecasts over eligible cells and apply the recipient adjustment afterward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config, projections
from ..projections import Frames

# Sorting by player_id last makes tie order reproducible run to run, which the
# shuffled nulls need and the degenerate environment model (identical lam for
# every player in a team-week) depends on entirely.
SORT_KEYS = ["week", "slot", "lam", "player_id"]
SORT_ASC = [True, True, False, True]


def _scaffold(frames: Frames, from_week: int, role_source: str | None) -> pd.DataFrame:
    if frames.scaffold is None:
        frames.scaffold = projections.build_projections(frames, from_week, role_source=role_source)
    return frames.scaffold.copy()


def _with_lam(proj: pd.DataFrame, lam) -> pd.DataFrame:
    out = proj.copy()
    out["lam"] = np.asarray(lam, dtype=float)
    return out.sort_values(SORT_KEYS, ascending=SORT_ASC).reset_index(drop=True)


def _pos_mean(proj: pd.DataFrame, frames: Frames) -> pd.Series:
    """Positional 'regular' mean TD rate — the shrinkage target, from the prior season."""
    if frames.cached_pos_means is None:
        frames.cached_pos_means = projections.positional_means(frames.pw_prior)
    means = frames.cached_pos_means
    return proj.position.map(means).fillna(0.0).astype(float)


def _shrunk(tds: pd.Series, games: pd.Series, mean: pd.Series, k: float) -> np.ndarray:
    """(tds + k*mean) / (games + k) — the model's own shrinkage, applied to one season."""
    return ((tds + k * mean) / (games + k)).to_numpy(dtype=float)


# --- Model 0a/0b: nulls -----------------------------------------------------
def shuffle_within_slot_week(seed: int):
    """Model 0a. Permute availability-free rates within each eligible (week, slot).

    Eligibility and recipient availability remain attached to their original cells.
    FLEX combines WR and TE. Seeds describe perturbations of the same outcomes,
    not independent seasons or an information-free prediction experiment.
    """

    def build(frames: Frames, from_week: int = 1, role_source: str | None = None) -> pd.DataFrame:
        proj = _scaffold(frames, from_week, role_source)
        proj = proj.sort_values(SORT_KEYS, ascending=SORT_ASC).reset_index(drop=True)
        rng = np.random.default_rng(seed)
        lam = proj.lam.to_numpy(dtype=float).copy()
        for _, group in proj[proj.hard_eligible].groupby(["week", "slot"], sort=True):
            idx = group.index.to_numpy()
            lam[idx] = (
                rng.permutation(lam[idx] / proj.avail_mult.iloc[idx]) * proj.avail_mult.iloc[idx]
            )
        return _with_lam(proj, lam)

    build.__name__ = f"shuffle_within_slot_week_{seed}"
    return build


def shuffle_within_player(seed: int):
    """Model 0b. Permute a player's eligible remaining-week availability-free rates.

    Each decision rebuilds and perturbs its own remaining-season surface. Rate and
    role updates across decisions survive, as do eligibility and recipient
    availability. This is not a permutation of the consumed decision-week sequence
    or an identity-versus-timing decomposition of season performance.
    """

    def build(frames: Frames, from_week: int = 1, role_source: str | None = None) -> pd.DataFrame:
        proj = _scaffold(frames, from_week, role_source)
        proj = proj.sort_values(SORT_KEYS, ascending=SORT_ASC).reset_index(drop=True)
        rng = np.random.default_rng(seed)
        lam = proj.lam.to_numpy(dtype=float).copy()
        for _, group in proj[proj.hard_eligible].groupby("player_id", sort=True):
            idx = group.index.to_numpy()
            lam[idx] = (
                rng.permutation(lam[idx] / proj.avail_mult.iloc[idx]) * proj.avail_mult.iloc[idx]
            )
        return _with_lam(proj, lam)

    build.__name__ = f"shuffle_within_player_{seed}"
    return build


# --- Model 1-5: baselines ---------------------------------------------------
def historical_rate(
    frames: Frames, from_week: int = 1, role_source: str | None = None
) -> pd.DataFrame:
    """Model 1. Prior-season TD/game, unregressed. The simplest player-identity model."""
    proj = _scaffold(frames, from_week, role_source)
    games = proj.prior_games.to_numpy(dtype=float)
    rate = np.divide(
        proj.prior_tds.to_numpy(dtype=float), games, out=np.zeros(len(proj)), where=games > 0
    )
    return _with_lam(proj, rate * proj.avail_mult)


def regressed_rate(
    frames: Frames, from_week: int = 1, role_source: str | None = None
) -> pd.DataFrame:
    """Model 2. Model 1 regressed toward the positional mean; no context multipliers.

    This is the shipped model's `prior_rate` term standing alone, so the gap to
    Model 6 is exactly what the current-season blend and the multiplier stack add.
    """
    proj = _scaffold(frames, from_week, role_source)
    mean = _pos_mean(proj, frames)
    rate = _shrunk(proj.prior_tds, proj.prior_games, mean, config.PRIOR_SEASON_SHRINK_GAMES)
    no_history = proj.prior_games.to_numpy(dtype=float) <= 0
    rate = np.where(no_history, config.NO_HISTORY_FACTOR * mean, rate)
    return _with_lam(proj, rate * proj.avail_mult)


def current_season_rate(
    frames: Frames, from_week: int = 1, role_source: str | None = None
) -> pd.DataFrame:
    """Model 3. Current season only, shrunk toward the positional mean.

    Tests whether the prior-season term earns its place: in week 1 this model is
    pure positional mean and knows nothing about the player at all.
    """
    proj = _scaffold(frames, from_week, role_source)
    mean = _pos_mean(proj, frames)
    rate = _shrunk(proj.cur_tds, proj.cur_games, mean, config.PRIOR_SEASON_SHRINK_GAMES)
    return _with_lam(proj, rate * proj.avail_mult)


def vegas_environment(
    frames: Frames, from_week: int = 1, role_source: str | None = None
) -> pd.DataFrame:
    """Model 4. Scoring environment only: no player TD history whatsoever.

    Expected TD opportunity is allocated by position and depth role, then scaled
    by the Vegas implied team total. Role is retained because without it a backup
    QB ties his starter and the model is not a serious benchmark; the point is to
    isolate how far team environment alone gets you, not to remove all structure.
    """
    proj = _scaffold(frames, from_week, role_source)
    lam = _pos_mean(proj, frames) * proj.role_mult * proj.vegas_mult * proj.avail_mult
    return _with_lam(proj, lam)


def player_plus_vegas(
    frames: Frames, from_week: int = 1, role_source: str | None = None
) -> pd.DataFrame:
    """Model 5. base_rate x vegas_mult — the shipped model with defense, home/away
    and role neutralised.

    The load-bearing comparison: once player history and team scoring environment
    are known, does the rest of the multiplier stack earn its complexity?
    """
    proj = _scaffold(frames, from_week, role_source)
    return _with_lam(proj, proj.base_rate * proj.vegas_mult * proj.avail_mult)


# --- ablations of the shipped model (instrument validation) -----------------
def no_vegas(frames: Frames, from_week: int = 1, role_source: str | None = None) -> pd.DataFrame:
    """Shipped model with vegas_mult neutralised.

    Recomposed from the components rather than divided out of `lam`, so a
    zero or missing multiplier cannot turn into an infinity.
    """
    proj = _scaffold(frames, from_week, role_source)
    lam = proj.base_rate * proj.def_mult * proj.home_mult * proj.avail_mult * proj.role_mult
    return _with_lam(proj, lam)


def base_rate_only(
    frames: Frames, from_week: int = 1, role_source: str | None = None
) -> pd.DataFrame:
    """Shipped base rate with no matchup context at all."""
    proj = _scaffold(frames, from_week, role_source)
    return _with_lam(proj, proj.base_rate * proj.avail_mult)
