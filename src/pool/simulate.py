"""Joint player-week touchdown outcomes, and the pot share a set of picks earns.

Pure: arrays and frames in, arrays out. No database, no clock, no I/O, for the reason
`rivals` has none -- this module is bound for the enforced decision closure.

The structure is measured rather than assumed. Against 2011-2025 complete coverage:

    corr(home credits, away credits) within a game        +0.110   (13 of 15 seasons)
    corr(QB, lead pass-catcher)      same team            +0.465
    corr(QB, lead RB)                same team            +0.013
    corr(lead RB, lead pass-catcher) same team            -0.036
    share of QB pool credits that are `throwing`           90.6%

Two things follow. A shared *game* factor is real, so one is drawn per game and both teams
ride it. A shared *team* factor is not: same-team non-QB pairs are no more correlated than
cross-team pairs, and running backs and receivers are mildly negative, because a team's
red-zone chances are finite and its skill players compete for them. A positive team factor
would have inflated exactly the correlations that are zero or negative.

What the +0.465 is instead is not correlation at all but **shared credit**. A credited
passing touchdown pays two players, the passer and the scorer, so a quarterback's score is
very nearly his receivers' scores -- 90.6% of it. That is modelled as the identity it is:
a pass-catcher's receiving touchdowns are sampled once and added to his quarterback.

Every player's expectation is preserved exactly. `E[Y_i] = lam_i` for everybody including
quarterbacks, so the simulator puts a joint distribution around the shipped point forecasts
without moving any of them. Phase 3B's no-promotion outcome stands.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Provisional until the calibration run fits them; every value here is measured or declared,
# never tuned to make an answer come out. `RECEIVE_SHARE` is the fraction of a non-quarter-
# back's scores that his passer is also credited for -- a rushing touchdown links to nobody.
THROW_SHARE = 0.906
RECEIVE_SHARE = {"QB": 0.0, "RB": 0.207, "WR": 0.975, "TE": 0.980}
K_GAME = 3.0


@dataclass(frozen=True)
class Params:
    k_game: float = K_GAME
    throw_share: float = THROW_SHARE
    receive_share: dict[str, float] = field(default_factory=lambda: dict(RECEIVE_SHARE))

    def receiving(self, position: str) -> float:
        return self.receive_share.get(position, 0.0)


@dataclass(frozen=True)
class Draws:
    """`values[cell, sim]`: one sampled touchdown count per player-week, per simulation.

    One row per player-week, shared by every entrant who picked that player. Two entrants
    on the same player must move together; sampling them apart would decide a pool on a
    coin flip that does not exist.
    """

    values: np.ndarray
    index: dict[tuple[int, str], int]
    sims: int
    rescaled: int = 0

    def totals(self, picks) -> np.ndarray:
        """Season total across `picks`, as (sims,). An unknown player-week contributes 0."""
        rows = [self.index[key] for key in picks if key in self.index]
        if not rows:
            return np.zeros(self.sims, dtype=np.int32)
        return self.values[rows].sum(axis=0)


def _cells(proj: pd.DataFrame, weeks) -> pd.DataFrame:
    frame = proj[proj.week.isin(list(weeks))]
    frame = frame.drop_duplicates(["week", "player_id"])
    return frame.reset_index(drop=True)


def sample(proj: pd.DataFrame, weeks, *, sims: int, seed: int, params: Params | None = None):
    """Draw `sims` joint seasons over `weeks`.

    One Gamma factor per game, both teams on it. A pass-catcher's receiving scores are
    sampled once and credited to his quarterback as well as to him; his other scores, and
    the quarterback's own rushing, are independent given the game factor.
    """
    params = params or Params()
    cells = _cells(proj, weeks)
    n = len(cells)
    rng = np.random.default_rng(seed)
    index = {(int(r.week), str(r.player_id)): i for i, r in enumerate(cells.itertuples())}
    if not n:
        return Draws(np.zeros((0, sims), dtype=np.int32), index, sims)

    lam = cells.lam.to_numpy(dtype=float)
    lam = np.where(np.isfinite(lam) & (lam > 0), lam, 0.0)
    position = cells.position.to_numpy()
    is_qb = position == "QB"
    rho = np.array([params.receiving(p) for p in position])

    game_key = cells.week.astype(str) + "|" + cells.game_id.astype(str)
    game_of = pd.factorize(game_key, sort=True)[0]
    team_key = cells.week.astype(str) + "|" + cells.team.astype(str)
    team_of = pd.factorize(team_key, sort=True)[0]

    # A team's pass-catchers link to its highest-rate quarterback. A backup carries his own
    # throwing rate independently: crediting one team's receiving scores to two passers
    # would invent touchdowns that were thrown once.
    starter = np.full(team_of.max() + 1, -1)
    for team in range(team_of.max() + 1):
        qbs = np.flatnonzero((team_of == team) & is_qb)
        if qbs.size:
            starter[team] = int(qbs[np.argmax(lam[qbs])])
    linked = np.array([starter[t] if not q else -1 for t, q in zip(team_of, is_qb, strict=True)])

    # Scale receiving down only where a team's catchers out-project its quarterback's arm.
    # Scaling the non-receiving part up by the same amount keeps E[Y_j] = lam_j exactly, so
    # the identity survives the guard rather than being clamped away.
    recv = lam * rho
    scale = np.ones(n)
    rescaled = 0
    for team, qb in enumerate(starter):
        if qb < 0:
            continue
        members = (team_of == team) & (linked >= 0)
        supply, arm = recv[members].sum(), params.throw_share * lam[qb]
        if supply > arm > 0:
            scale[members] = arm / supply
            rescaled += 1
        elif supply > 0 and arm <= 0:
            scale[members] = 0.0
            rescaled += 1
    recv = recv * scale
    other = lam - recv
    other = np.where(is_qb, (1 - params.throw_share) * lam, other)

    # Every quarterback keeps his whole throwing rate; only the route differs. A starter
    # takes his team's sampled receiving scores plus a residual for catchers the pool does
    # not model. A backup takes his independently, because his team's receiving touchdowns
    # were thrown once and crediting them twice would invent them.
    residual = np.zeros(n)
    for qb in np.flatnonzero(is_qb):
        team = team_of[qb]
        if starter[team] == qb:
            members = (team_of == team) & (linked >= 0)
            residual[qb] = max(0.0, params.throw_share * lam[qb] - recv[members].sum())
        else:
            residual[qb] = params.throw_share * lam[qb]

    factor = rng.gamma(params.k_game, 1.0 / params.k_game, size=(game_of.max() + 1, sims))
    spread = factor[game_of]
    receiving = rng.poisson(recv[:, None] * spread).astype(np.int32)
    values = receiving + rng.poisson(other[:, None] * spread).astype(np.int32)
    values += rng.poisson(residual[:, None] * spread).astype(np.int32)
    for cell in np.flatnonzero(linked >= 0):
        values[linked[cell]] += receiving[cell]
    return Draws(values, index, sims, rescaled)


def shares(mine: np.ndarray, rivals: np.ndarray) -> np.ndarray:
    """Share of the pot per simulation: 1 outright, 1/k in a k-way tie, 0 otherwise.

    A tie at the top splits the winnings, so this is the quantity the policy maximises and
    not the probability of an outright win. Season totals are small integers and ties are
    common; a rule that ignored them would systematically undervalue positions that reliably
    draw level.
    """
    if rivals.size == 0:
        return np.ones(len(mine), dtype=float)
    top = rivals.max(axis=1)
    level = (rivals == top[:, None]).sum(axis=1)
    return np.where(mine > top, 1.0, np.where(mine == top, 1.0 / (level + 1), 0.0))


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mean difference and its standard error, on shared draws.

    Candidates are compared on identical outcomes, so the rivals' maximum is the same in
    both arms and the difference is estimated far more precisely than either share. Two
    candidates whose difference does not clear this error are not ordered -- without that
    the tool would reorder picks on Monte Carlo noise every week, and the simulation count
    would quietly become a decision input.
    """
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(difference) < 2:
        return float(difference.sum()), float("inf")
    return float(difference.mean()), float(difference.std(ddof=1) / np.sqrt(len(difference)))
