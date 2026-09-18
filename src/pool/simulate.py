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

# Fitted and measured on 2011-2025 by `experiments/simulator_calibration.py`; see
# `experiments/results/phase4-simulator/`. Frozen deliberately: a decision whose sampling
# distribution moves with the data underneath it cannot be reconstructed, and
# `capture.reconstruct` would have nothing to restore.
#
# `THROW_SHARE` and `RECEIVE_SHARE` are *measured*, straight off `touchdown_credits.kind`
# and the offensive splits -- the fraction of a quarterback's credits that come from
# throwing, and the fraction of everyone else's that his passer is also paid for. A rushing
# touchdown links to nobody, which is why a running back sits at 0.207 and a receiver at
# 0.975. `K_GAME` is *fitted*, from the covariance between opposing teams in one game.
THROW_SHARE = 0.9064
RECEIVE_SHARE = {"QB": 0.0, "RB": 0.207, "WR": 0.975, "TE": 0.980}
K_GAME = 14.302


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

    def totals(self, picks, finalized=()) -> np.ndarray:
        """Sum unique picks, replacing sampled cells with known final outcomes."""
        fixed = {(int(w), str(pid)): int(tds) for w, pid, tds in finalized}
        keys = dict.fromkeys(picks)
        banked = sum(fixed[key] for key in keys if key in fixed)
        rows = [self.index[key] for key in keys if key in self.index and key not in fixed]
        total = self.values[rows].sum(axis=0) if rows else np.zeros(self.sims, dtype=np.int32)
        return total + banked


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
    # the identity survives the guard instead of being clamped away. It fires often -- in
    # roughly half of real team-weeks the projections put the two within a rounding of each
    # other -- which is why it has to preserve the expectation rather than merely cap.
    recv = lam * rho
    scale = np.ones(n)
    rescaled = 0
    for team, qb in enumerate(starter):
        if qb < 0:
            continue
        members = (team_of == team) & (linked >= 0)
        supply, arm = recv[members].sum(), params.throw_share * lam[qb]
        if supply > arm:
            scale[members] = (arm / supply) if supply > 0 else 0.0
            rescaled += 1
    recv = recv * scale
    other = np.where(is_qb, (1 - params.throw_share) * lam, lam - recv)

    factor = rng.gamma(params.k_game, 1.0 / params.k_game, size=(game_of.max() + 1, sims))
    spread = factor[game_of]
    values = rng.poisson(other[:, None] * spread).astype(np.int32)

    # A team's passing touchdowns are drawn once and *allocated* among its receivers, not
    # sampled per receiver. That is what the pool's own arithmetic does -- one pass pays the
    # catcher and the passer -- and independent per-receiver draws get it measurably wrong in
    # both directions: they leave receivers positively correlated when real ones compete for
    # a finite number of throws (observed -0.029, independent draws +0.051), and through that
    # they overstate the quarterback-to-receiver link by about forty percent (observed +0.358,
    # independent draws +0.502). The multinomial gives the substitution for free, from the
    # mechanism, rather than as a correction bolted on afterwards.
    for team, qb in enumerate(starter):
        if qb < 0:
            continue
        catchers = np.flatnonzero((team_of == team) & (linked >= 0) & (recv > 0))
        arm = params.throw_share * lam[qb]
        weights = np.append(recv[catchers], max(0.0, arm - recv[catchers].sum()))
        if weights.sum() <= 0:
            continue
        thrown = rng.poisson(weights.sum() * factor[game_of[qb]])
        caught = rng.multinomial(thrown, weights / weights.sum())
        values[qb] += thrown.astype(np.int32)  # every throw pays the passer
        if catchers.size:
            values[catchers] += caught[:, : catchers.size].T.astype(np.int32)

    # A team with no quarterback in the pool still scores its own rate, and a backup keeps
    # his whole arm independently: his team's receiving touchdowns were thrown once, and
    # crediting two passers for them would invent touchdowns.
    loose = np.flatnonzero((recv > 0) & (linked < 0) & ~is_qb)
    values[loose] += rng.poisson(recv[loose][:, None] * spread[loose]).astype(np.int32)
    backups = np.flatnonzero(is_qb & (np.arange(n) != starter[team_of]))
    values[backups] += rng.poisson(
        params.throw_share * lam[backups][:, None] * spread[backups]
    ).astype(np.int32)
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


def clustered_se(values: np.ndarray, edges) -> float:
    """Cluster-robust SE for draws sharing a rival path within each scenario block."""
    values = np.asarray(values, dtype=float)
    groups = len(edges) - 1
    if groups < 2:
        return float("inf")
    centered = values - values.mean()
    sums = np.array([
        centered[low:high].sum() for low, high in zip(edges[:-1], edges[1:], strict=True)
    ])
    return float(np.sqrt(groups / (groups - 1) * np.dot(sums, sums) / len(values)**2))


def paired(a: np.ndarray, b: np.ndarray, edges=None) -> tuple[float, float]:
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
    if edges is not None:
        return float(difference.mean()), clustered_se(difference, edges)
    return float(difference.mean()), float(difference.std(ddof=1) / np.sqrt(len(difference)))
