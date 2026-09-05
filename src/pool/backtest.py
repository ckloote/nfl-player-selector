"""Replay seasons with explicit input policies and independent no-reuse histories.

Historical replay uses prior-week closing lines, stats through W-1, reports through
W and usage roles. Final schedule revisions, report timing and stat corrections
remain approximations; timestamped observations support stricter snapshot replay.
Greedy, optimizer and random strategies consume model forecasts. Hindsight uses
finalized actuals as a retrospective ceiling. Operational picks are never read or written.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from statistics import pstdev

import numpy as np
import pandas as pd

from . import config, projections, scoring
from .optimizer import FORBIDDEN, build_matrix, plan_slot, solve

# Tunables that only affect the assignment, not the projection frame. Changing
# one of these during a sweep does not require rebuilding projections; changing
# anything else does.
PLAN_PARAMS = frozenset({"FUTURE_DISCOUNT", "CANDIDATES_PER_SLOT"})


@dataclass(frozen=True)
class Choice:
    player_id: str | None
    player_name: str | None
    team: str | None
    projected: float


@dataclass(frozen=True)
class Pick:
    week: int
    slot: str
    player_id: str | None
    player_name: str | None
    team: str | None
    projected: float
    actual: float


@dataclass
class Replay:
    season: int
    strategy: str
    picks: list[Pick] = field(default_factory=list)

    @property
    def total(self) -> float:
        return float(sum(p.actual for p in self.picks))

    @property
    def projected(self) -> float:
        return float(sum(p.projected for p in self.picks))

    @property
    def by_slot(self) -> dict[str, float]:
        return {
            slot: float(sum(p.actual for p in self.picks if p.slot == slot))
            for slot in config.SLOTS
        }

    @property
    def zero_picks(self) -> int:
        """Slot-weeks that scored nothing. Bust frequency matters in a
        winner-take-all pool independently of the mean."""
        return sum(1 for p in self.picks if p.player_id and not p.actual)

    @property
    def empty_slots(self) -> int:
        return sum(1 for p in self.picks if p.player_id is None)

    @property
    def players_used(self) -> int:
        return len({p.player_id for p in self.picks if p.player_id})


@dataclass
class Summary:
    """One row of the results table. `sd` is set only for repeated trials."""

    season: int
    strategy: str
    total: float
    by_slot: dict[str, float]
    projected: float
    zero_picks: int
    empty_slots: int
    players_used: int
    picks: list[Pick]
    sd: float | None = None
    trials: int = 1
    input_provenance: list[dict] = field(default_factory=list)

    @classmethod
    def of(cls, replay: Replay, **kw) -> Summary:
        return cls(
            season=replay.season,
            strategy=replay.strategy,
            total=replay.total,
            by_slot=replay.by_slot,
            projected=replay.projected,
            zero_picks=replay.zero_picks,
            empty_slots=replay.empty_slots,
            players_used=replay.players_used,
            picks=replay.picks,
            **kw,
        )


# --- inputs -----------------------------------------------------------------
def actual_tds(conn: sqlite3.Connection, season: int) -> dict[tuple[int, str], float]:
    """Complete weeks only; callers validate weeks before interpreting missing as zero."""
    weeks = scored_weeks(conn, season)
    totals = scoring.touchdown_totals(conn, season)
    out = {
        (int(w), pid): 0.0
        for w, pid in conn.execute(
            "SELECT week, player_id FROM player_weeks WHERE season = ?", (season,)
        )
        if w in weeks
    }
    out.update(
        {
            (int(r.week), r.player_id): float(r.pool_td)
            for r in totals.itertuples()
            if r.week in weeks
        }
    )
    return out


def scored_weeks(conn: sqlite3.Connection, season: int) -> list[int]:
    """Only weeks with complete scoring coverage for every scheduled game."""
    return scoring.complete_weeks(conn, season)


def weekly_projections(
    conn: sqlite3.Connection,
    season: int,
    weeks: Sequence[int],
    *,
    role_source: str | None = None,
    vegas_horizon: int | None = None,
    input_policy: str = "historical",
    decision_times: dict[tuple[int, int], str] | None = None,
    builder: Callable[..., pd.DataFrame] | None = None,
) -> dict[int, pd.DataFrame]:
    """One frozen projection frame per week — the expensive part of a replay.

    Independent of both the strategy and the assignment tunables, so a sweep
    over the future discount reuses these instead of rebuilding them.

    `builder` swaps in an alternative projection model with the signature of
    `projections.build_projections`; it must return the same frame contract
    (`projections.PROJECTION_COLUMNS`). Comparing a candidate model against the
    shipped one on identical frozen data is the whole point of the harness — see
    the experiment branches referenced in `docs/BACKTEST.md`.
    """
    role_source = projections.validate_policy(input_policy, role_source, vegas_horizon)
    loaded = weekly_inputs(
        conn,
        season,
        weeks,
        input_policy=input_policy,
        decision_times=decision_times,
        vegas_horizon=vegas_horizon,
    )
    build = projections.build_projections if builder is None else builder
    frames = {}
    for week, frame in loaded.items():
        result = build(frame, week, role_source=role_source)
        result.attrs["input_provenance"] = frame.provenance
        frames[week] = result
    return frames


def weekly_inputs(
    conn, season, weeks, *, input_policy="historical", decision_times=None, vegas_horizon=None
):
    """Load once per decision and share these inputs and shipped components across models."""
    projections.validate_policy(input_policy, vegas_horizon=vegas_horizon)
    if input_policy == "snapshots":
        missing = [(season, w) for w in weeks if (season, w) not in (decision_times or {})]
        if missing:
            raise ValueError(f"--decision-times is missing timestamps for {missing}")
    else:
        scoring.require_complete(conn, season - 1, projections.available_weeks(conn, season - 1))
        history = [w for w in projections.available_weeks(conn, season) if w < max(weeks)]
        if history:
            scoring.require_complete(conn, season, history)
    return {
        week: projections.load_frames(
            conn,
            season,
            as_of_week=week,
            vegas_horizon=vegas_horizon,
            input_policy=input_policy,
            decision_at=(decision_times or {}).get((season, week)),
        )
        for week in weeks
    }


# --- strategies -------------------------------------------------------------
EMPTY = Choice(None, None, None, 0.0)


def _row(players: pd.DataFrame, row: int, projected: float) -> Choice:
    p = players.iloc[row]
    return Choice(p.player_id, p.player_name, p.team, float(projected))


def _this_week(proj: pd.DataFrame, slot: str, week: int, used: set[str]):
    """Candidates playable this week, via the optimizer's own matrix builder.

    Going through `build_matrix` rather than filtering by hand means every
    strategy sees the same candidate pool and the same bye / ruled-out
    forbidding, so a win is a win about the decision rule.
    """
    players, values, _ = build_matrix(proj, slot, [week], week, used)
    if not len(players):
        return None, None, None
    column = values[:, 0]
    playable = np.flatnonzero(column > FORBIDDEN / 2)
    if not playable.size:
        return None, None, None
    return players, column, playable


def pick_optimizer(proj, slot, week, used, *, discount=None, **_) -> Choice:
    plan = plan_slot(proj, slot, week, used, discount=discount)
    row = plan.assignment.get(week)
    return EMPTY if row is None else _row(plan.players, row, plan.lam(row, week))


def pick_greedy(proj, slot, week, used, **_) -> Choice:
    got = _this_week(proj, slot, week, used)
    if got[0] is None:
        return EMPTY
    players, column, playable = got
    row = int(playable[np.argmax(column[playable])])
    return _row(players, row, column[row])


def pick_random(proj, slot, week, used, *, rng, top_n=None, **_) -> Choice:
    got = _this_week(proj, slot, week, used)
    if got[0] is None:
        return EMPTY
    players, column, playable = got
    top_n = config.RANDOM_TOP_N if top_n is None else top_n
    best = playable[np.argsort(-column[playable], kind="stable")][:top_n]
    row = int(rng.choice(best))
    return _row(players, row, column[row])


Chooser = Callable[..., Choice]
STRATEGIES: dict[str, Chooser] = {
    "optimizer": pick_optimizer,
    "greedy": pick_greedy,
    "random": pick_random,
}


# --- replay -----------------------------------------------------------------
def replay(
    frames: dict[int, pd.DataFrame],
    actuals: dict[tuple[int, str], float],
    season: int,
    strategy: str,
    *,
    weeks: Sequence[int] | None = None,
    **kw,
) -> Replay:
    """Walk the season once, one pick per slot per week, never reusing a player."""
    chooser = STRATEGIES[strategy]
    out = Replay(season=season, strategy=strategy)
    used: set[str] = set()
    for week in sorted(weeks if weeks is not None else frames):
        proj = frames[week]
        for slot in config.SLOTS:
            choice = chooser(proj, slot, week, used, **kw)
            if choice.player_id:
                used.add(choice.player_id)
            out.picks.append(
                Pick(
                    week=week,
                    slot=slot,
                    player_id=choice.player_id,
                    player_name=choice.player_name,
                    team=choice.team,
                    projected=choice.projected,
                    actual=actuals.get((week, choice.player_id), 0.0),
                )
            )
    return out


def hindsight(
    conn: sqlite3.Connection,
    season: int,
    weeks: Sequence[int],
    actuals: dict[tuple[int, str], float] | None = None,
) -> Replay:
    """The best possible season, chosen after the fact. A ceiling, not a rule.

    Candidates are everyone who scored at least once — nobody else can improve
    the total — and a week the player did not play is forbidden rather than
    zero, since you could not have fielded them.
    """
    actuals = actual_tds(conn, season) if actuals is None else actuals
    scoring.require_complete(conn, season, list(weeks))
    pw = scoring.pool_history(conn, season).rename(columns={"pool_td": "tds"})
    pw["slot"] = pw.position.map(config.slot_for_position)
    out = Replay(season=season, strategy="hindsight")
    weeks = list(weeks)
    for slot in config.SLOTS:
        sub = pw[(pw.slot == slot) & pw.week.isin(weeks)]
        scorers = sorted(sub.loc[sub.tds > 0, "player_id"].unique())
        if not scorers:
            out.picks.extend(Pick(w, slot, None, None, None, 0.0, 0.0) for w in weeks)
            continue
        grid = (
            sub[sub.player_id.isin(scorers)]
            .pivot_table(index="player_id", columns="week", values="tds", aggfunc="max")
            .reindex(index=scorers, columns=weeks)
        )
        scored = grid.to_numpy(dtype=float)
        values = np.where(np.isnan(scored), FORBIDDEN, scored)
        assignment, _ = solve(values, weeks)
        names = sub.drop_duplicates("player_id").set_index("player_id")
        for w in weeks:
            row = assignment.get(w)
            pid = scorers[row] if row is not None else None
            out.picks.append(
                Pick(
                    week=w,
                    slot=slot,
                    player_id=pid,
                    player_name=None if pid is None else names.player_name.get(pid),
                    team=None if pid is None else names.team.get(pid),
                    projected=0.0,
                    actual=0.0 if pid is None else actuals.get((w, pid), 0.0),
                )
            )
    return out


# --- orchestration ----------------------------------------------------------
def run_season(
    conn: sqlite3.Connection,
    season: int,
    strategies: Iterable[str],
    *,
    weeks: Sequence[int] | None = None,
    trials: int | None = None,
    seed: int = 0,
    discount: float | None = None,
    role_source: str | None = None,
    vegas_horizon: int | None = None,
    input_policy: str = "historical",
    decision_times: dict[tuple[int, int], str] | None = None,
    frames: dict[int, pd.DataFrame] | None = None,
    builder: Callable[..., pd.DataFrame] | None = None,
) -> list[Summary]:
    """Replay one season under each named strategy, sharing frozen projections."""
    strategies = list(strategies)
    weeks = list(weeks) if weeks is not None else projections.available_weeks(conn, season)
    scoring.require_complete(conn, season, weeks)
    actuals = actual_tds(conn, season)
    if frames is None:
        frames = weekly_projections(
            conn,
            season,
            weeks,
            role_source=role_source,
            vegas_horizon=vegas_horizon,
            input_policy=input_policy,
            decision_times=decision_times,
            builder=builder,
        )
    trials = config.RANDOM_TRIALS if trials is None else trials

    out: list[Summary] = []
    for name in strategies:
        if name == "hindsight":
            out.append(Summary.of(hindsight(conn, season, weeks, actuals)))
        elif name == "random":
            runs = [
                replay(
                    frames,
                    actuals,
                    season,
                    "random",
                    weeks=weeks,
                    rng=np.random.default_rng(seed + i),
                )
                for i in range(trials)
            ]
            totals = [r.total for r in runs]
            best = Summary.of(runs[0], sd=pstdev(totals) if len(totals) > 1 else 0.0, trials=trials)
            best.total = float(np.mean(totals))
            best.by_slot = {
                slot: float(np.mean([r.by_slot[slot] for r in runs])) for slot in config.SLOTS
            }
            best.zero_picks = int(round(float(np.mean([r.zero_picks for r in runs]))))
            out.append(best)
        else:
            out.append(
                Summary.of(replay(frames, actuals, season, name, weeks=weeks, discount=discount))
            )
    for row in out:
        row.input_provenance = projection_provenance(frames, season)
    return out


def projection_provenance(frames, season):
    return [
        dict(
            season=season,
            week=week,
            decision_at=frame.attrs.get("decision_at"),
            inputs=frame.attrs.get("input_provenance", []),
        )
        for week, frame in frames.items()
    ]


def sweep(
    conn: sqlite3.Connection,
    seasons: Sequence[int],
    discounts: Sequence[float],
    prior_weights: Sequence[float],
    *,
    role_source: str | None = None,
    vegas_horizon: int | None = None,
    input_policy: str = "historical",
    decision_times: dict[tuple[int, int], str] | None = None,
    log: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Grid-search the future discount and prior weight, one row per cell.

    Projections depend on the prior weight but not the discount, so the frames
    are rebuilt once per (season, prior weight) and reused across the discount
    axis. Greedy is scored once per those frames too — it ignores the discount.
    """
    rows, provenance = [], {}
    for season in seasons:
        weeks = scored_weeks(conn, season)
        actuals = actual_tds(conn, season)
        for pw in prior_weights:
            with config.override(PRIOR_WEIGHT_GAMES=float(pw)):
                frames = weekly_projections(
                    conn,
                    season,
                    weeks,
                    role_source=role_source,
                    vegas_horizon=vegas_horizon,
                    input_policy=input_policy,
                    decision_times=decision_times,
                )
                provenance[season] = projection_provenance(frames, season)
                greedy = replay(frames, actuals, season, "greedy", weeks=weeks).total
                for d in discounts:
                    total = replay(
                        frames, actuals, season, "optimizer", weeks=weeks, discount=float(d)
                    ).total
                    rows.append(
                        {
                            "season": season,
                            "prior_weight": float(pw),
                            "discount": float(d),
                            "optimizer": total,
                            "greedy": greedy,
                            "delta": total - greedy,
                        }
                    )
            if log:
                log(f"  {season} prior_weight={pw}: greedy {greedy:.0f}")
    out = pd.DataFrame(rows)
    out.attrs["input_provenance"] = [r for records in provenance.values() for r in records]
    return out
