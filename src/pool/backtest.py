"""Replay seasons with explicit input policies and independent no-reuse histories.

Historical replay uses prior-week closing lines, stats through W-1, reports through
W and usage roles. Final schedule revisions, report timing and stat corrections
remain approximations; timestamped observations support stricter snapshot replay.
Greedy, optimizer and random strategies consume model forecasts. Hindsight uses
finalized actuals as a retrospective ceiling. Operational picks are never read or written.

The `winprob` strategy is different in kind and is labelled as such everywhere it appears.
Seasons before the pool's reports began have no opponent picks, so a replay of one has to
invent its rivals -- "a test of the machinery, not of the idea", in the phase plan's own
words. Nothing here can turn that into evidence about people; it can only show that the
policy runs a season end to end without spending a player twice.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import pstdev

import numpy as np
import pandas as pd

from . import config, projections, recommend, rivals, scoring
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
    # What the season was replayed against, if anything, and where it left everyone. Both
    # are None/empty for an ordinary replay, which is played against nobody: the total is
    # just touchdowns. They are set when a `Field` was threaded through, and then the
    # total on its own is the wrong axis to read `winprob` on -- a policy that trades a
    # touchdown for a better share of the pot is doing its job, and will look worse here.
    against: str | None = None
    standing: tuple[tuple[str, float], ...] = ()

    @property
    def total(self) -> float:
        return float(sum(p.actual for p in self.picks))

    @property
    def rank(self) -> int | None:
        """Where my total finished among the invented rivals, 1-based, ties shared."""
        return _placing(self.total, self.standing)[0]

    @property
    def level(self) -> int:
        """How many of them I finished exactly level with -- a shared pot, not a win."""
        return _placing(self.total, self.standing)[1]

    @property
    def finish(self) -> str | None:
        return _finish(self.total, self.standing)

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
    against: str | None = None
    standing: tuple[tuple[str, float], ...] = ()

    @property
    def finish(self) -> str | None:
        """Derived rather than copied, because `run_season` overwrites `total` with a mean
        over trials for the random baseline. A stored finish would still describe trial
        zero, and the two columns of one row would be talking about different seasons."""
        return _finish(self.total, self.standing)

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
            against=replay.against,
            standing=replay.standing,
            **kw,
        )


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _placing(total: float, standing) -> tuple[int | None, int]:
    """Competition rank against a field, and how many of it I am exactly level with."""
    if not standing:
        return None, 0
    return (
        1 + sum(1 for _, tds in standing if tds > total),
        sum(1 for _, tds in standing if tds == total),
    )


def _finish(total: float, standing) -> str | None:
    """"2nd of 5", or "1st of 5, sharing with 2", or None against nobody.

    Never "won". A tie at the top splits the pot, and a line that called that a win would
    be the mistake `simulate.shares` exists to avoid, printed instead of simulated.
    """
    rank, level = _placing(total, standing)
    if rank is None:
        return None
    out = f"{_ordinal(rank)} of {len(standing) + 1}"
    return out if not level else f"{out}, sharing with {level}"


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
    the experiment branches referenced in `docs/archive/BACKTEST.md`.
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


# --- invented rivals --------------------------------------------------------
# Quoted rather than paraphrased, because this is the sentence the number has to travel
# with. From `docs/PHASE4_PLAN.md`: "seasons before ingestion have no opponent picks, so a
# backtest would be measuring the policy against invented rivals -- a test of the
# machinery, not of the idea."
INVENTED = (
    "No opponent picks exist for a finished season, so this field was invented -- "
    '"a test of the machinery, not of the idea", in the phase plan\'s own words'
)


@dataclass
class Opponent:
    """One invented rival, mid-season: what they have spent and what it has scored them."""

    display_name: str
    used: set[str] = field(default_factory=set)
    tds: float = 0.0
    picks: list[tuple[int, str, str | None]] = field(default_factory=list)


@dataclass(frozen=True)
class Field:
    """The specification of an invented opposition. A description, not a run of one.

    Nobody in here is real and no arrangement of the knobs can make them so. Worse than
    that: the default `behaviour` is the *same* greedy-with-noise hypothesis
    `recommend.pot_shares` assumes when it values a candidate, so a `winprob` replay under
    it is being scored against an opponent model that is true by construction. That is the
    policy's best case and it is not evidence about the five people in the pool.

    `behaviour` is a key of `rivals.PREDICTORS`, so the mis-specified case can be run too:
    under `naive` or `optimizer` the policy is exploiting a model that is wrong in a named
    way, and the gap between the two runs is the only thing here worth reading.

    Frozen and separate from `Opponents` so that one specification can be started fresh for
    each strategy in a comparison. The invented rivals never see my picks -- this pool lets
    two entrants hold the same player -- so the same seed gives every strategy the
    identical invented season, and the finishes are comparable.
    """

    count: int = 4
    behaviour: str = "greedy"
    top_n: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.behaviour not in rivals.PREDICTORS:
            raise ValueError(
                f"Unknown rival behaviour {self.behaviour!r}; "
                f"choose from {sorted(rivals.PREDICTORS)}."
            )
        if self.count < 1:
            raise ValueError("A field needs at least one invented rival.")

    @property
    def noise(self) -> int:
        return config.RIVAL_NOISE_TOP_N if self.top_n is None else self.top_n

    @property
    def label(self) -> str:
        """Who they are, in one clause -- short enough to head a table, and carrying the
        word that matters. `INVENTED` is the paragraph it travels with; the two are kept
        apart so a picks table can be headed without reprinting the argument under it."""
        among = "always their best" if self.noise < 2 else f"among their top {self.noise}"
        return (
            f"{self.count} invented rivals picking {self.behaviour}, {among} "
            f"(seed {self.seed})"
        )

    def start(self) -> Opponents:
        """A fresh run of this field, at week one of a season, having spent nothing."""
        return Opponents(
            self,
            [Opponent(f"Invented {chr(ord('A') + i)}") for i in range(self.count)],
            np.random.default_rng(self.seed),
        )


@dataclass
class Opponents:
    """One run of a `Field` through a season, advanced a week at a time beside my own."""

    spec: Field
    entrants: list[Opponent]
    rng: np.random.Generator

    def state(self, my_tds: float) -> rivals.PoolState:
        """The pool as the policy needs to see it, at the start of a week.

        Complete by construction: an invented rival has no unreported weeks and no name
        that failed to resolve, so `pool_complete` is true and the guard `recommend` prints
        for a real pool never fires here. That is another way of saying this is easier than
        the live problem.
        """
        return rivals.PoolState(
            tuple(
                rivals.RivalState(
                    f"invented-{i}",
                    o.display_name,
                    frozenset(o.used),
                    season_tds=int(round(o.tds)),
                )
                for i, o in enumerate(self.entrants)
            ),
            int(round(my_tds)),
        )

    def advance(self, proj: pd.DataFrame, week: int, actuals) -> None:
        """Let every invented rival fill this week's slots and bank what they scored."""
        predict = rivals.PREDICTORS[self.spec.behaviour]
        top_n = self.spec.noise
        for opponent in self.entrants:
            for slot in config.SLOTS:
                name = opponent.display_name
                state = rivals.RivalState(name, name, frozenset(opponent.used))
                # Remove spent players before naive's cap and ranking truncation. Its
                # standalone predictor deliberately ignores depletion; a legal entrant cannot.
                eligible = (
                    proj[~proj.player_id.isin(opponent.used)]
                    if self.spec.behaviour == "naive" else proj
                )
                ranked = predict(eligible, slot, week, state, top_n=top_n)
                if not ranked:
                    opponent.picks.append((week, slot, None))
                    continue
                choice = ranked[0 if len(ranked) == 1 else int(self.rng.integers(len(ranked)))]
                opponent.used.add(choice.player_id)
                opponent.tds += actuals.get((week, choice.player_id), 0.0)
                opponent.picks.append((week, slot, choice.player_id))

    @property
    def standing(self) -> tuple[tuple[str, float], ...]:
        return tuple((o.display_name, float(o.tds)) for o in self.entrants)


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


def decision_time(proj: pd.DataFrame, week: int) -> datetime:
    """A clock reading just before this week's first pick deadline.

    `recommend.advise_slot` needs one and the other strategies have none: they go through
    `build_matrix`, which forbids byes and ruled-out players and consults no clock at all.
    Handing this one the real time would drop every candidate whose game has since kicked
    off, and the gap between two strategies would stop being a gap between two decision
    rules. Read off the frame, it excludes nothing the others kept.
    """
    kicks = proj.loc[proj.week >= week, "kickoff"].dropna()
    first = min(datetime.fromisoformat(str(k)) for k in kicks)
    return first - timedelta(minutes=config.PICK_DEADLINE_MINUTES + 1)


def pick_winprob(
    proj, slot, week, used, *, against=None, memo=None, banked=0.0, discount=None, **_
) -> Choice:
    """This week's pick by expected share of the pot, against an invented field.

    Cache a coherent lineup in QB/RB/FLEX order. Each accepted deviation changes the
    comparison for remaining slots, so re-evaluate them with earlier choices locked.

    Deviates from expected touchdowns only where `SlotAdvice.divergent` -- a clear pot-share
    winner outside simulation noise -- which is the same rule the CLI prints under. Within
    noise it takes the expected-TD pick, so most weeks of most replays are identical to
    `optimizer` and the rows differ only where the second objective actually had something
    to say.
    """
    if against is None:
        raise ValueError("The `winprob` strategy needs invented rivals: pass `against=Field(...)`.")
    memo = {} if memo is None else memo
    if week not in memo:
        pool, now = against.state(banked), decision_time(proj, week)
        locked, choices = {}, {}
        advice = recommend.advise_week(
            proj, week, set(used), locked, now=now, pool=pool, discount=discount
        )
        for index, name in enumerate(config.SLOTS):
            one = next((a for a in advice if a.slot == name), None)
            if one is None or one.recommended is None:
                choices[name] = EMPTY
                continue
            chosen = one.recommended
            if one.divergent:
                candidates = one.candidates or [one.recommended, *one.alternatives]
                chosen = next(c for c in candidates if c.player_id == one.best_share.player_id)
            choices[name] = Choice(chosen.player_id, chosen.player_name, chosen.team, chosen.lam)
            locked[name] = {week: chosen.player_id}
            if one.divergent and index < len(config.SLOTS) - 1:
                advice = recommend.advise_week(
                    proj, week, set(used), locked, now=now, pool=pool, discount=discount
                )
        memo[week] = choices
    return memo[week].get(slot, EMPTY)


Chooser = Callable[..., Choice]
STRATEGIES: dict[str, Chooser] = {
    "optimizer": pick_optimizer,
    "greedy": pick_greedy,
    "random": pick_random,
    "winprob": pick_winprob,
}
# Strategies that cannot be run against nobody. Kept as data so the CLI can refuse the
# combination with a sentence instead of letting `replay` raise from three slots deep.
NEEDS_RIVALS = frozenset({"winprob"})


# --- replay -----------------------------------------------------------------
def replay(
    frames: dict[int, pd.DataFrame],
    actuals: dict[tuple[int, str], float],
    season: int,
    strategy: str,
    *,
    weeks: Sequence[int] | None = None,
    against: Field | None = None,
    **kw,
) -> Replay:
    """Walk the season once, one pick per slot per week, never reusing a player.

    With `against`, a run of that invented field is advanced alongside, one week behind the
    decision: the policy sees what everyone had spent and banked *before* the week, which
    is what it would see on a Tuesday. Every strategy may be given a field, not just
    `winprob` -- the invented rivals do not react to my picks, so the same specification
    plays the identical season under each one and the finishes can be compared.
    """
    chooser = STRATEGIES[strategy]
    opponents = None if against is None else against.start()
    label = None if against is None else against.label
    out = Replay(season=season, strategy=strategy, against=label)
    used: set[str] = set()
    for week in sorted(weeks if weeks is not None else frames):
        proj = frames[week]
        # Read before this week's picks are appended: what I have banked is what finished
        # weeks scored, and this week has not been played yet.
        banked = out.total
        memo: dict[int, dict[str, Choice]] = {}
        for slot in config.SLOTS:
            choice = chooser(
                proj, slot, week, used, against=opponents, memo=memo, banked=banked, **kw
            )
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
        if opponents is not None:
            opponents.advance(proj, week, actuals)
    if opponents is not None:
        out.standing = opponents.standing
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
    against: Field | None = None,
) -> list[Summary]:
    """Replay one season under each named strategy, sharing frozen projections.

    `against` is given to every strategy rather than only to `winprob`, so the table can
    say where each of them finished in the same invented pool. Without that a `winprob`
    row would be read on the touchdown column, which is the one axis it is not trying to
    win.
    """
    strategies = list(strategies)
    needs = sorted(set(strategies) & NEEDS_RIVALS)
    if needs and against is None:
        raise ValueError(f"{needs[0]!r} needs invented rivals; pass `against=Field(...)`.")
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
                    against=against,
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
                Summary.of(
                    replay(
                        frames,
                        actuals,
                        season,
                        name,
                        weeks=weeks,
                        against=against,
                        discount=discount,
                    )
                )
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
