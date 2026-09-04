# NFL Player Selector

A decision-support tool for a season-long NFL touchdown pool: each week you pick
one QB, one RB, and one WR/TE, you score a point for every TD your picks throw or
score, and every player can only be used **once per season**. Most TDs at the end
of the season wins, winner take all.

This turns weekly picks into a season-long resource-allocation problem: *when* do
you spend your best players, given their schedules, and how should your risk
appetite change with your position on the leaderboard?

- [`docs/DESIGN.md`](docs/DESIGN.md) — the pool rules, the model, and the system architecture
- [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) — phased build plan and status
- [`docs/BACKTEST.md`](docs/BACKTEST.md) — how the model performs against fifteen replayed seasons
- [`docs/PROJECTION_BENCHMARK.md`](docs/PROJECTION_BENCHMARK.md) — the projection layer scored on every player-week forecast, against seven alternatives
- [`docs/REVIEW.md`](docs/REVIEW.md) — September 2026 review: open defects, validation limitations, and recommended priorities

## Status

The [September 4 review](docs/REVIEW.md) identified open deadline-eligibility and
evaluation issues. Its findings qualify the research claims below and recommend
fixing the weekly workflow and validation before prioritizing leaderboard strategy.

Phases 0 and 1 are implemented: data import, a matchup-adjusted projection
model, the season-long assignment optimizer, and a CLI that gives weekly picks.
The Phase 4 backtesting harness is built too, and it reports an uncomfortable
result: replayed over 2011-2025 the optimizer is **statistically tied with
simply picking the best available player each week** (-0.73 TD/season, SE 2.39,
better in 8 of 15 seasons), though both beat a random baseline by ~13 TD/season.
The optimizer stays anyway — it is what produces the rest-of-season plan and the
season-cost ranking — but the plan is a forecast of intent, not a proven edge.
See [`docs/BACKTEST.md`](docs/BACKTEST.md) for the numbers, the ideas that were
tested and rejected, and how small an effect the harness can actually resolve.

The projection layer has since been benchmarked on its own, scoring every
player-week forecast rather than the 54 picks — 877,000 forecasts across eight
models and fifteen seasons. The shipped model leads the reported ranking
comparison; the contextual multipliers' top-10 advantage does not establish a
season-scoring gain. The model also over-projects its best players. A monotone
calibration correction preserves greedy rankings but can change assignment picks. See
[`docs/PROJECTION_BENCHMARK.md`](docs/PROJECTION_BENCHMARK.md).
Leaderboard-aware strategy (Phase 3) is not built yet.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python, dependency, and
environment management. Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from the repo root:

```bash
uv sync                         # creates .venv and installs everything from uv.lock
uv run pool refresh             # pulls nflverse data into data/pool.db
```

`uv sync` provisions the right Python (see `.python-version`) if it is missing,
so there is no separate `venv`/`pip` step. Every command below is prefixed with
`uv run`, which keeps the environment in sync with the lockfile before running.
If you prefer bare commands, `source .venv/bin/activate` once and drop the
prefix, or install the CLI globally with `uv tool install .`.

Data comes from [nflverse](https://github.com/nflverse) via `nflreadpy`. The
default season is 2026 (override with `--season` or `POOL_SEASON`); the prior
season is always used for the start-of-season prior.

## Weekly workflow

```bash
uv run pool refresh             # latest stats, lines, rosters, depth charts, injuries
uv run pool recommend           # picks for every open slot this week
uv run pool record --rb "Saquon Barkley"        # lock a slot (others stay open)
uv run pool record --qb "Herbert" --flex "Goedert"
uv run pool recommend           # re-optimizes the still-open slots
uv run pool plan                # rest-of-season assignment
uv run pool players --pos RB    # projection table for a slot
uv run pool picks / uv run pool unrecord 3 QB
```

## Backtesting

```bash
uv run pool refresh --season 2025          # imports 2024 (prior) and 2025
uv run pool backtest --season 2025         # replay one season against the baselines
uv run pool backtest --season 2017-2025 --detail
uv run pool sweep --season 2017-2025       # grid-search the discount and prior weight
```

Two layers are measured separately, because a better forecast is not the same
thing as a better season:

```bash
uv run pool models                         # the benchmark's projection models
uv run pool evaluate --season 2011-2025    # score every player-week forecast
uv run pool backtest --season 2017-2025 --projection player-vegas
```

`backtest` answers "does this decision rule win more touchdowns"; `evaluate`
answers "is this projection a better forecast". The first is limited to effects
above ~±3 TD/season; the second resolves roughly 5x smaller — but only for claims
about the forecast.

Each week the model sees only what was knowable an hour before kickoff, picks a
player per slot, and is scored on the touchdowns they actually went on to score
— against greedy best-available, a random top-10 baseline, and perfect
hindsight. Backfill more seasons with
`for y in 2017 2019 2021 2023 2025; do uv run pool refresh --season $y; done`.

`recommend` shows, per slot, the optimizer's pick, its expected TDs, its pick
deadline (one hour before kickoff), and alternatives ranked by **season cost**:
how many projected TDs the rest-of-season plan loses if you take that player
now instead. When the recommended player plays before the Sunday slate, it says
whether to commit or hold for injury news.

## Tuning

Every modeling knob lives in `src/pool/config.py` (shrinkage weights, home edge,
depth-chart role multipliers, future discount, information premium).

## Development

```bash
uv run pytest
uv run ruff check src tests && uv run ruff format src tests
```

`pytest` and `ruff` live in the `dev` dependency group, which `uv sync` installs
by default; `uv sync --no-dev` gives a runtime-only environment.

Dependency changes go through uv so that `uv.lock` stays authoritative — it is
committed, and it is what pins the exact versions everyone gets:

```bash
uv add scikit-learn             # add a runtime dependency
uv add --dev pytest-cov         # add a dev-only dependency
uv remove pandas                # drop one
uv lock --upgrade               # refresh every pin
uv sync                         # apply the lockfile to .venv
```
