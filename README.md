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

The weekly reliability work from the [September 4 review](docs/REVIEW.md) is
implemented: enforced deadlines, validated pick history, complete touchdown
accounting, local scoring, and feed freshness reporting. Evaluation repairs,
calibration changes, and leaderboard strategy remain later work.

**Published benchmark numbers below and in both research reports use the previous
passing/rushing/receiving scoring definition. They have not been rerun with
complete touchdown accounting.**

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
uv run pool refresh             # latest feeds, including compact play-by-play scoring credits
uv run pool recommend           # picks for every open slot this week
uv run pool record --rb "Saquon Barkley"        # log a submitted pick
uv run pool record --qb "Herbert" --flex "Goedert"
uv run pool recommend           # re-optimizes the still-open slots
uv run pool plan                # rest-of-season assignment
uv run pool players --pos RB    # projection table for a slot
uv run pool score --season 2026          # recompute all recorded picks from local data
uv run pool score --week 3 --season 2026  # recompute one week, including corrections
uv run pool score --week 3 --season 2026 --refresh  # refresh first
uv run pool picks                       # scores, pending reasons, and subtotals
uv run pool status --season 2026 --week 3 # feed attempts, coverage, freshness, fallbacks
uv run pool unrecord 3 QB               # remove an incorrect entry
```

Each invocation uses one Eastern decision time. A player becomes unavailable
exactly 60 minutes before confirmed kickoff. Re-running after Thursday's lock
re-solves the remaining choices while preserving recorded picks and future-week
eligibility. Games with unconfirmed kickoff times are excluded from current-week
advice; their future planning estimates remain visible. `players` identifies used,
expired, unavailable, and unconfirmed players.

`record` logs picks submitted elsewhere. Historical entries and corrections are
allowed, with warnings for elapsed deadlines or apparent unavailability. Names
are resolved against the requested season/week's stats and roster history,
including inactive players. Every supplied slot is validated before one atomic
write. Replacing a player clears the slot's score; re-recording the same player
preserves it. Removing a pick removes its contribution to totals.

Scoring counts every touchdown thrown or scored, including returns and recoveries.
The importer uses nflverse's explicit scorer identifier and separately credits the
passer on a credited passing touchdown, excluding negated plays and conversions.
See the [official play-by-play field definitions](https://nflfastr.com/reference/fast_scraper.html).
A game is complete only with an end-of-game marker, terminal scores matching the
schedule, and resolved touchdown identities. A complete game with no credits for
a pick scores **0**, including a player who did not play. Missing/incomplete feeds
or unresolved games remain **pending**. Completed picks score independently;
weekly and season subtotals are labeled incomplete while any picks remain pending.
Stored scores survive missing results and appear as “last scored” while pending.
If `score --refresh` partially fails, it retains existing scores, scores completed
unscored picks from available local results, and exits nonzero. Use plain `score`
to explicitly recompute existing scores from the retained local data.

Explicit refreshes bypass nflreadpy's cache, attempt independent feeds, and retain
the previous dataset on download, parsing, or missing-file failures. Partial
failures exit nonzero; expected unpublished preseason results are informational.
Advice continues with warnings when usable schedule and player history exist.
`status`, `recommend`, and `plan` distinguish missing coverage/model fallbacks from
fetch age. Default age limits are 1 hour for schedule/lines and 24 hours for stats,
rosters, injuries, depth charts, and touchdown feeds. Configure these through
`POOL_FRESHNESS_SCHEDULE_HOURS`, `POOL_FRESHNESS_PLAYER_STATS_HOURS`,
`POOL_FRESHNESS_ROSTERS_HOURS`, `POOL_FRESHNESS_INJURIES_HOURS`,
`POOL_FRESHNESS_DEPTH_CHARTS_HOURS`, and `POOL_FRESHNESS_TOUCHDOWNS_HOURS`.
Completed historical seasons are exempt from age warnings.

SQLite upgrades run transactionally on open, preserving imported history and
picks. Legacy timestamps retain an unspecified-timezone label. New import metadata
uses UTC; kickoff display remains Eastern. Existing databases lack complete TD
coverage until refreshed. Live projections may use legacy offensive-TD estimates
with a warning; finalized scoring and replay comparisons require complete coverage.

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
uv run ruff check src tests
uv run ruff format --check src tests
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
