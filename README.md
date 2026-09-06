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
- [`docs/EVALUATION.md`](docs/EVALUATION.md) - generated season scores, forecast diagnostics, methods and provenance
- [`docs/ANALYSIS.md`](docs/ANALYSIS.md) - dated, authored interpretation and research limits; not refreshed by reruns
- [`docs/REVIEW.md`](docs/REVIEW.md) — September 2026 review: open defects, validation limitations, and recommended priorities

## Status

The weekly workflow and Phase 2 validation repairs are implemented: explicit deadline
eligibility, complete touchdown accounting, timestamped input archives, shared baseline
comparisons, and reproducible season experiments. Production model constants remain fixed.
The saved study supports further diagnosis, not a production calibration change. Calibration
experiments and live-policy validation are planned in [Phase 3](docs/IMPLEMENTATION_PLAN.md);
leaderboard strategy comes later.

The [evaluation report](docs/EVALUATION.md) combines actual season scores from each model's
greedy and optimizer pick history with ranking and calibration diagnostics. Its
[ranking section](docs/EVALUATION.md#ranking-diagnostics) retains the distinct unit of TDs per
ranked candidate, not achieved season scores. The study covers 2011–2025 with 2010 prior history;
both era summaries are retrospective. Its saved configuration,
scoring coverage, seeds, source hashes and provenance accompany the results in
[`experiments/results/roster-snapshot-repair`](experiments/results/roster-snapshot-repair), which
reran the identical frozen dataset after the F11 candidate-pool repair. The superseded
[Phase 2 results](experiments/results/phase2-validation) remain published for comparison.
Previous reports are retained in [`docs/archive`](docs/archive) under their old scoring and
evaluation definitions.

Generated reports are for facts, metric definitions and provenance, not automated research
conclusions. Human/AI reasoning is maintained separately in the dated [analysis](docs/ANALYSIS.md).
Its interpretation of this retrospective study is not automatically refreshed by reruns.

The assignment solver supplies a rest-of-season plan and the projected cost of overriding it.
Its optimality for a fixed, pruned forecast matrix does not establish a rolling-policy scoring
advantage. One strictly increasing map shared by all candidates in a slot preserves greedy
ordering; position-specific WR/TE maps can change FLEX ranks. Calibration can also change
assignment and hold/commit decisions. Joint ablations do not isolate defense alone, and these
model comparisons do not establish that the available inputs are exhausted or that simulations
are validated.

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

`recommend` also records the decision: the whole pre-pruning surface for every slot and
remaining week, the advice and hold-or-commit call per slot, the used and locked state, the
feed observations that were visible, and the code, constants and model that produced it. The
log is append-only — a correction is a new event, and `record`/`unrecord` append their own —
so a captured decision can be reconstructed later even after the feeds have moved on.
Outcomes are never stored beside it; they are joined from finalized scoring when read. Use
`--no-capture` to skip recording. This evidence is what
[Phase 3](docs/IMPLEMENTATION_PLAN.md) needs and is not used to make picks.

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

`backtest` measures actual season scores; `evaluate` reports ranking and calibration diagnostics
alongside separate greedy/optimizer replays. A challenger's common-pool top-one rankings can
repeat a player, while its achieved replay uses each player at most once. Ranking differences
are never multiplied into purported season gains. Slot-week differences are averaged within
season; shuffled seeds are averaged within season before computing uncertainty across seasons.
Seed variability and empty-cell coverage are reported separately.

Replay commands share these input policies:

| Policy | Inputs and assumptions |
|---|---|
| `historical` (default) | Closing lines strictly before week W, masked before team/league averages; stats through W−1; rosters/injuries through W; usage roles. It has no observation times to consult, so it cannot tell which of week W's games had finished; that cut, final schedule revisions, weekly report timing and later stat corrections remain approximations. |
| `snapshots` | Latest successful archived feed observation at or before an explicit decision timestamp; depth-chart roles by default. Stats from completed games already observed at that timestamp are used, including earlier games in week W, matching the live path. Missing essential history fails; absent/stale optional feeds are recorded with fallbacks. |
| `legacy-closing` | Prior closing-line horizon, explicitly labeled sensitivity analysis. Only this policy permits `--vegas-horizon`. |

Historical and legacy replay reject the latest depth chart. Live recommendations retain their
depth-chart defaults and available current lines. Historical replay uses one decision immediately
before the first confirmed pick deadline. Snapshot replay also makes one decision per week;
this phase does not simulate repeated decisions within a week.

```bash
uv run pool evaluate --season 2024-2025 --model all --seeds 0-19 --csv /tmp/forecasts.csv
uv run pool evaluate --model player-vegas,regressed-rate --baseline player-vegas
uv run pool backtest --season 2025 --input-policy legacy-closing --vegas-horizon 0
uv run pool backtest --season 2026 --input-policy snapshots --decision-times decisions.csv
uv run pool benchmark --config experiments/phase2-validation.toml --output data/experiments/phase2-validation
# Continue only when configuration, code, dependencies and frozen dataset match:
uv run pool benchmark --config experiments/phase2-validation.toml --output data/experiments/phase2-validation --resume
```

Decision CSVs require `season,week,decision_at`, with one timezone-aware timestamp for every
requested week, such as `2026,1,2026-09-10T18:00:00-04:00`. Observations become available when
an import succeeds; source publication timestamps never backdate availability. Legacy database
rows acquire no invented observation times. Later imports may change measured final outcomes
but cannot alter projections reconstructed from earlier snapshots.
The benchmark reads the decision CSV once, while resolving its configuration, and carries the
timestamps in the resolved specification. They are therefore part of the run's configuration
hash: editing the file in place and resuming is rejected, and the workers never re-read the
path. A normalized `decision-times.csv` copy is saved beside the frozen dataset.

Every successful refresh atomically replaces a normalized feed and appends an observation,
including valid empty feeds. Payloads are compressed and deduplicated; full play-by-play is not
stored. Schedule/lines, player stats, rosters, injuries, depth charts and compact TD credits/results
are archived. Failed imports retain earlier data and add no successful observation. `status`
shows snapshot coverage and earliest/latest observation times. The benchmark uses a dedicated
research database, audits every required game/history season, then freezes a separate copy.
A documented, guarded [source correction](experiments/scoring-corrections.json) resolves two
duplicate source TD plays in one 2011 game; original observations remain archived.

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
