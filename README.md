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
- [`docs/PHASE3B_OUTCOME.md`](docs/PHASE3B_OUTCOME.md) - completed calibration experiment: no candidate promoted, rationale and next step
- [`docs/PHASE3C_PROTOCOL.md`](docs/PHASE3C_PROTOCOL.md) - dated prospective baseline protocol: window, decision events, parity criteria and floors, written but never run
- [`docs/PHASE3C_OUTCOME.md`](docs/PHASE3C_OUTCOME.md) - the decision to close that window without collecting, and what it gives up
- [`docs/PHASE4_PLAN.md`](docs/PHASE4_PLAN.md) - opponent ingestion, standings, and deciding by win probability: staging, and what can and cannot be validated
- [`docs/PHASE4_STAGE1_PLAN.md`](docs/PHASE4_STAGE1_PLAN.md) - implemented report ingestion, archive guarantees, and identity checks
- [`docs/REVIEW.md`](docs/REVIEW.md) — September 2026 review: open defects, validation limitations, and recommended priorities

## Status

The weekly workflow and Phase 2 validation repairs are implemented: explicit deadline
eligibility, complete touchdown accounting, timestamped input archives, shared baseline
comparisons, and reproducible season experiments. Production model constants remain fixed.
The [Phase 3B report](experiments/results/phase3-calibration/CALIBRATION.md) covers the completed
2016-2025 walk-forward calibration experiment. Position-specific log-affine maps improved
forecast accuracy, but no candidate passed all signed forecast and policy gates. The
[authored outcome](docs/PHASE3B_OUTCOME.md) is no promotion; production remains unchanged.
Phase 3C was declared as identity-only prospective collection and then
[closed without collecting](docs/PHASE3C_OUTCOME.md): its [protocol](docs/PHASE3C_PROTOCOL.md)
stands as written and unrun, no decision was ever captured under it, and no floor was
measured. The window required a source freeze for its duration, which the season's own
work — opponent ingestion, standings, a win-probability objective — could not accommodate.
The replay assumption it would have checked is still open, and a single captured decision
verified the same day settles it whenever that is worth doing.

Phase 4, stage 1 is implemented: weekly entrant reports are archived before parsing,
imported with player and entrant identity checks, and displayed as pool-reported standings.
`pool report list` includes failed imports and checks. Entrant scoring, remaining-pool
queries, and win probability remain later stages.

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

`recommend` prints the decision it captured, and `record --decision <id>` names it. With
more than one decision in a week the fallback link — the most recent advice for that slot
— is whichever happened last, which is not the same thing as the one the pick came from.

### Working a week

A week has one decision point before each **kickoff wave** — a distinct kickoff day, with
its deadline 60 minutes before that day's first kickoff. Week 1 of 2026 opens Wednesday,
plays again Thursday, holds the main slate Sunday and closes Monday; every later week is
Thursday, Sunday, Monday. Work each wave the same way:

> The Sunday deadline is the week's **first** Sunday kickoff, not the 1pm block. A week
> with an early London game moves it to 08:30 ET — 2026 weeks 4, 5 and 6 all do. Check the
> deadline rather than assuming noon; `pool captures` and the protocol's event table both
> report it.

```bash
uv run pool refresh --season 2026                    # immediately before the capture
uv run pool recommend --season 2026 --no-capture     # look as much as you like
uv run pool recommend --season 2026                  # capture once; prints the decision id
uv run pool record --decision <id> --qb "Herbert"    # only the slots you are committing now
```

- **Refresh immediately before every capture, no exceptions.** Parity divides by every
  captured decision against a floor of `1.0`, and a decision whose feeds were never
  archived at that instant cannot be verified — unverifiable is not verified. One
  unrefreshed capture fails the window's parity floor.
- **`recommend` commits nothing.** It writes only the capture log; `my_picks` is untouched.
  Only `record` commits a pick, and only a recorded pick locks its slot or spends the
  player for the season.
- **Use `--no-capture` to look.** Every plain `recommend` writes a permanent decision that
  then has to reconstruct and match a snapshot replay. Browse freely without capture;
  capture once per wave.
- **Honour the hold.** When the recommended player is in an early game and the best later
  alternative costs less than `INFO_PREMIUM_TD` season-TDs, the advice says hold. That is
  the signal to leave the slot open and decide at the next wave rather than commit now.
- **Name the decision when you record.** Without `--decision` the pick links to the most
  recent advice for that slot, which across several waves is whichever happened last, not
  the one you acted on.
- **News after you recorded is not a problem.** `pool unrecord <week> <slot>` removes the
  pick and frees the player; `pool record` over the slot replaces him. Either way the log
  keeps the history — the earlier entry is reported as withdrawn or superseded and the
  population counts only the pick that still stands.

No protocol governs this cadence any more — [3C is closed](docs/PHASE3C_OUTCOME.md) and
no capture is required at any wave. Capturing anyway costs nothing and keeps the option:
`pool captures` lists what you have and `pool verify-capture` checks it, and because the
enforced source fingerprint now covers only the decision path, an unrelated feature no
longer makes yesterday's captures unverifiable.

```bash
uv run pool captures --season 2026 --week 1          # captured decisions, or one in full
uv run pool verify-capture --season 2026             # reconstruction and live/snapshot parity
uv run pool baseline --config experiments/phase3c-baseline.toml \
  --out experiments/results/phase3c-baseline         # the descriptive export and dated note
```

`verify-capture` re-derives each decision's advice from its stored surface alone, then
rebuilds the same instant from the archived feeds and compares the model columns. The
rebuild runs under the constants the decision recorded, so a setting that has moved since
is not reported as a live/archive divergence. The archive is written by `refresh`, so a
decision made without one cannot be verified and is reported as unverified rather than
skipped; a window with no captures at all exits nonzero, because nothing verified is not
verification. The fingerprint it compares covers what a decision is a function of — the
import closure of `recommend`, `projections` and `snapshots`, plus `uv.lock` — and not the
whole tree; the whole-tree hash is still recorded beside it, so drift stays visible without
an unrelated module invalidating a capture it could not have changed. `baseline` describes
the captures under the dated [3C protocol](experiments/phase3c-baseline.toml), which
declares the window, the decision events, the populations and the floors; a missing floor
blocks the export. That window was closed without being run, and the command is kept
against a future one. It promotes nothing and changes nothing.

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

### Importing the pool's weekly report

```bash
uv run pool report import week1.csv --season 2026 --check
uv run pool report import week1.csv --season 2026 --me "Chris K."
uv run pool report list --season 2026
uv run pool standings --season 2026             # latest imported week, reported values
uv run pool standings --season 2026 --week 1
```

The initial `--format csv` parser accepts UTF-8 CSV with one row per entrant and slot:

```csv
week,entrant,slot,player_name,reported_week,reported_total,reported_rank
1,Chris K.,QB,Quarter One,3,3,1
1,Chris K.,RB,Runner One,,,
1,Chris K.,WR/TE,Flex One,,,
```

Include every entrant and all three slots: `QB`, `RB`, and `FLEX` (`WR`, `TE`, and
`WR/TE` also map to `FLEX`). The `entrant`, `slot`, and `player_name` columns are all
required, and `entrant` and `slot` must be filled in on every row. Supply
`week` on every row or use `--week`; when both are present they must agree. An optional
`season` column must agree with `--season`. Totals and rank are optional integers; put
them on one row per entrant or repeat consistent values. Missing totals stay unknown.
Duplicate slots, conflicting totals, and unknown columns are rejected.
The full [reference fixture](tests/fixtures/pool_report.csv) includes two entrants.
The delivery format has not yet been confirmed against a real pool report.

To report an entrant who submitted nothing for a slot, keep the row and leave
`player_name` empty. That stores a no-pick, which `standings` shows as `(no pick)` and
which stays distinct from a name that could not be resolved and from a week that was
never imported. Dropping the row instead is still rejected as a missing slot, because
nothing distinguishes it from a truncated file.

Every attempt commits the original bytes before parsing, including failed imports and
`--check`. Re-imports add an observation while sharing the same stored payload and
updating the week's picks. Parse outcomes live separately in metadata so the original
observation remains immutable. `--check` validates and prints counts and issues without
changing entrants, picks, or totals.

Unresolved player names are retained and listed with candidates; re-import after updating
the roster to resolve them. Entrant names are normalized across weeks. Additions, removals,
and renames leave the week exactly as it was and exit nonzero until acknowledged with
`--allow-roster-change`; review those differences before acknowledging them. A correction
that drops an entrant and a delivery that was truncated look identical, so nothing is
written until you say which it is; acknowledging then replaces that week's entrant set
while preserving the archived originals. `--me` identifies your row once, then every
import compares it with `my_picks`. A slot you have not recorded is a note; a slot where
your record and the report name different players — or where the report says you picked
nobody — is a mismatch. Unresolved names, unacknowledged roster changes, and mismatches
exit nonzero. `my_picks` is never edited by a report import. Incomplete game coverage
warns but permits the import.

`standings` shows the supplied picks, weekly count, running total, and rank, explicitly
labelled as reported. It uses the latest imported week, so checks and failed parses cannot
advance the display. Report archives appear in `status` but are excluded from projection
inputs, snapshot replay, and feed freshness checks.

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
uv run pool diagnose --run data/experiments/roster-snapshot-repair --out experiments/results/phase3a-readiness
uv run pool benchmark --config experiments/phase3-calibration.toml --output data/experiments/phase3-calibration
```

`diagnose` describes a saved study's rate errors by population, position, rate bin,
availability and forecast lead horizon, and writes a dated readiness note. Group definitions
are frozen before any outcome is read, horizons are never pooled — repeated forecasts of one
target week share its single outcome — and eligible zero rates are counted rather than
dropped, separately from hard exclusions. It fits no correction and states no conclusion.

A configuration carrying a `[calibration_experiment]` table runs the Phase 3B experiment
instead of a model bake-off: identity forecast/outcome pairs for every season, then one fit per
fold on the seasons that had already finished, then each candidate applied to the whole surface
before pruning and discount and replayed with its own no-reuse history. The map is
`exp(a) * lam ** b`, fitted pooled and by position with WR and TE separate; zero rates map to
zero and hard exclusions stay masks, so every candidate is scored on the same rows. Fold
schedule, families, population, weighting, fallback, primary estimand, the promotion conditions
and the decision margins are all declared in the dated configuration, whose text is hashed into
the run identity; a missing margin blocks the run, and so does one outside its own domain. The
run measures the outcome coverage its floors are stated over, reports retention beside it, counts
every training row it discards, and scores each candidate by forecast horizon and availability as
well as in the pooled primary estimand. The primary comparison is a paired season t with a Holm
step-down, and the step-down's own decision is what the promotion rule reads. See
[experiments/README.md](experiments/README.md) for the stages and artifacts.

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
