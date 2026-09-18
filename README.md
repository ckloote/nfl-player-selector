# NFL Player Selector

A decision-support tool for a season-long NFL touchdown pool: each week you pick
one QB, one RB, and one WR/TE, your score is the number of TDs your picks throw or
score, and every player can only be used **once per season**. Most TDs at the end
of the season wins, winner take all; entrants tied at the top split the pot.

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
- [`docs/PHASE4_PLAN.md`](docs/PHASE4_PLAN.md) - opponent ingestion, standings, and deciding by expected share of the pot: staging, and what can and cannot be validated
- [`docs/PHASE4_STAGE1_PLAN.md`](docs/PHASE4_STAGE1_PLAN.md) - implemented report ingestion, archive guarantees, and identity checks
- [`docs/PHASE4_STAGE2_PLAN.md`](docs/PHASE4_STAGE2_PLAN.md) - implemented shared scoring, computed standings, ties, and independent used pools
- [`docs/PHASE4_STAGE3_PLAN.md`](docs/PHASE4_STAGE3_PLAN.md) - implemented pot-share policy: the simulator, the opponent model, what it can and cannot be judged on
- [`experiments/results/phase4-simulator/CALIBRATION.md`](experiments/results/phase4-simulator/CALIBRATION.md) - the simulator fitted against 2011-2025, with the checks it fails beside the ones it passes
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

Phase 4 is implemented. Weekly entrant reports are archived before parsing, imported with
identity checks, and scored through the same core as your own picks. `pool standings` shows
computed ranks and reported values together, preserves ties, and tracks each entrant’s used
players. `pool report list` includes failed imports. See the
[stage 1](docs/PHASE4_STAGE1_PLAN.md) and [stage 2](docs/PHASE4_STAGE2_PLAN.md) plans.

[Stage 3](docs/PHASE4_STAGE3_PLAN.md) adds a second objective **beside** the first, never in
place of it: `pool recommend` still gives the expected-TD advice and now prints each
candidate's expected share of the pot next to it, with a sentence naming why the two
disagree when they do. A tie at the top splits the winnings, so the quantity is a share and
not a probability of an outright win. The
[simulator calibration](experiments/results/phase4-simulator/CALIBRATION.md) reports each of
its four checks pass or fail: one passes outright, two pass only in part, and same-team
substitution fails and is deferred with its route written down. Nothing here establishes that the policy wins
pools and nothing can: a season is one Bernoulli trial. What accrues instead is prospective
and weekly — `pool predict record` archives a prediction of every rival's picks before the
report that settles it, and `pool predict score` reads the record back. `pool backtest
--strategy winprob` replays a finished season and says, everywhere it prints, that it did so
against rivals who do not exist.

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
uv run pool predict record              # archive rival predictions before the week is visible
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

The existing 2026 captures require
`uv run pool verify-capture --season 2026 --allow-code-drift`, and have since stage 2. The
enforced closure has moved three times: stage 2 extracted the shared pick-scoring core into
`scoring.py`, a later review fix to the same file — scoring a pick with no game zero once its
week is final — moved it again, and stage 3 added `rivals.py` and `simulate.py` when
`recommend` gained the second objective. Each move was an isolated commit, with the pre-move
output recorded first. Every capture still reconstructs and matches replay under the
override, which explicitly reports that the fingerprint check was bypassed. A capture made
this week cannot reach parity until its own week has been scored — replay needs complete
touchdown coverage, and an unplayed week has none; that is pending, not a failure. See the
recorded verifications for [stage 2](docs/PHASE4_STAGE2_PLAN.md#implementation-verification--2026-09-15)
and [stage 3](docs/PHASE4_STAGE3_PLAN.md#implementation-verification--2026-09-17).

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
a pick scores **0**, including a player who did not play. Once every game in a week is
final, a pick with no game of its own also scores **0** — the pool treats a pick on a
player who is not playing as worth nothing — and says so, because the other way to reach
that state is a team the schedule does not match. Missing or incomplete feeds remain
**pending**, and so does a missing game while its week is still unfinished. Completed picks score independently;
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
uv run pool report import data/reports/week1.csv --season 2026 --check
uv run pool report import data/reports/week1.csv --season 2026 --me "Chris K."
uv run pool report list --season 2026
uv run pool standings --season 2026             # latest picks, computed season ranks
uv run pool standings --season 2026 --week 1
```

Keep report files in `data/reports/`. Git ignores that folder, because a report names every
entrant in the pool.

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

Every import commits the original bytes before parsing, including imports that fail.
Re-imports add an observation while sharing the same stored payload and updating the
week's picks. Parse outcomes live separately in metadata so the original observation
remains immutable. `--check` is a dry run: it validates and prints counts and issues but
archives and writes nothing, so checking a file repeatedly while you fix it leaves no
trace. A report can be imported as soon as it arrives, even before that week's games
finish; re-importing the week later replaces it in place.

Unresolved player names are retained and listed with candidates; re-import after updating
the roster to resolve them. A name that is not an exact match (ignoring case and
punctuation) but fits exactly one player by a partial name or a close spelling is imported
and printed as a note showing what you typed and who it matched, in both `--check` and the
import. Read those notes: a typo can match the wrong player. Entrant names are normalized
across weeks. Additions, removals,
and renames leave the week exactly as it was and exit nonzero until acknowledged with
`--allow-roster-change`; review those differences before acknowledging them. A correction
that drops an entrant and a delivery that was truncated look identical, so nothing is
written until you say which it is; acknowledging then replaces that week's entrant set
while preserving the archived originals. `--me` identifies your row once, then every
import that contains your row compares it with `my_picks`; a file without your row is a
roster change, not a mismatch. Include yourself: `standings` only lists entrants in the
report, and the comparison is what catches the pool registering a different pick than the
one you recorded. A slot you have not recorded is a note; a slot where
your record and the report name different players — or where the report says you picked
nobody — is a mismatch. Unresolved names, unacknowledged roster changes, and mismatches
exit nonzero. `my_picks` is never edited by a report import. Incomplete game coverage
warns but permits the import.

Report archives appear in `status` but are excluded from projection inputs, snapshot replay,
and feed freshness checks.

## Standings

```bash
uv run pool standings --season 2026          # latest imported picks and season standings
uv run pool standings --season 2026 --week 1 # choose the displayed picks, not the ranking cutoff
uv run pool score --season 2026             # update your own cached scores for pool picks
```

The table puts computed ranks, weekly TDs, season TDs, and used-player counts beside the
supplied weekly totals, season totals, and ranks. Reported values remain passthrough;
standings neither computes from them nor compares them against its totals. The pool's
organizer uses the same nflverse source, so agreement would not be independent validation.

Picks default to the latest imported week. Everyone is ranked through the last consecutive
completed week starting at week 1; a gap holds the cutoff back. The title states both weeks.
Later reported picks appear separately as in progress. Your recorded picks also appear there
when no report has arrived for their week. Where a report differs from your records, it is
shown with a note to review the `pool report import --me` comparison.

A completed game with no credits scores zero, and a reported no-pick is also a final zero.
So does a pick with no game once that week is final, reported as a note naming the player.
An unfinished week is pending; an unresolved name needs re-importing; a
missing report leaves three missing slots. Weekly and season subtotals explicitly remain
incomplete while any of these gaps exists. Ranks are provisional in that case. Tied entrants
share competition ranks (1, 2, 2, 4), and a tie for first splits the pot equally. A provisional
tie describes a split at the end, not a settled result. Totals count touchdowns of every kind,
including returns; two-point conversions do not count.

Used players include every imported week, even one still being played. Each entrant has an
independent set, unresolved names add an unknown count, missing reports leave it incomplete,
and repeated players are counted once with their weeks reported. The Python API
`standings.remaining_counts(conn, season, week, entrant_id)` counts known remaining players
by slot from `state.historical_pool`, spending only what was used **before** that week, so a
report that has already arrived for the week itself cannot shrink its own answer; callers
should inspect `used_pools` for unknown usage and missing weeks. The pot-share view below is
what reads these pools.

Entrant scores are computed without a cache or a migration. Your own `my_picks.tds` cache
still updates only with `pool score`; standings prompts for that command when it is empty
or stale for a finished game. With matching recorded and reported picks and a current cache,
`pool picks` and standings use the same scoring function and show the same TD count.

## Deciding by share of the pot

Once reports are imported, `pool recommend` prints a second opinion beside the expected-TD
advice. It never replaces it: the recommended pick is still the assignment solver's, and
the pot-share column sits next to it so the disagreement is visible rather than silent.

```
Pot share vs 4 rivals: you 1 TD, best rival Jamie 5. 2000 paired simulations, seed 20260916.
─────────────────────────────────── QB ───────────────────────────────────
  PICK: Lamar Jackson (BAL QB) — @ DAL, Vegas x1.22, opp-D x1.28
  Expected TDs 2.65 · pot share 46.2% ± 1.1% · deadline Sun 9/27 3:25PM ET
 Alternative           xTD   Season cost  Pot share  vs EV   Plan uses in
 Patrick Mahomes (KC)  2.48  -0.17        45.5%      tied    wk 9
 Jared Goff (DET)      2.66  -0.36        43.3%      -2.8%   wk 12
```

A **tie for first splits the winnings**, so the quantity maximised is an expected share —
a win counts 1 and a k-way tie counts 1/k — and not the chance of an outright win. Season
totals are small integers and ties are common; a rule that ignored them would
systematically undervalue positions that reliably draw level.

`vs EV` is the difference from the recommended pick's share, estimated on **identical
draws**: every candidate is evaluated against the same simulated seasons and the same rival
pick-paths, so a difference is estimated far more precisely than either level. A candidate
whose difference does not clear that paired standard error prints as `tied` and is not
ordered. Without that the tool would reorder picks on Monte Carlo noise every week, and the
simulation count would quietly become a decision input.

Where the two objectives disagree, the line under the slot says why — an overlap with the
rivals likely to take that player, or the standings — and names them. A divergence it cannot
attribute to either is printed as a defect rather than dressed up as advice. With no reports
imported it says there is no second view and gives the expected-TD advice alone.

```bash
uv run pool recommend --season 2026 --sensitivity   # re-rank across the fitted knobs
```

`--sensitivity` re-runs each slot across a declared stress range of the game-correlation
constant and the rival-noise width, and reports per slot whether a divergence separates at
every setting, at some of them and never the other way, or nowhere. It reads the noise band
rather than the raw ranking, so a slot whose candidates are all tied reports as tied and not
as knob-sensitive. On the scripted test cases the fitted `k_game` turns out not to matter at
any value in the range; the rival-noise width, which `config.py` declares provisional,
decides whether a divergence separates at all.

### The prediction log

The one place this work gets to be empirical. Predict every rival's picks, archive the
prediction, read the report, score it — and the prediction must be recorded before it could
have seen the answer.

```bash
uv run pool predict record --season 2026 --week 3          # archive before the week is visible
uv run pool predict record --season 2026 --week 3 --dry-run  # show it, archive nothing
uv run pool predict score --season 2026                    # hit rates and the PIT histogram
```

Three hypotheses are archived per rival per slot per week, not one: greedy from their own
remaining pool, the same optimizer this tool runs, and naive — the best player in the slot,
ignoring the one-per-season rule entirely. Which of these describes five particular people
is a fact about them rather than a choice to make in advance, and it can only ever be
measured if all three were written down before the week they describe. The gap between
naive and greedy is its own measurement: it says whether they track the constraint at all.
The ranked list matters more than its winner, so where the actual pick landed in each
ranking is recorded, not just whether the top one hit.

`predict record` exits non-zero when the record will not be scorable — the archive still
happens, because an unscorable record is still evidence, but a scripted run has to be told
the observation was lost. Reports carry their arrival time, so a report that landed after a
prediction cannot be used to make it.

`predict score` also reports the **probability integral transform** of each entrant's weekly
total: where the actual total fell in the simulated distribution, randomised within the
observed integer because the counts are discrete. What gets archived is the *commitment* — a
seed, a simulation count, the parameters and a content hash of the projection frame — and
not the numbers, so rebuilding it twice gives the identical distribution and a later
re-forecast cannot move it. It is reported with no verdict: 85 draws a season cannot test
uniformity with any power, and the lean is the readable part.

## Backtesting

```bash
uv run pool refresh --season 2025          # imports 2024 (prior) and 2025
uv run pool backtest --season 2025         # replay one season against the baselines
uv run pool backtest --season 2017-2025 --detail
uv run pool sweep --season 2017-2025       # grid-search the discount and prior weight
uv run pool backtest --season 2025 --strategy winprob,optimizer  # pot share, vs invented rivals
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

The `winprob` strategy replays the pot-share policy, and everything it prints carries the
reason it proves nothing:

```
Strategy    TDs         ...  % ceiling   Finish
winprob     12.0             100%        1st of 5
optimizer   12.0             100%        1st of 5
random      6.0 ± 0.8        50%         5th of 5

Against 4 invented rivals picking greedy, among their top 3 (seed 0). No opponent picks
exist for a finished season, so this field was invented — "a test of the machinery, not of
the idea", in the phase plan's own words.
```

A finished season has no opponent picks, so `--rivals N` invents them. Worse than invented:
by default they pick greedily among their best few, which is the *same* hypothesis the
policy assumes when it values a candidate, so it is being scored against an opponent model
that is true by construction — its best case, not a neutral one. `--rival-behaviour naive`
or `optimizer` runs the mis-specified case, and the gap between two such runs is the only
reading here worth anything. Whatever the behaviour, the one-player-per-season rule is
enforced on them.

Read that row on the **finish** column and not on TDs. Giving up a touchdown for a larger
share of the pot is the whole of what the policy does, so it is meant to lose the other one.
Every strategy in the run plays the same invented field — the rivals never see your picks,
so one seed gives them all the identical opposition — which is what makes the finishes
comparable. And a season is a single trial: a finish here, or fifteen of them, settles
nothing either way.

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
whether to commit or hold for injury news. With the pool's reports imported it also prints
each candidate's expected share of the pot; see [deciding by share of the
pot](#deciding-by-share-of-the-pot).

## Tuning

Every modeling knob lives in `src/pool/config.py` (shrinkage weights, home edge,
depth-chart role multipliers, future discount, information premium).

The pot-share block there is annotated with what each number is and is not. `K_GAME` and
`THROW_SHARE` are fitted and measured against 2011–2025. `RIVAL_NOISE_TOP_N` is
**provisional and says so**: the real distribution is what `pool predict` measures, and it
is meant to be replaced by that measurement in writing, with the date. `WINPROB_SEED` is
fixed so identical inputs give identical advice — a policy that answered differently on
re-run could not be reconstructed from its own capture.

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
