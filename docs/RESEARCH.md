# Research and audit

Nothing on this page is needed to pick. It covers what the tool keeps as evidence while you
pick — decision records and rival predictions — how the pot share is computed, and the
retrospective studies behind the model: backtests, forecast evaluation, calibration and tuning.
The weekly workflow is in the [README](../README.md); how the project got here is in
[HISTORY.md](HISTORY.md).

## Decision records

`pool week` and `pool recommend` save the decision they show. `pool week` does so only while a
slot is still open, and the `record` line it prints names the decision with `--decision`, so a
pick links to the advice it came from; `recommend` saves one on every run unless given
`--no-capture`.

A decision record holds the whole pre-pruning surface for every slot and
remaining week, the advice and hold-or-commit call per slot, the used and locked state, the
feed observations that were visible, and the code, constants and model that produced it. The
log is append-only — a correction is a new event, and `record`/`unrecord` append their own —
so a captured decision can be reconstructed later even after the feeds have moved on.
Outcomes are never stored beside it; they are joined from finalized scoring when read. This
evidence is what [Phase 3](IMPLEMENTATION_PLAN.md) needs and is not used to make picks.

`record --decision <id>` names the decision a pick came from: `week` writes it into the line
it prints, and `recommend` prints the id. Without it the fallback link — the most recent advice
for that slot — is whichever run happened last, which with a Thursday and a Sunday run in one
week is not always the one the pick came from.

No protocol governs how often decisions are captured any more —
[3C is closed](PHASE3C_OUTCOME.md) and no capture is required at any point in the week.
Capturing costs nothing and keeps the option: `pool research captures` lists what you have
and `pool research verify-capture` checks it, and because the enforced source fingerprint
covers only the decision path, an unrelated feature does not make yesterday's captures
unverifiable.

```bash
uv run pool research captures --season 2026 --week 1   # captured decisions
uv run pool research captures --decision 95ed0d28cbe6   # one in full; the listed id is enough
uv run pool research verify-capture --season 2026      # reconstruction and live/snapshot parity
```

The existing 2026 captures require `uv run pool research verify-capture --season 2026
--allow-code-drift`, and have since stage 2. The
enforced closure has moved four times: stage 2 extracted the shared pick-scoring core into
`scoring.py`, a later review fix to the same file — scoring a pick with no game zero once its
week is final — moved it again, stage 3 added `rivals.py` and `simulate.py` when
`recommend` gained the second objective, and on 2026-09-19 pick scoring moved out of
`scoring.py` into `results.py`, outside the closure, in the same PR as the rival fix below
(fingerprint `be2b9bbb…` to `23c28ae1…`; revision `67ab143` is the last before it). Each move
was an isolated change, with the pre-move output recorded first. Every capture still reconstructs and matches replay under the
override, which explicitly reports that the fingerprint check was bypassed. The exceptions
are the three captures that carry a pot share (`cf9cc9e5d64d`, `8edc0f08b08e`,
`fd9c03a2946a`): a later fix changed the pot-share arithmetic, so current code re-derives
different shares from them. They reconstruct exactly at the revision they recorded,
`89dca93`, and their expected-TD advice and replayed surfaces match today.

`95ed0d28cbe6` (week 2, made Friday 2026-09-18) used to fail parity for another reason. The
snapshot replay folds the pick deadline into `hard_eligible` and the live path does not, and
the rival simulation read that column, so on replay no rival could take a player from
Thursday's game. The two paths then disagreed about which rivals might hold an RB. A rival
may have made that pick before kickoff, so rivals now judge players by availability alone
on both paths, and this capture verifies under the override like the others. See the
recorded verifications for [stage 2](PHASE4_STAGE2_PLAN.md#implementation-verification--2026-09-15)
and [stage 3](PHASE4_STAGE3_PLAN.md#implementation-verification--2026-09-17).

Replay rebuilds the inputs a decision read, exactly as it read them. A week still being
played at the decision is restored as it stood, through the same legacy-touchdown fallback
the live path used, so a capture made mid-week reaches parity like any other. The archive
holds only what was observed by the decision, so waiting for a week to be scored changes
nothing about it. Research replays keep requiring complete history, because they ask a
different question.

`verify-capture` re-derives each decision's advice from its stored surface alone, then
rebuilds the same instant from the archived feeds and compares the model columns. The
rebuild runs under the constants the decision recorded, so a setting that has moved since
is not reported as a live/archive divergence. The archive is written by `refresh`, so a
decision made without one cannot be verified and is reported as unverified rather than
skipped; a season with no captures at all exits nonzero, because nothing verified is not
verification. The fingerprint it compares covers what a decision is a function of — the
import closure of `recommend`, `projections` and `snapshots`, plus the installed version of
every runtime dependency — and not the whole tree. It is read from the package itself, so it
works installed, and captures made before it replaced `uv.lock` report the decision path as
moved; the whole-tree hash is still recorded beside it, so drift stays visible without
an unrelated module invalidating a capture it could not have changed. Both checks live in
`src/pool/research/verify.py`.

The [3C protocol](../experiments/phase3c-baseline.toml) that these checks were built for
declared a collection window, decision events, populations and floors, and `pool baseline`
described captures under it. The window closed without being run, and the protocol
machinery and `baseline` were removed on 2026-09-19. Revision `bc57683` is the last one
that has them, for anyone reproducing that tooling exactly.

## The prediction log

The one place this work gets to be empirical. Predict every rival's picks, archive the
prediction, read the report, score it — and the prediction must be recorded before it could
have seen the answer.

```bash
uv run pool research predict record --season 2026 --week 3            # archive before kickoff
uv run pool research predict record --season 2026 --week 3 --dry-run  # show it, archive nothing
uv run pool research predict score --season 2026                  # hit rates and the PIT histogram
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
observed integer because the counts are discrete. What gets archived is the *outcomes*:
every player-week in the frame, drawn once before kickoff and stored (about 130 KB a week),
beside the seed, parameters and projection-frame hash that produced them. Scoring reads
them back, so neither a re-forecast nor a later change to the simulator or numpy can move
them. Commitments archived before outcomes were stored are drawn once from their seed the
first time `predict score` runs, then stored the same way and never drawn again. It is
reported with no verdict: 85 draws a season cannot test uniformity with any power, and the
lean is the readable part.

`pool week` makes the prediction for you on each run of the current week until its first
kickoff or its report, and saves again only when it changed; the scorer reads the last one
that beat both deadlines. `pool research predict record` remains for a prediction made by hand.

## How the pot share is computed

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

The Python API `standings.remaining_counts(conn, season, week, entrant_id)` counts known
remaining players by slot from `state.historical_pool`, spending only what was used **before**
that week, so a report that has already arrived for the week itself cannot shrink its own
answer; callers should inspect `used_pools` for unknown usage and missing weeks. The pot share
is what reads these pools.

## Backtesting

```bash
uv run pool refresh --season 2025                 # 2025, and 2024 (prior) until it is settled
uv run pool research backtest --season 2025       # replay one season against the baselines
uv run pool research backtest --season 2017-2025 --detail
uv run pool research sweep --season 2017-2025     # grid-search the discount and prior weight
# pot share, against invented rivals:
uv run pool research backtest --season 2025 --strategy winprob,optimizer
```

Two layers are measured separately, because a better forecast is not the same
thing as a better season:

```bash
uv run pool research models                       # the benchmark's projection models
uv run pool research evaluate --season 2011-2025  # score every player-week forecast
uv run pool research backtest --season 2017-2025 --projection player-vegas
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
uv run pool research evaluate --season 2024-2025 --model all --seeds 0-19 --csv /tmp/forecasts.csv
uv run pool research evaluate --model player-vegas,regressed-rate --baseline player-vegas
uv run pool research backtest --season 2025 --input-policy legacy-closing --vegas-horizon 0
uv run pool research backtest --season 2026 --input-policy snapshots --decision-times decisions.csv
uv run pool research benchmark --config experiments/phase2-validation.toml --output data/experiments/phase2-validation
# Continue only when configuration, code, dependencies and frozen dataset match:
uv run pool research benchmark --config experiments/phase2-validation.toml --output data/experiments/phase2-validation --resume
uv run pool research diagnose --run data/experiments/roster-snapshot-repair --out experiments/results/phase3a-readiness
uv run pool research benchmark --config experiments/phase3-calibration.toml --output data/experiments/phase3-calibration
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
[experiments/README.md](../experiments/README.md) for the stages and artifacts.

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
A documented, guarded [source correction](../experiments/scoring-corrections.json) resolves two
duplicate source TD plays in one 2011 game; original observations remain archived.

## Tuning

Every modeling knob lives in `src/pool/config.py` (shrinkage weights, home edge,
depth-chart role multipliers, future discount, information premium).

The pot-share block there is annotated with what each number is and is not. `K_GAME` and
`THROW_SHARE` are fitted and measured against 2011–2025. `RIVAL_NOISE_TOP_N` is
**provisional and says so**: the real distribution is what `pool research predict` measures,
and it is meant to be replaced by that measurement in writing, with the date. `WINPROB_SEED` is
fixed so identical inputs give identical advice — a policy that answered differently on
re-run could not be reconstructed from its own capture.
