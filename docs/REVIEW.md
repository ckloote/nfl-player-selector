# Project review — 2026-09-04

Reviewed revision: `53e8c83` (`Benchmark the projection layer on every player-week forecast`).
The working tree was clean when the review began. All findings were open at that
revision. The original evidence below is retained; the status notes distinguish
subsequent implementation from the original review.

The project has a sound separation between import, projections, assignment,
recommendation, and evaluation, with useful tests. Keeping the assignment solver
for planning and season-cost explanations is reasonable. Its season-scoring
advantage remains unproven. Live eligibility and the evaluation's interpretation
need attention before leaderboard-aware simulation becomes the priority.

## Scope and verification

The review covered the source, tests, README, design, implementation plan,
backtest report, projection benchmark, and local database coverage.

- Existing suite: **121 tests passed**; Ruff lint and formatting checks passed.
- Targeted checks used disposable fixtures and a read-only connection to the
  existing historical database. No real picks or imported data were changed.
- Selected 2024–2025 season replays and 2025 forecast comparisons were rerun.
  The full fifteen-season, eight-model benchmark was **not** independently rerun.
- The local database contained player stats for 2010–2025, a 2026 week-one
  roster, and a 2026 depth snapshot dated September 2. The recorded 2026 refresh
  timestamp was `2026-09-02T21:54:22`. No 2026 injury rows were present; preseason
  absence alone does not establish that the feed is broken.

The command sandbox failed to initialize during the review, so approved commands
ran outside it. That tooling failure is not a project defect.

## Findings

### F01 — High: recommendations do not enforce elapsed pick deadlines

**Current status:** Resolved in the weekly reliability phase: one Eastern decision time, forbidden deadline cells before assignment, unconfirmed kickoff handling, preserved locks/future weeks, and boundary/CLI tests.

**Evidence:** [`advise_slot`](../src/pool/recommend.py) defines playable candidates
using matrix value alone. Neither that function nor the recommendation CLI
compares candidate deadlines with the current time. Deadlines are displayed but
do not constrain the solve.

A fixture with an RB kicking off September 3 at 20:15 and a lower-projected RB
kicking off September 6 still recommended the first player when reviewed on
September 4. Its displayed deadline was September 3 at 19:15.

**Impact:** rerunning recommendations after Thursday's deadline can suggest a
pick that can no longer be submitted, contrary to the documented within-week
workflow.

**Resolution:** pass an explicit decision time into the live recommendation path
and forbid expired cells for the current week. Preserve those players' eligibility
in future weeks and preserve already-recorded picks. Verify the exact deadline
boundary, timezone conversion, and re-solving after an early game.

### F02 — High: the replay still admits future information through betting lines

**Phase 2 status:** Addressed in Phase 2: historical closing lines are masked for weeks ≥ W before all averages; snapshots resolve observed inputs by explicit timestamp. `tests/test_phase2.py` checks every model, week-one fallbacks, timestamp boundaries and later corrections. Final schedule revisions, weekly report timing and corrected historical stats remain approximations.

**Evidence:** [`load_frames`](../src/pool/projections.py) preserves closing lines
through `as_of_week + VEGAS_HORIZON_WEEKS`. A future game's closing line is not
the advance line that existed at the replay's decision time. Moreover,
`game_context` uses all retained lines to calculate league and team averages.

In the four-week test fixture, changing only week four's total line from 45 to
75 changed all 24 week-one forecasts; one changed from approximately 0.7243 to
0.6208 expected TDs. The current freezing test preserves this same horizon and
therefore does not detect the problem.

Sensitivity checks on the existing database produced:

| Season | Future-line horizon | Optimizer TDs | Greedy TDs | Difference |
|---|---:|---:|---:|---:|
| 2024 | 6 | 56 | 60 | −4 |
| 2024 | 0 | 62 | 60 | +2 |
| 2025 | 6 | 67 | 64 | +3 |
| 2025 | 0 | 68 | 64 | +4 |

These are sensitivity results, not evidence that a zero horizon is optimal.
Even horizon zero retains current-week closing lines and the existing weekly
roster/injury timing approximations. Giving all strategies the same leaked
inputs does not guarantee that their relative performance is unbiased.

**Resolution:** qualify the “known an hour before kickoff” claims. Save timestamped
line, roster, injury, and depth inputs for prospective use. For historical replay,
use genuine archived snapshots where available and label remaining approximations.
Check that changing data published after the decision time cannot change its
projections or choices.

Reproduce the historical sensitivity checks with:

```bash
uv run pool backtest --season 2024-2025 --strategy optimizer,greedy --input-policy legacy-closing --vegas-horizon 6
uv run pool backtest --season 2024-2025 --strategy optimizer,greedy --input-policy legacy-closing --vegas-horizon 0
```

### F03 — High: a common-pool ranking metric is presented as season scoring

**Phase 2 status:** Addressed in Phase 2: common-pool top-k is labeled TDs per ranked candidate with season-level uncertainty. Actual greedy/optimizer replays use separate no-reuse histories. The synthetic repeated-leader test demonstrates why rankings are not season scores. Evidence: corrected reports and saved ranking/replay artifacts.

**Evidence:** [`forecast_set` and `forecasts`](../src/pool/evaluate.py) rank every
challenger against players remaining after the baseline model's greedy walk.
The challenger does not deplete that pool with its own choices. It can therefore
receive credit for selecting the same player in multiple weeks if the baseline
has not spent that player.

This is a useful comparison of forecasts in common situations, but the comments
calling depleted top-one “exactly what greedy scores” do not hold for challengers.
On the existing 2025 data, with the shipped model as baseline:

| Measure | Shipped | Player-vegas | Challenger minus shipped |
|---|---:|---:|---:|
| Sum of top-one outcomes in the shared depleted pool | 64 | 74 | +10 |
| Actual greedy season replay | 64 | 62 | −2 |

Likewise, summing differences in top-ten average outcomes across slot-weeks gives
a scaled ranking statistic, not an achieved season-score improvement. The
benchmark's claim that contextual multipliers are “worth about 2.5 TD/season”
overstated what its primary metric establishes. The report also explicitly
disclaims season-level conclusions, creating an internal contradiction.

**Resolution:** label common-pool top-k results as ranking metrics, with their
units and baseline depletion policy stated. Use each challenger's actual replay
for claims about season scoring. Retain the common-pool comparison as a diagnostic
without treating its top-one result as an equivalent season-level gate.

The 2025 discrepancy is visible by comparing the top-one gate from
`pool evaluate --season 2025 --model shipped,player-vegas` with greedy backtests
using `--projection shipped` and `--projection player-vegas`.

### F04 — Medium: custom baselines can silently lose the shared candidate pool

**Phase 2 status:** Addressed in Phase 2: the requested selected deterministic baseline is validated before loading and passed through generation. Its depletion history is shared across all models/seeds, with candidate and mask assertions. Custom-baseline API and CLI export tests run without shipped.

**Evidence:** [`evaluate` in the CLI](../src/pool/cli.py) passes `--baseline` to
rendering but does not pass it to `forecast_set`. The latter defaults to `shipped`.
When that model is absent, `common` remains `None`, and each model falls back to
its own depletion history.

A 2025 comparison containing only `player-vegas` and `regressed-rate` produced
**288 player-week rows with different availability** between the models. This
reintroduces the per-model depletion bias described in the benchmark document.

**Resolution:** pass the requested baseline into forecast generation, require it
to be present or explicitly construct its common history, and reject invalid
baseline names before doing the expensive work. Verify identical availability
masks across all models in a comparison, including when `shipped` is absent.

### F05 — Medium: scoring omits touchdowns covered by the written rules

**Current status:** Resolved in the weekly reliability phase: compact scorer/passer credits from play-by-play, conservative game coverage, shared complete scoring for projections/replay/hindsight and `pool score`, with return/recovery and pending-result tests. The corrected Phase 2 benchmark uses complete accounting; the prior reports are archived under their old definition.

**Evidence:** [`transform_player_stats`](../src/pool/ingest.py), the database
schema, and [`actual_tds`](../src/pool/backtest.py) retain and sum only passing,
rushing, and receiving touchdowns. The written rule credits every touchdown a
pick scores or throws. nflverse also exposes `special_teams_tds`, covering
touchdowns on special-teams plays, in its
[official stat definitions](https://nflfastr.com/reference/nfl_stats_variables.html).

**Impact:** under the written rule, a return touchdown by an eligible player is
missing from imported scoring data and replay results.

**Resolution:** establish the exact pool scoring categories. If the written rule
is correct, import all applicable touchdown types without double-counting and
use one scoring definition for projections, historical actuals, and `pool score`.
If the pool excludes return touchdowns, document that exception instead.

### F06 — Medium: “untouched holdout” overstates research independence

**Phase 2 status:** Addressed in Phase 2: both era summaries are retrospective. The dated TOML specification, source/dataset/dependency fingerprints, seeds, coverage, per-season checkpoints and saved exports record the research configuration. No untouched-holdout claim is made.

**Evidence:** [`BACKTEST.md` §5](BACKTEST.md) records model searches on 2017–2025.
The later [`PROJECTION_BENCHMARK.md`](PROJECTION_BENCHMARK.md) calls 2019–2025 an
untouched holdout. Those seasons were already examined, and the earlier ablation
results helped select the benchmark metrics.

Not fitting a particular parameter on those seasons is useful separation, but it
does not erase their earlier influence on model and evaluation choices. This
does not make every reported result false; it weakens claims of independent
confirmation.

**Resolution:** distinguish an experiment's parameter-fitting split from a fresh
holdout across the research program. Preserve dated specifications, dataset
versions, seeds, and metric changes. Label retrospective comparisons appropriately
and reserve prospective data, or use a fully specified nested evaluation, for
stronger confirmation claims.

### F07 — Medium: several model and roadmap conclusions exceed the evidence

**Phase 2 status:** Documentation corrections complete in Phase 2. README, design, roadmap and replacement reports qualify all five claims below. Calibration validation by position/selection strata, transfer to live depth roles, and assignment/hold behavior remain Phase 3 work.

The following claims should be corrected when reconciling the design, plan,
backtest, and benchmark documents:

- **Calibration cannot change a pick:** a positive monotone transform preserves
  within-week greedy rankings. It does not generally preserve a sum-maximizing
  assignment or the fixed-TD information-premium decision. The benchmark itself
  reports assignment changes in two of seven seasons. A noisy observed effect
  is not proof that the correction is worth exactly zero expected TDs.
- **Defense alone must stay:** removing defense, home, and role together measures
  their combined contribution. It does not identify the marginal contribution of
  defense alone. The documented joint comparison cannot by itself refute dropping
  `def_mult`.
- **The information set is exhausted:** comparing eight largely hand-built models
  does not measure the best achievable prediction from their inputs. New usage
  features are plausible candidates, but this comparison does not establish a
  ceiling or rule out better estimation of existing features.
- **Poisson settles simulation readiness:** approximate variance/mean agreement
  in pooled projection bins does not establish conditional tail probabilities or
  independence between players. Those matter for finishing-first probabilities.
  Poisson is a reasonable starting model; the distribution question is not closed.
- **Greedy is provably worse:** exact assignment optimizes a specified fixed
  projection matrix. That mathematical property does not establish a strict
  advantage over a rolling greedy policy with uncertain, changing projections.
  The design's wording should reflect the distinction and the measured uncertainty.

**Resolution:** state conclusions at the level each experiment supports. Validate
calibration by relevant position and selection strata, and check transfer from
the replay's usage-based roles to the live depth-chart model. Evaluate assignment
and hold/commit behavior whenever the scale of projected TDs changes.

### F08 — Medium: shuffled benchmark models can restore unavailable players

**Phase 2 status:** Addressed in Phase 2: hard eligibility is independent of lambda and applied before pruning, ranking and assignment. Both nulls shuffle only eligible availability-free cells and reapply recipient Questionable adjustments. Twenty-seed exclusion, grouping, zero-estimate and strategy tests cover the contract.

**Evidence:** both shuffle builders in [`models/baselines.py`](../src/pool/models/baselines.py)
permute the final `lam`, which already includes availability. They do not preserve
the relationship between a player's injury status and a zero projection.

In a fixture, a player marked `Out` with `avail_mult = 0` received positive `lam`
from both shuffled models. Since the optimizer forbids cells using `lam <= 0`,
that player became playable. The synthetic reproduction demonstrates a contract
failure; its impact on published season results was not quantified.

**Resolution:** preserve hard eligibility independently of randomized forecasts.
Specify whether the null is intended to destroy matchup information, availability
information, or both; the current documentation promises common eligibility.
Verify that null models cannot restore ruled-out cells.

### F09 — Medium: out-of-range recorded weeks consume players

**Current status:** Resolved in the weekly reliability phase: shared schedule-based week validation, historical player lookup, atomic multi-slot recording, and score invalidation on replacement.

**Evidence:** [`record_pick`](../src/pool/state.py) validates position and reuse
but not whether the week belongs to the season. A fixture accepted week 99,
after which `used_ids` treated that player as spent for the season. The `record`
CLI does not run the week-range validation used by recommendation commands.

**Resolution:** validate the week against the loaded regular-season schedule
before writing a pick. Manual recording of legitimate historical picks can remain
supported. Verify that invalid input leaves existing picks and used-player state
unchanged.

### F10 — Medium: the documented random benchmark differs from the default runner

**Phase 2 status:** Addressed in Phase 2: evaluate defaults to seeds 0–19 for each shuffled model; deterministic models run once with seed sentinel −1. Per-seed metrics average within season before uncertainty across seasons; shuffle SD is separate. The committed benchmark config distinguishes shuffled models from shipped random-top-10 strategy trials.

**Evidence:** the benchmark describes 20 seeded permutations, but
[`models.BUILDERS`](../src/pool/models/__init__.py) registers the `random`
projection model with only `seed=0`, and the default evaluator does not iterate
over seeds. The season harness's 20 random-strategy trials are a separate mechanism.

**Resolution:** either implement and document the intended multi-seed forecast
evaluation, including its aggregation, or describe the shipped single-seed
benchmark accurately. Store a runnable configuration for the reported analysis
so its seed count and reported uncertainty can be reproduced.

## Current status and recommended sequence

The implementation status is broadly accurate: imports, projections, assignment,
the recommendation CLI, season replay, and forecast evaluation exist. The weekly
reliability phase now also implements `pool score`, feed freshness/coverage status,
and the live deadline workflow. F01, F05, and F09 are resolved. Opponent tracking,
standings, and win-probability strategy remain unfinished. F02–F04, F06, F08 and F10 are addressed in Phase 2. F07 documentation corrections are
complete; its empirical calibration validation remains Phase 3 work.

Verification includes mocked refresh → recommend → record → score integration,
migration rollback/idempotence, failed-feed preservation, and a temporary 2025
historical import: 272 complete regular-season games, 2,206 throwing/scoring
credits, and correct passing and return-TD pick scores. No benchmark was rerun during the weekly reliability step; Phase 2 supplies the replacement results.

The [roadmap](IMPLEMENTATION_PLAN.md) now distinguishes shipped behavior, corrected validation,
and pending calibration/simulation research. Per-type rates and alternative count distributions
remain research options without claims of established benefit or definitive rejection.

Phase 2 evidence is in [the benchmark](PROJECTION_BENCHMARK.md), [actual replays](BACKTEST.md),
[`experiments/phase2-validation.toml`](../experiments/phase2-validation.toml), and the saved compact
artifacts. The research dataset audits all 4,175 scheduled regular-season games from 2010–2025
before freezing. Explicit end-of-game score markers replace numeric play-ID ordering. One
[guarded source correction](../experiments/scoring-corrections.json) removes duplicate rushing-TD
records in the 2011 Detroit–New Orleans feed, reconciled to the official game report; original
observations remain archived. The operational database is unchanged.

Recommended order:

1. **Make the weekly workflow reliable.** Address deadline eligibility, scoring
   semantics, and pick validation; complete `pool score`; expose freshness and
   missing-feed status for the inputs on which a recommendation depends.
2. **Repair and document validation.** Fix custom-baseline handling and shuffled
   eligibility. Correct metric labels and temporal claims. Start preserving
   timestamped inputs, and save reproducible experiment configurations before
   making further holdout claims.
3. **Reassess calibration.** Use the corrected evaluation and verify behavior for
   the actual live role model, relevant positions, assignment choices, and
   hold/commit decisions. Improved calibration is valuable for expected-TD
   interpretation even without an established season-score gain.
4. **Then build leaderboard strategy.** Start with scoring and opponent-report
   state. In simulations, every entrant choosing the same player must receive the
   same sampled player-week outcome. Account for relevant player correlations,
   tie rules, uncertain opponent choices, and future policy updates. Validate
   these assumptions before treating a change in simulated win probability as
   an established advantage.

The review does not recommend replacing the assignment solver or shipping new
model constants on the strength of the two replayed seasons. Its recommendation
is to make existing decisions and the evidence supporting them dependable before
adding a more demanding objective.
