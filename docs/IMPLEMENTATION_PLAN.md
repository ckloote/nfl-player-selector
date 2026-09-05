# Implementation roadmap

The September 4 review's repair sequence takes precedence over the original feature phases.
The original benchmark and backtest reports are preserved in [archive](archive) with their old
scoring, temporal and evaluation definitions. Current results use the saved
[candidate-pool repair specification](../experiments/roster-snapshot-repair.toml), which reruns
the [Phase 2 specification](../experiments/phase2-validation.toml) on the identical frozen dataset
after F11. The superseded Phase 2 results remain published for comparison.

| Work | Status |
|---|---|
| Data import, projections, assignment, recommendation CLI | Implemented |
| Weekly reliability (review step 1) | Implemented: F01, F05, F09 |
| Validation repairs (review step 2) | Implemented: F02–F04, F06, F08, F10; F07 documentation corrected |
| Candidate-pool repair (review step 2) | Implemented: F11; benchmark rerun on the same frozen dataset |
| Calibration validation (review step 3) | Pending; no production calibration or tuning changes |
| Leaderboard strategy (review step 4) | Pending |

## Weekly reliability

The CLI supports refresh → recommend → record → score, with complete TD credits, conservative
game coverage, per-pick deadlines, unconfirmed-kickoff exclusions, preserved locks, atomic
validated recording and feed freshness warnings. Historical entries and corrections remain
supported. Missing final results stay pending instead of becoming zero. Versioned database
migrations preserve picks and imported history.

## Phase 2 — validation repairs

- Explicit hard eligibility is independent of forecast value. Out/Doubtful players remain
  excluded before pruning, ranking and assignment; eligible zero estimates and deterministic
  player-ID ties are retained. Deadline restrictions and recorded locks remain enforced.
- Model/baseline validation precedes loading. A selected deterministic baseline supplies one
  greedy depletion history shared by all comparisons. Candidate keys and eligibility/depletion
  masks are checked, including future projection cells.
- Common-pool top-1/3/5/10 results are ranking diagnostics in TDs per ranked candidate. Paired
  slot-week differences are averaged within season, with uncertainty across seasons. Empty cells
  and coverage are explicit. Each model's actual greedy/optimizer replay uses its own history.
- Both shuffled models run seeds 0–19, permuting only eligible availability-free forecasts and
  reapplying recipient Questionable adjustments. Deterministic models run once. Seed averages
  precede season uncertainty; seed variability is separate from cross-season evidence.
- Historical replay masks current/future closing lines before team/league averages, uses stats
  through W−1 and reports through W, and retains usage roles. `legacy-closing` is an explicitly
  labeled sensitivity policy. Latest depth charts are rejected for historical replay.
- Refresh atomically archives normalized schedule/lines, rosters, injuries, depth charts,
  stats and compact scoring data. Successful empty feeds create observations; failures preserve
  data. Payloads are compressed/deduplicated. Legacy observations are never invented.
- Snapshot replay resolves observations by an explicit timezone-aware timestamp for every
  requested week. It defaults to depth roles, records absent/stale optional inputs and fails
  on missing essential history. Forecast inputs are separate from final scoring actuals.
- The benchmark configuration fixes 2011–2025 and prior history from 2010, all ten models,
  shipped constants, both season strategies, shipped random-top-10 trials and hindsight.
  Every required game is audited before the research database is frozen. One documented
  upstream duplicate-TD correction has strict preconditions and preserves original observations.
- Checkpoints, resolved configuration, source/dataset/dependency fingerprints, feed provenance,
  coverage, forecasts, future surfaces, picks and seed/season metrics make results reviewable.
  Resume rejects incompatible inputs. Reports are generated from saved metrics.

Acceptance requires full coverage and all configured seasons, models and seeds. A favorable
model result or production parameter change is not an acceptance criterion. Final schedule
revisions, weekly roster/injury timing and later statistical corrections remain limitations of
historical replay. Both 2011–2018 and 2019–2025 summaries are retrospective.

## Phase 3 — calibration validation

F07's documentation corrections are complete; its empirical calibration work remains open.
The shipped Poisson calibration slope now sits between 0.817 and 0.892 in all fifteen seasons
(mean 0.867, SD 0.022) once F11 is repaired, so forecasts are too extreme by a consistent and
replicable amount rather than by an amount that varies with the season. That is the quantity
Phase 3 has to explain before it corrects anything.
Validate by position, projected-rate range and selected/available population. Check transfer
from historical usage roles to the live depth-chart model. Evaluate changes in assignment
and hold/commit choices when forecast scale changes. Positive monotone corrections preserve
greedy ranking but do not generally preserve sum-maximizing assignment.

Keep production constants fixed until a separately specified experiment supports a change.
Joint context ablations do not identify defense's marginal contribution. Comparing the current
models does not establish that their input information is exhausted. Pooled Poisson diagnostics
do not validate conditional tails or independence. Separate per-type rates, alternative usage
features and negative-binomial dispersion remain research options, without claims of either
proven benefit or permanent rejection based on the superseded experiments.

## Later — leaderboard strategy and optional interface work

Start with official opponent-report ingestion, opponent picks and standings. Simulations must
share one sampled player-week outcome across every entrant selecting that player, with relevant
player correlations, ties, uncertainty about opponent choices and future policy updates.
Validate conditional tails and simulation behavior before treating a simulated win-probability
gain as an established advantage. Repeated decisions within a week are beyond Phase 2 replay.

A local dashboard, deadline reminders and additional input providers remain optional. The
projection-frame interface and assignment solver continue to support expected-TD planning;
fixed-matrix optimality does not establish a rolling-policy advantage over greedy.
