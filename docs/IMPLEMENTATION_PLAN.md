# Implementation roadmap

The September 4 review's repair sequence takes precedence over the original feature phases.
The original benchmark and backtest reports are preserved in [archive](archive) with their old
scoring, temporal and evaluation definitions. The Phase 2 results use the saved
[candidate-pool repair specification](../experiments/roster-snapshot-repair.toml), which reruns
the [Phase 2 specification](../experiments/phase2-validation.toml) on the identical frozen dataset
after F11. The superseded Phase 2 results remain published for comparison.

| Work | Status |
|---|---|
| Data import, projections, assignment, recommendation CLI | Implemented |
| Weekly reliability (review step 1) | Implemented: F01, F05, F09 |
| Validation repairs (review step 2) | Implemented: F02–F04, F06, F08, F10; F07 documentation corrected |
| Candidate-pool repair (review step 2) | Implemented: F11; benchmark rerun on the same frozen dataset |
| Phase 3: readiness, calibration and shadow validation (review step 3) | 3A complete; 3B completed 2026-09-07 with no candidate promoted. Next: identity-only prospective baseline protocol/capture for 3C; production unchanged |
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
  Resume checks recorded identities, including the actual decision-times CSV contents as
  repaired in Phase 3A. Reports are generated from saved metrics and contain facts, methods
  and provenance; dated human/AI interpretation is separate in [ANALYSIS.md](ANALYSIS.md).

Acceptance requires full coverage and all configured seasons, models and seeds. A favorable
model result or production parameter change is not an acceptance criterion. Final schedule
revisions, weekly roster/injury timing and later statistical corrections remain limitations of
historical replay. Both 2011–2018 and 2019–2025 summaries are retrospective.

## Phase 3 — calibration validation

**Status as of 2026-09-07 (UTC):** 3A and the full 3B experiment are complete; 3C validation
remains planned. The [published calibration report](../experiments/results/phase3-calibration/CALIBRATION.md)
and [authored outcome](PHASE3B_OUTCOME.md) apply the unchanged signed margins: no candidate
passes every gate. Position-specific log-affine calibration passes the forecast gate but
does not establish either policy's required non-inferiority. Production stays unchanged;
no calibrated candidate advances into 3C. The next step is an identity-baseline prospective
protocol and capture, not further tuning on these inspected results.

Scope remains to diagnose rate errors, test past-only mappings and validate their decision
effects without changing the underlying model. Full projection/action logging is implemented
in 3A; continue prospective capture using it. Initial 2026 snapshots are not completed live
validation, and descriptive research does not require waiting for an entire prospective season.

| Stage | Dependency | Exit Deliverable |
|---|---|---|
| 3A: Evidence, capture and readiness | Implemented 2026-09-06 | Guarded diagnostics, reproducible capture, regression tests and a dated [readiness note](../experiments/results/phase3a-readiness/READINESS.md) |
| 3B: Chronological train/apply experiment | Completed 2026-09-07; no promotion | Frozen [specification](../experiments/phase3-calibration.toml), [published metrics](../experiments/results/phase3-calibration), fitted artifacts/surfaces and [outcome note](PHASE3B_OUTCOME.md) |
| 3C: Shadow-live transfer and policy validation | Next: identity baseline only; no 3B candidate qualified | Reviewed prospective baseline protocol, matched live/snapshot evidence, then a dated validation assessment; candidate promotion requires separate qualifying evidence |

### Phase 3A: Evidence And Readiness

**Implemented.** Each repair below has fixtures in `tests/test_phase3a.py`; the descriptive
export is `pool diagnose`, saved in
[experiments/results/phase3a-readiness](../experiments/results/phase3a-readiness). The
deliverable table records what was required.

| Deliverable | Required Tests And Acceptance |
|---|---|
| Guard `evaluate.poisson_glm` and calibration summaries | Check design rank/identifiability, finite inputs, convergence and iteration exhaustion. Constant log rates, all-zero outcomes, empty inputs and `n <= 2` must yield explicit unsupported-fit reasons, not plausible coefficients/intervals. Insufficient clusters (including one) must suppress cluster SEs; predeclare a minimum for inferential use. Test an identifiable converged fixture against a trusted reference. |
| Separate hold policy from display truncation in `recommend.advise_slot` | Evaluate the best feasible later alternative within the solver candidate pool before truncating display. Vary `n_alternatives` from zero through small/default/large values in a fixture whose best later option is outside the displayed set: recommended pick, hold flag, hold alternative and its cost must agree. Preserve deadline, hard-mask and locked-slot tests. |
| Align live/snapshot same-week stats policy in `projections.load_frames` | Define whether already-observed completed early-game stats enter later decisions, then use that rule in both paths. At the same timestamp, archived inputs, role source and used/locked state must produce equal current/future forecasts and advice. Test a post-Thursday/pre-Sunday decision, timestamp boundaries and later-import noninterference. Historical `stats < W` remains explicitly approximate. |
| Freeze decision-time inputs for `benchmark` resume | Copy and hash actual `season,week,decision_at` CSV contents into the run identity before workers start; workers use the frozen copy, not a mutable path. Same-path content edits must reject resume; missing/duplicate/invalid timestamps must fail. Verify unchanged inputs resume and a resumed run matches an uninterrupted run. |
| Append-only prospective decision capture | Store the full pre-pruning current/future surface, not just selected/displayed rows: decision/event ID and timestamp; player/season/target-week keys, position/slot and lead time; original and mapped rates; hard eligibility, availability adjustment and used/locked state; model/calibrator identity, code/config hashes, fitted-artifact hash and input observation identities/hashes. Record plan/advice, hold/commit, submitted actions and later corrections as append-only events, with outcomes joined separately. Test full-surface coverage, event linkage, no overwrite and replay after later feed corrections. |

Deliver a descriptive diagnostic export for all eligible, available/depleted, common-pool top-k
and each policy's own selected populations. Include position (WR and TE separately), rate bins,
availability status, selection and forecast lead horizon; freeze diagnostic group definitions
without selecting on future TDs or participation. Report counts, outcome coverage, intercept
and slope with fit status/uncertainty, forecast/actual levels, reliability and proper metrics.
Keep eligible zero rates visible: explicitly define their scoring and positive-rate GLM
exclusion/counts, including zero forecasts with positive outcomes; never silently drop them.

Current-week fits do not establish future-horizon calibration. Diagnose raw future rates
separately from fixed-discount planning values. Repeated future forecasts share target outcomes;
predeclare weighting and use appropriate outcome/player-season or season clusters, with paired
season uncertainty for model/policy comparisons, not row independence. Seed variation is separate.

**3A acceptance:** all repair fixtures pass; one captured decision reconstructs its complete
surface and advice; exclusions, failed fits and missing coverage are explicit. Save the source,
data and metric identities with a dated readiness note. Descriptive artifacts and authored
reasoning stay separate. None of these checks by itself authorizes production calibration.

*Met 2026-09-06.* `capture.reconstruct` re-derives a captured decision's advice from its stored
surface alone; unsupported fits carry a reason and no coefficient; missing outcomes are reported
as missing, never as zeros; identities are in
[identities.json](../experiments/results/phase3a-readiness/identities.json) beside the note. The
export is descriptive: it states no research or deployment conclusion, and production constants
are unchanged. Interpreting it remains a separate, dated authoring step.

### Phase 3B: Chronological Experiment

**Completed with no promotion, 2026-09-07 (UTC).** See the
[report](../experiments/results/phase3-calibration/CALIBRATION.md) and
[outcome note](PHASE3B_OUTCOME.md). The runner is
`pool benchmark --config experiments/phase3-calibration.toml`,
which executes three stages with a barrier between each: identity forecast/outcome pairs for
every season, then one fit per fold, then the candidates applied and replayed. Fitting lives
in [`calibration.py`](../src/pool/calibration.py); fixtures are in `tests/test_phase3b.py`.
The declarations below are keys of the dated specification, and `benchmark.resolve` refuses a
run that is missing any of them.

**Margins approved 2026-09-07 (UTC).** The project owner signed off on the unchanged
thresholds and their accepted policy downside before the full benchmark. See the
[margin decision and rationale](PHASE3B_MARGIN_SIGNOFF.md). This is approval of the
pre-run decision rule, not experiment completion or a candidate promotion.

1. **Freeze the specification before fitting.** Deliver a dated
   `experiments/phase3-calibration.toml` with source/data/input hashes, candidate families,
   populations/weights, rate and lead-horizon strata, cutoffs, refit schedule, minimum samples,
   unsupported-fit fallback, seeds and multiplicity policy. Use expanding folds: train on
   forecast/outcome pairs from 2011 through Y-1, apply to Y for Y=2016,...,2025; 2010 supplies
   base-model history. No target outcome from Y enters its fit. Future-surface training rows
   must also have target outcomes before the cutoff, not merely early forecast timestamps.
   Record historical final-stat/timing approximations. All these seasons have been explored;
   this is retrospective walk-forward evaluation, not an untouched holdout. Reserve subsequent
   prospective/shadow data for confirmation and date any amendments before their evaluation.
2. **Predeclare estimands and decision margins.** Primary forecast metric: paired change in
   season-mean Poisson deviance on a fixed all-eligible current-week population, with zero-rate
   scoring specified. Declare horizon-specific proper-score diagnostics and population weights;
   slope/intercept, level ratios and common-pool ranks are diagnostics, not substitutes for
   proper scores. Separately estimate achieved TD/season changes for each policy versus its
   own identity replay. Put numeric practical improvement and acceptable policy-loss margins,
   uncertainty method, coverage requirements and promotion/defer rules in the dated config
   before fitting; a missing margin blocks execution. Non-significance is not equivalence,
   and a median SE across comparisons is not a universal detection threshold.
3. **Fit only the declared small families.** Identity/no change is mandatory. Compare level-only
   `c * lam` (`c > 0`) and log-affine `exp(a) * lam**b` (`b > 0`) candidates for positive rates;
   map eligible zeros to zero and preserve hard exclusions. Declare any position/lead-time
   grouping and sparse-group fallback before fitting, rather than choosing them on evaluation
   labels. Save coefficients, fit status, training keys/cutoff, model/input identity and artifact
   hash per fold. Fit on past data only; applying the artifact must not require evaluation labels.
4. **Apply to the full surface, then replay.** Map current and future rates before optimizer
   discount/pruning, leaving the discount, information premium and base-model constants fixed.
   Preserve keys, hard masks, availability semantics, zeros and used/locked state. Export raw
   and mapped surfaces and run independent no-reuse greedy/optimizer histories per candidate.
   A shared strictly increasing map preserves within-slot ranks; separate WR/TE maps can change
   FLEX ordering. Nonlinear mapping can change assignments and rate scaling can change fixed-TD
   hold comparisons. Report static `advise_slot` sensitivity separately, not as waiting-policy TDs.

**Tests and acceptance:** with forecast inputs held fixed, perturb every evaluation scoring
label and prove fitted coefficients, mapped forecasts and decisions stay unchanged. Observations
after the training cutoff must not affect the fit; imports after a decision must not alter that
decision's reconstructed forecasts or advice. Legitimate past-game updates to later base-model
inputs remain allowed. Identity must reproduce unchanged rates, picks and scores. Test
zero/mask/key preservation, current/future application order, sparse/failed fits, cross-position
FLEX rank changes and deterministic resume. Reconcile pick sums, no reuse, matched coverage and
paired season metrics. Deliver the frozen config, fold artifacts, out-of-fold surfaces, picks,
metric tables and a separately authored decision note, including negative/inconclusive results.
A favorable result is not required to complete 3B; production remains unchanged pending 3C.

*Implementation completed 2026-09-06; full-run outcome recorded 2026-09-07.*
The map is `exp(a) * lam ** b`: `level` fixes the exponent at one,
`log_affine` estimates both, and identity is mandatory. Each is fitted pooled and by position
with WR and TE separate, on the availability-adjusted rate the optimizer consumes, clustered on
the shared target player-week. Candidates are registered as builders, so they share the shipped
scaffold and `evaluate.assert_comparable` checks the keys and masks; each replays its own
no-reuse history through `backtest.replay`. The apply stage rebuilds identity rather than
reusing the pairs stage, so "identity reproduces unchanged rates, picks and scores" is checked
by comparing the two, over whole forecast rows, future surfaces and pick histories rather than
season totals. The surface carries the decision-time position it was built with, so a bye-week
forecast trains in the group it is applied by. `experiments/verify.py` works from the saved
resolved configuration, checked against the identity the run recorded; it reconciles that every
fitted artifact still hashes as recorded and carries its own declared fit, that no training key
reaches its own apply season, and that each candidate surface is exactly that artifact applied
to identity's rates, and that each fold's digest covers the values its coefficients are a
function of rather than only the keys. Outcomes are settled from the finalized scoring ledger
rather than from the target week's candidate pool, so a player who leaves the pool contributes
the zero he recorded instead of an unavailable outcome; membership travels separately as
retention. The run measures the outcome coverage its floors are stated over, counts every
training row it discards, and reports the proper score by horizon and availability. The primary
comparison is a paired season t with a Holm step-down, and the step-down's own decision is what
the promotion rule reads -- per-step levels widen down the ranking, so a later local interval
can exclude zero on a step the procedure never reached.

### Phase 3C: Shadow-Live Validation

**Recommended next deliverable:** a dated identity-baseline prospective protocol, reviewed
before evaluating subsequent outcomes. No calibrated candidate qualified in 3B; starting
baseline collection is not an exception to that promotion rule.

1. Fix the collection/review window, decision-event schedule, live role/input policy,
   diagnostic populations, coverage requirements and live/snapshot parity criteria.
2. Use the existing capture infrastructure to retain identity's full surfaces, advice,
   input identities and submitted actions at real events. Reconstruct with the same
   timestamp, roles and used/locked state; keep submitted picks and production settings unchanged.
3. Review matched coverage and horizon/availability diagnostics at the declared review point.
   Report baseline readiness separately from any claim of improved policy or waiting value.

Do not immediately rerun a calibration sweep. The opposing current/future results for
level maps motivate a bounded, separately specified horizon-weighting experiment if that
research is pursued, not replacement of 3B's primary metric. Any such experiment needs
fresh design review and confirmation requirements before execution; the already inspected
seasons cannot be relabeled as an untouched holdout, and the existing margins stay fixed.

Continue timestamped capture at real decision events while running identity and any frozen
candidate that later earns separate approval in shadow, without changing submitted picks.
Fix the candidate, live role policy,
training cutoff and review schedule before observing shadow outcomes; amendments start a new
evaluation window. Compare historical usage-role findings with the actual depth-role/current-line
surface by position, availability, selection and horizon. Record forecast/proper-score changes,
assignment/hold disagreements and coverage with appropriate paired/clustered uncertainty.

Current replay makes one decision per week and calls `plan_slot`, not `advise_slot`. Static
hold/commit sensitivity and matched snapshots can test mechanics, but cannot measure the dynamic
value of waiting for news. A later multi-event policy replay is required for that claim: process
timestamped observations and hold/commit events in order, preserve each policy's used/locked
state, and never expose later news or outcomes early. Include Thursday-to-Sunday news changes,
no-news controls, elapsed deadlines and irreversible submitted-lock fixtures. A shadow advice
log alone is not a counterfactual achieved season score.

**3C acceptance:** demonstrate live/snapshot parity and faithful reconstruction of captured
events; meet the predeclared coverage and practical-evidence criteria for the intended deployment
population/horizons. If promotion relies on waiting-policy benefit, multi-event replay is an
additional prerequisite. Publish factual shadow comparisons plus a dated authored ship/defer/
no-change decision. Initial snapshots or a non-significant policy difference do not pass this
gate. Production changes require a separate review with appropriate live-transfer and policy
evidence; keeping the model unchanged is a valid completed outcome.

### Non-Goals

Do not simultaneously add usage features, separate TD-type models, alternative distributions,
information-premium tuning or future-discount tuning. Do not infer defense alone from joint
ablations, exhausted input information from this model set, or validated tails/independence
from pooled Poisson fits. Those questions remain open. Leaderboard tails, joint outcomes,
opponent modeling and win-probability optimization belong to later work, not Phase 3.

## Later — leaderboard strategy and optional interface work

Start with official opponent-report ingestion, opponent picks and standings. Simulations must
share one sampled player-week outcome across every entrant selecting that player, with relevant
player correlations, ties, uncertainty about opponent choices and future policy updates.
Validate conditional tails and simulation behavior before treating a simulated win-probability
gain as an established advantage. Repeated decisions within a week are beyond Phase 2 replay.

A local dashboard, deadline reminders and additional input providers remain optional. The
projection-frame interface and assignment solver continue to support expected-TD planning;
fixed-matrix optimality does not establish a rolling-policy advantage over greedy.
