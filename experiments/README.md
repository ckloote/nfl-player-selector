# Experiments

`phase2-validation.toml` is the dated specification. It uses all ten registered models,
shipped production constants, baseline `shipped`, historical inputs and usage roles, shuffled
seeds 0–19, both greedy and optimizer replay, shipped random-top-10 trials, and hindsight.
Seasons 2011–2025 require prior history beginning in 2010. The two eras are retrospective.

Run from the repository root with the pinned environment:

```bash
uv sync --locked
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run pool benchmark \
  --config experiments/phase2-validation.toml \
  --output data/experiments/phase2-validation
# Keep the same environment and add --resume for compatible checkpoints.
```

The runner creates or reuses `research.db` under the ignored output directory, imports missing
research feeds, checks the expected scheduled game counts and complete touchdown coverage,
then freezes `dataset.sqlite`. It does not open the operational pick database. Later refreshes
must use a new output directory/dataset. Missing or unresolved games fail the run. The one
reviewed upstream correction in `scoring-corrections.json` is applied only when its exact source
preconditions and reconciled player totals match; a changed source requires investigation.

Snapshot imports record availability at successful import, preserving source timestamps as
provenance. These observations were acquired during this work, not at historical pick deadlines.
The full historical benchmark therefore uses the documented prior-week-line approximation.
Only prospectively archived observations support genuine timestamp replay.

The output manifest records schema/scoring versions, specification, code revision, dirty state,
source-file hashes (including new untracked code), dependency versions, thread environment,
dataset/table hashes and feed provenance. Checkpoints contain artifact hashes and are accepted
only under the matching manifest identity. Four independent season workers share the same
frozen read-only database; each reuses weekly inputs and shipped components across its models.
The second run with `--resume` verifies every saved checkpoint before regenerating reports.

Artifact schema version 1:

| Artifact | Row definition and units |
|---|---|
| `forecasts.parquet` | Current decision-week player forecast, model, seed, timestamp, hard eligibility, baseline depletion, lambda, final actual TDs, ranks and actual replay-pick indicators |
| `surface-MODEL-SEED.parquet` | Forecast used for every remaining player/week at each decision week, including hard eligibility and recipient availability adjustment |
| `ranking.csv` | Model/seed/season/pool/k; mean of slot-week top-k candidate means, in TDs per ranked candidate, with candidate and empty-cell counts |
| `paired_ranking.csv` | Challenger/seed/season/pool/k minus deterministic baseline in identical cells; TDs per ranked candidate; jointly empty and covered cells |
| `calibration.csv` | Separate model/seed/season Poisson calibration intercept/slope with player-season cluster uncertainty; not pooled repeated seeds |
| `reliability.csv` | Eligible population lambda bins, including eligible zeros; candidate count, forecast/actual mean and played fraction by model/seed/season |
| `deviance.csv` | Model/seed/season Poisson deviance restricted to hard-eligible baseline lambda > 0.30; a forecast diagnostic |
| `spearman.csv` | Model/seed/season mean within-slot-week correlation on hard-eligible candidates |
| `replays.csv` | Model/seed/season/strategy; actual TDs, projected TDs, empty slots, unique players and zero-scoring picks |
| `picks.csv` | Every individual strategy choice and its final actual TDs; empty slots are retained |
| `*_summary.csv` | Seeds/trials averaged within season, then equally weighted seasons; cross-season SE and separate within-season shuffle/trial SD, including retrospective era summaries |
| `input-provenance.json` | Decision timestamps and, for snapshot policy, resolved observation identities, hashes, absence, age and staleness |
| `decision-times.csv` | Normalized copy of the frozen `season,week,decision_at` inputs the run used; the same records are in the resolved configuration and so in its hash |
| `coverage.csv` | Each scheduled game in every required history/evaluation season, with completion and reason |
| `presentation.json` | Publication-only source hashes for the report/figure renderers, plus the original metric source and dataset identities; does not replace the metric manifest |

`phase3c-baseline.toml` is the dated Phase 3C collection protocol. It is not a benchmark
specification: no run reads a frozen research dataset and nothing is fitted. It declares
the collection window, the two weekly decision events, the live input policy, the
populations, the parity criteria and the fidelity and coverage floors, and
`prospective.resolve` refuses it if any declaration is missing — a missing floor blocks
the collection the way a missing margin blocks Phase 3B. The file's text is hashed into
the window identity, so an edit to its reasoning starts a new window.

```bash
uv run pool baseline --config experiments/phase3c-baseline.toml \
  --out experiments/results/phase3c-baseline
```

It reads the operational database's append-only decision log rather than a research
dataset, and writes:

| Artifact | Row definition |
|---|---|
| `decisions.csv` | Each captured decision, its declared event, and whether it reconstructs and matches a snapshot replay of its own instant |
| `events.csv` | Scheduled decision events against captured ones, week by week; a week whose first game is already in the main slate schedules one, not two |
| `fidelity.csv` | Event capture, reconstruction and parity rates against the protocol's floors; both fidelity rates divide by every captured decision, not by the ones that could be checked |
| `outcome-coverage.csv` | Coverage over the declared current and future populations, with future target weeks outside the window reported as unsettled rather than missing |
| `submissions.csv` | Every recorded submission with its player, the decision it was attributed to, whether that link was named or inferred, any reader-side fallback, and its status: still standing, superseded, withdrawn, unattributed, or credited to a decision that never forecast that player |
| `strata.csv`, `fits.csv`, `reliability.csv`, `zero-accounting.csv`, `coverage.csv` | The Phase 3A descriptive tables over the captured surface, under the prospective populations |
| `identities.json` | The protocol hash, the distinct decision identities in the window and any constants drift between them, beside the module hashes that computed the description |
| `BASELINE.md` | The dated note: measurements and identities only, no conclusion |

Eligibility is reconciled with the deadline before any population is described, and parity
compares the model columns rather than the two eligibility columns, which differ by
construction: the live path leaves the deadline out of `hard_eligible` and the replay folds
it in. The captured side of the observation comparison comes from each decision's own
append-only input record rather than from re-resolving the archive at check time, so an
observation written afterwards but stamped before the decision is caught instead of
agreeing with itself. Corrections and removals are folded before `submitted` is a
population, so it holds the pick that still stands rather than every pick ever entered.

Seed −1 identifies a deterministic forecast/strategy. Seeds 0–19 identify shuffled projections
or shipped random-strategy trials; the `model` and `strategy` columns distinguish them.
Shuffled seeds do not multiply the number of independent seasons. An empty common comparison
cell is excluded from the mean and reported, not silently turned into a zero observation.
The `played` field means a player-stat row or TD credit exists. It is not an independently
verified game-day active or snap-participation label.

The candidate pool is the QB/RB/WR/TE players listed active on their own team's most recent
weekly roster snapshot at or before the decision week (latest visible stat teams are the
fallback when no roster feed is available), before assignment pruning. Reading the latest
snapshot rather than the union of every week to date keeps released players out and gives
moved players their current team; it also confines the one cutdown-era snapshot the feed
mislabels as a game week (2016 week 1) to that week. Per team rather than league-wide, because
from 2016 on the feed publishes no roster for a team on its bye, and one league-wide latest
week would drop those teams from every remaining week of the plan. In the saved 2011-2025
study, no team's fallback exceeds one week; the code does not impose that age limit.
Out/Doubtful and expired/unconfirmed current-week cells are hard
exclusions. Questionable scales forecasts by 0.85; zero forecasts can still be eligible.
Player ID breaks ties. Each actual strategy replay depletes its own pool. The hindsight
population is historical eligible scorer identities, so it is a retrospective scoring ceiling
rather than another forecast model.

Mean ranking differences never become purported season gains. Actual-score summary differences
are explicitly against the deterministic baseline's greedy season replay. Raw per-season
artifacts permit other paired strategy comparisons. Calibration and simulation remain separate
validation work; no favorable result or parameter adjustment was required for completion.

After a completed run, independently verify and publish the saved artifacts:

```bash
uv run python experiments/verify.py <experiment> <tests-passed>
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run pool benchmark \
  --config experiments/<experiment>.toml \
  --output data/experiments/<experiment> --resume
uv run python experiments/publish.py <experiment>
```

The verification script reconciles every configured seed, candidate/mask population, pick
history and score total, then records the matching implementation commit. Verification and
model resume still require the recorded metric implementation and inputs. Raw forecasts and
databases remain in the ignored output directory.

## Descriptive diagnostics

`pool diagnose` reads a completed run's saved surface and forecast exports and describes the
model's rate errors by population, position, rate bin, availability and forecast lead horizon.
It fits no correction, selects no model and states no conclusion.

```bash
uv run pool diagnose --run data/experiments/roster-snapshot-repair \
  --out experiments/results/phase3a-readiness
```

Group definitions are frozen before any outcome is read; none selects on future touchdowns or
on participation. Horizons are never pooled: repeated forecasts of one target week share its
single outcome, so horizon 0 clusters on player-season and later horizons cluster on the target
player-week. Raw rates and discounted planning values are exported separately. Eligible zero
rates cannot enter a fit on `log(lambda)` and are excluded from every fit and deviance and
counted instead, separately from hard exclusions, whose zero is a mask rather than a forecast.

| Artifact | Row definition |
|---|---|
| `strata.csv` | Population x axis level x horizon: counts, outcome coverage, forecast and actual levels, forecast/actual, played fraction, positive-rate deviance and the zero-rate counts |
| `fits.csv` | The Poisson calibration fit for each of those strata, with fit status, reason, convergence, iterations, clusters and whether cluster SEs are supported |
| `fits-by-season.csv` | The same fit per season for each population and horizon; season is the unit any later paired comparison would use |
| `reliability.csv` | Lambda-bin table by population, position and horizon |
| `zero-accounting.csv` | Eligible zero rates and hard exclusions, counted separately, with how many scored anyway |
| `coverage.csv` | Rows with and without a resolved outcome, by season and horizon |
| `identities.json`, `READINESS.md` | Source, dataset, configuration and diagnostics-module identities, and a dated note of measurements and limits |

## Phase 3B calibration experiment

`experiments/phase3-calibration.toml` is the dated, frozen specification for the chronological
train/apply experiment. It declares the fold schedule, the candidate families, the population,
the weighting, the fallback, the primary estimand and the decision margins before anything is
fitted; every one of those keys enters the resolved configuration and so the run identity, and
`benchmark.resolve` refuses a run that omits a margin. Amending any of them starts a new
experiment rather than continuing this one.

The project owner approved the unchanged margins on 2026-09-07 (UTC), before the full
benchmark. The [sign-off and rationale](../docs/PHASE3B_MARGIN_SIGNOFF.md) record the
accepted policy downside, the limits of the inference, and the distinction between
Phase 3C advancement and production deployment.

**Full-run outcome, 2026-09-07 (UTC): no promotion.** The
[published report](results/phase3-calibration/CALIBRATION.md) and
[authored outcome note](../docs/PHASE3B_OUTCOME.md) record ten paired evaluation seasons.
No candidate clears every forecast and policy gate; both coverage gates pass. Production
remains unchanged. The next recommended work is identity-only prospective baseline
capture/validation, not a threshold amendment or an automatically promoted candidate.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run pool benchmark \
  --config experiments/phase3-calibration.toml \
  --output data/experiments/phase3-calibration
uv run python experiments/verify.py phase3-calibration <tests-passed>
uv run python experiments/publish.py phase3-calibration
```

The run has three stages with a barrier between each, and each stage checkpoints and resumes
the same way a benchmark season does.

| Stage | Output | What it is |
|---|---|---|
| pairs | `pairs/<season>/` | The identity model over every configured season: the forecast/outcome pairs a fold trains on |
| folds | `folds/<Y>/<candidate>.json` | One fitted artifact per candidate per fold, with its coefficients, fit status, training keys digest, cutoff and content hash |
| apply | `apply/<season>/` | Identity and every candidate over the apply seasons: raw and mapped surfaces, independent replays, metrics |

The map is `exp(a) * lam ** b` on positive rates. `level` fixes the exponent at one and
estimates the scale alone; `log_affine` estimates both and refuses a non-positive exponent,
which would rank a better forecast below a worse one. Identity is mandatory. Each family is
fitted pooled and by position, with WR and TE separate because they share the FLEX slot. The
map acts on the availability-adjusted rate — the quantity the optimizer consumes and the one
Phase 3A described — before pruning and before the future discount. `avail_mult`,
`hard_eligible` and the keys are untouched, and a zero rate maps to zero, so the population the
primary metric scores is identical for every candidate.

Fold Y trains on `train_start..Y-1` only. A training row needs its target week played and
scored, not merely an early forecast timestamp, and the artifact records a digest of the
*values* that entered the fit — keys, rates, positions and outcomes — so that "this artifact was
fitted on these rows" is checkable afterwards rather than asserted; the keys alone would let one
run's fold sit in another's directory unnoticed. The artifact also names the checkpoint digest of
each training season it read, and a resumed fold is checked against both before it is accepted:
a restored directory carries its own checkpoint, so verifying a checkpoint against the files
beside it says nothing about which run produced them. Eligible zero rates cannot enter a fit on
`log(lambda)`: they are excluded and counted, in `training-accounting.csv`, as in the 3A export.
A group below the declared row or cluster minimum, or whose fit is unsupported, takes the
declared fallback and records which one it took.

An outcome is settled from the finalized scoring ledger at export time and travels on the
surface, absence counting as zero only where the week is completely scored. Reading it instead
off the target week's own forecast row would make it conditional on the player still being a
candidate then, which is a fact about roster churn rather than about what he scored: on 2016 that
alone left 9,502 eligible rows unscored, none of which had scored a touchdown, and dropping them
moved fold 2016's fitted scale from 0.9413 to 0.9355. Membership is reported separately as
retention.

| Artifact | Row definition |
|---|---|
| `folds.csv` | Every fitted group in every fold: coefficients, source, fit status and reason, training rows, digest and artifact hash |
| `paired-deviance.csv` | The primary estimand — paired change in season-mean Poisson deviance on the fixed all-eligible current-week population — as a paired season t with its Holm step-down adjustment and both intervals |
| `policy.csv` | Achieved TDs per season and each candidate against identity's replay of the *same* strategy, never against identity greedy, with the paired interval the non-inferiority bound is compared against |
| `decision-changes.csv` | How often a candidate's replay chose a different player than identity's did |
| `advice.csv`, `advice-changes.csv` | Static `advise_slot` sensitivity: replay makes one decision per week through `plan_slot`, so a flipped hold is a mechanism, not a touchdown gained by waiting |
| `coverage-out-of-fold.csv`, `coverage-margins.csv` | Whether the applied surface has an outcome to score against, by season and horizon and then collapsed to the two populations the declared floors are stated over, with retention — how much of the surface belonged to a player still in the pool at the week he was forecast for — reported beside it. Different from the schedule audit in `coverage.csv` and from each other |
| `zero-accounting-out-of-fold.csv` | Eligible zero rates and the two kinds of exclusion over the applied seasons, in the classes that mean different things |
| `training-accounting.csv` | The same for the training population, per fold: hard exclusions, eligible zero rates and unresolved outcomes, reconciling to the rows each fold actually fitted |
| `strata-out-of-fold.csv` | The proper score by forecast horizon and availability, per candidate — the diagnostics the pooled fit declares consequences for and cannot itself show |
| `surface-<candidate>-<seed>.parquet` | The out-of-fold surface, carrying the mapped rate as `lam`, the rate it was mapped from as `original_lam`, the decision-time `position` the map was selected by, and the target week's settled outcome (`actual_tds`, `outcome_complete`, `played`) beside `in_target_pool` |

`experiments/phase3-calibration-smoke.toml` is a two-season mechanics check, not evidence. It
exists so the stages, the barrier, the artifact hashes and resume can be exercised end to end.
Point a fresh output directory at a copy of an existing `dataset.sqlite` to skip re-ingestion.

The generated `CALIBRATION.md` states facts, methods and provenance. It selects no candidate:
the promotion rule is in the frozen configuration as `[calibration_experiment.promotion]`, and
applying it is a separate, dated authoring step, exactly as the bake-off keeps its report
separate from [docs/ANALYSIS.md](../docs/ANALYSIS.md). Keeping the model unchanged is a valid
outcome.

The rule is data rather than prose so that it cannot be relaxed once the estimates exist, and
`resolve` additionally hashes the specification *text* into the resolved configuration, so an
edit to the reasoning beside those keys starts a new experiment too. `verify.py` reads the
saved resolved configuration rather than the current TOML, checks it against the identity the
run recorded, and records that hash in `verification.json`; publication refuses a calibration
record that does not carry it.

## Measurements And Interpretation

Each run produces one `EVALUATION.md`, published beside its metrics and as
[`docs/EVALUATION.md`](../docs/EVALUATION.md). It presents season replay results, ranking
diagnostics and calibration diagnostics as distinct sections, with shared methods,
reproducibility and verification recorded once. All four figures remain in their matching
results sections. The three metric tables retain their separate units and populations.

The generated report contains measured tables, plots, counts, uncertainty estimates, metric
definitions, input assumptions and recorded provenance. It does not select a winning model,
attribute effects to player identity or timing, turn a standard error into a detection floor,
or recommend a research/deployment decision. Undefined statistics are explicit. Plot colors
and reference lines do not classify statistical significance. Calibration reports show both
intercept and slope; a unit slope alone does not establish calibration.

[docs/ANALYSIS.md](../docs/ANALYSIS.md) is separately authored human/AI interpretation, dated
and tied to a specific study and its source/dataset identities. A new result requires an
explicit review of that analysis, not automatic prose selected by thresholds in a renderer.
Publication never overwrites the analysis or the implementation plan. Superseded experiments
and archived reports retain their original historical record, including the older split
report filenames.

To update presentation for an already verified run, use only:

```bash
uv run python experiments/publish.py roster-snapshot-repair
```

Publication validates the saved completion, verification, manifest, configuration and frozen
dataset identities. It copies the compact metrics, compresses the exact pick CSV with a
deterministic gzip header, and renders the report text anew from the saved metric tables,
not from previously generated Markdown. Compact Markdown is not copied into the publication;
only the current rendered report is written. After publishing successfully, the two retired
`BACKTEST.md` and `PROJECTION_BENCHMARK.md` files are removed from the current experiment's
published directory and `docs/`, not from saved compact inputs, archives or other experiments.
It does not run forecasts, calibration fits or
season replays, and does not rewrite the original verification record or compact inputs.
The original metric source need not match the current renderer: `presentation.json` records
the current rendering-source hashes separately. This permits presentation changes without
claiming a new model run or weakening the benchmark's resume identity checks.

Changes anywhere under `src`, including report code or docstrings, still change the benchmark
source fingerprint. Do not rerun verification or rewrite a manifest to make old checkpoints
appear compatible with a newer source tree. Recompute in a new experiment when model/metric
changes require it; use the recorded implementation to resume the original computation.
