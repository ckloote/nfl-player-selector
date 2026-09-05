# Phase 2 experiments

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
| `coverage.csv` | Each scheduled game in every required history/evaluation season, with completion and reason |

Seed −1 identifies a deterministic forecast/strategy. Seeds 0–19 identify shuffled projections
or shipped random-strategy trials; the `model` and `strategy` columns distinguish them.
Shuffled seeds do not multiply the number of independent seasons. An empty common comparison
cell is excluded from the mean and reported, not silently turned into a zero observation.

The candidate pool is the QB/RB/WR/TE players listed active on their own team's most recent
weekly roster snapshot at or before the decision week (latest visible stat teams are the
fallback when no roster feed is available), before assignment pruning. Reading the latest
snapshot rather than the union of every week to date keeps released players out and gives
moved players their current team; it also confines the one cutdown-era snapshot the feed
mislabels as a game week (2016 week 1) to that week. Per team rather than league-wide, because
from 2016 on the feed publishes no roster for a team on its bye, and one league-wide latest
week would drop those teams from every remaining week of the plan. A team's fallback is never
more than one week old. Out/Doubtful and expired/unconfirmed current-week cells are hard
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
history and score total, then records the matching implementation commit. Publication copies
compact metrics, compresses the exact individual-pick CSV with a deterministic gzip header,
and generates report appendices from that verification record. Raw forecasts and databases
remain in the ignored output directory.
