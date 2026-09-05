# Corrected projection benchmark

Specification: 2026-09-04. Artifact schema 1; scoring version 2; database schema 2.

Code revision `fdf2638d943d423d9b7c2a28dc5dff2f19ddb5f3`; source fingerprint `2cff1d7117b804c6ba33bc162d61283c57ab05cd5f19a84d901dc3c4efeb08b8`; dirty tree: True. Frozen dataset SHA-256 `405df67ac6a2d73b1036ffdd483b56f1e9745934af110a1730c84d6e7e93fdf2`.

Models: random, within-player, historical-rate, regressed-rate, current-season-rate, vegas-environment, player-vegas, shipped, no-vegas, base-rate-only. Baseline: `shipped`; shuffled seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]; deterministic seed sentinel: -1. Policy: `historical`; roles: `usage`.

Reproduce: `pool benchmark --config experiments/phase2-validation.toml --output data/experiments/phase2-validation --resume`. Checkpoints require matching code, configuration, dependencies and frozen dataset. Saved compact artifacts are in `experiments/results/phase2-validation`; detailed forecasts and future surfaces remain in the ignored output directory.

- Scoring: one credit for each touchdown scored and each credited passing touchdown thrown; negated plays and conversions excluded; complete game coverage required.
- Historical: current-season closing lines for weeks >= decision week are masked before all team/league averages. Week one uses the shipped league fallback.
- Stats through W-1; weekly roster and injury rows through W; usage roles. Final schedule revisions, report timing within a week and later stat corrections remain approximations.
- One decision per week immediately before the first confirmed pick deadline. Unknown current-week kickoffs are hard exclusions; future planning estimates remain usable.
- Snapshot observations represent import availability, not backdated source publication. Finalized outcome scoring is separate from archived projection inputs.
- Ranking population: all active-roster pool-position candidate cells, before assignment pruning; hard exclusions unranked. Zero estimates eligible. Ties use player ID.
- Common-pool depletion uses the selected deterministic baseline greedy history. Top-k diagnostics are TDs per ranked candidate, never achieved season scores.
- Slot-week means are averaged within season. Seeds are averaged within season before uncertainty across seasons; seed SD is reported separately. Empty cells are reported.
- Each greedy/optimizer replay has its own no-reuse history. Random-top-10 strategy trials use shipped forecasts and differ from shuffled projection models. Hindsight is retrospective.
- All seasons and era summaries are retrospective; neither era is an untouched holdout. Production constants are fixed; no calibration correction or tuning is applied.
- Research scoring includes the checked correction in experiments/scoring-corrections.json: duplicate rushing touchdowns in 2011_13_DET_NO reconciled to the official Saints game report.

## Common-pool paired ranking diagnostics

Challenger minus baseline, TDs per ranked candidate; SE across seasons.

| model | k | mean | se | shuffle_sd | seasons |
| --- | --- | --- | --- | --- | --- |
| base-rate-only | 1 | -0.0590 | 0.0436 | 0.0000 | 15 |
| base-rate-only | 3 | -0.0657 | 0.0208 | 0.0000 | 15 |
| base-rate-only | 5 | -0.0432 | 0.0179 | 0.0000 | 15 |
| base-rate-only | 10 | -0.0365 | 0.0091 | 0.0000 | 15 |
| current-season-rate | 1 | -0.0696 | 0.0420 | 0.0000 | 15 |
| current-season-rate | 3 | -0.1024 | 0.0216 | 0.0000 | 15 |
| current-season-rate | 5 | -0.1071 | 0.0193 | 0.0000 | 15 |
| current-season-rate | 10 | -0.1575 | 0.0149 | 0.0000 | 15 |
| historical-rate | 1 | -0.3060 | 0.0530 | 0.0000 | 15 |
| historical-rate | 3 | -0.2613 | 0.0284 | 0.0000 | 15 |
| historical-rate | 5 | -0.2101 | 0.0243 | 0.0000 | 15 |
| historical-rate | 10 | -0.1406 | 0.0147 | 0.0000 | 15 |
| no-vegas | 1 | -0.0316 | 0.0214 | 0.0000 | 15 |
| no-vegas | 3 | -0.0139 | 0.0144 | 0.0000 | 15 |
| no-vegas | 5 | -0.0145 | 0.0102 | 0.0000 | 15 |
| no-vegas | 10 | -0.0099 | 0.0040 | 0.0000 | 15 |
| player-vegas | 1 | -0.0254 | 0.0477 | 0.0000 | 15 |
| player-vegas | 3 | -0.0747 | 0.0274 | 0.0000 | 15 |
| player-vegas | 5 | -0.0599 | 0.0179 | 0.0000 | 15 |
| player-vegas | 10 | -0.0642 | 0.0111 | 0.0000 | 15 |
| random | 1 | -0.7069 | 0.0343 | 0.1404 | 15 |
| random | 3 | -0.6628 | 0.0213 | 0.0774 | 15 |
| random | 5 | -0.6183 | 0.0165 | 0.0546 | 15 |
| random | 10 | -0.5157 | 0.0139 | 0.0372 | 15 |
| regressed-rate | 1 | -0.1132 | 0.0523 | 0.0000 | 15 |
| regressed-rate | 3 | -0.1446 | 0.0171 | 0.0000 | 15 |
| regressed-rate | 5 | -0.1300 | 0.0135 | 0.0000 | 15 |
| regressed-rate | 10 | -0.1304 | 0.0120 | 0.0000 | 15 |
| vegas-environment | 1 | -0.2381 | 0.0471 | 0.0000 | 15 |
| vegas-environment | 3 | -0.1673 | 0.0229 | 0.0000 | 15 |
| vegas-environment | 5 | -0.1403 | 0.0138 | 0.0000 | 15 |
| vegas-environment | 10 | -0.0674 | 0.0079 | 0.0000 | 15 |
| within-player | 1 | -0.0256 | 0.0223 | 0.0929 | 15 |
| within-player | 3 | -0.0304 | 0.0114 | 0.0416 | 15 |
| within-player | 5 | -0.0252 | 0.0074 | 0.0288 | 15 |
| within-player | 10 | -0.0099 | 0.0052 | 0.0147 | 15 |

Per-seed coverage, jointly empty cells and per-season diagnostics are saved in `ranking.csv` and `paired_ranking.csv`. Across exported ranking metric rows: 0 empty cells (repeated across k, pools and seeds; not independent observations). Calibration slopes/intercepts, reliability bins, tail deviance and within-slot Spearman results are saved separately by seed and season. They are diagnostics, not evidence of an achieved season gain or simulation readiness.


## Run verification

Implementation commit matching every recorded source-file hash: `79c79f2166cb3b5ac073a8f00670b00562b8fda4`. The run began from that source tree before its implementation commit; the manifest preserves the original parent revision and dirty fingerprint.

All 15 seasons completed: 720 model/seed/season runs, 5,262,384 forecast rows, 1,755 achieved season scores, and 91,260 individual replay picks. All 4,175 required games have complete scoring and player-stat coverage for both teams. No season was omitted.

2022 includes 271 completed games; the [Bills–Bengals game was canceled](https://www.buffalobills.com/news/nfl-says-neutral-site-afc-championship-game-is-possible-bills-bengals-week-17-ga). Its absence from the final schedule is a historical-replay approximation.

Verification: 214 pytest tests passed; Ruff lint and formatting passed. Saved forecasts were checked for identical candidate/mask populations, hard exclusions, timestamps and outcomes. Pick histories contain no player reuse; pick sums equal reported scores; selected-player flags and unique-player counts reconcile. A full `--resume` verified all checkpoint hashes. Synthetic interrupted/resumed and uninterrupted runs produced identical artifacts.

This run used `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`; keep these environment variables when resuming. The pinned dependencies and exact environment are recorded in the manifest. See [experiment instructions](../../README.md) for the full command and schema. Reports are generated from saved metrics and this verification record.
