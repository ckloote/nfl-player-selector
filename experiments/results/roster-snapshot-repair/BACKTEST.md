# Corrected season replays

Specification: 2026-09-04. Artifact schema 1; scoring version 2; database schema 2.

Code revision `2546c23abb58eb8a3367dc07f020cb49813d3ba7`; source fingerprint `a90e7cda4582e9320bca5283a7e45b472f5fc5247d65ad9bb2e8f6bdee7418f2`; dirty tree: True. Frozen dataset SHA-256 `405df67ac6a2d73b1036ffdd483b56f1e9745934af110a1730c84d6e7e93fdf2`.

Models: random, within-player, historical-rate, regressed-rate, current-season-rate, vegas-environment, player-vegas, shipped, no-vegas, base-rate-only. Baseline: `shipped`; shuffled seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]; deterministic seed sentinel: -1. Policy: `historical`; roles: `usage`.

Reproduce: `pool benchmark --config experiments/roster-snapshot-repair.toml --output data/experiments/roster-snapshot-repair --resume`. Checkpoints require matching code, configuration, dependencies and frozen dataset. Saved compact artifacts are in `experiments/results/roster-snapshot-repair`; detailed forecasts and future surfaces remain in the ignored output directory.

- Scoring: one credit for each touchdown scored and each credited passing touchdown thrown; negated plays and conversions excluded; complete game coverage required.
- Historical: current-season closing lines for weeks >= decision week are masked before all team/league averages. Week one uses the shipped league fallback.
- Stats through W-1; weekly roster and injury rows through W; usage roles. Final schedule revisions, report timing within a week and later stat corrections remain approximations.
- One decision per week immediately before the first confirmed pick deadline. Unknown current-week kickoffs are hard exclusions; future planning estimates remain usable.
- Snapshot observations represent import availability, not backdated source publication. Finalized outcome scoring is separate from archived projection inputs.
- Ranking population: pool-position players listed active on the latest weekly roster snapshot at or before the decision week, before assignment pruning; hard exclusions unranked. Zero estimates eligible. Ties use player ID.
- Common-pool depletion uses the selected deterministic baseline greedy history. Top-k diagnostics are TDs per ranked candidate, never achieved season scores.
- Slot-week means are averaged within season. Seeds are averaged within season before uncertainty across seasons; seed SD is reported separately. Empty cells are reported.
- Each greedy/optimizer replay has its own no-reuse history. Random-top-10 strategy trials use shipped forecasts and differ from shuffled projection models. Hindsight is retrospective.
- All seasons and era summaries are retrospective; neither era is an untouched holdout. Production constants are fixed; no calibration correction or tuning is applied.
- Research scoring includes the checked correction in experiments/scoring-corrections.json: duplicate rushing touchdowns in 2011_13_DET_NO reconciled to the official Saints game report.

## What the replays show

`shipped` scores 53.7 TDs per season under greedy, ahead of every alternative forecast. The nearest is `no-vegas` at -2.73 (1.31 SE), better in 4 of 15 seasons.

Replaying the rolling assignment against its own no-reuse history scores +2.87 TDs per season (1.68 SE) versus greedy, better in 10 of 15 seasons. That does not clear two standard errors, so these seasons cannot separate the two policies. Exact optimality on a fixed pruned matrix is a property of the solver, not evidence about the rolling policy, and this comparison is sensitive to the input policy the replay is run under.

`within-player` keeps each player's own forecasts and destroys only their order across weeks. It costs -3.14 TDs per season (1.46 SE), better in 5 of 15, against -34.42 for the fully shuffled null. So about 91% of the measured advantage over random is in telling players apart, and the remainder in timing them. That timing component sits 2.2 standard errors from zero.

Paired standard errors on these 15-season comparisons run 1.31-2.67 TDs per season, median 2.18. A typical comparison therefore needs roughly 4 TDs per season before these replays can tell it from zero. That floor, not the length of the model list, is what limits every season-level claim here.

![Season score against the baseline, with standard errors](figures/bakeoff.svg)

![Rolling assignment minus greedy, by season](figures/optimizer-by-season.svg)

## Achieved season scores

| model | strategy | tds_per_season | se | shuffle_sd | delta_vs_baseline_greedy | paired_se |
| --- | --- | --- | --- | --- | --- | --- |
| base-rate-only | greedy | 47.3333 | 1.9314 | 0.0000 | -6.4000 | 2.1883 |
| base-rate-only | optimizer | 47.0667 | 1.9309 | 0.0000 | -6.6667 | 2.1771 |
| current-season-rate | greedy | 45.9333 | 1.7388 | 0.0000 | -7.8000 | 2.2995 |
| current-season-rate | optimizer | 46.6667 | 1.7719 | 0.0000 | -7.0667 | 1.9135 |
| hindsight | hindsight | 157.8000 | 1.6071 | 0.0000 | 104.0667 | 2.0412 |
| historical-rate | greedy | 41.0667 | 1.5871 | 0.0000 | -12.6667 | 2.6702 |
| historical-rate | optimizer | 41.0000 | 1.3904 | 0.0000 | -12.7333 | 2.3329 |
| no-vegas | greedy | 51.0000 | 2.1224 | 0.0000 | -2.7333 | 1.3146 |
| no-vegas | optimizer | 54.1333 | 2.1753 | 0.0000 | 0.4000 | 2.1861 |
| player-vegas | greedy | 47.8667 | 1.5083 | 0.0000 | -5.8667 | 1.8410 |
| player-vegas | optimizer | 48.0000 | 1.6733 | 0.0000 | -5.7333 | 1.9383 |
| random | greedy | 19.3133 | 0.5891 | 7.9100 | -34.4200 | 1.9876 |
| random | optimizer | 19.1900 | 0.6111 | 7.5441 | -34.5433 | 2.0339 |
| regressed-rate | greedy | 43.0667 | 1.9868 | 0.0000 | -10.6667 | 2.2922 |
| regressed-rate | optimizer | 42.6667 | 1.9777 | 0.0000 | -11.0667 | 2.2897 |
| shipped | greedy | 53.7333 | 2.3227 | 0.0000 | 0.0000 | 0.0000 |
| shipped | optimizer | 56.6000 | 1.9756 | 0.0000 | 2.8667 | 1.6843 |
| shipped | random | 46.7433 | 1.0796 | 6.9501 | -6.9900 | 2.1784 |
| vegas-environment | greedy | 46.6000 | 2.1620 | 0.0000 | -7.1333 | 2.3962 |
| vegas-environment | optimizer | 46.0667 | 2.2178 | 0.0000 | -7.6667 | 2.4585 |
| within-player | greedy | 50.5900 | 1.5691 | 5.1875 | -3.1433 | 1.4567 |
| within-player | optimizer | 50.6733 | 1.0732 | 5.9595 | -3.0600 | 1.7523 |

`replays.csv` records actual TDs, empty slots and unique players for every seed/trial and season; `picks.csv` records every choice. Era summaries remain retrospective. Hindsight uses actual scorer identities and the full feasible scoring history; it is a reference ceiling with a different candidate population. The solver is exact for the pruned fixed matrix, which does not establish an advantage for its rolling policy. No production parameter was changed.


## Run verification

Implementation commit matching every recorded source-file hash: `61f204903757b8655c7ba1818a2cd6be82c614d9`. The run began from that source tree before its implementation commit; the manifest preserves the original parent revision and dirty fingerprint.

All 15 seasons completed: 720 model/seed/season runs, 5,045,952 forecast rows, 1,755 achieved season scores, and 91,260 individual replay picks. All 4,175 required games have complete scoring and player-stat coverage for both teams. No season was omitted.

2022 includes 271 completed games; the [Bills–Bengals game was canceled](https://www.buffalobills.com/news/nfl-says-neutral-site-afc-championship-game-is-possible-bills-bengals-week-17-ga). Its absence from the final schedule is a historical-replay approximation.

Verification: 218 pytest tests passed; Ruff lint and formatting passed. Saved forecasts were checked for identical candidate/mask populations, hard exclusions, timestamps and outcomes. Pick histories contain no player reuse; pick sums equal reported scores; selected-player flags and unique-player counts reconcile. A full `--resume` verified all checkpoint hashes. Synthetic interrupted/resumed and uninterrupted runs produced identical artifacts.

This run used `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`; keep these environment variables when resuming. The pinned dependencies and exact environment are recorded in the manifest. See [experiment instructions](../../README.md) for the full command and schema. Reports are generated from saved metrics and this verification record.
