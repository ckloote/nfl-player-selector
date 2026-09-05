# Corrected season replays

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

| model | strategy | tds_per_season | se | shuffle_sd | delta_vs_baseline_greedy | paired_se |
| --- | --- | --- | --- | --- | --- | --- |
| base-rate-only | greedy | 46.8667 | 2.1221 | 0.0000 | -6.6667 | 2.2010 |
| base-rate-only | optimizer | 46.7333 | 2.0574 | 0.0000 | -6.8000 | 2.2172 |
| current-season-rate | greedy | 45.9333 | 1.7388 | 0.0000 | -7.6000 | 2.2271 |
| current-season-rate | optimizer | 46.6000 | 1.7776 | 0.0000 | -6.9333 | 1.9060 |
| hindsight | hindsight | 157.8000 | 1.6071 | 0.0000 | 104.2667 | 2.0644 |
| historical-rate | greedy | 40.6667 | 1.5574 | 0.0000 | -12.8667 | 2.6581 |
| historical-rate | optimizer | 40.6000 | 1.3898 | 0.0000 | -12.9333 | 2.3247 |
| no-vegas | greedy | 50.9333 | 2.1480 | 0.0000 | -2.6000 | 1.3266 |
| no-vegas | optimizer | 54.2667 | 2.2813 | 0.0000 | 0.7333 | 1.9940 |
| player-vegas | greedy | 47.7333 | 1.5259 | 0.0000 | -5.8000 | 1.8132 |
| player-vegas | optimizer | 47.6667 | 1.6894 | 0.0000 | -5.8667 | 1.8844 |
| random | greedy | 19.3000 | 0.6790 | 8.2640 | -34.2333 | 1.9722 |
| random | optimizer | 19.2300 | 0.6666 | 8.1709 | -34.3033 | 1.9859 |
| regressed-rate | greedy | 42.2667 | 2.0964 | 0.0000 | -11.2667 | 2.1788 |
| regressed-rate | optimizer | 42.0000 | 1.9615 | 0.0000 | -11.5333 | 1.9417 |
| shipped | greedy | 53.5333 | 2.3009 | 0.0000 | 0.0000 | 0.0000 |
| shipped | optimizer | 56.6000 | 2.4178 | 0.0000 | 3.0667 | 1.8860 |
| shipped | random | 46.6667 | 1.0826 | 6.9672 | -6.8667 | 2.0999 |
| vegas-environment | greedy | 46.6000 | 2.1730 | 0.0000 | -6.9333 | 2.4465 |
| vegas-environment | optimizer | 45.8667 | 2.3009 | 0.0000 | -7.6667 | 2.5027 |
| within-player | greedy | 50.4700 | 1.4879 | 5.5767 | -3.0633 | 1.4259 |
| within-player | optimizer | 50.4767 | 1.1526 | 6.1439 | -3.0567 | 1.9511 |

`replays.csv` records actual TDs, empty slots and unique players for every seed/trial and season; `picks.csv` records every choice. Era summaries remain retrospective. Hindsight uses actual scorer identities and the full feasible scoring history; it is a reference ceiling with a different candidate population. The solver is exact for the pruned fixed matrix, which does not establish an advantage for its rolling policy. No production parameter was changed.


## Run verification

Implementation commit matching every recorded source-file hash: `79c79f2166cb3b5ac073a8f00670b00562b8fda4`. The run began from that source tree before its implementation commit; the manifest preserves the original parent revision and dirty fingerprint.

All 15 seasons completed: 720 model/seed/season runs, 5,262,384 forecast rows, 1,755 achieved season scores, and 91,260 individual replay picks. All 4,175 required games have complete scoring and player-stat coverage for both teams. No season was omitted.

2022 includes 271 completed games; the [Bills–Bengals game was canceled](https://www.buffalobills.com/news/nfl-says-neutral-site-afc-championship-game-is-possible-bills-bengals-week-17-ga). Its absence from the final schedule is a historical-replay approximation.

Verification: 214 pytest tests passed; Ruff lint and formatting passed. Saved forecasts were checked for identical candidate/mask populations, hard exclusions, timestamps and outcomes. Pick histories contain no player reuse; pick sums equal reported scores; selected-player flags and unique-player counts reconcile. A full `--resume` verified all checkpoint hashes. Synthetic interrupted/resumed and uninterrupted runs produced identical artifacts.

This run used `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`; keep these environment variables when resuming. The pinned dependencies and exact environment are recorded in the manifest. See [experiment instructions](../experiments/README.md) for the full command and schema. Reports are generated from saved metrics and this verification record.
