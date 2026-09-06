# Phase 3A Readiness Note

Generated 2026-09-06 from `data/experiments/roster-snapshot-repair`. Measurements and identities only; interpretation belongs in a separately authored, dated analysis.

## Identities

**Forecasts described**

- experiment: `roster-snapshot-repair`
- model: `shipped seed -1`
- source_fingerprint: `08886f8851956059ccb50dace43e2b8af4f0607f724ff8a5d14c04b6e82d5686`
- code_revision: `9b0525c86b59e15f9238a5fd01abe9f6ff824064`
- frozen_dataset: `405df67ac6a2d73b1036ffdd483b56f1e9745934af110a1730c84d6e7e93fdf2`
- configuration: `beaf203930155ab8be928b2b09597f9c7db6fe479bf86aa6fbb054578730bfbb`
- future_discount: `0.985`
- future_discount_source: `study`

**Implementation describing them**

- diagnostics_module: `75e2f9ff6e21ff839c8a9615aca2822a2edbfdc78e32550fc02b467a6465cb1c`
- evaluate_module: `2e5019c8c707aa5148816084db9c719809c9fa725c8462091ac5dc93512b72c7`
- min_inference_clusters: `30`
- lambda_bins: `['0.0', '0.05', '0.1', '0.15', '0.2', '0.3', '0.4', '0.6', '0.8', '1.0', 'inf']`
- horizon_buckets: `['0', '1', '2-3', '4-6', '7+']`
- populations: `['all_eligible', 'available', 'depleted', 'common_top1', 'common_top3', 'common_top5', 'common_top10', 'selected_greedy', 'selected_optimizer']`
- positions: `['QB', 'RB', 'WR', 'TE']`
- artifact_schema: `1`


## Population

- Surface rows: 964,429 over 15 seasons, 2,552 players, lead horizons 0-17.
- Rows with a resolved outcome: 856,673 (0.8883). A missing outcome is never read as a zero.
- Eligible zero rates (forecast zero but selectable): 0. These are the rows a fit on `log(lambda)` cannot take; they are excluded from every fit and from the deviance, and counted in `zero-accounting.csv`.
- Excluded as unavailable (ruled out, so masked to zero rather than forecast at zero): 2,097, of which 3 scored anyway. A player who was ruled out and played is a report error, not a calibration error.
- Excluded as undecidable (deadline elapsed or kickoff unconfirmed): 0. These keep a positive rate: the forecast was usable, the cell was not, and their touchdowns are evidence about neither the model nor the injury report.
- Neither kind of exclusion enters an eligible population or any fit.

## Group definitions

- Populations: all_eligible, available, depleted, common_top1, common_top3, common_top5, common_top10, selected_greedy, selected_optimizer. All of them except `all_eligible` rest on a decision-week fact -- whether the baseline had already spent the player, his rank in the common pool, whether a policy picked him -- so they exist only at horizon 0. `available` and `depleted` partition the eligible current-week rows between them.
- Positions: QB, RB, WR, TE, WR and TE separately throughout.
- Rate bins: [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, inf].
- Availability: questionable or full, from the availability multiplier. A ruled-out player is not eligible, so no eligible population contains one.
- Lead horizons: 0, 1, 2-3, 4-6, 7+, never pooled.
- Every definition is fixed before any outcome is read; none selects on future touchdowns or on participation.

## Fits

- Strata fitted: 188; unsupported: 2; supported but without cluster standard errors: 6 (fewer than 30 clusters).
  - n <= 2 for a two-parameter fit: 2
- Horizon 0 clusters on player-season; later horizons cluster on the shared target player-week, because repeated forecasts of one target share its single outcome.
- Raw rates are diagnosed separately from the discounted planning values; both are exported.

## Limits

- Descriptive and retrospective. These are in-sample diagnostic coefficients on seasons that have already been explored, not fitted correction artifacts and not an out-of-sample evaluation.
- A slope is not a correction. Neither a stratum's coefficient nor the set of them identifies a mapping; Phase 3B specifies and freezes that separately, before fitting.
- Nothing here authorizes a production change. Production constants are unchanged.
