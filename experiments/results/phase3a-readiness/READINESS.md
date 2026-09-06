# Phase 3A Readiness Note

Generated 2026-09-06 from `data/experiments/roster-snapshot-repair`. Measurements and identities only; interpretation belongs in a separately authored, dated analysis.

## Identities

- experiment: `roster-snapshot-repair`
- model: `shipped seed -1`
- source fingerprint: `08886f8851956059ccb50dace43e2b8af4f0607f724ff8a5d14c04b6e82d5686`
- code revision: `9b0525c86b59e15f9238a5fd01abe9f6ff824064`
- frozen dataset: `405df67ac6a2d73b1036ffdd483b56f1e9745934af110a1730c84d6e7e93fdf2`
- configuration: `beaf203930155ab8be928b2b09597f9c7db6fe479bf86aa6fbb054578730bfbb`
- diagnostics module: `04d543a387e4a6fdd47a1685460380bb789ab5dab1469688ed10f0f16760459a`
- artifact schema: `1`

## Population

- Surface rows: 964,429 over 15 seasons, 2,552 players, lead horizons 0-17.
- Rows with a resolved outcome: 856,673 (0.8883). A missing outcome is never read as a zero.
- Eligible zero rates (forecast zero but selectable): 0. These are the rows a fit on `log(lambda)` cannot take; they are excluded from every fit and from the deviance, and counted in `zero-accounting.csv`.
- Hard exclusions (ruled out, so masked to zero rather than forecast at zero): 2,097 with a known outcome, of which 3 scored anyway. These never enter an eligible population or a fit; a player who was ruled out and played is a report error, not a calibration error.

## Group definitions

- Populations: all_eligible, available, depleted, common_top1, common_top3, common_top5, common_top10, selected_greedy, selected_optimizer. Depletion, common-pool rank and the two selected populations are decision-week properties and exist only at horizon 0.
- Positions: QB, RB, WR, TE, WR and TE separately throughout.
- Rate bins: [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, inf].
- Availability: excluded, questionable, full, from the availability multiplier.
- Lead horizons: 0, 1, 2-3, 4-6, 7+, never pooled.
- Every definition is fixed before any outcome is read; none selects on future touchdowns or on participation.

## Fits

- Strata fitted: 248; unsupported: 2; supported but without cluster standard errors: 6 (fewer than 30 clusters).
  - n <= 2 for a two-parameter fit: 2
- Horizon 0 clusters on player-season; later horizons cluster on the shared target player-week, because repeated forecasts of one target share its single outcome.
- Raw rates are diagnosed separately from the discounted planning values; both are exported.

## Limits

- Descriptive and retrospective. These are in-sample diagnostic coefficients on seasons that have already been explored, not fitted correction artifacts and not an out-of-sample evaluation.
- A slope is not a correction. Neither a stratum's coefficient nor the set of them identifies a mapping; Phase 3B specifies and freezes that separately, before fitting.
- Nothing here authorizes a production change. Production constants are unchanged.
