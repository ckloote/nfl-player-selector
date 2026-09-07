# Authored Study Analysis

**Author:** OpenCode (AI-authored interpretation), 2026-09-05.
**Basis:** main at `abc8523` and the saved `roster-snapshot-repair` study below.
This is not a generated report and is not automatically refreshed by benchmark reruns or
publication. Later human/AI interpretation must be explicitly reviewed and dated. The generated
[evaluation report](EVALUATION.md) contains season scores, forecast diagnostics,
methods and provenance, not automated recommendations about research or production.

**Follow-up, 2026-09-07 (UTC):** the completed chronological calibration experiment and
its no-promotion disposition are recorded separately in [PHASE3B_OUTCOME.md](PHASE3B_OUTCOME.md).
Position-specific log-affine maps improve forecasts but do not establish the required policy
non-inferiority. Production remains unchanged. The September 5 study analysis below retains
its original scope, provenance and conclusions; it is not the new experiment's outcome note.

## Provenance

The exact study is [experiments/results/roster-snapshot-repair](../experiments/results/roster-snapshot-repair),
specified by [roster-snapshot-repair.toml](../experiments/roster-snapshot-repair.toml) and its
saved [resolved configuration](../experiments/results/roster-snapshot-repair/resolved-config.json).
It replays 2011-2025 with 2010 prior history, historical input approximations, usage roles,
fixed shipped constants, and shuffled seeds 0-19. No calibration mapping was applied.

- Metric implementation: `7a98f03150193572515617f29f60e635f21aedb9`, recorded in [verification.json](../experiments/results/roster-snapshot-repair/verification.json).
- Source hash: `08886f8851956059ccb50dace43e2b8af4f0607f724ff8a5d14c04b6e82d5686`.
- Dataset hash: `405df67ac6a2d73b1036ffdd483b56f1e9745934af110a1730c84d6e7e93fdf2`.

The [manifest](../experiments/results/roster-snapshot-repair/manifest.json) records a dirty run
based on `9b0525c`; its source hashes and the matching implementation above identify the metric
code, not the later main merge or report-rendering revision. The repaired study supersedes
[Phase 2 results](../experiments/results/phase2-validation); those and [archived reports](archive)
retain their original definitions and numbers.

## Saved Facts

From [replay_summary.csv](../experiments/results/roster-snapshot-repair/replay_summary.csv),
`all retrospective`: seeds are averaged within season before equal-season summaries; paired
SEs use the fifteen season differences. The `random` projection null is not the shipped
random-top-10 selection strategy.

| Model / Policy | TDs Per Season | Difference From Shipped Greedy | Paired SE |
|---|---:|---:|---:|
| Shipped / greedy | 53.7333 | 0 | - |
| Random projection / greedy | 19.3133 | -34.4200 | 1.9876 |
| Within-player / greedy | 50.3967 | -3.3367 | 1.4895 |
| Shipped / optimizer | 56.6000 | +2.8667 | 1.8044 |

In [calibration.csv](../experiments/results/roster-snapshot-repair/calibration.csv), the shipped
per-season slopes, pooling eligible current-week forecasts across positions within each
season, range from 0.816872 to 0.891524; their mean is 0.866961 and sample SD is 0.021879.
These are fifteen separate fits, not one pooled correction coefficient.

## Interpretation And Limits

**Policy comparisons.** The optimizer's nominal paired t(14) 95% interval is approximately
`[-1.00, 6.74]` TDs per season, computed as `2.8667 +/- t(14, .975) * 1.8044`.
This exploratory interval treats season differences as independent, representative draws with
a suitably behaved sampling distribution. It has no multiple-comparison adjustment and no
independent holdout confirmation. It allows both a small loss and a useful gain; failure to
exclude zero is not equivalence. A median SE across other comparisons is not a universal
4-TD detection floor: intervals and prospective power calculations are comparison-specific.

**What the null retains.** Under greedy, `(50.3967 - 19.3133) / (53.7333 - 19.3133)` is
90.3% retained advantage over the random projection null. This is descriptive, not a causal
identity/timing attribution. At every decision the within-player null permutes availability-free
rates across each player's eligible remaining-week surface, then reapplies recipient
availability. It retains updated player rates, eligibility and availability information;
it is not a fixed player-quality estimate with all temporal information removed.
Using matched optimizer histories gives 82.8% retention instead, from within-player 50.1667,
random 19.2067 and shipped 56.6000 TDs per season. The difference demonstrates policy dependence,
not an invariant percentage of signal attributable to player identity.

**Calibration populations.** Slopes below one in all fifteen pooled season fits support
persistent marginal miscalibration on this population. They do not establish a universal,
stable correction for every position, selected player, forecast horizon or live input policy.
For positive rates the diagnostic is `E[Y] = exp(a + b * log(lam))`: an intercept is necessary
to assess level as well as slope. A slope of one is insufficient when `a != 0`; even `a = 0`
and `b = 1` do not validate conditional tails or joint outcomes.

The following are descriptive retrospective checks from the study's local per-season
`data/experiments/roster-snapshot-repair/<season>/forecasts.parquet` files, not a new frozen
train/apply experiment. Filter to shipped hard-eligible forecasts and, for selected groups,
that policy's own replay-pick indicators. Fit one positive-rate Poisson GLM across seasons;
forecast/actual is the ratio of summed forecasts to summed actual TDs in the same group.

| Population | Pooled Slope b | Forecast / Actual |
|---|---:|---:|
| All eligible current-week forecasts | 0.866 | 1.026 |
| Own greedy-selected forecasts | 1.038 | 1.226 |
| Own optimizer-selected forecasts | 0.999 | 1.212 |

Near-unit selected slopes do not remove selected level overprediction. These groups are
diagnostic and defined by decisions, not by future TDs or future participation. They do not
justify fitting to whichever retrospective subset looks favorable. Diagnose position, rate,
availability and selection strata, and forecast lead time, before specifying a correction.
Current-week calibration cannot be assumed to fit the optimizer's future surface. Repeated
predictions of the same future player-week share one outcome and are not independent rows.

**Ranking versus policy.** The common-pool top-10 diagnostic and greedy replay differences
have Spearman correlation 0.8571 across the seven non-null alternatives to shipped in this
study, derived from the saved ranking and replay summaries. That is a descriptive same-data
association across related models, not a validated model-selection rule. Neither a scaled
ranking difference nor its ratio to achieved season differences is a general conversion
coefficient. Policy claims require each policy's own no-reuse replay and paired uncertainty.

## Research Decision

Proceed with the staged [Phase 3 plan](IMPLEMENTATION_PLAN.md): repair diagnostic and capture
gaps, specify chronological train/apply comparisons, and reserve prospective/shadow evidence
for live transfer and policy validation. All 2011-2025 seasons have already been explored;
walk-forward refitting would improve temporal discipline, not create an untouched holdout.
Initial 2026 snapshots permit capture work to start now, but are not completed live validation.
Descriptive research need not wait for an entire prospective season.

Keep production constants unchanged unless appropriate evidence supports a separately reviewed
change. Better proper forecast scores and better season-policy TDs are different claims.
Static hold/commit sensitivity is not the dynamic value of waiting: current replay makes one
decision per week via `plan_slot`, not `advise_slot`. Multi-event replay is needed later for
that policy question. Keeping the model unchanged is a valid outcome of Phase 3.
