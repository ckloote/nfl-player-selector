# Phase 3B Calibration Experiment

Specification `experiments/phase3-calibration.toml`, dated 2026-09-06, frozen before any fit. Folds train on 2011-Y-1 and apply to Y for Y = 2016-2025; no outcome from Y enters its own fit, and a training row needs its target week played and scored, not merely an early forecast timestamp. All of these seasons have been explored before: this is walk-forward evaluation, not an untouched holdout.

## Fitted Maps

The map is `exp(a) * lam ** b` on positive rates, applied to the availability adjusted lam before the optimizer prunes or discounts it. Eligible zero rates are excluded and counted and hard exclusions stay masks, so every candidate is scored on the same rows. Of 100 fitted groups, 0 were unsupported and 0 used the declared fallback rather than their own coefficient.

| fold | candidate | a | b | source | n | clusters | fit_status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2016 | cal-level-pooled | -0.0667 | 1.0000 | own | 248026 | 29423 | ok |
| 2016 | cal-logaffine-pooled | -0.1722 | 0.8288 | own | 248026 | 29423 | ok |
| 2017 | cal-level-pooled | -0.0796 | 1.0000 | own | 326652 | 42435 | ok |
| 2017 | cal-logaffine-pooled | -0.1899 | 0.8291 | own | 326652 | 42435 | ok |
| 2018 | cal-level-pooled | -0.0929 | 1.0000 | own | 400078 | 51602 | ok |
| 2018 | cal-logaffine-pooled | -0.2075 | 0.8288 | own | 400078 | 51602 | ok |
| 2019 | cal-level-pooled | -0.0886 | 1.0000 | own | 472189 | 60935 | ok |
| 2019 | cal-logaffine-pooled | -0.2142 | 0.8170 | own | 472189 | 60935 | ok |
| 2020 | cal-level-pooled | -0.0941 | 1.0000 | own | 538689 | 70255 | ok |
| 2020 | cal-logaffine-pooled | -0.2188 | 0.8177 | own | 538689 | 70255 | ok |
| 2021 | cal-level-pooled | -0.0910 | 1.0000 | own | 604483 | 79778 | ok |
| 2021 | cal-logaffine-pooled | -0.2116 | 0.8215 | own | 604483 | 79778 | ok |
| 2022 | cal-level-pooled | -0.1072 | 1.0000 | own | 676775 | 90028 | ok |
| 2022 | cal-logaffine-pooled | -0.2274 | 0.8195 | own | 676775 | 90028 | ok |
| 2023 | cal-level-pooled | -0.1177 | 1.0000 | own | 747850 | 99956 | ok |
| 2023 | cal-logaffine-pooled | -0.2363 | 0.8219 | own | 747850 | 99956 | ok |
| 2024 | cal-level-pooled | -0.1182 | 1.0000 | own | 819313 | 109668 | ok |
| 2024 | cal-logaffine-pooled | -0.2400 | 0.8195 | own | 819313 | 109668 | ok |
| 2025 | cal-level-pooled | -0.1111 | 1.0000 | own | 890624 | 119468 | ok |
| 2025 | cal-logaffine-pooled | -0.2380 | 0.8142 | own | 890624 | 119468 | ok |

Coefficients by position, across folds. WR and TE are fitted separately because they share the FLEX slot, so their two maps are the ones able to reorder it.

| candidate | group | folds | a_min | a_max | b_min | b_max | sources |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cal-level-position | QB | 10 | -0.1440 | -0.0984 | 1.0000 | 1.0000 | own |
| cal-level-position | RB | 10 | -0.1108 | -0.0357 | 1.0000 | 1.0000 | own |
| cal-level-position | TE | 10 | -0.1935 | -0.1618 | 1.0000 | 1.0000 | own |
| cal-level-position | WR | 10 | -0.0548 | 0.0283 | 1.0000 | 1.0000 | own |
| cal-logaffine-position | QB | 10 | -0.0732 | -0.0342 | 0.6245 | 0.6959 | own |
| cal-logaffine-position | RB | 10 | -0.3798 | -0.2940 | 0.7450 | 0.7817 | own |
| cal-logaffine-position | TE | 10 | -0.6300 | -0.4588 | 0.7300 | 0.8094 | own |
| cal-logaffine-position | WR | 10 | -0.4131 | -0.2871 | 0.7394 | 0.7668 | own |

## Forecast Score

Primary estimand: paired change in season-mean Poisson deviance on hard-eligible current-week rows with identity lambda above 0.0, which is the same population for every candidate because a map sends zero to zero. Deviance is a loss, so a negative `delta` is a better forecast than identity's. Uncertainty is the paired season t on 10 season differences, and `p_holm` is the Holm step-down adjustment over the pre-registered family, which stops at the first comparison it does not reject. **`reject` is the adjusted result and the only one the promotion rule reads.** `lo`/`hi` is the ordinary unadjusted interval a single comparison would report. `stage_lo`/`stage_hi` is each step's own local test boundary, and Holm's levels widen down the ranking, so a later step's local interval can exclude zero on a step the procedure never reached; it is a diagnostic, never a bound. Nothing here is a verdict: the rule is applied in a separate, dated note.

| model | delta | se | seasons | t | p_value | holm_rank | p_holm | reject | lo | hi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cal-logaffine-position | -0.0089 | 0.0019 | 10 | -4.6898 | 0.0011 | 1 | 0.0045 | True | -0.0132 | -0.0046 |
| cal-logaffine-pooled | -0.0049 | 0.0012 | 10 | -4.2097 | 0.0023 | 2 | 0.0068 | True | -0.0076 | -0.0023 |
| cal-level-pooled | 0.0015 | 0.0009 | 10 | 1.6156 | 0.1406 | 3 | 0.2813 | False | -0.0006 | 0.0036 |
| cal-level-position | 0.0012 | 0.0009 | 10 | 1.2994 | 0.2261 | 4 | 0.2813 | False | -0.0009 | 0.0032 |

## Achieved Season Scores

Each candidate replays its own no-reuse greedy and optimizer history and is compared against identity's replay of the same strategy, never against identity greedy. A better proper score and more touchdowns are different claims.

The declared non-inferiority bound is 0.5 touchdowns per season, and it is compared against `lo` -- the lower end of the paired interval -- rather than against the point estimate. A candidate that reproduced identity's whole pick history differs from it by exactly nothing in every season, which is `degenerate_zero_variance`: an interval of width zero, and a bound, rather than the missing inference a zero standard error would otherwise read as.

| model | strategy | tds_per_season | delta_vs_own_identity | paired_se | comparison | lo | hi | seasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cal-level-pooled | greedy | 54.4000 | 0.0000 | 0.0000 | degenerate_zero_variance | 0.0000 | 0.0000 | 10 |
| cal-level-pooled | optimizer | 55.8000 | 0.0000 | 0.0000 | degenerate_zero_variance | 0.0000 | 0.0000 | 10 |
| cal-level-position | greedy | 54.3000 | -0.1000 | 0.4069 | t | -1.0204 | 0.8204 | 10 |
| cal-level-position | optimizer | 56.0000 | 0.2000 | 0.2906 | t | -0.4574 | 0.8574 | 10 |
| cal-logaffine-pooled | greedy | 54.4000 | 0.0000 | 0.0000 | degenerate_zero_variance | 0.0000 | 0.0000 | 10 |
| cal-logaffine-pooled | optimizer | 55.6000 | -0.2000 | 0.6110 | t | -1.5822 | 1.1822 | 10 |
| cal-logaffine-position | greedy | 54.4000 | 0.0000 | 0.5578 | t | -1.2618 | 1.2618 | 10 |
| cal-logaffine-position | optimizer | 56.3000 | 0.5000 | 0.7491 | t | -1.1945 | 2.1945 | 10 |
| shipped | greedy | 54.4000 | NA | NA | NA | NA | NA | 10 |
| shipped | optimizer | 55.8000 | NA | NA | NA | NA | NA | 10 |

## Decision Changes

How often a candidate's replay chose a different player than identity's did. A shared strictly increasing map preserves greedy's order within a slot; separate WR and TE maps need not, because they compete in FLEX. The optimizer has no such guarantee even under a shared map: it maximises an assignment sum, and a nonlinear map can reorder assignments and reorder cells at different horizons once the discount is applied.

| model | strategy | decisions | changed |
| --- | --- | --- | --- |
| cal-level-pooled | greedy | 525 | 0 |
| cal-level-pooled | optimizer | 525 | 0 |
| cal-level-position | greedy | 525 | 33 |
| cal-level-position | optimizer | 525 | 44 |
| cal-logaffine-pooled | greedy | 525 | 0 |
| cal-logaffine-pooled | optimizer | 525 | 12 |
| cal-logaffine-position | greedy | 525 | 45 |
| cal-logaffine-position | optimizer | 525 | 66 |

## Static Hold Sensitivity

`advise_slot` compares a season cost in touchdowns against a fixed 0.1 premium, so rescaling the rates rescales one side of that comparison and not the other. These counts are that sensitivity and nothing more: replay makes one decision per week through `plan_slot`, so a flipped hold here is not a touchdown gained or lost. Measuring the value of waiting needs the multi-event replay Phase 3C specifies.

| model | decisions | pick_changed | hold_changed |
| --- | --- | --- | --- |
| cal-level-pooled | 525 | 0 | 2 |
| cal-level-position | 525 | 31 | 2 |
| cal-logaffine-pooled | 525 | 1 | 9 |
| cal-logaffine-position | 525 | 41 | 16 |

## Outcome Coverage

Whether the applied surface has an outcome to be scored against, which is what the declared coverage floors are conditions on. An outcome is settled from the finalized scoring ledger, absence counting as zero only where the target week is completely scored, so what is missing here is a week the feed has not finished rather than a player the pool has not kept. Over hard-eligible rows, measured on identity: a map preserves keys, masks and zeros, so every candidate is scored on exactly these rows. `meets_floor` is a comparison of two measured numbers, not a promotion decision.

| population | rows | outcomes_known | outcomes_missing | coverage | declared_floor | meets_floor |
| --- | --- | --- | --- | --- | --- | --- |
| current | 75797 | 75797 | 0 | 1.0000 | 0.9800 | True |
| future | 638509 | 638509 | 0 | 1.0000 | 0.9500 | True |

Retention is a different question, reported so that it cannot be mistaken for the first: what fraction of the surface belonged to a player still in the candidate pool at the week he was forecast for. It is the lower number, and it should be. Taking outcomes from the target week's own forecast row conflated the two and reported this as coverage, which understated coverage by exactly the players who left -- and dropped their outcomes, nearly all of them zeros, from the fit.

| population | rows | in_target_pool | retention |
| --- | --- | --- | --- |
| current | 75797 | 75797 | 1.0000 |
| future | 638509 | 534745 | 0.8375 |

## Training Population

What each fold discarded before fitting, in the classes that mean different things. A hard exclusion is a mask rather than a forecast; an eligible zero rate cannot enter a fit on `log(lambda)`; an unresolved outcome is a week that is not completely scored. The counts reconcile to the surface each fold started from, so "excluded and counted" covers the training population and not only the applied one.

| fold | surface_rows | hard_excluded | eligible_zero_lam | unresolved_outcome | fitted_rows | fitted_share |
| --- | --- | --- | --- | --- | --- | --- |
| 2016 | 249038 | 1012 | 0 | 0 | 248026 | 0.9959 |
| 2017 | 328046 | 1394 | 0 | 0 | 326652 | 0.9958 |
| 2018 | 401756 | 1678 | 0 | 0 | 400078 | 0.9958 |
| 2019 | 474194 | 2005 | 0 | 0 | 472189 | 0.9958 |
| 2020 | 540777 | 2088 | 0 | 0 | 538689 | 0.9961 |
| 2021 | 606575 | 2092 | 0 | 0 | 604483 | 0.9966 |
| 2022 | 678869 | 2094 | 0 | 0 | 676775 | 0.9969 |
| 2023 | 749944 | 2094 | 0 | 0 | 747850 | 0.9972 |
| 2024 | 821408 | 2095 | 0 | 0 | 819313 | 0.9974 |
| 2025 | 892721 | 2097 | 0 | 0 | 890624 | 0.9977 |

## Out-Of-Fold Diagnostics

The proper score by forecast horizon and availability, which the pooled fit declares two consequences for and cannot itself show. Late target weeks are forecast, and so represented, more often than early ones under equal row weight; and for an exponent away from one the Questionable multiplier is rescaled nonlinearly, so it is no longer a clean multiplier on the calibrated rate. These are declared diagnostics and cannot substitute for the primary estimand.

| model | horizon | availability | n | deviance | delta | paired_se |
| --- | --- | --- | --- | --- | --- | --- |
| cal-level-pooled | 0 | full | 72457 | 0.6961 | 0.0015 | 0.0010 |
| cal-level-pooled | 0 | questionable | 3340 | 0.7122 | 0.0008 | 0.0018 |
| cal-level-pooled | 1 | full | 72238 | 0.7049 | -0.0013 | 0.0010 |
| cal-level-pooled | 2-3 | full | 130497 | 0.7149 | -0.0028 | 0.0011 |
| cal-level-pooled | 4-6 | full | 162549 | 0.7257 | -0.0045 | 0.0012 |
| cal-level-pooled | 7+ | full | 273225 | 0.7506 | -0.0068 | 0.0014 |
| cal-level-position | 0 | full | 72457 | 0.6958 | 0.0012 | 0.0009 |
| cal-level-position | 0 | questionable | 3340 | 0.7125 | 0.0011 | 0.0018 |
| cal-level-position | 1 | full | 72238 | 0.7049 | -0.0013 | 0.0010 |
| cal-level-position | 2-3 | full | 130497 | 0.7151 | -0.0027 | 0.0011 |
| cal-level-position | 4-6 | full | 162549 | 0.7260 | -0.0043 | 0.0012 |
| cal-level-position | 7+ | full | 273225 | 0.7509 | -0.0065 | 0.0015 |
| cal-logaffine-pooled | 0 | full | 72457 | 0.6898 | -0.0048 | 0.0012 |
| cal-logaffine-pooled | 0 | questionable | 3340 | 0.7031 | -0.0082 | 0.0042 |
| cal-logaffine-pooled | 1 | full | 72238 | 0.6966 | -0.0095 | 0.0011 |
| cal-logaffine-pooled | 2-3 | full | 130497 | 0.7048 | -0.0130 | 0.0011 |
| cal-logaffine-pooled | 4-6 | full | 162549 | 0.7134 | -0.0168 | 0.0014 |
| cal-logaffine-pooled | 7+ | full | 273225 | 0.7339 | -0.0236 | 0.0018 |
| cal-logaffine-position | 0 | full | 72457 | 0.6860 | -0.0086 | 0.0019 |
| cal-logaffine-position | 0 | questionable | 3340 | 0.6937 | -0.0177 | 0.0061 |
| cal-logaffine-position | 1 | full | 72238 | 0.6903 | -0.0159 | 0.0017 |
| cal-logaffine-position | 2-3 | full | 130497 | 0.6969 | -0.0208 | 0.0019 |
| cal-logaffine-position | 4-6 | full | 162549 | 0.7032 | -0.0270 | 0.0023 |
| cal-logaffine-position | 7+ | full | 273225 | 0.7201 | -0.0373 | 0.0034 |

## Methods And Provenance

Source `bce82abeea3dfb43b8c3319a8a742d07939bcbff`, dataset `c0f564496c80`, configuration `0f079f9c5a12`. Production constants are unchanged and no calibrated model is deployed; the future discount, information premium and every base-model constant were fixed for the whole experiment. Candidate fallback order is ['own', 'pooled', 'identity'], with a group needing at least 500 rows and 30 clusters to use its own coefficient. Declared margins: `max_policy_loss_td_per_season` = 0.5, `min_deviance_improvement` = 0.005, `min_outcome_coverage_current` = 0.98, `min_outcome_coverage_future` = 0.95. Repeated forecasts of one target week share its single outcome, so fits cluster on the target player week and rows carry equal weight; late target weeks are therefore forecast, and so represented, more often than early ones. Identity is `shipped` and is mandatory: fold artifacts, out-of-fold surfaces, picks and these tables are saved beside the run. Applying the promotion rule is a separate, dated authoring step, and keeping the model unchanged is a valid outcome. The promotion conditions are frozen with the rest of the specification as `calibration_experiment.promotion` (alpha = 0.05, decision = holm_step_down, direction = improvement, max_promoted = 1, policy_strategies = both, require_coverage_floors = True, require_deviance_margin = True), and the specification text itself hashes to `4f0a88ea8874`. Candidates: cal-level-pooled, cal-level-position, cal-logaffine-pooled, cal-logaffine-position.


### Recorded Verification

Recorded metric implementation commit: `bce82abeea3dfb43b8c3319a8a742d07939bcbff`. The original manifest records the metric source and dataset identities.

All 25 stage-season records completed, covering 15 unique seasons: 65 model/seed/season runs, 489,534 forecast rows, 130 achieved season scores, and 6,810 individual replay picks. All 4,175 required games have complete scoring and player-stat coverage for both teams.

2022 includes 271 completed games; the [Bills–Bengals game was canceled](https://www.buffalobills.com/news/nfl-says-neutral-site-afc-championship-game-is-possible-bills-bengals-week-17-ga). Its absence from the final schedule is a historical-replay approximation.

Original verification record: 392 pytest tests passed. Lint: Ruff check passed. Formatting: Ruff format --check passed. These are recorded results, not checks executed by the presentation step.

Recorded checks:

- all configured models and seeds
- candidate/eligibility/depletion masks
- hard exclusions unranked
- explicit decision timestamps
- consistent finalized actuals
- no player reuse
- pick sums equal season scores
- unique-player counts
- forecast pick indicators match replays
- game coverage
- both teams player-stat coverage
- fitted artifacts hash as recorded
- every candidate's artifact carries its own declared fit
- no training key reaches its own apply season
- identity reproduces the pairs stage exactly
- candidate surfaces are their own artifact applied to identity
- candidate surfaces keep identity's keys, scaffold and outcomes
- every model's scored forecast rows are exactly its own horizon-zero surface
- fold digests cover the values that determined the coefficients
- training rows are accounted for before they are filtered

The pinned dependencies and thread environment are recorded in the metric manifest; retain that environment when resuming model computation. See [experiment instructions](../../README.md) for commands and schema.

### Presentation

Rendered from saved compact metrics; no forecasts, replays, or calibration fits were recomputed. `presentation.json` records the rendering-source hashes. The metric manifest and original verification record are unchanged.
