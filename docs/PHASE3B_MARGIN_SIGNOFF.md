# Phase 3B Margin Sign-Off

**Decision date:** 2026-09-07 (UTC).

**Decision maker:** project owner, following the pre-run margin discussion.

**Status:** approved before the full benchmark; no candidate selected or deployed.

This records owner approval of the unchanged margins and promotion rule in
[`experiments/phase3-calibration.toml`](../experiments/phase3-calibration.toml), reviewed at
commit `1f8a8ad6e2b72928d4032fd73e1f32c1f72f444f` on
[PR #4](https://github.com/ckloote/nfl-player-selector/pull/4). The full benchmark has not
been run at sign-off. This documentation decision changes neither the specification nor
production constants.

## Approved Decision Rule

At most one candidate may advance into Phase 3C, and only if every gate passes:

| Gate | Approved threshold and interpretation |
|---|---|
| Forecast improvement | Candidate-minus-identity season-mean Poisson deviance `delta < -0.005`, with Holm step-down rejection at `alpha = 0.05` across the four declared comparisons. |
| Policy non-inferiority | For both greedy and optimizer independently, the paired interval lower bound `lo > -0.5` TD/season against identity's replay of the same strategy. |
| Current outcome coverage | At least `0.98` of the declared hard-eligible current-week surface has resolved outcomes. |
| Future outcome coverage | At least `0.95` of the declared hard-eligible future surface has resolved outcomes. |

The forecast metric uses the fixed hard-eligible current-week population with positive
identity rates and equal season weighting. Inference remains paired-season t with the
declared Holm correction for the primary forecast comparisons. Policy bounds remain
ordinary two-sided 95% paired intervals, with the existing `degenerate_interval`
convention for valid zero-SE comparisons. Local `stage_lo`/`stage_hi` intervals are
diagnostics, not substitutes for the Holm step-down result.

Failure to clear any gate is a valid defer/no-change outcome. Passing the gates permits
consideration for Phase 3C; it does not compel selection or authorize production use.
Applying the rule remains a separate, dated decision after reviewing the verified results.

## Rationale

### Forecast Improvement

Retain `min_deviance_improvement = 0.005`. This is approximately a 0.7% reduction relative
to the Phase 3A reference deviance of 0.6977, a modest screening threshold appropriate to
advancing a small calibration model into further evaluation. That reference is scale
context, not the exact baseline of the new apply-season estimand. A deviance improvement
does not imply an equivalent increase in touchdowns or win probability.

The rule requires a point improvement greater than 0.005 and adjusted evidence against
zero improvement. It does not require a confidence bound establishing improvement greater
than 0.005. That stronger criterion is not the approved screening rule.

### Policy Risk

Retain `max_policy_loss_td_per_season = 0.5`. The owner accepts advancing a candidate with
better forecast accuracy and a small observed retrospective TD decline, provided both
policy lower bounds exceed -0.5 TD/season and the other gates pass. This is an explicit
non-inferiority allowance for further validation, not permission to deploy a model that
loses half a touchdown per season.

Half a touchdown is roughly 1% of the cited historical identity greedy score of 53.73
TD/season, but may still matter in an individual pool season. The bound, not merely the
observed loss, controls the gate. With ten seasons and positive SE, the lower bound is
approximately `mean difference - 2.262 * paired SE`; a near-zero mean therefore needs an
SE below approximately 0.22 to pass. A positive point estimate can fail when uncertainty
is large. Non-significance is not equivalence, and these retrospective bounds do not
guarantee future policy performance.

### Outcome Coverage

Retain `min_outcome_coverage_current = 0.98` and
`min_outcome_coverage_future = 0.95` as data-quality safeguards, not performance targets.
Outcomes come from finalized touchdown credits, with absent credits treated as zero only
in completely scored target weeks. Roster retention is a separate diagnostic. The earlier
smoke figure of 0.8907 measured retention and is not grounds for relaxing outcome coverage.

Coverage should normally be 1.000 under the fully audited retrospective runner. Unexpected
missingness warrants investigation even if an aggregate floor passes, particularly if it
is concentrated by position or forecast horizon. Passing coverage alone is not evidence
of forecast quality.

## Freeze And Next Steps

Smoke results have already been examined, and the historical seasons were explored before
this experiment. This sign-off freezes the decision rule before the full run; it is not
pristine preregistration against untouched data. The thresholds will not be relaxed to
make the full-run results pass. Any later methodological amendment must be explicitly
dated and treated as a new experiment, not attached to the existing run after inspection.

The owner will merge and run the tests and benchmark manually. This sign-off is not a
test result or experiment-completion record. Use the documented resume, verification and
publication workflow before the separately authored Phase 3B outcome decision. Production
remains unchanged pending the distinct Phase 3C evidence and review requirements.
