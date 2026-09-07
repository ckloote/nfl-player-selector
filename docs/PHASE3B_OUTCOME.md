# Phase 3B Outcome: No Promotion

**Author:** OpenCode (AI-authored interpretation), 2026-09-07 (UTC).

**Disposition:** defer calibration promotion; keep the shipped identity model and all
production constants unchanged. No candidate qualifies for Phase 3C under the signed rule.

This is the separately authored outcome of the completed chronological calibration
experiment, not an automatically generated recommendation. It applies the unchanged
[pre-run margin sign-off](PHASE3B_MARGIN_SIGNOFF.md) to the
[published report](../experiments/results/phase3-calibration/CALIBRATION.md) and its saved
tables. Phase 3B is complete with a no-promotion result; Phase 3C validation is not complete.

## Evidence And Provenance

The [resolved configuration](../experiments/results/phase3-calibration/resolved-config.json)
specifies annual expanding folds trained on 2011 through Y-1 and applied to 2016-2025,
using historical inputs, usage roles, and fixed shipped constants. The four candidates
are level and log-affine maps, each pooled or position-specific, against identity.

- Metric implementation: `bce82abeea3dfb43b8c3319a8a742d07939bcbff`.
- Metric source hash prefix: `6934e773fbcc`; dataset hash prefix: `c0f564496c80`.
- Configuration hash prefix: `0f079f9c5a12`; specification hash prefix: `4f0a88ea8874`.
- Full identities are preserved in the [manifest](../experiments/results/phase3-calibration/manifest.json).
- The [original verification](../experiments/results/phase3-calibration/verification.json)
  records 392 passing tests, Ruff checks, and 4,175 required games with complete scoring
  and player-stat coverage. These are recorded run checks, not tests executed by this note.

There are 15 unique historical seasons: 15 identity-pairs records and 10 apply records,
or 25 stage-season records in total. The inferential sample is **ten paired evaluation
seasons**, not 25 seasons or the hundreds of thousands of repeated forecast rows.

## Applying The Signed Gates

Deviance is a loss: candidate-minus-identity `delta` must be below `-0.005`, and the
comparison must pass Holm step-down rejection at 0.05. Each policy's ordinary paired
95% lower bound must exceed `-0.5 TD/season`, against identity's same strategy.

| Candidate | Deviance delta | Holm p | Forecast gate | Greedy lower bound | Optimizer lower bound | Both policy gates |
|---|---:|---:|---|---:|---:|---|
| `cal-level-pooled` | +0.001488 | 0.281253 | Fail | 0.000000 | 0.000000 | Pass |
| `cal-level-position` | +0.001173 | 0.281253 | Fail | -1.020438 | -0.457368 | Fail |
| `cal-logaffine-pooled` | -0.004947 | 0.006821 | Fail: magnitude | 0.000000 | -1.582201 | Fail |
| `cal-logaffine-position` | -0.008930 | 0.004546 | Pass | -1.261771 | -1.194522 | Fail |

Sources: [paired-deviance.csv](../experiments/results/phase3-calibration/paired-deviance.csv)
and [policy.csv](../experiments/results/phase3-calibration/policy.csv). Rounded values above
are for display; the decisions use the saved precision. In particular, pooled log-affine
improves by `0.00494724053196`, which does not exceed `0.005`; rounding cannot make it pass.
It also fails the optimizer policy gate, independently of that near-miss.

Both coverage gates pass for every candidate. Current coverage is 75,797/75,797 and future
coverage is 638,509/638,509, both 1.000. Future retention is 0.837490, a separate fact about
remaining in the target-week candidate pool, not missing outcomes. See
[coverage-margins.csv](../experiments/results/phase3-calibration/coverage-margins.csv).

**No candidate passes all gates.** No margin is relaxed, no diagnostic replaces the primary
metric, and no candidate is advanced by choosing only the strategy on which it looks better.

## Interpretation

**Forecast accuracy improved most under position-specific log-affine calibration.** Its
mean deviance reduction of 0.008930 is approximately 1.28% of identity's apply-season mean
of 0.695013. It improves all ten seasons, exceeding 0.005 in seven, and passes the full
forecast gate. Its ordinary interval does not establish improvement greater than 0.005,
but the signed rule required that practical margin for the point estimate, not the bound.
All 100 fitted groups were supported and used their own coefficients, with no fallback.

**Policy non-inferiority is unestablished, not disproved.** For that candidate, greedy
averages 54.4 TD/season, equal to identity, with a difference interval of
`[-1.261771, +1.261771]`. Optimizer averages 56.3 versus identity's 55.8, a difference of
`+0.5` with interval `[-1.194522, +2.194522]`. Neither rules out losses beyond the allowance.
The optimizer's differences range from -4 to +4 TD; its last two seasons contribute +7 TD
against a ten-season net gain of +5 TD. That variability explains the uncertainty, not a
reason to discard seasons. Non-significance is not equivalence or evidence of harm.

The position-specific log-affine map changes 45 of 525 greedy decisions and 66 of 525
optimizer decisions. Pooled level changes neither history, so its zero-width policy
intervals reflect unchanged replays under the declared convention, not general evidence
that changed policies are safe. The 16 static hold flips for position-specific log-affine
are sensitivity results, not touchdowns gained by waiting.

**Future-horizon gains do not override the current-week result.** Position-specific
log-affine improvements increase from approximately 0.0159 at horizon 1 to 0.0373 at
horizon 7+, on the reported full-availability strata. Level-only maps improve the reported
future strata while slightly worsening the primary current-week score. This motivates a
question about fitting across horizons, not a post-hoc replacement of the frozen estimand.
The horizon populations differ, and repeated forecasts share target outcomes.

Sources for these checks are the saved [season deviance](../experiments/results/phase3-calibration/deviance.csv),
[replays](../experiments/results/phase3-calibration/replays.csv),
[decision changes](../experiments/results/phase3-calibration/decision-changes.csv), and
[stratified scores](../experiments/results/phase3-calibration/strata-out-of-fold.csv).

## Recommended Next Step

Proceed with **identity-only prospective evidence collection and live/snapshot validation**
using the existing Phase 3A capture infrastructure. Before reviewing subsequent outcomes,
write a dated Phase 3C baseline protocol fixing the collection/review window, decision
events, live role/input policy, populations, coverage requirements, and parity checks.
Keep submitted picks and production settings unchanged. This is baseline validation, not
promotion of any failed Phase 3B candidate, and initial captures do not establish live
policy benefit. Multi-event replay remains necessary before any claim about waiting for news.

If calibration research continues, first specify a separate, bounded experiment on horizon
weighting, retaining the small candidate families and explicit policy safeguards. Its
design and evidence requirements need fresh review before execution. These inspected
historical results may motivate it but cannot become an untouched confirmation set;
do not tune until a candidate passes or relax the current margins to rescue one.

The immediate deliverable is the identity-baseline protocol and ongoing capture, not another
calibration sweep. The [implementation plan](IMPLEMENTATION_PLAN.md#phase-3c-shadow-live-validation)
records the sequence. The ten retrospective seasons, historical timing/stat corrections,
and lack of prospective confirmation remain limitations. This outcome does not establish
that calibration cannot help; it establishes that none of these candidates earned promotion.
