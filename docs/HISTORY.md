# Project history

How the project reached its current state, and where each stage is written down. The weekly
workflow is in the [README](../README.md); research methods and results are in
[RESEARCH.md](RESEARCH.md).

## Documents

- [`docs/IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — phased build plan and status
- [`docs/EVALUATION.md`](EVALUATION.md) - generated season scores, forecast diagnostics, methods and provenance
- [`docs/ANALYSIS.md`](ANALYSIS.md) - dated, authored interpretation and research limits; not refreshed by reruns
- [`docs/PHASE3B_OUTCOME.md`](PHASE3B_OUTCOME.md) - completed calibration experiment: no candidate promoted, rationale and next step
- [`docs/PHASE3C_PROTOCOL.md`](PHASE3C_PROTOCOL.md) - dated prospective baseline protocol: window, decision events, parity criteria and floors, written but never run
- [`docs/PHASE3C_OUTCOME.md`](PHASE3C_OUTCOME.md) - the decision to close that window without collecting, and what it gives up. Its tooling (`prospective.py`, `pool baseline`) was removed on 2026-09-19; revision `bc57683` is the last that has it, and its reconstruction and parity checks remain as `pool research verify-capture`
- [`docs/PHASE4_PLAN.md`](PHASE4_PLAN.md) - opponent ingestion, standings, and deciding by expected share of the pot: staging, and what can and cannot be validated
- [`docs/PHASE4_STAGE1_PLAN.md`](PHASE4_STAGE1_PLAN.md) - implemented report ingestion, archive guarantees, and identity checks
- [`docs/PHASE4_STAGE2_PLAN.md`](PHASE4_STAGE2_PLAN.md) - implemented shared scoring, computed standings, ties, and independent used pools
- [`docs/PHASE4_STAGE3_PLAN.md`](PHASE4_STAGE3_PLAN.md) - implemented pot-share policy: the simulator, the opponent model, what it can and cannot be judged on
- [`experiments/results/phase4-simulator/CALIBRATION.md`](../experiments/results/phase4-simulator/CALIBRATION.md) - the simulator fitted against 2011-2025, with the checks it fails beside the ones it passes
- [`docs/REVIEW.md`](REVIEW.md) — September 2026 review: open defects, validation limitations, and recommended priorities
- [`docs/PROJECT_REVIEW_2026-09-17.md`](PROJECT_REVIEW_2026-09-17.md) — the September 17
  review: pot-share input semantics, package and capture integrity, and the simplification that
  produced `pool week`
- [`docs/archive`](archive) — the original benchmark and backtest reports, under their old
  scoring and evaluation definitions

## Status on 18 September 2026, before `pool week`

The weekly workflow and Phase 2 validation repairs are implemented: explicit deadline
eligibility, complete touchdown accounting, timestamped input archives, shared baseline
comparisons, and reproducible season experiments. Production model constants remain fixed.
The [Phase 3B report](../experiments/results/phase3-calibration/CALIBRATION.md) covers the completed
2016-2025 walk-forward calibration experiment. Position-specific log-affine maps improved
forecast accuracy, but no candidate passed all signed forecast and policy gates. The
[authored outcome](PHASE3B_OUTCOME.md) is no promotion; production remains unchanged.
Phase 3C was declared as identity-only prospective collection and then
[closed without collecting](PHASE3C_OUTCOME.md): its [protocol](PHASE3C_PROTOCOL.md)
stands as written and unrun, no decision was ever captured under it, and no floor was
measured. The window required a source freeze for its duration, which the season's own
work — opponent ingestion, standings, a win-probability objective — could not accommodate.
The replay assumption it would have checked is still open, and a single captured decision
verified the same day settles it whenever that is worth doing.

Phase 4 is implemented. Weekly entrant reports are archived before parsing, imported with
identity checks, and scored through the same core as your own picks. `pool standings` shows
computed ranks and reported values together, preserves ties, and tracks each entrant’s used
players. `pool report list` includes failed imports. See the
[stage 1](PHASE4_STAGE1_PLAN.md) and [stage 2](PHASE4_STAGE2_PLAN.md) plans.

[Stage 3](PHASE4_STAGE3_PLAN.md) adds a second objective **beside** the first, never in
place of it: `pool recommend` still gives the expected-TD advice and now prints each
candidate's expected share of the pot next to it, with a sentence naming why the two
disagree when they do. A tie at the top splits the winnings, so the quantity is a share and
not a probability of an outright win. The
[simulator calibration](../experiments/results/phase4-simulator/CALIBRATION.md) reports each of
its four checks pass or fail: one passes outright, two pass only in part, and same-team
substitution fails and is deferred with its route written down. Nothing here establishes that the policy wins
pools and nothing can: a season is one Bernoulli trial. What accrues instead is prospective
and weekly — `pool predict record` archives a prediction of every rival's picks before the
report that settles it, and `pool predict score` reads the record back. `pool backtest
--strategy winprob` replays a finished season and says, everywhere it prints, that it did so
against rivals who do not exist.

The [evaluation report](EVALUATION.md) combines actual season scores from each model's
greedy and optimizer pick history with ranking and calibration diagnostics. Its
[ranking section](EVALUATION.md#ranking-diagnostics) retains the distinct unit of TDs per
ranked candidate, not achieved season scores. The study covers 2011–2025 with 2010 prior history;
both era summaries are retrospective. Its saved configuration,
scoring coverage, seeds, source hashes and provenance accompany the results in
[`experiments/results/roster-snapshot-repair`](../experiments/results/roster-snapshot-repair), which
reran the identical frozen dataset after the F11 candidate-pool repair. The superseded
[Phase 2 results](../experiments/results/phase2-validation) remain published for comparison.
Previous reports are retained in [`docs/archive`](archive) under their old scoring and
evaluation definitions.

Generated reports are for facts, metric definitions and provenance, not automated research
conclusions. Human/AI reasoning is maintained separately in the dated [analysis](ANALYSIS.md).
Its interpretation of this retrospective study is not automatically refreshed by reruns.

The assignment solver supplies a rest-of-season plan and the projected cost of overriding it.
Its optimality for a fixed, pruned forecast matrix does not establish a rolling-policy scoring
advantage. One strictly increasing map shared by all candidates in a slot preserves greedy
ordering; position-specific WR/TE maps can change FLEX ranks. Calibration can also change
assignment and hold/commit decisions. Joint ablations do not isolate defense alone, and these
model comparisons do not establish that the available inputs are exhausted or that simulations
are validated.

## The weekly routine before `pool week`

Until 18 September 2026 the README taught a routine written for the Phase 3C collection
window: refresh immediately before every capture, browse with `recommend --no-capture`,
capture once per kickoff wave, and name that capture with `record --decision`. The window was
[closed without collecting](PHASE3C_OUTCOME.md), so none of that was required any more,
and `pool week` replaced it: it refreshes when the data is stale, saves a decision only while a
slot is open, and prints the `record` line with the decision already named. The
[protocol](PHASE3C_PROTOCOL.md) that defined the old routine stands as written.

## Stored pick scores, before 19 September 2026

Until 19 September 2026 `pool picks` read each pick's touchdowns from `my_picks.tds`, a copy
that only `pool score` wrote, so a finished game showed "not scored; run pool score" and
standings warned when the copy was stale. Standings, `pool week` and the rival predictions
already scored picks from the results each time. Now `pool picks` does too: pick scoring moved
from `scoring.py` to `results.py`, the column stays in the schema unread, and `pool score` is a
hidden name for `pool picks`.

`scoring.py` is in the decision closure, so this moved the decision fingerprint, together with
a fix in the same PR that lets rivals hold a player whose game has started. The fingerprint
went from `be2b9bbb…` to `23c28ae1…`, and revision `67ab143` is the last with the old one.
Decisions captured before verify with `pool research verify-capture --allow-code-drift`, and
reconstruct exactly at the revision each one recorded.
