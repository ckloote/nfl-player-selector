# Phase 3C Baseline Protocol

**Specification date:** 2026-09-07 (UTC).

**Status: closed without collection, 2026-09-07 (UTC).** No decision was ever captured
under this protocol and no floor was measured. See the
[outcome](PHASE3C_OUTCOME.md) for the decision and what it gives up. Everything below is
kept as written — it resolves, it is tested, and it is the specification to start from if
this window is ever re-declared against a new date. Nothing in it was run.

**Decision maker:** project owner, before the 2026 week 1 pick deadline.

**Scope of approval, unchanged by the amendment.** Signing authorises collection, fidelity
verification and the descriptive diagnostics below. It authorises no candidate promotion,
no production change, no achieved-score claim and no waiting-value claim. The declared
floors and window will not be relaxed after inspection, and a failing floor will be
reported as an outcome rather than adjusted away.

### Sign-off history

**Signed 2026-09-07** on [PR #6](https://github.com/ckloote/nfl-player-selector/pull/6) at
commit `48a9f3303110a659d38dca735644bdf905b0cd05`, covering specification
`958118e635d5b2718a10dbe0846335e0e39306a4445a23386fb370656fa474ab`.

**Closed 2026-09-07** without collection, before the first captured decision. The
amendment below was never signed, and the window it describes was never opened.

**Amended 2026-09-07**, the same day, before the first captured decision. The declared
schedule named two decision events per week, but every week of this window has three
kickoff waves and week 1 has four: `classify_event` filed a Thursday decision as the
Sunday one and a Monday-game decision -- made before its own kickoff -- as `after_deadline`.
Decisions are now classified by the wave whose deadline they beat, and undeclared waves are
reported with `scheduled = 0` rather than required. The required schedule, the floors, the
populations, the parity criteria and the window are untouched.

The specification is now
`14541e7253acb97f20cc9cf045fbbc21d9b8dfae44ab3e645b54ab7b682e8b75`, so the approval above
no longer covers it and the amended protocol needs a fresh sign-off before the first
capture. That hash is what every export carries: a window whose export reports a different
one was not run under the protocol that was signed.

This is an amendment against no evidence. The operational database held zero captured
decisions when it was written, so there was nothing to inspect and nothing that could have
suggested it; it is a correction to a preregistration still in its pre-registration state,
not an adjustment to a window in flight. After the first capture the rule in
[Freeze And Limits](#freeze-and-limits) applies instead, and a change of this kind would
start a new window rather than amend this one.

This is the dated collection protocol for Phase 3C, the counterpart to the
[Phase 3B margin sign-off](PHASE3B_MARGIN_SIGNOFF.md). It fixes the window, the decision
events, the live input policy, the populations, the parity criteria and the fidelity and
coverage floors before any of them can be chosen from the evidence. The machine-readable
form is [`experiments/phase3c-baseline.toml`](../experiments/phase3c-baseline.toml);
`prospective.resolve` refuses a protocol that omits any declaration, and hashes the file,
so an amendment starts a new collection window rather than reinterpreting this one.

**Nothing is under evaluation.** No candidate passed the Phase 3B gates, so the
[outcome](PHASE3B_OUTCOME.md) is no promotion and there is nothing to run in shadow.
This window collects identity only. Signing it authorises collection and the checks
below; it authorises no production change, no candidate, and no policy claim.

## The Window

| Declaration | Value |
|---|---|
| Season | 2026 |
| Collection weeks | 1 through 6 |
| Review point | after week 6 is completely scored |
| Required decision events | two per week: `thursday_deadline`, `sunday_slate` |
| Other kickoff waves | captured when worked, classified by their own deadline, `scheduled = 0` |
| Model / calibrator | `shipped` / `identity` |
| Input policy / roles | `live` / depth chart |
| Constants | shipped, unchanged |

Six weeks is sized to validate that the capture is faithful, that live and snapshot agree
and that coverage holds on forward data. It is not sized to detect a forecast or policy
difference, and the review must not report one. Extending the window is an amendment, and
an amendment starts a new window rather than lengthening this one after the fact.

### Decision events

`thursday_deadline` is the decision made before the week's first confirmed kickoff
deadline. `sunday_slate` is the decision made before the main Sunday slate begins. A week
whose first game is already in the main slate schedules one event, not two: counting an
impossible event as a miss would make the schedule, rather than the operator, decide
whether the window passed.

These two are what the capture floor requires. They are not the only decisions a week
offers. A week has as many decision points as it has kickoff waves -- distinct kickoff
days, each with its own deadline -- and in this window every week has a Monday game while
week 1 opens on Wednesday and plays again on Thursday, giving four. Picks are really made
at those waves, so captures are expected there too.

A captured decision is therefore classified by the wave whose deadline it beat, not by the
first declared event it happens to precede. Without that a Thursday decision is filed as
the Sunday one, three days early, and a Monday-game decision is filed as `after_deadline`
when it was made before its own kickoff. `after_deadline` keeps its meaning: a decision
that beat no deadline at all.

Waves the protocol does not declare are reported with `scheduled = 0` and enter neither
side of the capture rate. Describing a decision correctly is not the same as demanding it:
requiring a capture at every wave would put roughly twenty mandatory events behind a floor
of `1.0`, where a single missed Monday would fail a window that was otherwise sound. The
obligation stays at the two events that carry the news change this window is about, and
every extra decision is still held to reconstruction and parity, because those divide by
every captured decision rather than by the scheduled ones.

Capturing both is what puts the Thursday-to-Sunday news change in the record. It does not
measure the value of waiting for news. Current replay makes one decision per week and
calls `plan_slot`, never `advise_slot`, so no achieved-touchdown replay has ever exercised
the hold-or-commit branch. Measuring that requires a multi-event replay that processes
timestamped observations and hold/commit events in order, preserves each policy's
used and locked state, and never exposes later news early. It does not exist. This window
does not substitute for it, and no reading of these captures may stand in for it.

### Live participation

Real picks are submitted during the window. Submissions happen on the pool's own site, so
the link back to a decision is recorded rather than inferred: `pool recommend` prints the
decision it captured and `pool record --decision <id>` names it. With two decisions in a
week the fallback link — the most recent advice for that slot — is whichever happened
last, which is not the same thing as the one the pick came from. How each link was made
is recorded beside it. 3C changes no submitted pick and no production setting.

## Populations

Declared for the prospective log rather than inherited from Phase 3A. A bake-off has two
replayed policies; a live week has one recommendation and one submission, and calling
either of them a policy's selection would name a comparison that was never run.

`all_eligible`, `available`, `depleted`, `recommended`, `submitted`, and
`common_top1/3/5/10` by rank within slot among the eligible unused candidates at the
decision itself. `submitted` is the pick that still stands in a slot: corrections,
removals and re-entries are folded over the ordered action history first, across decision
ids, because a withdrawal recorded without naming a decision arrives linked to whichever
advice came last. Reading only the submissions would leave every player ever entered in
the population. A submission credited to a decision that never forecast that player is
reported as unmatched rather than counted, so a link that names nothing describable is
visible instead of silently absent. Positions keep WR and TE separate. Lead horizons use the frozen buckets
and are never pooled: repeated forecasts of one target week share its single outcome.

The fold reads the declared collection weeks and nothing else. An action taken in another
week belongs to another window: its decision is not in this one, so reading it here would
report a correctly linked pick as an attribution failure and would let a submission made
after this window closed move its totals. Within the declared weeks nothing is filtered,
so a submission whose decision was never captured stays visible as the failure it is.

Every recorded submission is accounted for in the audit exactly once, including the
entries later actions displaced. Whether one is *described* is read off the population
itself rather than asserted beside it, because being submitted and being describable are
different facts: eligibility reconciles the deadline and the live surface does not, so a
pick made against an earlier decision can appear on a later decision's captured surface
with its kickoff already gone. Those are reported as outside the eligible population,
with the status that excluded them, rather than counted as described.

Eligibility reconciles the deadline. The live path leaves the deadline out of
`hard_eligible` and applies it in the recommender, so a cell whose kickoff had already
passed is still marked eligible on the captured surface. Describing that as an eligible
population would mix cells the tool would let you pick with cells it would not. After
reconciliation the two exclusion classes mean what they mean in Phase 3A: a ruled-out
player is masked to zero, and an undecidable one keeps his positive rate because the
forecast was usable and only the cell was not.

## Parity And Reconstruction

Two distinct checks, because they can fail independently.

**Reconstruction** re-derives each decision's advice from its stored surface alone, under
the settings and source fingerprint it was captured with. If the re-derived advice differs
from the advice recorded beside it, something the decision depended on was never written
down, and no diagnostic over that surface describes the decision that was actually made.

**Parity** rebuilds the same instant from the archived feeds and compares it to the
captured surface on `base_rate`, `def_mult`, `vegas_mult`, `home_mult`, `avail_mult`,
`role_mult` and `lam`, keyed by week, slot and player. Every replay in this project
assumes the live path and the archive are the same function of the same inputs; until now
that was checked only inside one process on a staged database, never against a decision
that had actually been captured.

The rebuild runs under the constants the decision recorded, not the current ones. The
projection multipliers are read when they are called, so replaying under today's would
report a configuration change as a live/archive divergence -- and would report it while
reconstruction still passed, because the stored surface already has the old multiplier in
it. A structured setting the log cannot faithfully restore leaves the decision
unverifiable, which counts against the floor rather than being excused by it.

`hard_eligible` is deliberately not compared. The live path omits the deadline from it and
the replay folds it in, so the two differ by construction and comparing them would report
the contract as a defect. The deadline is reconciled instead: the captured
`decision_status` must reproduce exactly the cells the replay blocked.

The same instant must also name the same feed observations, or the two sides agreed about
inputs neither of them read. The captured side of that comparison is the decision's own
append-only input record, never a fresh resolution of the archive: resolving both sides at
check time compares the archive against itself, so an observation written afterwards but
stamped before the decision would change what the replay reads while the comparison went
on reporting agreement. Observations are compared by identity rather than by content
alone, because two readings of a feed can carry identical bytes and still be two
readings.

## Floors

Fidelity and data-quality requirements, not performance targets. A missing floor blocks
execution, for the reason a missing margin blocked Phase 3B: a threshold chosen once the
numbers exist is not a threshold.

| Floor | Value | What it requires |
|---|---|---|
| `min_event_capture_rate` | 1.0 | Every scheduled decision event was captured. A missed event is evidence that does not exist and cannot be recreated. |
| `min_reconstruction_rate` | 1.0 | Every captured decision re-derives its recorded advice. Anything less is a capture defect, not a tolerance. |
| `min_parity_rate` | 1.0 | Every captured decision matches a snapshot replay of its own instant. |
| `min_outcome_coverage_current` | 0.98 | Resolved outcomes on the hard-eligible current-week surface. Matches the Phase 3B floor so the two read on one scale. |
| `min_outcome_coverage_future` | 0.95 | Resolved outcomes on the future surface whose target week is inside the window. |

Both fidelity rates divide by every captured decision, not by the ones that could be
checked. The archive is written by `refresh`, so a decision whose feeds were never
snapshotted cannot be verified — and an unverifiable decision is not a verified one.
Putting it in the denominator is what keeps the floor from passing by shrinking, and it
puts the obligation where it belongs: refresh before deciding.

Outcome coverage is declared over the target weeks this window can settle. A week-1
decision forecasts through week 18, and at a week-6 review those later weeks have no
outcomes and cannot have any. Counting them as missing coverage would measure the
calendar; reading absence there as zero would be worse. They are excluded from the floor
and reported separately as unsettled. Inside a completely scored week, an absent
touchdown credit is a zero — the rule the capture already states.

## What The Review May Conclude

Baseline readiness: whether the capture was faithful, whether it was complete, whether
live and snapshot agreed, and what the descriptive diagnostics look like on forward data.
The declared scope is carried in the protocol as data rather than prose, so it cannot
widen to fit an interesting number.

Out of scope for this window, and refused by the protocol: any candidate, any promotion,
any achieved-score or waiting-value claim, any production change. Meeting every floor
establishes that the evidence is fit to review. It does not establish that the model is
good, that any policy improved, or that waiting for news is worth anything. Failing a
floor is a valid and informative outcome; keeping the model unchanged is a valid completed
outcome of Phase 3C.

## Freeze And Limits

The 2026 season had not started when this was written and no decision had been captured,
so the window is genuinely prospective. That is the one methodological advantage this
phase has over 3B, whose seasons had all been explored before its margins were signed, and
it survives only if the floors are not relaxed to make the collected window pass. They
will not be. Any later methodological amendment must be explicitly dated and treated as a
new window, not attached to this one after inspection. The event-classification amendment
recorded above was made the same day, before any decision existed to inspect; it is the
last change this document takes without starting a new window.

Known limits at the first sign-off. The operational database had never captured a
decision: the capture tables are created by the next migration on open, `my_picks` was
empty, and the only 2026 observations were the schedule, rosters and depth charts imported
2026-09-05, with no injury report. A refresh is a prerequisite for a meaningful week-1
decision and for any parity check at all, and its freshness warnings belong in the record
rather than dismissed. That refresh was run on 2026-09-07 at 16:27 UTC: injuries entered
the 2026 archive for the first time, and `player_stats` and `touchdowns` remain unpublished
because no 2026 game has been played, which both the live path and the replay see as
absent. Nothing has been captured yet. Six weeks of one season is a small sample under one set of shipped constants;
if a constant moves mid-window the decisions are two different functions of the same name,
and the export reports that drift rather than pooling across it.
