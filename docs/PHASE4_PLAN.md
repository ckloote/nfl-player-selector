# Phase 4 Plan: Opponents, Standings, And Winning

**Drafted:** 2026-09-07 (UTC), before any of it exists.

**Stage 1 implemented:** 2026-09-11, following the refined
[stage 1 plan](PHASE4_STAGE1_PLAN.md): three tables, raw bytes committed before parsing,
and separate parse metadata. Stages 2 and 3 remain planned.

**Corrected:** 2026-09-13. Reports do not wait for a week to resolve. The person running
the pool sends them by hand: sometimes before a week's games have finished, and always
before the next week's first lock. The 2026 week 1 report arrived before that Sunday's
games. The passages below that assumed after-the-fact delivery are rewritten in place,
because stages 2 and 3 will be built from this text, and each now says what the real
cadence changes.

Phase 3 ended without a promotion and without a collection window: 3B found no candidate
worth shipping, and [3C was closed](PHASE3C_OUTCOME.md) rather than freeze the source tree
through the season this tool is meant to be used in. What remains is the work the pool
itself asks for, in the order its dependencies force:

1. **Ingest the rest of the pool's picks** — the entrants, their submissions, their
   remaining players.
2. **Standings** — score every entrant on the same terms `my_picks` is scored on.
3. **Win probability** — decide by the chance of finishing first rather than by expected
   touchdowns.

The first two are ordinary features. The third is a change to what the tool is *for*, and
most of this document is about why that one cannot be judged the way the first two can.

## The Constraint That Shapes The Order

You cannot fit or test an opponent model without opponent data, and you cannot compute a
win probability without an opponent model. Stage 3 is therefore gated on stage 1 by
substance, not by scheduling.

The person running the pool sends every entrant's picks by hand, so history is delivered
rather than queried. **Keep every weekly report from the first one**: a report that
arrives and is discarded is a week of evidence gone, and that is the only deadline in this
plan.

The delivery also fixes what stage 3 is allowed to know. A week's report can arrive while
that week is still being played, but it always arrives before the next week's first lock.
So at any decision for week N, every entrant's picks and remaining pool through week N-1
are guaranteed to be in hand, and their week-N picks sometimes are too. Where you stand is
reliably known. The current week's opposition is sometimes observable and never
guaranteed, so it has to be predicted whenever the report has not arrived — and an early
report can include picks whose games have not locked, which their owners can still change.

## Stage 1 — Opponent Ingestion

**What the pool provides:** every entrant's picks for a week, sent by hand at some point
before the next week's first lock. Per entrant, per week, complete — which is what the design below assumed, so the
shape stands. The remaining unknown is only the delivery format, and that decides the
`fetch_` and `transform_` pair, not the schema or the staging; the schema can be built
before the first report arrives.

**Storage.** New tables `pool_entrants` and `pool_picks`, additive, mirroring `my_picks`
in shape so the scoring seam is shared. Raw observations go into the **existing**
`input_observations` and `input_payloads` under a new feed name — append-only, content
addressed, provenance for free.

Deliberately **not** registered in `snapshots.TABLES` or `freshness.FEEDS`.
`capture.observed_inputs` iterates `TABLES` explicitly (`capture.py:121`) and
`snapshots.restore` does the same, so an unregistered feed is invisible to the decision
record and to replay — which is right, because opponent picks are not a projection input
and have no business being restored into a projection rebuild. It also keeps
`snapshots.py` and `freshness.py` untouched, so this work does not move the decision
fingerprint through them.

**It will move it once, through `db.py`.** The migration list lives in a module the
enforced closure covers, so the schema change re-fingerprints the decision path exactly
once. Nothing depends on that today — 3C is closed and no protocol is running — but it
argues for doing the schema work in one pass, early, rather than dribbling migrations out
across the season.

**Shape.** Follow `ingest.py`: a `fetch_` that returns raw rows, a `transform_` that
normalises them, and a write that archives the observation in the same transaction. The
CLI gains an import command and `pool standings` to show what was ingested.

**Done when** every entrant's picks for a week can be imported, re-imported without
duplication, and read back with the observation they came from.

## Stage 2 — Standings

Mechanical once stage 1 lands, but it inherits a discipline the repo already has and must
not lose:

- **Pending is not zero.** A complete game with no credits scores 0; an incomplete or
  unresolved game stays pending. Weekly and season subtotals are labelled incomplete while
  any pick is pending. `scoring.pick_results` and `pool_history` already do this for
  `my_picks`; entrants go through the same function or the leaderboard will quietly
  disagree with `pool picks`.
- **Each entrant has their own used pool.** One player per entrant per season. This is not
  presentation — stage 3 needs to know what each opponent can still pick, and it is the
  only place that state will exist.
- **Everyone is compared as of the last week whose games are all final.** Not the last
  week imported: a report can arrive while its week is still being played, and `my_picks`
  may hold the current week before any report does. Ranking a half-played week would
  invent a lead or a deficit, so the standings are stated as of the last resolved week,
  and anything beyond it — mine or theirs — is shown separately as in progress.
- **Ties are real.** The pool has a tie rule; the leaderboard must implement whatever it
  is rather than sorting and hoping.

**Done when** `pool leaderboard` reproduces the pool's own standings for a scored week. If
it disagrees with the official table, the ingestion or the scoring is wrong and that is
worth knowing before anything is built on top.

## Stage 3 — Win Probability

### What actually changes

Today's objective is expected season touchdowns. `optimizer.build_matrix` scores every
(player, week) cell and `solve` runs `scipy.optimize.linear_sum_assignment`, maximising a
**separable sum** — that is what makes it exact and fast.

Win probability is not a sum of cell values. It depends on the joint distribution of a
whole plan, on what opponents hold, and on correlations the matrix has no way to express:
two of my picks in the same game move together, and a player picked by both me and a rival
cancels out between us entirely. **The assignment solver cannot be given a new objective.**
It stays as the expected-TD planner and becomes the starting point for something else.

### The four pieces

**A simulator.** Sample player-week touchdown outcomes jointly. Two requirements the
repo already states: one sampled outcome per player-week is shared by every entrant who
picked that player, and within-game correlation is represented rather than assumed away.
Independently useful — it turns every projection into a distribution instead of a point.

**An opponent model.** What will each rival pick in weeks not yet played? This is still
required, not a convenience: a week's report sometimes arrives before you choose, but
nothing guarantees it, so the policy has to work without one. When the report *has*
arrived, use the real picks rather than predictions of them. Each of my slots locks
separately, so an early report can land before my later slots lock, and that is when
seeing them can change a decision. A reported pick whose game has not locked is likely,
not final. The defensible baseline for prediction is greedy-from-remaining-pool on
projected touchdowns, which is also what this tool's own `greedy` policy does, run against
the remaining pool stage 2 already tracks.

The reporting cadence also hands it a clean validation loop, which is the one place this
stage gets to be empirical: predict every entrant's week-N pick, read the report, and score
the prediction. The prediction must be recorded before that report is imported, or it is
scored against an answer it could have seen. Because reports can arrive early, that is an
ordering rule rather than a calendar one, and the archive already makes it checkable:
every imported report records when it was observed. That accrues an observation per
entrant per week rather than one per season, and it is checkable from the first report
onward, long before any policy is built on top of it.

**A policy.** The search space is far too large to enumerate, so the tractable form is a
one-step lookahead: hold the expected-TD plan for the rest of the season, vary only this
week's pick, and score each candidate by simulated P(finish first). That is a real policy
and it should be described as exactly that, not as "the optimal win-probability plan".

**A seam.** `backtest.STRATEGIES` already maps a name to a policy function, which is where
a third policy belongs — but its signature is `(proj, slot, week, used, ...)` and carries
no standings or opponent state. Widening it is part of this stage, and it touches
`backtest.py` and `recommend.py`; the latter is inside the enforced fingerprint, which is
correct, because this genuinely changes what a decision is.

### The part that cannot be finessed

**A season is one Bernoulli trial.** You either win the pool or you do not. No season, and
no five seasons, can establish that a win-probability policy raised your chance of winning.
Any claim of the form "this worked" is unavailable here, and the plan should stop
pretending otherwise before it starts.

Historical replay does not rescue it either: seasons before ingestion have no opponent
picks, so a backtest would be measuring the policy against invented rivals — a test of the
machinery, not of the idea.

What *can* be established, and what stage 3 should be judged on:

| Claim | How it is checked |
|---|---|
| The simulator is calibrated | Simulated player-week touchdown frequencies against 15 seasons of observed ones, including the conditional tails, not just the mean |
| The opponent model predicts picks | Hit rate against the picks entrants actually made, once stage 1 has weeks of them |
| The policy behaves as intended | In simulation: takes variance when behind, sheds it when ahead, and converges on the expected-TD pick when the standings are level |
| The advantage exists in the model | P(win) under win-max against P(win) under TD-max, **inside the simulator** |

The last row is a simulated gain and must be reported as one. The repo's own earlier note
put it correctly: validate conditional tails and simulation behaviour before treating a
simulated win-probability gain as an established advantage.

### How it ships

**Alongside, not instead of.** `pool recommend` keeps giving the expected-TD advice and
gains the win-probability view next to it, so the two are visible together and their
disagreement is the interesting output. Silently swapping the objective would replace a
decision rule that is understood with one that cannot be validated, and hide the swap.

This also matches the season's shape: early on, with everyone level, the two objectives
mostly agree. They diverge late, when chasing or protecting a lead is what actually
decides the pool — which is exactly when a visible second opinion is worth having.

## Out Of Scope

- **The multi-event replay.** Measuring the value of waiting for news needs a replay that
  processes timestamped observations and hold/commit events in order, preserving each
  policy's used and locked state and never exposing later news early. It still does not
  exist, and nothing here substitutes for it.
- **Any calibration change.** Phase 3B's no-promotion outcome stands; the shipped model
  and production constants are unchanged by all of the above.
- **Reopening 3C.** If capture fidelity is ever worth certifying, one captured decision
  verified the same day answers it. That does not need a window and is not part of this.

## Sequencing

Stages 1 and 2 are ordinary feature work on `main` with no ceremony. Stage 3's first two
pieces — the simulator and its calibration — depend on nothing but history and can begin
whenever; the policy waits on real opponent data, and how long that takes is set by
whatever the answer to stage 1's open question turns out to be.
