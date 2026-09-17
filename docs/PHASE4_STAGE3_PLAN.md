# Phase 4, Stage 3: Win Probability

**Drafted:** 2026-09-16 (UTC), before any of it exists. Implements stage 3 of
[the Phase 4 plan](PHASE4_PLAN.md), which stages 1 and 2 have now cleared by substance:
every entrant's picks are archived and attributable, every entrant is scored by the
function that scores mine, and each rival's used pool is tracked independently.

Stages 1 and 2 were ordinary feature work. This one is not. It changes what the tool is
*for* — from expected season touchdowns to expected share of the pot — and the phase plan
is explicit that the change cannot be judged the way the first two were. Most of what
follows is about keeping that distinction visible in the code rather than only in prose.

## Where The Season Actually Stands

Checked against `data/pool.db` before drafting, so the plan is written against the season
as it is rather than as it was imagined:

| | |
|---|---|
| Schema | `user_version = 4`. No migration since stage 1. |
| Season | 2026, 272 regular-season games over 18 weeks. |
| Scored | Week 1 only. `complete_weeks` is `[1]`; `resolved_through` is `1`. |
| Imported | One report, week 1, fifteen picks, all resolved, no blanks. |
| Entrants | Five: adam, brad, jamie, keith, and cj with `is_me = 1`. |
| Next lock | Week 2 kicks off **2026-09-17T20:15 ET** — tomorrow. |

So there are 17 weeks left and four rivals to predict. That number is the entire empirical
budget this stage will ever have, and it only decreases.

## The Deadline This Stage Has

Stage 1 had one deadline: a report that arrives and is discarded is a week of evidence
gone. Stage 3 has the mirror of it.

The phase plan names the one place this stage gets to be empirical: **predict every
entrant's week-N pick, record the prediction, read the report, score the prediction.** It
also names the rule that makes the score mean anything — the prediction must be recorded
before it could have seen the answer.

The plan states that as an ordering rule against the report's arrival. That is necessary
and **not sufficient**. A prediction made after a week's games have been played, but
before its report lands, has still seen something a real prediction could not: the
projections it runs on are rebuilt from stats through the latest complete week, so the
predictor would be choosing rivals' week-N picks with week-N outcomes already in its
inputs. The honest deadline is therefore the harder one:

> A prediction for week N counts only if it was archived **before week N's first kickoff**
> *and* **before week N's report observation**. Both are timestamps already in
> `input_observations`, so both are checkable rather than asserted.

Predictions failing either test are kept and reported as unscorable. They are never
silently dropped and never silently counted.

**Week 2 is conceded.** Building this properly cannot happen before tomorrow evening, and
a payload shape guessed under time pressure would be the thing regretted by week 6. The
log starts at week 3, which leaves **16 weeks × 4 rivals × 3 slots = 192 prediction
observations** if nothing is missed. That is the budget. Commit 1 below is ordered first
for this reason and no other: it is the only part of this stage with a deadline, and it
is buildable without the simulator, the policy, or anything else here.

## Storage: No Migration

A prediction is an observation of what the model said at a time. `input_observations` and
`input_payloads` already are exactly that — append-only by trigger, content-addressed,
timestamped, with a `feed` column and a free-form `coverage` JSON. Stage 1 put the raw
report bytes there under a new feed name for the same reasons. This stage adds a second
feed name, `pool_prediction`, and **no table and no migration**.

That is not only cheaper, it is the design. The check this stage rests on is a comparison
of two timestamps, and putting both in the same append-only table makes the comparison
native and the ordering untamperable: neither row can be edited or deleted.

Deliberately **not** registered in `snapshots.TABLES` or `freshness.FEEDS`, exactly as
`pool_report` is not. `snapshots.TABLES` is built from `freshness.FEEDS` plus
`touchdowns` (`snapshots.py:17`), and `capture.observed_inputs` iterates it, so an
unregistered feed is invisible to the decision record, to freshness warnings, and to
replay — which is right. A prediction of what a rival will do is not a projection input
and has no business being restored into a projection rebuild.

**The whole prediction log therefore costs zero fingerprint moves.** `db.py` is untouched.

## The Closure, And The One Fingerprint Move

`benchmark.decision_modules()` computes the enforced closure from `DECISION_ROOTS =
("recommend", "projections", "snapshots")` by following imports. Stage 3 adds decision
logic, so the closure must widen — the phase plan says so, and it is correct, because this
genuinely changes what a decision is. The risk is widening it *further than that*.

The hazard is concrete. The policy needs rival standings and remaining pools. Those live
in `standings.py`, which imports `entrants.py`, which is a CSV parser. If `recommend.py`
reaches for `standings`, the enforced fingerprint acquires a report parser, and every
change to a column header would invalidate real captured decisions.

The house already has the answer and it is the shape `recommend.py` is built in:
**`advise_slot` never touches the database.** It takes `proj`, `used_ids` and `locked` as
data, prepared by the caller. Stage 3 follows it exactly. Three new modules, split by
whether they decide or merely read:

| Module | In the closure? | What it holds |
|---|---|---|
| `rivals.py` | **yes**, via `recommend` | Pure. `RivalState`, the three candidate pick predictors, the ranked-candidate form the simulator samples from. No `db`, no `sqlite3`, no I/O. |
| `simulate.py` | **yes**, via `recommend` | Pure. Joint player-week sampling, season totals, pot share. Takes arrays and frames, returns arrays. |
| `predictions.py` | **no** | Reads the database, builds `RivalState` from `standings`, archives predictions, reads them back, scores hit rates. |

`predictions.py` importing `rivals.py` does not pull `predictions` into the closure — the
closure follows imports outward from the roots, not inward. So the reader can depend on
the decision logic without contaminating it, which is the same one-way seam `standings.py`
has with `scoring.py` today.

**The move happens once.** `rivals.py` and `simulate.py` land unimported by anything in
the closure (commits 1–3), where they change nothing. Commit 4 wires `recommend.py` to
them and adds the new `config` constants, and that single commit moves the fingerprint:
`DECISION_SOURCES` gains `src/pool/rivals.py` and `src/pool/simulate.py`, and
`src/pool/recommend.py` and `src/pool/config.py` change hash.

This is the **third** move this season — stage 1 through `db.py`, stage 2 through
`scoring.py`, now this. The 2026 captures already need `--allow-code-drift`, so the
marginal cost is smaller than stage 2's was; it is still paid deliberately, under stage 2's
own procedure:

1. The move is one commit, alone, touching no other closure module.
2. **Before** that commit, `pool verify-capture --season 2026 --allow-code-drift` is run
   and its output pasted into this document, so the last verification under the stage-2
   fingerprint is on record.
3. The README says the 2026 captures have drifted again, and why.

Unlike the previous two, this move is not an accounting cost. A decision made under a
different opponent model or a different simulator *is* a different decision, so the
fingerprint covering them is the fingerprint doing its job.

## Piece 1 — The Opponent Model And Its Log

### What is predicted

For each rival, for each slot, for each week not yet reported: which player will they
pick? The phase plan names the defensible baseline — greedy-from-remaining-pool on
projected touchdowns, which is what `backtest.pick_greedy` already does, run against the
remaining pool `standings.used_pools` already tracks.

That is one hypothesis about a human being. Three are recorded, because recording the
other two costs nothing now and cannot be done retroactively:

| Predictor | Hypothesis about the rival |
|---|---|
| `greedy` | Takes the best available player this week, respecting what they have spent. |
| `optimizer` | Plans the rest of the season the way this tool does (`plan_slot`, one week's assignment). |
| `naive` | Takes the best projected player in the slot, **ignoring** their own used pool. |

`naive` is there to answer a real and otherwise unanswerable question: do these people
track the one-use constraint at all? Its hit rate against `greedy`'s is the cheapest
measurement of that fact, and the answer changes the simulator.

**The ranked top five is recorded, not just the argmax.** The simulator does not need to
know a rival's most likely pick; it needs a *distribution* over their picks, and a hit
rate on the argmax alone cannot calibrate one. Recording where the actual pick landed in
each predictor's ranking — 1st, 2nd, 5th, off the list — yields the sampling distribution
directly. This is the single highest-value thing in the archived payload and it is the
one that cannot be reconstructed later.

### The archived payload

One observation per `(season, week)`, one payload, written under feed `pool_prediction`
by an `archive_prediction` that follows `entrants.archive_report` line for line —
`observed_at` in UTC microseconds, content-addressed via `INSERT OR IGNORE` into
`input_payloads`, coverage JSON naming the week and what was predicted.

```json
{"season": 2026, "week": 3, "predicted_at": "...",
 "projection_hash": "...", "code": {"decision_hash": "..."},
 "rivals": {"adam": {"QB": {"greedy":  [{"player_id": "...", "player_name": "...", "lam": 0.61}, ...],
                            "optimizer": [...], "naive": [...],
                            "remaining": {"QB": 41, "RB": 55, "FLEX": 96}}}}}
```

`remaining` is `standings.remaining_counts`, stored because a prediction made against a
pool of 41 quarterbacks is a different prediction from one made against 8, and by week 14
nobody will remember which it was.

Re-running in the same week appends another observation rather than replacing one — the
table forbids anything else. Scoring uses the **last prediction archived strictly before
both deadlines**: the latest information state that could not have seen the answer.

### Scoring the log

`pool predict score` reports, per predictor: hit rate overall, by slot, by rival, and by
week; the rank distribution of the actual pick within each predictor's top five; and the
count of unscorable weeks with the reason each failed. No predictor is promoted by this
command — it reports, and the choice of which predictor the simulator uses is a decision
made in this document and revised in writing, with the revision dated.

### An unresolved name is not a miss

Stage 2 established that a name the report could not resolve spends *somebody we cannot
name*. The same rule applies here: a rival-week whose actual pick has `player_id` NULL is
**unscorable**, not a miss. Scoring it as a miss would make an import defect look like a
model defect, and the rate of the two is not the same number.

## Piece 2 — The Simulator

### What it must do

Three requirements the repo has already committed to, none negotiable:

1. **One sampled outcome per player-week, shared by every entrant who picked that
   player.** Week 1 already produced the case: brad and keith both picked Joe Burrow.
   Sampling him twice would let one of them beat the other on a coin flip that does not
   exist.
2. **Within-game correlation represented rather than assumed away.** Two of my picks in
   the same game move together.
3. **Ties come out of the simulation.** Season totals are small integers, ties are common,
   and a tie for first splits the pot. `standings.Standing.share` already carries `1/k`;
   the simulator computes the same quantity forward.

### The model

**Revised 2026-09-16, before any of it was built**, against 16 seasons of complete
touchdown coverage in `data/experiments/roster-snapshot-repair/dataset.sqlite`. The first
draft proposed a two-level game x team multiplicative mixture. Measurement rejected half of
it and promoted a deferred limitation into a required component. The measurements are in
the table below; the model follows them rather than the other way round.

| Measured, 2011-2025 REG, complete games | Value | Consequence |
|---|---|---|
| Team-game pool credits, var/mean | 1.60 | Overdispersed. Plain Poisson is out. |
| Corr(home credits, away credits) within a game | **+0.110**, positive in 13/15 seasons | A shared *game* factor is real. Shootout beats game script. |
| Corr(QB, lead pass-catcher), **same team** | **+0.465** | The dominant correlation in the entire system. |
| Corr(QB, lead RB), same team | +0.013 | No general team factor. |
| Corr(lead RB, lead pass-catcher), same team | **-0.036** | Same-team non-QB pairs are *substitutes*, not complements. |
| Corr(QB, opposing pass-catcher), cross-team | +0.069 | The game-factor baseline. |
| Share of QB pool credits that are `throwing` | **90.6%** | A QB's score is almost entirely his receivers' scores. |

Two conclusions, both load-bearing.

**The shared team factor `T_t` is deleted.** It predicts that same-team pairs are more
correlated than cross-team pairs. For non-QB pairs that is false in the data: same-team
QB x RB (+0.013) is *below* cross-team QB x pass-catcher (+0.069), and same-team
RB x pass-catcher is negative. Two backs and a receiver compete for the same finite
red-zone chances. A positive `T_t` would have inflated exactly the correlations that are
zero or negative, and it would have done so invisibly.

**The QB/pass-catcher shared credit is promoted from deferred limitation to required
component.** At +0.465 it is not a refinement; at a 90.6% throwing share it *is* how
quarterback scoring works. The first draft deferred it on the grounds that splitting `lam`
into throwing and scoring parts meant changing `projections.py`, which the phase plan puts
out of scope. That reasoning was wrong: the split is a property of **history**, readable
from `touchdown_credits.kind`, which already labels every credit `throwing` or `scoring`.
It is estimated inside the simulator and `projections.py` is never touched.

So, for a game `g`, team `t`, week `w`:

```
G_g ~ Gamma(mean 1, var 1/k_game)                   one draw per game, both teams

pass-catchers  j on t:   R_j ~ Poisson(lam_j * G_g)          receiving/scoring credits
QB rushing:              Q_run ~ Poisson((1 - a) * lam_QB * G_g)
QB throwing:             Q_throw = SUM_j R_j  +  U_t
      where U_t ~ Poisson(max(0, a * lam_QB - SUM_j lam_j) * G_g)
Y_QB = Q_run + Q_throw
```

`a = 0.906` is the measured throwing share. `U_t` is the residual: touchdowns thrown to
receivers the candidate pool does not model — backup tight ends, third-down backs — and it
is exactly the term that keeps the quarterback's expectation right:

```
E[Y_QB] = (1-a)*lam_QB + SUM_j lam_j + (a*lam_QB - SUM_j lam_j) = lam_QB
```

**`E[Y_i] = lam_i` for every player, exactly.** `lam_i` remains `projections.lam` as
shipped — `base_rate * def_mult * vegas_mult * home_mult * avail_mult * role_mult`.
Nothing about the projection model changes; Phase 3B's no-promotion outcome stands. The
simulator adds a joint distribution around the shipped point forecasts and is not
permitted to move any of them.

One parameter is fitted, `k_game`, from the cross-team covariance:

| Quantity | Model implies |
|---|---|
| `Cov(Y_i, Y_j)`, same game, opposing teams | `lam_i * lam_j / k_game` |
| `Var(Y_i)`, non-QB | `lam_i + lam_i^2 / k_game` |
| `Cov(Y_QB, R_j)`, same team | `lam_j * (1 + lam_QB/k_game)` — dominated by the shared credit, not by `k_game` |

Fit `k_game` on the cross-team covariance, then check the implied marginal variance and the
implied same-team QB correlation against the observed +0.465. Those two are **not** fitted;
they are the specification checks, and the calibration report prints them whether they pass
or fail. `a` is measured, not fitted, and is reported per position group.

`k_game` and `a` are frozen into `config.py` once, with the run that produced them named in
a comment. They are not re-fit per invocation: a decision whose sampling distribution moves
with the data underneath it cannot be reconstructed, and `capture.reconstruct` would have
nothing to restore.

**Implementation constraint.** `SUM_j lam_j` over modelled pass-catchers can exceed
`a * lam_QB` for a team whose receivers are collectively projected above their quarterback.
The residual clamps at zero, which breaks the expectation identity for that team. The
simulator must detect it, scale the receiving terms to restore the identity, and count the
occurrences; the calibration report states how often it fired. Silently clamping would
under-credit exactly the highest-scoring offences.

### The one that is still named, not fixed

Same-team **substitution** is not modelled. `Corr(lead RB, lead pass-catcher) = -0.036`
says a team's red-zone chances are finite and its skill players compete for them; the model
above gives those pairs correlation zero through the shared `G_g`, which is slightly too
high. The effect is an order of magnitude smaller than the QB stack it would sit beside, it
is measured, and its direction is known: the tool will very slightly overrate holding a
running back and a receiver from the same team. Deferred with a number attached rather than
with an adjective.

**The week 1 illustration, corrected.** My own picks were Justin Herbert at QB and Quentin
Johnston at FLEX — same team, same game, `2026_01_ARI_LAC`. Herbert threw one touchdown and
Johnston caught none, so it paid one slot. Under the revised model that pairing is
correlated at roughly the measured +0.465 rather than through a team factor that does not
exist, which is the difference between a policy that understands a stack and one that does
not.

### What is checked, and what that check is worth

The shipped `lam` carries whatever calibration bias `docs/EVALUATION.md` records; the
simulator inherits it and does not correct it. Two things follow and both belong in the
report. A shared multiplicative bias largely cancels when candidates in the same slot are
*ranked* against each other, which is what the policy does. It does not cancel in the
absolute pot share, which is therefore a number to compare across candidates and not a
number to believe.

## Piece 3 — The Policy

### The shape, and why it is this shape

The assignment solver maximises a separable sum. Pot share is not one. The solver is not
given a new objective; it stays the expected-TD planner and becomes the starting point for
a **one-step lookahead**, exactly as the phase plan specifies: hold the expected-TD plan
for the rest of the season, vary only this week's pick, and score each candidate by its
simulated expected share of the pot.

That is a real policy and the code and the CLI both call it one. It is not "the optimal
win-probability plan" and nothing in this stage may describe it that way.

### Common random numbers, which is what makes it tractable

The naive form re-simulates the season per candidate: 80 candidates per slot, three slots,
thousands of draws each. The tractable form samples the outcome table **once** and
evaluates every candidate against the same draws.

```
Y[sim, week, player]     sampled once, for every player anyone could still pick
```

Then, per simulation draw:

- each rival's season total is a fixed vector — their future picks depend on their
  remaining pool and the projections, not on outcomes, so they are computed once;
- my total is `locked total + Y[sim, week, candidate] + my re-solved future plan's total`.

`recommend.advise_slot` already re-solves the rest of the season once per playable
candidate through `forced_total`, so the future-plan part introduces no new cost class —
it reads the plan that re-solve already produced. Per candidate the remaining work is one
vector add and one comparison against the rivals' running maximum: a few thousand-element
numpy operations, not a loop.

The pairing matters for more than speed. Every candidate is compared on identical outcome
draws, so the *difference* between two candidates has far lower variance than either
candidate's absolute share. That is the quantity the decision actually depends on.

### What actually moves the answer

Worth writing down before the code is built, because the DESIGN §3.4 sketch promises a
lever this model does not have.

`Var(Y_i) = lam_i + lam_i^2 * (c - 1)` is **strictly increasing in `lam_i`**. Two candidates
with the same projected rate therefore have the *same* marginal distribution: there is no
"higher-variance player at equal expected touchdowns" to reach for. Per-player variance is
not an independent dial, and any description of the policy that says "takes variance when
behind" by choosing a spikier player is describing something that cannot happen here.

The levers that do exist are all **correlation**, and they are real:

| Lever | Available when | What it does |
|---|---|---|
| Differentiate | I am behind | Picking a player my target did *not* pick makes our scores independent. Mirroring them freezes the gap, and a frozen gap is a loss. |
| Mirror | I am ahead, against essentially one threat | Sharing their player matches their gains and runs out the clock. |
| Stack my own slots | More than one of my slots is still open | Two of my picks on the same team are positively correlated, which raises `Var(my total)` at unchanged `E[my total]`. This is the one genuine variance play, and it is only available before my other slots lock. |

Measured in the model, two-horse race, one contested slot, `lam` equal on both sides:

| Weeks left | My lead | Differentiate minus mirror, in pot share |
|---|---|---|
| 1 | −4 | **+0.036** |
| 1 | −2 | **+0.076** |
| 1 | level | +0.000 |
| 1 | +2 | **−0.075** |
| 2 | −2 | +0.037 |
| 4 | −2 | +0.014 |
| 8 | −2 | +0.006 |

Three things follow, and the CLI should reflect all three. The sign flips exactly at level
**when the rival's pick is independent of mine**, which is the sanity check. It is not the
general case: if every rival is about to take the player I would take, mirroring them buys a
guaranteed k-way split and differentiating buys a chance at the whole pot, so differentiation
wins even from level. `docs/DESIGN.md` says the two objectives agree early, when everyone is
level. That is true only when my candidates are equally unrelated to what rivals hold, and
the implementation measured it at 0.25 against 0.44 in the case where they are not. The magnitude decays fast in weeks remaining — with 17 weeks left
one contested slot is swamped by everything else, so **early in the season the two objectives
genuinely agree and the tool should say so rather than manufacture a difference**. And
mirroring only pays against a *single* live threat: with four rivals still in range,
correlating with one of them leaves the other three untouched, and differentiating won even
from in front.

### Rival noise, and where its value comes from

Deterministic rivals would make the policy overconfident about blocking and
differentiation: it would know exactly which player the leader is about to take. The
DESIGN §3.4 sketch already says "with noise".

The noise is the rank distribution Piece 1 measures. With probability derived from the
observed frequency, a rival takes their predictor's 2nd, 3rd, ... choice instead of the
1st, reusing the `config.RANDOM_TOP_N` shape `backtest.pick_random` established.

**Until the log has weeks in it, that distribution is a declared guess, and the plan says
so out loud.** An initial value is written into `config.py` with a comment naming it as
provisional, and it is revised exactly once, in writing, when the log has enough weeks —
with the revision dated in this document and the fingerprint move it causes acknowledged.
This is the loop closing: the thing Piece 1 measures is the thing Piece 3 consumes.

### Determinism

The seed is a config constant. Identical inputs must give identical advice, or
`capture.reconstruct` cannot rebuild a decision and the capture machinery this repo spent
Phase 3 on stops meaning anything for the new view.

### Refusing to rank on noise

Two candidates whose pot shares differ by less than the paired Monte Carlo standard error
are **not ordered**. The CLI prints them as tied within simulation noise. Without this the
tool will confidently reorder picks on nothing, every week, and the number of simulations
will silently become a decision input.

## Piece 4 — The Seam

`backtest.STRATEGIES` maps a name to a policy function. Its choosers already absorb
unknown keywords (`def pick_greedy(proj, slot, week, used, **_)`), so widening the
signature is free at the callee. The work is in `replay`, which must thread a rival-state
object through the week loop and keep it updated as simulated picks are spent.

A `winprob` entry is added, and it is labelled for what it is. A historical replay has no
opponent picks, so it would be measuring the policy against invented rivals: **a test of
the machinery, not of the idea.** The CLI says so wherever it prints the row, and the
phase plan's own sentence to that effect is quoted there.

The check that does mean something is not a season replay at all. It is three scripted
scenarios run inside the simulator, with the standings set by hand:

| Scenario | The policy must |
|---|---|
| Behind by 4, 2 weeks left, one leader | differentiate — refuse the leader's likely pick at equal `lam` |
| Ahead by 4, 2 weeks left, one live chaser | mirror — prefer the chaser's likely pick at equal `lam` |
| Ahead, but three rivals still in range | *not* mirror; the single-threat condition fails |
| Level, and no rival would hold either candidate | converge on the expected-TD pick |
| Level, but every rival is about to take the same player I would | **differentiate anyway** |

Fixed seed, deterministic, and they fail loudly if the policy stops behaving as described.

## How It Ships

**Alongside, not instead of.** `pool recommend` keeps giving the expected-TD advice and
gains the win-probability view beside it. Silently swapping the objective would replace a
decision rule that is understood with one that cannot be validated, and hide the swap.

```
Alternative          Matchup      xTD   Season cost   Pot share   vs EV   Deadline   Note
```

A difference inside the noise band prints in `vs EV` as `tied` rather than as a signed
number. Printing ±0.1% there invites reading an ordering into it, and the paired draws
exist precisely to know when there is not one.

Below the table, when the two disagree, one sentence naming the reason in the terms the
policy actually used. Those terms are **overlap first, standing second**: differentiating
from a player rivals are likely to hold, mirroring one they are likely to hold, or — when
neither candidate is on anybody's list — the standing alone. That ordering corrects this
section, which said "behind and buying variance, ahead and buying floor" and had the
mechanism backwards. Being behind does not by itself prefer one candidate over another;
what prefers one is whose season it is correlated with. The sentence names rivals out of
the same first-week sampling sets the rollouts drew from — `rivals.options` is shared with
`rivals.rollout` for exactly this reason — so the explanation cannot drift away from the
simulation it explains. A divergence with no statable reason is a bug report, not advice,
and prints as one.

Degenerate cases get a sentence rather than a column of zeroes:

- **No reports imported.** No rivals, no view. Points at `pool report import`.
- **Nothing separates.** When every alternative sits inside the noise band of the
  expected-TD pick, say so with the band and the draw count, rather than printing a
  spuriously precise share. This replaces the "everyone level, early season" case, whose
  premise the suite disproved: levelness is not what makes the objectives agree. It is the
  measured form of the same sentence, and it can be wrong in a way the asserted one could
  not.
- **A week that was played and never reported.** Three more players are spent per such
  week and none of them can be named, so every rival's remaining pool is overstated and so
  is what the simulation lets them score. The same defect as an unresolved name, arriving
  through a different door: `RivalState` carries `missing_weeks` beside `unknown`, and
  `pool_complete` fails on either. **Played** is load-bearing. Advice for week N counts
  spending through week N-1, so asking one week ahead of the schedule brings in a week that
  has not kicked off, in which nobody has picked and no report can exist. Counting that as
  missing evidence invents three spent players per rival out of a week that has not
  happened, which is what the first version of this did. Kickoff is the line here for the
  same reason it is the line for a prediction: before it, absence says nothing.
- **Season decided.** Share is 0 or 1 and nothing is left to optimise.

`pool predict record` and `pool predict score` are the new commands, shaped as a Typer
sub-app like `pool report import` / `pool report list`. `pool recommend` prints a nudge
when the current week has no prediction on record and its first kickoff has not passed —
the deadline, as a reminder, at the only moment it can still be met.

`capture.record_decision` records the new view: each candidate's share, its standard
error, the simulation count, the seed, the rival-state hash and which predictor produced
the rival picks. A capture that stored only the EV advice could not afterwards say what
the other half of the screen said.

## The Part That Cannot Be Finessed

Restated here because it is the reason this document exists and it must not be softened in
the implementation.

**A season is one Bernoulli trial.** You win the pool, share it, or do not. No season, and
no five seasons, can establish that a win-probability policy raised your chance of winning.
Historical replay does not rescue it: seasons before ingestion have no opponent picks.

What can be established, and what this stage is judged on:

| Claim | How it is checked |
|---|---|
| The simulator is calibrated | Simulated player-week TD frequencies against 2011–2025 observed ones, including `P(Y≥1)`, `P(Y≥2)`, `P(Y≥3)` by lambda bin — the conditional tails, not just the mean |
| The correlation is represented | Observed against simulated covariance for same-team and same-game-opposing pairs, with QB/receiver pairs reported separately |
| The opponent model predicts picks | Hit rate and rank distribution against the picks entrants actually made, from week 3 onward |
| The joint model holds on the picks people actually make | **Prospective PIT.** Each week, before kickoff, the simulator implies a distribution for every entrant's weekly total. After the week, record where the realised total fell in it. Five entrants x 17 weeks = 85 draws; a uniform rank histogram is the pass. This tests `lam`, `G_g` and the quarterback mechanism together, on the only population that matters — the players these five people actually pick — and it costs one archived array per week |
| The policy is not an artefact of a fitted constant | **Sensitivity.** Re-rank this week's candidates across the plausible range of `k_game` and the rival-noise parameter. A pick that is stable across that range does not depend on getting them right; one that flips is reported as undetermined rather than advised |
| The policy behaves as intended | The three scripted scenarios above, in simulation |
| The advantage exists in the model | Expected pot share under win-max against TD-max, **inside the simulator** |

The last row is **simulated and self-referential**: the simulator both generates the
outcomes and scores the policy, so the number is what the advantage would be if the model
were right. It is an upper bound under a correct model, it is reported with that sentence
attached, and it is never quoted without it.

## Tests

New `tests/test_rivals.py`, `tests/test_simulate.py`, `tests/test_predictions.py`, and
additions to `tests/test_recommend.py`. Built on the existing shapes: `conftest.proj_row`
and `make_proj` for frames, `tests/test_workflow`'s schedule and `play`/`end` builders and
`tests/test_standings`'s multi-entrant CSV for the database-facing parts.

The claims this document makes that would be silent if wrong:

**The log**

1. A prediction archived before both deadlines is scored; one archived after the report is
   kept and reported unscorable.
2. A prediction archived after the week's first kickoff is unscorable even though the
   report has not arrived — the harder deadline, as a test.
3. Re-predicting the same week appends an observation and scoring uses the last one that
   beat both deadlines.
4. **The prediction feed is invisible where it must be**: after archiving,
   `capture.observed_inputs` names no `pool_prediction` feed, `freshness.report` warns
   about nothing, and `snapshots.restore` restores no prediction rows.
5. A rival-week whose actual pick never resolved is unscorable, not a miss.
6. All three predictors and the ranked top five are archived, and the rank of the actual
   pick is recovered from the archive alone.
7. `remaining_counts` at prediction time is archived and read back with the prediction.

**The simulator**

8. One player picked by two entrants gets one sampled outcome, credited to both.
9. A quarterback and his own pass-catcher covary strongly, through shared credit rather
    than through a team factor; a quarterback and his own running back do not; two players
    in unrelated games do not.
10. A two-point conversion never enters a sampled total and a return touchdown does —
    the simulator samples what `scoring` credits, not offensive touchdowns.
11. Simulated marginal frequencies match the fitted moments within tolerance at fixed seed,
    and `E[Y_i]` returns `lam_i` for every player including quarterbacks — the identity the
    residual term exists to preserve.
11b. A team whose modelled pass-catchers out-project its quarterback triggers the rescale
    rather than a silent clamp, and the occurrence is counted.
12. A k-way tie for first yields each tied entrant exactly `1/k`, and the shares over all
    entrants sum to 1 in every draw.
13. The same seed and the same inputs give the same answer, twice.

**The policy**

14. Level standings, with my candidates disjoint from anything a rival would hold: the
    two objectives agree, and no divergence is reported.
14b. Level standings where every rival converges on the player I would take: the policy
    **differentiates anyway**. Mirroring buys a guaranteed k-way split; differentiating
    buys a chance at the whole pot. Being level does not by itself make them agree.
15. Behind late against one leader: at equal `lam`, the policy prefers the candidate the
    leader is *unlikely* to take, and the EV ranking is indifferent between them.
16. Ahead late against one chaser: at equal `lam`, the preference reverses. With three
    rivals still in range it does not reverse — mirroring needs a single threat.
16b. With both a QB and a FLEX slot open and me behind, the policy prefers the same-team
    pair over two unrelated players of equal total `lam`; ahead, it prefers them split.
17. Differentiation is real — with the leader highly likely to take a player, my
    identical pick scores lower pot share than an equivalent-EV alternative.
18. Two candidates within the paired Monte Carlo standard error are reported as tied
    rather than ordered.
19. My re-solved future plan excludes the candidate this week's pick spends.
20. A rival's reported pick, where the report has arrived, is used instead of a prediction
    of it.
21. With no reports imported, `recommend` gives the EV advice and says why there is no
    second view, rather than failing.

**The seams**

22. `recommend.advise_slot` with no rival state returns exactly today's advice — same
    pick, same alternatives, same hold flag, byte for byte.
23. `rivals.py` and `simulate.py` import no `db`, no `sqlite3` and nothing that reads the
    filesystem — the property that keeps the closure from acquiring a CSV parser.
24. `benchmark.decision_modules()` equals `DECISION_SOURCES` with `rivals.py` and
    `simulate.py` in it and `predictions.py`, `standings.py` and `entrants.py` out of it.
25. `backtest.STRATEGIES["winprob"]` replays a season, and its output is labelled as run
    against invented rivals.

**The shipping surface**

26. A decision captured with rival state reconstructs from its own record, shares and all;
    the same decision captured without it reports itself as differing from itself. Both
    directions, because the first alone would pass on a capture that stored nothing.
27. The rivals named in a divergence sentence come from the same first-week sampling sets
    the rollouts drew from, not from a second computation that agrees with them today.
28. A rival whose week's report never arrived is not a complete pool, and `recommend` says
    so before printing shares computed against an opposition with too many players left.
29. A divergence the policy cannot attribute to overlap or to standing prints as a defect
    rather than as advice.

Claim 22 is the acceptance test for "alongside, not instead of". It is what fails if the
win-probability work ever starts quietly changing the expected-TD advice.
Claim 26 is the acceptance test for the capture: a second objective that cannot be
reconstructed is a second objective that was never really recorded.
Claim 24 is the acceptance test for the closure argument, and claim 4 for the storage one.

`tests/test_phase3a.py`'s `outside` set gains `predictions`. `rivals` and `simulate` are
not added to it — they are not readers, CLI or research harness; they are decision logic
that is simply not wired yet, and in commit 4 they join `DECISION_SOURCES` instead.

## Order Of Work

Eight commits, in this order, each green on its own. Commits 1–3 are independently
valuable: if the policy is never built, the prediction log and a calibrated simulator are
still worth having, and the log is the only part that expires.

0. **This document.**
1. **The prediction log.** `rivals.py` (the three predictors, pure), `predictions.py`
   (state, archive, readback, hit rates), `pool predict record` / `pool predict score`,
   `tests/test_rivals.py` and `tests/test_predictions.py`, and the `test_phase3a.py`
   `outside` edit. No fingerprint move. **This one is time-critical** — every week it
   slips is an observation that cannot be recovered.
2. **The simulator.** `simulate.py` and `tests/test_simulate.py`. Pure, unimported by the
   closure, constants still hard-coded in the test.
3. **The calibration.** `experiments/phase4-simulator.toml` and its runner, results under
   `experiments/results/phase4-simulator`, and a written report covering the tails, the
   covariance decomposition, the over-identification check and the named QB/receiver gap.
   The fitted `k_game` and the measured throwing share `a` are chosen here, with the
   two specification checks printed pass or fail.
4. **The policy, and the one fingerprint move.** The fitted constants into `config.py`,
   `recommend.py` importing `rivals` and `simulate` and gaining the win-probability view,
   `DECISION_SOURCES` gaining both modules. One commit, alone, with the pre-move
   `verify-capture` output recorded in this document first.
5. **The CLI.** The pot-share column, the divergence sentence, the noise guard, the
   degenerate-case sentences, the missing-prediction nudge, and `capture.record_decision`
   storing the new view.
6. **The behaviour scenarios and the prospective checks.** The scripted standings
   positions as tests; the weekly PIT array archived alongside the prediction log under the
   same `pool_prediction` feed, since it is the same kind of claim — what the model said
   before it could see the answer; and the `k_game` / rival-noise sensitivity sweep behind
   `pool recommend --sensitivity`.
7. **The seam.** `backtest.STRATEGIES["winprob"]` and rival state threaded through
   `replay`, labelled where it prints.
8. **Docs.** `README.md` status and the drifted 2026 captures; `docs/DESIGN.md` §3.4,
   which describes this module as later work, and §3.5 for the new commands;
   `docs/PHASE4_PLAN.md`, whose stage 3 section is marked implemented;
   `docs/IMPLEMENTATION_PLAN.md` stage status.

## Verification Record, Before The Fingerprint Move

Run on 2026-09-16, immediately before the commit that makes `recommend.py` import `rivals`
and `simulate`, so the last verification under the stage-2 fingerprint is on the record.
This is stage 2's procedure, followed for the third move of the season.

```
$ pool verify-capture --season 2026 --allow-code-drift

decision      week  surface           reconstructed  replay  note
ea7b0ff08ff6  1     thursday_deadline yes            yes     fingerprint moved; accepted by override
6e71a2be2ae2  1     sunday_slate      yes            yes     fingerprint moved; accepted by override
55134df98e15  1     sunday_slate      yes            yes     fingerprint moved; accepted by override
155e11a74ea3  1     sunday_slate      yes            yes     fingerprint moved; accepted by override
071a5bc55c96  2     thursday_deadline yes            yes     fingerprint moved; accepted by override

5 passed only because the fingerprint check was overridden. The source these
checks enforce had moved and they were accepted anyway.
All 5 captured decisions reconstruct and match replay.
```

All five reconstruct and match replay. They already required `--allow-code-drift` after
stage 2's move through `scoring.py`, so this move does not change what is needed to verify
them; it changes only how many named causes stand between them and a clean check.

Unlike the previous two moves this one is not an accounting cost. A decision made under a
different opponent model or a different simulator **is** a different decision, so a
fingerprint that refuses to certify the old ones is the fingerprint working.

## Out Of Scope, And Deliberately Deferred

- **Any change to `projections.lam`.** Phase 3B's no-promotion outcome stands. The
  simulator wraps the shipped forecast and does not move it.
- **Throwing/scoring component rates.** The fix for the QB/receiver identity, named above,
  requires a projection-model change and is therefore out of scope by the line above.
- **The multi-event replay.** Measuring the value of waiting for news still needs a replay
  that processes timestamped observations and hold/commit events in order. It does not
  exist and nothing here substitutes for it.
- **Reopening 3C.** One captured decision verified the same day answers capture fidelity
  whenever that is worth doing. It needs no window and is not part of this.
- **A learned opponent model.** Sixteen weeks times four rivals is not a training set. The
  log exists to choose between three declared hypotheses and to calibrate one noise
  parameter, not to fit a model.
