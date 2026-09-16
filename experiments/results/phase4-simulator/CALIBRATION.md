# Phase 4 Stage 3: Simulator Calibration

**Run:** 2026-09-16 (UTC), by `experiments/simulator_calibration.py`, against the frozen
`roster-snapshot-repair` study — 2011–2025, 103,027 shipped hard-eligible player-weeks with
positive lambda, 3,555 team-game pairs, 7,415 team-weeks.

Two parameters are **measured**, one is **fitted**, and four checks are **neither**. The
checks are the point: matching a covariance says nothing about whether the marginal spread
or the conditional tails come out right, and the shared-credit mechanism is asserted by the
model rather than estimated from it. Each is reported pass or fail.

Nothing here reads the operational pick database, and nothing here changes `projections.lam`.

## Measured

| Parameter | Value | Source |
|---|---|---|
| `THROW_SHARE` | **0.9064** | `touchdown_credits.kind`: the share of a quarterback's pool credits that are `throwing` |
| `RECEIVE_SHARE` WR | **0.975** | `rec_td / (rush_td + rec_td)` — the share of his scores his passer is also paid for |
| `RECEIVE_SHARE` TE | **0.980** | as above |
| `RECEIVE_SHARE` RB | **0.207** | as above; a rushing touchdown links to no passer |

A quarterback's score is almost entirely his receivers' scores. That is not a modelling
choice, it is the pool's own scoring rule — one credited pass pays two players — and 90.6%
is how much of him it accounts for.

## Fitted

| Parameter | Value | How |
|---|---|---|
| `K_GAME` | **14.302** | Moment estimator on opposing teams sharing one game factor: `k = Σ Lam_A·Lam_B / Σ (Y_A−Lam_A)(Y_B−Lam_B)` |

Fitted on the model's own rates rather than on raw totals, so every matchup difference the
projection already knows about is not counted a second time.

## Check 1 — Marginal spread: **PASS**

> observed / model = **1.029** over 89,000 non-quarterback player-weeks (1.00 is exact)

`k_game` was fitted on a *cross-team covariance* and is here asked to reproduce a
*single-player variance*. It does, to within three percent. This is the strongest result in
the run: the one fitted number is not absorbing slack.

## Check 2 — Conditional tails: **PASS on shape, FAIL on level, and the level is inherited**

Two model columns, because two different things can be wrong and only one is this stage's
business. `model_*` uses the shipped lambda as it stands. `shape_*` first rescales each
bin's lambdas to the mean actually observed in that bin, which removes the projection's own
calibration tilt and leaves only the question the simulator owns: **given the right average,
does the mixture put the right mass in the tail?**

| lambda bin | n | lam | actual | tilt | obs P(≥1) | model | shape | obs P(≥2) | model | shape |
|---|---|---|---|---|---|---|---|---|---|---|
| (0.00, 0.05] | 12726 | 0.037 | 0.055 | 1.49 | 0.0519 | 0.0359 | **0.0530** | 0.0027 | 0.0007 | 0.0016 |
| (0.05, 0.10] | 24732 | 0.073 | 0.082 | 1.12 | 0.0750 | 0.0705 | **0.0786** | 0.0068 | 0.0028 | 0.0035 |
| (0.10, 0.15] | 14114 | 0.122 | 0.125 | 1.03 | 0.1125 | 0.1143 | **0.1171** | 0.0117 | 0.0074 | 0.0077 |
| (0.15, 0.20] | 7869 | 0.173 | 0.201 | 1.16 | 0.1754 | 0.1583 | **0.1807** | 0.0230 | 0.0142 | 0.0187 |
| (0.20, 0.30] | 10385 | 0.246 | 0.267 | 1.08 | 0.2279 | 0.2162 | **0.2317** | 0.0344 | 0.0272 | 0.0314 |
| (0.30, 0.40] | 6347 | 0.347 | 0.352 | 1.01 | 0.2948 | 0.2898 | **0.2931** | 0.0526 | 0.0500 | 0.0511 |
| (0.40, 0.60] | 6277 | 0.482 | 0.437 | 0.91 | 0.3506 | 0.3768 | **0.3492** | 0.0757 | 0.0878 | 0.0746 |
| (0.60, 0.90] | 3030 | 0.715 | 0.600 | 0.84 | 0.4469 | 0.5009 | **0.4432** | 0.1267 | 0.1638 | 0.1249 |
| (0.90, 10.0] | 857 | 1.087 | 0.777 | 0.71 | 0.5473 | 0.6443 | **0.5271** | 0.1820 | 0.2938 | 0.1853 |

**The shape is right.** Once the level is corrected, `shape_ge1` tracks the observed
frequency within a couple of points everywhere, and `shape_ge2` is close from lambda 0.3
upward — which is the range anything pickable lives in.

**The level is wrong, and it is not this stage's to fix.** The `tilt` column is the shipped
model's calibration slope showing through: it over-predicts high-rate players by 29% in the
top bin and under-predicts the bottom bin by 49%. That is the Phase 3B finding, unchanged
and unchallenged here. The simulator inherits it and is forbidden from correcting it.

**One real residual.** Below lambda 0.15 the model understates `P(≥2)` by roughly two-fold
even after the tilt is removed: low-rate players have fatter tails than the mixture allows.
Nobody picks a 0.07 player, so it does not bind on a decision, but it is a genuine
misspecification and is recorded as one rather than rounded away.

## Check 3 — Shared credit: **the structure is right; the ranking is not**

| Pair | Observed (lead by lambda) | Simulated | Reality, lead by **targets** |
|---|---|---|---|
| QB × lead pass-catcher | +0.358 | +0.502 | **+0.525** exact identity, **+0.482** real passer |
| QB × lead back | +0.027 | +0.181 | |
| Lead catcher × lead back | −0.029 | +0.052 | |

Read alone, the first row looks like a 40% overstatement. It is not, and the third column is
why. If a quarterback's credits were *exactly* his receivers' receiving touchdowns — the
identity the model asserts — then real 2011–2025 outcomes, taking each team's true lead
receiver by targets, give **+0.525**. Against the real passer, **+0.482**. The simulated
+0.502 sits between them.

So the mechanism is right. What the gap to +0.358 measures is something else: **how well
`lam` identifies which receiver leads a team.** The simulator's lead-by-lambda receiver is
by construction the one carrying the largest share of the arm; reality's lead-by-lambda
receiver is frequently the team's third option. The projection's within-team ordering error
attenuates the observed figure, and no change to the simulator would close it.

The consequence for the policy is worth stating plainly, because it is directional and it
will show up in advice: **the model is more confident than it should be that the specific
receiver it names is the one coupled to the quarterback.** It will therefore somewhat
overstate the value of stacking a named quarterback with a named receiver. It does not
overstate the existence or the size of the stacking effect in general.

## Check 4 — Same-team substitution: **FAIL, quantified, and deferred with a route**

> team passing touchdowns per game: mean 1.525, var 1.348, **var/mean = 0.884**

A team's passing touchdowns are *under*-dispersed. Red-zone chances are finite, so a team's
receivers compete, which is why the observed lead-catcher/lead-back correlation is −0.029
while the model gives +0.052.

The model cannot produce this, and one specific attempt proved it cannot. Drawing the team's
passing total and *allocating* it multinomially among receivers looks like it should create
competition. It does not: **splitting a Poisson total multinomially returns independent
Poissons.** The change was implemented and the calibration output did not move by a single
digit in any of the three correlations — which is the cleanest possible demonstration that
no allocation rule whatever can manufacture substitution on top of a Poisson total.

Substitution requires an under-dispersed total, e.g. a binomial team count with `var/mean =
0.884` in place of the Poisson. That is the route, it is a change to one draw, and it is not
taken here: it carries its own calibration burden and this run's job is to report. The
allocation structure is kept because it is where that draw goes.

Effect size: the correlation involved is −0.029 against a modelled +0.052, i.e. about 0.08
of correlation on pairs that are nearly uncorrelated either way. Against the +0.5 quarterback
stack it is small. The direction is known — the tool will very slightly overrate holding a
back and a receiver from the same team.

## What this run does not establish

The simulator's marginal spread and tail shape match 15 seasons; its dominant correlation
matches the identity that generates it. None of that says a win-probability policy built on
it will win a pool. A season is one Bernoulli trial and no calibration result changes that.

## Reproducing

```bash
uv run python experiments/simulator_calibration.py roster-snapshot-repair
```

Outputs `fitted.json`, `tails.csv`, `marginal-spread.csv`, `shared-credit.csv` and
`identity-and-dispersion.csv` beside this file.
