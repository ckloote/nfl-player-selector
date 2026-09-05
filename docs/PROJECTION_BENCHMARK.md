# Projection benchmark: how good is the forecast, and where does it come from?

> Historical report: all published numbers below use the previous scoring
> definition (passing + rushing + receiving TDs). They have not been rerun with
> the complete throwing/scoring credits now used by the application. Refresh
> each prior/replay season before new comparisons. The evaluation limitations
> in [REVIEW.md](REVIEW.md) remain open; this workflow change makes no model
> performance claim.

`BACKTEST.md` scores 54 picks a season and gets one number out. That resolves
effects larger than about ±3 TD/season and nothing smaller, which is every model
refinement anyone has tried. The same frozen projections contain **7,600 player-week
forecasts a season** — 110,000 over 2011–2025, 877,000 across the eight models
compared here. This is what that resolution buys.

**The short answer: the shipped model wins the bake-off, and the multiplier stack
the season harness could not see is worth about 2.5 TD/season. The model is also
measurably mis-calibrated — over-projecting its best players by ~20% — and fixing
that provably cannot change a single pick.** Section 5 is the useful part: two
findings that only exist because forecast quality and pool value were measured
separately.

## 1. Running it

```bash
uv run pool models                                  # the model list
uv run pool evaluate --season 2011-2025             # the bake-off
uv run pool evaluate --season 2011-2025 --csv eval.csv
uv run pool backtest --season 2017-2025 --projection player-vegas
```

`evaluate` builds the same frozen frames `backtest` does, via the same
`builder=` seam, and keeps only the week being forecast — the later weeks a frame
carries will be rebuilt before anyone picks from them. Eight models over fifteen
seasons takes about six minutes and is byte-for-byte reproducible.

## 2. The models

All eight share the shipped model's candidate pool, bye handling and injury
availability, and differ only in `lam`. Availability is an eligibility filter,
not a modelling choice under test.

| Model | `lam` |
|---|---|
| `random` | shipped λ shuffled within each slot-week — 20 seeded permutations |
| `within-player` | each player's λ shuffled across their own weeks |
| `historical-rate` | prior-season TD/game, unregressed |
| `regressed-rate` | prior season regressed to the positional mean |
| `current-season-rate` | current season only, shrunk |
| `vegas-environment` | positional mean × role × Vegas — no player TD history at all |
| `player-vegas` | `base_rate × vegas_mult` — shipped with defense, home/away and role off |
| `shipped` | the production model, frozen |

## 3. The metric, and why it is not the obvious one

Two choices were made against measured behaviour on the two ablations
`BACKTEST.md` already sized (dropping Vegas, −8.56 TD/season; base rate only,
−8.11), not from first principles. Both turned out to matter more than the
modelling.

**Rank among players still available, not in the full pool.** The pool spends each
player once, so by midseason the decision is made well down the ranking, and that
is where models separate. Ranked in the full pool, top-1 is the same handful of
stars every model likes: it recovers only −4.9 of the −8.6 TD/season that dropping
Vegas actually costs. Ranked among still-available players it recovers −8.8.

**And that depleted pool must be shared.** Scoring each model against its *own*
greedy walk rewards it for picking badly: a bad pick leaves a better player behind,
inflating every later top-k. This is not hypothetical — with per-model pools, 12 of
15 shrinkage perturbations "improved" the primary metric. Against one common pool
the same sweep splits 8 up, 7 down.

**k = 10, not k = 1.** Sensitivity falls with k but noise falls faster:

| metric | drop Vegas | base rate only |
|---|---|---|
| depleted top-1 | −6.78 (SE 1.55) | −8.11 (SE 2.80) |
| depleted top-10 | −1.76 (SE **0.35**) | −1.74 (SE **0.50**) |

Top-1 *is* the decision, but it reproduces the season harness's own resolution, so
selecting on it buys nothing. Top-10 is the pre-registered **primary**; depleted
top-1 is the confirmatory **gate**; a change must move both in the same direction.

## 4. The bake-off

Fifteen seasons, realised TDs per pick, ranked among still-available players:

| Model | top1 | top3 | top10 | Spearman | primary vs shipped | gate vs shipped |
|---|---:|---:|---:|---:|---:|---:|
| `shipped` | **1.112** | **0.989** | **0.839** | **0.400** | — | — |
| `player-vegas` | 1.062 | 0.970 | 0.799 | 0.337 | −2.11 (SE 0.53) 2/15 | −2.60 (SE 1.65) 4/15 |
| `vegas-environment` | 0.949 | 0.923 | 0.798 | 0.377 | −2.18 (SE 0.43) 1/15 | −8.47 (SE 2.50) 3/15 |
| `current-season-rate` | 0.956 | 0.892 | 0.687 | 0.208 | −7.95 (SE 0.79) 0/15 | −8.07 (SE 2.28) 2/15 |
| `regressed-rate` | 0.941 | 0.853 | 0.722 | 0.299 | −6.13 (SE 0.80) 0/15 | −8.87 (SE 2.40) 1/15 |
| `historical-rate` | 0.821 | 0.801 | 0.719 | 0.337 | −6.26 (SE 0.87) 0/15 | −15.13 (SE 2.86) 0/15 |
| `within-player` *(null)* | 1.006 | 0.955 | 0.824 | 0.387 | −0.82 (SE 0.38) 5/15 | −5.47 (SE 1.93) 4/15 |
| `random` *(null)* | 0.336 | 0.285 | 0.311 | 0.029 | −27.51 (SE 0.68) 0/15 | −40.33 (SE 2.25) 0/15 |

**The shipped model wins every column.** That is Outcome A of the original plan:
the current architecture is supported and the simplification hypothesis fails.

**The multiplier stack earns its complexity — which the season harness could not
show.** `player-vegas` is the shipped model with `def_mult`, `home_mult` and
`role_mult` switched off. Measured one at a time through season totals, all three
were null (−0.67, −1.22, +0.11, none clearing ±3). Measured together on forecasts
they cost **−2.11 TD/season at SE 0.53** — t = 4.0 — and that replicates:

| | primary | gate |
|---|---|---|
| dev 2011–2018 | −1.45 (SE 0.52) 1/8 | −2.88 (SE 2.57) 3/8 |
| holdout 2019–2025 | −2.87 (SE 0.94) 1/7 | −2.29 (SE 2.18) 1/7 |

This is the resolution the instrument was built for. It is also, note, an argument
*against* the simplification the plan expected to find, and against
`BACKTEST.md`'s suggestion of dropping `def_mult` on complexity grounds.

The nulls behave: `random` loses by 27–40 TD/season with Spearman 0.03.
`within-player` — same players, weeks scrambled — costs 5.47 TD/season on the gate,
which sizes the week-to-week matchup signal. Treat that as an upper bound: the
shuffle also moves a player's injury-week zero into a healthy week, so part of it
is availability rather than matchup.

## 5. Two things that only show up when A and C are measured apart

### The model is over-extreme, and correcting it cannot change a pick

Calibration slope from a Poisson GLM, `E[Y] = exp(a + b·log λ)`, with standard
errors clustered on player-season (a player's error repeats across all their weeks;
naive errors would make everything significant):

| Model | b | 95% CI |
|---|---:|---|
| `shipped` | **0.880** | [0.865, 0.895] |
| `player-vegas` | 0.992 | [0.968, 1.016] |
| `regressed-rate` | 1.010 | [0.977, 1.042] |
| `vegas-environment` | 1.107 | [1.084, 1.130] |

`b < 1` means the forecasts are too spread out. The pooled projected/actual ratio
is 1.02 and says nothing, because it averages two opposite errors:

| λ bin | n | proj | actual | proj/act |
|---|---:|---:|---:|---:|
| 0.00–0.05 | 16,740 | 0.036 | 0.054 | **0.67** |
| 0.05–0.10 | 28,216 | 0.073 | 0.079 | 0.93 |
| 0.15–0.20 | 10,140 | 0.173 | 0.201 | 0.86 |
| 0.40–0.60 | 6,296 | 0.484 | 0.444 | 1.09 |
| 0.80–1.00 | 1,614 | 0.892 | 0.722 | **1.23** |
| 1.00+ | 6,815 | 1.768 | 1.467 | 1.21 |

The baselines localize it: `player-vegas` and `regressed-rate` are calibrated at
b ≈ 1.0. **The tilt comes from the multiplier stack, not the base rate** — the same
multipliers that improve the ranking.

Fitting `λ' = e^a · λ^b` on 2011–2018 (a = −0.1405, b = 0.8933) and applying it to
the untouched holdout moves the slope 0.866 → 0.967 and the top-1 projected/actual
ratio 1.281 → 1.052. It replicates cleanly.

It also cannot possibly help the pool. A power transform is monotone, so it never
reorders anyone: greedy's picks are **identical in all 7 holdout seasons** (delta
exactly 0.00). The optimizer sums λ across weeks so it *can* move, and does in 2 of
7 seasons, for +0.71 TD/season (SE 0.89) — a null. The correction is worth shipping
for the *level*, which Phase 3's Monte Carlo needs and which the optimizer's
cross-week comparisons use. It is worth nothing for expected TDs.

### A better forecast can be a worse pool model

`base-rate-only` beats the shipped model on Poisson deviance (−0.0045) and on
calibration slope (1.009 vs 0.880) while costing **10.3 TD/season**. `player-vegas`
does the same thing on a smaller scale: better deviance (−0.0100, SE 0.0044), worse
by 2.1 TD/season.

Deviance is therefore reported as a layer-A diagnostic and is **not** a gate. Had
it been a co-primary, `base-rate-only` would have passed half of it.

## 6. Shrinkage constants: nothing ships

The one place the instrument should have been decisive. Four constants, ranges and
primary metric pre-registered before any result was read, coordinate sweep on
2011–2018, one pass, no adaptive refinement.

Only one cell moved the primary and the gate in the same direction:
`PRIOR_WEIGHT_GAMES = 3` (shipped: 7), primary +0.86 (SE 0.34) 6/8, gate +1.88
(SE 1.32) 6/8, with a monotone trend across its neighbours (3: +0.86, 5: +0.70,
10: −0.38, 14: −0.30) rather than an isolated win. The gate earned its keep here:
`DEF_SHRINK_GAMES = 32` won the primary in **8 of 8** dev seasons and reverses to
−1.75 on the gate.

Frozen and applied once to 2019–2025:

| | dev 2011–2018 | holdout 2019–2025 |
|---|---|---|
| primary (top-10) | +0.86 (SE 0.34) 6/8 | +0.46 (SE 0.58) 4/7 |
| gate (top-1) | +1.88 (SE 1.32) 6/8 | **−0.86 (SE 1.50) 4/7** |
| season replay (greedy) | — | −0.57 (SE 3.01) 3/7 |

The gate reverses sign and the primary drops to t = 0.8. **`PRIOR_WEIGHT_GAMES`
stays at 7.0.** This is the per-slot experiment's pattern again (+3.67 dev, +0.17
holdout) at smaller scale, and it is the second time the holdout has killed a
result that looked solid on the development block.

A limitation worth recording: the surviving candidate sat at the edge of its
pre-registered range, so the trend may continue below 3. Extending the grid after
seeing that would be exactly the adaptive refinement the pre-registration forbids;
it needs a fresh one.

## 7. What this does and does not establish

**Does:** the shipped projection model is the best of the eight on ranking; its
contextual multipliers are worth ~2.5 TD/season and replicate on a holdout; it is
over-extreme by a measurable and correctable amount; touchdown counts are close
enough to Poisson that a negative-binomial is not called for (variance/mean by λ
bin runs 1.23 at the middle down to 0.91 at the top, and within-bin λ spread
inflates even that); and no shrinkage constant
survives a holdout.

**Does not:** say anything about layer C. A forecast win is not a pool win, the
±3 TD/season floor still governs every season-level claim, and the two are
demonstrably capable of pointing in opposite directions — §5 has two examples.

## 8. Proposed and deliberately not done

Recorded for the same reason `BACKTEST.md` §5 records rejected ideas: so nobody
rebuilds them, and so the reasons can be argued with later. The research plan
this phase came from was larger than what ran, and the cuts were made for
reasons the results have since either confirmed or complicated.

| Proposed | Why not | Status now |
|---|---|---|
| A learned/regularized ensemble over the existing features | Different cost class — expanding-window refits inside the point-in-time framework — and its whole feature list is information already in the model | **Deferred.** §4 says the information set is close to exhausted, so this would likely rediscover `base_rate × vegas`. Worth doing once, as an upper bound, after new features |
| Red-zone / goal-line usage from play-by-play | Needs a new nflverse import path | **Deferred, and now the best candidate.** It is the only proposal carrying information the current set lacks |
| A 6 × 2 model × strategy matrix through the season harness | Twelve season-level comparisons at the ±3 TD floor, on seasons already searched, is the machine that produced the per-slot false positive | **Dropped.** Replaced by one pre-registered challenger |
| A negative-binomial count model | — | **Closed.** Counts are not overdispersed (§7) |
| Backfilling 1999–2009 to enlarge the sample | No injury reports before ~2009 and no weekly rosters before ~2002, so `player_pool` falls to its last-stats-row branch and the candidate universe changes | **Rejected.** The model has no cross-season training data — a replay of season Y reads only Y−1 and Y — so older seasons add replay seasons, not information. They may still be worth it for structural quantities that want long history, but only after the candidate universe is made comparable |
| Simplifying to `player-vegas`, and dropping `def_mult` | This was the *expected* outcome | **Refuted** (§4). The multiplier stack is worth −2.11 TD/season with a holdout replication |

Two methodological cuts are worth naming because they were wrong in the plan and
corrected against data, not argument. The primary metric was pre-registered as
top-10 **of the full pool**; §3 shows that recovers barely half of a known
effect, and the depleted-pool ranking replaced it. And Poisson deviance was
pre-registered as a **co-primary gate**; §5 shows a model 10 TD/season worse
passes it, so it was demoted to a diagnostic.

## 9. What is worth doing next

1. **Ship the calibration correction**, but for the right reason. It is worth
   nothing for expected TDs and is a prerequisite for Phase 3: a Monte Carlo over
   win probability needs λ to mean what it says, and today the best players' λ is
   ~20% too high.
2. **Phase 3 (win probability)**, unchanged from `BACKTEST.md`'s ranking. The
   overdispersion question is now closed in Poisson's favour, so it is less blocked
   than it looked.
3. **Red-zone / goal-line usage**, if projections are revisited. Every model here
   shares one information set, and the bake-off says that set is close to
   exhausted: the spread between the shipped model and `player-vegas` is 2 TD, and
   between shipped and a model with *no player history at all*
   (`vegas-environment`) only 8. New information is the only thing left that is not
   capped by the ceiling this benchmark just measured.
4. **Do not** revisit the shrinkage constants, drop `def_mult`, or simplify to
   `player-vegas`. All three are now measured, and all three say no.
