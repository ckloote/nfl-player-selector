# Archived results — superseded definitions

This report is preserved as research history. Its numbers use the previous offensive-only scoring, closing-line replay horizon and earlier evaluation semantics. Its season-gain, confirmatory-gate, untouched-holdout and model-ceiling interpretations are superseded. Do not combine these results with the corrected Phase 2 artifacts. Two headline claims below are specifically refuted: dropping the Vegas multiplier is sized at −8.56 TDs per season here and −3.11 on the identical nine seasons once future closing lines are masked, and “the optimizer is not worth it” (−0.73, better in 8 of 15) becomes +2.87 (better in 10 of 15) under the corrected input policy. Neither the old nor the new optimizer figure clears two standard errors. See [REVIEW.md](../REVIEW.md) F02 and the current [backtest report](../BACKTEST.md).

---

# Backtest: does the optimizer actually work?

> Historical report: all published numbers below use the previous scoring
> definition (passing + rushing + receiving TDs). They have not been rerun with
> the complete throwing/scoring credits now used by the application. Refresh
> each prior/replay season before new comparisons. The evaluation limitations
> in [REVIEW.md](REVIEW.md) remain open; this workflow change makes no model
> performance claim.

The tool has always been able to produce a confident pick sheet. This is the
harness that checks whether those picks are any good, by replaying finished
seasons with data frozen at each pick deadline and scoring the result against
what the players actually did.

**The short answer: the projection model is worth real touchdowns; the
season-long assignment optimizer on top of it is not.** Over fifteen seasons
the optimizer is indistinguishable from simply taking the best available player
each week, while both beat a random baseline by ~13 TD/season. Section 6 covers
the second, more useful finding: how small an effect this harness can actually
resolve, which turns out to be the constraint on everything else.

## 1. Running it

```bash
uv run pool refresh --season 2025        # imports 2024 (prior) and 2025
uv run pool backtest --season 2025
uv run pool backtest --season 2011-2025 --detail
uv run pool sweep    --season 2011-2025 --csv sweep.csv
```

`refresh --season Y` imports both `Y-1` and `Y`, and seasons are stored under
disjoint keys, so backfilling is a handful of refreshes. Data is good back to
2010:

```bash
for y in 2011 2013 2015 2017 2019 2021 2023 2025; do uv run pool refresh --season $y; done
```

A backtest needs the prior season's stats (they are the model's starting prior)
and the replay season's results. It refuses to run without them rather than
returning a plausible-looking number: with no prior season every positional mean
collapses to zero and the projections become meaningless.

## 2. What "frozen at the pick deadline" means

This is the part that has to be right; everything else is bookkeeping. When
replaying week W, `projections.load_frames(conn, season, as_of_week=W)` returns
only what was knowable an hour before kickoff:

| Source | Visible at week W | Why |
|---|---|---|
| `games` — teams, weeks, kickoff | all REG weeks | the schedule is published in advance |
| `games.spread_line` / `.total_line` | weeks ≤ W + horizon, NaN beyond | see below |
| `player_weeks`, prior season | all | the prior season is complete |
| `player_weeks`, replay season | **week < W** | week W has not been played |
| `rosters` | week ≤ W, latest per player | catches in-season signings and cuts |
| `injuries` | week ≤ W | the report is published before kickoff |
| final scores | never read by the model | — |

**Vegas lines are horizon-limited.** A finished season carries closing lines for
every week, but live the database holds roughly six weeks of lines and almost
nothing beyond (verified: at the start of 2026, weeks 1–6 were fully lined,
week 7 half, weeks 8+ empty). Replaying with the whole season's lines would make
far-future matchups look more knowable than they are — biasing the future
discount upward, which is one of the two parameters being tuned. Lines past
`W + VEGAS_HORIZON_WEEKS` (default 6) are blanked, and the existing
team-average/league-average fallback fills them exactly as it does live.

**Role comes from usage, not the depth chart.** `depth_charts` stores a single
snapshot per season with no week column, and the stored 2025 snapshot is dated
**2026-03-14** — a post-free-agency chart. It cannot be rewound, so backtests
default to `--role-source usage`: rank players within their own (team, position)
by touches per game to date, then read the same `DEPTH_MULT` table. The choice
barely moves the result (over 2021–2025 the optimizer-minus-greedy delta was
+2.75 with usage, +3.00 with the leaking snapshot, +4.50 with no role factor at
all — all inside noise).

The freeze is enforced by a test asserting the frozen frame is a *pure function
of the visible slice*: build week-W projections from a full database and from
one physically truncated after week W-1, and require the frames to be identical.
Stated that way it catches any future leak, including through a column nobody
thought about — which is exactly how the depth-chart snapshot got in unnoticed.

## 3. The strategies

All four see the same frozen projections, so the comparison isolates the
decision rule rather than the model. Candidates come from the optimizer's own
`build_matrix`, so every rule faces the same pool and the same exclusions.

| Strategy | Rule |
|---|---|
| `optimizer` | the shipped rule: solve the rest-of-season assignment, play whoever it assigns to this week |
| `greedy` | the best available player this week, no thought for later weeks — **the one to beat** |
| `random` | uniform among the top 10 this week, averaged over 20 seeded trials |
| `hindsight` | one assignment over what actually happened; unreachable, a scale reference only |

## 4. Results

Fifteen seasons, 2011–2025, at the shipped defaults (`PRIOR_WEIGHT_GAMES = 7.0`,
`FUTURE_DISCOUNT = 0.985`, usage roles, six-week line horizon). Mean TDs/season:

| | QB | RB | FLEX | total |
|---|---|---|---|---|
| optimizer *(shipped)* | 34.67 | 13.13 | 9.27 | **57.07** |
| greedy | 36.20 | 13.47 | 8.13 | **57.80** |
| random top-10 | 27.5 | 9.0 | 8.1 | ~44.5 |
| hindsight (2017–25) | 73.6 | 44.9 | 41.3 | 159.8 |

**Optimizer minus greedy: −0.73 TD/season, SE 2.39, better in 8 of 15 seasons.**
A null result. Split by era, it is null in both halves independently:

| Seasons | optimizer − greedy | Better in |
|---|---|---|
| 2017–2025 | −1.11 (SE 3.33) | 5/9 |
| 2011–2016 | −0.17 (SE 3.65) | 3/6 |

Both real strategies beat random by ~13 TD/season, so the projection model has
genuine signal. Both reach only ~36% of the hindsight ceiling, which mostly says
the ceiling is a fantasy — it picks whoever happened to score three touchdowns,
every week.

### The parameter sweep

`pool sweep`, mean optimizer-minus-greedy TDs per season over 2017–2025:

| prior weight \ discount | 0.80 | 0.90 | 0.95 | 0.985 | 1.00 |
|---|---|---|---|---|---|
| 5 | −0.22 | −0.33 | −2.22 | 0.00 | +0.11 |
| 7 *(current)* | −1.44 | −1.89 | −3.00 | −1.11 | −1.00 |
| 10 | +1.78 | +0.67 | +1.11 | +0.89 | +1.89 |

The whole surface spans −3.0 to +1.9 against a per-season SD of 8–11 — every
cell is inside noise, and the optimizer's own season total barely moves across
the grid (56.8 to 59.1). **No parameters were changed.** Adopting
`prior_weight = 10` because it scored +1.89 would be fitting nine noisy numbers.

### Which factors carry signal

Paired ablation, 2017–2025, greedy as a clean readout. Only the top two clear
the detection floor of §6; **the rest are consistent with zero** and should not
be acted on:

| Change | Effect (TD/season) | Credible? |
|---|---|---|
| drop Vegas implied total | −8.56 (SE 1.72), 0/9 | **yes** |
| base rate only (no matchup) | −8.11 (SE 3.07), 0/9 | **yes** |
| drop home/away | −1.22 (SE 1.38), 2/9 | no |
| drop opponent defense | −0.67 (SE 3.44), 4/9 | no |
| drop role multiplier | +0.11 (SE 1.35), 5/9 | no |

So: **player identity and the Vegas implied team total are the model.**
Opponent defense, role, and home/away are not measurably doing anything, though
"not measurable" here means "smaller than ±3 TD", not "zero".

### Why the assignment layer does not pay

`DESIGN.md` argues greedy "burns Josh Allen in week 1 against a top defense when
week 9 offers him against the league's worst." That effect is real but far
smaller than the framing implies. Decomposing a projection into *who the player
is* versus *which week you play him* (2025, week-1 view):

| Slot | between-player SD | within-player SD (across weeks) | matchup share |
|---|---|---|---|
| QB | 0.815 | 0.117 | **13%** |
| RB | 0.258 | 0.080 | **24%** |
| FLEX | 0.130 | 0.051 | **28%** |

Player identity dominates. Scheduling operates on the remaining 13–28%, and the
optimizer's edge over greedy *in its own currency* is ~+6 projected TD across a
season (2025: 84.5 projected vs greedy's 78.6, of which +3 was realised).
Against a season-to-season SD of ~10 actual touchdowns, an edge that size needs
on the order of a hundred seasons to detect. Fifteen cannot.

The solver is doing real work — only ~20 of 54 picks match greedy's — the
quantity it optimizes is just a small share of what decides the season.

Both strategies also over-project the players they *pick* (proj/act 1.23–1.44)
while the ratio across all player-weeks sits at 0.99–1.06. **That second number
does not mean the model is calibrated, and this doc used to say it did.**
Splitting it by projection bin ([`PROJECTION_BENCHMARK.md`](PROJECTION_BENCHMARK.md))
shows two opposite errors cancelling: the model under-projects the bottom of the
pool (0.67 below λ = 0.05) and over-projects the top (1.21–1.23 above λ = 0.8),
for a calibration slope of 0.880 [0.865, 0.895]. The selected-cell gap is
therefore only partly the optimizer's curse — the top of the distribution is
genuinely over-projected before anything selects from it. Correcting it is
measured there, and turns out to be worth nothing for expected TDs: the
correction is monotone, so it cannot reorder a single pick.

## 5. Ideas tested and rejected

Recorded so nobody rebuilds them. Both are preserved as runnable branches rather
than prose, so a future attempt starts from working code and a known baseline:

| Branch | Experiment |
|---|---|
| `experiment/per-type-rates` | separate pass/rush/receive rates with matched defensive splits |
| `experiment/per-slot-base-rates` | opportunity-based rate for WR/TE, TD history for QB |

Each plugs into `backtest.weekly_projections(..., builder=...)` and is runnable
with `pool backtest --projection <name>`, so it can be re-measured against the
shipped model on identical frozen data at any time.

**Per-type rates (pass/rush/receive), as the original design specified.** Built
in full, including matched per-type defensive splits. Result: **−0.89 TD/season
(SE 1.33) under greedy, −2.89 (SE 1.54) under the optimizer**, over 2017–2025.
No evidence of benefit. Two reasons: it refines the opponent-defense multiplier,
which carries no measurable signal to begin with, and it splits already-sparse
touchdown counts three ways, adding estimation variance. A caveat for anyone
revisiting: the prototype reused the same shrinkage constants for each type,
and sparser counts arguably need heavier shrinkage.

**Per-slot base rates (opportunity for WR/TE, TD history for QB).** This one is
a cautionary tale worth reading before trusting any future result from this
harness. Searching ~12 model variants against 2017–2025 turned up an apparently
strong finding: the FLEX slot scored *worse than random* (−1.09 TD/season, 2/9
seasons), its top-ranked pick lost to a coin flip among its own top 3, and
switching WR/TE to an opportunity-based rate was worth **+3.67 TD/season
(SE 1.59, 7/9 seasons)**. A plausible mechanism came with it — ranking receivers
by prior TD rate selects last season's lucky finishers, who regress.

It did not replicate. Backfilling 2011–2016 as a genuine holdout, with the model
spec frozen beforehand:

| Comparison | 2017–2025 (searched) | 2011–2016 (holdout) | All 15 |
|---|---|---|---|
| per-slot − current, total | +3.67 (SE 1.59) 7/9 | **+0.17 (SE 3.03) 3/6** | +2.27 (SE 1.54) 10/15 |
| per-slot − current, FLEX | +3.11 (SE 1.24) 7/9 | +0.67 (SE 1.69) 3/6 | +2.13 (SE 1.02) 10/15 |

The underlying "FLEX is worse than random" defect did not replicate either: on
the holdout the current model's FLEX beat random by +1.77. The unbiased estimate
of the effect is the holdout column, and it is ~0. **Not shipped.**

## 6. What this harness can and cannot detect

The most useful thing this exercise produced. The paired season-to-season SD is
~6 TD for a projection change and ~9 TD for a decision-rule change. With the 15
seasons that exist:

- **Resolvable:** effects larger than roughly **±3 TD/season** for a model
  tweak, **±5 TD/season** for a rule change. *Scoring the projection layer
  directly instead lifts this by roughly 5x for a model change — see
  [`PROJECTION_BENCHMARK.md`](PROJECTION_BENCHMARK.md) — but only for claims
  about the forecast. Season-level claims are still governed by this floor.*
- **Not resolvable:** anything smaller — which is *every* model refinement
  tested here.

So the harness can catch disasters and confirm large effects (Vegas, player
identity, and the optimizer-vs-greedy null all clear the bar). It **cannot
adjudicate fine-grained model tuning**, and any future "+2 TD/season" result
from it should be treated as unproven until it survives a holdout.

Extending back to 1999 would roughly double the sample and bring the floor to
~±3.6 TD for rule changes, at the cost of mixing in a materially different era
of football.

## 7. Known imperfections

- **Closing lines.** `spread_line`/`total_line` are closing numbers, so even
  week W's line embeds Saturday-night news a Thursday picker would not have.
  Unavoidable from this data, and it applies equally to every strategy.
- **All three slots are picked at the start of the week.** Real picks lock
  slot-by-slot at each player's kickoff. This is the conservative direction —
  the live tool can wait for Sunday inactives — so it should if anything
  understate the live tool.
- **Injury reports are the final status**, not a Wednesday snapshot.
- **Hindsight ignores roster eligibility**, so it is a loose ceiling.

## 8. Decisions taken, and what is worth doing next

**The optimizer stays, unchanged.** The evidence is a null, not a defeat: the
true effect sits somewhere around [−5, +4] TD/season and the harness cannot
narrow it further. Meanwhile the assignment solve is what produces `pool plan`
and the season-cost ranking in `recommend` — the tool's actual explanatory
value, which greedy cannot generate — and Phase 3's Monte Carlo needs a
rest-of-season policy to simulate. Read the plan as a forecast of intent, which
is what `DESIGN.md` always called it, rather than as a proven edge.

Deliberately **not** done: the 15-season split shows the optimizer worse at QB
(34.67 vs 36.20) and better at FLEX (9.27 vs 8.13), suggesting a per-slot hybrid
rule. That is a ~1.5 TD effect found by slicing data already searched, well under
the §6 floor. Acting on it would repeat exactly the mistake the holdout caught.

Ranked by expected value, given all of the above:

1. **Phase 3 (win probability), not more projection work.** In a winner-take-all
   pool the objective is finishing first. The optimizer's near-tie with greedy on
   *mean* touchdowns says nothing about variance, and the two rules have visibly
   different shapes (18/54 zero-scoring picks vs greedy's 20/54). This is also
   the one area where the payoff could plausibly exceed the detection floor.
2. **Red-zone and end-zone usage** from play-by-play, if projections are revisited.
   Opportunity is a genuinely different feature rather than a refinement of the
   TD-rate estimator, so it is not capped by the saturation in §4 — but note the
   per-slot result above, and require a holdout before believing any gain.
3. ~~**Consider dropping `def_mult`** on simplicity grounds.~~ **Refuted.**
   Measured on forecasts rather than season totals, `def_mult`, `home_mult` and
   `role_mult` together are worth −2.11 TD/season (SE 0.53), replicated on a
   2019–2025 holdout. Individually null at ±3 TD; jointly visible at t = 4.
   See [`PROJECTION_BENCHMARK.md`](PROJECTION_BENCHMARK.md).
4. **Do not** pursue per-type rates, further shrinkage tuning, or discount
   tuning. All measured flat — and shrinkage has since been measured properly
   and still comes out flat, with a holdout to prove it.
