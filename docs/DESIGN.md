# Design: NFL Touchdown Pool Optimizer

## 1. The game

Pool rules, restated as constraints:

1. Each week, pick exactly one **QB**, one **RB**, and one **WR/TE**.
2. You score one point for every TD a pick **scores or throws** that week
   (a QB gets credit for passing *and* rushing TDs; a RB for rushing *and*
   receiving TDs, etc.).
3. A player may be used **at most once per season**, across all weeks.
4. Highest cumulative TD total after the final week wins. **Winner take all.**
5. Each pick locks independently, **1 hour before that player's game starts**.
   The three picks in a week need not be submitted together: you could lock a
   RB before Thursday's game, a QB before Sunday's, and a WR before Monday's.
6. At the end of each week, an official report confirms every entrant's picks
   and running totals.

Consequences worth calling out:

- Over an 18-week season you will use 18 QBs, 18 RBs, and 18 WR/TEs — 54
  distinct players. The talent pool at each position is much shallower than 18
  "great" options, so the bottom of your schedule is filled by matchup plays.
  Finding the weeks where a mediocre player has a great spot matters as much as
  scheduling the stars.
- Because the schedule is known in advance, this is fundamentally an
  **assignment problem**: assign players to weeks to maximize total expected
  TDs, subject to one-use-per-player. Exact assignment maximizes the fixed pruned forecast matrix. Its rolling
  policy is compared with greedy using each policy's own history; uncertain, changing
  projections prevent that mathematical optimum from proving a season-score advantage.
- Winner-take-all means the true objective is **probability of finishing
  first**, not expected TDs. Those diverge exactly when you're not in the
  middle of the pack: a trailing player should prefer a high-variance pick with
  a lower mean, a leader should prefer a high-floor pick and/or copy the
  second-place player's likely picks to deny them ground.
- Per-pick deadlines create an **in-week timing game**. Committing to a
  Thursday player means giving up Friday/Saturday injury news and Sunday
  inactives for that slot — so a Thursday pick has to be *better enough* than
  the best Sunday alternative to pay for the information you forfeit. In the
  other direction, your own early-week results are known before your later
  picks lock: if your Thursday RB busts in a week where you're protecting a
  lead, you can dial up variance on the QB and WR/TE picks that haven't locked
  yet. The recommender must therefore work slot-by-slot within a week, not
  only week-by-week.
- Opponents' current-week picks are hidden until the end-of-week report, so
  in-week you're playing against a one-week-old picture of the leaderboard and
  everyone's remaining player pools. That's still highly informative — the
  used-player lists are exact — but blocking/mirroring decisions are made
  against *predicted* opponent picks, never observed ones.

## 2. What the tool does

The current CLI implements expected-TD advice, picks and scoring. Opponent tracking,
boom/bust displays and leaderboard adaptation described below are proposed features.

Each week, the tool answers: **"Who should I pick this week, and why?"**

It outputs, per position slot:

- The recommended pick, with expected TDs, a variance/boom-bust indicator, and
  a one-line rationale (opponent, Vegas context, usage trend).
- The top ~5 alternatives, so a human can veto (news the model hasn't seen,
  gut feel).
- A view of the *rest-of-season plan* — which weeks the optimizer currently
  intends to spend each remaining top player — so you can see the cost of
  deviating.

Because picks lock slot-by-slot at different times, "this week" is not one
decision: the tool supports recording picks one slot at a time and
re-recommending for the still-open slots with everything known at that moment
(locked picks, injury news, and — late in the season — your own early-game
results).

It also tracks state: your used players, your weekly scores, and every
opponent's picks and totals (entered from the official end-of-week report),
and adjusts recommendations based on your leaderboard position.

## 3. Architecture

Four layers, deliberately decoupled so each can improve independently:

```
┌────────────────────────────────────────────────────────┐
│  CLI (weekly workflow: refresh → recommend → record)   │
├────────────────────────────────────────────────────────┤
│  Optimizer            │  Risk / leaderboard module     │
│  (assignment solver)  │  (Monte Carlo win probability) │
├────────────────────────────────────────────────────────┤
│  Projection model (per-player, per-week TD forecast)   │
├────────────────────────────────────────────────────────┤
│  Data layer (stats, schedules, injuries, Vegas lines)  │
└────────────────────────────────────────────────────────┘
```

### 3.1 Data layer

- **Source:** the [nflverse](https://github.com/nflverse) data via the
  `nflreadpy` Python package (the maintained successor to `nfl_data_py`) —
  free and includes weekly player stats (with TD breakdowns by type), season
  schedules with betting lines (spreads and totals), weekly rosters, depth
  charts, and injury reports, both historical and current. Files for the
  current season appear as the season progresses (e.g. no weekly stats or
  injury reports before Week 1), so the importer treats a missing file as
  "not published yet" rather than an error.
- **Storage:** a local SQLite database (`data/pool.db`). Two categories of
  tables:
  - *Imported*: weekly player stats (prior season + current season), schedules
    and Vegas lines, weekly rosters, depth charts, injury reports. Team-defense
    strength is derived from opponents' stat lines at query time rather than
    imported as its own table.
  - *Pool state*: your picks and results, plus `pool_entrants`, `pool_picks`, and
    `pool_report_totals` for official weekly reports. Entrant picks retain the
    reported name even when player resolution fails; totals and ranks preserve
    what the pool reported. Entrant scoring and remaining-pool comparisons are
    Phase 4, stage 2 work.
- **Pool reports:** `report import` commits the delivered bytes to the existing
  content-addressed archive before parsing. A second transaction writes resolved
  records and parse outcomes in `meta`, keyed by observation id; immutable observations
  retain their original archive coverage. Failed files and `--check` attempts remain
  recoverable. The `pool_report` feed is deliberately excluded from projection inputs,
  snapshot restoration, and freshness checks. See [the stage 1 plan](PHASE4_STAGE1_PLAN.md).
- **Refresh:** one command (`refresh`) re-pulls current-season data. Everything
  downstream reads only from the database, so the model and optimizer never
  care where data came from — which also makes backtesting on past seasons
  trivial.

### 3.2 Projection model

Goal: for every (player, week) pair in the remaining season, a **distribution**
of TDs, not just a point estimate — the risk module needs variance, not just
means.

Model TDs as a **Poisson-like count** with a per-game rate λ built from:

| Factor | How it enters |
|---|---|
| Player baseline TD rate | TDs per game from a blend of last season and this season (see updating below), using complete thrown/scored TD credits; separate per-type rates remain a research proposal |
| Opponent defense | Multiplier from TDs allowed by the opponent to that position, regressed toward league average (defense stats are noisy early in the season) |
| Vegas implied team total | A team scoring-environment estimate; scale λ by implied team points relative to league average |
| Home/away | Small fixed adjustment |
| Role | Depth-chart rank multiplier: a backup QB is worth a small fraction of a starter; RB2/WR2 keep most of their value (committees and multiple starters) |
| Availability | An explicit hard-eligibility mask excludes Out/Doubtful cells and byes; Questionable scales λ by 0.85; eligible zero forecasts remain selectable |

**Start-of-season prior (before any 2026 games):** last season's rates,
regressed toward positional means (touchdown rates are notoriously noisy —
regress heavily), adjusted for team changes and role changes where known.

**In-season updating:** a Bayesian shrinkage blend — effectively
`λ = (w · prior_rate + games_played · observed_rate) / (w + games_played)`
where `w` is a prior weight of roughly 6–8 games. Early in the season the model
trusts last year; by midseason this year's data dominates. This is simple,
transparent, and hard to beat without much more sophistication.

**Variance:** Poisson is a starting count model. Pooled variance/mean agreement does
not validate conditional tails, dependence between players or finishing-first probabilities.
Position and selection strata, shared player outcomes, player correlations, tie handling and
future policy changes need validation before leaderboard simulation is trusted. Negative-binomial
dispersion is an unvalidated option, not a shipped feature or a closed research question.

A strictly increasing calibration map shared by every candidate in a slot preserves
within-week greedy ranks under the same eligibility mask and tie rule. Separate WR and TE
maps can change FLEX ordering even if each map is increasing. A nonlinear shared map can
change sum-maximizing assignments; even a level rescale can change comparisons with the
fixed-TD information premium. [Phase 3](IMPLEMENTATION_PLAN.md) must validate these behaviors,
current and future forecast horizons, and transfer from historical usage roles to live depth
roles before production calibration. Pooled slopes alone do not specify a universal correction;
see the dated [analysis](ANALYSIS.md).

The model layer exposes one interface: a (player, week) projection frame
carrying a TD rate per row — today a Poisson `lam`, later the parameters of a
distribution over TD counts. Everything above it depends only on that frame, so
the model can be upgraded (better features, ML) without touching the optimizer.

### 3.3 Optimizer

The scheduling core. Key observation: **the three position slots are
independent** — a QB pick never competes with a RB pick for eligibility — so
the problem decomposes into three separate assignment problems, one per slot:

> Assign each remaining week exactly one *available* player from the position
> pool, each player to at most one week, maximizing total expected TDs.

This is the classic linear assignment problem, solved exactly and instantly by
`scipy.optimize.linear_sum_assignment` on a (players × remaining weeks) matrix
of discounted projected TDs after candidate pruning. Hard exclusions are applied before
pruning and assignment; missing cells and used players are forbidden. Eligible zero estimates
remain feasible. Current deadline restrictions and recorded locks are preserved.

**Rolling horizon:** only this week's pick is ever committed. Each week the
projections are refreshed and the whole remaining-season assignment is
re-solved. The future plan is a *forecast of intent*, recomputed as information
arrives — that's how "revise the model as stats accumulate" propagates into
decisions automatically.

**Within-week, slot-by-slot decisions:** the assignment is computed at week
granularity, but the recommendation layer applies an **early-commitment rule**
for players whose games kick off before the week's main slate: commit early
unless a considered later-game alternative costs less than the fixed TD
"information premium." This is a heuristic, not an estimated dynamic value of waiting.
The comparison considers every candidate in the solver's own pool and is independent of
`n_alternatives`, which sizes only the displayed alternatives. Zero, default and large
display sizes give the same pick, hold flag, hold alternative and cost.
Re-solving after early games have locked (or finished) preserves recorded locks and
excludes elapsed current-week choices. Current season replay calls `plan_slot`, not
`advise_slot`, once per week, so it does not validate this hold/commit workflow.

**Future-uncertainty discount:** a projected TD in week 17 is worth slightly
less than one now (injury risk, role changes, late-season benching of
locked-playoff-seed starters). Apply a small per-week decay (~1–2%/week,
tunable, and a candidate for calibration in backtesting) to future cells of the
assignment matrix. This nudges the solver away from hoarding stars for
far-future matchups that may never materialize.

### 3.4 Risk / leaderboard module

Turns "maximize expected TDs" into "maximize probability of winning the pool."

- **State tracked:** every opponent's cumulative score *and used players*,
  imported from the official end-of-week report (they face the same one-use
  constraint, so their remaining arsenal is exactly knowable, one week behind).
- **Monte Carlo simulation:** simulate the rest of the season a few thousand
  times. Your picks follow the optimizer plan; opponents are modeled as playing
  a near-optimal assignment over their own remaining pools (with noise). Each
  simulation draws actual TD outcomes from the projection distributions. Output:
  P(you finish first) for each candidate pick this week.
- **Decision rule:** rank this week's candidate picks by *win probability*, not
  expected TDs. This automatically produces the intuitive behavior:
  - **Trailing:** high-variance picks and *differentiation* — avoid picking the
    same player the leader is likely to pick this week (their actual pick is
    unknown until the report, so this is played against their modeled best
    move); you can't gain ground on mirrored picks.
  - **Leading:** high-floor picks and *blocking/mirroring* — favor your
    chaser's best remaining options' equivalents; matched outcomes preserve a
    lead.
  - **Mid-pack / early season:** win probability and expected TDs agree, and
    the module recommends the EV pick.
- The CLI shows both rankings (EV and win-probability) side by side with a note
  when they diverge and why.

This module is later work, after Phase 3 calibration and policy validation. Expected-TD
planning is useful without it, but is not generally equivalent to maximizing win probability.

### 3.5 Interface

A CLI (Python, `typer`), because the weekly workflow is short and scripted:

```
pool refresh                    # pull latest stats/lines/injuries into SQLite
pool recommend --week 4         # open slots only: picks, alternatives, plan,
                                #   and hold-vs-commit advice for early games
pool record --week 4 --rb "B.Robinson"     # lock one slot (slots lock at
pool record --week 4 --qb "J.Allen"        #   different times, so recording
                                           #   is per-slot; repeatable)
pool report import week4.csv    # ingest the official end-of-week report
                                #   --check archives and validates without writing picks
pool report list                # archived attempts, times, counts, and status
pool standings                  # picks, totals, and ranks reported for the latest imported week
pool plan                       # full remaining-season assignment view
```

The initial report parser accepts one CSV row per entrant/slot, with optional reported
totals. `--me` enables comparison against recorded picks; `--allow-roster-change`
acknowledges entrant-set changes. See the [CSV format and examples](../README.md#importing-the-pools-weekly-report).
Computed standings, remaining-player comparisons, and manual `opponent record` are deferred.

A web dashboard is a possible Phase 4 nicety, not a requirement.

## 4. Design decisions & trade-offs

- **Exact assignment solver over heuristics:** the LP-assignment problem is
  tiny (≤ ~300 players × 18 weeks per slot); exact solutions are free, so no
  greedy heuristics anywhere in the core.
- **Simple statistical model over ML:** with 18 games/season, TD data is far
  too sparse to train fancy models without overfitting. Poisson rates + Vegas
  lines + shrinkage is transparent and debuggable. Comparing a few hand-built models
  does not establish a ceiling for these inputs; the projection-frame interface permits
  better features or estimation. Joint context ablations do not identify defense alone.
- **Leaderboard from the official weekly report:** the pool publishes
  everyone's picks and totals at week's end, so opponent state is entered once
  a week from that report (CSV import, with a hand-written CSV as fallback)
  rather than scraped or guessed. The one-week lag is a fact of the game, not
  a tooling gap — everyone plays under it. The tool works (in EV mode) even if
  you skip opponent tracking entirely.
- **SQLite over files:** pool state (picks, opponents) needs transactional
  updates and joins against stats; SQLite gives that with zero infrastructure.
- **Backtesting as a first-class concern:** because data access is behind the
  DB layer, we can replay any past season week by week — freeze data at each
  week, run the recommender, score the result. This is how we'll validate the
  model, tune the future-discount, and honestly answer "does this beat naive
  strategies?" before trusting it with real picks.

## 5. Validation and input history

The shared projection loader implements historical, archived-snapshot and legacy-closing
policies. Historical lines at weeks ≥ W are removed before every team/league average;
week-one uses the shipped fallback. Historical stats stop at W−1 and weekly reports at W.
One stats contract covers the two paths that have observation times: a completed game
already observed when the decision was made is available to it, so an early-week result
enters both live advice and snapshot replay of the same instant. Historical replay keeps
its W−1 cut because it cannot tell which of week W's games had finished. Final schedule
revisions, within-week roster/injury timing and later stat corrections remain approximations.
Historical roles use usage; only observed snapshots support replay of the depth-chart model.

Snapshots use compressed deduplicated payloads and append-only UTC observations. Feed
replacement and archival share a transaction. Source timestamps describe provenance, while
observation time governs availability. Legacy rows have no inferred historical availability.
Snapshot replay resolves every feed into a separate in-memory database with no fallback to
current tables. Outcome scoring uses finalized results separately. One explicit timestamp
per requested week is required; absent/stale optional feeds and essential-history errors are
reported. There is no machine-clock dependency in replay decisions.

Common-pool top-1/3/5/10 rankings share one selected deterministic baseline's greedy depletion.
They measure TDs per ranked candidate, not achieved season gains. Hard eligibility and candidate
keys are asserted across models/seeds. Shuffles permute availability-free values only over
eligible slot-week cells (`random`) or each player's eligible remaining-week cells
(`within-player`) and reapply each recipient's availability adjustment. The within-player
permutation is rebuilt on every decision's remaining-week surface; rate updates, eligibility
and availability remain. It is not a season-long fixed-rate null or a causal separation of
player identity from timing.
Deterministic models run once; shuffled models use seeds 0–19. Seeds are averaged within
season before estimating uncertainty across seasons. Empty comparison cells are reported.

Actual greedy and optimizer replays maintain independent no-reuse histories. Shipped random-top-10
strategy trials and hindsight are separate reference strategies. The configuration in
`experiments/phase2-validation.toml` fixes 2011–2025 replays, 2010 prior history, shipped
constants and retrospective era summaries. The runner audits all game coverage, freezes a
research database, fingerprints code/data/configuration, checkpoints each season, and exports
forecasts, future surfaces, picks, per-seed metrics and reports. Operational picks are untouched.

Generated reports contain facts, methods and provenance; human/AI interpretation belongs in
the separately authored [analysis](ANALYSIS.md), which reruns do not refresh. The saved historical
study supports diagnosis, not calibrated production. Live decisions are captured append-only beside the feed archive: the pre-pruning current and
future surface, each slot's advice and hold/commit call, used/locked state, the resolved input
observations and the code/constant/model identity. Corrections are new events and outcomes are
joined at read time, so nothing already recorded changes. Initial 2026 feed snapshots are not completed live validation.

## 6. Out of scope (for now)

- Automatic scraping of your pool's website (manual entry instead).
- DFS-style lineup pricing/salary logic — this pool has no salaries.
- Live in-game data; the tool operates on the weekly cadence.
