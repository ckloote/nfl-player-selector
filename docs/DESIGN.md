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
  TDs, subject to one-use-per-player. Picking greedily each week ("best
  available now") is provably worse — it burns Josh Allen in week 1 against a
  top defense when week 9 offers him against the league's worst.
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
  - *Pool state*: your picks and results, each opponent's picks and results,
    weekly standings. (Only your own picks exist today; the opponent and
    standings tables arrive with Phase 3.)
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
| Player baseline TD rate | TDs per game from a blend of last season and this season (see updating below), with per-type rates kept separate (pass/rush/receive) since matchups affect them differently |
| Opponent defense | Multiplier from TDs allowed by the opponent to that position, regressed toward league average (defense stats are noisy early in the season) |
| Vegas implied team total | The single best public predictor of scoring; scale λ by implied team points relative to league average |
| Home/away | Small fixed adjustment |
| Role | Depth-chart rank multiplier: a backup QB is worth a small fraction of a starter; RB2/WR2 keep most of their value (committees and multiple starters) |
| Availability | Injury status and bye weeks zero out or discount λ; only active-roster players are in the pool |

**Start-of-season prior (before any 2026 games):** last season's rates,
regressed toward positional means (touchdown rates are notoriously noisy —
regress heavily), adjusted for team changes and role changes where known.

**In-season updating:** a Bayesian shrinkage blend — effectively
`λ = (w · prior_rate + games_played · observed_rate) / (w + games_played)`
where `w` is a prior weight of roughly 6–8 games. Early in the season the model
trusts last year; by midseason this year's data dominates. This is simple,
transparent, and hard to beat without much more sophistication.

**Variance:** Poisson gives variance = mean for free, but real TD scoring is
lumpier (multi-TD games cluster). Fit a negative-binomial dispersion parameter
per position from historical data. "Boom/bust" players are those whose TD
production concentrates (goal-line backs, deep-threat WRs on high-total games);
the dispersion measure is what the risk module leans on.

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
of projected expected TDs (bye weeks and already-used players get −∞).

**Rolling horizon:** only this week's pick is ever committed. Each week the
projections are refreshed and the whole remaining-season assignment is
re-solved. The future plan is a *forecast of intent*, recomputed as information
arrives — that's how "revise the model as stats accumulate" propagates into
decisions automatically.

**Within-week, slot-by-slot decisions:** the assignment is computed at week
granularity, but the recommendation layer applies an **early-commitment rule**
for players whose games kick off before the week's main slate: commit early
only when the early player's projected edge over the best later-game
alternative exceeds an "information premium" (a tunable threshold representing
the value of Friday injury news and Sunday inactives — calibratable from
historical late-week injury-downgrade rates). Otherwise the tool says "hold
this slot," and re-solving after early games have locked (or finished) is a
first-class operation: locked slots are frozen, everything else re-optimizes.

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

This module is deliberately built last (Phase 3): early in the season, EV
maximization *is* the right strategy, so the tool is fully useful before this
exists.

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
                                #   (everyone's picks + totals); manual
                                #   `pool opponent record` as fallback
pool standings                  # leaderboard + remaining-arsenal comparison
pool plan                       # full remaining-season assignment view
```

A web dashboard is a possible Phase 4 nicety, not a requirement.

## 4. Design decisions & trade-offs

- **Exact assignment solver over heuristics:** the LP-assignment problem is
  tiny (≤ ~300 players × 18 weeks per slot); exact solutions are free, so no
  greedy heuristics anywhere in the core.
- **Simple statistical model over ML:** with 18 games/season, TD data is far
  too sparse to train fancy models without overfitting. Poisson rates + Vegas
  lines + shrinkage is transparent, debuggable, and near the practical ceiling.
  The projection-frame interface leaves the door open.
- **Leaderboard from the official weekly report:** the pool publishes
  everyone's picks and totals at week's end, so opponent state is entered once
  a week from that report (a CSV/paste import, with manual entry as fallback)
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

## 5. Out of scope (for now)

- Automatic scraping of your pool's website (manual entry instead).
- DFS-style lineup pricing/salary logic — this pool has no salaries.
- Live in-game data; the tool operates on the weekly cadence.
