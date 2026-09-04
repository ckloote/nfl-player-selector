# Implementation Plan

Phased so that the tool is *useful from Phase 1 onward* — each phase ships a
working improvement, and with the season starting in early September, Phase 1
is scoped to be usable for real picks within the first weeks.

Stack: Python 3.11+ managed with `uv` (environments, dependencies, and the
`uv_build` backend; `uv.lock` is committed), `nflreadpy` (nflverse data),
SQLite, `scipy` (assignment solver), `numpy`/`pandas`, `typer` + `rich` (CLI),
`pytest`.

CLI commands are written below as `pool <cmd>`; run them as `uv run pool <cmd>`,
or activate `.venv` first. See the README for setup.

The [2026-09-04 project review](REVIEW.md) records open correctness and validation
findings, including issues in phases marked implemented below. Its recommended
sequence is workflow reliability, evaluation repairs, calibration checks, then
leaderboard strategy. The original phase breakdown and research history remain
below; pending proposals and stronger research claims need the reconciliation
described in that review.

| Phase | Status |
|---|---|
| 0 — Scaffolding & data | **Done** (see the schema note under Phase 0) |
| 1 — Projections + optimizer + CLI | **Done** (see notes under Phase 1) |
| 2 — In-season learning | Partly started: the prior/current shrinkage blend and depth-chart roles shipped with Phase 1; per-type rates, dispersion, and `score` remain |
| 3 — Leaderboard-aware strategy | Not started |
| 4 — Validation & polish | Backtesting harness **done**; projection benchmark and calibration report **done** (see Phase 4); the optional niceties remain |

---

## Phase 0 — Scaffolding & data foundation

**Goal:** clean project skeleton and reliable data in a local database.

- Project layout (`src/pool/`, `tests/`, `pyproject.toml`), `uv`-managed
  environment and lockfile, lint/format (`ruff`), CI-friendly test setup
  (`uv sync --locked` then `uv run pytest`).
- SQLite schema: imported tables (`player_weeks`, `games` — schedule and Vegas
  lines in one table, `rosters`, `depth_charts`, `injuries`) plus the
  `my_picks` pool-state table and a `meta` key/value table.
  - Team-defense strength is **derived at query time** from opponents' stat
    lines (`projections.defense_multipliers`), not stored as an imported
    table; byes are likewise derived from the absence of a schedule row.
  - Opponent picks and standings are **Phase 3** tables and do not exist yet.
- `pool refresh`: idempotent import of prior-season (2025) stats and the
  current (2026) schedule/lines via `nflreadpy`; re-runnable all season for
  current-season data.
- **Done when:** `pool refresh` populates the DB from scratch; queries can list
  any player's 2025 weekly TD lines and any team's 2026 schedule; tests cover
  the import transforms.
- *Verified against live nflverse data:* refresh builds the DB in ~3s and is
  idempotent — a second run leaves every table byte-identical and preserves
  recorded picks. Current-season files that don't exist yet degrade to
  "not published yet" instead of erroring, which is the normal preseason state.
- *Coverage:* ~74% overall. Every import transform has unit tests; what remains
  untested in `ingest.py` (~50%) is the network fetch layer and the `refresh`
  orchestration around it. `cli.py` (~37%) is covered only on its error paths —
  no test yet renders a real pick sheet.

## Phase 1 — Projections + optimizer + CLI (minimum useful product)

**Goal:** real weekly recommendations, EV-maximizing, before leaderboard logic.

- Projection v1: per-player Poisson TD rate from regressed prior-season rates ×
  opponent-defense multiplier × Vegas team-total scaling × home/away; byes and
  ruled-out players excluded. Exposed as a whole (player, week) frame from
  `build_projections` / `projections_for`, not the per-player
  `project(player, week)` call originally sketched here — everything
  downstream consumes that frame.
- Optimizer: per-slot linear assignment over remaining weeks
  (`linear_sum_assignment`), future-week discount, respects used players.
- CLI: `recommend` (pick + top-5 alternatives + rationale per open slot, plus
  rest-of-season plan), `record` (log picks **per slot**, since QB/RB/WR-TE
  lock at different times), `plan`. Re-running `recommend` with some slots
  already locked freezes those and re-optimizes the rest.
- Early-commitment advice: when a candidate plays before the main slate, show
  the edge over the best later-game alternative and a hold/commit suggestion
  (fixed information-premium threshold in v1; calibrated later).
- **Done when:** `pool recommend --week N` produces a defensible pick sheet in
  seconds, slots can be locked independently, and recorded picks are correctly
  excluded from future weeks.
- *Verified against live 2026 data:* `recommend` returns in under a second;
  locking one slot leaves the other two open and re-solves the rest of the
  season around it; a full plan assigns 53 slot-weeks to 53 distinct players
  with no reuse of a recorded pick.
- *Implementation notes:* the projection already blends current-season
  observations into the prior (it is the same shrinkage formula, so there was
  no reason to defer it), and a depth-chart role multiplier was added after
  real-data testing surfaced backup QBs as candidates. Alternatives are ranked
  by *season cost* (optimal plan value minus the plan value if that player is
  forced into this week), which is the honest price of overriding the plan.
  The ruled-out/questionable path (`INJURY_MULT` → a forbidden cell in the
  assignment matrix) is covered by tests at all three layers — the transform,
  the multiplier, and the resulting exclusion from both the assignment and the
  advice — but has still never run against real injury data, since
  current-season reports don't publish until Week 1.

## Phase 2 — In-season learning & data depth

**Goal:** the model revises itself as 2026 stats accumulate.

- ~~Bayesian shrinkage blend of prior-season and current-season rates~~
  (shipped in Phase 1; tune `PRIOR_WEIGHT_GAMES` once real 2026 data exists).
- Starter detection from depth charts shipped in Phase 1; refine multipliers
  with current-season usage (a RB2 with 60% of the carries is not a backup).
- Per-type rates (pass/rush/receive TDs) with matching defensive splits.
- Negative-binomial dispersion per position → variance/boom-bust scores
  surfaced in the CLI.
- Injury-status ingestion into availability; weekly `refresh` fully automates
  the data → projection pipeline.
- Scoring: `pool score --week N` pulls actual TDs for my picks and updates my
  running total.
- **Done when:** projections demonstrably shift with in-season data (test with
  synthetic updates), and weekly workflow is refresh → recommend → record →
  score with no manual data edits.

## Phase 3 — Leaderboard-aware strategy

**Goal:** optimize P(win the pool), not expected TDs.

- Opponent tracking: `pool report import` to ingest the official end-of-week
  report (every entrant's picks and totals; CSV or pasted text), with
  `pool opponent record` as a manual fallback; `pool standings` with
  remaining-arsenal comparison (who has already burned which stars).
- Monte Carlo engine: simulate remaining season (mine + modeled opponent
  behavior) drawing from projection distributions; per-candidate-pick win
  probability.
- Decision layer: rank by win probability; surface EV vs. win-prob divergence
  with an explanation (e.g., "trailing by 9 with 5 weeks left → high-variance
  pick and diverge from leader"). Opponent picks for the current week are
  unobserved (report lag), so blocking/mirroring runs against modeled picks.
- In-week adaptivity: when re-recommending open slots after an early game,
  fold your own already-known TD results into the risk posture (e.g., a
  Thursday bust while protecting a lead shifts remaining slots toward floor).
- **Done when:** given contrived standings (big lead / big deficit), the
  recommender visibly shifts toward floor/mirror or ceiling/contrarian picks,
  with tests asserting those directional behaviors.

## Phase 4 — Validation & polish (parallel/ongoing)

**Goal:** trust and convenience.

- ~~Backtesting harness: replay 2024/2025 week-by-week with data frozen at each
  pick deadline; compare cumulative TDs vs. baselines (greedy best-available,
  random top-10, perfect hindsight). Use it to tune the future-discount and
  prior weight.~~ Shipped as `pool backtest` / `pool sweep` (`backtest.py`), and
  replaying 2017-2025 rather than just two seasons — see
  [`BACKTEST.md`](BACKTEST.md).
- ~~Calibration report: projected vs. actual TD distributions.~~ Shipped as
  `pool evaluate` (`evaluate.py`, `models/`) — see
  [`PROJECTION_BENCHMARK.md`](PROJECTION_BENCHMARK.md). It scores every
  player-week forecast (877k across eight models and fifteen seasons) rather
  than the 54 picks a replay scores, which is roughly 5x the resolution for a
  model change. Findings: the shipped model wins the eight-way bake-off; the
  contextual multipliers are jointly worth −2.11 TD/season (SE 0.53) with a
  holdout replication, having been individually null at the season level; the
  model's calibration slope is 0.880 [0.865, 0.895], and the correction for it
  is monotone so it cannot change any pick (greedy identical in 7/7 holdout
  seasons); touchdown counts are not overdispersed; and no shrinkage constant
  survived its holdout, so none was changed. The single projected/actual ratio
  the backtest reported was actively misleading — it averaged 0.67 at the bottom
  of the pool with 1.23 at the top.
- Optional: simple local web dashboard (read-only view of plan/standings),
  pick-deadline reminders.
- **Done when:** a documented backtest shows the optimizer beating the greedy
  baseline over a full season replay, with tuned parameters checked in.
  - *Restated after the fact:* one season cannot support that claim either way.
    The per-season SD of the optimizer-minus-greedy delta is ~10 TD, so a single
    replay is a coin flip dressed as evidence. The honest criterion is the mean
    paired delta across many seasons, reported with its standard error.
- *Verified against 2011-2025 (fifteen full replays):* the harness works and the
  answer is **negative** — the optimizer averages **-0.73 TD/season against
  greedy** (SE 2.39, won 8 of 15), and the null holds independently in both
  halves (2017-2025: -1.11; 2011-2016: -0.17). Both beat the random baseline by
  ~13 TD/season, so the projection model has real signal; the assignment layer
  does not add to it. A 3x5 sweep of `PRIOR_WEIGHT_GAMES` x `FUTURE_DISCOUNT`
  spans -3.0 to +1.9 TD/season — entirely inside noise — so **no parameters were
  changed**; the shipped values are confirmed, not tuned. The mechanism is
  measured rather than guessed: only 13% (QB) to 28% (FLEX) of projection
  variance comes from *which week* a player is used, so the quantity the
  assignment optimizes is a small share of what decides the season.
- *Decision:* the optimizer **stays, unchanged**. The result is a null, not a
  defeat — the true effect sits in roughly [-5, +4] TD/season and the data
  cannot narrow it — and the assignment solve is what produces `plan` and the
  season-cost ranking, which greedy cannot, and which Phase 3 needs.
- *The harness's own limits, which constrain every phase after this one:* the
  paired season-to-season SD is ~6 TD for a projection change and ~9 TD for a
  rule change, so with fifteen seasons only effects beyond **~±3 TD/season
  (model) or ~±5 (rule)** are resolvable. Two candidate improvements were built
  and rejected on that basis — per-type rates (no benefit) and per-slot base
  rates (+3.67 on the seasons it was found on, **+0.17 on a 2011-2016 holdout**).
  Treat any future small win from this harness as unproven until it survives a
  holdout.
- *Implementation notes:* point-in-time freezing lives in
  `projections.load_frames(..., as_of_week=W)` rather than a parallel loader, so
  live and replay share one code path. Backtests take their role multiplier from
  usage to date (`projections.usage_roles`), because `depth_charts` holds one
  snapshot per season with no week column — the stored 2025 chart is dated
  2026-03-14. Two ingest bugs blocked all of this and are fixed: `season_type`
  does not exist before 2025 (injuries), and depth charts up to 2024 use a
  different schema entirely (`week`/`depth_team` rather than `dt`/`pos_rank`).
  `config.override` plus `None`-resolved defaults in `build_matrix` make the
  sweep actually reach the solver; bound as import-time defaults, every cell
  silently returned the same number.

---

## Sequencing notes

- Phases 0–1 are the critical path to first real use; 2 and 3 layer on without
  reworking interfaces (the projection frame and the assignment matrix are the
  stable seams).
- The backtesting harness (Phase 4) is worth starting as soon as Phase 1
  exists — it's the fastest way to catch modeling mistakes, and requires no
  new data work thanks to the DB layer. *Borne out:* it was built after Phase 1
  and immediately showed the assignment layer is not paying for itself, which
  re-orders what is worth doing next — better projections (Phase 2) and win
  probability (Phase 3) both target larger effects than tuning the scheduler.
- Manual-entry commands (`record`, `opponent record`) are deliberately dumb and
  early; they're what make the state real while fancier layers are built.
