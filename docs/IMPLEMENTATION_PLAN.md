# Implementation Plan

Phased so that the tool is *useful from Phase 1 onward* — each phase ships a
working improvement, and with the season starting in early September, Phase 1
is scoped to be usable for real picks within the first weeks.

Stack: Python 3.11+, `nfl_data_py` (nflverse data), SQLite, `scipy` (assignment
solver), `numpy`/`pandas`, `typer` (CLI), `pytest`.

---

## Phase 0 — Scaffolding & data foundation

**Goal:** clean project skeleton and reliable data in a local database.

- Project layout (`src/pool/`, `tests/`, `pyproject.toml`), lint/format
  (`ruff`), CI-friendly test setup.
- SQLite schema: imported tables (weekly player stats, schedules, team defense,
  Vegas lines, injuries/byes) and pool-state tables (my picks, opponent picks,
  standings).
- `pool refresh`: idempotent import of prior-season (2025) stats and the
  current (2026) schedule/lines via `nfl_data_py`; re-runnable all season for
  current-season data.
- **Done when:** `pool refresh` populates the DB from scratch; queries can list
  any player's 2025 weekly TD lines and any team's 2026 schedule; tests cover
  the import transforms.

## Phase 1 — Projections + optimizer + CLI (minimum useful product)

**Goal:** real weekly recommendations, EV-maximizing, before leaderboard logic.

- Projection v1: per-player Poisson TD rate from regressed prior-season rates ×
  opponent-defense multiplier × Vegas team-total scaling × home/away; byes and
  ruled-out players excluded. Exposed as `project(player, week)`.
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

## Phase 2 — In-season learning & data depth

**Goal:** the model revises itself as 2026 stats accumulate.

- Bayesian shrinkage blend of prior-season and current-season rates (prior
  weight ≈ 6–8 games, tunable).
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

- Backtesting harness: replay 2024/2025 week-by-week with data frozen at each
  pick deadline; compare cumulative TDs vs. baselines (greedy best-available,
  random top-10, perfect hindsight). Use it to tune the future-discount and
  prior weight.
- Calibration report: projected vs. actual TD distributions.
- Optional: simple local web dashboard (read-only view of plan/standings),
  pick-deadline reminders.
- **Done when:** a documented backtest shows the optimizer beating the greedy
  baseline over a full season replay, with tuned parameters checked in.

---

## Sequencing notes

- Phases 0–1 are the critical path to first real use; 2 and 3 layer on without
  reworking interfaces (`project()` and the assignment matrix are the stable
  seams).
- The backtesting harness (Phase 4) is worth starting as soon as Phase 1
  exists — it's the fastest way to catch modeling mistakes, and requires no
  new data work thanks to the DB layer.
- Manual-entry commands (`record`, `opponent record`) are deliberately dumb and
  early; they're what make the state real while fancier layers are built.
