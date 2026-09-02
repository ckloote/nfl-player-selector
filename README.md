# NFL Player Selector

A decision-support tool for a season-long NFL touchdown pool: each week you pick
one QB, one RB, and one WR/TE, you score a point for every TD your picks throw or
score, and every player can only be used **once per season**. Most TDs at the end
of the season wins, winner take all.

This turns weekly picks into a season-long resource-allocation problem: *when* do
you spend your best players, given their schedules, and how should your risk
appetite change with your position on the leaderboard?

- [`docs/DESIGN.md`](docs/DESIGN.md) — the pool rules, the model, and the system architecture
- [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) — phased build plan and status

## Status

Phases 0 and 1 are implemented: data import, a matchup-adjusted projection
model, the season-long assignment optimizer, and a CLI that gives weekly picks.
Leaderboard-aware strategy (Phase 3) and backtesting (Phase 4) are not built yet.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pool refresh          # pulls nflverse data into data/pool.db
```

Data comes from [nflverse](https://github.com/nflverse) via `nflreadpy`. The
default season is 2026 (override with `--season` or `POOL_SEASON`); the prior
season is always used for the start-of-season prior.

## Weekly workflow

```bash
pool refresh                    # latest stats, lines, rosters, depth charts, injuries
pool recommend                  # picks for every open slot this week
pool record --rb "Saquon Barkley"        # lock a slot (others stay open)
pool record --qb "Herbert" --flex "Goedert"
pool recommend                  # re-optimizes the still-open slots
pool plan                       # rest-of-season assignment
pool players --pos RB           # projection table for a slot
pool picks / pool unrecord 3 QB
```

`recommend` shows, per slot, the optimizer's pick, its expected TDs, its pick
deadline (one hour before kickoff), and alternatives ranked by **season cost**:
how many projected TDs the rest-of-season plan loses if you take that player
now instead. When the recommended player plays before the Sunday slate, it says
whether to commit or hold for injury news.

## Tuning

Every modeling knob lives in `src/pool/config.py` (shrinkage weights, home edge,
depth-chart role multipliers, future discount, information premium).

## Development

```bash
.venv/bin/pytest
.venv/bin/ruff check src tests && .venv/bin/ruff format src tests
```
