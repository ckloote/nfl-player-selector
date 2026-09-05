"""Independently verify a completed benchmark run: `verify.py <experiment> [tests-passed]`."""

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from pool import benchmark, models
from pool import evaluate as ev

experiment = sys.argv[1] if len(sys.argv) > 1 else "phase2-validation"
tests_passed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
out = Path("data/experiments") / experiment
spec = benchmark.resolve(f"experiments/{experiment}.toml")
manifest = json.loads((out / "manifest.json").read_text())
assert manifest["code"]["code_hash"] == benchmark.code_identity()["code_hash"]
expected = {
    (name, seed)
    for name in spec["models"]
    for seed in (spec["seeds"] if name in models.NULLS else [-1])
}
# The run's own shape rather than the fifteen-season one this script was written against:
# every (model, seed) under both strategies, the baseline's random trials, and hindsight.
replay_rows = (
    len(expected) * len(spec["strategies"]) + spec["random_trials"] + int(spec["hindsight"])
)
# `audit` walks the history season too, so the schedule is wider than the replayed seasons:
# 4,175 games over 2010-2025 against 3,919 over 2011-2025.
audited = range(spec["history_start"], max(spec["seasons"]) + 1)
scheduled = sum(spec["expected_games"][str(season)] for season in audited)
counts = []
for season in spec["seasons"]:
    d = out / str(season)
    assert (d / "checkpoint.json").exists(), season
    df = pd.read_parquet(d / "forecasts.parquet")
    assert set(df[["model", "seed"]].itertuples(index=False, name=None)) == expected
    assert df.schema_version.eq(1).all()
    assert df.loc[~df.hard_eligible, ["rank_in_slot", "rank_available"]].isna().all().all()
    assert df.decision_at.notna().all()
    base = df[df.model.eq(spec["baseline"])]
    for _, group in df.groupby(["model", "seed"]):
        ev.assert_comparable(base, group)
    assert df.groupby(["week", "slot", "player_id"]).actual_tds.nunique().eq(1).all()
    scores = pd.read_csv(d / "replays.csv")
    picks = pd.read_csv(d / "picks.csv")
    assert len(scores) == replay_rows
    selected = picks[picks.player_id.notna()]
    assert not selected.duplicated(["model", "seed", "strategy", "player_id"]).any()
    sums = picks.groupby(["model", "seed", "strategy"]).actual.sum()
    pd.testing.assert_series_equal(
        sums.sort_index(),
        scores.set_index(["model", "seed", "strategy"]).total.sort_index(),
        check_names=False,
        check_dtype=False,
    )
    used = selected.groupby(["model", "seed", "strategy"]).player_id.nunique()
    pd.testing.assert_series_equal(
        used.sort_index(),
        scores.set_index(["model", "seed", "strategy"]).unique_players.sort_index(),
        check_names=False,
        check_dtype=False,
    )
    for strategy in ("greedy", "optimizer"):
        chosen = {
            (r.model, r.seed, r.week, r.player_id)
            for r in selected[selected.strategy.eq(strategy)].itertuples()
        }
        actual = {
            (r.model, r.seed, r.week, r.player_id)
            for r in df[df["picked_" + strategy]].itertuples()
        }
        assert chosen == actual, (season, strategy)
    counts.append(
        dict(
            season=season,
            forecast_rows=len(df),
            model_seed_runs=len(expected),
            season_scores=len(scores),
            picks=len(picks),
            empty_slots=int(scores.empty_slots.sum()),
        )
    )
    print(season, "verified", flush=True)
coverage = pd.read_csv(out / "coverage.csv")
assert len(coverage) == scheduled and coverage.complete.eq(1).all()
c = sqlite3.connect(f"file:{out}/dataset.sqlite?mode=ro", uri=True)
gaps = c.execute(
    """SELECT g.game_id FROM games g LEFT JOIN player_weeks p ON p.game_id=g.game_id
 WHERE g.game_type='REG' AND g.season BETWEEN ? AND ?
 GROUP BY g.game_id HAVING COUNT(DISTINCT p.team) != 2""",
    (min(audited), max(audited)),
).fetchall()
assert not gaps
c.close()
validation = dict(
    schema_version=1,
    implementation_commit=benchmark.code_identity()["revision"],
    source_hash=manifest["code"]["code_hash"],
    dataset_hash=manifest["dataset_hash"],
    seasons=counts,
    scheduled_games=scheduled,
    complete_games=int(coverage.complete.eq(1).sum()),
    player_stat_game_gaps=0,
    pytest_passed=tests_passed,
    lint="Ruff check passed",
    format="Ruff format --check passed",
    checks=[
        "all configured models and seeds",
        "candidate/eligibility/depletion masks",
        "hard exclusions unranked",
        "explicit decision timestamps",
        "consistent finalized actuals",
        "no player reuse",
        "pick sums equal season scores",
        "unique-player counts",
        "forecast pick indicators match replays",
        "game coverage",
        "both teams player-stat coverage",
    ],
)
benchmark.write_json(out / "verification.json", validation)
print(
    json.dumps(
        {
            key: sum(row[key] for row in counts)
            for key in ("forecast_rows", "model_seed_runs", "season_scores", "picks", "empty_slots")
        },
        sort_keys=True,
    ),
    flush=True,
)
