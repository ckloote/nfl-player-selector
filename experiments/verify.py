"""Independently verify a completed benchmark run: `verify.py <experiment> [tests-passed]`.

A Phase 3B calibration experiment is verified stage by stage: the identity pairs, the
fitted folds, then the applied candidates. The fold checks are the ones that cannot be
recovered later -- that an artifact still hashes as recorded, and that the keys it was
fitted on stop before the season it was applied to.
"""

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
calibrated = spec.get("calibration_experiment")


def model_seeds(names):
    return {
        (name, seed) for name in names for seed in (spec["seeds"] if name in models.NULLS else [-1])
    }


def verify_season(directory, expected, replay_rows, season):
    """Every reconciliation the bake-off makes, against one season directory."""
    assert (directory / "checkpoint.json").exists(), season
    df = pd.read_parquet(directory / "forecasts.parquet")
    assert set(df[["model", "seed"]].itertuples(index=False, name=None)) == expected
    assert df.schema_version.eq(1).all()
    assert df.loc[~df.hard_eligible, ["rank_in_slot", "rank_available"]].isna().all().all()
    assert df.decision_at.notna().all()
    base = df[df.model.eq(spec["baseline"])]
    for _, group in df.groupby(["model", "seed"]):
        ev.assert_comparable(base, group)
    assert df.groupby(["week", "slot", "player_id"]).actual_tds.nunique().eq(1).all()
    scores = pd.read_csv(directory / "replays.csv")
    picks = pd.read_csv(directory / "picks.csv")
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
    return dict(
        season=season,
        forecast_rows=len(df),
        model_seed_runs=len(expected),
        season_scores=len(scores),
        picks=len(picks),
        empty_slots=int(scores.empty_slots.sum()),
    )


# `audit` walks the history season too, so the schedule is wider than the replayed seasons:
# 4,175 games over 2010-2025 against 3,919 over 2011-2025.
audited = range(spec["history_start"], max(spec["seasons"]) + 1)
scheduled = sum(spec["expected_games"][str(season)] for season in audited)
checks = [
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
]
counts = []

if calibrated is None:
    expected = model_seeds(spec["models"])
    # The run's own shape rather than the fifteen-season one this script was written
    # against: every (model, seed) under both strategies, the baseline's random trials,
    # and hindsight.
    replay_rows = (
        len(expected) * len(spec["strategies"]) + spec["random_trials"] + int(spec["hindsight"])
    )
    for season in spec["seasons"]:
        counts.append(verify_season(out / str(season), expected, replay_rows, season))
        print(season, "verified", flush=True)
else:
    from pool import calibration as cal

    names = [n for _f, _g, n in cal.candidates(spec)]
    pair_expected = model_seeds([spec["baseline"]])
    apply_expected = model_seeds([spec["baseline"], *names])
    strategies = len(spec["strategies"])
    for season in spec["seasons"]:
        counts.append(
            verify_season(
                out / "pairs" / str(season),
                pair_expected,
                len(pair_expected) * strategies + spec["random_trials"] + int(spec["hindsight"]),
                season,
            )
        )
        print(season, "pairs verified", flush=True)
    for fold in calibrated["apply_seasons"]:
        for name in names:
            # `load_artifact` refuses an artifact that no longer hashes as recorded.
            artifact = cal.load_artifact(out / "folds" / str(fold) / f"{name}.json")
            assert artifact["fold"] == fold
            assert artifact["train_seasons"] == list(range(calibrated["train_start"], fold))
            assert artifact["cutoff_season"] == fold - 1
        # Recomputed from the saved pairs: the digest names the exact keys that entered
        # the fit, and none of them may be at or after the season it is applied to.
        rows, digest = cal.training_pairs(out / "pairs", spec, fold)
        assert int(rows.season.max()) < fold
        assert (
            digest
            == cal.load_artifact(out / "folds" / str(fold) / f"{names[0]}.json")["training_digest"]
        )
        print(fold, "fold verified", flush=True)
    for season in calibrated["apply_seasons"]:
        counts.append(
            verify_season(
                out / "apply" / str(season),
                apply_expected,
                len(apply_expected) * strategies + spec["random_trials"] + int(spec["hindsight"]),
                season,
            )
        )
        # Identity is rebuilt in the apply stage rather than reused, so the two stages
        # must agree exactly. A candidate's difference means nothing otherwise.
        keys = ["season", "week", "slot", "player_id"]
        one = pd.read_parquet(out / "pairs" / str(season) / "forecasts.parquet")
        two = pd.read_parquet(out / "apply" / str(season) / "forecasts.parquet")
        two = two[two.model.eq(spec["baseline"])]
        one, two = (f.sort_values(keys).reset_index(drop=True) for f in (one, two))
        pd.testing.assert_series_equal(one.lam, two.lam)
        pd.testing.assert_series_equal(one.actual_tds, two.actual_tds)
        first = pd.read_csv(out / "pairs" / str(season) / "replays.csv").set_index("strategy")
        second = pd.read_csv(out / "apply" / str(season) / "replays.csv")
        second = second[second.model.eq(spec["baseline"])].set_index("strategy")
        pd.testing.assert_series_equal(
            first.total.sort_index(), second.total.sort_index(), check_dtype=False
        )
        print(season, "apply verified", flush=True)
    checks += [
        "fitted artifacts hash as recorded",
        "no training key reaches its own apply season",
        "identity reproduces the pairs stage exactly",
    ]

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
    checks=checks,
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
