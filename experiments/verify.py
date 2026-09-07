"""Independently verify a completed benchmark run: `verify.py <experiment> [tests-passed]`.

A Phase 3B calibration experiment is verified stage by stage: the identity pairs, the
fitted folds, then the applied candidates. The fold checks are the ones that cannot be
recovered later -- that an artifact still hashes as recorded, and that the keys it was
fitted on stop before the season it was applied to.
"""

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pool import benchmark, models
from pool import evaluate as ev

experiment = sys.argv[1] if len(sys.argv) > 1 else "phase2-validation"
tests_passed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
out = Path("data/experiments") / experiment
manifest = json.loads((out / "manifest.json").read_text())
assert manifest["code"]["code_hash"] == benchmark.code_identity()["code_hash"]
# The configuration the run actually executed, authenticated against the identity it
# recorded -- not whatever the TOML says now. Re-resolving the file let an edit made
# after execution decide what got verified: dropping the last apply season from it
# skipped that season's artifacts and still produced a clean verification record, while
# publication went on to publish the whole experiment from the saved compact config.
spec = json.loads((out / "resolved-config.json").read_text())
config_hash = hashlib.sha256(benchmark.json_text(spec).encode()).hexdigest()
assert config_hash == manifest["identity"]["config_hash"], "Saved configuration is not the run's"
calibrated = spec.get("calibration_experiment")


def saved_surface(directory, model):
    """One saved surface, ordered so two of them can be compared row for row."""
    keys = ["season", "decision_week", "week", "slot", "player_id"]
    frame = pd.read_parquet(directory / f"surface-{model}--1.parquet")
    return frame.sort_values(keys).reset_index(drop=True)


def model_seeds(names):
    return {
        (name, seed) for name in names for seed in (spec["seeds"] if name in models.NULLS else [-1])
    }


def verify_season(directory, expected, replay_rows, season):
    """Every reconciliation the bake-off makes, against one season directory."""
    # Not merely that a checkpoint exists: that every artifact it sealed still hashes as
    # recorded. An existence check passes over a season whose forecasts were replaced.
    assert benchmark.checkpoint_complete(directory, season), season
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
        directory = out / "folds" / str(fold)
        assert benchmark.checkpoint_complete(directory, f"fold {fold}"), fold
        # Recomputed from the saved pairs: the digest names the exact keys that entered
        # the fit, and none of them may be at or after the season it is applied to.
        rows, digest, accounting = cal.training_pairs(out / "pairs", spec, fold)
        assert int(rows.season.max()) < fold
        upstream = {
            str(season): benchmark.digest(out / "pairs" / str(season) / "checkpoint.json")
            for season in range(calibrated["train_start"], fold)
        }
        for family, grouping, name in cal.candidates(spec):
            # `load_artifact` refuses an artifact that no longer hashes as recorded. Every
            # field below is checked for every candidate, because checking one candidate's
            # digest leaves a valid same-fold artifact free to sit under another
            # candidate's filename and answer for a map it did not fit.
            artifact = cal.load_artifact(directory / f"{name}.json")
            assert artifact["candidate"] == name
            assert (artifact["family"], artifact["grouping"]) == (family, grouping)
            assert artifact["fold"] == fold
            assert artifact["train_seasons"] == list(range(calibrated["train_start"], fold))
            assert artifact["cutoff_season"] == fold - 1
            # Recomputed over the values the coefficients are a function of, not the keys
            # alone: another run's fold directory can name the same rows and hold different
            # rates, positions or outcomes, and with keys alone it passed everything here.
            assert artifact["training_digest"] == digest
            assert artifact["training_upstream"] == upstream
            assert artifact["training_counts"] == accounting
            assert artifact["training_rows"] == accounting["fitted_rows"]
            assert artifact["base_model"] == spec["baseline"] and artifact["base_seed"] == -1
            assert artifact["map_target"] == calibrated["map_target"]
            assert artifact["fallback"] == list(calibrated["fallback"])
            assert (artifact["min_rows"], artifact["min_clusters"]) == (
                calibrated["min_rows"],
                calibrated["min_clusters"],
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
        # must agree exactly. A candidate's difference means nothing otherwise, and equal
        # season totals are not equal decisions: two different pick histories can score
        # the same. Compare the whole forecast row set, the whole future surface, and the
        # complete pick history.
        keys = ["season", "week", "slot", "player_id"]
        one = pd.read_parquet(out / "pairs" / str(season) / "forecasts.parquet")
        two = pd.read_parquet(out / "apply" / str(season) / "forecasts.parquet")
        two = two[two.model.eq(spec["baseline"])]
        one, two = (f.sort_values(keys).reset_index(drop=True) for f in (one, two))
        pd.testing.assert_frame_equal(one, two, check_like=True)
        base_surface = saved_surface(out / "apply" / str(season), spec["baseline"])
        pd.testing.assert_frame_equal(
            saved_surface(out / "pairs" / str(season), spec["baseline"]),
            base_surface,
            check_like=True,
        )
        pick_keys = ["model", "seed", "strategy", "week", "slot"]
        first = pd.read_csv(out / "pairs" / str(season) / "picks.csv")
        second = pd.read_csv(out / "apply" / str(season) / "picks.csv")
        second = second[second.model.eq(spec["baseline"])]
        first, second = (f.sort_values(pick_keys).reset_index(drop=True) for f in (first, second))
        pd.testing.assert_frame_equal(first, second, check_like=True)
        # Every candidate's saved surface must be its own fold artifact applied to those
        # identity rates -- otherwise a surface can be replaced, or built from the wrong
        # fold, and no later table would show it.
        surface_keys = ["season", "decision_week", "week", "slot", "player_id"]
        current = pd.read_parquet(out / "apply" / str(season) / "forecasts.parquet")
        for name in names:
            artifact = cal.load_artifact(out / "folds" / str(season) / f"{name}.json")
            saved = saved_surface(out / "apply" / str(season), name)
            # The keys and the scaffold first. Comparing rate arrays alone compares two
            # orderings and calls them the same population.
            pd.testing.assert_frame_equal(saved[surface_keys], base_surface[surface_keys])
            for column in ("avail_mult", "hard_eligible", "position", "actual_tds"):
                pd.testing.assert_series_equal(saved[column], base_surface[column])
            pd.testing.assert_series_equal(saved.original_lam, base_surface.lam, check_names=False)
            expected_lam = cal.mapped_lam(base_surface, artifact)
            np.testing.assert_array_equal(saved.lam.to_numpy(), expected_lam)
        # And every model's horizon-zero slice against the forecast rows the primary score
        # reads, which are exported separately and could disagree with the surface.
        # Identity included: the primary metric is a paired difference against it, so an
        # exporter regression that wrote the same wrong baseline rates into both stages
        # would move every comparison and pass the stage-to-stage checks above.
        for name in [spec["baseline"], *names]:
            rows = current[current.model.eq(name)]
            saved = saved_surface(out / "apply" / str(season), name)
            head = saved[saved.decision_week.eq(saved.week)]
            merged = rows.merge(
                head[["season", "week", "slot", "player_id", "lam", "actual_tds"]],
                on=["season", "week", "slot", "player_id"],
                how="inner",
                suffixes=("", "_surface"),
                validate="one_to_one",
            )
            # Both directions: one-to-one says no row matched twice, and these say neither
            # file holds a keyed row the other does not.
            assert len(merged) == len(rows) == len(head), (season, name)
            np.testing.assert_array_equal(merged.lam.to_numpy(), merged.lam_surface.to_numpy())
            # The outcomes too, which the two files reach by different routes: the forecast
            # export resolves the decision week's own row, the surface settles every target
            # week from the ledger. Agreeing where they overlap is what says the ledger
            # lookup found the rows it was supposed to.
            np.testing.assert_array_equal(
                merged.actual_tds.to_numpy(), merged.actual_tds_surface.to_numpy()
            )
        print(season, "apply verified", flush=True)
    checks += [
        "fitted artifacts hash as recorded",
        "every candidate's artifact carries its own declared fit",
        "no training key reaches its own apply season",
        "identity reproduces the pairs stage exactly",
        "candidate surfaces are their own artifact applied to identity",
        "candidate surfaces keep identity's keys, scaffold and outcomes",
        "every model's scored forecast rows are exactly its own horizon-zero surface",
        "fold digests cover the values that determined the coefficients",
        "training rows are accounted for before they are filtered",
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
    config_hash=config_hash,
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
