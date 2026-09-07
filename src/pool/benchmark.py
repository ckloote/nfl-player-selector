"""Configuration-driven, checkpointed research runs and reports from saved artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import shutil
import sqlite3
import subprocess
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest, config, db, ingest, models, projections, scoring, snapshots
from . import evaluate as ev

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[2]
ASSUMPTIONS = [
    "Scoring: one credit for each touchdown scored and each credited passing touchdown thrown; "
    "negated plays and conversions excluded; complete game coverage required.",
    "Historical: current-season closing lines for weeks >= decision week are masked before "
    "all team/league averages. Week one uses the shipped league fallback.",
    "Stats through W-1; weekly roster and injury rows through W; usage roles. Final schedule "
    "revisions, report timing within a week and later stat corrections remain approximations.",
    "One decision per week immediately before the first confirmed pick deadline. Unknown "
    "current-week kickoffs are hard exclusions; future planning estimates remain usable.",
    "Snapshot observations represent import availability, not backdated source publication. "
    "Finalized outcome scoring is separate from archived projection inputs.",
    "Ranking population: pool-position players listed active on their own team's latest "
    "weekly roster snapshot at or before the decision week, before assignment pruning; hard "
    "exclusions unranked. Zero estimates eligible. Ties use player ID.",
    "Common-pool depletion uses the selected deterministic baseline greedy history. Top-k "
    "diagnostics are TDs per ranked candidate, never achieved season scores.",
    "Slot-week means are averaged within season. Seeds are averaged within season before "
    "uncertainty across seasons; seed SD is reported separately. Empty cells are reported.",
    "Each greedy/optimizer replay has its own no-reuse history. Random-top-10 strategy trials "
    "use shipped forecasts and differ from shuffled projection models. Hindsight is retrospective.",
    "All seasons and era summaries are retrospective; neither era is an untouched holdout. "
    "Production constants are fixed; no calibration correction or tuning is applied.",
    "Research scoring includes the checked correction in experiments/scoring-corrections.json: "
    "duplicate rushing touchdowns in 2011_13_DET_NO reconciled to the official Saints game report.",
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_text(value):
    return json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json_text(value))
    tmp.replace(path)


def code_identity():
    paths = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "pyproject.toml", ROOT / "uv.lock"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(
        ["git", "diff", "HEAD", "--", "src", "pyproject.toml", "uv.lock"], cwd=ROOT
    )
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    return dict(
        revision=revision,
        source_hashes=hashes,
        code_hash=hashlib.sha256(json_text(hashes).encode()).hexdigest(),
        dirty=bool(status.strip()),
        dirty_diff_hash=hashlib.sha256(diff).hexdigest(),
        working_tree_status=status,
    )


def constants():
    return {
        k: sorted(v) if isinstance(v, (set, frozenset)) else v
        for k, v in vars(config).items()
        if k.isupper() and k not in ("DB_PATH", "DEFAULT_SEASON", "FRESHNESS_HOURS")
    }


def resolve(path):
    with Path(path).open("rb") as stream:
        spec = tomllib.load(stream)
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported benchmark config schema")
    ev.validate_models(spec["models"], spec["baseline"])
    spec["seeds"] = (
        models.parse_seeds(spec["seeds"]) if isinstance(spec["seeds"], str) else spec["seeds"]
    )
    if not spec["seeds"] or any(s < 0 or not isinstance(s, int) for s in spec["seeds"]):
        raise ValueError("Benchmark seeds must be a nonempty list of nonnegative integers")
    spec["seeds"] = sorted(set(spec["seeds"]))
    spec["seasons"] = sorted(set(spec["seasons"]))
    if not spec["seasons"] or spec["history_start"] > min(spec["seasons"]) - 1:
        raise ValueError("Complete prior-season history is required")
    spec["role_source"] = projections.validate_policy(
        spec["input_policy"], spec.get("role_source"), spec.get("vegas_horizon")
    )
    if spec.get("constants") != "shipped":
        raise ValueError("This validation runner requires constants = 'shipped'")
    if set(spec["strategies"]) != {"greedy", "optimizer"}:
        raise ValueError("Benchmark requires both greedy and optimizer replays")
    if spec.get("scoring_corrections"):
        spec["resolved_corrections"] = json.loads((ROOT / spec["scoring_corrections"]).read_text())
    spec["resolved_decision_times"] = decision_time_records(spec)
    spec["workers"] = int(spec.get("workers", 1))
    if not 1 <= spec["workers"] <= 4:
        raise ValueError("workers must be between 1 and 4")
    if "calibration_experiment" in spec:
        resolve_calibration(spec, path)
    spec["production_constants"] = constants()
    return spec


def resolve_calibration(spec, path=None):
    """Validate the Phase 3B experiment declaration, before a single pair is read.

    Everything checked here ends up in the specification dict, and the specification is
    what the run identity hashes, so a fold schedule, a family, a fallback or a margin
    cannot be changed after the fact without invalidating every checkpoint. The margins
    especially: the plan says a missing margin blocks execution, because a threshold
    chosen once the estimate is on screen is not a threshold.
    """
    from . import calibration as cal

    c = spec["calibration_experiment"]
    if not spec.get("export_future_forecasts"):
        raise ValueError("A calibration experiment fits the future surface; export it")
    if spec["models"] != [spec["baseline"]]:
        raise ValueError("A calibration experiment runs one base model: its own baseline")
    # A candidate keeps the shipped scaffold and replaces only its `lam`, so a different
    # baseline would fit coefficients on one model's rates and apply them to another's.
    if spec["baseline"] != cal.IDENTITY:
        raise ValueError(
            f"A calibration experiment maps the {cal.IDENTITY!r} scaffold; "
            f"this run declares baseline {spec['baseline']!r}"
        )
    unknown = sorted(set(c.get("families", [])) - set(cal.FAMILIES))
    if not c.get("families") or unknown:
        raise ValueError(
            f"Unknown calibration families {unknown}; choose from {list(cal.FAMILIES)}"
        )
    unknown = sorted(set(c.get("groupings", [])) - set(cal.GROUPINGS))
    if not c.get("groupings") or unknown:
        raise ValueError(
            f"Unknown calibration groupings {unknown}; choose from {list(cal.GROUPINGS)}"
        )
    c["apply_seasons"] = sorted(set(c.get("apply_seasons", [])))
    outside = [y for y in c["apply_seasons"] if y not in set(spec["seasons"])]
    if not c["apply_seasons"] or outside:
        raise ValueError(f"Apply seasons {outside} are not replayed seasons of this run")
    if c.get("train_start") is None or c["train_start"] <= spec["history_start"]:
        raise ValueError("train_start must be a replayed season after history_start")
    if min(c["apply_seasons"]) <= c["train_start"]:
        raise ValueError("The first apply season must leave at least one training season")
    missing = [
        y for y in range(c["train_start"], max(c["apply_seasons"])) if y not in spec["seasons"]
    ]
    if missing:
        raise ValueError(f"Training seasons {missing} are not replayed by this run")
    for name, expected in (
        ("map_target", cal.MAP_TARGET),
        ("fit_horizons", cal.FIT_HORIZONS),
        ("row_weight", cal.ROW_WEIGHT),
        ("cluster", cal.CLUSTER),
        ("zero_rate_policy", cal.ZERO_RATE_POLICY),
        # Declared method. Each of these was previously accepted unread while execution
        # was hard-coded, so a specification could name an estimand the run did not
        # compute and nothing would say so.
        ("refit", cal.REFIT),
        ("primary_metric", cal.PRIMARY_METRIC),
        ("primary_population", cal.PRIMARY_POPULATION),
        ("uncertainty", cal.UNCERTAINTY),
        ("multiplicity", cal.MULTIPLICITY),
        ("zero_variance_policy", cal.ZERO_VARIANCE_POLICY),
    ):
        if c.get(name) != expected:
            raise ValueError(
                f"calibration_experiment.{name} must be {expected!r}; "
                f"this run declares {c.get(name)!r}"
            )
    if list(c.get("fallback", ())) != list(cal.FALLBACK):
        raise ValueError(f"calibration_experiment.fallback must be {list(cal.FALLBACK)}")
    for name in ("min_rows", "min_clusters"):
        if int(c.get(name, 0)) <= 0:
            raise ValueError(f"calibration_experiment.{name} must be a positive integer")
    if not c.get("specification_date"):
        raise ValueError("A calibration experiment must be dated before it is fitted")
    margins = c.get("margins") or {}
    # `isinstance(True, int)` and `isinstance(nan, float)` are both true, and neither is a
    # threshold. A margin has to be a finite real number to be compared against anything.
    absent = [
        k
        for k in cal.REQUIRED_MARGINS
        if isinstance(margins.get(k), bool)
        or not isinstance(margins.get(k), (int, float))
        or not math.isfinite(margins[k])
    ]
    if absent:
        raise ValueError(
            f"calibration_experiment.margins is missing {absent}; a missing margin blocks the run"
        )
    # A finite number is not yet a threshold. A coverage floor outside [0, 1] is either
    # vacuous or unreachable, and a negative improvement or loss margin inverts the
    # comparison it appears in, so each would freeze a condition that cannot do its job.
    outside = [k for k in cal.COVERAGE_MARGINS if not 0.0 <= margins[k] <= 1.0]
    negative = [k for k in cal.NONNEGATIVE_MARGINS if margins[k] < 0.0]
    if outside or negative:
        raise ValueError(
            f"calibration_experiment.margins is out of domain: coverage floors {outside} must "
            f"lie in [0, 1] and margins {negative} must not be negative"
        )
    promotion = c.get("promotion") or {}
    wrong = [k for k, allowed in cal.PROMOTION.items() if promotion.get(k) not in allowed]
    if wrong:
        raise ValueError(
            f"calibration_experiment.promotion declares unsupported {wrong}; the rule is frozen "
            "with the specification, so it must be data rather than a comment"
        )
    # A name already registered as a calibrated candidate is this runner's own, from an
    # earlier run in the same process. Only a shipped model is a collision.
    collisions = [
        n
        for _f, _g, n in cal.candidates(spec)
        if n in models.BUILDERS and n not in models.CALIBRATED
    ]
    if collisions:
        raise ValueError(f"Calibration candidates {collisions} would shadow registered models")
    # The primary metric is the all-eligible current-week population, not the tail the
    # bake-off scores, so the floor travels in the specification rather than defaulting.
    spec["deviance_floor"] = float(c.get("deviance_floor", 0.0))
    spec["advice_sensitivity"] = bool(c.get("advice_sensitivity", True))
    # The parsed keys above are the ones the run executes; the file also carries the prose
    # that says what they mean. Hashing the text puts that prose inside the run identity
    # too, so an edit to the reasoning starts a new experiment rather than reinterpreting
    # a finished one.
    if path is not None:
        c["specification_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decision_time_records(spec):
    """Read the decision CSV's contents into the specification, before anything runs.

    Resume compared a path. Editing the file in place left the configuration hash
    untouched, so a run could be continued against timestamps it had never seen, and
    the workers -- separate processes, each re-reading the path relative to its own
    working directory -- were the first thing to look at the file at all. The contents
    belong to the run identity, the same way the reviewed scoring correction does.
    """
    path = spec.get("decision_times")
    snapshot_policy = spec["input_policy"] == "snapshots"
    if snapshot_policy and path is None:
        raise ValueError("input_policy = 'snapshots' requires decision_times")
    if path is not None and not snapshot_policy:
        raise ValueError("decision_times requires input_policy = 'snapshots'")
    if path is None:
        return []
    times = snapshots.decision_times(ROOT / path)
    return [
        dict(season=int(season), week=int(week), decision_at=stamp)
        for (season, week), stamp in sorted(times.items())
    ]


def frozen_decision_times(spec):
    """The frozen mapping, in the shape the replay layer expects."""
    records = spec.get("resolved_decision_times") or []
    return {(r["season"], r["week"]): r["decision_at"] for r in records} or None


def require_decision_times(conn, spec):
    """Every week the run will replay needs a timestamp, checked before any worker.

    `weekly_inputs` also refuses a missing week, but only once a worker has reached
    that season -- after the manifest is written and hours into a fifteen-season run.
    """
    times = frozen_decision_times(spec)
    if times is None:
        return
    missing = [
        (season, week)
        for season in spec["seasons"]
        for week in projections.available_weeks(conn, season)
        if (season, week) not in times
    ]
    if missing:
        raise ValueError(f"decision_times is missing timestamps for {missing}")


def audit(conn, spec):
    rows = []
    for season in range(spec["history_start"], max(spec["seasons"]) + 1):
        cov = scoring.coverage(conn, season).sort_values("game_id")
        expected = spec["expected_games"].get(str(season))
        if expected is None or len(cov) != expected:
            raise ValueError(
                f"Schedule audit failed for {season}: {len(cov)} games, expected {expected}"
            )
        scoring.require_complete(conn, season, projections.available_weeks(conn, season))
        if not conn.execute("SELECT 1 FROM player_weeks WHERE season = ?", (season,)).fetchone():
            raise ValueError(f"Missing required {season} player history")
        rows.append(cov)
    return pd.concat(rows, ignore_index=True)


def apply_corrections(conn, spec, log=print):
    """Apply versioned research corrections only when all documented preconditions match.

    No scoring waivers: the corrected scorer ledger must equal independently sourced
    player totals. Preserve the original observations and append a correction observation.
    """
    from . import snapshots

    for fix in spec.get("resolved_corrections", {}).get("corrections", []):
        if not spec["history_start"] <= fix["season"] <= max(spec["seasons"]):
            continue
        gid = fix["game_id"]
        game = conn.execute("SELECT * FROM games WHERE game_id = ?", (gid,)).fetchone()
        result = conn.execute("SELECT * FROM game_results WHERE game_id = ?", (gid,)).fetchone()
        if not game or not result:
            continue  # no import yet; the final coverage audit still blocks
        credits = db.read_df(conn, "SELECT * FROM touchdown_credits WHERE game_id = ?", (gid,))
        totals = credits.groupby("player_id").size().to_dict()
        target = fix["corrected_result"]
        if result["complete"] and [result["home_score"], result["away_score"]] == target:
            if totals != fix["expected_player_totals"]:
                raise ValueError(
                    f"Source changed: scorer totals no longer match correction {fix['id']}"
                )
            continue
        keys = set(credits[["play_id", "player_id", "kind"]].itertuples(index=False, name=None))
        expected = {tuple(k) for k in fix["remove_credits"] + fix["retained_credits"]}
        if (
            [game["home_score"], game["away_score"]] != target
            or [result["home_score"], result["away_score"]] != fix["expected_result"]
            or result["reason"] != "terminal scores do not match schedule"
            or len(credits) != fix["expected_credits"]
            or not expected <= keys
        ):
            raise ValueError(
                f"Source changed: preconditions failed for research correction {fix['id']}"
            )
        with db.transaction(conn):
            conn.executemany(
                "DELETE FROM touchdown_credits WHERE game_id=? AND play_id=? "
                "AND player_id=? AND kind=?",
                [(gid, *k) for k in fix["remove_credits"]],
            )
            corrected = db.read_df(
                conn, "SELECT * FROM touchdown_credits WHERE game_id = ?", (gid,)
            )
            if corrected.groupby("player_id").size().to_dict() != fix["expected_player_totals"]:
                raise ValueError(f"Scorer reconciliation failed for {fix['id']}")
            conn.execute(
                "UPDATE game_results SET home_score=?, away_score=?, complete=1, "
                "reason=? WHERE game_id=?",
                (*target, "complete; research correction " + fix["id"], gid),
            )
            snapshots.archive(conn, fix["season"], "touchdowns")
        log(f"Applied documented research correction: {fix['id']}")


def prepare_dataset(output, spec, log):
    frozen = output / "dataset.sqlite"
    if frozen.exists():
        return frozen
    research = output / "research.db"
    conn = db.connect(research)
    try:
        apply_corrections(conn, spec, log)
        for season in range(spec["history_start"] + 1, max(spec["seasons"]) + 2, 2):
            pair = [season - 1, season]
            if all(
                len(scoring.coverage(conn, yr)) == spec["expected_games"].get(str(yr))
                and scoring.coverage(conn, yr).complete.all()
                for yr in pair
            ):
                continue
            log(f"Backfilling research data {season - 1}–{season}")
            ingest.refresh(conn, season, log=log)
        apply_corrections(conn, spec, log)
        audit(conn, spec)
        tmp = output / "dataset.sqlite.tmp"
        destination = sqlite3.connect(tmp)
        conn.backup(destination)
        destination.close()
        tmp.replace(frozen)
    finally:
        conn.close()
    return frozen


def input_metadata(conn, spec):
    hashes = {}
    for table in (
        "games",
        "player_weeks",
        "rosters",
        "injuries",
        "depth_charts",
        "touchdown_credits",
        "game_results",
    ):
        frame = db.read_df(conn, f"SELECT * FROM {table}")
        frame = frame.sort_values(list(frame.columns), na_position="first")
        hashes[table] = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
    provenance = db.read_df(conn, "SELECT * FROM feed_status ORDER BY season, feed").to_dict(
        "records"
    )
    return dict(
        table_hashes=hashes,
        feed_provenance=provenance,
        observation_count=conn.execute("SELECT COUNT(*) FROM input_observations").fetchone()[0],
    )


def save_frame(path, frame):
    frame = frame.copy()
    frame.attrs = {}
    if "schema_version" not in frame:
        frame.insert(0, "schema_version", SCHEMA_VERSION)
    keys = [
        k
        for k in (
            "fold",
            "season",
            "model",
            "seed",
            "candidate",
            "group",
            "strategy",
            "population",
            "horizon",
            "availability",
            "decision_week",
            "week",
            "slot",
            "player_id",
            "rank",
            "k",
            "bin",
        )
        if k in frame
    ]
    if keys:
        frame = frame.sort_values(keys, na_position="first").reset_index(drop=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(tmp, index=False)
    else:
        frame.to_csv(tmp, index=False, float_format="%.12g")
    tmp.replace(path)


CALIBRATION_COLUMNS = (
    "intercept",
    "slope",
    "se_intercept",
    "se_slope",
    "slope_lo",
    "slope_hi",
    "n",
    "clusters",
    "dropped_zero_lam",
)


def write_checkpoint(directory):
    """Seal a completed stage directory by hashing everything it wrote.

    Partial writes are excluded, not recorded. Every writer here renames a `.tmp` sibling
    into place, so an interruption can leave one behind -- including `checkpoint.json.tmp`,
    which this function's own write then renames away. Recording it produced a checkpoint
    naming a file that could not exist, and the next resume rejected the recovered run.
    """
    files = {
        p.name: digest(p)
        for p in sorted(directory.iterdir())
        if p.name != "checkpoint.json" and p.suffix != ".tmp"
    }
    write_json(directory / "checkpoint.json", dict(schema_version=SCHEMA_VERSION, files=files))


def checkpoint_complete(directory, label):
    """True if `directory` holds a checkpoint whose artifacts still hash as recorded."""
    checkpoint = directory / "checkpoint.json"
    if not checkpoint.exists():
        return False
    saved = json.loads(checkpoint.read_text())
    if any(
        not (directory / name).exists() or digest(directory / name) != checksum
        for name, checksum in saved["files"].items()
    ):
        raise ValueError(f"Checkpoint artifact hash mismatch for {label}")
    return True


def resolve_surface_outcomes(surfaces, forecasts, actuals, scored, played):
    """Settle each future forecast's outcome from the ledger, not from pool membership.

    A surface row says "at decision week w, this player was projected for week t". What
    happened in week t is a fact about week t. Reading it off week t's own forecast row
    made it conditional on the player still being a candidate then, and he need not be:
    on 2016's apply season 9,502 eligible rows -- 444 at horizon 1 rising to 5,772 at
    seven or more -- had no outcome for that reason alone, every week was completely
    scored, and not one of them had scored a touchdown. Dropping them removed a
    population that is almost entirely zeros, which biases the level upward rather than
    merely narrowing it: fold 2016's scale moved 0.9413 to 0.9355 once they were back.

    Absence is zero only where the week is completely scored, the rule `capture` states
    and the current-week export already follows. `in_target_pool` keeps the membership
    question separate rather than letting it masquerade as a missing outcome.
    """
    weeks = surfaces.week.astype(int)
    complete = weeks.isin(scored).to_numpy()
    pairs = list(zip(weeks, surfaces.player_id, strict=True))
    resolved = np.array([actuals.get(pair, 0.0) for pair in pairs], dtype=float)
    out = surfaces.copy()
    out["outcome_complete"] = complete
    out["actual_tds"] = np.where(complete, resolved, np.nan)
    out["played"] = [pair in played for pair in pairs]
    current = set(zip(forecasts.week.astype(int), forecasts.player_id, strict=True))
    out["in_target_pool"] = [pair in current for pair in pairs]
    return out


def evaluate_season(conn, season, spec, directory, log, candidates=None):
    directory.mkdir(parents=True, exist_ok=True)
    chosen = {name: models.get(name) for name in spec["models"]}
    for name, builder in (candidates or {}).items():
        # `season_frames` validates every name against the model registry, and a fitted
        # candidate is not in it until its fold has been fitted -- which is inside this
        # process, because a builder closing over an artifact cannot be pickled here.
        models.register_calibrated(name, builder)
    chosen |= candidates or {}
    base_frames = None
    advice_parts = []
    actuals = backtest.actual_tds(conn, season)
    # A future forecast's outcome is a fact about the target week, and the ledger settles
    # it whether or not the player is still in that week's candidate pool. Both are read
    # once here, outside the model loop, because every model shares them.
    scored = set(backtest.scored_weeks(conn, season))
    played = ev.played_pairs(conn, season)
    reference = None
    metrics = {
        name: []
        for name in (
            "ranking",
            "paired_ranking",
            "calibration",
            "reliability",
            "deviance",
            "spearman",
            "replays",
            "picks",
        )
    }
    forecast_parts, provenance = [], []
    # The frozen contents carried in the specification, never a re-read of the path:
    # a worker is a separate process with its own working directory, and by the time it
    # runs, the file on disk is no longer part of anything the run identity covers.
    times = frozen_decision_times(spec)
    for model, seed, common, loaded, frames in ev.season_frames(
        conn,
        season,
        chosen,
        baseline=spec["baseline"],
        seeds=spec["seeds"],
        role_source=spec["role_source"],
        input_policy=spec["input_policy"],
        vegas_horizon=spec.get("vegas_horizon"),
        decision_times=times,
    ):
        df = ev.forecasts(
            conn,
            season,
            model=model,
            seed=seed,
            frames=frames,
            weeks=list(frames),
            available_from=common,
            strategies=(),
            input_policy=spec["input_policy"],
        )
        if reference is None:
            reference = df.copy()
            for week, frame in loaded.items():
                provenance.append(
                    dict(week=week, decision_at=frame.decision_at, inputs=frame.provenance)
                )
        ev.assert_comparable(reference, df)
        for strategy in spec["strategies"]:
            replay = backtest.replay(frames, actuals, season, strategy)
            row, picks = ev.replay_records(replay, model, seed)
            metrics["replays"].append(pd.DataFrame([row]))
            metrics["picks"].append(pd.DataFrame(picks))
            picked = {(p.week, p.player_id) for p in replay.picks if p.player_id}
            df[f"picked_{strategy}"] = [
                (w, p) in picked for w, p in zip(df.week, df.player_id, strict=True)
            ]
        for rank in spec["ranking_pools"]:
            metrics["ranking"].append(ev.ranking_seasons(df, spec["ranking_k"], rank))
            if model != spec["baseline"]:
                paired = pd.concat([reference, df], ignore_index=True)
                paired.attrs["cells"] = df.attrs["cells"]
                for k in spec["ranking_k"]:
                    metrics["paired_ranking"].append(
                        ev.paired_ranking_seasons(paired, spec["baseline"], k, rank)
                    )
        if spec["calibration"]:
            # Artifact schema 1 columns only. The guarded fit's status, reason and
            # zero-rate breakdown belong to the Phase 3A diagnostics export; widening a
            # published artifact's schema in place would leave two different files both
            # claiming to be schema 1.
            fit = ev.calibration(df)
            metrics["calibration"].append(
                pd.DataFrame(
                    [
                        dict(
                            model=model,
                            seed=seed,
                            season=season,
                            **{k: fit[k] for k in CALIBRATION_COLUMNS},
                        )
                    ]
                )
            )
            rel = ev.reliability(df)
            rel["bin"] = rel["bin"].astype(str)
            metrics["reliability"].append(rel.assign(model=model, seed=seed, season=season))
            paired = (
                df if model == spec["baseline"] else pd.concat([reference, df], ignore_index=True)
            )
            dev = ev.deviance_seasons(
                paired, spec["baseline"], spec.get("deviance_floor", ev.DEVIANCE_FLOOR)
            )
            metrics["deviance"].append(dev[dev.model == model])
            metrics["spearman"].append(ev.spearman_seasons(df))
        forecast_parts.append(df)
        if model == spec["baseline"] and seed == -1:
            base_frames = frames
        if spec.get("advice_sensitivity"):
            from . import calibration as cal

            advice_parts.append(cal.advice_rows(frames, common, model, seed, season))
        if spec["export_future_forecasts"]:
            columns = ["player_id", "week", "slot", "game_id", "lam", "avail_mult", "hard_eligible"]
            if spec.get("calibration_experiment"):
                # The role the frame was built with, at the decision week that built it.
                # Recovering it later by joining the decision week's own forecast row
                # loses every player who has no such row -- a bye week is exactly that --
                # and those rows then trained pooled while being applied by position.
                # Only in a calibration experiment, for the reason `original_lam` is: two
                # different files must not both claim artifact schema 1.
                columns.append("position")
            surfaces = pd.concat(
                [
                    f[columns].assign(decision_week=w, model=model, seed=seed, season=season)
                    for w, f in frames.items()
                ],
                ignore_index=True,
            )
            if spec.get("calibration_experiment"):
                # The mapped rate is `lam` and the rate it was mapped from travels beside
                # it, so a saved surface says what the map did without rejoining the
                # identity export to find out.
                if candidates and model in candidates:
                    if base_frames is None:
                        raise ValueError(
                            "A calibrated candidate was built before its identity surface; "
                            "the baseline must be evaluated first"
                        )
                    keys = ["decision_week", "week", "slot", "player_id"]
                    raw = pd.concat(
                        [
                            f[["week", "slot", "player_id", "lam"]].assign(decision_week=w)
                            for w, f in base_frames.items()
                        ],
                        ignore_index=True,
                    ).rename(columns={"lam": "original_lam"})
                    surfaces = surfaces.merge(raw, on=keys, how="left", validate="one_to_one")
                else:
                    surfaces["original_lam"] = surfaces.lam
                surfaces = resolve_surface_outcomes(surfaces, df, actuals, scored, played)
            save_frame(directory / f"surface-{model}-{seed}.parquet", surfaces)
        if model == spec["baseline"]:
            for trial in range(spec["random_trials"]):
                trial_seed = spec["random_seed"] + trial
                replay = backtest.replay(
                    frames,
                    actuals,
                    season,
                    "random",
                    rng=np.random.default_rng(trial_seed),
                    top_n=spec["random_top_n"],
                )
                row, picks = ev.replay_records(replay, model, trial_seed)
                metrics["replays"].append(pd.DataFrame([row]))
                metrics["picks"].append(pd.DataFrame(picks))
            if spec["hindsight"]:
                row, picks = ev.replay_records(
                    backtest.hindsight(conn, season, list(frames), actuals), "hindsight", -1
                )
                metrics["replays"].append(pd.DataFrame([row]))
                metrics["picks"].append(pd.DataFrame(picks))
        if seed in (-1, max(spec["seeds"])):
            log(f"  {season} {model}: completed through seed {seed}")
    save_frame(directory / "forecasts.parquet", pd.concat(forecast_parts, ignore_index=True))
    for name, parts in metrics.items():
        if parts:
            save_frame(directory / f"{name}.csv", pd.concat(parts, ignore_index=True))
    write_json(directory / "input-provenance.json", provenance)
    if advice_parts:
        save_frame(directory / "advice.csv", pd.concat(advice_parts, ignore_index=True))
    write_checkpoint(directory)


def markdown_table(frame, columns):
    # Avoid an optional tabulate dependency in the reproducible report path.
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        cells = [
            "NA" if pd.isna(v) else f"{v:.4f}" if isinstance(v, float) else str(v) for v in row
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def reports(output, spec, manifest, config_path=None):
    """Generate compact exports and an evaluation report from saved season metrics."""
    compact = output / "compact"
    compact.mkdir(exist_ok=True)
    names = (
        "ranking",
        "paired_ranking",
        "calibration",
        "reliability",
        "deviance",
        "spearman",
        "replays",
        "picks",
    )
    data = {}
    for name in names:
        paths = [output / str(s) / f"{name}.csv" for s in spec["seasons"]]
        data[name] = pd.concat([pd.read_csv(p) for p in paths if p.exists()], ignore_index=True)
        save_frame(compact / f"{name}.csv", data[name])
    eras = {"all retrospective": [min(spec["seasons"]), max(spec["seasons"])], **spec["eras"]}
    summaries = {
        name: [] for name in ("ranking_summary", "paired_ranking_summary", "replay_summary")
    }
    for era, (lo, hi) in eras.items():
        rank = data["ranking"][data["ranking"].season.between(lo, hi)]
        pair = data["paired_ranking"][data["paired_ranking"].season.between(lo, hi)]
        replay = data["replays"][data["replays"].season.between(lo, hi)]
        summaries["ranking_summary"].append(
            ev.summarize_seeds(rank, "actual", ("model", "rank", "k")).assign(era=era)
        )
        summaries["paired_ranking_summary"].append(
            ev.summarize_seeds(pair, "delta", ("model", "rank", "k")).assign(era=era)
        )
        summaries["replay_summary"].append(
            ev.replay_summary(replay, spec["baseline"]).assign(era=era)
        )
    for name, parts in summaries.items():
        data[name] = pd.concat(parts, ignore_index=True)
        save_frame(compact / f"{name}.csv", data[name])
    write_json(compact / "manifest.json", manifest)
    shutil.copy2(output / "resolved-config.json", compact / "resolved-config.json")
    shutil.copy2(output / "coverage.csv", compact / "coverage.csv")
    for name, text in render_reports(data, spec, manifest, output, config_path).items():
        (compact / name).write_text(text)
    return compact


def calibration_reports(output, spec, manifest, config_path=None):
    """Compact exports and a factual report for a Phase 3B run, from saved artifacts only.

    Nothing here selects a candidate. The promotion rule is in the frozen specification
    and the decision that applies it is a separately authored, dated note, the same
    separation the bake-off keeps between its generated report and `docs/ANALYSIS.md`.
    """
    from . import calibration as cal

    compact = output / "compact"
    compact.mkdir(exist_ok=True)
    seasons = spec["calibration_experiment"]["apply_seasons"]
    names = (
        "ranking",
        "paired_ranking",
        "calibration",
        "reliability",
        "deviance",
        "spearman",
        "replays",
        "picks",
        "advice",
    )
    data = {}
    for name in names:
        paths = [output / "apply" / str(s) / f"{name}.csv" for s in seasons]
        parts = [pd.read_csv(p) for p in paths if p.exists()]
        if not parts:
            continue
        data[name] = pd.concat(parts, ignore_index=True)
        save_frame(compact / f"{name}.csv", data[name])
    data["folds"] = cal.fold_table(output / "folds", spec)
    save_frame(compact / "folds.csv", data["folds"])
    data["policy"] = cal.policy_table(data["replays"], spec["baseline"])
    save_frame(compact / "policy.csv", data["policy"])
    data["decision_changes"] = cal.decision_changes(data["picks"], spec["baseline"])
    save_frame(compact / "decision-changes.csv", data["decision_changes"])
    if "advice" in data:
        data["advice_changes"] = cal.advice_changes(data["advice"], spec["baseline"])
        save_frame(compact / "advice-changes.csv", data["advice_changes"])
    data["paired_deviance"] = cal.paired_inference(
        ev.paired_deviance(
            _forecast_deviance(output, spec), spec["baseline"], spec["deviance_floor"]
        )
    )
    save_frame(compact / "paired-deviance.csv", data["paired_deviance"])
    # The declared coverage floors and the promised horizon/availability diagnostics are
    # conditions on the applied surface, so they are measured from it rather than from
    # the schedule audit `coverage.csv` records.
    coverage = cal.coverage_tables(output / "apply", spec)
    data.update(coverage)
    save_frame(compact / "coverage-out-of-fold.csv", coverage["coverage_out_of_fold"])
    save_frame(compact / "coverage-margins.csv", coverage["coverage_margins"])
    # Named for the population it describes. The training population is accounted for
    # separately, from the folds, because its seasons are the ones before any apply
    # season and none of them appears here.
    save_frame(compact / "zero-accounting-out-of-fold.csv", coverage["zero_accounting"])
    data["training_accounting"] = cal.training_accounting(output / "folds", spec)
    save_frame(compact / "training-accounting.csv", data["training_accounting"])
    data["strata"] = cal.stratified_scores(output / "apply", spec, spec["baseline"])
    save_frame(compact / "strata-out-of-fold.csv", data["strata"])
    write_json(compact / "manifest.json", manifest)
    shutil.copy2(output / "resolved-config.json", compact / "resolved-config.json")
    shutil.copy2(output / "coverage.csv", compact / "coverage.csv")
    (compact / "CALIBRATION.md").write_text(
        render_calibration_report(data, spec, manifest, output, config_path)
    )
    return compact


def _forecast_deviance(output, spec):
    """The current-week forecast rows of every candidate, for one paired comparison."""
    parts = [
        pd.read_parquet(output / "apply" / str(season) / "forecasts.parquet")
        for season in spec["calibration_experiment"]["apply_seasons"]
    ]
    return pd.concat(parts, ignore_index=True)


_POSITION_SUMMARY_COLUMNS = [
    "candidate",
    "group",
    "folds",
    "a_min",
    "a_max",
    "b_min",
    "b_max",
    "sources",
]


def _position_summary(folds):
    """Per-position coefficients collapsed across folds, so the table stays readable.

    The range rather than a mean: a map that is stable across folds and one that swings
    between them are different findings, and an average would report them identically.
    """
    rows = folds[folds.grouping == "position"]
    out = rows.groupby(["candidate", "group"], as_index=False).agg(
        folds=("fold", "nunique"),
        a_min=("a", "min"),
        a_max=("a", "max"),
        b_min=("b", "min"),
        b_max=("b", "max"),
    )
    used = rows.groupby(["candidate", "group"]).source.agg(lambda s: "/".join(sorted(set(s))))
    return out.merge(used.rename("sources"), on=["candidate", "group"])


def render_calibration_report(data, spec, manifest, output, config_path=None):
    """Facts, methods and provenance. No selection, no recommendation, no verdict."""
    from . import calibration as cal

    c = spec["calibration_experiment"]
    experiment = Path(output).name
    config_path = config_path or f"experiments/{experiment}.toml"
    folds = data["folds"]
    unsupported = folds[folds.fit_status != ev.FIT_OK]
    inherited = folds[folds.source != "own"]
    report = (
        "# Phase 3B Calibration Experiment\n\n"
        f"Specification `{config_path}`, dated {c['specification_date']}, frozen before any fit. "
        f"Folds train on {c['train_start']}-Y-1 and apply to Y for "
        f"Y = {min(c['apply_seasons'])}-{max(c['apply_seasons'])}; no outcome from Y enters its "
        "own fit, and a training row needs its target week played and scored, not merely an "
        "early forecast timestamp. All of these seasons have been explored before: this is "
        "walk-forward evaluation, not an untouched holdout.\n\n"
        "## Fitted Maps\n\n"
        f"The map is `exp(a) * lam ** b` on positive rates, applied to the "
        f"{c['map_target'].replace('_', ' ')} before the optimizer prunes or discounts it. "
        f"Eligible zero rates are {c['zero_rate_policy'].replace('_', ' ')} and hard exclusions "
        f"stay masks, so every candidate is scored on the same rows. Of "
        f"{len(folds)} fitted groups, {len(unsupported)} were unsupported and {len(inherited)} "
        "used the declared fallback rather than their own coefficient.\n\n"
        + markdown_table(
            folds[folds.grouping == "pooled"][
                ["fold", "candidate", "a", "b", "source", "n", "clusters", "fit_status"]
            ],
            ["fold", "candidate", "a", "b", "source", "n", "clusters", "fit_status"],
        )
        + "\n\nCoefficients by position, across folds. WR and TE are fitted separately because "
        "they share the FLEX slot, so their two maps are the ones able to reorder it.\n\n"
        + markdown_table(_position_summary(folds), _POSITION_SUMMARY_COLUMNS)
        + "\n\n## Forecast Score\n\n"
        f"Primary estimand: paired change in season-mean Poisson deviance on hard-eligible "
        f"current-week rows with identity lambda above {spec['deviance_floor']}, which is the "
        "same population for every candidate because a map sends zero to zero. Deviance is a "
        "loss, so a negative `delta` is a better forecast than identity's. Uncertainty is the "
        f"paired season t on {int(data['paired_deviance'].df.max()) + 1} season differences, "
        "and `p_holm` is the Holm step-down adjustment over the pre-registered family, which "
        "stops at the first comparison it does not reject. **`reject` is the adjusted result "
        "and the only one the promotion rule reads.** `lo`/`hi` is the ordinary unadjusted "
        "interval a single comparison would report. `stage_lo`/`stage_hi` is each step's own "
        "local test boundary, and Holm's levels widen down the ranking, so a later step's "
        "local interval can exclude zero on a step the procedure never reached; it is a "
        "diagnostic, never a bound. Nothing here is a verdict: the rule is applied in a "
        "separate, dated note.\n\n"
        + markdown_table(
            data["paired_deviance"],
            [
                "model",
                "delta",
                "se",
                "seasons",
                "t",
                "p_value",
                "holm_rank",
                "p_holm",
                "reject",
                "lo",
                "hi",
            ],
        )
        + "\n\n## Achieved Season Scores\n\n"
        "Each candidate replays its own no-reuse greedy and optimizer history and is compared "
        "against identity's replay of the same strategy, never against identity greedy. A "
        "better proper score and more touchdowns are different claims.\n\n"
        f"The declared non-inferiority bound is {c['margins']['max_policy_loss_td_per_season']} "
        "touchdowns per season, and it is compared against `lo` -- the lower end of the paired "
        "interval -- rather than against the point estimate. A candidate that reproduced "
        "identity's whole pick history differs from it by exactly nothing in every season, "
        "which is `degenerate_zero_variance`: an interval of width zero, and a bound, rather "
        "than the missing inference a zero standard error would otherwise read as.\n\n"
        + markdown_table(
            data["policy"],
            [
                "model",
                "strategy",
                "tds_per_season",
                "delta_vs_own_identity",
                "paired_se",
                "comparison",
                "lo",
                "hi",
                "seasons",
            ],
        )
        + "\n\n## Decision Changes\n\n"
        "How often a candidate's replay chose a different player than identity's did. A shared "
        "strictly increasing map preserves greedy's order within a slot; separate WR and TE maps "
        "need not, because they compete in FLEX. The optimizer has no such guarantee even under a "
        "shared map: it maximises an assignment sum, and a nonlinear map can reorder assignments "
        "and reorder cells at different horizons once the discount is applied.\n\n"
        + markdown_table(
            data["decision_changes"]
            .groupby(["model", "strategy"], as_index=False)
            .agg(decisions=("decisions", "sum"), changed=("changed", "sum")),
            ["model", "strategy", "decisions", "changed"],
        )
    )
    if "advice_changes" in data:
        report += (
            "\n\n## Static Hold Sensitivity\n\n"
            f"`advise_slot` compares a season cost in touchdowns against a fixed "
            f"{config.INFO_PREMIUM_TD} premium, so rescaling the rates rescales one side of "
            "that comparison and not the other. These counts are that sensitivity and nothing "
            "more: replay makes one decision per week through `plan_slot`, so a flipped hold "
            "here is not a touchdown gained or lost. Measuring the value of waiting needs the "
            "multi-event replay Phase 3C specifies.\n\n"
            + markdown_table(
                data["advice_changes"]
                .groupby("model", as_index=False)
                .agg(
                    decisions=("decisions", "sum"),
                    pick_changed=("pick_changed", "sum"),
                    hold_changed=("hold_changed", "sum"),
                ),
                ["model", "decisions", "pick_changed", "hold_changed"],
            )
        )
    promotion = ", ".join(f"{k} = {v}" for k, v in sorted(c["promotion"].items()))
    report += (
        "\n\n## Outcome Coverage\n\n"
        "Whether the applied surface has an outcome to be scored against, which is what the "
        "declared coverage floors are conditions on. An outcome is settled from the finalized "
        "scoring ledger, absence counting as zero only where the target week is completely "
        "scored, so what is missing here is a week the feed has not finished rather than a "
        "player the pool has not kept. Over hard-eligible rows, measured on identity: a map "
        "preserves keys, masks and zeros, so every candidate is scored on exactly these rows. "
        "`meets_floor` is a comparison of two measured numbers, not a promotion decision.\n\n"
        + markdown_table(
            data["coverage_margins"],
            [
                "population",
                "rows",
                "outcomes_known",
                "outcomes_missing",
                "coverage",
                "declared_floor",
                "meets_floor",
            ],
        )
        + "\n\nRetention is a different question, reported so that it cannot be mistaken for "
        "the first: what fraction of the surface belonged to a player still in the candidate "
        "pool at the week he was forecast for. It is the lower number, and it should be. "
        "Taking outcomes from the target week's own forecast row conflated the two and "
        "reported this as coverage, which understated coverage by exactly the players who "
        "left -- and dropped their outcomes, nearly all of them zeros, from the fit.\n\n"
        + markdown_table(
            data["coverage_margins"],
            ["population", "rows", "in_target_pool", "retention"],
        )
        + "\n\n## Training Population\n\n"
        "What each fold discarded before fitting, in the classes that mean different "
        "things. A hard exclusion is a mask rather than a forecast; an eligible zero rate "
        "cannot enter a fit on `log(lambda)`; an unresolved outcome is a week that is not "
        "completely scored. The counts reconcile to the surface each fold started from, so "
        '"excluded and counted" covers the training population and not only the applied '
        "one.\n\n"
        + markdown_table(
            data["training_accounting"],
            [
                "fold",
                "surface_rows",
                "hard_excluded",
                "eligible_zero_lam",
                "unresolved_outcome",
                "fitted_rows",
                "fitted_share",
            ],
        )
        + "\n\n## Out-Of-Fold Diagnostics\n\n"
        "The proper score by forecast horizon and availability, which the pooled fit declares "
        "two consequences for and cannot itself show. Late target weeks are forecast, and so "
        f"represented, more often than early ones under {c['row_weight']} row weight; and for "
        "an exponent away from one the Questionable multiplier is rescaled nonlinearly, so it "
        "is no longer a clean multiplier on the calibrated rate. These are declared "
        "diagnostics and cannot substitute for the primary estimand.\n\n"
        + markdown_table(
            data["strata"][data["strata"].model != spec["baseline"]],
            ["model", "horizon", "availability", "n", "deviance", "delta", "paired_se"],
        )
        + "\n\n## Methods And Provenance\n\n"
        f"Source `{manifest['code']['revision']}`"
        f"{' (dirty)' if manifest['code']['dirty'] else ''}, dataset "
        f"`{manifest['dataset_hash'][:12]}`, configuration "
        f"`{manifest['identity']['config_hash'][:12]}`. Production constants are unchanged and "
        "no calibrated model is deployed; the future discount, information premium and every "
        "base-model constant were fixed for the whole experiment. Candidate fallback order is "
        f"{list(c['fallback'])}, with a group needing at least {c['min_rows']} rows and "
        f"{c['min_clusters']} clusters to use its own coefficient. Declared margins: "
        + ", ".join(f"`{k}` = {v}" for k, v in sorted(c["margins"].items()))
        + ". Repeated forecasts of one target week share its single outcome, so fits cluster on "
        f"the {c['cluster'].replace('_', ' ')} and rows carry {c['row_weight']} weight; late "
        "target weeks are therefore forecast, and so represented, more often than early ones. "
        f"Identity is `{spec['baseline']}` and is mandatory: fold artifacts, out-of-fold "
        "surfaces, picks and these tables are saved beside the run. Applying the promotion rule "
        "is a separate, dated authoring step, and keeping the model unchanged is a valid "
        f"outcome. The promotion conditions are frozen with the rest of the specification as "
        f"`calibration_experiment.promotion` ({promotion}), "
        f"and the specification text itself hashes to `{c['specification_sha256'][:12]}`. "
        f"Candidates: {', '.join(n for _f, _g, n in cal.candidates(spec))}.\n"
    )
    return report


def render_reports(data, spec, manifest, output, config_path=None):
    """Render one evaluation report, without model-selection interpretations.

    This pure presentation step is also used to republish existing metrics without
    rerunning models or changing their recorded source/dataset identity.
    """
    # A generated report that names a different experiment sends its reader to the
    # wrong numbers, so both names come from the run rather than from a constant.
    experiment = Path(output).name
    config = config_path or f"experiments/{experiment}.toml"
    report = (
        "# Evaluation Report\n\n"
        "Generated measurements and methodological notes. Interpretation is maintained "
        "separately in [the authored analysis](ANALYSIS.md), is tied to a named experiment, "
        "and is not updated by report generation. NA denotes an unavailable metric.\n\n"
        "## Study Overview\n\n"
        f"Experiment: `{experiment}`. Specification: {spec['specification_date']}. "
        f"Seasons: {', '.join(map(str, spec['seasons']))}.\n\n"
        f"Models: {', '.join(spec['models'])}. Baseline: `{spec['baseline']}`; "
        f"shuffled seeds: {spec['seeds']}; deterministic seed sentinel: -1. "
        f"Policy: `{spec['input_policy']}`; roles: `{spec['role_source']}`.\n\n"
        "## Season Replay Results\n\n"
        "Actual TDs per season from each model/strategy's own no-reuse history. "
        "Differences and paired SEs are relative to the baseline's greedy replay; "
        "seeds/trials are averaged within season before computing SEs across seasons.\n\n"
    )
    replays = data["replay_summary"]
    replays = replays[replays.era == "all retrospective"]
    report += markdown_table(
        replays,
        [
            "model",
            "strategy",
            "tds_per_season",
            "se",
            "shuffle_sd",
            "delta_vs_baseline_greedy",
            "paired_se",
        ],
    )
    report += (
        "\n\n`replays.csv` records actual TDs, empty slots and unique players for every "
        "seed/trial and season; `picks.csv` records every choice. Hindsight uses actual scorer "
        "identities and the full feasible scoring history; it is a reference ceiling with a "
        "different candidate population.\n\n"
        "## Ranking Diagnostics\n\n"
        "Challenger minus baseline in the common pool: TDs per ranked candidate, "
        "not achieved season scores. SEs are across seasons.\n\n"
    )
    paired = data["paired_ranking_summary"]
    paired = paired[(paired.era == "all retrospective") & (paired["rank"] == "rank_available")]
    report += markdown_table(paired, ["model", "k", "mean", "se", "shuffle_sd", "seasons"])
    rank = data["ranking"]
    report += (
        "\n\nPer-seed coverage, jointly empty cells and per-season diagnostics are saved "
        "in `ranking.csv` and `paired_ranking.csv`. "
        f"Across exported ranking metric rows: {int(rank.empty_cells.sum())} empty cells "
        "(repeated across k, pools and seeds; not independent observations).\n\n"
        "## Calibration Diagnostics\n\n"
    )
    cal = data["calibration"]
    cal = cal[cal.model.eq(spec["baseline"])].sort_values("season")
    report += (
        f"`{spec['baseline']}`; hard-eligible forecasts with positive lambda, including "
        "baseline-spent players. Separate diagnostic fits on each evaluated season: "
        "`E[Y] = exp(intercept) * lambda ** slope`. SEs use player-season clusters. "
        "These are in-sample diagnostic coefficients, not fitted correction artifacts. "
        "Intercept 0 and slope 1 define the identity reference; slope alone does not "
        "establish calibration. `dropped_zero_lam` counts excluded eligible zero forecasts.\n\n"
    )
    cal_columns = [
        "season",
        "intercept",
        "slope",
        "se_intercept",
        "se_slope",
        "n",
        "clusters",
        "dropped_zero_lam",
    ]
    # Fits with no positive-rate rows do not export uncertainty or cluster fields.
    report += markdown_table(cal.reindex(columns=cal_columns), cal_columns)
    report += (
        "\n\nCalibration slopes/intercepts, reliability bins, tail deviance and within-slot "
        "Spearman results are saved separately by model, seed and season in `calibration.csv`, "
        "`reliability.csv`, `deviance.csv` and `spearman.csv`.\n\n"
        "## Methods And Limitations\n\n"
        + "\n".join(f"- {a}" for a in manifest["assumptions"])
        + "\n\n"
    )
    report += (
        "The solver is exact for the pruned fixed matrix, which does not establish an "
        "advantage for its rolling policy. Forecast diagnostics are not evidence of an "
        "achieved season gain or simulation readiness. No production parameter was changed.\n\n"
        "## Reproducibility And Verification\n\n"
        f"Artifact schema {manifest['schema_version']}; "
        f"scoring version {manifest['scoring_version']}; "
        f"database schema {manifest['database_schema']}.\n\n"
        f"Code revision `{manifest['code']['revision']}`; source fingerprint "
        f"`{manifest['code']['code_hash']}`; dirty tree: {manifest['code']['dirty']}. "
        f"Frozen dataset SHA-256 `{manifest['dataset_hash']}`.\n\n"
        f"Reproduce: `pool benchmark --config {config} "
        f"--output {output} --resume`. Checkpoints require matching "
        "code, configuration, dependencies and frozen dataset. Saved compact artifacts are in "
        f"`experiments/results/{experiment}`; detailed forecasts and future surfaces remain "
        "in the ignored output directory.\n"
    )
    return {"EVALUATION.md": report}


def _season_worker(frozen, season, spec, directory, folds=None):
    conn = sqlite3.connect(f"file:{Path(frozen).resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        evaluate_season(
            conn, season, spec, Path(directory), print, _candidates(spec, folds, season)
        )
    finally:
        conn.close()
    return season


def _candidates(spec, folds, season):
    """Bind this season's fitted maps, in the process that is about to use them."""
    if folds is None:
        return None
    from . import calibration

    return calibration.candidate_builders(folds, spec, season)


def season_sweep(conn, frozen, spec, seasons, root, log, folds=None):
    """Evaluate every season that has no verified checkpoint yet, in parallel.

    Shared by the plain benchmark and by both of the calibration experiment's season
    stages, so resume verifies artifact hashes the same way in all three.
    """
    root = Path(root)
    pending = []
    for season in seasons:
        directory = root / str(season)
        if checkpoint_complete(directory, season):
            log(f"Resume: verified completed season {season} in {root.name}")
        else:
            pending.append(season)
    if spec["workers"] == 1 or len(pending) == 1:
        for season in pending:
            log(f"Evaluating {season}")
            evaluate_season(
                conn, season, spec, root / str(season), log, _candidates(spec, folds, season)
            )
        return
    if not pending:
        return
    with ProcessPoolExecutor(
        max_workers=spec["workers"], mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = {
            executor.submit(_season_worker, frozen, season, spec, root / str(season), folds): season
            for season in pending
        }
        for future in as_completed(futures):
            log(f"Checkpoint complete: {future.result()}")


def run(config_path, output, *, resume=False, log=print):
    spec = resolve(config_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not resume:
        raise ValueError("Experiment already exists; use --resume or a fresh output directory")
    frozen = prepare_dataset(output, spec, log)
    conn = sqlite3.connect(f"file:{frozen.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        coverage = audit(conn, spec)
        require_decision_times(conn, spec)
        code = code_identity()
        dependencies = {
            name: importlib.metadata.version(name)
            for name in ("nfl-pool", "nflreadpy", "pandas", "numpy", "scipy", "pyarrow", "polars")
        }
        identity = dict(
            config_hash=hashlib.sha256(json_text(spec).encode()).hexdigest(),
            code_hash=code["code_hash"],
            dataset_hash=digest(frozen),
            dependencies=dependencies,
            python=platform.python_version(),
            thread_environment={
                k: os.environ.get(k)
                for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
        )
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["identity"] != identity:
                raise ValueError(
                    "Cannot resume: configuration, code, dependencies "
                    "or dataset fingerprint changed"
                )
        else:
            manifest = dict(
                schema_version=SCHEMA_VERSION,
                identity=identity,
                code=code,
                dataset_hash=identity["dataset_hash"],
                dependencies=dependencies,
                database_schema=conn.execute("PRAGMA user_version").fetchone()[0],
                scoring_version=scoring.SCORING_VERSION,
                assumptions=ASSUMPTIONS,
                input_metadata=input_metadata(conn, spec),
            )
            write_json(output / "resolved-config.json", spec)
            save_frame(output / "coverage.csv", coverage)
            if spec["resolved_decision_times"]:
                # The normalised copy the run actually used, beside the frozen dataset.
                save_frame(
                    output / "decision-times.csv", pd.DataFrame(spec["resolved_decision_times"])
                )
            write_json(manifest_path, manifest)
        if spec.get("calibration_experiment"):
            from . import calibration as cal

            cal.run_stages(conn, frozen, output, spec, log)
            compact = calibration_reports(output, spec, manifest, config_path)
        else:
            season_sweep(conn, frozen, spec, spec["seasons"], output, log)
            compact = reports(output, spec, manifest, config_path)
        write_json(
            output / "complete.json",
            dict(schema_version=SCHEMA_VERSION, identity=identity, seasons=spec["seasons"]),
        )
        log(f"Benchmark complete. Reports and compact results: {compact}")
        return compact
    finally:
        conn.close()
