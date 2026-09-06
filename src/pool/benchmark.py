"""Configuration-driven, checkpointed research runs and reports from saved artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
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

from . import backtest, config, db, ingest, models, projections, scoring
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
    spec["workers"] = int(spec.get("workers", 1))
    if not 1 <= spec["workers"] <= 4:
        raise ValueError("workers must be between 1 and 4")
    spec["production_constants"] = constants()
    return spec


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
            "season",
            "model",
            "seed",
            "strategy",
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


def evaluate_season(conn, season, spec, directory, log):
    directory.mkdir(parents=True, exist_ok=True)
    chosen = {name: models.get(name) for name in spec["models"]}
    actuals = backtest.actual_tds(conn, season)
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
    from . import snapshots

    times = snapshots.decision_times(spec["decision_times"]) if spec.get("decision_times") else None
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
            dev = ev.deviance_seasons(paired, spec["baseline"])
            metrics["deviance"].append(dev[dev.model == model])
            metrics["spearman"].append(ev.spearman_seasons(df))
        forecast_parts.append(df)
        if spec["export_future_forecasts"]:
            surfaces = pd.concat(
                [
                    f[
                        [
                            "player_id",
                            "week",
                            "slot",
                            "game_id",
                            "lam",
                            "avail_mult",
                            "hard_eligible",
                        ]
                    ].assign(decision_week=w, model=model, seed=seed, season=season)
                    for w, f in frames.items()
                ],
                ignore_index=True,
            )
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
    files = {p.name: digest(p) for p in sorted(directory.iterdir()) if p.name != "checkpoint.json"}
    write_json(directory / "checkpoint.json", dict(schema_version=SCHEMA_VERSION, files=files))


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


def _season_worker(frozen, season, spec, directory):
    conn = sqlite3.connect(f"file:{Path(frozen).resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        evaluate_season(conn, season, spec, Path(directory), print)
    finally:
        conn.close()
    return season


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
            write_json(manifest_path, manifest)
        pending = []
        for season in spec["seasons"]:
            directory = output / str(season)
            checkpoint = directory / "checkpoint.json"
            if checkpoint.exists():
                saved = json.loads(checkpoint.read_text())
                if any(
                    not (directory / name).exists() or digest(directory / name) != checksum
                    for name, checksum in saved["files"].items()
                ):
                    raise ValueError(f"Checkpoint artifact hash mismatch for {season}")
                log(f"Resume: verified completed season {season}")
            else:
                pending.append(season)
        if spec["workers"] == 1:
            for season in pending:
                log(f"Evaluating {season}")
                evaluate_season(conn, season, spec, output / str(season), log)
        elif pending:
            with ProcessPoolExecutor(
                max_workers=spec["workers"], mp_context=multiprocessing.get_context("spawn")
            ) as executor:
                futures = {
                    executor.submit(
                        _season_worker, frozen, season, spec, output / str(season)
                    ): season
                    for season in pending
                }
                for future in as_completed(futures):
                    log(f"Checkpoint complete: {future.result()}")
        compact = reports(output, spec, manifest, config_path)
        write_json(
            output / "complete.json",
            dict(schema_version=SCHEMA_VERSION, identity=identity, seasons=spec["seasons"]),
        )
        log(f"Benchmark complete. Reports and compact results: {compact}")
        return compact
    finally:
        conn.close()
