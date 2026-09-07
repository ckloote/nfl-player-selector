"""Temporal isolation, comparable populations, snapshots, and reproducible artifacts."""

import hashlib
import json
import re
import runpy
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from experiments import appendix, figures
from pool import backtest, benchmark, db, models, snapshots
from pool import evaluate as ev
from pool import projections as P
from pool.cli import app
from pool.optimizer import plan_slot
from tests.conftest import proj_row
from tests.test_backtest import PRIOR, SEASON, WEEKS


def archive_all(conn, stamp="2024-09-01T00:00:00Z", optional=True):
    for season in (PRIOR, SEASON):
        for feed in snapshots.TABLES:
            if optional or feed in ("schedule", "player_stats", "touchdowns"):
                snapshots.archive(conn, season, feed, observed_at=stamp)


@pytest.mark.parametrize("name", sorted(models.BUILDERS))
@pytest.mark.parametrize("week", [1, 3])
def test_current_and_future_lines_cannot_change_any_model_or_choice(seeded, name, week):
    builder = models.get(name) or P.build_projections
    before = builder(P.load_frames(seeded, SEASON, week), week, role_source="usage")
    with seeded:
        seeded.execute(
            "UPDATE games SET total_line=100, spread_line=-40 WHERE season=? AND week>=?",
            (SEASON, week),
        )
    after = builder(P.load_frames(seeded, SEASON, week), week, role_source="usage")
    pd.testing.assert_frame_equal(before, after)
    for strategy in ("greedy", "optimizer"):
        assert (
            backtest.replay({week: before}, {}, SEASON, strategy).picks
            == backtest.replay({week: after}, {}, SEASON, strategy).picks
        )
    if week == 1:
        assert before.vegas_mult.eq(1).all()


def test_custom_baseline_and_seed_order(seeded):
    names = ["player-vegas", "regressed-rate", "random", "within-player"]
    kwargs = dict(baseline="player-vegas", seeds=[2, 3], strategies=("greedy",))
    a = ev.forecast_set(seeded, [SEASON], {n: models.get(n) for n in names}, **kwargs)
    b = ev.forecast_set(seeded, [SEASON], {n: models.get(n) for n in reversed(names)}, **kwargs)
    pd.testing.assert_frame_equal(a, b)
    base = a[a.model == "player-vegas"]
    for _, group in a.groupby(["model", "seed"]):
        ev.assert_comparable(base, group)
    assert a[a.model == "regressed-rate"].seed.unique().tolist() == [-1]
    assert sorted(a[a.model == "random"].seed.unique()) == [2, 3]


@pytest.mark.parametrize("baseline", ["shipped", "random"])
def test_invalid_baseline_fails_before_loading(monkeypatch, baseline):
    monkeypatch.setattr(backtest, "weekly_inputs", lambda *a, **kw: pytest.fail("loaded inputs"))
    with pytest.raises(ValueError, match="Baseline|shuffled baseline"):
        ev.forecast_set(None, [2025], {"random": models.get("random")}, baseline=baseline)


@pytest.mark.parametrize("name", ["random", "within-player"])
def test_shuffles_preserve_exclusions_and_recipient_questionable_adjustment(seeded, name):
    loaded = P.load_frames(seeded, SEASON, 1)
    base = P.build_projections(loaded, 1, role_source="usage")
    # Include an eligible zero, a Questionable recipient and an Out cell.
    base.loc[0, ["lam", "avail_mult", "hard_eligible"]] = [0.0, 0.0, False]
    base.loc[1, "lam"] *= 0.85
    base.loc[1, "avail_mult"] = 0.85
    base.loc[2, "lam"] = 0.0
    loaded.scaffold = base
    for seed in range(20):
        out = models.seeded(name, seed)(loaded, 1, role_source="usage")
        indexed = out.set_index(["player_id", "week"])
        original = base.set_index(["player_id", "week"])
        pd.testing.assert_series_equal(
            indexed.hard_eligible.sort_index(), original.hard_eligible.sort_index()
        )
        assert indexed.loc[tuple(base.loc[0, ["player_id", "week"]]), "lam"] == 0
        for _, group in base[base.hard_eligible].groupby(
            ["slot", "week"] if name == "random" else ["player_id"]
        ):
            keys = pd.MultiIndex.from_frame(group[["player_id", "week"]])
            changed = indexed.loc[keys]
            assert sorted((changed.lam / changed.avail_mult).round(9)) == sorted(
                (group.lam / group.avail_mult).round(9)
            )
        for strategy in ("greedy", "optimizer"):
            run = backtest.replay({1: out}, {}, SEASON, strategy)
            assert not any(
                p.player_id == base.iloc[0].player_id and p.week == base.iloc[0].week
                for p in run.picks
            )


def test_zero_is_eligible_and_pruning_uses_hard_mask():
    frame = pd.DataFrame(
        [
            proj_row("b", "B", "QB", 1, 0),
            proj_row("a", "A", "QB", 1, 0),
            proj_row("out", "O", "QB", 1, 100, status="Out"),
        ]
    )
    plan = plan_slot(frame, "QB", 1, set(), max_players=1)
    assert plan.pick_for(1).player_id == "a"
    assert plan.total == 0


def test_common_rankings_can_repeat_a_challenger_but_replay_cannot(seeded):
    frames = backtest.weekly_projections(seeded, SEASON, WEEKS)
    for frame in frames.values():
        frame.loc[frame.player_id.eq("AAA-RB1"), "lam"] = 100
    common = {w: set() for w in WEEKS}
    ranked = ev.forecasts(
        seeded, SEASON, model="synthetic", frames=frames, weeks=WEEKS, available_from=common
    )
    assert (
        len(
            ranked[
                (ranked.slot == "RB")
                & (ranked.rank_available == 1)
                & ranked.player_id.eq("AAA-RB1")
            ]
        )
        == 4
    )
    replay = backtest.replay(frames, backtest.actual_tds(seeded, SEASON), SEASON, "greedy")
    assert sum(p.player_id == "AAA-RB1" for p in replay.picks) == 1
    assert replay.players_used == len(replay.picks)


def test_seeds_are_averaged_before_season_uncertainty():
    frame = pd.DataFrame(
        [
            dict(model="m", season=s, seed=i, metric=v)
            for s, vals in [(1, [0, 2]), (2, [4, 6])]
            for i, v in enumerate(vals)
        ]
    )
    out = ev.summarize_seeds(frame, "metric").iloc[0]
    assert out["mean"] == 3
    assert out.se == 2
    assert out.seasons == 2
    assert out.shuffle_sd == pytest.approx(np.sqrt(2))
    duplicated = pd.concat([frame, frame.assign(seed=frame.seed + 2)], ignore_index=True)
    assert ev.summarize_seeds(duplicated, "metric").iloc[0].se == 2


def test_jointly_empty_cells_are_reported(seeded):
    frame = ev.forecast_set(
        seeded,
        [SEASON],
        {"shipped": None, "player-vegas": models.get("player-vegas")},
        strategies=(),
    )
    frame.loc[(frame.week == 1) & (frame.slot == "QB"), "rank_available"] = np.nan
    out = ev.paired_top_k(frame).iloc[0]
    assert out.jointly_empty_cells == 1
    assert out.covered_cells == 11
    assert out.units == "TDs per ranked candidate"
    assert "delta_per_season" not in out.index


def test_snapshots_boundaries_timezones_corrections_and_depth(seeded):
    with seeded:
        seeded.execute(
            "INSERT INTO depth_charts VALUES (?, 'AAA-QB1', 'AAA', 'QB', 2, 'old')", (SEASON,)
        )
    archive_all(seeded)
    decision = "2024-09-01T00:00:00Z"
    a = P.load_frames(seeded, SEASON, 1, input_policy="snapshots", decision_at=decision)
    before = P.build_projections(a, 1)
    with seeded:
        seeded.execute("UPDATE depth_charts SET rank=1")
        seeded.execute("UPDATE games SET total_line=90")
        seeded.execute("DELETE FROM touchdown_credits")
    archive_all(seeded, "2024-09-02T00:00:00Z")
    b = P.load_frames(
        seeded, SEASON, 1, input_policy="snapshots", decision_at="2024-08-31T20:00:00-04:00"
    )
    pd.testing.assert_frame_equal(before, P.build_projections(b, 1))
    assert b.depth.iloc[0]["rank"] == 2
    assert before[before.player_id.eq("AAA-QB1")].role_mult.eq(0.15).all()
    with pytest.raises(ValueError, match="Missing snapshot"):
        P.load_frames(
            seeded, SEASON, 1, input_policy="snapshots", decision_at="2024-08-31T23:59:59.999999Z"
        )


def test_snapshot_optional_absence_staleness_and_deadline(seeded):
    archive_all(seeded, optional=False)
    stamp = "2024-09-11T12:00:00-04:00"  # exact first deadline in the fixture
    loaded = P.load_frames(seeded, SEASON, 1, input_policy="snapshots", decision_at=stamp)
    assert any(p["missing"] for p in loaded.provenance if p["feed"] == "depth_charts")
    assert all(p["stale"] for p in loaded.provenance if not p["missing"])
    frame = P.build_projections(loaded, 1)
    assert not frame[frame.week.eq(1)].hard_eligible.any()
    assert frame[frame.week.gt(1)].hard_eligible.all()


def test_snapshot_dedup_empty_and_atomic_rollback(seeded):
    snapshots.archive(seeded, SEASON, "injuries", observed_at="2024-01-01T00:00:00Z")
    snapshots.archive(seeded, SEASON, "injuries", observed_at="2024-01-02T00:00:00Z")
    assert seeded.execute("SELECT COUNT(*) FROM input_payloads").fetchone()[0] == 1
    assert seeded.execute("SELECT COUNT(*) FROM input_observations").fetchone()[0] == 2
    assert (
        json.loads(seeded.execute("SELECT coverage FROM input_observations LIMIT 1").fetchone()[0])[
            "rows"
        ]
        == 0
    )
    count = seeded.execute("SELECT COUNT(*) FROM rosters").fetchone()[0]
    with pytest.raises(RuntimeError), db.transaction(seeded):
        db.replace_season(
            seeded, "rosters", SEASON, db.read_df(seeded, "SELECT * FROM rosters LIMIT 0")
        )
        snapshots.archive(seeded, SEASON, "rosters")
        raise RuntimeError("abort")
    assert seeded.execute("SELECT COUNT(*) FROM rosters").fetchone()[0] == count
    assert seeded.execute("SELECT COUNT(*) FROM input_observations").fetchone()[0] == 2
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        seeded.execute("DELETE FROM input_observations")


@pytest.mark.parametrize("command", ["evaluate", "backtest", "sweep"])
def test_cli_rejects_incompatible_temporal_options(command, tmp_path):
    result = CliRunner().invoke(
        app, [command, "--vegas-horizon", "0", "--db", str(tmp_path / "none.db")]
    )
    assert result.exit_code != 0
    assert "legacy-closing" in result.output
    result = CliRunner().invoke(
        app, [command, "--input-policy", "snapshots", "--db", str(tmp_path / "snapshot.db")]
    )
    assert result.exit_code != 0
    assert "decision-times" in result.output


def test_decision_csv_rejects_naive_and_duplicate_times(tmp_path):
    path = tmp_path / "times.csv"
    path.write_text("season,week,decision_at\n2024,1,2024-09-01T00:00:00\n")
    with pytest.raises(ValueError, match="timezone"):
        snapshots.decision_times(path)
    path.write_text(
        "season,week,decision_at\n2024,1,2024-09-01T00:00:00Z\n2024,1,2024-09-02T00:00:00Z\n"
    )
    with pytest.raises(ValueError, match="Duplicate"):
        snapshots.decision_times(path)


def test_benchmark_resume_and_fingerprint_rejection(seeded, tmp_path, monkeypatch):
    spec = benchmark.resolve(Path("experiments/phase2-validation.toml"))
    spec.update(
        seasons=[SEASON],
        history_start=PRIOR,
        models=["shipped", "random"],
        seeds=[0, 1],
        random_trials=2,
        workers=1,
        eras={"test retrospective": [SEASON, SEASON]},
        expected_games={str(PRIOR): 8, str(SEASON): 8},
    )
    monkeypatch.setattr(benchmark, "resolve", lambda path: spec)
    identity = dict(code_hash="test", revision="test", dirty=False)
    monkeypatch.setattr(benchmark, "code_identity", lambda: identity)
    out = tmp_path / "experiment"
    out.mkdir()
    destination = sqlite3.connect(out / "research.db")
    seeded.backup(destination)
    destination.close()
    original = benchmark.evaluate_season

    def interrupt(*args):
        original(*args)
        raise RuntimeError("interrupted after season checkpoint")

    monkeypatch.setattr(benchmark, "evaluate_season", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        benchmark.run("unused", out, log=lambda x: None)
    monkeypatch.setattr(benchmark, "evaluate_season", lambda *a: pytest.fail("recomputed season"))
    compact = benchmark.run("unused", out, resume=True, log=lambda x: None)
    assert {p.name for p in compact.glob("*.md")} == {"EVALUATION.md"}
    before = {p.name: benchmark.digest(p) for p in compact.iterdir()}
    benchmark.run("unused", out, resume=True, log=lambda x: None)
    assert before == {p.name: benchmark.digest(p) for p in compact.iterdir()}
    # A separate uninterrupted run produces the same saved metrics and forecasts.
    other = tmp_path / "uninterrupted"
    other.mkdir()
    destination = sqlite3.connect(other / "research.db")
    seeded.backup(destination)
    destination.close()
    monkeypatch.setattr(benchmark, "evaluate_season", original)
    independent = benchmark.run("unused", other, log=lambda x: None)
    # Saved metrics must match byte for byte. The reports from the two runs differ where they
    # name their own experiment, which is the point of naming it: a generated report
    # that cites another run's config sends its reader to the wrong numbers.
    docs = {"EVALUATION.md"}
    after = {p.name: benchmark.digest(p) for p in independent.iterdir()}
    assert {k: v for k, v in before.items() if k not in docs} == {
        k: v for k, v in after.items() if k not in docs
    }

    def anonymise(path, run, name):
        text = (path / name).read_text()
        return (
            text.replace(str(run), "<run>")
            .replace(f"experiments/results/{run.name}", "<results>")
            .replace(f"Experiment: `{run.name}`", "Experiment: `<experiment>`")
        )

    for name in docs:
        assert anonymise(compact, out, name) == anonymise(independent, other, name)
    assert benchmark.digest(out / str(SEASON) / "forecasts.parquet") == benchmark.digest(
        other / str(SEASON) / "forecasts.parquet"
    )
    identity["code_hash"] = "changed"
    with pytest.raises(ValueError, match="fingerprint"):
        benchmark.run("unused", out, resume=True, log=lambda x: None)


def test_evaluate_custom_baseline_cli_exports_reproducibility_metadata(seeded, tmp_path):
    path = seeded.execute("PRAGMA database_list").fetchone()[2]
    output = tmp_path / "forecasts.csv"
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--season",
            str(SEASON),
            "--model",
            "player-vegas,regressed-rate",
            "--baseline",
            "player-vegas",
            "--csv",
            str(output),
            "--db",
            path,
        ],
    )
    assert result.exit_code == 0, (result.output, result.exception)
    meta = json.loads(output.with_suffix(".metadata.json").read_text())
    assert meta["baseline"] == "player-vegas"
    assert meta["input_policy"] == "historical"
    assert len(meta["input_provenance"]) == 4
    assert pd.read_csv(output).decision_at.notna().all()
    picks = pd.read_csv(output.with_suffix(".picks.csv"))
    for _, group in picks.groupby(["model", "strategy"]):
        assert group.player_id.nunique() == len(group)


def test_all_snapshot_decisions_are_required_before_input_loading(seeded, monkeypatch):
    monkeypatch.setattr(P, "load_frames", lambda *a, **kw: pytest.fail("loaded inputs"))
    with pytest.raises(ValueError, match="missing timestamps"):
        backtest.weekly_inputs(
            seeded,
            SEASON,
            WEEKS,
            input_policy="snapshots",
            decision_times={(SEASON, 1): "2024-09-01T00:00:00Z"},
        )


def test_research_correction_checks_exact_scorer_ledger(tmp_path):
    spec = benchmark.resolve("experiments/phase2-validation.toml")
    fix = spec["resolved_corrections"]["corrections"][0]
    conn = db.connect(tmp_path / "correction.db")
    gid = fix["game_id"]
    with conn:
        conn.execute(
            "INSERT INTO games(game_id,season,week,game_type,kickoff,home_team,away_team,"
            "home_score,away_score) VALUES (?,2011,13,'REG','2011-12-04T20:30','NO','DET',31,17)",
            (gid,),
        )
        conn.execute(
            "INSERT INTO game_results VALUES (?,2011,13,0,"
            "'terminal scores do not match schedule',37,23,'2026-09-04T00:00:00Z')",
            (gid,),
        )
        for i, (pid, count) in enumerate(fix["expected_player_totals"].items()):
            original_id = next(
                (p for p, player, _ in fix["retained_credits"] if player == pid), i + 1
            )
            for j in range(count):
                conn.execute(
                    "INSERT INTO touchdown_credits(game_id,play_id,player_id,kind) "
                    "VALUES (?,?,?,'scoring')",
                    (gid, original_id + j * 100, pid),
                )
        conn.executemany(
            "INSERT INTO touchdown_credits(game_id,play_id,player_id,kind) VALUES (?,?,?,?)",
            [(gid, *key) for key in fix["remove_credits"]],
        )
    benchmark.apply_corrections(conn, spec, log=lambda _: None)
    assert conn.execute("SELECT complete FROM game_results").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM touchdown_credits").fetchone()[0] == 10
    benchmark.apply_corrections(conn, spec, log=lambda _: None)  # idempotent
    with conn:
        conn.execute("DELETE FROM touchdown_credits WHERE player_id='00-0027966'")
    with pytest.raises(ValueError, match="Source changed"):
        benchmark.apply_corrections(conn, spec, log=lambda _: None)


@pytest.mark.parametrize("command", ["backtest", "sweep"])
def test_snapshot_replay_commands_report_optional_absence_and_age(seeded, tmp_path, command):
    archive_all(seeded, optional=False)
    times = tmp_path / "times.csv"
    times.write_text(
        "season,week,decision_at\n"
        + "".join(f"{SEASON},{week},2024-09-10T12:00:00Z\n" for week in WEEKS)
    )
    path = seeded.execute("PRAGMA database_list").fetchone()[2]
    args = [
        command,
        "--season",
        str(SEASON),
        "--input-policy",
        "snapshots",
        "--decision-times",
        str(times),
        "--db",
        path,
    ]
    if command == "backtest":
        args += ["--strategy", "greedy,optimizer"]
    else:
        args += ["--discount", "1", "--prior-weight", "7", "--csv", str(tmp_path / "sweep.csv")]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, (result.output, result.exception)
    assert "missing /" in result.output and "Age hours" in result.output
    if command == "sweep":
        meta = json.loads((tmp_path / "sweep.metadata.json").read_text())
        assert any(feed["missing"] for row in meta["input_provenance"] for feed in row["inputs"])


def test_invalid_backtest_policy_does_not_open_a_database(monkeypatch):
    from pool import cli

    monkeypatch.setattr(cli, "_conn", lambda path: pytest.fail("opened database"))
    result = CliRunner().invoke(app, ["backtest", "--input-policy", "snapshots"])
    assert result.exit_code != 0
    assert "decision-times" in result.output


# --- generated measurements and figures -------------------------------------
def _replay_rows(scores):
    """`replays` rows for {(season, model): total} under both strategies."""
    return pd.DataFrame(
        [
            dict(season=season, model=model, seed=-1, strategy=strategy, total=total)
            for (season, model), total in scores.items()
            for strategy in ("greedy", "optimizer")
        ]
    )


def _report_data(scores, ranks):
    replays = _replay_rows(scores)
    seasons = sorted({season for season, _ in scores})
    summary = ev.replay_summary(replays, "shipped").assign(era="all retrospective")
    paired = pd.DataFrame(
        [
            dict(
                era="all retrospective",
                rank="rank_available",
                k=10,
                model=m,
                mean=v,
                se=0.0,
                shuffle_sd=0.0,
                seasons=len(seasons),
            )
            for m, v in ranks.items()
        ]
    )
    ranking = pd.DataFrame([dict(season=s, covered_cells=40, empty_cells=0) for s in seasons])
    return {
        "replays": replays,
        "replay_summary": summary,
        "paired_ranking_summary": paired,
        "ranking": ranking,
        "calibration": pd.DataFrame(
            [
                dict(
                    season=s,
                    model="shipped",
                    intercept=-0.1,
                    slope=0.86,
                    se_intercept=0.01,
                    se_slope=0.02,
                    n=100,
                    clusters=20,
                    dropped_zero_lam=0,
                )
                for s in seasons
            ]
        ),
    }, dict(
        baseline="shipped",
        seasons=seasons,
        models=sorted({m for _, m in scores}),
        seeds=[0],
        specification_date="2026-09-05",
        input_policy="historical",
        role_source="usage",
    )


@pytest.mark.parametrize("seasons", [[2012, 2018], [2025]])
def test_reports_measure_results_without_interpreting_them(seasons):
    manifest = dict(
        schema_version=1,
        scoring_version=2,
        database_schema=2,
        code=dict(revision="original-run", code_hash="original-source", dirty=False),
        dataset_hash="frozen-data",
        assumptions=["The recorded input policy."],
    )
    prose = []
    for delta, slope in [(-8.0, 0.86), (0.0, 1.0), (8.0, 1.2)]:
        scores = {
            (s, m): v
            for s in seasons
            for m, v in [("shipped", 50.0), ("within-player", 50.0 + delta), ("random", 20.0)]
        }
        data, spec = _report_data(scores, {"within-player": delta / 100, "random": -0.9})
        data["calibration"]["slope"] = slope
        reports = benchmark.render_reports(data, spec, manifest, Path("data/experiments/test"))
        assert set(reports) == {"EVALUATION.md"}
        report = reports["EVALUATION.md"]
        assert f"| within-player | greedy | {50 + delta:.4f}" in report
        assert f"| within-player | 10 | {delta / 100:.4f}" in report
        assert f"| {seasons[0]} | -0.1000 | {slope:.4f}" in report
        assert report.count("original-run") == report.count("original-source") == 1
        assert report.count("The recorded input policy.") == 1
        assert "(ANALYSIS.md)" in report
        assert "NA" in report if len(seasons) == 1 else "| 0.0000 |" in report
        assert "Actual TDs per season" in report
        assert "TDs per ranked candidate, not achieved season scores" in report
        headings = [
            "## Study Overview",
            "## Season Replay Results",
            "## Ranking Diagnostics",
            "## Calibration Diagnostics",
            "## Methods And Limitations",
            "## Reproducibility And Verification",
        ]
        assert [line for line in report.splitlines() if line.startswith("## ")] == headings
        assert report.index("original-source") > report.index(headings[-1])
        for claim in (
            "What the replays show",
            "What the diagnostics show",
            "clears two",
            "timing component",
            "telling players apart",
            "Use it to rank",
            "standing property",
            "resolution floor",
            "ahead of every",
            "**behind**",
        ):
            assert claim not in report
        prose.append("\n".join(line for line in report.splitlines() if not line.startswith("|")))
    assert prose[0] == prose[1] == prose[2]
    empty_fit = ev.calibration(pd.DataFrame(dict(hard_eligible=[True], lam=[0.0])))
    assert empty_fit["fit_status"] == "unsupported"
    assert empty_fit["reason"] == "no positive-rate rows"
    data["calibration"] = pd.DataFrame(
        [dict(season=s, model="shipped", **empty_fit) for s in seasons]
    )
    report = benchmark.render_reports(data, spec, manifest, Path("data/experiments/test"))[
        "EVALUATION.md"
    ]
    # No coefficient and no interval, but the excluded population is a count, not an
    # absence: one eligible zero-rate row was dropped and no positive-rate row remained.
    assert f"| {seasons[0]} | NA | NA | NA | NA | 0 | 0 | 1 |" in report


def test_republish_renders_saved_metrics_without_changing_the_run(tmp_path, monkeypatch):
    """Presentation may change after the recorded model source; the metrics must not."""
    output = tmp_path / "data/experiments/test"
    compact = output / "compact"
    compact.mkdir(parents=True)
    (tmp_path / "docs").mkdir()
    published = tmp_path / "experiments/results/test"
    published.mkdir(parents=True)
    archive = tmp_path / "docs/archive"
    archive.mkdir()
    superseded = tmp_path / "experiments/results/phase2-validation"
    superseded.mkdir()
    analysis = tmp_path / "docs/ANALYSIS.md"
    analysis.write_text("Separately reviewed interpretation.\n")
    dataset = output / "dataset.sqlite"
    dataset.write_bytes(b"frozen dataset")
    data, spec = _report_data(
        {(2025, "shipped"): 50.0, (2025, "within-player"): 42.0}, {"within-player": -0.1}
    )
    data["reliability"] = _reliability([("(0.05, 0.1]", 10, 0.08, 0.1)]).assign(model="shipped")
    identity = dict(
        code_hash="recorded-model-source",
        dataset_hash=benchmark.digest(dataset),
        config_hash=hashlib.sha256(benchmark.json_text(spec).encode()).hexdigest(),
    )
    manifest = dict(
        identity=identity,
        schema_version=1,
        scoring_version=2,
        database_schema=2,
        code=dict(revision="original-run", code_hash=identity["code_hash"], dirty=False),
        dataset_hash=identity["dataset_hash"],
        assumptions=["Recorded methodology."],
    )
    verification = _record([2025], 272, 272)
    verification.update(source_hash=identity["code_hash"], dataset_hash=identity["dataset_hash"])
    for path, value in [
        (output / "manifest.json", manifest),
        (compact / "manifest.json", manifest),
        (output / "complete.json", dict(identity=identity)),
        (output / "verification.json", verification),
        (compact / "resolved-config.json", spec),
    ]:
        benchmark.write_json(path, value)
    for name, frame in data.items():
        frame.to_csv(compact / f"{name}.csv", index=False)
    for name in ("BACKTEST.md", "PROJECTION_BENCHMARK.md"):
        (compact / name).write_text("Stale generated conclusion.\n")
        for directory in (published, tmp_path / "docs", archive, superseded):
            (directory / name).write_text("Previous presentation.\n")
    (compact / "EVALUATION.md").write_text("Stale combined report.\n")
    (compact / "unrelated.md").write_text("Not publisher-owned output.\n")
    (tmp_path / "docs/unrelated.md").write_text("Keep unrelated documentation.\n")
    (published / "unrelated.md").write_text("Keep experiment notes.\n")
    historical = {p: benchmark.digest(p) for d in (archive, superseded) for p in d.iterdir()}
    originals = {p: benchmark.digest(p) for p in output.rglob("*") if p.is_file()}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["publish.py", "test"])
    monkeypatch.setitem(sys.modules, "appendix", appendix)
    monkeypatch.setitem(sys.modules, "figures", figures)
    monkeypatch.setattr(benchmark, "code_identity", lambda: pytest.fail("consulted runtime model"))
    monkeypatch.setattr(benchmark, "reports", lambda *a: pytest.fail("recomputed compact metrics"))
    monkeypatch.setattr(benchmark, "evaluate_season", lambda *a: pytest.fail("reran models"))
    script = str(benchmark.ROOT / "experiments/publish.py")
    runpy.run_path(script, run_name="__main__")
    text = (published / "EVALUATION.md").read_text()
    assert "Stale generated conclusion" not in text
    assert "Stale combined report" not in text
    assert "| within-player | greedy | 42.0000" in text
    assert "| within-player | 10 | -0.1000" in text
    assert "| 2025 | -0.1000 | 0.8600" in text
    assert "original-run" in text and "recorded-model-source" in text
    assert "presentation.json" in text
    assert text.count("### Recorded Verification") == text.count("### Presentation") == 1
    assert text.count("Recorded methodology.") == text.count("Frozen dataset SHA-256") == 1
    for stem in ("bakeoff", "optimizer-by-season", "reliability", "calibration-slope"):
        assert text.count(f"(figures/{stem}.svg)") == 1
    assert text.index("## Season Replay Results") < text.index("figures/bakeoff.svg")
    assert text.index("figures/optimizer-by-season.svg") < text.index("## Ranking Diagnostics")
    assert text.index("## Calibration Diagnostics") < text.index("figures/reliability.svg")
    assert text.index("figures/calibration-slope.svg") < text.index("## Methods And Limitations")
    docs_text = (tmp_path / "docs/EVALUATION.md").read_text()
    assert (
        docs_text.replace("../experiments/results/test/figures/", "figures/")
        .replace("(../experiments/README.md)", "(../../README.md)")
        .replace("(ANALYSIS.md)", "(../../../docs/ANALYSIS.md)")
        == text
    )
    for directory in (published, tmp_path / "docs"):
        assert {p.name for p in directory.glob("*.md")} == {
            "EVALUATION.md",
            "unrelated.md",
            "README.md" if directory == published else "ANALYSIS.md",
        }
    assert (tmp_path / "docs/unrelated.md").read_text() == "Keep unrelated documentation.\n"
    assert (published / "unrelated.md").read_text() == "Keep experiment notes.\n"
    assert historical == {p: benchmark.digest(p) for p in historical}
    presentation = json.loads((published / "presentation.json").read_text())
    assert presentation["metric_source_hash"] == identity["code_hash"]
    assert presentation["source_hashes"]["src/pool/benchmark.py"] == benchmark.digest(
        benchmark.ROOT / "src/pool/benchmark.py"
    )
    assert originals == {p: benchmark.digest(p) for p in originals}
    assert analysis.read_text() == "Separately reviewed interpretation.\n"
    before = {p: benchmark.digest(p) for p in published.rglob("*") if p.is_file()}
    runpy.run_path(script, run_name="__main__")
    assert before == {p: benchmark.digest(p) for p in published.rglob("*") if p.is_file()}
    assert historical == {p: benchmark.digest(p) for p in historical}
    assert originals == {p: benchmark.digest(p) for p in originals}
    assert not (tmp_path / "docs/BACKTEST.md").exists()
    assert not (tmp_path / "docs/PROJECTION_BENCHMARK.md").exists()
    benchmark.write_json(compact / "resolved-config.json", dict(spec, baseline="different"))
    with pytest.raises(SystemExit, match="Compact configuration"):
        runpy.run_path(script, run_name="__main__")


def _calibration(slopes, se=0.05):
    return pd.DataFrame(
        [dict(season=2011 + i, model="m", slope=s, se_slope=se) for i, s in enumerate(slopes)]
    )


def _viewbox(svg, axis):
    """Every plotted coordinate on one axis, with the viewBox extent it must fit."""
    box = svg.split('viewBox="0 0 ')[1].split('"')[0].split()
    extent = float(box[0 if axis == "x" else 1])
    values = []
    for attr in (f"c{axis}=", f"{axis}1=", f"{axis}2="):
        for chunk in svg.split(attr)[1:]:
            values.append(float(chunk.split('"')[1]))
    return extent, values


def _viewbox_ys(svg):
    return _viewbox(svg, "y")


def test_calibration_figure_holds_every_plotted_interval():
    """`vegas-environment` reaches a slope of 1.137 and an upper bound of 1.234.
    Against a hardcoded top of 1.03 its points fell outside the viewport."""
    svg = figures.slope_by_season({"calibration": _calibration([0.9, 1.137], se=0.05)}, "m")
    height, ys = _viewbox_ys(svg)
    assert ys and all(0 <= y <= height for y in ys)


def test_both_fit_warnings_fit_inside_the_slope_figure():
    """Concatenated onto the estimate line, the two warnings ran about 77px past the
    720px canvas and the reader lost the end of whichever came second. Monospace at
    11px overflows near 105 characters, so every caption line has to stay under it."""
    cal = pd.DataFrame(
        {
            "model": "shipped",
            "season": range(2011, 2026),
            "slope": [np.nan, np.nan] + [0.85 + i * 0.01 for i in range(13)],
            "se_slope": [np.nan] * 5 + [0.03] * 10,
            "intercept": -0.1,
        }
    )
    svg = figures.slope_by_season({"calibration": cal}, "shipped")
    captions = [
        m.group(1)
        for m in re.finditer(r'font-size="11"[^>]*>([^<]*)</text>', svg)
        if not re.fullmatch(r"[\d.]+", m.group(1))
    ]
    assert any("unsupported" in c for c in captions)
    assert any("cluster SEs" in c for c in captions)
    assert max(len(c) for c in captions) <= 105
    # The plot is given the extra room rather than drawn over the caption.
    assert 'viewBox="0 0 720 335"' in svg


def test_the_shipped_calibration_range_is_unchanged():
    """The adaptive bounds must not redraw a figure the data already fitted."""
    svg = figures.slope_by_season({"calibration": _calibration([0.82, 0.89])}, "m")
    assert ">1.00<" in svg and ">0.80<" in svg and ">1.05<" not in svg


def test_a_single_season_publishes_without_a_resolution_band():
    """One season has no cross-season paired SE, so the median floor is NaN and
    the bake-off tick loop raised `cannot convert float NaN to integer`."""
    scores = {(2024, "shipped"): 50.0, (2024, "within-player"): 42.0}
    data, _ = _report_data(scores, {"within-player": -0.1})
    svg = figures.bakeoff(data, "shipped")
    assert "nan" not in svg.lower()
    assert "Shaded" not in svg
    height, ys = _viewbox_ys(svg)
    assert all(0 <= y <= height for y in ys)


def test_a_single_season_bakeoff_keeps_its_zero_reference():
    """The zero line is always drawn, so zero must always be in the domain. Saved 2025
    results, whose comparisons all lose, put it at x=819.9 on a 720-wide canvas."""
    scores = {(2025, "shipped"): 50.0, (2025, "within-player"): 42.0, (2025, "no-vegas"): 44.0}
    data, _ = _report_data(scores, {"within-player": -0.1, "no-vegas": -0.2})
    svg = figures.bakeoff(data, "shipped")
    width, xs = _viewbox(svg, "x")
    assert xs and all(0 <= x <= width for x in xs)


def test_random_alone_is_still_a_comparison():
    """`random` is set aside because it would squash the real comparisons. With
    `models = ["shipped", "random"]` there are none to squash, and dropping it anyway
    left `min()` an empty argument."""
    scores = {}
    for season in (2011, 2012):
        scores[(season, "shipped")] = 50.0
        scores[(season, "random")] = 20.0
    data, _ = _report_data(scores, {"random": -0.9})
    svg = figures.bakeoff(data, "shipped")
    assert ">random<" in svg and "omitted" not in svg
    height, ys = _viewbox_ys(svg)
    assert all(0 <= y <= height for y in ys)


def test_a_bakeoff_without_a_challenger_says_so():
    """A baseline-only run has nothing to compare; the report still references the file."""
    data, _ = _report_data({(2011, "shipped"): 50.0}, {})
    svg = figures.bakeoff(data, "shipped")
    assert "nothing to compare" in svg and svg.rstrip().endswith("</svg>")


def _reliability(bins):
    return pd.DataFrame(
        [
            dict(model="m", season=2025, bin=name, n=n, proj=proj, actual=actual)
            for name, n, proj, actual in bins
        ]
    )


def test_a_bin_that_scored_nothing_has_no_ratio():
    """The saved 2025 `vegas-environment` bin holds one candidate projected at 0.0483
    with no touchdowns. Dividing by zero made its ratio the axis bound, and the tick
    loop raised `Maximum allowed size exceeded` — in 17 of 150 season/model runs."""
    data = {
        "reliability": _reliability(
            [
                ("(-0.001, 0.05]", 40, 0.04, 0.06),
                ("(0.05, 0.1]", 20, 0.08, 0.09),
                ("(0.8, 1.0]", 1, 0.0483, 0.0),
                ("(1.0, 1.5]", 25, 1.2, 1.1),
                ("(1.5, inf]", 30, 1.7, 1.5),
            ]
        )
    }
    svg = figures.reliability(data, "m")
    assert "inf" not in svg and "nan" not in svg.lower()
    assert "no TDs" in svg and "have no ratio" in svg
    height, ys = _viewbox_ys(svg)
    assert ys and all(0 <= y <= height for y in ys)
    # The defined bins still plot, and no line is drawn across the undefined one.
    assert ">0.67<" in svg and ">1.13<" in svg
    assert svg.count("<polyline") == 2


def _record(seasons, scheduled, complete):
    return dict(
        implementation_commit="abc1234",
        pytest_passed=229,
        lint="Recorded lint result",
        format="Recorded formatting result",
        checks=["Recorded scoring check"],
        scheduled_games=scheduled,
        complete_games=complete,
        seasons=[
            dict(
                season=season,
                forecast_rows=1000,
                model_seed_runs=48,
                season_scores=117,
                picks=5967,
            )
            for season in seasons
        ],
    )


def test_the_appendix_counts_come_from_the_record():
    """It announced "All 15 seasons completed" and "All 4,175 required games" beside
    counts summed from the run, so a one-season publication claimed fourteen it never
    ran. The 2022 canceled game is only remarked on when 2022 was replayed."""
    solo = appendix.run_verification(_record([2025], 272, 272))
    assert "15 seasons" not in solo and "4,175" not in solo
    assert "The 2025 season completed" in solo and "All 272 required games" in solo
    assert "Bills" not in solo
    assert "Recorded lint result" in solo and "Recorded scoring check" in solo
    assert "A full `--resume` verified" not in solo
    assert "OPENBLAS_NUM_THREADS=1" not in solo

    full = appendix.run_verification(_record(range(2011, 2026), 4175, 4175))
    assert "All 15 seasons completed" in full and "All 4,175 required games" in full
    assert "Bills" in full


def test_the_appendix_does_not_claim_coverage_it_lacks():
    """The completeness sentence is generated too: a short run must not inherit it."""
    text = appendix.run_verification(_record([2024, 2025], 544, 543))
    assert "543 of 544 required games" in text and "All 544" not in text


def test_the_appendix_distinguishes_stage_records_from_unique_seasons():
    """Pairs and apply replay some of the same years, not independent extra seasons."""
    record = _record([*range(2011, 2026), *range(2016, 2026)], 4175, 4175)
    text = appendix.run_verification(record)
    assert "25 stage-season records completed, covering 15 unique seasons" in text
    assert "All 25 seasons completed" not in text
    # Work totals include both stages even though the season count is deduplicated.
    assert "1,200 model/seed/season runs" in text
    assert "25,000 forecast rows" in text
    assert "2,925 achieved season scores" in text
    assert "149,175 individual replay picks" in text
    assert "All 4,175 required games" in text
    assert text.count("Bills") == 1


def test_figures_do_not_classify_significance_or_state_conclusions():
    scores = {}
    for i, season in enumerate((2011, 2012, 2013)):
        scores[(season, "shipped")] = 50.0
        scores[(season, "within-player")] = 46.9 + 0.1 * i
    data, _ = _report_data(scores, {"within-player": -0.1})
    svg = figures.bakeoff(data, "shipped")
    assert "cannot resolve" not in svg
    assert "Shaded" not in svg and "threshold" not in svg
    assert f'fill="{figures.OVER}"' not in svg
    assert f'fill="{figures.INK}"' in svg
    slope = figures.slope_by_season({"calibration": _calibration([0.82, 0.89])}, "m")
    assert "1.0 is calibrated" not in slope and "too extreme" not in slope
    assert "1.96 SE" in slope and "Intercepts in table" in slope
    assert "is the finding" not in figures.optimizer_by_season(data, "shipped")
