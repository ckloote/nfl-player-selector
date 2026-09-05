"""Temporal isolation, comparable populations, snapshots, and reproducible artifacts."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from experiments import figures
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
    # Saved metrics must match byte for byte. The two reports differ only where they
    # name their own experiment, which is the point of naming it: a generated report
    # that cites another run's config sends its reader to the wrong numbers.
    docs = {"BACKTEST.md", "PROJECTION_BENCHMARK.md"}
    after = {p.name: benchmark.digest(p) for p in independent.iterdir()}
    assert {k: v for k, v in before.items() if k not in docs} == {
        k: v for k, v in after.items() if k not in docs
    }

    def anonymise(path, run, name):
        text = (path / name).read_text()
        return text.replace(str(run), "<run>").replace(
            f"experiments/results/{run.name}", "<results>"
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


# --- generated findings and figures -----------------------------------------
def _replay_rows(scores):
    """`replays` rows for {(season, model): total} under both strategies."""
    return pd.DataFrame(
        [
            dict(season=season, model=model, seed=-1, strategy=strategy, total=total)
            for (season, model), total in scores.items()
            for strategy in ("greedy", "optimizer")
        ]
    )


def _findings_data(scores, ranks):
    replays = _replay_rows(scores)
    seasons = sorted({season for season, _ in scores})
    summary = pd.DataFrame(
        [dict(era="all retrospective", model=m, paired_se=1.5) for m in ranks],
    )
    paired = pd.DataFrame(
        [
            dict(era="all retrospective", rank="rank_available", k=10, model=m, mean=v)
            for m, v in ranks.items()
        ]
    )
    ranking = pd.DataFrame([dict(season=s, covered_cells=40) for s in seasons])
    return {
        "replays": replays,
        "replay_summary": summary,
        "paired_ranking_summary": paired,
        "ranking": ranking,
    }, {"baseline": "shipped", "seasons": seasons}


def test_a_comparison_with_no_paired_spread_still_reports():
    """Seasons that lose the same amount every time have zero paired variance.
    Saved results for 2012 and 2018 at seed 0 give `within-player` exactly -8 in
    both, and dividing the mean by that standard error ended the whole report."""
    scores = {}
    for season in (2012, 2018):
        scores[(season, "shipped")] = 50.0
        scores[(season, "within-player")] = 42.0
        scores[(season, "random")] = 20.0
    data, spec = _findings_data(scores, {"within-player": -0.1, "random": -0.9})
    text = benchmark.replay_findings(data, spec)
    assert "-8.00" in text
    assert "no paired spread" in text
    assert "nan" not in text.lower()


@pytest.mark.parametrize(
    ("ranks", "expected", "forbidden"),
    [
        ({"a": -0.1, "b": -0.2, "c": -0.3, "d": -0.4}, "much as their", "against their"),
        ({"a": -0.4, "b": -0.3, "c": -0.2, "d": -0.1}, "against their", "Use it to rank"),
    ],
)
def test_the_ranking_conclusion_follows_the_correlation(ranks, expected, forbidden):
    """The conclusion asserted agreement whatever the correlation came out. The
    same models give Spearman +0.86 over fifteen seasons, -0.20 over 2011-2012
    and -0.02 over 2020-2021; only the first of those supports the sentence."""
    scores = {}
    for i, season in enumerate((2011, 2012)):
        scores[(season, "shipped")] = 50.0
        for j, model in enumerate(ranks):
            scores[(season, model)] = 50.0 - (j + 1) - i
    data, spec = _findings_data(scores, ranks)
    data["calibration"] = pd.DataFrame(
        [dict(season=s, model="shipped", slope=0.86, se_slope=0.02) for s in (2011, 2012)]
    )
    text = benchmark.diagnostic_findings(data, spec)
    assert expected in text
    assert forbidden not in text
    assert "never" in text or "not" in text


def _calibration(slopes, se=0.05):
    return pd.DataFrame(
        [dict(season=2011 + i, model="m", slope=s, se_slope=se) for i, s in enumerate(slopes)]
    )


def _viewbox_ys(svg):
    height = float(svg.split('viewBox="0 0 ')[1].split('"')[0].split()[1])
    values = []
    for attr in ("cy=", "y1=", "y2="):
        for chunk in svg.split(attr)[1:]:
            values.append(float(chunk.split('"')[1]))
    return height, values


def test_calibration_figure_holds_every_plotted_interval():
    """`vegas-environment` reaches a slope of 1.137 and an upper bound of 1.234.
    Against a hardcoded top of 1.03 its points fell outside the viewport."""
    svg = figures.slope_by_season({"calibration": _calibration([0.9, 1.137], se=0.05)}, "m")
    height, ys = _viewbox_ys(svg)
    assert ys and all(0 <= y <= height for y in ys)


def test_the_shipped_calibration_range_is_unchanged():
    """The adaptive bounds must not redraw a figure the data already fitted."""
    svg = figures.slope_by_season({"calibration": _calibration([0.82, 0.89])}, "m")
    assert ">1.00<" in svg and ">0.80<" in svg and ">1.05<" not in svg


def test_a_single_season_publishes_without_a_resolution_band():
    """One season has no cross-season paired SE, so the median floor is NaN and
    the bake-off tick loop raised `cannot convert float NaN to integer`."""
    scores = {(2024, "shipped"): 50.0, (2024, "within-player"): 42.0}
    data, _ = _findings_data(scores, {"within-player": -0.1})
    svg = figures.bakeoff(data, "shipped", None)
    assert "nan" not in svg.lower()
    assert "Shaded" not in svg
    height, ys = _viewbox_ys(svg)
    assert all(0 <= y <= height for y in ys)


def test_the_shaded_band_is_a_typical_scale_not_a_threshold():
    """`within-player` is -3.14 with its own SE of 1.46 — 2.16 SE from zero, and
    resolved by the report's criterion — yet sits inside a +/-4.35 median band."""
    scores = {}
    for i, season in enumerate((2011, 2012, 2013)):
        scores[(season, "shipped")] = 50.0
        scores[(season, "within-player")] = 46.9 + 0.1 * i
    data, _ = _findings_data(scores, {"within-player": -0.1})
    svg = figures.bakeoff(data, "shipped", 4.35)
    assert "cannot resolve" not in svg
    assert "typical scale" in svg
    # Resolved against its own SE, so it is coloured as a real difference even
    # though the median-SE band would swallow it.
    assert f'fill="{figures.OVER}"' in svg
