"""Phase 3B: the chronological calibration experiment.

Every test states the failure it prevents. The failures worth preventing here are the
ones that would make a favourable result meaningless rather than the ones that would
make the code crash: a fit that saw the season it is scored on, a map that changed who
was eligible, an identity candidate that did not reproduce identity, or a hold flipped
by a rescale and then counted as touchdowns gained by waiting.
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pool import benchmark, capture, config, db, models
from pool import calibration as cal
from pool import evaluate as ev
from pool.recommend import advise_slot
from tests.test_phase3a import _reference_mle

HISTORY, SEASONS, APPLY = 2022, [2023, 2024, 2025], [2024, 2025]
WEEKS = [1, 2, 3, 4]
TEAMS = ["AAA", "BBB", "CCC", "DDD"]
PAIRS = [("AAA", "BBB"), ("CCC", "DDD")]


# --- a multi-season fixture --------------------------------------------------
def _people():
    """Two players per position per team, including tight ends.

    The shared backtest fixture has no tight end, and a FLEX slot with only receivers
    in it cannot show what a separate WR and TE map does to their ordering.
    """
    return [
        (f"{team}-{pos}{depth}", f"{team} {pos}{depth}", pos, team, depth)
        for team in TEAMS
        for pos in ("QB", "RB", "WR", "TE")
        for depth in (1, 2)
    ]


def _seed_seasons(conn, seasons, flipped=()):
    """A four-week, four-team season for each year, with scoring that drifts by season.

    The drift matters: if every season were identical, a fold fitted on earlier seasons
    would be indistinguishable from one fitted on later ones and the walk-forward
    schedule would be untestable.
    """
    people = _people()
    opponent = {}
    for home, away in PAIRS:
        opponent[home], opponent[away] = away, home
    games, weeks_rows, rosters = [], [], []
    for season in seasons:
        for week in WEEKS:
            for home, away in PAIRS:
                games.append(
                    (
                        f"g{season}-{week}-{home}",
                        season,
                        week,
                        "REG",
                        f"{season}-09-{10 + week:02d}T13:00",
                        home,
                        away,
                        -3.0,
                        45.0,
                    )
                )
            for pid, name, pos, team, depth in people:
                # Starters score, backups do not, and one starter per season goes quiet
                # so the outcome level moves from season to season.
                scores = depth == 1 and not (season % 2 == 0 and pid.endswith("WR1"))
                # A flipped season is the label perturbation: the backups score instead.
                tds = int(scores if season not in flipped else not scores)
                weeks_rows.append(
                    (
                        season,
                        week,
                        "REG",
                        pid,
                        name,
                        pos,
                        team,
                        opponent[team],
                        tds if pos == "QB" else 0,
                        0,
                        tds if pos != "QB" else 0,
                        30 if pos == "QB" else 0,
                        15 if pos == "RB" else 0,
                        8 if pos in ("WR", "TE") else 0,
                    )
                )
                rosters.append((season, week, pid, name, pos, team, "ACT", pos))
    with conn:
        conn.executemany(
            "INSERT INTO games(game_id, season, week, game_type, kickoff, home_team, away_team,"
            " spread_line, total_line) VALUES (?,?,?,?,?,?,?,?,?)",
            games,
        )
        conn.executemany(
            "INSERT INTO player_weeks(season, week, season_type, player_id, player_name, position,"
            " team, opponent, pass_td, rush_td, rec_td, attempts, carries, targets)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            weeks_rows,
        )
        conn.executemany(
            "INSERT INTO rosters(season, week, player_id, player_name, position, team, status,"
            " depth_chart_position) VALUES (?,?,?,?,?,?,?,?)",
            rosters,
        )
    with conn:
        conn.execute("UPDATE games SET home_score = 21, away_score = 21, kickoff_known = 1")
        db.backfill_game_ids(conn)
        conn.execute(
            "INSERT INTO game_results SELECT game_id, season, week, 1, 'complete', "
            "home_score, away_score, '2026-09-01T00:00:00+00:00' FROM games"
        )
        credits = []
        for i, row in enumerate(conn.execute("SELECT * FROM player_weeks")):
            for j in range(row["pass_td"] + row["rush_td"] + row["rec_td"]):
                credits.append(
                    (
                        row["game_id"],
                        i * 100 + j,
                        row["player_id"],
                        "throwing" if row["position"] == "QB" else "scoring",
                    )
                )
        conn.executemany(
            "INSERT INTO touchdown_credits(game_id, play_id, player_id, kind) VALUES (?,?,?,?)",
            credits,
        )
    return conn


def _spec(**overrides):
    spec = benchmark.resolve(Path("experiments/phase3-calibration-smoke.toml"))
    spec.update(
        seasons=SEASONS,
        history_start=HISTORY,
        workers=1,
        eras={"fixture": [min(APPLY), max(APPLY)]},
        expected_games={str(y): 8 for y in [HISTORY, *SEASONS]},
        scoring_corrections=None,
        resolved_corrections={},
    )
    spec["calibration_experiment"] = {
        **spec["calibration_experiment"],
        "train_start": SEASONS[0],
        "apply_seasons": list(APPLY),
        # The fixture is four weeks of four teams; the shipped thresholds would push
        # every group onto the identity fallback and test nothing about fitting.
        "min_rows": 10,
        "min_clusters": 2,
    }
    spec.update(overrides)
    return spec


def _build(directory, monkeypatch, spec, flipped=()):
    monkeypatch.setattr(benchmark, "resolve", lambda path: spec)
    monkeypatch.setattr(
        benchmark, "code_identity", lambda: dict(code_hash="t", revision="t", dirty=False)
    )
    conn = _seed_seasons(db.connect(directory / "source.db"), [HISTORY, *SEASONS], flipped)
    out = directory / "run"
    out.mkdir()
    destination = sqlite3.connect(out / "research.db")
    conn.backup(destination)
    destination.close()
    conn.close()
    benchmark.run("unused", out, log=lambda *_: None)
    return out


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    """One completed Phase 3B run, shared by the tests that only read its artifacts."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        spec = _spec()
        out = _build(tmp_path_factory.mktemp("phase3b"), monkeypatch, spec)
    return out, spec


@pytest.fixture(scope="module")
def relabelled(tmp_path_factory):
    """The same experiment with every scoring label in the last apply season inverted.

    Season 2025 is applied but never trained on -- folds 2024 and 2025 stop at 2023 and
    2024 -- so nothing fitted may move when its outcomes do. Rebuilding the whole run
    rather than perturbing a dataframe is the point: a leak could enter anywhere between
    the frozen dataset and the saved artifact, and only an end-to-end rebuild would find it.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        spec = _spec()
        out = _build(tmp_path_factory.mktemp("phase3b-relabelled"), monkeypatch, spec, (2025,))
    return out, spec


def _artifact(run, fold, candidate):
    return cal.load_artifact(run / "folds" / str(fold) / f"{candidate}.json")


# --- the level estimator -----------------------------------------------------
def _level_rows(n=400, scale=0.8, seed=0):
    rng = np.random.default_rng(seed)
    lam = rng.uniform(0.05, 1.2, n)
    return rng.poisson(scale * lam).astype(float), lam, np.arange(n) % 40


def test_the_level_fit_recovers_a_known_scale():
    """`c = sum(y) / sum(lam)` is the maximum likelihood, and a general-purpose optimiser
    that shares no code with it must land in the same place."""
    y, lam, cluster = _level_rows()
    fit = cal.fit_level(y, lam, cluster, min_clusters=2)
    assert fit["fit_status"] == ev.FIT_OK
    assert fit["slope"] == 1.0
    assert fit["intercept"] == pytest.approx(np.log(y.sum() / lam.sum()), abs=1e-12)
    # The same model as a two-parameter fit with the exponent pinned: profile the
    # reference likelihood at b = 1 by offsetting, which is what the family means.
    reference = _reference_mle(y, np.log(lam))
    assert fit["intercept"] == pytest.approx(reference[0], abs=0.05)


def test_a_level_fit_reports_no_slope_interval_because_it_estimated_no_slope():
    """A fixed exponent reported with an interval would read as an estimate that happened
    to be one, and a later table could not tell the two families apart."""
    y, lam, cluster = _level_rows()
    fit = cal.fit_level(y, lam, cluster, min_clusters=2)
    assert fit["family"] == "level"
    assert np.isnan(fit["se_slope"]) and np.isnan(fit["slope_lo"]) and np.isnan(fit["slope_hi"])
    assert np.isfinite(fit["se_intercept"]) and np.isfinite(fit["intercept_lo"])


def test_clustering_widens_the_level_interval():
    """Repeated forecasts of one target are not independent rows, and an interval that
    treats them as independent is narrow by however often the tool happened to look."""
    y, lam, _ = _level_rows()
    independent = cal.fit_level(y, lam, np.arange(len(y)), min_clusters=2)
    clustered = cal.fit_level(y, lam, np.zeros(len(y)) + np.arange(len(y)) // 40, min_clusters=2)
    assert clustered["se_intercept"] > independent["se_intercept"]


@pytest.mark.parametrize(
    ("y", "lam", "reason"),
    [
        ([], [], "empty input"),
        ([1.0], [0.5], "n <= 1 for a one-parameter fit"),
        ([0.0, 0.0, 0.0], [0.4, 0.5, 0.6], "all-zero outcomes"),
        ([1.0, 0.0, 2.0], [0.4, 0.0, 0.6], "non-positive rates"),
        ([1.0, -1.0, 2.0], [0.4, 0.5, 0.6], "negative outcomes"),
        ([1.0, np.nan, 2.0], [0.4, 0.5, 0.6], "non-finite inputs"),
    ],
)
def test_an_unfittable_level_population_gets_a_reason_not_a_coefficient(y, lam, reason):
    """A scale of zero, or one derived from a single row, is arithmetic rather than
    calibration. Every one of these once returned a number a table would have printed."""
    fit = cal.fit_level(y, lam, np.arange(len(y)), min_clusters=2)
    assert fit["fit_status"] == ev.FIT_UNSUPPORTED
    assert fit["reason"] == reason
    assert np.isnan(fit["intercept"])


def test_a_non_increasing_exponent_is_refused_rather_than_applied():
    """`b <= 0` ranks a better forecast below a worse one. It fits and it is not a
    calibration, so it must not reach an artifact as a usable coefficient."""
    rng = np.random.default_rng(3)
    lam = rng.uniform(0.05, 1.0, 600)
    # Outcomes falling as the rate rises: the maximum likelihood exponent is negative.
    y = rng.poisson(0.4 / (1.0 + 4.0 * lam)).astype(float)
    fit = cal.fit_log_affine(y, lam, np.arange(len(y)) % 50, min_clusters=2)
    assert fit["fit_status"] == ev.FIT_UNSUPPORTED
    assert "not positive" in fit["reason"]
    assert np.isnan(fit["slope"])


def test_the_level_fit_refuses_ragged_input():
    with pytest.raises(ValueError, match="equal length"):
        cal.fit_level([1.0, 2.0], [0.3], [0, 1])


# --- the map -----------------------------------------------------------------
def _proj(lams, positions=None):
    positions = positions or ["WR"] * len(lams)
    return pd.DataFrame(
        {
            "player_id": [f"p{i}" for i in range(len(lams))],
            "position": positions,
            "slot": [{"QB": "QB", "RB": "RB", "WR": "FLEX", "TE": "FLEX"}[p] for p in positions],
            "lam": [float(v) for v in lams],
            "avail_mult": [1.0] * len(lams),
            "hard_eligible": [True] * len(lams),
        }
    )


def _pooled(a, b):
    payload = dict(
        schema_version=cal.SCHEMA_VERSION,
        grouping="pooled",
        groups={cal.POOLED_GROUP: {"a": a, "b": b, "source": "own"}},
        default={"a": a, "b": b, "source": "own"},
    )
    return payload


def _by_position(table, default=(0.0, 1.0)):
    return dict(
        schema_version=cal.SCHEMA_VERSION,
        grouping="position",
        groups={k: {"a": v[0], "b": v[1], "source": "own"} for k, v in table.items()},
        default={"a": default[0], "b": default[1], "source": "pooled"},
    )


def test_a_map_leaves_zero_rates_masks_and_keys_exactly_where_they_were():
    """A calibration map that changed who was eligible would not be a calibration map,
    and the primary metric's population is only fixed across candidates because a map
    sends zero to zero."""
    proj = _proj([0.0, 0.3, 0.0, 0.9], ["RB", "WR", "TE", "QB"])
    proj.loc[2, "hard_eligible"] = False  # a mask, not a forecast
    proj.loc[1, "avail_mult"] = 0.85
    mapped = cal.mapped_lam(proj, _pooled(-0.2, 0.9))
    assert mapped[0] == 0.0 and mapped[2] == 0.0
    assert mapped[1] > 0 and mapped[3] > 0
    # Only `lam` may move; the frame the map was handed is untouched.
    assert list(proj.lam) == [0.0, 0.3, 0.0, 0.9]
    assert list(proj.hard_eligible) == [True, True, False, True]
    assert list(proj.avail_mult) == [1.0, 0.85, 1.0, 1.0]


def test_the_identity_map_is_exactly_the_identity():
    proj = _proj([0.0, 0.15, 0.62, 1.4])
    assert list(cal.mapped_lam(proj, _pooled(0.0, 1.0))) == list(proj.lam)


def test_a_shared_increasing_map_preserves_the_order_inside_a_slot():
    """The pruning rank, the greedy argmax and the frame's own sort all read `lam` within
    one slot-week, and a strictly increasing transform moves none of them."""
    proj = _proj([0.9, 0.4, 0.62, 0.15], ["WR", "TE", "WR", "TE"])
    mapped = cal.mapped_lam(proj, _pooled(-0.14, 0.89))
    assert list(np.argsort(-mapped)) == list(np.argsort(-proj.lam.to_numpy()))


def test_separate_wr_and_te_maps_can_reorder_the_flex_slot():
    """WR and TE compete in FLEX, so two different increasing maps are not one monotone
    transform of the pooled column and the ordering they produce is a result to measure."""
    proj = _proj([0.50, 0.45], ["WR", "TE"])
    shared = cal.mapped_lam(proj, _pooled(-0.1, 0.9))
    assert shared[0] > shared[1]
    split = cal.mapped_lam(proj, _by_position({"WR": (-0.6, 1.0), "TE": (0.0, 1.0)}))
    assert split[1] > split[0], "a separate TE map must be able to overtake a WR"


def test_a_position_without_its_own_map_takes_the_declared_default():
    proj = _proj([0.4, 0.4], ["WR", "QB"])
    mapped = cal.mapped_lam(proj, _by_position({"WR": (0.5, 1.0)}, default=(0.0, 1.0)))
    assert mapped[0] == pytest.approx(0.4 * np.exp(0.5))
    assert mapped[1] == pytest.approx(0.4)


# --- artifacts ---------------------------------------------------------------
def test_an_edited_fold_artifact_is_refused(experiment, tmp_path):
    """Coefficients are the experiment. A hand-edited artifact that still loaded would
    let a run be attributed to a fit that never produced it."""
    run, _spec_ = experiment
    path = run / "folds" / str(APPLY[0]) / "cal-level-pooled.json"
    tampered = tmp_path / "tampered.json"
    payload = json.loads(path.read_text())
    payload["groups"][cal.POOLED_GROUP]["a"] = 99.0
    tampered.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="does not match its recorded hash"):
        cal.load_artifact(tampered)


def test_registering_a_candidate_cannot_shadow_a_shipped_model():
    """Every saved artifact is keyed by model name, so two different functions answering
    to `shipped` would be indistinguishable in the exports afterwards."""
    with pytest.raises(ValueError, match="would shadow"):
        models.register_calibrated("shipped", lambda *a, **k: None)
    name = "cal-test-pooled"
    builder = models.calibrated(lambda proj: proj.lam)
    models.register_calibrated(name, builder)
    models.register_calibrated(name, builder)  # a second season binds its own fold
    assert models.get(name) is builder
    models.BUILDERS.pop(name)
    models.CALIBRATED.discard(name)


# --- the walk-forward schedule ------------------------------------------------
def test_a_fold_is_fitted_only_on_seasons_that_had_already_finished(experiment):
    """The whole experiment rests on this. A fold that saw its own apply season would
    make every downstream number a description of the answer."""
    run, spec = experiment
    for fold in APPLY:
        for _f, _g, name in cal.candidates(spec):
            artifact = _artifact(run, fold, name)
            assert artifact["fold"] == fold
            assert artifact["train_seasons"] == list(range(SEASONS[0], fold))
            assert artifact["cutoff_season"] == fold - 1
    early, late = (_artifact(run, f, "cal-level-pooled") for f in APPLY)
    assert len(late["train_seasons"]) > len(early["train_seasons"])
    assert late["training_digest"] != early["training_digest"]


def test_no_training_key_reaches_the_season_it_is_applied_to(experiment):
    """The digest is the check: it covers the exact keys that entered the fit, so
    "the fit saw nothing from Y" is verifiable rather than asserted."""
    run, spec = experiment
    for fold in APPLY:
        rows, digest = cal.training_pairs(run / "pairs", spec, fold)
        assert int(rows.season.max()) < fold
        assert rows.actual_tds.notna().all(), "a pair without an outcome is not a pair"
        assert (rows.lam > 0).all() and rows.hard_eligible.all()
        assert digest == _artifact(run, fold, "cal-level-pooled")["training_digest"]


def test_relabelling_the_evaluation_season_moves_no_coefficient_and_no_pick(experiment, relabelled):
    """The headline acceptance test. Inverting every scoring label in a season that is
    applied but never trained on must leave every fitted coefficient exactly where it was,
    and must leave the earlier apply season's forecasts and picks untouched.

    The relabelled season's own forecasts do move, and legitimately: the base model reads
    its completed weeks to project its later ones, so changing what happened in week one
    changes the week-two forecast. That is a base-model input, not an evaluation label
    reaching back into a fit. What must not move is the map, and it does not: the
    coefficients are identical and each run's surface is still exactly its own artifact
    applied to its own identity rates.
    """
    run, spec = experiment
    other, _ = relabelled
    for fold in APPLY:
        for _f, _g, name in cal.candidates(spec):
            assert (
                _artifact(run, fold, name)["artifact_hash"]
                == _artifact(other, fold, name)["artifact_hash"]
            ), f"{name} fold {fold} moved with an outcome it never saw"

    untouched = APPLY[0]
    keys = ["model", "season", "week", "slot", "player_id"]
    a, b = (
        pd.read_parquet(d / "apply" / str(untouched) / "forecasts.parquet")
        .sort_values(keys)
        .reset_index(drop=True)
        for d in (run, other)
    )
    pd.testing.assert_frame_equal(a[keys], b[keys])
    pd.testing.assert_series_equal(a.lam, b.lam)
    picks = [
        pd.read_csv(d / "apply" / str(untouched) / "picks.csv").sort_values(
            ["model", "strategy", "week", "slot"]
        )
        for d in (run, other)
    ]
    assert list(picks[0].player_id) == list(picks[1].player_id)

    # And the invariance is not vacuous: the fit does read the labels it is given.
    rows, digest = cal.training_pairs(run / "pairs", spec, APPLY[-1])
    assert cal.fit_family("log_affine", rows, 2)["intercept"] != pytest.approx(
        cal.fit_family("log_affine", rows.assign(actual_tds=rows.actual_tds * 7 + 3), 2)[
            "intercept"
        ]
    )
    assert digest == cal.training_pairs(run / "pairs", spec, APPLY[-1])[1]


@pytest.mark.parametrize("directory", ["experiment", "relabelled"])
def test_a_saved_surface_is_exactly_its_own_artifact_applied_to_the_identity_rates(
    directory, request
):
    """The map in force has to be the one the fold recorded. Recomputing it from the saved
    artifact and the saved identity rates is the check that no other transform crept in
    between fitting and replay."""
    run, spec = request.getfixturevalue(directory)
    for season in APPLY:
        surface = run / "apply" / str(season)
        positions = pd.read_parquet(surface / "forecasts.parquet")
        positions = positions[positions.model.eq(spec["baseline"])][
            ["week", "player_id", "position"]
        ].rename(columns={"week": "decision_week"})
        for _f, _g, name in cal.candidates(spec):
            saved = pd.read_parquet(surface / f"surface-{name}--1.parquet")
            keys = ["decision_week", "week", "slot", "player_id"]
            saved = saved.sort_values(keys).reset_index(drop=True)
            frame = saved.merge(positions, on=["decision_week", "player_id"], how="left")
            frame["position"] = frame.position.fillna("unknown")
            # `lam` on a saved candidate surface is the mapped rate; the map is recomputed
            # from `original_lam`, so the mapped column has to go before the rename.
            frame = frame.drop(columns=["lam"]).rename(columns={"original_lam": "lam"})
            expected = cal.mapped_lam(frame, _artifact(run, season, name))
            np.testing.assert_allclose(saved.lam.to_numpy(), expected, rtol=1e-12, atol=1e-15)


def test_a_sparse_group_falls_back_and_says_so(experiment):
    """A group promoted by lowering the bar after its coefficient looked interesting is
    a search, not a fit. The fallback is declared, taken, and recorded per group."""
    run, spec = experiment
    artifact = _artifact(run, APPLY[0], "cal-level-position")
    sources = {group: entry["source"] for group, entry in artifact["groups"].items()}
    assert set(sources) == set(cal.POSITION_GROUPS)
    assert set(sources.values()) <= {"own", "pooled", "identity"}
    inherited = {g: s for g, s in sources.items() if s != "own"}
    for group in inherited:
        fit = artifact["groups"][group]["fit"]
        assert fit["fit_status"] == ev.FIT_UNSUPPORTED or fit["n"] < artifact["min_rows"]
        if fit["fit_status"] == ev.FIT_UNSUPPORTED:
            assert fit["reason"] and np.isnan(fit["intercept"])


# --- the applied run ----------------------------------------------------------
def test_identity_reproduces_its_own_rates_picks_and_score(experiment):
    """The apply stage rebuilds the base model rather than reusing the pairs stage, so
    the two must agree exactly. A candidate's difference means nothing otherwise."""
    run, spec = experiment
    for season in APPLY:
        pairs = pd.read_parquet(run / "pairs" / str(season) / "forecasts.parquet")
        applied = pd.read_parquet(run / "apply" / str(season) / "forecasts.parquet")
        applied = applied[applied.model.eq(spec["baseline"])].reset_index(drop=True)
        keys = ["season", "week", "slot", "player_id"]
        a = pairs.sort_values(keys).reset_index(drop=True)
        b = applied.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_series_equal(a.lam, b.lam)
        pd.testing.assert_series_equal(a.actual_tds, b.actual_tds)
        one = pd.read_csv(run / "pairs" / str(season) / "replays.csv")
        two = pd.read_csv(run / "apply" / str(season) / "replays.csv")
        two = two[two.model.eq(spec["baseline"])]
        for strategy in ("greedy", "optimizer"):
            assert float(one[one.strategy.eq(strategy)].total.iloc[0]) == pytest.approx(
                float(two[two.strategy.eq(strategy)].total.iloc[0])
            )


def test_every_candidate_keeps_the_identity_keys_and_masks(experiment):
    """`assert_comparable` runs inside the benchmark, but only across the models the run
    happens to hold. Check the saved surfaces too: the population must be the same one."""
    run, spec = experiment
    for season in APPLY:
        directory = run / "apply" / str(season)
        base = pd.read_parquet(directory / f"surface-{spec['baseline']}--1.parquet")
        keys = ["decision_week", "week", "slot", "player_id"]
        base = base.sort_values(keys).reset_index(drop=True)
        for _f, _g, name in cal.candidates(spec):
            other = pd.read_parquet(directory / f"surface-{name}--1.parquet")
            other = other.sort_values(keys).reset_index(drop=True)
            pd.testing.assert_frame_equal(base[keys], other[keys])
            pd.testing.assert_series_equal(base.hard_eligible, other.hard_eligible)
            pd.testing.assert_series_equal(base.avail_mult, other.avail_mult)
            pd.testing.assert_series_equal(base.lam, other.original_lam, check_names=False)
            zero = base.lam.eq(0)
            assert (other.lam[zero] == 0).all(), "a zero rate must survive the map as zero"


def test_the_map_reaches_the_future_surface_and_not_only_the_decision_week(experiment):
    """A map applied after pruning, or only to the current week, would leave the plan the
    optimizer solves in the base model's units while the pick was made in the map's. Both
    parts of the surface must move, and by the same fold's coefficients."""
    run, spec = experiment
    for season in APPLY:
        directory = run / "apply" / str(season)
        for _f, _g, name in cal.candidates(spec):
            artifact = _artifact(run, season, name)
            coefficients = {(g["a"], g["b"]) for g in artifact["groups"].values()}
            if coefficients == {(0.0, 1.0)}:
                continue  # an all-identity fold maps nothing, by construction
            surface = pd.read_parquet(directory / f"surface-{name}--1.parquet")
            positive = surface[surface.original_lam > 0]
            current = positive[positive.week == positive.decision_week]
            future = positive[positive.week > positive.decision_week]
            assert len(current) and len(future), "the fixture must plan beyond this week"
            assert not np.allclose(current.lam, current.original_lam)
            assert not np.allclose(future.lam, future.original_lam)
            # Zero rates are the exception on both sides, and stay zero on both sides.
            zero = surface[surface.original_lam == 0]
            assert (zero.lam == 0).all()


def test_a_reported_hold_change_is_a_sensitivity_and_not_a_touchdown(experiment):
    """`advise_slot` compares a season cost in touchdowns against a fixed premium, so a
    rescale moves one side of that comparison and not the other. Replay never calls it,
    so a flipped hold is a mechanism to report, not a touchdown gained by waiting."""
    run, spec = experiment
    advice = pd.read_csv(run / "compact" / "advice.csv")
    assert set(advice.model) == {spec["baseline"], *(n for _f, _g, n in cal.candidates(spec))}
    assert (advice.premium == config.INFO_PREMIUM_TD).all()
    changes = cal.advice_changes(advice, spec["baseline"])
    assert (changes.hold_changed <= changes.decisions).all()
    replays = pd.read_csv(run / "compact" / "replays.csv")
    assert "hold" not in replays.columns and "hold_changed" not in replays.columns


def test_a_level_rescale_can_flip_a_hold_against_the_fixed_premium():
    """Scaling every rate scales every season cost with it, while `INFO_PREMIUM_TD` stays
    at 0.10 touchdowns. The threshold is not scale free: the same ranking, the same pick
    and the same alternative can commit at one scale and hold at another. A level
    calibration is therefore not a no-op for the recommender even when it changes no order.
    """
    from tests.conftest import proj_row

    proj = pd.DataFrame(
        [
            proj_row("early", "Early", "RB", 1, 1.20, kickoff="2026-09-10T20:00"),
            proj_row("late", "Late", "RB", 1, 0.90, kickoff="2026-09-13T13:00"),
            proj_row("early", "Early", "RB", 2, 0.10, kickoff="2026-09-17T20:00"),
            proj_row("late", "Late", "RB", 2, 0.80, kickoff="2026-09-20T13:00"),
        ]
    )
    now = pd.Timestamp("2026-09-10T12:00").to_pydatetime()
    base = advise_slot(proj, "RB", 1, set(), {}, now=now)
    scaled = advise_slot(proj.assign(lam=proj.lam * 0.1), "RB", 1, set(), {}, now=now)
    # Same recommendation, same Thursday player, same later alternative.
    assert base.recommended.player_id == scaled.recommended.player_id == "early"
    assert base.recommended.early and scaled.recommended.early
    assert base.hold_alternative.player_id == scaled.hold_alternative.player_id == "late"
    # The cost scales with the rates; the premium does not, so the decision flips.
    assert scaled.hold_alternative.cost == pytest.approx(base.hold_alternative.cost * 0.1)
    assert base.hold_alternative.cost > config.INFO_PREMIUM_TD
    assert scaled.hold_alternative.cost < config.INFO_PREMIUM_TD
    assert (base.hold, scaled.hold) == (False, True)


# --- resume -------------------------------------------------------------------
def test_a_completed_experiment_resumes_without_recomputing_anything(experiment):
    """Resume verifies artifact hashes for the fold stage the same way it does for a
    season, so a fitted map cannot be silently swapped between two halves of a run."""
    run, spec = experiment
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(benchmark, "resolve", lambda path: spec)
        monkeypatch.setattr(
            benchmark, "code_identity", lambda: dict(code_hash="t", revision="t", dirty=False)
        )
        before = {p.name: benchmark.digest(p) for p in sorted((run / "folds" / "2024").iterdir())}
        benchmark.run("unused", run, resume=True, log=lambda *_: None)
        after = {p.name: benchmark.digest(p) for p in sorted((run / "folds" / "2024").iterdir())}
    assert before == after


def test_resume_refuses_a_fold_whose_artifact_changed_underneath_it(experiment, tmp_path):
    directory = tmp_path / "folds" / "2024"
    directory.mkdir(parents=True)
    (directory / "cal-level-pooled.json").write_text("{}")
    benchmark.write_checkpoint(directory)
    assert benchmark.checkpoint_complete(directory, "fold 2024")
    (directory / "cal-level-pooled.json").write_text('{"a": 1}')
    with pytest.raises(ValueError, match="Checkpoint artifact hash mismatch"):
        benchmark.checkpoint_complete(directory, "fold 2024")


# --- the frozen specification --------------------------------------------------
def test_the_shipped_specification_declares_every_margin():
    """A missing margin blocks execution: a threshold chosen once the estimate is on
    screen is not a threshold."""
    spec = benchmark.resolve(Path("experiments/phase3-calibration.toml"))
    margins = spec["calibration_experiment"]["margins"]
    assert set(cal.REQUIRED_MARGINS) <= set(margins)
    assert all(isinstance(margins[k], (int, float)) for k in cal.REQUIRED_MARGINS)
    assert spec["deviance_floor"] == 0.0
    assert spec["models"] == [spec["baseline"]] == ["shipped"]


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"margins": {}}, "missing"),
        ({"families": ["quadratic"]}, "Unknown calibration families"),
        ({"groupings": ["team"]}, "Unknown calibration groupings"),
        ({"apply_seasons": [2099]}, "not replayed seasons"),
        ({"train_start": 2009}, "after history_start"),
        ({"map_target": "raw_rate"}, "map_target"),
        ({"fallback": ["own", "identity"]}, "fallback must be"),
        ({"min_rows": 0}, "positive integer"),
        ({"specification_date": None}, "dated before it is fitted"),
    ],
)
def test_an_undeclared_experiment_is_refused_before_it_runs(patch, message, tmp_path):
    """Each of these is a choice that has to exist before the numbers do."""
    import tomllib

    spec = tomllib.load(open("experiments/phase3-calibration.toml", "rb"))
    spec["calibration_experiment"].update(patch)
    with pytest.raises(ValueError, match=message):
        benchmark.resolve_calibration(spec)


def test_a_calibration_run_must_export_the_surface_it_intends_to_fit():
    import tomllib

    spec = tomllib.load(open("experiments/phase3-calibration.toml", "rb"))
    spec["export_future_forecasts"] = False
    with pytest.raises(ValueError, match="export it"):
        benchmark.resolve_calibration(spec)


# --- the report ----------------------------------------------------------------
def test_the_report_states_the_population_and_recommends_nothing(experiment):
    """The generated report carries facts, methods and provenance. Selecting a candidate
    is a separately authored, dated step, exactly as the bake-off keeps it."""
    run, _spec_ = experiment
    text = (run / "compact" / "CALIBRATION.md").read_text()
    assert "walk-forward evaluation, not an untouched holdout" in text
    assert "Production constants are unchanged" in text
    assert "separate, dated authoring step" in text
    for banned in ("we recommend", "should be deployed", "the winner", "ship "):
        assert banned not in text.lower()


def test_each_candidate_is_measured_against_its_own_strategy(experiment):
    """Comparing a candidate's optimizer replay against identity's greedy replay would
    attribute the solver to the map."""
    run, spec = experiment
    policy = pd.read_csv(run / "compact" / "policy.csv")
    replays = pd.read_csv(run / "compact" / "replays.csv")
    for strategy in ("greedy", "optimizer"):
        base = replays[replays.model.eq(spec["baseline"]) & replays.strategy.eq(strategy)]
        for _f, _g, name in cal.candidates(spec):
            row = policy[policy.model.eq(name) & policy.strategy.eq(strategy)]
            other = replays[replays.model.eq(name) & replays.strategy.eq(strategy)]
            expected = (other.set_index("season").total - base.set_index("season").total).mean()
            assert float(row.delta_vs_own_identity.iloc[0]) == pytest.approx(expected)


def test_the_primary_score_covers_every_eligible_positive_rate_not_only_the_tail(experiment):
    """The bake-off scores a `lam > 0.30` tail. The declared primary population is every
    hard-eligible current-week row with a positive rate, and it has to be the same rows
    for every candidate or the comparison is not paired."""
    run, spec = experiment
    assert spec["deviance_floor"] == 0.0
    deviance = pd.read_csv(run / "compact" / "deviance.csv")
    counts = deviance.groupby("model").n.sum()
    assert counts.nunique() == 1, "candidates must be scored on identical rows"
    forecasts = pd.concat(
        [pd.read_parquet(run / "apply" / str(s) / "forecasts.parquet") for s in APPLY],
        ignore_index=True,
    )
    eligible = forecasts[forecasts.model.eq(spec["baseline"]) & forecasts.hard_eligible]
    assert int(counts.iloc[0]) == int((eligible.lam > 0).sum())


# --- capture -------------------------------------------------------------------
def test_a_captured_decision_records_the_rate_it_was_made_on_and_the_one_before_it(tmp_path):
    """A decision that stored only one of the two could not say afterwards whether the
    calibrator or the base model moved."""
    from tests.test_phase3a import _decide

    conn = _seed_seasons(db.connect(tmp_path / "cap.db"), [HISTORY, SEASONS[0]])
    conn.close()
    conn = db.connect(tmp_path / "cap.db")
    proj = pd.DataFrame(
        [
            dict(
                player_id="p1",
                player_name="P One",
                position="RB",
                slot="RB",
                team="AAA",
                week=1,
                opponent="BBB",
                home=True,
                kickoff="2026-09-13T13:00",
                kickoff_known=1,
                game_id="g1",
                base_rate=0.5,
                def_mult=1.0,
                vegas_mult=1.0,
                home_mult=1.0,
                avail_mult=1.0,
                hard_eligible=True,
                report_status=None,
                role_mult=1.0,
                depth_rank=1,
                lam=0.45,
                prior_games=10,
                prior_tds=5,
                cur_games=0,
                cur_tds=0,
            )
        ]
    )
    from pool.recommend import advise_week

    original = pd.Series([0.5])
    advice = advise_week(proj, 1, set(), {}, now=pd.Timestamp("2026-09-13T09:00").to_pydatetime())
    decision = capture.record_decision(
        conn,
        2026,
        1,
        proj,
        advice,
        set(),
        {},
        decision_at="2026-09-13T09:00:00-04:00",
        calibrator="cal-level-pooled@2026",
        artifact_hash="abc123",
        original_lam=original,
    )
    events = capture.events(conn, 2026, 1)
    surface_hash = events[events.kind.eq("surface")].surface_hash.iloc[0]
    stored = capture.load_surface(conn, surface_hash)
    assert float(stored.mapped_lam.iloc[0]) == pytest.approx(0.45)
    assert float(stored.original_lam.iloc[0]) == pytest.approx(0.5)
    recorded = capture.recorded_identity(conn, decision)
    assert recorded["calibrator"] == "cal-level-pooled@2026"
    assert recorded["calibrator_artifact_hash"] == "abc123"
    conn.close()
    assert _decide is not None
