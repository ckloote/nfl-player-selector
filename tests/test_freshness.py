"""Mocked feed integration: refresh, recommend, record, score and status."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import polars as pl
import pytest
from typer.testing import CliRunner

from pool import config, db, freshness, ingest, scoring, state
from pool.cli import app
from tests.test_workflow import end, play

runner = CliRunner()
NOW = datetime(2026, 9, 1, tzinfo=UTC)


@pytest.fixture
def feeds(monkeypatch):
    schedule = pd.DataFrame(
        [
            dict(
                game_id=f"g{s}",
                season=s,
                week=1,
                game_type="REG",
                gametime="13:00",
                gameday=f"{s}-09-13",
                weekday="Sunday",
                home_team="A",
                away_team="B",
                home_score=28 if s == 2025 else None,
                away_score=7 if s == 2025 else None,
                spread_line=3.0,
                total_line=45.0,
            )
            for s in (2025, 2026)
        ]
    )

    def stats(s):
        return pd.DataFrame(
            [
                dict(
                    season=s,
                    week=1,
                    season_type="REG",
                    player_id=pid,
                    player_display_name=name,
                    position=pos,
                    team="A",
                    opponent_team="B",
                    passing_tds=2 if pos == "QB" else 0,
                    rushing_tds=1 if pos == "RB" else 0,
                    receiving_tds=1 if pos == "WR" else 0,
                    attempts=30 if pos == "QB" else 0,
                    carries=10 if pos == "RB" else 0,
                    targets=5 if pos == "WR" else 0,
                )
                for pid, name, pos in [
                    ("q1", "Quarter One", "QB"),
                    ("r1", "Runner One", "RB"),
                    ("f1", "Flex One", "WR"),
                ]
            ]
        )

    def roster(s):
        raw = stats(s).rename(columns={"player_id": "gsis_id", "player_display_name": "full_name"})
        return raw.assign(status="ACT")

    def injuries(s):
        return roster(s).assign(report_status=None, practice_status=None)

    def depth(s):
        return roster(s).assign(club_code="A", depth_team="1", game_type="REG")

    def pbp(s):
        return pd.DataFrame(
            [
                play(f"g{s}", "f1", pass_touchdown=1, passer_player_id="q1"),
                play(f"g{s}", "r1", play_id=2),
                end(f"g{s}"),
            ]
        )

    monkeypatch.setattr(ingest, "fetch_schedules", lambda: schedule.copy())
    monkeypatch.setattr(ingest, "fetch_player_stats", stats)
    monkeypatch.setattr(ingest, "fetch_weekly_rosters", roster)
    monkeypatch.setattr(ingest, "fetch_season_roster", roster)
    monkeypatch.setattr(ingest, "fetch_injuries", injuries)
    monkeypatch.setattr(ingest, "fetch_depth_charts", depth)
    monkeypatch.setattr(ingest, "fetch_touchdowns", pbp)
    monkeypatch.setattr(
        state,
        "eastern_now",
        lambda now=None: datetime(2026, 9, 1) if now is None else now.replace(tzinfo=None),
    )
    monkeypatch.setenv("COLUMNS", "240")
    return schedule


def test_mocked_full_weekly_workflow_and_refresh_option(tmp_path, feeds):
    path = tmp_path / "workflow.db"
    for args in [
        ["refresh"],
        ["recommend", "--week", "1"],
        [
            "record",
            "--week",
            "1",
            "--qb",
            "Quarter One",
            "--rb",
            "Runner One",
            "--flex",
            "Flex One",
        ],
    ]:
        out = runner.invoke(app, [*args, "--db", str(path)])
        assert out.exit_code == 0, (out.stdout, out.exception)
    pending = runner.invoke(app, ["score", "--week", "1", "--db", str(path)])
    assert pending.exit_code == 0 and "3 pending" in pending.stdout
    feeds.loc[feeds.season.eq(2026), ["home_score", "away_score"]] = [28, 7]
    result = runner.invoke(app, ["score", "--refresh", "--db", str(path)])
    assert result.exit_code == 0, result.stdout
    assert "Season 2026 subtotal: 3 TDs; 0 pending" in result.stdout
    shown = runner.invoke(app, ["picks", "--db", str(path)])
    assert "Season 2026 subtotal: 3 TDs; 0 pending" in shown.stdout
    status = runner.invoke(app, ["status", "--week", "1", "--db", str(path)])
    assert status.exit_code == 0 and "1/1 games complete" in status.stdout
    assert "Last success (UTC)" in status.stdout


@pytest.mark.parametrize("failure", ["404", "network", "schema"])
def test_failures_retain_previous_data_and_success_time_attempt_other_feeds(
    tmp_path, feeds, monkeypatch, failure
):
    conn = db.connect(tmp_path / "refresh.db")
    assert not ingest.refresh(conn, 2026, log=lambda _: None).failures
    before = db.read_df(conn, "SELECT * FROM player_weeks ORDER BY season, player_id")
    saved = conn.execute(
        "SELECT last_success FROM feed_status WHERE season=2025 AND feed='player_stats'"
    ).fetchone()[0]

    def fail(s):
        if failure == "404":
            return None
        if failure == "schema":
            return pd.DataFrame({"bad_schema": [1]})
        raise ConnectionError("offline")

    calls = []
    original_depth = ingest.fetch_depth_charts
    monkeypatch.setattr(ingest, "fetch_player_stats", fail)
    monkeypatch.setattr(
        ingest, "fetch_depth_charts", lambda s: calls.append(s) or original_depth(s)
    )
    result = ingest.refresh(conn, 2026, log=lambda _: None)
    assert result.failures and calls == [2025, 2026]
    pd.testing.assert_frame_equal(
        before, db.read_df(conn, "SELECT * FROM player_weeks ORDER BY season, player_id")
    )
    status = conn.execute(
        "SELECT * FROM feed_status WHERE season=2025 AND feed='player_stats'"
    ).fetchone()
    assert status["last_success"] == saved and status["failure"]
    assert status["outcome"] == ("missing" if failure == "404" else "failed")


def test_unpublished_preseason_is_informational_and_preserves_rows(tmp_path, feeds, monkeypatch):
    conn = db.connect(tmp_path / "preseason.db")
    ingest.refresh(conn, 2026, log=lambda _: None)
    old = ingest.fetch_player_stats
    monkeypatch.setattr(ingest, "fetch_player_stats", lambda s: None if s == 2026 else old(s))
    result = ingest.refresh(conn, 2026, log=lambda _: None)
    assert not result.failures
    status = conn.execute(
        "SELECT * FROM feed_status WHERE season=2026 AND feed='player_stats'"
    ).fetchone()
    assert status["outcome"] == "unpublished" and status["last_success"]
    assert conn.execute("SELECT COUNT(*) FROM player_weeks WHERE season=2026").fetchone()[0] == 3


def test_roster_404_cannot_replace_weekly_history_with_season_snapshot(
    tmp_path, feeds, monkeypatch
):
    conn = db.connect(tmp_path / "roster.db")
    ingest.refresh(conn, 2026, log=lambda _: None)
    with conn:
        conn.execute("UPDATE rosters SET week=2")
    monkeypatch.setattr(ingest, "fetch_weekly_rosters", lambda s: None)
    result = ingest.refresh(conn, 2026, log=lambda _: None)
    assert any("rosters" in failure for failure in result.failures)
    assert set(db.read_df(conn, "SELECT week FROM rosters").week) == {2}


def test_refresh_bypasses_existing_package_cache_and_restores_config(monkeypatch):
    from nflreadpy.config import CacheMode, DataFormat, get_config, update_config
    from nflreadpy.downloader import get_downloader

    downloader = get_downloader()
    url = downloader._build_url(
        "nflverse-data", "stats_player/stats_player_week_2026", DataFormat.PARQUET
    )
    original = get_config().cache_mode
    update_config(cache_mode=CacheMode.MEMORY)
    downloader.cache.set(url, pl.DataFrame({"value": [1]}))

    class Response:
        content = pd.DataFrame({"value": [2]}).to_parquet(index=False)
        headers = {"Last-Modified": "Wed, 02 Sep 2026 12:00:00 GMT"}

        def raise_for_status(self):
            return None

    calls = []

    def fetch(*args, **kwargs):
        calls.append(args)
        response = Response()
        for hook in list(downloader.session.hooks["response"]):
            hook(response)
        return response

    monkeypatch.setattr(downloader.session, "get", fetch)
    try:
        assert ingest.fetch_player_stats(2026).value.iloc[0] == 1
        with ingest.uncached_downloads():
            fresh = ingest.fetch_player_stats(2026)
            assert fresh.value.iloc[0] == 2
            assert fresh.attrs["source_timestamp"] == Response.headers["Last-Modified"]
        assert len(calls) == 1 and get_config().cache_mode == CacheMode.MEMORY
        with pytest.raises(RuntimeError), ingest.uncached_downloads():
            raise RuntimeError("failed refresh")
        assert get_config().cache_mode == CacheMode.MEMORY
    finally:
        downloader.cache.clear()
        update_config(cache_mode=original)


def test_freshness_is_per_season_historical_inputs_exempt_and_thresholds_configurable(
    tmp_path, feeds
):
    conn = db.connect(tmp_path / "status.db")
    ingest.refresh(conn, 2026, log=lambda _: None)
    old = (NOW - timedelta(hours=2)).isoformat()
    with conn:
        conn.execute("UPDATE feed_status SET last_success=?", (old,))
    _, current = freshness.report(conn, 2026, 1, NOW)
    assert any("schedule: stale" in w for w in current)
    assert not any("rosters: stale" in w for w in current)
    _, past = freshness.report(conn, 2025, 1, NOW)
    assert not any("stale" in w for w in past)
    with config.override(FRESHNESS_HOURS={feed: 1 for feed in freshness.FEEDS}):
        _, warnings = freshness.report(conn, 2026, 1, NOW)
        assert any("rosters: stale" in w for w in warnings)
    freshness.record_status(conn, 2025, "schedule", "success", NOW.isoformat())
    assert (
        conn.execute(
            "SELECT last_success FROM feed_status WHERE season=2026 AND feed='schedule'"
        ).fetchone()[0]
        == old
    )


def test_coverage_and_fallback_warnings_independent_of_recent_fetch(tmp_path, feeds):
    conn = db.connect(tmp_path / "missing.db")
    ingest.refresh(conn, 2026, log=lambda _: None)
    with conn:
        conn.execute("DELETE FROM injuries")
        conn.execute("DELETE FROM depth_charts")
        conn.execute("DELETE FROM game_results")
        conn.execute("UPDATE games SET kickoff_known=0, total_line=NULL")
    _, warnings = freshness.report(conn, 2026, 1, NOW)
    text = "\n".join(warnings)
    assert "Unconfirmed kickoff" in text
    assert "Injuries missing" in text and "no week 1 coverage" in text
    assert "Depth chart missing" in text and "Missing betting lines" in text
    assert "legacy offensive-TD" in text


def test_refresh_failure_returns_nonzero_and_score_refresh_preserves_previous_scores(
    tmp_path, feeds, monkeypatch
):
    path = tmp_path / "failure.db"
    conn = db.connect(path)
    ingest.refresh(conn, 2026, log=lambda _: None)
    state.record_pick(conn, 2026, 1, "QB", "q1", "Quarter One", "QB")
    with conn:
        conn.execute("UPDATE my_picks SET tds=4")

    def fail():
        raise ConnectionError("schedule download failed")

    monkeypatch.setattr(ingest, "fetch_schedules", fail)
    out = runner.invoke(app, ["score", "--refresh", "--db", str(path)])
    assert out.exit_code == 1 and "stored scores retained" in out.stdout
    assert state.picks(conn, 2026).tds.iloc[0] == 4
    assert runner.invoke(app, ["refresh", "--db", str(path)]).exit_code == 1
    assert (
        conn.execute(
            "SELECT outcome FROM feed_status WHERE season=2026 AND feed='rosters'"
        ).fetchone()[0]
        == "success"
    )


@pytest.mark.parametrize(
    "bad", [pd.DataFrame({"bad": [1]}), pd.DataFrame([play(touchdown="broken"), end()])]
)
def test_scoring_parse_failure_preserves_imported_credits_and_coverage(tmp_path, bad):
    conn = db.connect(tmp_path / "scoring.db")
    with conn:
        conn.execute(
            "INSERT INTO games(game_id,season,week,game_type,kickoff,home_team,away_team,"
            "home_score,away_score) VALUES ('g1',2026,1,'REG','2026-09-13T13:00','A','B',28,7)"
        )
    scoring.import_touchdowns(conn, 2026, pd.DataFrame([play(), end()]))
    with pytest.raises(ValueError):
        scoring.import_touchdowns(conn, 2026, bad)
    assert scoring.touchdown_totals(conn, 2026).pool_td.sum() == 1
    assert scoring.coverage(conn, 2026).complete.all()


def test_snapshot_failure_rolls_back_the_replacement(tmp_path, feeds, monkeypatch):
    from pool import snapshots

    conn = db.connect(tmp_path / "atomic-snapshots.db")
    ingest.refresh(conn, 2026, log=lambda _: None)
    before = db.read_df(conn, "SELECT * FROM player_weeks ORDER BY season, player_id")
    observations = conn.execute(
        "SELECT COUNT(*) FROM input_observations WHERE feed='player_stats'"
    ).fetchone()[0]
    original_archive, original_fetch = snapshots.archive, ingest.fetch_player_stats

    def changed(season):
        return original_fetch(season).assign(passing_tds=99)

    def fail_archive(conn, season, feed, **kw):
        if feed == "player_stats":
            raise RuntimeError("archive disk failure")
        return original_archive(conn, season, feed, **kw)

    monkeypatch.setattr(ingest, "fetch_player_stats", changed)
    monkeypatch.setattr(snapshots, "archive", fail_archive)
    result = ingest.refresh(conn, 2026, log=lambda _: None)
    assert any("archive disk failure" in failure for failure in result.failures)
    pd.testing.assert_frame_equal(
        before, db.read_df(conn, "SELECT * FROM player_weeks ORDER BY season, player_id")
    )
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM input_observations WHERE feed='player_stats'"
        ).fetchone()[0]
        == observations
    )


def test_valid_empty_feed_is_archived_and_source_time_does_not_backdate(
    tmp_path, feeds, monkeypatch
):
    conn = db.connect(tmp_path / "empty-snapshots.db")
    original = ingest.fetch_injuries

    def empty(season):
        raw = original(season).iloc[:0].copy()
        raw.attrs["source_timestamp"] = "1990-01-01T00:00:00Z"
        return raw

    monkeypatch.setattr(ingest, "fetch_injuries", empty)
    assert not ingest.refresh(conn, 2026, log=lambda _: None).failures
    rows = conn.execute("SELECT * FROM input_observations WHERE feed='injuries'").fetchall()
    assert len(rows) == 2
    for row in rows:
        assert row["source_timestamp"] == "1990-01-01T00:00:00Z"
        assert datetime.fromisoformat(row["observed_at"]).year >= 2026
        assert '"rows": 0' in row["coverage"]
