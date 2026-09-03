import pandas as pd

from pool import ingest


def test_transform_schedule_builds_kickoff_and_filters_seasons():
    raw = pd.DataFrame(
        {
            "game_id": ["2026_01_A_B", "2025_01_A_B"],
            "season": [2026, 2025],
            "week": [1, 1],
            "game_type": ["REG", "REG"],
            "gameday": ["2026-09-13", "2025-09-07"],
            "gametime": ["13:00", None],
            "weekday": ["Sunday", "Sunday"],
            "home_team": ["B", "B"],
            "away_team": ["A", "A"],
            "home_score": [None, 20],
            "away_score": [None, 17],
            "spread_line": [3.0, -1.0],
            "total_line": [45.5, 44.0],
        }
    )
    out = ingest.transform_schedule(raw, [2026])
    assert list(out.game_id) == ["2026_01_A_B"]
    assert out.kickoff.iloc[0] == "2026-09-13T13:00"
    # missing gametime falls back to 13:00
    assert ingest.transform_schedule(raw, [2025]).kickoff.iloc[0] == "2025-09-07T13:00"


def test_transform_player_stats_keeps_pool_positions_and_reg_season():
    raw = pd.DataFrame(
        {
            "season": [2025] * 3,
            "week": [1, 1, 20],
            "season_type": ["REG", "REG", "POST"],
            "player_id": ["p1", "p2", "p1"],
            "player_display_name": ["A", "B", "A"],
            "position": ["QB", "CB", "QB"],
            "team": ["X", "Y", "X"],
            "opponent_team": ["Y", "X", "Y"],
            "passing_tds": [2, None, 1],
            "rushing_tds": [1, 0, 0],
            "receiving_tds": [0, 0, 0],
            "attempts": [30, 0, 30],
            "carries": [3, 0, 3],
            "targets": [0, 0, 0],
        }
    )
    out = ingest.transform_player_stats(raw)
    assert len(out) == 1
    assert out.iloc[0].pass_td == 2 and out.iloc[0].rush_td == 1


def test_transform_rosters_drops_missing_ids_and_other_positions():
    raw = pd.DataFrame(
        {
            "season": [2026] * 3,
            "week": [1] * 3,
            "gsis_id": ["p1", None, "p3"],
            "full_name": ["A", "B", "C"],
            "position": ["RB", "RB", "DE"],
            "team": ["X"] * 3,
            "status": ["ACT"] * 3,
            "depth_chart_position": ["RB", "RB", "DE"],
            "game_type": ["REG"] * 3,
        }
    )
    out = ingest.transform_rosters(raw)
    assert list(out.player_id) == ["p1"]


def test_transform_injuries_filters_and_dedupes():
    raw = pd.DataFrame(
        {
            "season": [2025] * 5,
            "week": [1, 1, 1, 20, 2],
            "season_type": ["REG", "REG", "REG", "POST", "REG"],
            "gsis_id": ["p1", None, "p3", "p1", "p1"],
            "full_name": ["A", "B", "C", "A", "A"],
            "team": ["X"] * 5,
            "position": ["RB", "RB", "LB", "RB", "RB"],
            "report_status": ["Out", "Out", "Out", "Out", "Questionable"],
            "practice_status": ["DNP", "DNP", "DNP", "DNP", "Limited"],
        }
    )
    out = ingest.transform_injuries(raw)
    # missing gsis_id, a non-pool position, and the postseason row are all dropped
    assert sorted(zip(out.player_id, out.week, strict=True)) == [("p1", 1), ("p1", 2)]
    assert out.set_index(["player_id", "week"]).loc[("p1", 2), "report_status"] == "Questionable"


def test_transform_injuries_keeps_last_duplicate_report():
    """The same player can be listed twice in a week; the later row wins."""
    raw = pd.DataFrame(
        {
            "season": [2025, 2025],
            "week": [1, 1],
            "season_type": ["REG", "REG"],
            "gsis_id": ["p1", "p1"],
            "full_name": ["A", "A"],
            "team": ["X", "X"],
            "position": ["RB", "RB"],
            "report_status": ["Questionable", "Out"],
            "practice_status": ["Limited", "DNP"],
        }
    )
    out = ingest.transform_injuries(raw)
    assert len(out) == 1
    assert out.iloc[0].report_status == "Out"


def test_transform_injuries_accepts_the_pre_2025_game_type_column():
    """2024 and earlier label the split `game_type`, not `season_type`. Reading
    the wrong one raised KeyError, so `refresh --season 2025` died importing the
    prior season it always pulls."""
    raw = pd.DataFrame(
        {
            "season": [2024, 2024],
            "game_type": ["REG", "WC"],
            "week": [1, 19],
            "gsis_id": ["p1", "p1"],
            "full_name": ["A B", "A B"],
            "team": ["AAA", "AAA"],
            "position": ["QB", "QB"],
            "report_status": ["Out", "Out"],
            "practice_status": [None, None],
        }
    )
    out = ingest.transform_injuries(raw)
    assert list(out.week) == [1]  # the playoff row is dropped


def test_transform_depth_charts_reads_the_legacy_weekly_schema():
    """Seasons up to 2024 publish one row per week with a `depth_team` rank;
    2025 onward publishes dated snapshots. Only the newer columns existed in
    the transform, so `refresh --season 2024` raised KeyError."""
    raw = pd.DataFrame(
        {
            "season": [2024] * 4,
            "club_code": ["AAA"] * 4,
            "week": [1, 1, 18, 18],
            "game_type": ["REG"] * 4,
            "depth_team": ["1", "2", "2", "1"],
            "gsis_id": ["starter", "backup", "starter", "backup"],
            "position": ["QB"] * 4,
        }
    )
    out = ingest.transform_depth_charts(raw, 2024, max_week=18).set_index("player_id")
    assert out.loc["backup", "rank"] == 1  # week 18 is the latest chart
    assert out.loc["starter", "rank"] == 2
    assert out.loc["backup", "as_of"] == "2024-W18"


def test_transform_depth_charts_trusts_the_schedule_over_a_mislabelled_week():
    """The 2024 file labels 1821 week-19 rows "REG"; the regular season is 18."""
    raw = pd.DataFrame(
        {
            "season": [2024, 2024],
            "club_code": ["AAA", "AAA"],
            "week": [18, 19],
            "game_type": ["REG", "REG"],
            "depth_team": ["1", "2"],
            "gsis_id": ["p1", "p1"],
            "position": ["QB", "QB"],
        }
    )
    assert ingest.transform_depth_charts(raw, 2024, max_week=18).as_of.iloc[0] == "2024-W18"
