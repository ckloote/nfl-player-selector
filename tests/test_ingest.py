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
