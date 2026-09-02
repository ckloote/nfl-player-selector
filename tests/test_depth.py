import pandas as pd

from pool import config, ingest
from pool import projections as P


def test_transform_depth_charts_takes_latest_snapshot_and_best_rank():
    raw = pd.DataFrame(
        {
            "dt": ["2026-08-01", "2026-09-02", "2026-09-02", "2026-09-02", "2026-09-02"],
            "team": ["A", "A", "A", "A", "A"],
            "gsis_id": ["q1", "q1", "q2", "w1", "w1"],
            "pos_abb": ["QB", "QB", "QB", "WR", "WR"],
            "pos_slot": [9, 9, 9, 1, 2],
            "pos_rank": [2, 1, 2, 2, 1],
            "player_name": ["Q1", "Q1", "Q2", "W1", "W1"],
        }
    )
    out = ingest.transform_depth_charts(raw, 2026).set_index("player_id")
    assert out.loc["q1", "rank"] == 1  # old snapshot (rank 2) ignored
    assert out.loc["q2", "rank"] == 2
    assert out.loc["w1", "rank"] == 1  # best of the listed slots


def test_role_multipliers_backup_qb_is_heavily_discounted():
    depth = pd.DataFrame(
        {
            "player_id": ["q1", "q2", "q9", "r3"],
            "position": ["QB", "QB", "QB", "RB"],
            "rank": [1, 2, 7, 3],
            "team": ["A"] * 4,
            "season": [2026] * 4,
            "as_of": [None] * 4,
        }
    )
    m = P.role_multipliers(depth)
    assert m["q1"] == 1.0
    assert m["q2"] == config.DEPTH_MULT["QB"][2]
    assert m["q9"] == config.DEPTH_MULT["QB"][3]  # beyond table -> last value
    assert m["r3"] == config.DEPTH_MULT["RB"][3]
