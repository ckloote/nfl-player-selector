import pandas as pd
import pytest

from pool import db, state


@pytest.fixture
def conn():
    return db.connect(":memory:")


def test_record_pick_rejects_reuse_and_wrong_slot(conn):
    state.record_pick(conn, 2026, 1, "QB", "q1", "Josh Allen", "QB")
    with pytest.raises(state.PickError):
        state.record_pick(conn, 2026, 5, "QB", "q1", "Josh Allen", "QB")
    with pytest.raises(state.PickError):
        state.record_pick(conn, 2026, 1, "RB", "q1", "Josh Allen", "QB")
    # re-recording the same slot/week replaces
    state.record_pick(conn, 2026, 1, "QB", "q2", "Lamar Jackson", "QB")
    assert state.used_ids(conn, 2026) == {"q2"}
    assert state.locked_by_slot(conn, 2026)["QB"] == {1: "q2"}
    assert state.remove_pick(conn, 2026, 1, "QB") and state.used_ids(conn, 2026) == set()


def test_find_player_matching():
    pool = pd.DataFrame(
        {
            "player_id": ["1", "2", "3"],
            "player_name": ["Josh Allen", "Josh Jacobs", "Ja'Marr Chase"],
            "position": ["QB", "RB", "WR"],
            "team": ["BUF", "GB", "CIN"],
        }
    )
    assert list(state.find_player(pool, "josh allen").player_id) == ["1"]
    assert len(state.find_player(pool, "Josh")) == 2
    assert list(state.find_player(pool, "Josh", ("QB",)).player_id) == ["1"]
    assert list(state.find_player(pool, "jamarr chase").player_id) == ["3"]
    assert list(state.find_player(pool, "Chase").player_id) == ["3"]
    assert len(state.find_player(pool, "Zzzz")) == 0


def test_current_week_from_schedule(conn):
    from datetime import datetime

    conn.executemany(
        "INSERT INTO games(game_id, season, week, game_type, kickoff, home_team, away_team) "
        "VALUES (?,?,?,?,?,?,?)",
        [
            ("g1", 2026, 1, "REG", "2026-09-13T13:00", "A", "B"),
            ("g2", 2026, 2, "REG", "2026-09-20T13:00", "A", "B"),
        ],
    )
    assert state.current_week(conn, 2026, datetime(2026, 9, 1)) == 1
    assert (
        state.current_week(conn, 2026, datetime(2026, 9, 13, 18)) == 2
    )  # 4h after last wk1 kickoff
    assert state.current_week(conn, 2026, datetime(2027, 1, 1)) == 2
