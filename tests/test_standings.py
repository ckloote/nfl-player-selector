"""Computed pool standings share personal scoring and preserve incomplete evidence."""

import pandas as pd
import pytest

from pool import scoring
from tests import test_workflow as workflow
from tests.test_workflow import end, play, record


@pytest.fixture
def local(tmp_path):
    yield from workflow.local.__wrapped__(tmp_path)


@pytest.mark.parametrize(
    "rows,pid,slot,expected",
    [([play(), end()], "r1", "RB", 1), ([end()], "q1", "QB", 0),
     ([play()], "r1", "RB", None)],
)
def test_shared_score_matches_personal_pick(local, rows, pid, slot, expected):
    conn, _ = local
    record(conn, **{slot: pid})
    scoring.import_touchdowns(conn, 2026, pd.DataFrame(rows))
    board = scoring.score_board(conn, 2026)
    game = scoring.resolve_pick_game(conn, 2026, 1, pid, "g1")
    shared = scoring.score_pick(board, 1, pid, game)
    mine = scoring.pick_results(conn, 2026, recompute=True).iloc[0]
    assert shared.tds == expected
    assert shared.pending == mine.pending_reason
    if expected is None:
        assert pd.isna(mine.tds)
    else:
        assert type(shared.tds) is int
        assert shared.tds == mine.tds
