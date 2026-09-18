"""Three hypotheses about a rival, ranked, and pure enough for the decision closure."""

import ast
from pathlib import Path

import pandas as pd
import pytest

from pool import rivals
from tests.conftest import proj_row

PAT = rivals.RivalState("pat", "Pat", frozenset({"q1"}))
OPEN = rivals.RivalState("open", "Open", frozenset())


@pytest.fixture
def week1():
    return pd.DataFrame(
        [
            proj_row("q1", "Quarter One", "QB", 1, 5.0),
            proj_row("q2", "Quarter Two", "QB", 1, 4.0),
            proj_row("q3", "Quarter Three", "QB", 1, 3.0),
        ]
    )


def test_greedy_skips_what_they_have_already_spent(week1):
    assert [c.player_id for c in rivals.predict_greedy(week1, "QB", 1, PAT)] == ["q2", "q3"]


def test_naive_ignores_the_used_pool_entirely(week1):
    """The gap between this and greedy is the only measurement of whether they track
    the one-player-per-season rule at all, so it must really ignore it."""
    assert [c.player_id for c in rivals.predict_naive(week1, "QB", 1, PAT)] == ["q1", "q2", "q3"]
    assert rivals.predict_naive(week1, "QB", 1, PAT) == rivals.predict_naive(week1, "QB", 1, OPEN)


def test_the_optimizer_hoards_where_greedy_spends():
    """A player worth more next week is kept for next week, which is the whole difference
    between the two hypotheses and the only thing that makes recording both worthwhile."""
    proj = pd.DataFrame(
        [
            proj_row("a", "Best Later", "RB", 1, 3.0),
            proj_row("a", "Best Later", "RB", 2, 5.0),
            proj_row("b", "Best Now", "RB", 1, 2.0),
            proj_row("b", "Best Now", "RB", 2, 1.0),
        ]
    )
    assert rivals.predict_greedy(proj, "RB", 1, OPEN)[0].player_id == "a"
    assert rivals.predict_optimizer(proj, "RB", 1, OPEN)[0].player_id == "b"


def test_every_predictor_keeps_at_most_top_n(week1):
    for name, predict in rivals.PREDICTORS.items():
        assert len(predict(week1, "QB", 1, OPEN, top_n=2)) == 2, name
        assert len(predict(week1, "QB", 1, OPEN)) <= rivals.TOP_N, name


def test_a_slot_with_nothing_left_predicts_nothing(week1):
    spent = rivals.RivalState("spent", "Spent", frozenset({"q1", "q2", "q3"}))
    assert rivals.predict_greedy(week1, "QB", 1, spent) == []
    assert rivals.predict_optimizer(week1, "QB", 1, spent) == []
    assert rivals.predict_naive(week1, "QB", 1, spent), "naive ignores the pool, so it still ranks"


def test_rank_of_locates_the_actual_pick_and_refuses_to_invent_one(week1):
    ranked = rivals.predict_greedy(week1, "QB", 1, PAT)
    assert rivals.rank_of(ranked, "q2") == 1
    assert rivals.rank_of(ranked, "q3") == 2
    assert rivals.rank_of(ranked, "q1") is None, "outside the kept list is not a rank"
    assert rivals.rank_of(ranked, None) is None, "an unresolved name has no rank"


def test_predict_all_returns_every_hypothesis(week1):
    every = rivals.predict_all(week1, "QB", 1, PAT)
    assert set(every) == set(rivals.PREDICTORS)


def test_an_incomplete_used_pool_is_flagged_rather_than_trusted():
    assert OPEN.pool_complete
    assert not rivals.RivalState("x", "X", frozenset({"q1"}), unknown=1).pool_complete


def _imports(module) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                found.add(node.module.split(".")[0])
            else:
                found |= {a.name for a in node.names}
    return found


def test_rivals_reads_nothing_from_the_world():
    """This module is bound for the enforced decision closure. If it ever reaches for the
    database it drags a CSV parser in with it, and a changed column header starts
    invalidating real captured decisions. The seam is the point, so it is asserted."""
    assert not _imports(rivals) & {
        "sqlite3", "db", "entrants", "standings", "predictions", "os", "pathlib", "io"
    }
