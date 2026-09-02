import pytest

from pool import optimizer as O


def test_each_player_used_once_and_stars_spent_in_best_week(make_proj):
    # Star is great everywhere; Role player only good in week 2. Optimal: Star wk1, Role wk2.
    proj = make_proj(
        [
            proj_row("star", "Star", "RB", 1, 1.0),
            proj_row("star", "Star", "RB", 2, 1.1),
            proj_row("role", "Role", "RB", 1, 0.2),
            proj_row("role", "Role", "RB", 2, 0.9),
        ]
    )
    plan = O.plan_slot(proj, "RB", 1, set(), {}, discount=1.0)
    assert plan.pick_for(1).player_id == "star"
    assert plan.pick_for(2).player_id == "role"
    assert plan.total == pytest.approx(1.9)


def test_bye_and_used_and_locked_handling(make_proj):
    proj = make_proj(
        [
            proj_row("a", "A", "QB", 1, 2.0),
            proj_row("a", "A", "QB", 3, 2.0),  # bye in week 2
            proj_row("b", "B", "QB", 1, 1.0),
            proj_row("b", "B", "QB", 2, 1.0),
            proj_row("b", "B", "QB", 3, 1.0),
            proj_row("c", "C", "QB", 1, 0.5),
            proj_row("c", "C", "QB", 2, 0.5),
            proj_row("c", "C", "QB", 3, 0.5),
        ]
    )
    plan = O.plan_slot(proj, "QB", 1, set(), {}, discount=1.0)
    assert plan.pick_for(2).player_id != "a"
    # 'a' used already -> never assigned
    plan2 = O.plan_slot(proj, "QB", 1, {"a"}, {}, discount=1.0)
    assert len(plan2.assignment) == 2  # only b and c remain for three weeks
    assert all(plan2.pick_for(w).player_id != "a" for w in plan2.assignment)
    # week 1 locked to 'b' -> week 1 not in plan, 'b' unavailable elsewhere
    plan3 = O.plan_slot(proj, "QB", 1, set(), {1: "b"}, discount=1.0)
    assert 1 not in plan3.weeks
    assert all(plan3.pick_for(w).player_id != "b" for w in plan3.assignment)


def test_forced_total_never_exceeds_optimum(make_proj):
    proj = make_proj(
        [proj_row("a", "A", "QB", w, lam) for w, lam in [(1, 2.0), (2, 2.5)]]
        + [proj_row("b", "B", "QB", w, lam) for w, lam in [(1, 1.5), (2, 1.0)]]
    )
    plan = O.plan_slot(proj, "QB", 1, set(), {}, discount=1.0)
    assert plan.total == pytest.approx(4.0)  # a wk2 + b wk1
    row_a = plan.players.index[plan.players.player_id == "a"][0]
    assert O.forced_total(plan, row_a, 1) == pytest.approx(3.0)
    assert O.forced_total(plan, row_a, 1) <= plan.total


def test_future_discount_prefers_using_value_now(make_proj):
    proj = make_proj(
        [
            proj_row("a", "A", "QB", 1, 1.0),
            proj_row("a", "A", "QB", 2, 1.0),
            proj_row("b", "B", "QB", 1, 0.5),
            proj_row("b", "B", "QB", 2, 0.5),
        ]
    )
    plan = O.plan_slot(proj, "QB", 1, set(), {}, discount=0.9)
    assert plan.pick_for(1).player_id == "a"


from tests.conftest import proj_row  # noqa: E402
