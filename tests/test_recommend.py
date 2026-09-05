from datetime import datetime
from functools import partial

import pytest

from pool import config
from pool.recommend import advise_slot, main_slate_start
from tests.conftest import proj_row

advise_slot = partial(advise_slot, now=datetime(2026, 9, 1))

THU = "2026-09-10T20:15"
SUN = "2026-09-13T13:00"


def _proj(make_proj, thu_lam, sun_lam):
    return make_proj(
        [
            proj_row("thu", "Thursday Guy", "RB", 1, thu_lam, kickoff=THU),
            proj_row("sun", "Sunday Guy", "RB", 1, sun_lam, kickoff=SUN),
            proj_row("thu", "Thursday Guy", "RB", 2, 0.3, kickoff=SUN),
            proj_row("sun", "Sunday Guy", "RB", 2, 0.3, kickoff=SUN),
        ]
    )


def test_main_slate_is_first_sunday_kickoff(make_proj):
    proj = _proj(make_proj, 1.0, 0.9)
    assert main_slate_start(proj, 1).isoformat(timespec="minutes") == SUN


def test_small_edge_on_early_pick_says_hold(make_proj):
    proj = _proj(make_proj, 0.95, 0.90)
    a = advise_slot(proj, "RB", 1, set(), {})
    assert a.recommended.player_id == "thu" and a.recommended.early
    assert a.hold and a.hold_alternative.player_id == "sun"
    assert a.hold_alternative.cost == pytest.approx(0.05)
    assert a.recommended.deadline.isoformat(timespec="minutes") == "2026-09-10T19:15"


def test_large_edge_on_early_pick_says_commit(make_proj):
    proj = _proj(make_proj, 0.90 + config.INFO_PREMIUM_TD + 0.2, 0.90)
    a = advise_slot(proj, "RB", 1, set(), {})
    assert a.recommended.player_id == "thu" and not a.hold


def test_locked_slot_reports_lock(make_proj):
    proj = _proj(make_proj, 1.0, 0.9)
    a = advise_slot(proj, "RB", 1, set(), {1: "sun"})
    assert a.locked_player == "Sunday Guy" and a.recommended is None


def test_alternatives_sorted_by_season_cost(make_proj):
    proj = make_proj(
        [
            proj_row(f"p{i}", f"P{i}", "QB", w, lam)
            for i, lam in enumerate([2.0, 1.8, 1.2, 0.9])
            for w in (1, 2)
        ]
    )
    a = advise_slot(proj, "QB", 1, set(), {})
    assert a.recommended.cost == 0
    costs = [c.cost for c in a.alternatives]
    assert costs == sorted(costs) and all(c >= 0 for c in costs)


def test_ruled_out_player_is_neither_recommended_nor_listed(make_proj):
    """A player who is Out must not surface even when he is the best name available."""
    proj = make_proj(
        [
            proj_row("out", "Out Guy", "RB", 1, 0.0, kickoff=SUN, status="Out"),
            proj_row("fit", "Fit Guy", "RB", 1, 0.5, kickoff=SUN),
            proj_row("out", "Out Guy", "RB", 2, 2.0, kickoff=SUN),
            proj_row("fit", "Fit Guy", "RB", 2, 0.5, kickoff=SUN),
        ]
    )
    a = advise_slot(proj, "RB", 1, set(), {})
    assert a.recommended.player_id == "fit"
    assert "out" not in {c.player_id for c in a.alternatives}
    # still spendable in a week he is healthy for
    assert a.plan.pick_for(2).player_id == "out"


def test_slot_with_no_playable_candidates_recommends_nothing(make_proj):
    proj = make_proj(
        [
            proj_row("a", "A", "RB", 1, 0.0, kickoff=SUN, status="Out"),
            proj_row("b", "B", "RB", 1, 0.0, kickoff=SUN, status="Doubtful"),
        ]
    )
    a = advise_slot(proj, "RB", 1, set(), {})
    assert a.recommended is None
    assert a.alternatives == []
    assert not a.hold
