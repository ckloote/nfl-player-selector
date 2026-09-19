"""Projection frames built by hand: one candidate at a time, and a small pool of them.

`proj_row` is one row of what `projections.build_projections` returns, with neutral
multipliers. `frame` is four quarterbacks and a back and receiver in unrelated games, for
the pot-share tests; `rival` is an opponent with nothing reported.
"""

from datetime import UTC, datetime

import pandas as pd

from pool import rivals

# Before the fixture's kickoffs, so nothing is deadline-blocked out of the candidate pool.
NOW = datetime(2026, 9, 12, 12, 0, tzinfo=UTC)


def proj_row(
    pid,
    name,
    slot,
    week,
    lam,
    kickoff="2026-09-13T13:00",
    team="AAA",
    opp="BBB",
    position=None,
    home=True,
    status=None,
):
    return dict(
        player_id=pid,
        player_name=name,
        position=position or {"QB": "QB", "RB": "RB", "FLEX": "WR"}[slot],
        slot=slot,
        team=team,
        week=week,
        opponent=opp,
        home=home,
        kickoff=kickoff,
        kickoff_known=1,
        game_id=f"g{week}",
        base_rate=lam,
        def_mult=1.0,
        vegas_mult=1.0,
        home_mult=1.0,
        avail_mult={"Out": 0.0, "Doubtful": 0.0, "Questionable": 0.85}.get(status, 1.0),
        hard_eligible=status not in ("Out", "Doubtful"),
        report_status=status,
        role_mult=1.0,
        depth_rank=1,
        lam=lam,
        prior_games=10,
        prior_tds=5,
        cur_games=0,
        cur_tds=0,
    )


def cell(pid, slot, lam, team, game, position=None, week=1):
    row = proj_row(pid, pid.upper(), slot, week, lam, team=team, position=position)
    return row | {"game_id": game}


def frame(qb_a=1.00, qb_b=0.99, weeks=(1,)):
    """Four quarterbacks in unrelated games, plus a filler back and receiver.

    `qa` edges `qb`, so both the solver and a greedy rival take `qa` -- which makes that pair
    a clean test of differentiation: near-identical in expected touchdowns, differing only in
    who else is likely to be holding them. `qz` and `qy` are exactly equal and both beneath
    what any rival would take, so they are exchangeable and nothing should order them.
    """
    rows = []
    for week in weeks:
        rows += [
            cell("qa", "QB", qb_a, "A", f"g{week}a", week=week),
            cell("qb", "QB", qb_b, "B", f"g{week}b", week=week),
            cell("qz", "QB", 0.90, "G", f"g{week}g", week=week),
            cell("qy", "QB", 0.90, "H", f"g{week}h", week=week),
            cell("q5", "QB", 0.80, "I", f"g{week}i", week=week),
            cell("q6", "QB", 0.70, "J", f"g{week}j", week=week),
            cell("ra", "RB", 0.5, "C", f"g{week}c", week=week),
            cell("rb", "RB", 0.4, "D", f"g{week}d", week=week),
            cell("fa", "FLEX", 0.5, "E", f"g{week}e", week=week),
            cell("fb", "FLEX", 0.4, "F", f"g{week}f", week=week),
        ]
    return pd.DataFrame(rows)


def rival(name, tds, used=()):
    return rivals.RivalState(name, name.title(), frozenset(used), 0, tds)
