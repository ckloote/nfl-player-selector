"""Alternative projection models, kept runnable so they can be re-measured.

Each builder matches the signature of `projections.build_projections`:

    build(frames, from_week=1, role_source=None) -> DataFrame

returning the same frame contract (`projections.PROJECTION_COLUMNS`), so it can
be handed to `backtest.weekly_projections(..., builder=...)` or
`evaluate.forecasts(..., builder=...)` and scored against the shipped model on
identical frozen data.

Every model shares the shipped model's candidate universe, availability handling
and bye-week treatment — they differ only in how `lam` is computed. That is
deliberate: a difference in eligibility would confound the comparison, and the
comparison is the entire point. See `docs/PROJECTION_BENCHMARK.md`.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from .baselines import (
    base_rate_only,
    current_season_rate,
    historical_rate,
    no_vegas,
    player_plus_vegas,
    regressed_rate,
    shuffle_within_player,
    shuffle_within_slot_week,
    vegas_environment,
)

Builder = Callable[..., pd.DataFrame]

# `None` means the shipped model — `projections.build_projections` itself.
BUILDERS: dict[str, Builder | None] = {
    "random": shuffle_within_slot_week(seed=0),
    "within-player": shuffle_within_player(seed=0),
    "historical-rate": historical_rate,
    "regressed-rate": regressed_rate,
    "current-season-rate": current_season_rate,
    "vegas-environment": vegas_environment,
    "player-vegas": player_plus_vegas,
    "shipped": None,
    # Ablations of the shipped model, used to validate the evaluation
    # instrument against effects the season backtest already sized.
    "no-vegas": no_vegas,
    "base-rate-only": base_rate_only,
}

# The bake-off proper: Models 0-6 of the benchmark plan, in report order.
BAKEOFF = [
    "random",
    "within-player",
    "historical-rate",
    "regressed-rate",
    "current-season-rate",
    "vegas-environment",
    "player-vegas",
    "shipped",
]

# Models whose ranking is degenerate or deliberately destroyed, so a paired
# comparison against them is a null test rather than a candidate comparison.
NULLS = frozenset({"random", "within-player"})


def get(name: str) -> Builder | None:
    """Look up a builder by name, or raise with the list of valid names."""
    if name not in BUILDERS:
        raise KeyError(f"unknown projection model {name!r}; choose from {sorted(BUILDERS)}")
    return BUILDERS[name]


__all__ = ["BAKEOFF", "BUILDERS", "NULLS", "Builder", "get"]
