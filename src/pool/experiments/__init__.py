"""Alternative projection models, kept runnable so they can be re-measured.

Each module exposes `build(frames, from_week=1, role_source=None)` matching
`projections.build_projections`, so it can be handed to
`backtest.weekly_projections(..., builder=...)` and scored against the shipped
model on identical frozen data.
"""
