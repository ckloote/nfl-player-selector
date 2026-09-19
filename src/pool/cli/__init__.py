"""`pool` command-line interface.

The commands live in modules by what they do: `week` and `advice` decide, `picks` records
and scores, `reports` reads the pool's reports, `data` backs up and exports, and `common`
holds what they share. Each is registered here, in the order `pool --help` lists them.
"""

from __future__ import annotations

import typer

from .. import research
from ..research import cli as _research_commands  # noqa: F401 -- registers `pool research`
from . import advice, data, picks, reports, week

app = typer.Typer(help="NFL touchdown pool decision support.", no_args_is_help=True)
for command in (
    reports.standings,
    picks.refresh,
    advice.recommend,
):
    app.command()(command)
app.command("week")(week.week_command)
for command in (
    picks.record,
    picks.unrecord,
    picks.picks,
    picks.status,
    advice.plan,
    advice.players,
    data.backup,
):
    app.command()(command)
# Picks are scored when they are read now; `score` is kept so the old habit still works.
app.command(hidden=True)(picks.score)
app.add_typer(reports.report_app, name="report")
app.add_typer(data.export_app, name="export")
app.add_typer(research.app, name="research")
