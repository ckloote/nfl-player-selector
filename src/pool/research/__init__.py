"""Research: replay, benchmarks, calibration, diagnostics and decision checks.

Nothing the weekly commands run imports this package; `pool research` loads each module
only inside the command that needs it. A test holds that boundary, so a study can change
here without touching what `pool week` computes or how fast it starts.

The command group is created here rather than in `cli`, which registers its commands:
`research.cli` imports the options it shares from `pool.cli`, and `pool.cli` mounts this
group, so whichever of the two is imported first finds the group already made.
"""

import typer

app = typer.Typer(
    help="Replay seasons, benchmark models, check captured decisions and score predictions.",
    no_args_is_help=True,
)
