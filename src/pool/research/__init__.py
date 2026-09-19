"""Research: replay, benchmarks, calibration, diagnostics and decision checks.

Nothing the weekly commands run imports this package; `pool research` loads each module
only inside the command that needs it. A test holds that boundary, so a study can change
here without touching what `pool week` computes or how fast it starts.
"""
