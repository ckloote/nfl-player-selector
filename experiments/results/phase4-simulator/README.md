# Phase 4 stage 3 — simulator calibration

Fits and checks the joint outcome model in `src/pool/simulate.py` against 2011–2025.
See [CALIBRATION.md](CALIBRATION.md) for the authored reading; the CSVs and `fitted.json`
are the artifacts it is written from.

Two parameters measured (`THROW_SHARE`, `RECEIVE_SHARE`), one fitted (`K_GAME` = 14.302),
four checks that are neither. Marginal spread passes at 1.029. Tail *shape* passes once the
shipped model's own calibration tilt is removed; the tilt itself is Phase 3B's finding and
is not corrected here. The quarterback/receiver coupling matches the identity that generates
it. Same-team substitution is not modelled, is measured at `var/mean = 0.884` on team
passing totals, and is deferred with its route written down.

Inputs are the frozen `data/experiments/roster-snapshot-repair` study. The operational pick
database is never opened.
