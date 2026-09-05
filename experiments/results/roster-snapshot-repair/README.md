# Verified results: roster-snapshot-repair

These compact metrics and generated reports correspond to the resolved configuration and manifest in this directory. Individual picks are stored as deterministic `picks.csv.gz` (the exact exported CSV, compressed). Forecast matrices and the research/frozen databases remain in the ignored `data/experiments/roster-snapshot-repair` directory.

Published with `uv run python experiments/publish.py roster-snapshot-repair` after `uv run python experiments/verify.py roster-snapshot-repair` and a benchmark resume check. See [schema and reproduction instructions](../../README.md).

Matching source implementation: `7a98f03150193572515617f29f60e635f21aedb9`. Historical input timing remains approximate; both eras are retrospective. No production parameter was tuned or calibrated in this phase.
