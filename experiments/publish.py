"""Publish verified artifacts without rerunning models: `publish.py <experiment>`."""

import gzip
import json
import shutil
import sys
from pathlib import Path

import appendix
import figures
import pandas as pd

from pool import benchmark

# Which figures go where, and the section each is inserted above. The markers come
# from `benchmark.reports`; a missing one means the report changed shape and the
# figure would land somewhere arbitrary, so it is an error rather than a silent skip.
PLACEMENT = {
    "PROJECTION_BENCHMARK.md": (
        "## Common-pool paired ranking diagnostics",
        [
            ("reliability", "Forecast over outcome by projection bin"),
            ("calibration-slope", "Poisson calibration slope by season"),
        ],
    ),
    "BACKTEST.md": (
        "## Achieved season scores",
        [
            ("bakeoff", "Season score against the baseline, with standard errors"),
            ("optimizer-by-season", "Rolling assignment minus greedy, by season"),
        ],
    ),
}

experiment = sys.argv[1] if len(sys.argv) > 1 else "phase2-validation"
output = Path("data/experiments") / experiment
if not (output / "complete.json").exists() or not (output / "verification.json").exists():
    raise SystemExit(f"Run the benchmark and experiments/verify.py {experiment} first.")
verification = json.loads((output / "verification.json").read_text())
manifest = json.loads((output / "manifest.json").read_text())
if verification["source_hash"] != benchmark.code_identity()["code_hash"]:
    raise SystemExit("Verified benchmark source differs from the current implementation.")
if verification["dataset_hash"] != benchmark.digest(output / "dataset.sqlite"):
    raise SystemExit("Verified benchmark dataset changed.")

published = Path("experiments/results") / experiment
published.mkdir(parents=True, exist_ok=True)
for path in sorted((output / "compact").iterdir()):
    if path.name == "picks.csv":
        with path.open("rb") as source, (published / "picks.csv.gz").open("wb") as target:
            with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
                shutil.copyfileobj(source, compressed)
    else:
        shutil.copy2(path, published / path.name)
shutil.copy2(output / "verification.json", published / "verification.json")

# Figures are a rendering of the exported metrics, so they are drawn here rather than
# in the run: a caption change must not invalidate a checkpoint.
spec = json.loads((output / "compact" / "resolved-config.json").read_text())
metrics = {
    name: pd.read_csv(output / "compact" / f"{name}.csv")
    for name in ("reliability", "calibration", "replays", "replay_summary")
}
era = metrics["replay_summary"]
era = era[era.era.eq("all retrospective")]
# A single-season run has no cross-season paired SE, so this median is NaN. There is
# no band to draw from it; the figure is drawn without one rather than shading a NaN.
median_se = era.paired_se[era.paired_se > 0].dropna().median()
floor = 2 * median_se if pd.notna(median_se) else None
figure_dir = published / "figures"
figure_dir.mkdir(exist_ok=True)
for name, svg in figures.render(metrics, spec["baseline"], floor).items():
    (figure_dir / f"{name}.svg").write_text(svg)


def with_figures(text, name, prefix):
    marker, wanted = PLACEMENT[name]
    if marker not in text:
        raise SystemExit(f"{name}: no '{marker}' section to place figures above")
    block = "".join(f"![{alt}]({prefix}{stem}.svg)\n\n" for stem, alt in wanted)
    return text.replace(marker, block + marker, 1)


for name in ("PROJECTION_BENCHMARK.md", "BACKTEST.md"):
    text = (output / "compact" / name).read_text() + appendix.run_verification(verification)
    beside = with_figures(text, name, "figures/")
    (published / name).write_text(beside.replace("(../experiments/README.md)", "(../../README.md)"))
    (Path("docs") / name).write_text(
        with_figures(text, name, f"../experiments/results/{experiment}/figures/")
    )

(published / "README.md").write_text(
    f"# Verified results: {experiment}\n\n"
    "These compact metrics and generated reports correspond to the resolved configuration and "
    "manifest in this directory. Individual picks are stored as deterministic `picks.csv.gz` "
    "(the exact exported CSV, compressed). Forecast matrices and the research/frozen "
    "databases remain in "
    f"the ignored `data/experiments/{experiment}` directory.\n\n"
    f"Published with `uv run python experiments/publish.py {experiment}` after "
    f"`uv run python experiments/verify.py {experiment}` and a benchmark resume check. "
    "See [schema and reproduction instructions](../../README.md).\n\n"
    f"Matching source implementation: `{verification['implementation_commit']}`. "
    "Historical input timing remains approximate; both eras are retrospective. No production "
    "parameter was tuned or calibrated in this phase.\n"
)
print(f"Published {len(list(published.iterdir()))} artifacts to {published} and docs/")
