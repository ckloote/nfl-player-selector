"""Render and publish verified saved metrics: `publish.py <experiment>`."""

import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import appendix
import figures
import pandas as pd

from pool import benchmark

# Insert each figure group before the following section. Missing or duplicate headings
# indicate a renderer mismatch rather than a valid publication.
PLACEMENT = {
    "## Ranking Diagnostics": [
        ("bakeoff", "Season score against the baseline, with standard errors"),
        ("optimizer-by-season", "Rolling assignment minus greedy, by season"),
    ],
    "## Methods And Limitations": [
        ("reliability", "Forecast over outcome by projection bin"),
        ("calibration-slope", "Poisson calibration slope by season"),
    ],
}

experiment = sys.argv[1] if len(sys.argv) > 1 else "phase2-validation"
output = Path("data/experiments") / experiment
if not (output / "complete.json").exists() or not (output / "verification.json").exists():
    raise SystemExit(f"Run the benchmark and experiments/verify.py {experiment} first.")
verification = json.loads((output / "verification.json").read_text())
manifest = json.loads((output / "manifest.json").read_text())
complete = json.loads((output / "complete.json").read_text())
if complete["identity"] != manifest["identity"]:
    raise SystemExit("Completion record differs from the benchmark identity.")
if (
    verification["source_hash"] != manifest["code"]["code_hash"]
    or verification["source_hash"] != manifest["identity"]["code_hash"]
):
    raise SystemExit("Verified benchmark source differs from the recorded implementation.")
if (
    verification["dataset_hash"] != manifest["dataset_hash"]
    or verification["dataset_hash"] != manifest["identity"]["dataset_hash"]
    or verification["dataset_hash"] != benchmark.digest(output / "dataset.sqlite")
):
    raise SystemExit("Verified benchmark dataset changed.")
if json.loads((output / "compact" / "manifest.json").read_text()) != manifest:
    raise SystemExit("Compact metrics name a different benchmark manifest.")
spec = json.loads((output / "compact" / "resolved-config.json").read_text())
if (
    hashlib.sha256(benchmark.json_text(spec).encode()).hexdigest()
    != manifest["identity"]["config_hash"]
):
    raise SystemExit("Compact configuration differs from the benchmark identity.")
# Verification records which configuration it checked, so publication can insist it was
# this one: without it a record could attest to a narrower run than the one published.
# Bake-off records saved before the field existed carry no claim to contradict, and their
# runs cannot be re-verified under this source anyway; a calibration run must carry it.
if "config_hash" in verification:
    if verification["config_hash"] != manifest["identity"]["config_hash"]:
        raise SystemExit("Verified configuration differs from the published one.")
elif spec.get("calibration_experiment"):
    raise SystemExit("Calibration verification must record the configuration it checked.")

published = Path("experiments/results") / experiment
published.mkdir(parents=True, exist_ok=True)
for path in sorted((output / "compact").iterdir()):
    # Markdown is rendered below. Saved compact inputs may still contain retired reports.
    if path.suffix == ".md":
        continue
    if path.name == "picks.csv":
        with path.open("rb") as source, (published / "picks.csv.gz").open("wb") as target:
            with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
                shutil.copyfileobj(source, compressed)
    else:
        shutil.copy2(path, published / path.name)
shutil.copy2(output / "verification.json", published / "verification.json")

calibrated = spec.get("calibration_experiment")


def presentation(sources):
    """Rendering does not execute the recorded model implementation. Preserve its identity
    and record the current presentation sources separately, including uncommitted edits."""
    benchmark.write_json(
        published / "presentation.json",
        dict(
            schema_version=1,
            metric_source_hash=verification["source_hash"],
            dataset_hash=verification["dataset_hash"],
            source_hashes={path: benchmark.digest(benchmark.ROOT / path) for path in sources},
        ),
    )


PRESENTATION_NOTE = (
    "\n### Presentation\n\n"
    "Rendered from saved compact metrics; no forecasts, replays, or calibration fits "
    "were recomputed. `presentation.json` records the rendering-source hashes. "
    "The metric manifest and original verification record are unchanged.\n"
)

if calibrated is not None:
    # The Phase 3B report is written by the run from its own saved tables, so publication
    # copies it rather than re-rendering it: there is no second renderer to drift. The
    # bake-off's figures are drawn for the bake-off's metric set and are not reused.
    presentation(("src/pool/benchmark.py", "experiments/publish.py", "experiments/appendix.py"))
    report = (output / "compact" / "CALIBRATION.md").read_text()
    report += appendix.run_verification(verification)
    report += PRESENTATION_NOTE
    # The appendix links are written relative to `docs/`; this report lives two levels
    # deeper and is not copied there, because Phase 3B is an experiment rather than the
    # project's standing evaluation.
    (published / "CALIBRATION.md").write_text(
        report.replace("(../experiments/README.md)", "(../../README.md)")
    )
    (published / "README.md").write_text(
        f"# Verified results: {experiment}\n\n"
        "A Phase 3B chronological calibration experiment. `folds.csv` holds every fitted "
        "group in every fold with the fit that produced it and the hash of the artifact it "
        "came from; `paired-deviance.csv` is the primary forecast estimand; `policy.csv` "
        "compares each candidate against identity's replay of the same strategy; "
        "`decision-changes.csv` and `advice-changes.csv` count assignment and static "
        "hold disagreements. Individual picks are stored as deterministic `picks.csv.gz`.\n\n"
        "The generated [report](CALIBRATION.md) states facts, methods and provenance. It "
        "selects no candidate: the promotion rule is in the resolved configuration here, "
        "and applying it is a separate, dated authoring step. Production constants are "
        "unchanged and nothing in this directory deploys a calibrated model.\n\n"
        f"Published with `uv run python experiments/publish.py {experiment}` after "
        f"`uv run python experiments/verify.py {experiment}` and a benchmark resume check. "
        "Fold artifacts, out-of-fold surfaces and the frozen database remain in the ignored "
        f"`data/experiments/{experiment}` directory. "
        f"Matching source implementation: `{verification['implementation_commit']}`.\n"
    )
    print(f"Published {len(list(published.iterdir()))} artifacts to {published}")
    raise SystemExit(0)

# Figures are a rendering of the exported metrics, so they are drawn here rather than
# in the run: a caption change must not invalidate a checkpoint.
metrics = {
    name: pd.read_csv(output / "compact" / f"{name}.csv")
    for name in (
        "reliability",
        "calibration",
        "replays",
        "replay_summary",
        "ranking",
        "paired_ranking_summary",
    )
}
figure_dir = published / "figures"
figure_dir.mkdir(exist_ok=True)
for name, svg in figures.render(metrics, spec["baseline"]).items():
    (figure_dir / f"{name}.svg").write_text(svg)

presentation(
    (
        "src/pool/benchmark.py",
        "experiments/publish.py",
        "experiments/figures.py",
        "experiments/appendix.py",
    )
)


def with_figures(text, prefix):
    for marker, wanted in PLACEMENT.items():
        if text.count(marker) != 1:
            raise SystemExit(f"Expected one '{marker}' section to place figures above")
        block = "".join(f"![{alt}]({prefix}{stem}.svg)\n\n" for stem, alt in wanted)
        text = text.replace(marker, block + marker, 1)
    return text


for name, text in benchmark.render_reports(metrics, spec, manifest, output).items():
    text += appendix.run_verification(verification)
    text += PRESENTATION_NOTE
    beside = with_figures(text, "figures/")
    (published / name).write_text(
        beside.replace("(../experiments/README.md)", "(../../README.md)").replace(
            "(ANALYSIS.md)", "(../../../docs/ANALYSIS.md)"
        )
    )
    (Path("docs") / name).write_text(
        with_figures(text, f"../experiments/results/{experiment}/figures/")
    )

(published / "README.md").write_text(
    f"# Verified results: {experiment}\n\n"
    "These compact metrics and the generated [evaluation report](EVALUATION.md) "
    "correspond to the resolved configuration and "
    "manifest in this directory. Individual picks are stored as deterministic `picks.csv.gz` "
    "(the exact exported CSV, compressed). Forecast matrices and the research/frozen "
    "databases remain in "
    f"the ignored `data/experiments/{experiment}` directory.\n\n"
    f"Published with `uv run python experiments/publish.py {experiment}` after "
    f"`uv run python experiments/verify.py {experiment}` and a benchmark resume check. "
    "See [schema and reproduction instructions](../../README.md).\n\n"
    f"Matching source implementation: `{verification['implementation_commit']}`. "
    "`presentation.json` identifies the sources used to render these reports and figures; "
    "it does not replace the original metric provenance. Interpretation is maintained "
    "separately in [docs/ANALYSIS.md](../../../docs/ANALYSIS.md), not generated here. "
    "Historical input timing remains approximate; all reported eras are retrospective.\n"
)
# Retire only the current presentation copies, never the saved inputs or other experiments.
for directory in (published, Path("docs")):
    for name in ("BACKTEST.md", "PROJECTION_BENCHMARK.md"):
        (directory / name).unlink(missing_ok=True)
print(f"Published {len(list(published.iterdir()))} artifacts to {published} and docs/")
