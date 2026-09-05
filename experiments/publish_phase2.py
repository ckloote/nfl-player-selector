"""Publish verified Phase 2 artifacts without rerunning models or changing the research DB."""

import gzip
import json
import shutil
from pathlib import Path

from pool import benchmark

output = Path("data/experiments/phase2-validation")
if not (output / "complete.json").exists() or not (output / "verification.json").exists():
    raise SystemExit("Run the full benchmark and experiments/verify_phase2.py before publishing.")
verification = json.loads((output / "verification.json").read_text())
manifest = json.loads((output / "manifest.json").read_text())
if verification["source_hash"] != benchmark.code_identity()["code_hash"]:
    raise SystemExit("Verified benchmark source differs from the current implementation.")
if verification["dataset_hash"] != benchmark.digest(output / "dataset.sqlite"):
    raise SystemExit("Verified benchmark dataset changed.")

published = Path("experiments/results/phase2-validation")
published.mkdir(parents=True, exist_ok=True)
for path in sorted((output / "compact").iterdir()):
    if path.name == "picks.csv":
        with path.open("rb") as source, (published / "picks.csv.gz").open("wb") as target:
            with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as compressed:
                shutil.copyfileobj(source, compressed)
    else:
        shutil.copy2(path, published / path.name)
shutil.copy2(output / "verification.json", published / "verification.json")

counts = {
    key: sum(row[key] for row in verification["seasons"])
    for key in ("forecast_rows", "model_seed_runs", "season_scores", "picks", "empty_slots")
}
appendix = (
    "\n\n## Run verification\n\n"
    f"Implementation commit matching every recorded source-file hash: "
    f"`{verification['implementation_commit']}`. The run began from that source tree before "
    "its implementation commit; the manifest preserves the original parent revision and dirty "
    "fingerprint.\n\n"
    f"All 15 seasons completed: {counts['model_seed_runs']:,} model/seed/season runs, "
    f"{counts['forecast_rows']:,} forecast rows, "
    f"{counts['season_scores']:,} achieved season scores, "
    f"and {counts['picks']:,} individual replay picks. All 4,175 required games have complete "
    "scoring and player-stat coverage for both teams. No season was omitted.\n\n"
    "2022 includes 271 completed games; the [Bills–Bengals game was canceled]"
    "(https://www.buffalobills.com/news/"
    "nfl-says-neutral-site-afc-championship-game-is-possible-bills-bengals-week-17-ga). "
    "Its absence from the final schedule is a historical-replay approximation.\n\n"
    f"Verification: {verification['pytest_passed']} pytest tests passed; Ruff lint and formatting "
    "passed. Saved forecasts were checked for identical candidate/mask populations, hard "
    "exclusions, timestamps and outcomes. Pick histories contain no player reuse; pick sums "
    "equal reported scores; selected-player flags and unique-player counts reconcile. "
    "A full `--resume` verified all checkpoint hashes. Synthetic interrupted/resumed and "
    "uninterrupted runs produced identical artifacts.\n\n"
    "This run used `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`; keep these environment variables "
    "when resuming. The pinned dependencies and exact environment are recorded in the manifest. "
    "See [experiment instructions](../experiments/README.md) for the full command and schema. "
    "Reports are generated from saved metrics and this verification record.\n"
)
for name in ("PROJECTION_BENCHMARK.md", "BACKTEST.md"):
    text = (output / "compact" / name).read_text() + appendix
    (published / name).write_text(text.replace("(../experiments/README.md)", "(../../README.md)"))
    (Path("docs") / name).write_text(text)

(published / "README.md").write_text(
    "# Verified Phase 2 results\n\n"
    "These compact metrics and generated reports correspond to the resolved configuration and "
    "manifest in this directory. Individual picks are stored as deterministic `picks.csv.gz` "
    "(the exact exported CSV, compressed). Forecast matrices and the research/frozen "
    "databases remain in "
    "the ignored `data/experiments/phase2-validation` directory.\n\n"
    "Published with `uv run python experiments/publish_phase2.py` after "
    "`uv run python experiments/verify_phase2.py` and a complete benchmark resume check. "
    "See [schema and reproduction instructions](../../README.md).\n\n"
    f"Matching source implementation: `{verification['implementation_commit']}`. "
    "Historical input timing remains approximate; both eras are retrospective. No production "
    "parameter was tuned or calibrated in this phase.\n"
)
print(f"Published {len(list(published.iterdir()))} artifacts to {published} and docs/")
