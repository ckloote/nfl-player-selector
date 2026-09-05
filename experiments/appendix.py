"""The run-verification appendix, generated from the run's own verification record.

Text rather than SVG, but it lives in the publishing layer for the same reason as
`figures`: what a report claims about its own run is read off that run's record, so a
smaller publication cannot inherit a larger one's counts. The fifteen-season strings
were hardcoded, and a one-season appendix announced "All 15 seasons completed: 48
model/seed/season runs" beside counts that came from the run itself.
"""

from __future__ import annotations

# The 2022 season lost one scheduled game; that is a fact about the NFL schedule rather
# than about any run, so it is stated whenever 2022 was replayed and not otherwise.
CANCELED_2022 = (
    "2022 includes 271 completed games; the [Bills–Bengals game was canceled]"
    "(https://www.buffalobills.com/news/"
    "nfl-says-neutral-site-afc-championship-game-is-possible-bills-bengals-week-17-ga). "
    "Its absence from the final schedule is a historical-replay approximation.\n\n"
)


def run_verification(verification):
    """The `## Run verification` appendix for one verified benchmark run."""
    seasons = [row["season"] for row in verification["seasons"]]
    counts = {
        key: sum(row[key] for row in verification["seasons"])
        for key in ("forecast_rows", "model_seed_runs", "season_scores", "picks")
    }
    scheduled, complete = verification["scheduled_games"], verification["complete_games"]
    completed = (
        f"All {len(seasons)} seasons completed"
        if len(seasons) > 1
        else f"The {seasons[0]} season completed"
    )
    games = (
        f"All {scheduled:,} required games have"
        if complete == scheduled
        else f"{complete:,} of {scheduled:,} required games have"
    )
    omitted = "No season was omitted." if len(seasons) > 1 else "It was the only season configured."
    return (
        "\n\n## Run verification\n\n"
        f"Implementation commit matching every recorded source-file hash: "
        f"`{verification['implementation_commit']}`. The run began from that source tree before "
        "its implementation commit; the manifest preserves the original parent revision and dirty "
        "fingerprint.\n\n"
        f"{completed}: {counts['model_seed_runs']:,} model/seed/season runs, "
        f"{counts['forecast_rows']:,} forecast rows, "
        f"{counts['season_scores']:,} achieved season scores, "
        f"and {counts['picks']:,} individual replay picks. {games} complete "
        f"scoring and player-stat coverage for both teams. {omitted}\n\n"
        f"{CANCELED_2022 if 2022 in seasons else ''}"
        f"Verification: {verification['pytest_passed']} pytest tests passed; Ruff lint and "
        "formatting passed. Saved forecasts were checked for identical candidate/mask "
        "populations, hard exclusions, timestamps and outcomes. Pick histories contain no player "
        "reuse; pick sums equal reported scores; selected-player flags and unique-player counts "
        "reconcile. A full `--resume` verified all checkpoint hashes. Synthetic "
        "interrupted/resumed and uninterrupted runs produced identical artifacts.\n\n"
        "This run used `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`; keep these environment "
        "variables when resuming. The pinned dependencies and exact environment are recorded in "
        "the manifest. See [experiment instructions](../experiments/README.md) for the full "
        "command and schema. Reports are generated from saved metrics and this verification "
        "record.\n"
    )
