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
    return (
        "\n\n## Run verification\n\n"
        f"Recorded metric implementation commit: `{verification['implementation_commit']}`. "
        "The original manifest records the metric source and dataset identities.\n\n"
        f"{completed}: {counts['model_seed_runs']:,} model/seed/season runs, "
        f"{counts['forecast_rows']:,} forecast rows, "
        f"{counts['season_scores']:,} achieved season scores, "
        f"and {counts['picks']:,} individual replay picks. {games} complete "
        "scoring and player-stat coverage for both teams.\n\n"
        f"{CANCELED_2022 if 2022 in seasons else ''}"
        f"Original verification record: {verification['pytest_passed']} pytest tests passed. "
        f"Lint: {verification['lint']}. Formatting: {verification['format']}. "
        "These are recorded results, not checks executed by the presentation step.\n\n"
        "Recorded checks:\n\n"
        + "\n".join(f"- {check}" for check in verification["checks"])
        + "\n\nThe pinned dependencies and thread environment are recorded in the metric "
        "manifest; retain that environment when resuming model computation. "
        "See [experiment instructions](../experiments/README.md) for commands and schema.\n"
    )
