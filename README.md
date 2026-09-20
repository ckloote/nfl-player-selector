# NFL Player Selector

A decision-support tool for a season-long NFL touchdown pool: each week you pick
one QB, one RB, and one WR/TE, your score is the number of TDs your picks throw or
score, and every player can only be used **once per season**. Most TDs at the end
of the season wins, winner take all; entrants tied at the top split the pot.

This turns weekly picks into a season-long resource-allocation problem: *when* do
you spend your best players, given their schedules, and how should your risk
appetite change with your position on the leaderboard?

- [`docs/DESIGN.md`](docs/DESIGN.md) — the pool rules, the model, and the system architecture
- [`docs/RESEARCH.md`](docs/RESEARCH.md) — backtesting, evaluation, calibration, decision records
  and the prediction log
- [`docs/HISTORY.md`](docs/HISTORY.md) — how the project got here: phases, reviews, outcomes and
  archived reports

Version **1.0.0** is available from [GitHub Releases](https://github.com/ckloote/nfl-player-selector/releases/tag/v1.0.0).
See the [release notes](docs/releases/1.0.0.md) for installation and known limitations.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python, dependency, and
environment management. Install it once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from the repo root:

```bash
uv sync                         # creates .venv and installs everything from uv.lock
uv run pool refresh             # pulls nflverse data into data/pool.db
```

`uv sync` provisions the right Python (see `.python-version`) if it is missing,
so there is no separate `venv`/`pip` step. Every command below is prefixed with
`uv run`, which keeps the environment in sync with the lockfile before running.
If you prefer bare commands, `source .venv/bin/activate` once and drop the
prefix, or install the CLI globally with `uv tool install .`.

An installed `pool` runs the weekly commands (`week`, `refresh`, `recommend`, `record`,
`unrecord`, `report`, `standings`) from any directory. Set `POOL_DB` to an absolute path,
because the default `data/pool.db` is relative to wherever you run it. The research commands
sit under `pool research` (`backtest`, `sweep`, `evaluate`, `benchmark`, `captures`,
`verify-capture`, `predict` and more; see [RESEARCH.md](docs/RESEARCH.md)). The studies among
them still need the checkout, because a study is reproduced from a revision.

Data comes from [nflverse](https://github.com/nflverse) via `nflreadpy`. The
default season is 2026 (override with `--season` or `POOL_SEASON`); the prior
season is always used for the start-of-season prior.

## Each week

```bash
uv run pool week                                  # this week's decisions; refreshes stale data first
uv run pool record --week 5 --rb "Kyren Williams" # the line `week` prints, once you have submitted
uv run pool report template --week 5              # when the pool's report arrives: type it in,
uv run pool report import data/reports/week5.csv  # then import it
```

`pool week` is the routine. It refreshes the data when it is older than the limits under
[scoring and data](#scoring-and-data) — an hour for the schedule and betting lines — and then
shows every slot:

```
Week 5 · 2026 · data 3 min old · times ET

 Slot  Player                            xTD   Cost  Deadline
 QB    HOLD Baker Mayfield (TB) @ DAL   1.97         Thu 7:15PM
         Dak Prescott (DAL)             2.33  -0.04  Thu 7:15PM
         Joe Burrow (CIN)               2.04  -0.09  Sun 12:00PM
         Jaxson Dart (NYG)              1.98  -0.20  Sun 12:00PM
       Hold: Baker Mayfield plays Thursday and is only 0.09 TD ahead of Joe
       Burrow (CIN, Sun 12:00PM). Wait for injury news, and take him before Thu
       7:15PM only if the gap grows.
 RB    PICK Kyren Williams (LA) vs BUF  1.10         Mon 7:15PM
         Chase Brown (CIN)              0.75  -0.13  Sun 12:00PM
         ...
 FLEX  PICK Trey McBride (ARI) vs DET   0.68         Sun 3:25PM
         ...

Nothing needs submitting before Sunday.
Submit next: RB, FLEX (first deadline Sun 3:25PM). Once submitted, record with:
  uv run pool record --week 5 --rb "Kyren Williams" --flex "Trey McBride" --decision a653…
Waiting: QB. Run `uv run pool week` again before Thu 7:15PM.
Rival rankings and distribution for week 5 saved; they count until first
kickoff, Thu 8:15PM.
Full detail: uv run pool recommend --week 5
```

- **PICK** is the recommendation: the player the rest-of-season plan takes this week. Beneath
  it are up to three alternatives, each with its **cost** — how many projected TDs the rest of
  the season loses if you take that player now instead. A higher xTD this week can still cost
  more than it gains.
- **HOLD** means the recommended player plays before Sunday but is barely ahead of the best
  later option, so waiting for injury news is worth more than the edge. Look again before his
  deadline and take him only if the gap has grown; otherwise the slot goes to the later game.
- **locked** is a slot you have recorded, with its score so far.

Every player has his own deadline, 60 minutes before his kickoff; the table shows each one.
The Sunday deadline is the first Sunday kickoff, not the 1pm block: 8:30 AM ET in a week with
a morning London game. Games with unconfirmed kickoff times are left out of this week's advice
and kept in future planning.

**Weeks with games before Sunday.** Each day with games before Sunday is its own decision;
Sunday and Monday are one. The `record` line covers the slots due on the next of those days
and names the rest, with when to come back. So on a Thursday you submit and record only what
plays Thursday, and a Sunday run picks up the rest — without the Thursday players, whose
deadlines have passed, and with Thursday's results in. Thanksgiving week (week 12) has four
such days: Wednesday, Thursday, Friday, then Sunday and Monday. A slot whose recommendation
changed since the previous run says what that run said.

**Recording.** `record` logs picks submitted elsewhere; recording is what locks a slot and
spends a player, and looking at advice never does. Paste the line `week` prints: it names the
players exactly as `record` looks them up and the decision they came from. Historical entries
and corrections are allowed, with warnings for elapsed deadlines or apparent unavailability.
Names are resolved against the requested season/week's stats and roster history, including
inactive players. Every supplied slot is validated before one atomic write. Replacing a player
clears the slot's score; re-recording the same player preserves it.

**News after you recorded is not a problem.** `pool unrecord <week> <slot>` removes the pick
and frees the player; `pool record` over the slot replaces him. Either way the history is kept.

**What is saved without asking.** Each `pool week` run with an open slot saves the decision it
showed — the inputs, the advice and the code that produced it — which is what the pasted
`--decision` names. Before the week's first kickoff and report arrival it also saves rival
rankings and the model's implied touchdown distribution. These are saved independently when
their inputs change: a rate outside the top five or a change to simulation settings can
update the distribution even when the rankings stay the same. A failed save is retried on
the next eligible run without duplicating the other archive. Nothing is backfilled after
either cutoff, so a week whose first game is on a Wednesday (weeks 1 and 12) needs a run by then.
Nothing here needs managing; [`docs/RESEARCH.md`](docs/RESEARCH.md) covers reading it back.

**More detail.** `pool recommend` is the same decision in full: matchups and multipliers, six
alternatives with where the plan would otherwise use them, the pot-share comparison, and
`--sensitivity`. Each run saves a decision too; `--no-capture` skips it.

```bash
uv run pool recommend --week 5          # the full view of the same advice
uv run pool plan                        # rest-of-season assignment
uv run pool players --pos RB            # projection table for a slot
uv run pool picks                       # your picks, scores, pending reasons and subtotals
uv run pool status --week 3             # feed attempts, coverage, freshness, fallbacks
uv run pool unrecord 3 QB               # remove an incorrect entry
uv run pool research predict score      # how the saved rival predictions have done
```

## The pot share

Once the pool's reports are imported, the advice gains a second opinion: each candidate's
expected **share of the pot**. It never replaces the expected-TD pick; `week` shows it as a
column and `recommend` shows the comparison in full, with a sentence naming why the two
disagree when they do. A tie for first splits the winnings, so a win counts 1 and a k-way tie
counts 1/k; it is a share, not the chance of an outright win. It is experimental: nothing can
establish that it wins pools, because a season is one trial. How it is computed, and what the
prediction log measures instead, is in [`docs/RESEARCH.md`](docs/RESEARCH.md).

**It is withheld, not estimated, when the season so far is not known.** Every entrant's picks
for every earlier week must be on record and final: the simulation starts at the week being
decided, so an earlier score it cannot see would otherwise count as zero. The screen says what
is missing and how to fix it, and gives the expected-TD advice unchanged:

| Withheld because | Fix |
| --- | --- |
| No entrant is marked as you | Re-import a report once with `--me "Your Name"` |
| A played week has no report, or an entrant is missing from it | `pool report import` for that week |
| An earlier pick's game is unfinished, or its result is not refreshed | `pool refresh` once the games are over |
| An earlier reported name did not resolve | Re-import that week's report once it resolves |
| An earlier week has not been played (you asked about a later week) | Ask about the next week to be decided |

A report that arrives before its week is played is used as it stands: a named pick is
simulated as that player, and a no-pick holds the slot empty, spending nobody and scoring
nothing. A name in it that did not resolve is simulated as an unknown choice, exactly as if
that report had not arrived, and the screen names it as a guess.

## Reports and standings

The pool's report arrives as a picture or a PDF, so it is typed in. Start from a template
with every entrant's name already written, type the picks, check, then import:

```bash
uv run pool report template --week 3                                 # data/reports/week3.csv
uv run pool report import data/reports/week3.csv --check             # validate, write nothing
uv run pool report import data/reports/week3.csv --me "Chris K."     # the first import names you
uv run pool report list
uv run pool standings                   # latest picks, computed season ranks
uv run pool standings --week 1          # choose the displayed picks, not the ranking cutoff
```

Keep report files in `data/reports/`. Git ignores that folder, because a report names every
entrant in the pool.

A template has one row per entrant:

```csv
week,entrant,QB,RB,FLEX
3,Chris K.,Quarter One,Runner One,Flex One
3,Pat,Quarter Two,,Flex Two
```

Type each pick under its slot; an empty cell is a no-pick (Pat's RB above). The names come
from the last imported week, the set the import compares against, so a new or departed
entrant is still a roster change to acknowledge. Your own row is left blank on purpose: type
it from the report too, so the import can catch the pool registering a different pick from
the one you recorded. `template` never overwrites a file; `--out` writes somewhere else.

The importer also reads one row per entrant and slot, and tells the two layouts apart by the
header:

```csv
week,entrant,slot,player_name,reported_week,reported_total,reported_rank
1,Chris K.,QB,Quarter One,3,3,1
1,Chris K.,RB,Runner One,,,
1,Chris K.,WR/TE,Flex One,,,
```

Either way, include every entrant and all three slots: `QB`, `RB`, and `FLEX` (`WR`, `TE`,
and `WR/TE` also mean `FLEX`, as a slot value or a column header). `entrant` must be filled
in on every row, and so must `slot` in the long layout. Supply `week` on every row or use
`--week`; when both are present they must agree. An optional `season` column must agree
with `--season`. The reported week TDs, season total and rank (`reported_week`,
`reported_total`, `reported_rank`) are optional integer columns in either layout; in the long
one, put them on one row per entrant or repeat consistent values. Missing totals stay
unknown. Duplicate slots, conflicting totals, and unknown columns are rejected. The full
[reference fixture](tests/fixtures/pool_report.csv) includes two entrants.

An entrant who submitted nothing for a slot is a no-pick: an empty cell, or in the long
layout a row with `player_name` empty. `standings` shows it as `(no pick)`, distinct from a
name that could not be resolved and from a week that was never imported. A missing slot
column, or a dropped slot row, is still rejected, because nothing distinguishes it from a
truncated file.

Every import commits the original bytes before parsing, including imports that fail.
Re-imports add an observation while sharing the same stored payload and updating the
week's picks. Parse outcomes live separately in metadata so the original observation
remains immutable. `--check` is a dry run: it validates and prints counts and issues but archives
and writes nothing, so checking a file repeatedly while you fix it leaves no trace. A report
can be imported as soon as it arrives, even before that week's games finish; re-importing the
week later replaces it in place.

Unresolved player names are retained and listed with candidates; re-import after updating
the roster to resolve them. A name that is not an exact match (ignoring case and
punctuation) but fits exactly one player by a partial name or a close spelling is imported
and printed as a note showing what you typed and who it matched, in both `--check` and the
import. Read those notes: a typo can match the wrong player. Entrant names are normalized
across weeks.

Additions, removals and renames leave the week exactly as it was and exit nonzero until
acknowledged with `--allow-roster-change`; review those differences before acknowledging them.
A correction that drops an entrant and a delivery that was truncated look identical, so nothing
is written until you say which it is; acknowledging then replaces that week's entrant set while
preserving the archived originals. An entrant left with no picks at all after a correction
was a misspelling, and is no longer counted anywhere: not in `standings`, and not as an
opponent in the pot share. That includes your own row: if a corrected report spells your name
differently, pass the new spelling with `--me` on that same import.

`--me` identifies your row once, then every import that contains your row compares it with
your recorded picks; a file without your row is a roster change, not a mismatch. Standings
work without it, but the pot share and the rival predictions do not: without it every entrant
reads as a rival, you included. An import that leaves no entrant marked as you says so. Include
yourself: `standings` only lists entrants in the report, and the comparison is what catches the
pool registering a different pick than the one you recorded. A slot you have not recorded is a
note; a slot where your
record and the report name different players — or where the report says you picked nobody —
is a mismatch. Unresolved names, unacknowledged roster changes, and mismatches exit nonzero.
A report import never edits your recorded picks. Incomplete game coverage warns but permits
the import.

**Standings** put computed ranks, weekly TDs, season TDs and used-player counts beside the
supplied weekly totals, season totals and ranks. Reported values remain passthrough; the
pool's organizer uses the same nflverse source, so agreement would not be independent
validation. Picks default to the latest imported week. Everyone is ranked through the last
consecutive completed week starting at week 1; a gap holds the cutoff back, and the title
states both weeks. Later reported picks appear separately as in progress, and so do your
recorded picks for a week with no report yet. Where a report differs from your records, it is
shown with a note to review the `pool report import --me` comparison.

A missing report leaves three missing slots, never three zeroes, and subtotals stay
incomplete while any gap exists; ranks are provisional in that case. Tied entrants share
competition ranks (1, 2, 2, 4), and a tie for first splits the pot equally; a provisional tie
describes a split at the end, not a settled result. A reported no-pick is a final zero, and an
unresolved name needs re-importing. Used players
include every imported week, even one still being played; each entrant has an independent set,
unresolved names add an unknown count, and repeated players are counted once with their weeks
reported. Your own picks are scored from the results each time they are read, so with matching
recorded and reported picks, `pool picks` and standings show the same TD count.

## Scoring and data

Scoring counts every touchdown thrown or scored, including returns and recoveries; two-point
conversions do not count. The importer uses nflverse's explicit scorer identifier and
separately credits the passer on a credited passing touchdown, excluding negated plays and
conversions. See the
[official play-by-play field definitions](https://nflfastr.com/reference/fast_scraper.html).
A game is complete only with an end-of-game marker, terminal scores matching the
schedule, and resolved touchdown identities. A complete game with no credits for
a pick scores **0**, including a player who did not play. Once every game in a week is
final, a pick with no game of its own also scores **0** — the pool treats a pick on a
player who is not playing as worth nothing — and says so, because the other way to reach
that state is a team the schedule does not match. Missing or incomplete feeds remain
**pending**, and so does a missing game while its week is still unfinished. Weekly and season
subtotals are labeled incomplete while any picks remain pending. Nothing is stored: a pick's
touchdowns are read from the results whenever it is shown, so a correction upstream appears
after the next refresh. `pool score`, which used to fill a stored copy, is kept as another
name for `pool picks`.

Refreshes bypass nflreadpy's cache, attempt independent feeds, and retain the previous
dataset on download, parsing, or missing-file failures. A refresh covers the current season
and the one before it, whose statistics are the model's starting history. Once that prior
season is settled (every game final with complete touchdown coverage, and every feed loaded
at least once), a refresh records only its schedule and skips its other downloads;
`pool refresh --full` downloads it again, for upstream stat corrections. `pool refresh` exits nonzero on a
partial failure, while expected unpublished preseason results are informational; `pool week`
reports a failure in one line and still advises from the data it has. Its automatic refresh
continues at the usual freshness intervals until the current season is settled, including
complete touchdown coverage, player statistics and a successful load of every feed. Final
schedule scores alone do not stop retries. `week --refresh` and `week --no-refresh` still
override the automatic choice. Advice continues with warnings when usable schedule and
player history exist. `status`, `recommend`, `week` and `plan`
distinguish missing coverage and model fallbacks from fetch age. Default age
limits are 1 hour for schedule/lines and 24 hours for stats, rosters, injuries, depth charts,
and touchdown feeds. Configure these through `POOL_FRESHNESS_SCHEDULE_HOURS`,
`POOL_FRESHNESS_PLAYER_STATS_HOURS`, `POOL_FRESHNESS_ROSTERS_HOURS`,
`POOL_FRESHNESS_INJURIES_HOURS`, `POOL_FRESHNESS_DEPTH_CHARTS_HOURS`, and
`POOL_FRESHNESS_TOUCHDOWNS_HOURS`. Completed historical seasons are exempt from age warnings
and are not refreshed by `week`.

SQLite upgrades run transactionally on open, preserving imported history and picks. New
import metadata uses UTC; kickoff display remains Eastern. Existing databases lack complete
touchdown coverage until refreshed. Live projections may use legacy
offensive-TD estimates with a warning where touchdown coverage is incomplete; finalized scoring
and replay comparisons require complete coverage. Report archives appear in `status` but are
excluded from projection inputs, snapshot replay, and feed freshness checks.

## Keeping your data

The nflverse feeds can be downloaded again, but your recorded picks, the pool's archived
reports and the decisions `pool week` saved exist only in `data/pool.db`. Back it up after
each report import:

```bash
uv run pool backup                          # data/backups/pool-20260920-2215.db
uv run pool export picks --csv picks.csv    # your picks and what each scored, as CSV
```

`backup` copies the database with SQLite's online backup, checks that the copy reads back
whole, and never overwrites a file (`--to` names another). Git ignores `data/backups/`, as it
does the database, so copy that folder somewhere off this machine now and then. To restore,
copy a backup over `data/pool.db` while no `pool` command is running.
The destination filesystem must support hard links, which publish the verified copy
without overwriting another file; otherwise the backup fails and removes its temporary copy.

`export picks` writes one row per pick: week, slot, player, team and position, then the
touchdowns, or why the pick is still pending. Without `--csv` it prints the same thing.
The CSV destination must be a new file: exporting again requires another filename or
manually removing the previous CSV. Existing files and symlinks are always refused.

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check src tests
```

CI runs these three checks, after `uv sync --locked`, on every push to `main` and every
pull request (`.github/workflows/ci.yml`). `--locked` fails when `uv.lock` does not match
`pyproject.toml`, so a dependency change has to commit its updated lockfile.

A separate `wheel-install` job builds a wheel and installs it with its runtime dependencies
in a temporary environment. From outside the checkout it checks package metadata, the
installed `pool` entry point, and a historical fixture pick and its captured identity.
It uses no live feeds and installs no development dependencies. Run the same check locally:

```bash
(
  set -e
  wheel_dir="$(mktemp -d)"
  trap 'rm -rf "$wheel_dir"' EXIT
  uv build --wheel --out-dir "$wheel_dir"
  uv run --no-project --python "$(cat .python-version)" python tests/wheel_smoke.py "$wheel_dir"/*.whl
)
```

The standalone driver also accepts an existing wheel path; it cleans up its temporary
environment and database on success or failure. The package-copy pytest regression remains
a fast local check.

The suite runs as of 18 September 2026, week 2 of the season its fixtures describe, whatever
today's date is: `tests/conftest.py` pins the clock with `time-machine`.

Tests are grouped by what they cover. Everything under `tests/research/` exercises the research
code — replays, benchmarks, calibration, diagnostics, published reports and the
`verify-capture` checks — and is marked `research`. For the everyday code alone, which takes
about 40 seconds rather than two and a half minutes:

```bash
uv run pytest -m "not research"
```

Shared builders (the synthetic seasons, the small 2026 database, reports and a captured
Friday decision) live in `tests/support/`; test modules import from there, never from each
other.

`pytest`, `ruff` and `time-machine` live in the `dev` dependency group, which `uv sync`
installs by default; `uv sync --no-dev` gives a runtime-only environment.

Dependency changes go through uv so that `uv.lock` stays authoritative — it is
committed, and it is what pins the exact versions everyone gets:

```bash
uv add scikit-learn             # add a runtime dependency
uv add --dev pytest-cov         # add a dev-only dependency
uv remove pandas                # drop one
uv lock --upgrade               # refresh every pin
uv sync                         # apply the lockfile to .venv
```
