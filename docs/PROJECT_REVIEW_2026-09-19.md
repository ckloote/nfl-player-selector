# Follow-up project review — 2026-09-19

Reviewed `47b09cf` against the September 17 review and the subsequent implementation.
The main simplifications and correctness repairs are present. Research is separated
from the weekly interface, the closed collection protocol is removed, personal
scores are computed through the shared scoring path, and `pool week` provides the
shorter workflow. The remaining problems are concentrated in new integration and
file-handling code.

The original review changed documentation only. Reproductions used disposable databases.
The operational database was opened read-only to create a SQLite backup for CLI
checks and to inspect the existing PIT migration receipt.

**Verified**

- `.venv/bin/pytest -q`: **715 passed in 141.59 seconds**.
- `.venv/bin/ruff check .`: passed.
- `.venv/bin/ruff format --check src tests`: passed; 85 files formatted.
- `week --no-refresh`, `picks`, and `standings`: successful on a temporary backup
  of the operational database at an 80-column terminal width.
- Separate fixture reproductions confirmed the five findings below. No live feeds
  were downloaded, historical experiments rerun, or newly built wheel installed.
  The suite includes the installed-package-layout regression test.

**Findings**

1. **Medium, data loss: pick export can overwrite its source database.**

   [`cli/data.py`](../src/pool/cli/data.py), line 63, unconditionally writes the CSV
   after closing the input connection. Giving `--csv` the same file as `--db`
   replaces the SQLite database with CSV and reports success. A disposable schema-only
   database reproduced this with exit code 0; reopening it returned
   `file is not a database`. This also applies to path aliases that identify the
   same file.

   Reject destinations identifying the database, including symlink/hard-link aliases.
   Prefer refusing existing output files by default, consistent with report templates
   and backups; any explicit overwrite option must still protect the database.
   This is an uncommon argument mistake with an irreversible consequence, rather
   than corruption during an ordinary export to a new CSV.

2. **Medium, data loss: backup deletes an existing file it does not own.**

   [`keeping.py`](../src/pool/keeping.py), lines 44–46, constructs
   `<destination>.partial` and unlinks it before opening the source. With source
   `backup.db.partial` and destination `backup.db`, the command deletes its source
   and then fails with `unable to open database file`. Reproduced in a temporary
   directory; the source no longer existed afterward. An unrelated existing file
   at that intermediate path is also deleted.

   Allocate a unique temporary file beside the destination and clean up only that
   owned file. Publish without overwriting an existing destination: the current
   `exists()` check followed later by `rename()` also does not enforce the documented
   no-overwrite promise if another process creates the destination in between.

3. **Medium: automatic prediction deduplication can retain an obsolete simulation.**

   [`weekly.save_predictions`](../src/pool/weekly.py), lines 267–280, decides both
   rival-pick and PIT saves from equality of the rival-pick payload, plus the
   existence of any PIT commitment for the week. Those are different inputs:
   the rival payload contains ranked top-five candidates, while PIT samples the
   complete current-week projection surface and simulation parameters.

   Reproduction: save a week with six quarterbacks, then change only the sixth
   quarterback's rate from 0.7 to 0.1. Every archived rival ranking remains equal.
   The next save returns `unchanged`, leaves one PIT commitment, and its stored
   surface still has 0.7. That player's eventual reported pick would be evaluated
   against the older distribution despite the later run having updated inputs.

   Deduplicate the two claims independently. For PIT, compare the complete current-week
   sampling inputs, including parameters, simulation count and seed, with the latest
   commitment. Also retry a failed PIT save even if an older commitment exists;
   writing the new pick prediction first currently makes that partial failure look
   complete on the next run.

4. **Medium: final game scores prematurely stop automatic refresh.**

   [`weekly.needs_refresh`](../src/pool/weekly.py), lines 36–42, treats every scheduled
   game having a score as sufficient to stop refreshing. Schedule results can arrive
   before the final touchdown feed, or while another feed has failed. The last
   week's personal picks and standings can consequently remain pending indefinitely
   under the advertised `pool week` workflow unless the user explicitly refreshes.

   A fixture with final schedule scores and incomplete touchdown coverage returned
   `needs_refresh=False`, while `ingest.settled` correctly returned `False`.
   Reuse the existing settled-season predicate instead of maintaining this weaker
   definition. Update `test_a_finished_season_is_not_refreshed`, which currently
   treats schedule scores alone as sufficient, and add the delayed-final-feed case.

5. **Low: suggested follow-up commands lose the user's execution context.**

   [`cli/reports.py`](../src/pool/cli/reports.py), line 74, prints a template's
   `report import ... --check` command without quoting the path or preserving
   `--db` and `--season`. A generated `typed report.csv` produced a pasted command
   that exited 2 with an unexpected argument. Even without spaces, using a custom
   database or season checks the report against the defaults instead.

   The weekly screen's full-detail command in
   [`cli/week.py`](../src/pool/cli/week.py), line 112, also drops `--db`; its report
   and rerun guidance has similar omissions. The actual recording command already
   preserves and quotes this context correctly.

   Use shell-quoted argument lists for actionable commands and retain explicit
   database/season options. Keep copyable commands on a single output line, as
   `record_command` already does. This needs a small shared command-formatting helper,
   not another workflow abstraction.

**What else is worth doing**

- **Make standings fit the normal terminal.** The live-data smoke check still showed
  `Jam…`, `Kei…`, truncated player names, and truncated headings at 80 columns. Eleven
  columns include three reported-value columns that were entirely empty. Default to
  rank, entrant, week total and season total; put player details and reported-value
  reconciliation in a detail view. This is the next visible workflow simplification
  after `pool week`.
- **Add a real wheel smoke test to CI.** The package-copy regression verifies the
  original source-layout failure, but does not test wheel contents, distribution
  metadata, or the installed `pool` entry point. Build/install in a temporary
  environment outside the checkout and exercise recording/capture against a fixture.
  No packaging failure was observed in this review; this is a remaining coverage gap.
- **Finish legacy PIT migration before a sampler/environment change.** New commitments
  correctly store their outcomes. Legacy `_freeze` still uses the sampler installed
  when first scored; the golden test is not a runtime compatibility check for an old
  database restored after an upgrade. The operational database's one schema-1 PIT
  commitment already has a freeze receipt, so this is a migration-hardening task for
  other databases/backups, not a current unfrozen record found in this checkout.
  Reject unsupported reconstruction or explicitly migrate under the known environment.
- **Clean up two small consistency gaps.** README's recording section still says
  replacing a player clears a cached score, although scores are now calculated at
  read time. Extending the format check to `experiments` finds one unformatted file,
  `experiments/simulator_calibration.py`; the documented and CI format scope passes.
- **Keep further scope small.** Consolidating overlapping research command orchestration
  remains unfinished from the first review, but is lower priority now that it is
  isolated. Do that when the runners need another change. Stable entrant aliases and
  membership rules can likewise wait for an actual pool case. The increased test count
  is not itself bloat: retain behavioral coverage and add focused regressions for the
  failures above rather than deleting tests to reduce the count. No additional model
  or calibration project is required to finish this cleanup.

Address the two data-preservation defects first, then prediction persistence and
refresh completeness, then command guidance and standings presentation. The original
pot-share semantic repairs, scoring consolidation, research isolation and protocol
retirement do not need another broad rewrite.

## Implementation completion — four medium findings

Findings 1–4 are addressed in this patch:

1. CSV exports exclusively create their destination and report filesystem failures with
   a nonzero exit. Existing CSVs, the source database, alternate path spellings, hard links,
   symlinks and dangling symlinks are refused without changing their contents. New-file
   exports retain the same CSV as stdout. There is no overwrite option.
2. Backups use a unique `mkstemp` file beside the destination, close both SQLite
   connections on success and failure, check integrity, and publish using `os.link`.
   Publication refuses a destination created during the copy. Cleanup removes only this
   invocation's temporary file, preserving unrelated `.partial` files and sources with
   that suffix. Unsupported hard links fail without an overwrite fallback.
3. Automatic rival-ranking and PIT saves deduplicate independently. Conditional PIT
   commits compare the complete current-week frame in row order, parameters, seed and
   simulation count against the latest schema-2 commitment with stored outcomes before
   sampling. Numeric object columns are inferred before comparison to match Parquet's
   stored dtypes. Schema-1 records do not suppress a new commitment. Failed saves retry
   during the existing window without duplicating a successful archive; actual archive
   timestamps and both kickoff/report cutoffs are retained. Status wording and the README
   cover both rankings and the distribution.
4. Automatic refresh uses `ingest.settled`, so final schedule scores alone cannot stop
   retries for missing touchdown coverage or feeds. Freshness intervals and explicit
   `--refresh`/`--no-refresh` choices remain intact.

Verification used disposable databases and mocked feeds:

- `.venv/bin/pytest -q tests/test_keeping.py tests/test_pit.py tests/test_week.py tests/test_freshness.py`:
  **112 passed**.
- `.venv/bin/pytest -q`: **753 passed in 139.97 seconds**.
- `.venv/bin/ruff check .`: passed.
- `.venv/bin/ruff format --check src tests`: passed; **85 files formatted**.
- `git diff --check`: passed.

Regression coverage includes backup copy/check/publication failures and connection cleanup,
the sixth-ranked player's changed rate, parameter/seed/simulation-count changes, ranking-only
changes, failed-save retries with an older commitment, legacy records, cutoff enforcement,
and delayed final feeds. The operational database was not used for implementation checks.

Finding 5 (command guidance), standings presentation, wheel-install CI, legacy PIT migration
hardening, version changes and release publication remain outside this patch. No database
migration or dependency was added.

## Remaining work for 1.0

The four medium findings above are complete. The remaining engineering requirements for
1.0 are command guidance and wheel-install CI; the wheel test is not the only open item.

- [ ] **Fix command guidance (finding 5).** Suggested report-import, weekly rerun and
  full-detail commands must quote paths and preserve the selected database and season.
  Verify that the printed commands run with a database/report path containing spaces and
  a nondefault season, and keep copyable commands on one line.
- [ ] **Add wheel-install CI.** Build the wheel, install it into a clean environment, and
  run the installed `pool` command from outside the checkout without source-tree imports.
  Verify distribution metadata and exercise recording/capture with a disposable fixture
  database and no live feed downloads. Keep the existing lint, formatting and suite checks.
- [ ] **Prepare and publish 1.0.** After both engineering items pass CI and merge, update
  the package version from `0.1.0` to `1.0.0` and any corresponding lockfile metadata,
  record release notes and known limitations, then tag and publish the release.

Explicitly deferred beyond 1.0:

- Standings presentation at normal terminal widths, including truncated names/headings
  and the separation of summary standings from reported-value details.
- Legacy PIT migration hardening for restored databases with unfrozen schema-1 records.
  The existing operational record has a freeze receipt. Revisit this before changing the
  sampler/environment or claiming compatibility for unfrozen legacy commitments, and
  document that limitation in the 1.0 release notes.
- The minor documentation/experimental-formatting consistency gaps and broader research
  orchestration, entrant-alias and membership refinements listed above.

These deferrals do not reopen the completed data-preservation, automatic prediction-save
or settled-season refresh fixes. This patch records the release checklist; it does not
implement the remaining requirements or publish a release.
