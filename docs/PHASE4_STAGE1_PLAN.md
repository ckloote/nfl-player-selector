# Phase 4, Stage 1: Opponent Ingestion

**Drafted:** 2026-09-11 (UTC), before any of it exists. Implements stage 1 of
[the Phase 4 plan](PHASE4_PLAN.md); stages 2 and 3 are unchanged by it.

The pool reports every entrant's picks once a week has resolved. This stage makes that
report land somewhere durable, identified, and re-readable. It computes nothing: no
scoring, no leaderboard, no remaining-pool query, no opponent model. Those are stage 2
and stage 3, and each of them is a different kind of mistake to make. What stage 1 owes
them is a record that is complete, attributable, and safe to re-derive from.

## The Only Deadline

The phase plan names one: **a report that arrives and is discarded is a week of evidence
gone.** Everything below follows from taking that literally rather than as encouragement.

The load-bearing consequence is that **archiving the delivered bytes is not the last step
of a successful import, it is the first step of any import at all.** `pool report import`
writes the observation and commits it, and *then* parses. A file this parser cannot read
still produces an observation, so the week survives a format surprise, a bad column
header, and a wrong guess about the schedule. A parser is a thing you can fix next
Tuesday. A report you threw away is not.

That is two transactions rather than one, deliberately. Rolling the archive back because
the parse failed would discard exactly the artifact the deadline exists to protect.

It also dissolves most of the remaining unknown. The format is still unknown, but a
format we parse badly today can be re-parsed from the archive later, because the bytes
are what was kept. The schema below does not depend on the format, and the parser is one
function behind a `--format` selector, so a second delivery format is a new function, not
a migration.

## Storage

One migration, `MIGRATIONS[4]`, additive, DDL only. No existing table is altered and no
existing row is touched.

```sql
CREATE TABLE pool_entrants (
    season       INTEGER NOT NULL,
    entrant_id   TEXT NOT NULL,   -- normalised display name; stable across weeks
    display_name TEXT NOT NULL,   -- as the report spells it
    is_me        INTEGER NOT NULL DEFAULT 0,
    first_seen   TEXT NOT NULL,   -- observed_at of the report that introduced them
    PRIMARY KEY (season, entrant_id)
);

CREATE TABLE pool_picks (
    season         INTEGER NOT NULL,
    week           INTEGER NOT NULL,
    entrant_id     TEXT NOT NULL,
    slot           TEXT NOT NULL,
    player_id      TEXT,          -- NULL while the reported name is unresolved
    player_name    TEXT NOT NULL, -- as the report spells it; never discarded
    game_id        TEXT,
    observation_id INTEGER NOT NULL REFERENCES input_observations(observation_id),
    PRIMARY KEY (season, week, entrant_id, slot),
    FOREIGN KEY (season, entrant_id) REFERENCES pool_entrants(season, entrant_id)
);
CREATE INDEX pool_picks_player ON pool_picks(season, player_id);

CREATE TABLE pool_report_totals (
    season         INTEGER NOT NULL,
    week           INTEGER NOT NULL,
    entrant_id     TEXT NOT NULL,
    reported_week  INTEGER,       -- the pool's own weekly count, if the report gives one
    reported_total INTEGER,       -- the pool's own running total, if the report gives one
    reported_rank  INTEGER,
    observation_id INTEGER NOT NULL REFERENCES input_observations(observation_id),
    PRIMARY KEY (season, week, entrant_id)
);
```

**Three tables, where the phase plan sketched two.** The report carries two grains — a
pick (entrant × week × slot) and a standing (entrant × week) — and folding the second
into `pool_picks` would repeat each total three times and invent a rule for what to do
when the three copies disagree. The third table exists because stage 2's acceptance test
is *"`pool leaderboard` reproduces the pool's own standings for a scored week"*, and that
test needs the pool's own standings written down. They arrive in the report; if we do not
store them, checking our arithmetic means re-reading a PDF by hand.

**What is deliberately absent:**

- **No `tds` column on `pool_picks`.** `my_picks.tds` is a cache that `pool score` fills.
  Giving entrant picks one would create a stored number that can disagree with both the
  report and with `scoring`, and a disagreement between our scoring and the pool's is the
  single most valuable signal stage 2 produces. It should surface as a comparison, not be
  resolvable by picking whichever column was written last. Stage 2 can decide whether it
  wants a cache; stage 1 will not pre-empt it.
- **No `recorded_at`.** `my_picks.recorded_at` records when *I* wrote a pick down. For an
  entrant pick the meaningful time is when the report was observed, and that lives on the
  observation the row already points at. A second timestamp would be a second answer.
- **`player_id` is nullable, `player_name` is not.** Discussed below.

Both `REFERENCES` clauses are documentation: this database does not run with
`PRAGMA foreign_keys = ON`, and migrations 2 and 3 declare references the same way.

## What This Does Not Touch

Per the phase plan, the report feed is **not** registered in `snapshots.TABLES` or
`freshness.FEEDS`. That is not an omission to tidy up later; it is the design.

| Mechanism | Iterates | Effect |
|---|---|---|
| `capture.observed_inputs` (`capture.py:121`) | `snapshots.TABLES` | The report is never a decision input |
| `snapshots.restore` | `snapshots.TABLES` | The report is never restored into a replay |
| `prospective.parity` | `decision_inputs` rows | Nothing to compare, so nothing to drift |
| `freshness.report` | `freshness.FEEDS` | No staleness limit, and `feed_coverage` is never asked for a table it has no entry for |

All four are correct. Opponent picks are not a projection input; restoring them into a
projection rebuild would be restoring something the projection cannot read. And a report
has no freshness semantics to begin with — it arrives once, for a week that is already
over, and it is never more or less stale than the week it describes. Registering it would
mean inventing an hours limit for a fact that does not decay.

Two visible consequences, both honest and both worth stating rather than discovering:

- `pool status` prints `snapshots.coverage`, which groups the whole observation table, so
  a `pool_report` row will appear there. It should: it is a real observation of a real
  input, and seeing that week 4's report was archived at a particular time is useful.
- `benchmark` records `observation_count` as a plain count over the same table, so it will
  include report observations. No dataset hash moves — `table_hashes` is an explicit list
  of seven tables and gains nothing — and no comparison is enforced on the count.

Because `snapshots.archive` is keyed on `TABLES[feed]`, this feed cannot use it and must
not. It writes its own archive path (below), which is the better fit anyway: for a
delivered report the raw file *is* the observation, whereas `snapshots.archive` exists to
capture the normalised state of tables that were replaced by a download.

## The One Fingerprint Move

The migration list lives in `db.py`, which is inside the enforced decision closure
(`snapshots` imports `db`). Adding `MIGRATIONS[4]` therefore moves `decision_hash` exactly
once. This is expected, was argued for in the phase plan, and costs nothing today: 3C is
closed and no protocol is running.

Two things follow for how the work is ordered:

1. **The whole schema lands in one commit, first**, rather than dribbling DDL out across
   the season. One move is a fact; five moves is a habit.
2. **The db.py change is DDL and nothing else.** No new helper, no extension to
   `backfill_game_ids`. Game-id resolution happens at import time in the new module, where
   it can be tested directly and where it cannot re-fingerprint the decision path.

`tests/test_phase3a.py` pins `benchmark.decision_modules()` — the list of paths, not their
hashes — so adding tables to `db.py` does not touch it. Adding a new module does not
either, since the closure is computed from what the decision *imports*. The one edit that
test needs is to its `outside` set, which asserts that the readers and importers are not
in the closure: `entrants` joins `cli`, `ingest`, `diagnostics`, `prospective`, `capture`
and `benchmark` there. That assertion is the point of the test, so extending it is a
review decision made on purpose.

## Identity

Two identity problems, and they fail in different directions.

### Entrants, across weeks

An entrant's used pool is the whole reason stage 3 can exist, and it is accumulated one
report at a time. If a rename splits one entrant into two, every used-pool query is wrong
for the rest of the season and nothing announces it.

`entrant_id` is the normalised display name (`state._norm`: lowercased, alphanumerics and
spaces). That makes "Chris K." and "chris k" the same entrant, which is what we want, and
makes "Chris K." and "CK" different entrants, which we cannot fix by guessing.

So the import compares the week's entrant set against the previously ingested week and
**fails on any change** — an addition, a removal, a rename — unless `--allow-roster-change`
is given. Week 1 establishes the set. The pool's entrant list is fixed after week 1 in
practice, so this tripwire should never fire; the season it does fire is the season it
saves. The failure happens after the rows are written, so the flag is a re-run, not a
re-import.

### Players, from names

The report names players as text. Resolution is
`state.find_player(state.historical_pool(conn, season, week), name, positions=config.SLOTS[slot])`
— the same function `pool record` uses, restricted by the slot the report already tells
us, which removes most of the ambiguity for free. The week is resolved, so
`player_weeks` is populated and the pool to match against is as good as it will ever be.

On exactly one match, `player_id` and `game_id` are filled. On zero or several:

- **the row is still written**, with `player_id` NULL and the reported `player_name` intact;
- the import prints every unresolved name with its candidates and **exits non-zero**;
- re-running the import re-resolves, because the PK is `(season, week, entrant_id, slot)`
  and the write is an upsert.

Dropping an unresolved entrant row would produce a leaderboard that is quietly short a
player, which is worse than a leaderboard that says it is incomplete. This is the same
discipline as *pending is not zero*, applied one stage earlier: **unresolved is not
absent.**

No alias table. If the pool spells names the way nflverse does, one is friction for
nothing; if it does not, a re-import after the mapping exists is the natural fix and the
archived bytes make it re-derivable. Build it when a real report proves it is needed.

`game_id` resolves the way `state.record_picks` resolves it — `player_weeks` for the
season/week/player first, `db.resolve_game` on the team as a fallback, NULL if neither
answers. Stage 2 joins on it to reach `scoring.touchdown_totals`.

## The Archive

A new function in the new module, not `snapshots.archive`:

```python
def archive_report(conn, season, week, raw: bytes, *, source: str,
                   observed_at=None, coverage: dict) -> int
```

- `sha256(raw)` → `input_payloads(content_hash, SCHEMA_VERSION, 'zlib-bytes', zlib.compress(raw))`,
  `INSERT OR IGNORE`, so the same file archived twice stores one copy of the bytes.
- `input_observations(season, 'pool_report', observed_at, NULL, content_hash, coverage, SCHEMA_VERSION)`.

The codec is `zlib-bytes`, distinct from `zlib-json`. Nothing reads these payloads today —
`snapshots.restore` only iterates `TABLES` and so never sees one — but if something ever
does, an unrecognised codec makes it fail loudly instead of parsing raw CSV as JSON.

`input_observations` has no `week` column, so the week goes in `coverage` alongside what
the reading contained:
`{"week": 4, "entrants": 12, "picks": 36, "unresolved": 0, "parsed": true, "source": "week4.csv"}`.
An unparseable file archives with `"parsed": false` and the counts absent. That row is the
week-4 report, and it is enough to recover everything later.

**Every import attempt is an observation.** Importing the same file three times while
fixing a parser leaves three observations and one payload. That is the honest record — a
reading is a reading, and the observation table is append-only for the same reason the
decision log is — and the content addressing means it costs one row, not one copy.

## Code Shape

A new module, `src/pool/entrants.py`, following `ingest.py`'s split: a parse step that
touches the outside world's format, pure transforms that are unit-tested, and a write that
archives in the same breath.

```python
PARSERS = {"csv": parse_csv}            # a new format is a new entry, not a migration

def parse_csv(raw: bytes) -> pd.DataFrame            # format-facing; the only part unknown
def transform_report(frame, season, week) -> tuple   # → (entrants, picks, totals); pure
def resolve_players(conn, season, week, picks)       # name → player_id, game_id; pure-ish
def archive_report(conn, season, week, raw, ...)     # committed before anything is parsed
def import_report(conn, season, week, raw, *, fmt, me=None, check=False) -> ImportResult
def entrant_picks(conn, season, week=None)           # readback, joined to its observation
def reported_totals(conn, season, week=None)
```

`ImportResult` carries the observation id, the row counts, the unresolved names with their
candidates, the entrant-set difference against the previous week, and the `my_picks`
comparison — so the CLI renders a result rather than re-deriving one.

The module imports `db`, `state`, `config` and `scoring`. Nothing in the decision closure
imports it, and nothing ever should.

### My own row

The report contains my picks too, and I already know what they should be. That makes it a
free end-to-end check on the entire ingestion path — the parser, the entrant identity, the
name resolution and the slot mapping — against a row whose right answer is already in the
database.

`--me "<display name>"` on the first import sets `is_me`. From then on, every import
compares my reported picks to `my_picks` for that week and reports any disagreement. It
does not resolve it: either the report is wrong or my record is, and both are worth a
human look. A disagreement is a warning and a non-zero exit, never an edit to `my_picks`.

### Week validation

`state.validate_week` rejects a week outside the season. A week whose games are not all
complete per `scoring.coverage` is a warning, not a rejection — the pool publishes when it
publishes, and refusing the file would violate the one deadline this stage has.

If the parser finds a week in the file and `--week` is also given and they disagree, the
import fails before writing anything except the archive.

## CLI

A `report` sub-app, matching the surface `docs/DESIGN.md` already documents:

```
pool report import <path> [--week N] [--season Y] [--format csv]
                          [--me NAME] [--allow-roster-change] [--check]
pool report list                    # reports ingested: week, observation, time, counts
pool standings [--week N]           # the pool's reported table, read back
```

`--check` archives and parses and prints, and writes no rows — which is what you want the
first time an unfamiliar file lands, and which still honours the deadline because the
archive happens regardless.

`pool standings` at this stage shows **what the pool said**: entrant, their three picks,
the reported weekly count, the reported total and rank, as of the last ingested week,
labelled as reported. It computes nothing. Stage 2 adds our computed column beside the
reported one, which is exactly where the comparison it is judged on belongs — the
reported column does not go away, it gains a neighbour.

## Tests

New `tests/test_entrants.py`. The first four are the phase plan's "done when" restated as
assertions; the rest are the claims this document makes that would be silent if wrong.

1. `transform_report` turns a reference file into the three frames, with slots mapped to
   `config.SLOTS` and unknown slots rejected.
2. An import writes entrants, picks and totals, and every `pool_picks.observation_id`
   points at the observation that carried them.
3. Re-importing the same bytes leaves the row counts unchanged, adds one observation, and
   adds no payload.
4. `entrant_picks` reads a week back joined to the observation it came from.
5. **A file that cannot be parsed still leaves an observation**, with `parsed: false`, and
   the command exits non-zero. This is the deadline, as a test.
6. An unresolvable name is stored with `player_id` NULL and `player_name` intact, is
   listed with its candidates, and exits non-zero — and a later import, after the roster
   resolves it, fills it in without touching anything else.
7. A changed entrant set fails without `--allow-roster-change` and succeeds with it.
8. Reported picks for `is_me` that disagree with `my_picks` are reported, and `my_picks`
   is not modified.
9. **The report feed is invisible where it must be**: after an import,
   `capture.observed_inputs` names no `pool_report` feed, and `snapshots.restore` succeeds
   and restores no entrant rows. This is the load-bearing claim of the whole storage
   design, and nothing else in the suite would notice if it stopped being true.
10. A week outside the season is rejected; an incomplete week warns and imports.
11. `--check` archives and parses and writes no `pool_picks` rows.

`tests/test_workflow.py` already asserts `user_version == max(db.MIGRATIONS)`, so the
migration is covered there without a new test. `tests/test_phase3a.py`'s `outside` set
gains `entrants`.

## Order Of Work

Four commits, in this order, each green on its own.

1. **Schema.** `MIGRATIONS[4]`, DDL only. The one fingerprint move, made deliberately and
   once.
2. **The module.** `entrants.py` with parse, transform, resolve, archive, write and
   readback, plus `tests/test_entrants.py`. This is where the work is.
3. **The CLI.** `pool report import`, `pool report list`, `pool standings`, and the
   `test_phase3a.py` closure-set edit.
4. **Docs.** `README.md` status; `docs/DESIGN.md` §3.1, which still says the opponent and
   standings tables are later work; `docs/IMPLEMENTATION_PLAN.md` stage status.

## Open, And Deliberately Deferred

- **The delivery format.** Still unknown, and no longer blocking: the archive takes bytes,
  the schema does not depend on the format, and `PARSERS` takes a second entry whenever the
  first report tells us what it is. The reference CSV in the test fixtures is a guess at
  the shape, and it is allowed to be wrong.
- **An alias table for player names.** Not until a real report proves nflverse spellings
  do not match.
- **Manual entry (`pool opponent record`).** `docs/DESIGN.md` names it as the fallback for
  a format we cannot parse. `--check` plus a hand-written CSV covers the same ground with
  no new code path, so it stays unbuilt until it is missed.
- **Scoring entrant picks, the leaderboard, remaining-pool queries.** Stage 2, by
  construction. `pool_picks` carries `player_id` and `game_id` so stage 2 reaches
  `scoring.touchdown_totals` without another migration.
