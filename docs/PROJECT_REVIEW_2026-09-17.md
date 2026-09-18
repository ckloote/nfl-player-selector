# Project review — 2026-09-17

Review of `895f46c` (the Phase 4 Stage 3 merge), covering implementation, documented
scope, correctness, research infrastructure, tests, and the weekly user workflow.
This is a review and proposed direction; no application code, model settings, saved
studies, or operational data were changed.

The weekly application is substantially implemented. Its main excess is the amount of
research administration exposed as part of using and maintaining it. There are also
correctness gaps in the newer pot-share integration that should be fixed before
relying on that view. Simplification should preserve the scoring and eligibility
rules while reducing the number of workflows and representations that must agree.

**Verification and limits**

- Full suite: **675 passed in 152.77 seconds** using `.venv/bin/pytest -q`.
- Lint: `.venv/bin/ruff check .` passed.
- The documented formatting check, `.venv/bin/ruff format --check src tests`, failed:
  **18 files would be reformatted**, 33 were already formatted.
- Exercised `recommend --no-capture`, `picks`, and `standings` against a temporary
  SQLite backup. The original database was opened read-only for that backup.
- Reproduced the entrant-state defects below with synthetic fixtures and checked
  the installed-package layout separately. No live feeds were refreshed, messages
  sent, or full historical experiments rerun.
- A wheel build was attempted but the environment's Snap-packaged `uv` could not
  start because of its container capabilities. The packaging finding was reproduced
  by copying the package into an isolated `site-packages` layout, not by installing
  a newly built wheel.

**Findings to address first**

1. **High: incomplete past scores silently become zero in pot-share advice.**

   In [`predictions._decision_outcomes`](../src/pool/predictions.py), lines 228–241,
   `pick.tds or 0` and `result.tds or 0` collapse pending scores into final zeroes.
   The simulation starts at the requested week, so those missing earlier outcomes
   are not simulated elsewhere. This loses the careful incomplete-total semantics
   implemented in `standings.py`.

   Reproduction: a fully resolved rival with one banked TD became a zero-TD rival
   after its prior game's coverage was marked incomplete. `pool_complete` remained
   true, because it measures known used players, not score completeness. The normal
   pot-share warning path consequently has no way to distinguish this from a real
   zero. Looking ahead while the preceding week is unfinished triggers the same
   underlying problem.

   Carry scoring completeness into `PoolState`. The simplest behavior is to withhold
   pot-share advice when required earlier scores are unresolved, explain the missing
   inputs, and continue giving expected-TD advice. Simulating unresolved earlier
   picks would be a larger alternative, not necessary for this repair.

2. **High: an import without `--me` can make you your own opponent.**

   [`entrants.import_report`](../src/pool/entrants.py) permits the initial import
   without identifying the user's row. [`predictions.pool_state`](../src/pool/predictions.py),
   lines 195–201, treats every row without `is_me` as a rival. A report containing
   `Me` and `Rival`, imported successfully without `--me`, produced two rivals,
   `me` and `rival`, in addition to the simulated personal entry.

   Require an established personal identity before enabling the pot-share and rival
   prediction views, or explicitly support a configured rivals-only report mode.
   Make this a one-time setup choice, with an actionable prompt when it is missing.
   Expected-TD recommendations should still work.

3. **Medium: reported no-picks are replaced with simulated picks.**

   [`predictions.known_picks`](../src/pool/predictions.py), line 111, discards rows
   without a player ID. This includes a report's explicit empty slot, which the
   importer and standings correctly distinguish from an unknown pick.
   [`rivals.rollout`](../src/pool/rivals.py), lines 219–227, then fills any unpinned
   week with a predicted player.

   Reproduction: an imported rival with blank RB and FLEX slots had neither slot
   pinned. The rollout therefore remained free to assign players to both. This
   overstates that rival's remaining scores and spends players they did not use.

   Preserve three states across the simulation boundary: unknown choice, known
   player, and known no-pick. A known no-pick must hold the slot at zero without
   consuming a player. An unresolved reported name should remain visibly uncertain.

4. **Medium: correcting an entrant name leaves a phantom opponent.**

   [`entrants._write_report`](../src/pool/entrants.py), lines 336–344, removes obsolete
   pick and total rows while preserving entrant identities.
   [`standings._board`](../src/pool/standings.py), line 239, subsequently selects every
   historical identity, and `pool_state` simulates every non-personal one.

   Reproduction: imported `Me`/`Rivla`, then corrected the same report to
   `Me`/`Rival` with `--allow-roster-change`. Both imports succeeded. The corrected
   report contained two entrants, but standings contained three and the simulator
   included both `rivla` and `rival` as opponents.

   Separate stable entrant identity from display names, with a small alias/correction
   mechanism. At minimum, an identity with no remaining picks after a correction
   must not become an active competitor. Preserve the raw archived reports; genuine
   departures with historical picks need an explicit membership rule rather than
   deletion of their history.

**Status of findings 1–4:** resolved on branch `claude/review-findings-1-4`.

- `2d628a5` (finding 1): pot share is withheld, with named reasons and fixes, until every
  entrant's earlier weeks are final. This includes looking ahead past an unfinished week.
- `dbf3de5` (finding 2): pot share and `predict record` require an identity. An import
  without one still succeeds, with a warning.
- `98dae12` (finding 3): a reported no-pick holds its slot empty. An unresolved name is
  simulated as unknown and named as a guess.
- `9c73dc8` (finding 4): the minimal fix, not the alias mechanism. An identity with no
  reported picks is not an entrant, and `--me` can move to a corrected spelling.

Decided and deferred:
- A `report alias` command waits for the report-format enhancement below.
- There is no membership rule for mid-season departures or late joiners. Either one now
  withholds pot share every later week, naming the entrant, until a rule follows the
  pool's actual policy.

`verify-capture --allow-code-drift` on a copy of the live database verifies 5 of 8 2026
captures, identically before and after these changes. The failures predate them:
`8edc0f08b08e` and `fd9c03a2946a` (week 2, re-derived advice differs) and `cf9cc9e5d64d`
(week 3, finding 7).

5. **Medium: the documented global installation is incompatible with normal capture.**

   The README recommends `uv tool install .`. But
   [`benchmark.code_identity`](../src/pool/benchmark.py), lines 166–179, assumes a
   repository layout containing `src/`, `pyproject.toml`, `uv.lock`, and a usable
   Git checkout. [`capture._code_identity`](../src/pool/capture.py) calls it during
   normal `recommend`, `record`, and `unrecord` operations.

   In an isolated installed-package layout, `code_identity()` failed with
   `FileNotFoundError` for `pyproject.toml`. A source copy without Git has a further
   failure path. `--no-capture` can avoid this for recommendations but does not make
   the documented installed workflow complete.

   Move runtime identity into a small package-aware module. Use installed source or
   build metadata and actual dependency versions; attach Git information when
   available. Keep research-run identity stricter and separate. Add one installed
   distribution smoke test that records a synthetic pick outside the source tree.

6. **Medium: the prospective simulator commitment does not freeze the simulator.**

   [`pit.commit`](../src/pool/pit.py), lines 73–80, saves parameters, seed, simulation
   count, and a projection hash. [`pit._draws`](../src/pool/pit.py), lines 93–103,
   always runs the current `simulate.sample`. It neither preserves the resulting
   draws nor validates the sampler implementation or dependency versions. A seed
   fixes results only under the same algorithm and environment.

   Reproduction: rebuilt an archived commitment, replaced the sampler with a changed
   implementation, and rebuilt the same commitment again. Different draws were
   accepted without a drift warning. Thus future simulator improvements can silently
   change the evidence used to judge earlier forecasts.

   For this small weekly workload, storing compressed sampled player outcomes is
   the simplest robust choice. Alternatively preserve a versioned sampler and its
   environment, and reject unsupported reconstruction. Avoid growing a second
   independent source-fingerprint system.

7. **Medium: capture verification imposes a different input contract from live advice.**

   [`projections.load_frames`](../src/pool/projections.py), lines 124–138, requires
   complete earlier-week scoring in restored snapshots. Live loading permits
   incomplete scoring and documented offensive-stat fallbacks. Consequently a valid
   live decision made with incomplete history, or an ahead-of-week preview, can be
   impossible to verify even if its actual inputs were faithfully archived.

   The Stage 3 verification record documents this failure for a week-3 capture, then
   says it will verify once week 2 is scored. That does not follow from the code:
   [`snapshots.restore`](../src/pool/snapshots.py) selects observations at or before
   the original decision timestamp. A later refresh cannot repair missing coverage
   in that old snapshot. The README also incorrectly says parity requires the
   captured week's own results; the gate is on earlier weeks.

   Separate faithful input reconstruction from research outcome-completeness gates.
   Replay the input/fallback policy actually used, and assess outcome coverage
   separately when scoring an experiment. Correct the documentation; do not imply
   that waiting will change immutable inputs.

**Status of findings 5–7:** resolved on branch `claude/review-findings-5-7`, stacked on
`claude/review-findings-1-4`.

- `92daa69` (finding 5): capture fingerprints the running package through a new
  `identity` module. The enforced hash covers the decision closure plus installed runtime
  dependency versions, instead of `uv.lock`. Git is attached when present.
  `benchmark.code_identity` stays strict for research. A wheel built with `uv build` and
  installed outside the checkout records picks and captures decisions, and its
  fingerprint equals the checkout's.
- `a82579c` (finding 6): PIT commitments store their drawn outcomes. A schema-1
  commitment is frozen the first time `predict score` reads it, and a test pins
  `simulate.sample`'s output. On a copy of the live database, 2026 week 2 froze to
  exactly what it rebuilt to before (the same digest). The live database freezes it on
  its next `predict score`.
- `48b6e73` (finding 7): snapshot replay reconstructs by default, and research replays opt
  into the coverage gate. `cf9cc9e5d64d` now gets past "archive cannot be resolved", and
  its replayed surface matches the stored one exactly.

The three captures that still fail `verify-capture` (`cf9cc9e5d64d`, `8edc0f08b08e`,
`fd9c03a2946a`) are code drift, not missing evidence. All three carry a pot share. All
three reconstruct exactly under `89dca93`, the revision two of them recorded; the third
was captured from a dirty tree at `4dcd1d5`. Their expected-TD picks are unchanged today;
only the shares differ, because `cc8565f` changed the pot-share arithmetic.

**Completion against the documentation**

| Area | Assessment |
| --- | --- |
| Refresh, projections, assignment, deadlines, recording, scoring | Implemented, with substantial regression coverage. Preserve the core safeguards. |
| Historical evaluation and Phase 3B calibration | Implemented; the recorded no-promotion outcome is a completed experiment, not unfinished implementation. Full experiments were not rerun for this review. |
| Phase 3C | Tooling implemented; collection deliberately closed without running. The roadmap's overview and opening Phase 3 status still say collection is pending, contradicting its own later section and the outcome note. |
| Reports and standings | Implemented; identity correction needs repair. The README still calls the input format provisional and unconfirmed against the actual delivery format. |
| Pot-share advice and predictions | Implemented, with the integration defects above. Model assumptions remain provisional; implementation completion is not evidence of a policy advantage. |
| Multi-event replay, learned opponent behavior, simulator substitution refinement | Explicitly deferred. These are possible future work, not reasons to continue expanding the current roadmap automatically. |

The simulator report is candid about miscalibration and failed/partial checks. Retain
that distinction. The operational display should call pot share a model estimate and
identify its displayed error as Monte Carlo precision, not overall forecast accuracy.
Also replace the categorical “the season is decided” in `cli._share_notes`: identical
outcomes in 2,000 samples do not establish mathematical certainty.

**Where the size comes from**

Counts include comments and docstrings; they measure maintenance surface, not executable
complexity or proof of unnecessary code.

| Surface | Current size |
| --- | ---: |
| Python under `src/pool` | 14,362 lines |
| `backtest`, `benchmark`, `calibration`, `diagnostics`, `evaluate`, `prospective`, and `models/` | 6,380 lines, about 44% of source |
| `cli.py` | 2,093 lines |
| `prospective.py`, for the closed collection workflow and verification | 1,398 lines |
| `optimizer.py` | 154 lines |
| Python tests | 10,816 lines; 675 collected cases |
| Main CLI | 21 command/group entries, including nine research/audit entries |

The optimizer is already compact. Replacing it with greedy selection would give up
planning and opportunity-cost explanations while saving little code. The historical
study's uncertain optimizer advantage is not proof of equivalence or a reason to
discard those useful features. SQLite and a local CLI are also appropriate choices;
there is no reason to introduce a service, ORM, plugin framework, or additional
dataframe-engine rewrite to simplify this project.

**What to keep, isolate, and retire**

| Action | Scope | Benefit and constraint |
| --- | --- | --- |
| Keep | Deadline eligibility, no reuse, atomic multi-slot recording, correction history, explicit pending scores, report archives, shared scoring | These prevent wrong picks or totals. They are product behavior, not test scaffolding. |
| Keep quietly | Compact decision records and deduplicated input snapshots | Useful for explaining a past recommendation. Routine browsing should not create an obligation to administer a research protocol. |
| Isolate | Evaluation, calibration, benchmark workers, diagnostic exports, model registry, sensitivity/replay studies | Put them in a `research` package/command group with lazy imports. This reduces everyday coupling and help clutter; moving code alone does not reduce total lines. |
| Retire from the active workflow | Phase 3C window/floor enforcement, collection-event accounting, population and submission audit exports, `baseline` | This is the clearest actual deletion opportunity because collection was explicitly closed. First extract useful reconstruction/parity checks, then preserve the historical protocol, reports, and a reproducible revision. |
| Consolidate | Overlapping `backtest`, `evaluate`, `sweep`, and `benchmark` configuration/orchestration | Keep a quick replay path and one specified experiment runner. Remove duplicated option parsing/report orchestration rather than the distinct underlying metrics. |
| Simplify | Personal cached scores versus computed entrant scores | Prefer one read-time scoring path. Consider retiring routine cache maintenance and making `score` a compatibility alias; preserve any explicitly desired “last scored” history separately. |
| Archive in documentation | Completed phase plans, old review timelines, superseded result narratives | Keep provenance accessible from a history index. The main README should teach the current workflow. |

Split `cli.py` by responsibility—weekly commands, reports/standings, research, and
rendering—and keep orchestration thin. Extract package identity from `benchmark.py`
before doing that split: currently capture drags research machinery back into the
weekly path. Existing private cross-module calls, including
`standings._score_rows` and `capture._store_surface`, identify small shared functions
that deserve explicit ownership; they do not require generic abstraction layers.

Do not delete saved studies or change their recorded identities to make a refactor
appear compatible. Retain the exact historical revision/environment for reproducing
old runs. A future study can use the refactored runner under a new identity.

**A simpler weekly workflow**

The current README starts with a large command menu and then adds refresh/capture
rituals, decision IDs, replay verification, and prediction collection. It still says
“Refresh immediately before every capture, no exceptions” and “Use `--no-capture`
to look” before later explaining that the governing protocol is closed. Remove those
obsolete obligations from the weekly guide.

A proposed default workflow is:

```text
pool week                         refresh and show the next decisions
pool record --rb "Player Name"    log what was submitted to the pool
pool report import report.csv     update opponents and computed standings
```

`pool week` is a proposed command, not something implemented by this review. It can
be a thin wrapper around existing functions rather than another independent workflow.
Alternatively add `recommend --refresh` and use that as the primary entry point.

The weekly screen should show season/week, next actual pick deadline, three slot
states (locked, pick, hold), the recommended names, and the immediate next action.
Show two or three alternatives with full names; put multiplier details, long future
plans, sensitivity tables, and capture metadata behind explicit detail options.
Preserve per-player deadlines; the collection protocol's kickoff-wave grouping must
not become a new rule that prevents later valid picks.

In the local smoke check, one open slot produced **47 output lines**, and the
alternatives table truncated player names at an 80-column width. Recommendation
computation took approximately **2.75 seconds** on that copied database; improving
presentation is more urgent than replacing the solver for speed. This is one local
measurement, not a general performance benchmark.

Offer a copy-ready recording command, or a stored-recommendation selection action,
so the user need not retype names and research IDs. Recording must continue to mean
logging an actual external submission; looking at advice must not lock a pick.

If prospective opponent measurement remains desired, archive it automatically once
per relevant input revision before first kickoff, as part of the weekly workflow.
After the window closes, give a concise status instead of expecting another manual
command to produce an unscorable record. Keep prediction scoring in the research
detail view. This retains useful evidence while eliminating a recurring chore.

**Can the tests be pared down?**

Yes, in organization and in features they support. There is no evidence that deleting
working regression checks wholesale would improve this project. A complete run takes
about two and a half minutes, and the new findings show that broad coverage still
misses some important handoffs.

- Keep tests of deadlines, no reuse, rollback, missing-versus-zero results, complete
  TD accounting, report correction, deterministic reconstruction, and data leakage.
- Separate core and research tests by directory/marker. A weekly-interface change
  should have a short focused test command, while the full suite remains available.
- Move shared fixtures out of individual test modules. For example, `conftest.seeded`
  imports `_seed` from `test_backtest`, and several later suites import helpers from
  `test_phase2`/`test_workflow`. Retiring a study should not break unrelated setup.
- Rename phase-number suites by the behavior they protect. Phase boundaries are
  historical context, not a useful long-term map of the code.
- Remove tests when their closed-protocol feature is intentionally retired. Consolidate
  repeated fixture setup and equivalent parameter cases; retain distinct failures.
- Replace exact source-layout expectations as appropriate when identity is redesigned.
  Tests of what changes an identity are more durable than permanently pinning the
  set of module names. Keep the underlying reproducibility checks.
- Add focused tests for the findings above and one installed-package workflow test.
  These exercise boundaries the current suite misses, rather than mirroring functions.

A minimal CI check for lint, formatting, and the suite would prevent the documented
format command from drifting again. No CI configuration is present in this checkout.
Formatting should be a separate mechanical change because source hashes are currently
part of capture identity.

**Enhancements worth doing after the repairs**

1. **Match the actual report delivery format.** Validate with a representative redacted
   report, make the normal import one step, and retain the existing review output for
   ambiguous names. Stable aliases and one-time personal identity setup also repair
   the most consequential ingestion friction.
2. **Show data readiness where the decision is made.** Distinguish missing reports,
   unresolved scores, stale inputs, and an experimental pot-share estimate. Offer the
   one command that fixes each problem without flooding normal output with provenance.
3. **Avoid repeatedly downloading settled prior seasons.** `refresh` currently attempts
   every feed for both years, including play-by-play. Bootstrap prior history once,
   refresh current data normally, and offer explicit full/historical refresh for
   corrections. Do not lose the ability to correct old scoring.
4. **Add a simple backup/export path.** Picks, entrant aliases, and archived reports are
   more valuable than disposable downloaded feeds. A documented SQLite backup and
   human-readable pick export are sufficient initially.
5. **Improve probability estimates only through a bounded experiment.** The saved
   simulator report identifies inherited rate bias and imperfect dependence. Keep
   expected-TD advice primary while collecting useful opponent evidence. Do not
   automatically reopen calibration searches or add a learned model because the
   infrastructure exists.

**Recommended order**

First repair pot-share input semantics (findings 1–4), then package/capture integrity
(findings 5–7). Next simplify the README and weekly display, separate research code,
and remove the closed-protocol workflow while preserving useful audit records.
Consolidate scoring and test fixtures after those boundaries are clear. Only then
consider further modeling work. This yields a smaller everyday product and a more
trustworthy second objective without rebuilding the application.
