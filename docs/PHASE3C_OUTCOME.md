# Phase 3C Outcome: Closed Without Collection

**Decision date:** 2026-09-07 (UTC).

**Decision maker:** project owner, before the 2026 week 1 pick deadline.

**Disposition:** the Phase 3C window is closed without being run. No decision was ever
captured under it, no floor was measured, and no reading of any kind is claimed. The
shipped model and every production constant remain unchanged, as they were under the
[Phase 3B no-promotion outcome](PHASE3B_OUTCOME.md).

This is a decision not to collect, recorded so a later reader is not left inferring one
from an empty results directory. The [protocol](PHASE3C_PROTOCOL.md) and its
[machine-readable form](../experiments/phase3c-baseline.toml) are kept intact and still
resolve; the read side — `pool captures`, `pool verify-capture`, `pool baseline` — is kept
and still tested. Nothing here is a finding about the model, the capture, or the replay.

## What Was Built, And Stands

The window was declared in full before it was abandoned, and everything it needed exists:

- The dated protocol, twice reviewed, once signed and once amended before any capture.
- Append-only prospective capture: surfaces, advice, submitted actions, input identities.
- Reconstruction, live/snapshot parity, event coverage, the declared populations, the
  submission audit and the descriptive export, all under test.

None of it is removed. `pool recommend` still captures by default, so a decision log
accumulates whether or not a protocol governs it. The window can be re-declared later
against a new date; what cannot be recovered is the specific weeks that go unmeasured.

## Why It Was Closed

The window required a source-tree freeze for its duration. Reconstruction and parity are
only interpretable if the code that re-derives a decision is the code that made it, and
until this decision the fingerprint they compared covered the entire `src/` tree — so
adding an unrelated feature would have failed six weeks of evidence retroactively.

The 2026 season is also when this tool is meant to be used. Opponent-pick ingestion,
standings and a win-probability objective are the season's actual work, and none of them
can proceed under a whole-tree freeze without branch and worktree discipline on every
command. Weighed against that, six weeks bought descriptive diagnostics on forward data
and little else that could not be bought later.

## What Is Given Up, Precisely

**Permanently:** decisions captured in 2026 weeks 1-6 will not have been certified as
reconstructible from their stored surfaces. If a later analysis wants those specific weeks
as validated evidence, that is not recoverable.

**Deferred, not lost:** the assumption under every replay in this project — that the live
path and archived replay are the same function of the same inputs — remains checked only
inside one process on a staged database, never against a decision that was really
captured. Parity is a per-decision check, so a single captured decision verified the same
day settles it whenever the question is next worth asking. It is an afternoon, not a
window.

**Not given up:** any gate on the win-probability work. Phase 3C validated capture
fidelity, never policy. Evaluating a change to the decision objective needs opponent data
and a multi-event replay that does not exist, and that requirement is untouched by this.

## What Changed With It

The enforced source fingerprint is narrowed from the whole tree to the modules a decision
is actually a function of — the import closure of `recommend`, `projections` and
`snapshots`, plus `uv.lock`. The whole-tree hash is still recorded beside it, so drift
stays visible; only the decision-scoped hash is enforced. This is a defect fix rather than
a concession: a fingerprint that fails a capture because a leaderboard was added was never
measuring the thing it claimed to measure.

It also does the work this closure was meant to avoid needing. Captures accumulated during
ordinary feature development stay reconstructible, so the option to verify them later
survives without any freeze, branch discipline, or protocol.

## Status Of The Roadmap

Phase 3 ends here: 3A delivered the evidence and capture layer, 3B completed with no
promotion, 3C is closed without collection. The next work is the season's own — opponent
ingestion, standings, then a win-probability objective — and it carries its own evaluation
requirements, which are not satisfied by anything in this phase.
