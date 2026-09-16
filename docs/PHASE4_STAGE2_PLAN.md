# Phase 4, Stage 2: Standings

**Drafted:** 2026-09-15 (UTC), before any of it exists. Implements stage 2 of
[the Phase 4 plan](PHASE4_PLAN.md); stage 3 is unchanged by it.

**Implemented:** 2026-09-15. Shared scoring, standings, CLI, and documentation landed
in separate commits following the sequence below. See the verification record at the end
for the pre-existing capture drift and the final results.

Stage 1 made the pool's weekly report land somewhere durable, identified and re-readable, and
deliberately computed nothing with it. `pool standings` shows what the pool said. This stage is
the other half: score every entrant on the same terms `my_picks` is scored on, rank them with
their ties intact, and track what each one has spent. It adds no table and no feed. Everything
it needs is already in the database.

Stage 3 is gated on this by substance rather than by scheduling. Its policy has to know where
everyone stands and what each opponent can still pick, and after this stage that is the only
place either fact will exist.

## What Stage 1 Left, And What Is Here Now

Checked against `data/pool.db` before drafting, so the plan is written against the season as it
actually stands rather than as it was imagined:

| | |
|---|---|
| Schema | `pool_entrants`, `pool_picks`, `pool_report_totals` at `user_version = 4`. `player_id` and `game_id` are on every pick, which is what lets this stage reach `scoring.touchdown_totals` without a migration — stage 1 put them there for exactly this. |
| Entrants | Five for 2026: adam, brad, jamie, keith, and cj with `is_me = 1`. |
| Picks | Fifteen rows, all week 1, every one resolved, no blanks, from one successful import. |
| Results | Week 1 is 16/16 games complete with 146 touchdown credits. `complete_weeks` is `[1]`. |

Two accidents of the real week are worth keeping, because each exercises a rule that would
otherwise have only fixtures behind it. **Joe Burrow was picked by both brad and keith**, so one
player's touchdowns credit two entrants — the property stage 3's simulator is required to
preserve, visible in week 1. And **adam and cj are level on one touchdown**, so shared ranks are
exercised from the first week, though a tie *for first* is not.

`my_picks.tds` is NULL for all three of my week-1 rows, because `pool score` has never been run
against the finished week. Standings will show my row as 1 while `pool picks` says
`pending: not scored`. Both are right about different questions, and the answer is to run the
command, not to reconcile the two numbers — but it will look like a bug the first time, so the
standings say so.

## The Premise That Was Wrong

The phase plan made the pool's own table this stage's acceptance test:

> **Done when** `pool leaderboard` reproduces the pool's own standings for a scored week. If it
> disagrees with the official table, the ingestion or the scoring is wrong and that is worth
> knowing before anything is built on top.

That reasoning needs the pool's table to be an independent authority, and it is not. **The
person running the pool works from the same nflverse data this tool does**, confirmed against
the report he sent on 2026-09-15, which matches. Reproducing his table would have compared our
arithmetic against a reflection of itself and called the agreement evidence.

So the comparison is not built, not deferred, and not kept as a warning that never fires. The
phase plan's paragraph is corrected in place, the way that document has taken its other
corrections. `pool_report_totals` stays exactly as stage 1 built it: the reported columns are
passthrough, they show what was supplied, and nothing computes from them or against them.

What replaces it is the check that still has teeth, and it was always the more important one:
**standings and `pool picks` report the same number for my own row, because they go through the
same function.** That is not a comparison of two sources; it is a single definition, asserted.

## Storage

**No migration.** The only candidate is a `tds` cache on `pool_picks`, which stage 1 left out on
purpose and left this stage to decide.

It stays out. Stage 1's argument was that a stored number can disagree with the report and with
`scoring`; the stronger argument is now visible from here. The cache is what *creates* two of
`pick_results`' four pending reasons — `not scored; run pool score` and `results changed; run
pool score` exist only because `my_picks.tds` can go stale. A cache-free entrant path has no
such state, so it can only ever disagree with `pool picks` when *my* cache is stale, which
`pool picks` already labels. Cache-free is what makes the acceptance test above provable rather
than merely usually true.

There is nothing to cache anyway. Five entrants times three slots times eighteen weeks is 270
rows a season, scored from two queries.

## The One Fingerprint Move

The phase plan requires entrants to go through the same scoring function as `my_picks` —
"entrants go through the same function or the leaderboard will quietly disagree with
`pool picks`". Sharing means `pick_results` delegates, which means editing `scoring.py`, which
`benchmark.decision_modules()` confirms is inside the enforced decision closure.

Every alternative is worse. A new module that `scoring` imports changes `scoring.py` *and*
widens the closure. Putting the core in `state.py` changes two closure modules. Not sharing
abandons the requirement.

**This costs more than stage 1's move did, and the difference should be stated rather than
discovered.** Stage 1 argued its migration "costs nothing today: 3C is closed and no protocol is
running." That is no longer the whole picture: five 2026 decisions are captured in this
database, under three identities. After this move `capture.reconstruct` refuses them, and
`pool verify-capture --season 2026` passes them only under `--allow-code-drift`, which prints in
red that they passed only because the check was overridden.

Worth noting that the fingerprint has already moved twice inside this season — three identities
across five captures — so this is a known cost being paid a third time, not a new kind of cost.
It is spent deliberately:

1. The move happens in **one commit, alone**, touching no other closure module, so the drift has
   exactly one named cause.
2. **Before** that commit, `pool verify-capture --season 2026` is run green and its output
   pasted into this document, so the last verification under the old fingerprint is on record.
3. The README says the 2026 captures now need `--allow-code-drift`, and why.

## Code Shape

### The shared core, in `scoring.py`

`pick_results` inlines three things entrants also need. They move out; everything about the
`my_picks.tds` cache stays exactly where it is.

```python
@dataclass(frozen=True)
class PickScore:
    game_id: str | None
    tds: int | None      # None exactly when `pending` is nonempty
    pending: str         # "" when the count is final

@dataclass(frozen=True)
class ScoreBoard:        # coverage and credits, read once
    season: int
    games: pd.DataFrame
    credits: dict[tuple[str, str], int]

def score_board(conn, season) -> ScoreBoard
def resolve_pick_game(conn, season, week, player_id, recorded=None) -> str | None
def score_pick(board, week, player_id, game_id) -> PickScore
def resolved_through(conn, season) -> int | None
```

`ScoreBoard` is read once per call and shared by every pick in it. Two callers reading coverage
separately could straddle a refresh and rank a week on two different sets of evidence.

`resolve_pick_game` is the existing re-resolution — this week's stat line first, then the team's
scheduled game, then what was recorded. Entrant picks need it for the same reason mine do: a
recorded team can be stale, and where a player actually appeared cannot.

`resolved_through` is the largest week W such that **every** week from 1 to W is complete. A
prefix, not a maximum. Weeks 1 and 3 final with 2 half-played must not produce a total "as of
week 3" that silently omits a week, because ranking on that invents a lead or a deficit. It
lives in `scoring.py` rather than with the standings because stage 3's policy is inside the
closure and will need it, and folding it into this one move costs nothing extra.

**What stays in `pick_results`:** `recompute`, `preserve_existing`, both cache-staleness
reasons, and the `UPDATE my_picks` block. Entrants have no cache and never will.

**The refactor is proven by tests it does not touch.** `tests/test_workflow.py` already pins
every pending reason, the stat-game override, schedule-correction invalidation, week scoping and
the atomic rollback. The commit that moves this code edits none of them. Green and untouched is
the proof; if any of them needs adjusting, the refactor changed behaviour and is wrong.

Four equivalences are easy to lose and each is a real defect if lost: `score_pick` must return a
Python `int` rather than a `numpy.int64`, because the value is bound into an `UPDATE` and
compared with `!=`; the fallback stays `gid or row.game_id`, not `if gid is None`, because the
falsy-empty-string case is existing behaviour; the week guard stays written so that
`None not in index` is what routes an unresolved game to `game unresolved`; and the write stays
`with conn:` rather than `db.transaction`, because a test pins that the whole `executemany`
rolls back as one.

### The standings, in `src/pool/standings.py`

A new module, outside the closure, importing `config`, `db`, `entrants`, `scoring` and `state`.
Nothing in the closure imports it and nothing ever should; `tests/test_phase3a.py`'s `outside`
set gains `standings`, the same one-line review decision stage 1 made for `entrants`.

`cli.py` already defines a `standings` command, which would shadow the module name. The house
already has the answer at `cli.py:15` — `from . import backtest as bt` — so the import is
aliased the same way.

```python
def entrant_scores(conn, season, *, weeks=None) -> list[EntrantPick]
def used_pools(conn, season, *, through=None) -> dict[str, UsedPool]
def remaining_counts(conn, season, week, entrant_id) -> dict[str, int]
def board(conn, season, *, week=None) -> Board
```

`Board` carries the season, the as-of week, the ranked rows, the in-progress picks, the leaders,
and whether my own cache is stale — so the CLI renders a result rather than re-deriving one,
which is the shape `ImportResult` established in stage 1.

`entrants.py` is untouched. Its readbacks already return everything this needs.

## The Three Pick States, Scored

Stage 1 built four distinguishable states and nothing has yet had to interpret them. Conflating
any two of them is the defect this stage exists to avoid, so they are written down as a table
before they are written as code:

| The report says | Scores |
|---|---|
| A pick, and its game is final | the credit count, including a real 0 |
| A pick, and its game is not final | **pending** — no number at all |
| No pick submitted (`player_name` NULL) | **0, final** — a fact about the week, not an absence |
| A name that never resolved (`player_id` NULL) | **unresolved** — neither a zero nor a pending game |
| Nothing; the week was never imported | **missing** — the total is incomplete |

This is *pending is not zero* applied to the states stage 1 invented. An unresolved **name** is
not a pending **game**: the game may well be finished, and calling the pick pending would
suggest waiting for a result that has already arrived. What is missing is an identity, and the
fix is a re-import, not patience.

A weekly or season subtotal is labelled incomplete while any pick under it is pending,
unresolved, or missing, which is the discipline `_render_scores` already applies to mine.

## Everyone Is Ranked As Of The Last Resolved Week

`resolved_through` sets the as-of week, and it is deliberately not the last week imported. A
report can arrive while its week is still being played — that is the corrected cadence the phase
plan records — and `my_picks` can hold the current week before any report does. Ranking a
half-played week would invent a standing.

Restricting to fully final weeks removes *feed* incompleteness, not *pick-level* incompleteness.
Inside a final week a pick can still be pending on an unresolvable game, unresolved on a name,
or missing because that entrant's week was never imported. So a row is final only when it has
none of the three, and the table says it is provisional otherwise — even at week 1.

Everything beyond the as-of week is shown separately as in progress, from two sources unioned
without inventing a merge rule: reported picks for later weeks, and my own recorded picks for
weeks where no report has arrived yet, labelled as recorded rather than reported. Where both
exist and name different players, the report's row is shown and the note points at
`pool report import`, whose `--me` comparison already owns that finding and should not be
re-litigated here.

## Ties Are Real, And At The Top They Split The Pot

Ranking is competition ranking: sort by touchdowns descending and assign a number only when the
total changes, so 5/4/3/1/1 gives ranks 1, 2, 3, 4, 4. Tied entrants print the same number.
Order within a tie is by display name purely so the table is stable, and the shared rank is what
makes clear that name order is not standing.

Only a tie for first has consequences. Every entrant level with the maximum is a leader, and
each carries `share = 1/k`; everyone else carries `0.0`. `share` is a real field rather than
something the renderer computes, because stage 3 maximises expected share of the pot and must
read the same number this table prints.

Four sentences, chosen by how many leaders there are and whether the board is final:

| Leaders | Final | Printed |
|---|---|---|
| 1 | yes | `Jamie leads with 5 TDs.` |
| 1 | no | `As it stands Jamie leads with 5 TDs, with 2 picks pending.` |
| more than 1 | season over | `Jamie and Keith are tied for first with 5 TDs. A tie for first splits the winnings: each takes 1/2 of the pot.` |
| more than 1 | otherwise | `As it stands Jamie and Keith are tied for first with 5 TDs, with 2 picks pending. A tie at the end splits the winnings 1/2 each.` |

**Corrected 2026-09-15**, after review. The settled sentence was first gated on `Board.final`,
which only says the ranked prefix has no gaps — true after one cleanly scored week of
eighteen, so a week 1 tie announced a split pot. It is gated on `Board.season_complete`
instead: every regular-season week ranked. `final` still drives the `(provisional)` label,
where it is the right question.

The word *leader* is never printed when there is more than one, and the split is never stated as
settled while the board is provisional. Season totals are small integers, so ties are common
enough that getting this wrong would be wrong most weeks.

## Each Entrant's Own Used Pool

One player per entrant per season. `used_pools` accumulates it from every **imported** week, and
that is deliberately a different cut-off from the ranking, which stops at the last **resolved**
week. Ranking a half-played week invents a lead; a player an opponent has already submitted is
spent whether or not his game has finished. Two questions, two as-of weeks, each labelled.

The three pick states resolve differently here too:

- a resolved pick spends its player;
- a no-pick spends nobody, which is a real fact and not a gap;
- an unresolved name spends **somebody we cannot name**, so it increments a separate `unknown`
  count and enters the used set not at all.

That third rule is the one worth arguing for. Adding an unnameable player to the used set would
shrink an opponent's remaining pool by a ghost; leaving it out silently would claim a
completeness we do not have. `unknown` is the honest third answer, and stage 3 has to branch on
it rather than trust a set that is quietly short.

**Repeats are observed, not enforced.** `state.record_picks` enforces one-player-per-season for
my picks because I am submitting them. Entrant picks we only observe, so a player appearing
twice for one entrant is counted once and reported as a repeat with the weeks it happened in.
Silently deduplicating it would discard the cheapest detector of a name-resolution error this
stage has.

`remaining_counts` returns per-slot counts over `state.historical_pool`, filtered by
`config.SLOTS`. Counts only: listing or ranking what an opponent has left needs projections, and
that is stage 3's job.

## CLI

No new command. The phase plan named `pool leaderboard` because it was drafted before
`pool standings` existed, and the two views differ on three axes at once — week against season,
reported against computed, last-imported against last-final. The first two are columns. Only the
third is a rule, and a rule is a labelled column, not a second command.

So `pool standings` gains the neighbour stage 1 promised it:

```
pool standings [--week N] [--season Y]
```

```
Rank | Entrant | QB | RB | FLEX | Week TDs | Season TDs | Used | Reported week | Reported total
```

The default week stays the latest **imported** week, and the title states both grains —
`week 1 picks; ranked on season totals through week 1` — so the two never have to be inferred
from each other. Pick cells keep stage 1's three-state rendering. Week TDs shows its pending
count when the week is unfinished; Season TDs is marked incomplete when any week up to the
as-of week is. Reported columns are unchanged passthrough.

The command exits 0. There is no failure mode left for it to signal: findings about whether a
report can be trusted belong to `pool report import`, which already reports and fails on them.

Below the table, in order: the as-of statement, the tie or leader sentence, any incompleteness,
the in-progress block, `Totals are touchdown counts, not points.`, and — while `my_picks.tds` is
empty for a finished week — the line telling me to run `pool score` so both commands print the
same number.

Degenerate cases get a sentence rather than an empty table: no week with all its games final
cannot be ranked at all, and nothing imported points at `pool report import`.

## Tests

New `tests/test_standings.py`, built on the existing shapes: the workflow suite's schedule and
its `play`/`end` row builders, `scoring.import_touchdowns`, then a multi-entrant CSV through
`entrants.import_report`. Week 1 final, week 2 permanently in progress — which is the shape the
as-of rule and the pending rule both need.

The claims this document makes that would be silent if wrong:

1. An entrant pick and one of mine on the same player score the same number and the same reason.
2. A complete game with no touchdowns scores zero; an incomplete one has no number at all.
3. A pick's game resolves to where the player actually played, not where he was recorded.
4. A slot the report says went unpicked scores zero rather than staying pending.
5. A name we could not resolve is neither a zero nor a pending game.
6. A week never imported leaves a total incomplete rather than counting zero.
7. Subtotals are labelled incomplete while any pick under them is unfinal.
8. Each entrant has their own used pool, and a shared player is spent by both.
9. An unresolved name leaves a used pool incomplete rather than quietly smaller.
10. A player an entrant used twice is reported as a repeat, not silently counted once.
11. The used pool counts every imported week while the ranking stops at the resolved one.
12. Standings stop at the last week whose games are all final, not the last imported.
13. A week imported before its games finish is shown in progress rather than ranked.
14. A gap in completed weeks holds the as-of week at the last unbroken one.
15. A week I recorded before any report arrived appears in progress and in no rank.
16. A pick whose game never resolves keeps a final week from being called final.
17. Entrants level on touchdowns share a rank rather than being ordered by name.
18. A tie for first says the pot splits and never names one of them the leader.
19. A tie below first is shown as a tie and splits nothing.
20. A provisional tie for first is not stated as a settled split.
21. A return touchdown counts for an entrant and a two-point conversion never does.
22. One player's touchdowns are credited to every entrant who picked him.
23. The standings call their numbers touchdowns and never points.
24. The computed columns sit beside the reported ones without replacing them.
25. **Standings and `pool picks` report the same number for my own row.**
26. The `pool score` nudge fires when my own cache is empty.

Number 25 is the acceptance test. It is the assertion that fails if anyone ever reimplements the
pending logic instead of calling the shared one, and it is the reason the shared core is worth a
fingerprint move.

`tests/test_phase3a.py`'s `outside` set gains `standings`. One existing assertion flips on
purpose: `test_cli_import_list_and_reported_standings` asserts that `standings` output contains
no computed values, which was correct for stage 1 and is the thing stage 2 changes. It must now
assert that the computed columns are present *and* that the reported ones survived.

## Order Of Work

Five commits, in this order, each green on its own.

0. **This document.**
1. **The shared core.** `scoring.py` only, plus the three tests that assert entrants and I score
   identically. The one fingerprint move, made deliberately and once, with the pre-move
   `verify-capture` output recorded here first.
2. **The standings.** `standings.py` and `tests/test_standings.py`. No CLI.
3. **The CLI.** The computed columns, the ranks, the tie sentences, the in-progress block, the
   `test_phase3a.py` closure-set edit and the flipped stage 1 assertion.
4. **Docs.** `README.md` status and a standings section; `docs/DESIGN.md` §3.1 and §3.5, which
   still call this later work; `docs/PHASE4_PLAN.md`, whose stage 2 acceptance paragraph is
   corrected in place; `docs/IMPLEMENTATION_PLAN.md` stage status.

## What The Real Data Cannot Validate Yet

Fixtures cover all of these. They are listed because a fixture-covered rule that has never met a
real week is a different kind of confidence, and this is the season to find out which ones bite.

- **A tie for first.** Week 1 has jamie alone at five. The only real tie is below first.
- **The in-progress split.** Week 1 is the only imported week and it is final, so the section is
  empty. Week 2 is the first real exercise.
- **No-picks and unresolved names.** All fifteen week-1 rows resolved and none was blank, so
  neither state has occurred outside fixtures.
- **Used-pool accumulation.** With one week, every entrant has spent exactly three players and a
  repeat is arithmetically impossible.
- **The prefix rule in `resolved_through`.** Weeks 2 through 18 are uniformly missing their
  scoring feed, so no gap exists to hold the as-of week back until mid-season.

## Implementation Verification — 2026-09-15

Before editing `scoring.py`, the required strict command exited 1: three of five
captures already had decision-path drift. A green strict baseline is therefore
unavailable in the starting tree; no capture or fingerprint was rewritten to manufacture
one. The override run exited 0, with all five reconstructing and matching replay.
This records the actual baseline in place of the planned all-green prerequisite.

```text
$ pool verify-capture --season 2026
┏━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Decision     ┃ Week ┃ Event             ┃ Reconstructs ┃ Parity ┃ Detail                                                                                                                                                 ┃
┡━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ ea7b0ff08ff6 │ 1    │ thursday_deadline │ no           │ no     │ Decision ea7b0ff08ff64b99ad80edec801896c2 was captured under source fingerprint fc80de7aacbf3ad522842159a9f09686c8b5b6278ad1878f6fe0a86f4d228054,      │
│              │      │                   │              │        │ running 204f5c2dc00340ef40978444824c5316770d08978588147a3fd4af79b7917e32 (decision path). The recommender's behaviour is not carried by the recorded   │
│              │      │                   │              │        │ constants alone; pass allow_code_drift=True to reconstruct anyway.; source tree differs from the one the decision was captured under                   │
│ 6e71a2be2ae2 │ 1    │ sunday_slate      │ no           │ no     │ Decision 6e71a2be2ae24e9aa55f35be9e65dba3 was captured under source fingerprint fc80de7aacbf3ad522842159a9f09686c8b5b6278ad1878f6fe0a86f4d228054,      │
│              │      │                   │              │        │ running 204f5c2dc00340ef40978444824c5316770d08978588147a3fd4af79b7917e32 (decision path). The recommender's behaviour is not carried by the recorded   │
│              │      │                   │              │        │ constants alone; pass allow_code_drift=True to reconstruct anyway.; source tree differs from the one the decision was captured under                   │
│ 55134df98e15 │ 1    │ sunday_slate      │ no           │ no     │ Decision 55134df98e1546d693d75373ac6969cd was captured under source fingerprint fc80de7aacbf3ad522842159a9f09686c8b5b6278ad1878f6fe0a86f4d228054,      │
│              │      │                   │              │        │ running 204f5c2dc00340ef40978444824c5316770d08978588147a3fd4af79b7917e32 (decision path). The recommender's behaviour is not carried by the recorded   │
│              │      │                   │              │        │ constants alone; pass allow_code_drift=True to reconstruct anyway.; source tree differs from the one the decision was captured under                   │
│ 155e11a74ea3 │ 1    │ sunday_slate      │ yes          │ yes    │ source outside the decision path moved (accepted)                                                                                                      │
│ 071a5bc55c96 │ 2    │ thursday_deadline │ yes          │ yes    │ -                                                                                                                                                      │
└──────────────┴──────┴───────────────────┴──────────────┴────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
1 verified against a source tree that moved outside the decision path, with the enforced fingerprint unchanged. That fingerprint covers what a decision is a function of; the whole-tree hash is recorded beside it.
3 of 5 captured decisions did not verify.

$ pool verify-capture --season 2026 --allow-code-drift
┏━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Decision     ┃ Week ┃ Event             ┃ Reconstructs ┃ Parity ┃ Detail                                                ┃
┡━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ ea7b0ff08ff6 │ 1    │ thursday_deadline │ yes          │ yes    │ decision path fingerprint moved; accepted by override │
│ 6e71a2be2ae2 │ 1    │ sunday_slate      │ yes          │ yes    │ decision path fingerprint moved; accepted by override │
│ 55134df98e15 │ 1    │ sunday_slate      │ yes          │ yes    │ decision path fingerprint moved; accepted by override │
│ 155e11a74ea3 │ 1    │ sunday_slate      │ yes          │ yes    │ source outside the decision path moved (accepted)     │
│ 071a5bc55c96 │ 2    │ thursday_deadline │ yes          │ yes    │ -                                                     │
└──────────────┴──────┴───────────────────┴──────────────┴────────┴───────────────────────────────────────────────────────┘
3 passed only because the fingerprint check was overridden. The source these checks enforce had moved and they were accepted anyway.
1 verified against a source tree that moved outside the decision path, with the enforced fingerprint unchanged. That fingerprint covers what a decision is a function of; the whole-tree hash is recorded beside it.
All 5 captured decisions reconstruct and match replay.
```

### Completed implementation

- The shared core moved only `scoring.py` within the decision closure, in commit
  `f9a6c8f`. `tests/test_workflow.py` was unchanged and passed. The enforced hash after
  that refactor is `f9682cdd5dbe57bafcd197865dd5fcdbcaeeb9f5bd155bf4769c5dd4bf59987f`.
- Standings (`ff48e77`) and the CLI (`4df2681`) remain outside the closure. No migration,
  entrant-score cache, change to `entrants.py`, or comparison with reported totals was added.
- `pytest -q`: **530 passed**. `ruff check .`: **All checks passed**. The 17 standings
  tests include parameterized shared-core equivalences, the scored states, independent
  used pools, missing weeks, the completed-week prefix, ties, in-progress sources, and CLI
  parity. The final wording adjustment also passed all 17 standings cases.
- Real week 1: Jamie 5 (rank 1), Keith 4 (rank 2), Brad 3 (rank 3), Adam 1 and CJ 1
  (shared rank 4). All five used pools contain three players. The current personal
  week-1 cache is now `[0, 1, 0]` with no pending reasons, and CJ's computed total is 1.
  This differs from the drafting-time cache observation above; implementation verification
  was read-only and did not run `pool score` against the real database.
- Strict post-refactor verification exits 1 with `5 of 5 captured decisions did not verify`.
  The explicit override exits 0 with the following output; it is not strict verification.

```text
$ pool verify-capture --season 2026 --allow-code-drift
┏━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Decision     ┃ Week ┃ Event             ┃ Reconstructs ┃ Parity ┃ Detail                                                ┃
┡━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ ea7b0ff08ff6 │ 1    │ thursday_deadline │ yes          │ yes    │ decision path fingerprint moved; accepted by override │
│ 6e71a2be2ae2 │ 1    │ sunday_slate      │ yes          │ yes    │ decision path fingerprint moved; accepted by override │
│ 55134df98e15 │ 1    │ sunday_slate      │ yes          │ yes    │ decision path fingerprint moved; accepted by override │
│ 155e11a74ea3 │ 1    │ sunday_slate      │ yes          │ yes    │ decision path fingerprint moved; accepted by override │
│ 071a5bc55c96 │ 2    │ thursday_deadline │ yes          │ yes    │ decision path fingerprint moved; accepted by override │
└──────────────┴──────┴───────────────────┴──────────────┴────────┴───────────────────────────────────────────────────────┘
5 passed only because the fingerprint check was overridden. The source these checks enforce had moved and they were accepted anyway.
All 5 captured decisions reconstruct and match replay.
```
