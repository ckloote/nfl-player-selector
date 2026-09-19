"""`pool research`: replay, benchmarks, diagnostics, decision checks and the prediction log.

Every research module is imported inside the command that uses it, so `pool week` and
the other weekly commands never load them; `tests/test_research_boundary.py` holds that.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from .. import capture, config, db, entrants, pit, predictions, projections, scoring, snapshots

app = typer.Typer(
    help="Replay seasons, benchmark models, check captured decisions and score predictions.",
    no_args_is_help=True,
)
predict_app = typer.Typer(
    help="Predict rival picks before a week can be seen, and score the record.",
    no_args_is_help=True,
)
app.add_typer(predict_app, name="predict")
console = Console()

SeasonOpt = typer.Option(config.DEFAULT_SEASON, "--season", "-s", help="Season year")
DbOpt = typer.Option(None, "--db", help="SQLite path (default data/pool.db)")
WeekOpt = typer.Option(None, "--week", "-w", help="Week (default: current)")
SeasonsOpt = typer.Option(
    "2025", "--season", "-s", help="Season, list, or range: 2025 | 2024,2025 | 2017-2025"
)
RoleSourceOpt = typer.Option(None, "--role-source", help="usage, depth, or none")
HorizonOpt = typer.Option(
    None, "--vegas-horizon", help="Closing-line horizon (legacy-closing policy only)"
)
PolicyOpt = typer.Option(
    "historical", "--input-policy", help="historical, snapshots, legacy-closing"
)
DecisionOpt = typer.Option(
    None, "--decision-times", help="CSV: season,week,decision_at (with timezone)"
)
CsvOpt = typer.Option(None, "--csv", help="Write every cell to a CSV")
ProjectionOpt = typer.Option(
    "shipped", "--projection", help="Projection model; see `pool research models` for the list"
)


def _conn(path: Path | None):
    return db.connect(path)


def _used_cell(block: dict) -> str:
    unknown = block["unknown"]
    return str(len(block["used"])) + (f" + {unknown} unknown" if unknown else "")


@predict_app.command("record")
def predict_record(
    week: int | None = WeekOpt,
    season: int = SeasonOpt,
    db_path: Path | None = DbOpt,
    write: bool = typer.Option(True, "--record/--dry-run", help="Archive it, or only show it"),
):
    """Predict every rival's picks for a week and archive it before the week can be seen.

    Exits non-zero when the prediction will not be scorable -- the archive still happens,
    because a record that cannot be scored is still evidence, but a scripted run has to be
    told that this week's observation was lost.
    """
    from ..cli import _projections, _week

    conn = _conn(db_path)
    try:
        wk = _week(conn, season, week)
        if not predictions.identified(conn, season):
            console.print(entrants.IDENTITY_PROMPT, style="red", markup=False)
            raise typer.Exit(1)
        now = datetime.now(UTC)
        kickoff = predictions.first_kickoff(conn, season, wk)
        arrivals, _ = predictions.report_arrivals(conn, season)
        arrival = arrivals.get(wk)
        late = []
        if kickoff is None:
            late.append(f"week {wk} has no confirmed kickoff")
        elif now >= kickoff:
            late.append(f"week {wk} kicked off at {kickoff.isoformat(timespec='minutes')}")
        if arrival is not None:
            late.append(f"week {wk}'s report already arrived at {arrival}")
        proj = _projections(conn, season, wk)
        payload = predictions.predict(conn, season, wk, proj)
        if not payload["rivals"]:
            console.print(f"No rivals to predict for {season}. Run pool report import first.")
            raise typer.Exit(1)
        observation = predictions.archive(conn, season, wk, payload) if write else None
        # The same act: what the model says before the week can be seen. The picks are one
        # claim about these five people, the distribution behind their totals is another,
        # and both expire at the same kickoff.
        committed = pit.commit(conn, season, wk, proj) if write else None
    finally:
        conn.close()
    table = Table(
        "Rival",
        "Slot",
        *predictions.rivals.PREDICTORS,
        "Remaining",
        "Used",
        title=f"Predicted rival picks \u2014 {season}, week {wk}",
    )
    for block in payload["rivals"].values():
        for slot, byname in block["slots"].items():
            table.add_row(
                block["display_name"],
                slot,
                *(
                    byname[name][0]["player_name"] if byname.get(name) else "\u2014"
                    for name in predictions.rivals.PREDICTORS
                ),
                str(block["remaining"].get(slot, "\u2014")),
                _used_cell(block),
            )
    console.print(table)
    console.print(
        f"Top {payload['top_n']} kept per predictor; the full rankings are in the archive, "
        "not in this table."
    )
    if observation is not None:
        console.print(f"Archived prediction as observation {observation}.")
        console.print(
            f"Committed week {wk}'s implied distribution as observation {committed} "
            f"({config.WINPROB_SIMS} draws, seed {config.WINPROB_SEED})."
        )
    else:
        console.print("[yellow]Dry run: nothing archived.[/yellow]")
    if late:
        console.print("[red]This prediction will not be scorable: " + "; ".join(late) + ".[/red]")
        console.print(
            "A prediction counts only if it was archived before the week's first kickoff "
            "and before its report. Both deadlines are checked against the archive."
        )
        raise typer.Exit(1)


def _rate_table(frame, title: str, first: str) -> Table:
    table = Table(first, "Predictor", "Scored", "Hits", "Hit rate", "In top N", title=title)
    for row in frame.itertuples():
        table.add_row(
            str(getattr(row, first.lower().replace(" ", "_"), "")),
            row.predictor,
            str(int(row.n)),
            str(int(row.hits)),
            f"{row.hit_rate:.0%}",
            f"{row.in_top_n:.0%}",
        )
    return table


@predict_app.command("score")
def predict_score(season: int = SeasonOpt, db_path: Path | None = DbOpt):
    """Score archived predictions against the picks entrants actually made."""
    conn = _conn(db_path)
    try:
        scored, notes = predictions.score(conn, season)
    finally:
        conn.close()
    if scored.empty:
        console.print(f"No scorable predictions for {season}.")
    else:
        overall = predictions.hit_rates(scored)
        table = Table(
            "Predictor",
            "Scored",
            "Hits",
            "Hit rate",
            "In top N",
            title=f"Rival-pick prediction accuracy \u2014 {season}",
        )
        for row in overall.itertuples():
            table.add_row(
                row.predictor,
                str(int(row.n)),
                str(int(row.hits)),
                f"{row.hit_rate:.0%}",
                f"{row.in_top_n:.0%}",
            )
        console.print(table)
        console.print(_rate_table(predictions.hit_rates(scored, "slot"), "By slot", "Slot"))
        console.print(
            _rate_table(predictions.hit_rates(scored, "display_name"), "By rival", "Display name")
        )
        ranks = Table(
            "Predictor", "Rank of the actual pick", "Count", title="Where the actual pick landed"
        )
        for (name, rank), n in (
            scored.groupby(["predictor", scored["rank"].astype("Int64")], dropna=False)
            .size()
            .items()
        ):
            ranks.add_row(name, "outside top N" if pd.isna(rank) else str(int(rank)), str(int(n)))
        console.print(ranks)
        console.print(
            "Hit rate is the share of rival slot-weeks whose actual pick the predictor "
            "ranked first. Totals are picks, not touchdowns."
        )
    for note in notes:
        where = "" if note["week"] is None else f"Week {note['week']}: "
        # A style rather than markup: the reason can quote a name, and markup is off so a
        # bracket in it prints as written.
        console.print(f"{where}{note['reason']}", style="yellow", markup=False)
    _print_pit(conn_path=db_path, season=season)


def _print_pit(conn_path, season: int) -> None:
    """The other claim on the same feed: where realised weekly totals fell in the model.

    Reported without a verdict. Eighty-five draws over a full season is not enough to test
    uniformity with any power, so a pass/fail printed off a part-season would be a coin
    flip wearing a conclusion. The lean is the readable part: mass at the top means the
    model is under-predicting the players these five actually pick.
    """
    conn = _conn(conn_path)
    try:
        scored, notes = pit.score(conn, season)
    finally:
        conn.close()
    if scored.empty:
        console.print(f"[dim]No scorable weekly distributions for {season} yet.[/dim]")
    else:
        table = Table(
            "Bin",
            "Count",
            "Expected if flat",
            title=f"Where realised weekly totals fell — {season} ({len(scored)} draws)",
        )
        for row in pit.uniformity(scored).itertuples():
            table.add_row(f"{row.low:.1f}-{row.high:.1f}", str(row.count), f"{row.expected:.1f}")
        console.print(table)
        console.print(
            "Committed before each kickoff as a seed and a frame hash, so the distribution "
            "could not have been chosen to fit. No pass or fail: a season is too few draws "
            "to test uniformity, and the lean is what is worth reading."
        )
    for note in notes:
        where = "" if note["week"] is None else f"Week {note['week']}: "
        console.print(f"{where}{note['reason']}", style="dim", markup=False)


# --- backtesting ------------------------------------------------------------
def _seasons(spec: str) -> list[int]:
    """Parse "2025", "2024,2025", or "2017-2025"."""
    out: list[int] = []
    try:
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-", 1))
                out.extend(range(lo, hi + 1))
            else:
                out.append(int(part))
    except ValueError:
        console.print(f"[red]Cannot read {spec!r} as a season, list, or range.[/red]")
        raise typer.Exit(1) from None
    return out


def _floats(spec: str, name: str) -> list[float]:
    try:
        return [float(x) for x in spec.split(",") if x.strip()]
    except ValueError:
        console.print(f"[red]Cannot read {spec!r} as a list of {name} values.[/red]")
        raise typer.Exit(1) from None


def _builder(name: str):
    """Resolve a projection-model name, or exit naming the valid ones."""
    from . import models

    try:
        return models.get(name)
    except KeyError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from None


def _backtest_ready(conn, season: int) -> list[int]:
    """Weeks available to replay, or a message naming the refresh that fixes it."""
    weeks = projections.available_weeks(conn, season)
    prior = conn.execute(
        "SELECT COUNT(*) FROM player_weeks WHERE season = ?", (season - 1,)
    ).fetchone()[0]
    if not projections.available_weeks(conn, season):
        msg = f"No {season} schedule loaded; run `pool refresh --season {season}` first."
    elif not prior:
        msg = (
            f"No {season - 1} stats loaded — the prior season is the model's starting "
            f"prior. Run `pool refresh --season {season}`, which imports both."
        )
    elif not weeks:
        msg = (
            f"No complete {season} touchdown coverage; run `pool refresh --season {season}` "
            "before replay comparisons."
        )
    else:
        try:
            scoring.require_complete(
                conn, season - 1, projections.available_weeks(conn, season - 1)
            )
            scoring.require_complete(conn, season, weeks)
            return weeks
        except ValueError as exc:
            msg = str(exc)
    console.print(f"[red]{msg}[/red]")
    raise typer.Exit(1)


@app.command()
def backtest(
    season: str = SeasonsOpt,
    strategy: str = typer.Option(
        "optimizer,greedy,random,hindsight", "--strategy", help="Comma-separated strategies"
    ),
    trials: int = typer.Option(config.RANDOM_TRIALS, help="Trials for the random baseline"),
    seed: int = typer.Option(0, help="Seed for the random baseline and the invented rivals"),
    rivals: int = typer.Option(4, help="Invented rivals the winprob strategy plays against"),
    rival_behaviour: str = typer.Option(
        "greedy", help="How the invented rivals pick: greedy, naive or optimizer"
    ),
    discount: float | None = typer.Option(None, help="Override FUTURE_DISCOUNT"),
    prior_weight: float | None = typer.Option(None, help="Override PRIOR_WEIGHT_GAMES"),
    role_source: str | None = RoleSourceOpt,
    vegas_horizon: int | None = HorizonOpt,
    input_policy: str = PolicyOpt,
    decision_times: Path | None = DecisionOpt,
    projection: str = ProjectionOpt,
    detail: bool = typer.Option(False, "--detail", help="Show every week's picks"),
    db_path: Path | None = DbOpt,
):
    """Replay finished seasons with data frozen at each pick deadline."""
    from . import backtest as bt

    builder = _builder(projection)
    role_source, times = _input_options(input_policy, role_source, vegas_horizon, decision_times)
    conn = _conn(db_path)
    console.print(
        f"Projection: {projection}; random strategy trials: {trials}; "
        f"trial seeds: {seed}–{seed + trials - 1}; "
        f"discount: {config.FUTURE_DISCOUNT if discount is None else discount}; "
        f"prior weight: {config.PRIOR_WEIGHT_GAMES if prior_weight is None else prior_weight}"
    )
    names = [s.strip() for s in strategy.split(",") if s.strip()]
    unknown = [n for n in names if n not in bt.STRATEGIES and n != "hindsight"]
    if unknown:
        known = [*bt.STRATEGIES, "hindsight"]
        console.print(f"[red]Unknown strategy {unknown[0]!r}; choose from {known}.[/red]")
        raise typer.Exit(1)
    # Built only when something asks for it, so an ordinary replay stays an ordinary
    # replay: the invented rivals cost a greedy rollout a week and, more to the point,
    # put a caveat under a table that does not need one.
    field = None
    if set(names) & bt.NEEDS_RIVALS:
        try:
            field = bt.Field(count=rivals, behaviour=rival_behaviour, seed=seed)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from None

    deltas: list[float] = []
    overrides = {} if prior_weight is None else {"PRIOR_WEIGHT_GAMES": prior_weight}
    for yr in _seasons(season):
        weeks = _backtest_ready(conn, yr)
        with config.override(**overrides):
            rows = bt.run_season(
                conn,
                yr,
                names,
                weeks=weeks,
                trials=trials,
                seed=seed,
                discount=discount,
                role_source=role_source,
                vegas_horizon=vegas_horizon,
                input_policy=input_policy,
                decision_times=times,
                builder=builder,
                against=field,
            )
        _render_input_provenance(rows[0].input_provenance)
        by_name = {r.strategy: r for r in rows}
        ceiling = by_name["hindsight"].total if "hindsight" in by_name else 0.0
        base = by_name["greedy"].total if "greedy" in by_name else None
        if base is not None and "optimizer" in by_name:
            deltas.append(by_name["optimizer"].total - base)
        _render_backtest(yr, weeks, rows, base, ceiling)
        if detail:
            _render_picks(by_name.get("optimizer") or rows[0])

    if len(deltas) > 1:
        mean = sum(deltas) / len(deltas)
        sd = (sum((d - mean) ** 2 for d in deltas) / (len(deltas) - 1)) ** 0.5
        wins = sum(1 for d in deltas if d > 0)
        console.print(
            f"\n[bold]optimizer - greedy:[/bold] mean {mean:+.2f} TD/season over "
            f"{len(deltas)} seasons (SD {sd:.2f}, won {wins}/{len(deltas)}). "
            f"Standard error {sd / len(deltas) ** 0.5:.2f} — treat anything smaller as noise."
        )


def _render_backtest(season: int, weeks: list[int], rows, base: float | None, ceiling: float):
    against = next((r.against for r in rows if r.against), None)
    columns = [
        "Strategy",
        "TDs",
        "vs greedy",
        *config.SLOTS,
        "Projected",
        "Proj/act",
        "Zero picks",
        "% ceiling",
    ]
    t = Table(
        *columns,
        *(["Finish"] if against else []),
        title=f"Backtest {season} (weeks {weeks[0]}-{weeks[-1]})",
    )
    for r in rows:
        total = f"{r.total:.1f}" + (f" ± {r.sd:.1f}" if r.sd else "")
        delta = "—" if base is None or r.strategy == "greedy" else f"{r.total - base:+.1f}"
        ratio = f"{r.projected / r.total:.2f}" if r.projected and r.total else "—"
        t.add_row(
            r.strategy,
            total,
            delta,
            *(f"{r.by_slot[s]:.0f}" for s in config.SLOTS),
            f"{r.projected:.1f}" if r.projected else "—",
            ratio,
            f"{r.zero_picks}/{len(r.picks)}",
            f"{100 * r.total / ceiling:.0f}%" if ceiling else "—",
            *([r.finish or "—"] if against else []),
        )
    console.print(t)
    if against:
        _render_invented(rows, against)


def _render_invented(rows, against: str) -> None:
    """The caveat the finish column cannot be read without.

    Printed under every table that has one, not once per session and not in the help
    text, because the row is what gets copied into a message to somebody.
    """
    from . import backtest as bt

    standing = next((r.standing for r in rows if r.standing), ())
    console.print(f"[yellow]Against {against}. {bt.INVENTED}.[/yellow]")
    console.print(
        "  They finished " + ", ".join(f"{name} {tds:.0f}" for name, tds in standing) + "."
    )
    console.print(
        "  Read [bold]winprob[/bold] on the finish column and not on TDs: giving up a "
        "touchdown for a larger share of the pot is the whole of what it does, so it is "
        "meant to lose the other one. And a season is a single trial — a finish here, "
        "or fifteen of them, settles nothing about the policy either way."
    )


def _render_picks(summary) -> None:
    if summary.against:
        console.print(f"[yellow]Against {summary.against}.[/yellow]")
    t = Table(
        "Week",
        "Slot",
        "Player",
        "xTD",
        "Actual",
        "Running",
        title=f"{summary.strategy} picks, {summary.season}",
    )
    running = 0.0
    for p in summary.picks:
        running += p.actual
        t.add_row(
            str(p.week),
            p.slot,
            f"{p.player_name} ({p.team})" if p.player_name else "—",
            f"{p.projected:.2f}",
            f"{p.actual:.0f}",
            f"{running:.0f}",
        )
    console.print(t)


@app.command()
def sweep(
    season: str = SeasonsOpt,
    discount: str = typer.Option("0.9,0.95,0.985,1.0", help="FUTURE_DISCOUNT grid"),
    prior_weight: str = typer.Option("5,7,10", help="PRIOR_WEIGHT_GAMES grid"),
    role_source: str | None = RoleSourceOpt,
    vegas_horizon: int | None = HorizonOpt,
    input_policy: str = PolicyOpt,
    decision_times: Path | None = DecisionOpt,
    csv_out: Path | None = CsvOpt,
    db_path: Path | None = DbOpt,
):
    """Grid-search the future discount and prior weight against the greedy baseline."""
    from . import backtest as bt

    role_source, times = _input_options(input_policy, role_source, vegas_horizon, decision_times)
    conn = _conn(db_path)
    seasons = _seasons(season)
    for yr in seasons:
        _backtest_ready(conn, yr)
    discounts = _floats(discount, "discount")
    weights = _floats(prior_weight, "prior weight")
    console.print(
        f"Sweeping {len(weights)}x{len(discounts)} cells over {len(seasons)} season(s)..."
    )
    df = bt.sweep(
        conn,
        seasons,
        discounts,
        weights,
        role_source=role_source,
        vegas_horizon=vegas_horizon,
        input_policy=input_policy,
        decision_times=times,
    )
    _render_input_provenance(df.attrs["input_provenance"])
    if csv_out:
        df.to_csv(csv_out, index=False)
        from . import benchmark as bench

        bench.write_json(
            csv_out.with_suffix(".metadata.json"),
            dict(
                schema_version=bench.SCHEMA_VERSION,
                scoring_version=scoring.SCORING_VERSION,
                seasons=seasons,
                discounts=discounts,
                prior_weights=weights,
                input_policy=input_policy,
                role_source=role_source,
                vegas_horizon=vegas_horizon,
                decision_times=[
                    dict(season=s, week=w, decision_at=t) for (s, w), t in (times or {}).items()
                ],
                input_provenance=df.attrs["input_provenance"],
                constants=bench.constants(),
                code=bench.code_identity(),
                assumptions=bench.ASSUMPTIONS,
                input_metadata=bench.input_metadata(conn, {}),
            ),
        )
        console.print(f"Wrote {csv_out} and reproducibility metadata")

    grid = df.pivot_table(index="prior_weight", columns="discount", values="delta", aggfunc="mean")
    sd = df.groupby(["prior_weight", "discount"]).delta.std().mean()
    t = Table(
        "prior weight",
        *(f"{d:g}" for d in grid.columns),
        title="Mean optimizer-minus-greedy TDs per season",
    )
    best = grid.stack().idxmax()
    for w, row in grid.iterrows():
        cells = []
        for d, v in row.items():
            cell = f"{v:+.2f}"
            cells.append(f"[bold green]{cell}[/bold green]" if (w, d) == best else cell)
        marker = " *" if w == config.PRIOR_WEIGHT_GAMES else ""
        t.add_row(f"{w:g}{marker}", *cells)
    console.print(t)
    console.print(
        f"Best cell: prior weight {best[0]:g}, discount {best[1]:g}. "
        f"Current config: prior weight {config.PRIOR_WEIGHT_GAMES:g}, "
        f"discount {config.FUTURE_DISCOUNT:g} (*)."
    )
    console.print(
        f"[yellow]Per-season SD of the delta is {sd:.1f} TD.[/yellow] Differences smaller than "
        "that are noise; prefer a cell that wins in most seasons over the maximum."
    )


# --- projection benchmark ---------------------------------------------------
@app.command("models")
def list_models():
    """List the projection models the benchmark can run."""
    from . import models

    t = Table("Model", "Role", title="Projection models")
    roles = {
        "random": "null — lambda shuffled within slot-week",
        "within-player": "null — each player's lambda shuffled across their weeks",
        "historical-rate": "baseline — prior-season TD/game, unregressed",
        "regressed-rate": "baseline — prior season regressed to the positional mean",
        "current-season-rate": "baseline — current season only, shrunk",
        "vegas-environment": "baseline — team scoring environment, no player TD history",
        "player-vegas": "baseline — base rate x Vegas; the principal challenger",
        "shipped": "the production model, frozen",
        "no-vegas": "ablation — shipped without the Vegas multiplier",
        "base-rate-only": "ablation — shipped base rate, no matchup context",
    }
    for name in models.BUILDERS:
        t.add_row(name, roles.get(name, ""))
    console.print(t)


def _model_set(spec: str) -> dict:
    """Resolve a comma-separated model list, or `bakeoff` / `all`."""
    from . import models

    if spec == "bakeoff":
        names = list(models.BAKEOFF)
    elif spec == "all":
        names = list(models.BUILDERS)
    else:
        names = [n.strip() for n in spec.split(",") if n.strip()]
    return {n: _builder(n) for n in names}


def _primary_k() -> int:
    """Read when `evaluate` runs, so defining the command loads no research module."""
    from . import evaluate as ev

    return ev.PRIMARY_K


@app.command()
def evaluate(
    season: str = SeasonsOpt,
    model: str = typer.Option("bakeoff", "--model", "-m", help="Models, or bakeoff / all"),
    baseline: str = typer.Option("shipped", help="Model every comparison is paired against"),
    seeds: str = typer.Option("0-19", help="Shuffled model seeds: integers or ranges"),
    k: int = typer.Option(
        default_factory=_primary_k,
        help="Primary top-k for paired comparison (default: the study's primary k)",
        show_default=False,
    ),
    role_source: str | None = RoleSourceOpt,
    vegas_horizon: int | None = HorizonOpt,
    input_policy: str = PolicyOpt,
    decision_times: Path | None = DecisionOpt,
    csv_out: Path | None = CsvOpt,
    db_path: Path | None = DbOpt,
):
    """Report common-pool ranking diagnostics and separate achieved season scores."""
    from . import evaluate as ev
    from . import models

    chosen = _model_set(model)
    try:
        ev.validate_models(chosen, baseline)
        seed_values = models.parse_seeds(seeds)
        if k < 1:
            raise ValueError("k must be positive")
    except (ValueError, KeyError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    role_source, times = _input_options(input_policy, role_source, vegas_horizon, decision_times)
    conn = _conn(db_path)
    seasons = _seasons(season)
    for yr in seasons:
        _backtest_ready(conn, yr)
    console.print(f"Evaluating {len(chosen)} model(s) over {len(seasons)} season(s)...")
    df = ev.forecast_set(
        conn,
        seasons,
        chosen,
        role_source=role_source,
        vegas_horizon=vegas_horizon,
        baseline=baseline,
        seeds=seed_values,
        input_policy=input_policy,
        decision_times=times,
        log=None,
    )
    console.print(
        f"{len(df):,} forecast rows; shuffled seeds {seed_values}; "
        "deterministic models run once. Ranking uses all hard-eligible candidates "
        "before assignment pruning."
    )
    if csv_out:
        from . import benchmark as bench

        df.to_csv(csv_out, index=False)
        bench.save_frame(csv_out.with_suffix(".replays.csv"), pd.DataFrame(df.attrs["replays"]))
        bench.save_frame(csv_out.with_suffix(".picks.csv"), pd.DataFrame(df.attrs["picks"]))
        bench.write_json(
            csv_out.with_suffix(".metadata.json"),
            dict(
                schema_version=bench.SCHEMA_VERSION,
                scoring_version=scoring.SCORING_VERSION,
                baseline=baseline,
                models=list(chosen),
                seeds=seed_values,
                seasons=seasons,
                input_policy=input_policy,
                role_source=role_source,
                vegas_horizon=vegas_horizon,
                decision_times=[
                    dict(season=s, week=w, decision_at=t) for (s, w), t in (times or {}).items()
                ],
                input_provenance=df.attrs["input_provenance"],
                constants=bench.constants(),
                code=bench.code_identity(),
                assumptions=bench.ASSUMPTIONS,
                input_metadata=bench.input_metadata(conn, {}),
            ),
        )
        console.print(f"Wrote {csv_out}, replay picks, scores and reproducibility metadata")
    _render_evaluation(df, baseline, k)


def _input_options(policy, role, horizon, path):
    try:
        role = projections.validate_policy(policy, role, horizon)
        if policy == "snapshots" and path is None:
            raise ValueError("--input-policy snapshots requires --decision-times PATH")
        if path is not None and policy != "snapshots":
            raise ValueError("--decision-times requires --input-policy snapshots")
        times = snapshots.decision_times(path) if path is not None else None
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(
        f"Input policy: {policy}; roles: {role}; scoring: all thrown/scored TD credits. "
        "Historical schedules, weekly reports and corrected stats are approximations."
    )
    from . import benchmark as bench

    console.print(
        "Candidates: active-roster QB/RB/WR/TE, with latest stat-team fallback; "
        "hard exclusions before pruning/ranking, eligible zeros retained, player-ID ties. "
        f"Code fingerprint: {bench.code_identity()['code_hash']}"
    )
    return role, times


def _render_input_provenance(records):
    observed = [r for r in records if r["inputs"]]
    if not observed:
        return
    table = Table("Decision", "Input season", "Feed", "Observed (UTC)", "Age hours", "Status")
    for record in observed:
        for feed in record["inputs"]:
            table.add_row(
                f"{record['season']} W{record['week']}: {record['decision_at']}",
                str(feed["season"]),
                feed["feed"],
                feed["observed_at"] or "none",
                "—" if feed["age_hours"] is None else f"{feed['age_hours']:.2f}",
                "missing / fallback"
                if feed["missing"]
                else "stale"
                if feed["stale"]
                else "observed",
            )
    console.print(table)


def _render_evaluation(df, baseline: str, k: int) -> None:
    from . import evaluate as ev

    _render_input_provenance(df.attrs["input_provenance"])
    console.print(
        "Ranking diagnostics: TDs per ranked candidate; shared baseline greedy depletion."
    )
    console.print(ev.top_k(df).to_string(index=False))
    if df.model.nunique() > 1:
        console.print(f"Paired ranking differences vs {baseline}; uncertainty across seasons:")
        console.print(ev.paired_top_k(df, baseline, k).to_string(index=False))
        console.print("Paired deviance on baseline lambda > 0.30 (diagnostic):")
        console.print(ev.paired_deviance(df, baseline).to_string(index=False))
    console.print("Calibration computed separately for each seed (slope, intercept):")
    for (model, seed), group in df.groupby(["model", "seed"]):
        fit = ev.calibration(group)
        console.print(
            f"{model} seed={seed}: slope {fit['slope']:.3f}, intercept {fit['intercept']:.3f}"
        )
    console.print("Achieved season TDs: each strategy uses its own player history.")
    console.print(
        ev.replay_summary(pd.DataFrame(df.attrs["replays"]), baseline).to_string(index=False)
    )


@app.command()
def diagnose(
    run: Annotated[Path, typer.Option("--run", help="A completed benchmark output directory")],
    out: Annotated[Path, typer.Option("--out", help="Where to write the diagnostic export")],
    model: str = typer.Option("shipped", help="Model whose saved surface to diagnose"),
    seed: int = typer.Option(-1, help="Seed; -1 is the deterministic sentinel"),
    seasons: str | None = typer.Option(None, "--season", help="Season or range, e.g. 2019-2025"),
):
    """Describe a saved study's rate errors by population, position, rate, availability
    and forecast horizon, and write a dated readiness note.

    Descriptive only: it fits no correction and selects no model. Group definitions are
    frozen before any outcome is read.
    """
    from . import diagnostics

    try:
        diagnostics.export(
            run,
            out,
            model=model,
            seed=seed,
            seasons=_seasons(seasons) if seasons else None,
            log=console.print,
        )
    except (ValueError, OSError, KeyError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


@app.command()
def captures(
    season: int = SeasonOpt,
    week: int | None = WeekOpt,
    decision: str | None = typer.Option(None, "--decision", help="Show one decision in full"),
    db_path: Path | None = DbOpt,
):
    """List captured decisions, or show one with its identity, inputs and advice."""
    from . import verify

    conn = _conn(db_path)
    if decision:
        _one_capture(conn, _decision_id(conn, decision))
        return
    found = verify.decisions(conn, season, week)
    if not len(found):
        console.print(f"[yellow]No captured decisions for {season}.[/yellow]")
        return
    events = capture.events(conn, season)
    t = Table("Decision", "Week")
    t.add_column("Made", no_wrap=True)
    t.add_column("Events")
    t.add_column("Submitted")
    for row in found.itertuples():
        mine = events[events.decision_id.eq(row.decision_id)]
        kinds = mine.kind.value_counts().to_dict()
        submitted = mine[mine.kind.eq("submitted")]
        t.add_row(
            row.decision_id[:12],
            str(row.week),
            _made(row.decision_at),
            ", ".join(f"{k}x{v}" for k, v in sorted(kinds.items())),
            ", ".join(f"{r.slot}:{r.player_id}" for r in submitted.itertuples()) or "-",
        )
    console.print(t)


def _made(decision_at: str) -> str:
    """When a decision was made, on the Eastern clock its deadlines are read on."""
    from .. import state
    from ..cli import _fmt_dt

    return _fmt_dt(state.eastern_now(datetime.fromisoformat(decision_at)))


def _decision_id(conn, given: str) -> str:
    """The full id of the one decision `given` starts, or exit naming why there is none.

    The listing prints the first twelve characters, so that is what gets typed back.
    """
    found = [
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT decision_id FROM decision_events "
            "WHERE substr(decision_id, 1, length(?)) = ?",
            (given, given),
        )
    ]
    if len(found) == 1:
        return found[0]
    if found:
        console.print(f"[red]{given} starts {len(found)} decisions; give more of the id.[/red]")
    else:
        console.print(f"[red]No captured decision {given}[/red]")
    raise typer.Exit(1)


def _one_capture(conn, decision_id: str) -> None:
    events = db.read_df(
        conn,
        "SELECT * FROM decision_events WHERE decision_id = ? ORDER BY event_id",
        (decision_id,),
    )
    if not len(events):
        console.print(f"[red]No captured decision {decision_id}[/red]")
        raise typer.Exit(1)
    identity = capture.recorded_identity(conn, decision_id)
    head = events[events.kind.eq("surface")]
    console.print(f"[bold]{decision_id}[/bold]")
    console.print(
        f"  {int(events.season.iloc[0])} week {int(events.week.iloc[0])} "
        f"at {events.decision_at.iloc[0]}"
    )
    console.print(f"  model {identity.get('model')} / calibrator {identity.get('calibrator')}")
    console.print(f"  source {identity.get('code_hash')} (revision {identity.get('revision')})")
    drift = capture.constants_drift(identity.get("constants", {}))
    if drift:
        console.print("[yellow]  Constants have moved since this decision:[/yellow]")
        for name, change in sorted(drift.items()):
            console.print(f"    {name}: recorded {change['recorded']}, now {change['current']}")
    if len(head):
        detail = json.loads(head.detail.iloc[0])
        console.print(
            f"  surface {detail['rows']:,} rows over weeks "
            f"{detail['weeks'][0]}-{detail['weeks'][-1]}; {len(detail['used'])} used"
        )
    inputs = db.read_df(
        conn,
        "SELECT season, feed, observed_at, missing, age_hours, stale FROM decision_inputs "
        "WHERE decision_id = ? ORDER BY season, feed",
        (decision_id,),
    )
    t = Table("Season", "Feed", "Observed (UTC)", "Age (h)", "State")
    for row in inputs.itertuples():
        state_text = "missing" if row.missing else ("stale" if row.stale else "fresh")
        t.add_row(
            str(int(row.season)),
            row.feed,
            row.observed_at or "-",
            "-" if row.age_hours is None or pd.isna(row.age_hours) else f"{row.age_hours:.1f}",
            state_text,
        )
    console.print(t)
    for row in events[events.kind.isin(["advice", "hold", "commit"])].itertuples():
        detail = json.loads(row.detail)
        if row.kind == "advice":
            pick = detail.get("recommended")
            console.print(
                f"  {row.slot}: "
                + (
                    f"{pick['player_name']} ({pick['lam']:.3f} xTD)"
                    if pick
                    else (
                        f"locked {detail['locked_player']}"
                        if detail.get("locked_player")
                        else "no candidate"
                    )
                )
            )
        else:
            console.print(f"    {row.slot} {row.kind} against premium {detail['premium']}")


@app.command()
def verify_capture(
    season: int = SeasonOpt,
    week: int | None = WeekOpt,
    decision: str | None = typer.Option(None, "--decision", help="Verify one decision"),
    db_path: Path | None = DbOpt,
    allow_code_drift: bool = typer.Option(
        False, help="Verify against a source tree the decisions were not captured under"
    ),
):
    """Reconstruct captured decisions and check them against a snapshot replay.

    Reconstruction re-derives the advice from the stored surface alone: if it differs
    from what was recorded, something the decision depended on was never written down.
    Parity rebuilds the same instant from the archived feeds: if that differs, the live
    path and replay are not the same function, which is what every replay assumes.
    """
    from . import verify

    conn = _conn(db_path)
    found = verify.decisions(conn, season, week)
    if decision:
        found = found[found.decision_id.eq(_decision_id(conn, decision))]
        if not len(found):
            console.print(f"[red]No captured decision {decision} in {season}[/red]")
            raise typer.Exit(1)
    if not len(found):
        # Nothing verified is not verification: exiting zero here would tell a caller the
        # season's decisions check out when there are none to check.
        console.print(f"[red]No captured decisions for {season}; nothing was verified.[/red]")
        raise typer.Exit(1)
    t = Table("Decision", "Week")
    t.add_column("Made", no_wrap=True)
    for name in ("Reconstructs", "Parity", "Detail"):
        t.add_column(name)
    failed = 0
    overridden = 0
    tolerated = 0
    for row in found.itertuples():
        rebuilt = verify.reconstruction(conn, row.decision_id, allow_code_drift=allow_code_drift)
        matched = verify.parity(conn, row.decision_id, allow_code_drift=allow_code_drift)
        ok = rebuilt["ok"] and matched["ok"]
        failed += not ok
        notes = [n for n in (rebuilt.get("reason"), matched.get("reason")) if n]
        # Drift outside the enforced fingerprint is accepted by design, and saying nothing
        # about it would leave a verified row indistinguishable from one where the tree had
        # moved -- reported by the very code the fingerprint does not cover.
        #
        # The override is checked first because it is the case the notice must never be
        # confused with: under `--allow-code-drift` the enforced fingerprint itself can have
        # moved, and calling that "outside the decision path" would assert the opposite of
        # what happened. Testing it first also settles a legacy whole-tree capture, whose
        # two flags always move together and for which "outside" means nothing.
        drift = rebuilt.get("drift") or {}
        if ok and drift.get("code_hash_changed"):
            overridden += 1
            scope = drift.get("fingerprint_scope") or "enforced"
            notes.append(f"[red]{scope} fingerprint moved; accepted by override[/red]")
        elif ok and drift.get("whole_tree_changed"):
            tolerated += 1
            notes.append("[yellow]source outside the decision path moved (accepted)[/yellow]")
        t.add_row(
            row.decision_id[:12],
            str(row.week),
            _made(row.decision_at),
            "[green]yes[/green]" if rebuilt["ok"] else "[red]no[/red]",
            "[green]yes[/green]" if matched["ok"] else "[red]no[/red]",
            "; ".join(notes) or "-",
        )
    console.print(t)
    # Before the verdict, not after it: an override is worth reporting on a run that ends
    # in failure too, and neither notice changes the exit code.
    if overridden:
        console.print(
            f"[red]{overridden} passed only because the fingerprint check was overridden. "
            "The source these checks enforce had moved and they were accepted anyway.[/red]"
        )
    if tolerated:
        console.print(
            f"[yellow]{tolerated} verified against a source tree that moved outside the "
            "decision path, with the enforced fingerprint unchanged. That fingerprint covers "
            "what a decision is a function of; the whole-tree hash is recorded beside it."
            "[/yellow]"
        )
    if failed:
        console.print(f"[red]{failed} of {len(found)} captured decisions did not verify.[/red]")
        raise typer.Exit(1)
    console.print(
        f"[green]All {len(found)} captured decisions reconstruct and match replay.[/green]"
    )


@app.command()
def benchmark(
    config_path: Annotated[Path, typer.Option("--config", help="Experiment TOML")],
    output: Annotated[Path, typer.Option(help="Ignored research dataset and experiment artifacts")],
    resume: bool = typer.Option(False, help="Continue compatible season checkpoints"),
):
    """Run a specified benchmark against a frozen, fully audited research database."""
    from . import benchmark as bench

    try:
        bench.run(config_path, output, resume=resume, log=console.print)
    except (ValueError, OSError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
