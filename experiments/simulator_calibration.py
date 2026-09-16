"""Fit and check the simulator's joint structure: `simulator_calibration.py [study]`.

Two parameters are measured rather than tuned -- the throwing share and the receiving
shares come straight out of `touchdown_credits.kind` and `player_weeks` -- and one is
fitted, `k_game`, from the covariance between opposing teams in the same game.

The three checks below are **not** fitted, which is the point of running them. Matching a
covariance says nothing about whether the marginal spread or the conditional tails come out
right, and the shared-credit mechanism is asserted by the model rather than estimated from
it. Each is printed pass or fail; none of them is allowed to be quietly split the
difference with.

Inputs are the frozen study: `dataset.sqlite` for the observed credits and each season's
`forecasts.parquet` for the shipped lambda paired with what actually happened. Nothing here
reads the operational pick database.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pool import simulate

STUDY = sys.argv[1] if len(sys.argv) > 1 else "roster-snapshot-repair"
SEASONS = range(2011, 2026)
CHUNK = 250          # games simulated at once; the full population does not fit
CHUNK_SIMS = 500     # draws per chunk, pooled across chunks for the correlation check
SEED = 20260916
SOURCE = Path("data/experiments") / STUDY
OUT = Path("experiments/results/phase4-simulator")
BINS = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.9, 10.0]


def forecasts() -> pd.DataFrame:
    """Shipped, hard-eligible player-weeks: the population the tool actually picks from.

    A player who was eligible and did not play stays in, scoring zero, because that is
    exactly what the pool would have paid for picking him.
    """
    frames = []
    for season in SEASONS:
        path = SOURCE / str(season) / "forecasts.parquet"
        frame = pd.read_parquet(path)
        frame = frame[frame.model.eq("shipped") & frame.hard_eligible]
        frames.append(
            frame[["season", "week", "team", "opponent", "position", "lam", "actual_tds"]]
        )
    out = pd.concat(frames, ignore_index=True)
    return out[out.lam.gt(0)].reset_index(drop=True)


def measured_shares(conn) -> dict:
    """The two parameters that are read off history rather than fitted to it."""
    kind = pd.read_sql(
        "SELECT t.kind, p.position, COUNT(*) n FROM touchdown_credits t "
        "JOIN games g ON g.game_id = t.game_id "
        "JOIN player_weeks p ON p.player_id = t.player_id AND p.game_id = t.game_id "
        "WHERE g.game_type = 'REG' AND g.season BETWEEN 2011 AND 2025 "
        "AND p.position IN ('QB','RB','WR','TE') GROUP BY t.kind, p.position",
        conn,
    ).pivot(index="position", columns="kind", values="n").fillna(0)
    offence = pd.read_sql(
        "SELECT position, SUM(rush_td) rush, SUM(rec_td) rec FROM player_weeks "
        "WHERE season BETWEEN 2011 AND 2025 AND season_type = 'REG' "
        "AND position IN ('QB','RB','WR','TE') GROUP BY position",
        conn,
    ).set_index("position")
    return {
        "throw_share": float(kind.loc["QB", "throwing"] / kind.loc["QB"].sum()),
        "receive_share": {
            position: (0.0 if position == "QB" else float(row.rec / (row.rush + row.rec)))
            for position, row in offence.iterrows()
        },
    }


def team_games(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per team-week: modelled rate against what that team's modelled players did."""
    grouped = rows.groupby(["season", "week", "team", "opponent"], as_index=False).agg(
        lam=("lam", "sum"), actual=("actual_tds", "sum")
    )
    pairs = grouped.merge(
        grouped,
        left_on=["season", "week", "team", "opponent"],
        right_on=["season", "week", "opponent", "team"],
        suffixes=("", "_opp"),
    )
    return pairs[pairs.team < pairs.team_opp].reset_index(drop=True)


def fit_k_game(pairs: pd.DataFrame) -> float:
    """A moment estimator on the model's own rates, not on raw totals.

    `Cov(A, B) = Lam_A * Lam_B / k_game` for opposing teams sharing one game factor, so
    matching the summed cross-product of residuals to the summed product of rates gives
    `k_game` directly. Using raw totals instead would fold every matchup difference the
    projection already knows about into the factor and count it twice.
    """
    residual = (pairs.actual - pairs.lam) * (pairs.actual_opp - pairs.lam_opp)
    return float((pairs.lam * pairs.lam_opp).sum() / residual.sum())


def check_marginal_spread(rows: pd.DataFrame, k: float) -> dict:
    """Does the fitted factor produce the spread actually observed, one player at a time?"""
    catchers = rows[rows.position.ne("QB")]
    model = (catchers.lam + catchers.lam**2 / k).sum()
    observed = ((catchers.actual_tds - catchers.lam) ** 2).sum()
    return dict(
        check="marginal spread, non-quarterbacks",
        observed=float(observed),
        model=float(model),
        ratio=float(observed / model),
        n=int(len(catchers)),
    )


def check_tails(rows: pd.DataFrame, k: float) -> pd.DataFrame:
    """Conditional tails by lambda bin: the mean can be right while these are wrong.

    Two model columns, because two different things can be wrong and only one of them is
    this stage's business. `model_*` uses the shipped lambda as it stands. `shape_*` first
    rescales each bin's lambdas to the mean actually observed in that bin, which removes the
    projection's own calibration tilt and leaves only the question this stage can answer:
    given the right average, does the Gamma-Poisson mixture put the right mass in the tail?
    A gap in `model_*` that closes in `shape_*` is inherited from `projections.lam` and is
    out of scope here; a gap that survives belongs to the simulator.
    """
    catchers = rows[rows.position.ne("QB")].copy()
    catchers["bin"] = pd.cut(catchers.lam, BINS)
    out = []
    for name, sub in catchers.groupby("bin", observed=True):
        lam = sub.lam.to_numpy()
        tilt = sub.actual_tds.mean() / lam.mean() if lam.mean() else 1.0
        row = dict(bin=str(name), n=int(len(sub)), lam=float(lam.mean()),
                   actual=float(sub.actual_tds.mean()), tilt=float(tilt))
        for threshold in (1, 2, 3):
            row[f"obs_ge{threshold}"] = float((sub.actual_tds >= threshold).mean())
            # Poisson mixed over Gamma(k, 1/k) is Negative Binomial with shape k, so the
            # implied tails are closed-form and need no simulation to state.
            row[f"model_ge{threshold}"] = float(
                stats.nbinom.sf(threshold - 1, k, k / (k + lam)).mean()
            )
            row[f"shape_ge{threshold}"] = float(
                stats.nbinom.sf(threshold - 1, k, k / (k + lam * tilt)).mean()
            )
        out.append(row)
    return pd.DataFrame(out)


def check_shared_credit(rows: pd.DataFrame, params, k: float, seed: int) -> dict:
    """The correlation the model asserts rather than estimates, on one population twice.

    Every modelled player on a team is simulated, not just its leading three. A quarterback
    throws to his whole roster, so concentrating his arm onto one receiver manufactures a
    correlation the real one does not have -- the first version of this check did exactly
    that and reported +0.536 against an observed +0.358. Games are simulated in chunks and
    the draws pooled, because the whole population times the draw count does not fit in
    memory at once.
    """
    rows = rows.copy()
    rows["role"] = np.where(
        rows.position.eq("QB"), "QB", np.where(rows.position.eq("RB"), "RB", "PC")
    )
    rows["unit"] = rows.season.astype(str) + "-" + rows.week.astype(str) + "-" + rows.team
    rows["game"] = [
        f"{season}-{week}-" + "-".join(sorted((team, opp)))
        for season, week, team, opp in zip(
            rows.season, rows.week, rows.team, rows.opponent, strict=True
        )
    ]
    rows["pid"] = rows.index.astype(str)

    lead = rows.sort_values("lam").drop_duplicates(["unit", "role"], keep="last")
    leaders = {(row.unit, row.role): row.pid for row in lead.itertuples()}
    observed = lead.pivot_table(
        index="unit", columns="role", values="actual_tds", aggfunc="first"
    ).dropna()
    complete = set(observed.index)

    games = sorted(rows.game.unique())
    pooled: dict[str, list] = {"QB": "", "PC": "", "RB": ""}
    pooled = {role: [] for role in pooled}
    rng = np.random.default_rng(seed)
    for start in range(0, len(games), CHUNK):
        part = rows[rows.game.isin(set(games[start : start + CHUNK]))]
        frame = pd.DataFrame(
            dict(
                player_id=part.pid, position=part.position, team=part.unit,
                game_id=part.game, week=1, lam=part.lam,
            )
        ).reset_index(drop=True)
        draws = simulate.sample(
            frame, [1], sims=CHUNK_SIMS, seed=int(rng.integers(1 << 30)), params=params
        )
        for unit in sorted(set(part.unit) & complete):
            for role in pooled:
                pooled[role].append(draws.values[draws.index[(1, leaders[(unit, role)])]])
    drawn = {role: np.concatenate(values) for role, values in pooled.items()}
    return dict(
        observed_qb_pc=float(np.corrcoef(observed.QB, observed.PC)[0, 1]),
        simulated_qb_pc=float(np.corrcoef(drawn["QB"], drawn["PC"])[0, 1]),
        observed_qb_rb=float(np.corrcoef(observed.QB, observed.RB)[0, 1]),
        simulated_qb_rb=float(np.corrcoef(drawn["QB"], drawn["RB"])[0, 1]),
        observed_pc_rb=float(np.corrcoef(observed.PC, observed.RB)[0, 1]),
        simulated_pc_rb=float(np.corrcoef(drawn["PC"], drawn["RB"])[0, 1]),
        team_weeks=int(len(observed)),
        draws_per_unit=CHUNK_SIMS,
        k_game=float(k),
    )


def check_identity_and_dispersion(conn) -> dict:
    """Two diagnostics that decide how to read the shared-credit check above.

    The first asks what the coupling would be if a quarterback's credits were *exactly* his
    receivers' receiving touchdowns -- the identity the model asserts -- using each team's
    true lead receiver by targets rather than by projected rate. If that lands near the
    simulated figure, the mechanism is right and the check's gap is the projection failing
    to name which receiver leads a team, not the simulator coupling them too hard.

    The second measures the dispersion of a team's passing touchdowns. Substitution between
    receivers can only come from a total that is *less* variable than Poisson: splitting a
    Poisson total multinomially returns independent Poissons, so no allocation rule whatever
    can manufacture competition on top of one.
    """
    pw = pd.read_sql(
        "SELECT season, week, team, position, pass_td, rush_td, rec_td, attempts, targets "
        "FROM player_weeks WHERE season BETWEEN 2011 AND 2025 AND season_type = 'REG' "
        "AND position IN ('QB','RB','WR','TE')",
        conn,
    )
    pw["unit"] = pw.season.astype(str) + "-" + pw.week.astype(str) + "-" + pw.team
    total = pw.groupby("unit").pass_td.sum()
    lead = pw[pw.position.ne("QB")].sort_values("targets").drop_duplicates("unit", keep="last")
    passer = pw[pw.position.eq("QB")].sort_values("attempts").drop_duplicates("unit", keep="last")
    joined = pd.DataFrame(
        {
            "lead_rec": lead.set_index("unit").rec_td,
            "identity_qb": pw.groupby("unit").rec_td.sum(),
            "real_qb": passer.set_index("unit").pass_td + passer.set_index("unit").rush_td,
        }
    ).dropna()
    return dict(
        team_pass_mean=float(total.mean()),
        team_pass_var=float(total.var()),
        team_pass_dispersion=float(total.var() / total.mean()),
        corr_lead_identity_qb=float(np.corrcoef(joined.lead_rec, joined.identity_qb)[0, 1]),
        corr_lead_real_qb=float(np.corrcoef(joined.lead_rec, joined.real_qb)[0, 1]),
        lead_share_of_team_receiving=float(
            (joined.lead_rec / joined.identity_qb.replace(0, np.nan)).mean()
        ),
        team_weeks=int(len(joined)),
    )


def main() -> None:
    import sqlite3

    OUT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(f"file:{SOURCE / 'dataset.sqlite'}?mode=ro", uri=True)
    rows = forecasts()
    shares = measured_shares(conn)
    pairs = team_games(rows)
    k = fit_k_game(pairs)
    params = simulate.Params(k_game=k, **shares)

    spread = check_marginal_spread(rows, k)
    tails = check_tails(rows, k)
    credit = check_shared_credit(rows, params, k, SEED)
    truth = check_identity_and_dispersion(conn)
    fitted = dict(
        study=STUDY,
        seasons=[min(SEASONS), max(SEASONS)],
        rows=int(len(rows)),
        games=int(len(pairs)),
        k_game=k,
        sims=CHUNK_SIMS,
        seed=SEED,
        **shares,
    )
    tails.to_csv(OUT / "tails.csv", index=False)
    pd.DataFrame([spread]).to_csv(OUT / "marginal-spread.csv", index=False)
    pd.DataFrame([credit]).to_csv(OUT / "shared-credit.csv", index=False)
    pd.DataFrame([truth]).to_csv(OUT / "identity-and-dispersion.csv", index=False)
    (OUT / "fitted.json").write_text(json.dumps(fitted, indent=2, sort_keys=True) + "\n")

    print(f"rows {len(rows)}  team-games {len(pairs)}")
    print(f"MEASURED throw_share {shares['throw_share']:.4f}")
    print("MEASURED receive_share " + ", ".join(
        f"{k2} {v:.3f}" for k2, v in sorted(shares["receive_share"].items())))
    print(f"FITTED   k_game {k:.3f}\n")
    print(f"CHECK marginal spread  observed/model = {spread['ratio']:.3f}  (1.00 is exact)")
    print(f"CHECK shared credit    QB-catcher observed {credit['observed_qb_pc']:+.3f} "
          f"simulated {credit['simulated_qb_pc']:+.3f}")
    print(f"CHECK shared credit    QB-back    observed {credit['observed_qb_rb']:+.3f} "
          f"simulated {credit['simulated_qb_rb']:+.3f}")
    print(f"CHECK shared credit    catch-back observed {credit['observed_pc_rb']:+.3f} "
          f"simulated {credit['simulated_pc_rb']:+.3f}")
    print(f"      over {credit['team_weeks']} team-weeks")
    print(f"  ...with the TRUE lead receiver, reality gives "
          f"{truth['corr_lead_identity_qb']:+.3f} under the exact identity and "
          f"{truth['corr_lead_real_qb']:+.3f} against the real passer")
    print(f"CHECK team passing TDs var/mean = {truth['team_pass_dispersion']:.3f} "
          f"(under 1.00 means substitution the model does not have)\n")
    print(tails.to_string(index=False))


if __name__ == "__main__":
    main()
