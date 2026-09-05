"""Complete pool touchdown credits and conservative game-level scoring coverage."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pandas as pd

from . import config, db, state

CREDIT_COLUMNS = ["game_id", "play_id", "player_id", "kind", "player_name", "team"]
RESULT_COLUMNS = [
    "game_id",
    "season",
    "week",
    "complete",
    "reason",
    "home_score",
    "away_score",
    "imported_at",
]
# Extra fields may exist, but these fields are essential to declare even a zero-TD game complete.
PBP_REQUIRED = {
    "game_id",
    "play_id",
    "touchdown",
    "pass_touchdown",
    "td_player_id",
    "passer_player_id",
    "two_point_attempt",
    "extra_point_attempt",
    "play_type",
    "desc",
    "total_home_score",
    "total_away_score",
}


def transform_touchdowns(
    raw: pd.DataFrame, games: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep one scorer and, on credited passing TDs, one passer per play.

    td_player_id is authoritative even on returns, recoveries and laterals. Summary
    categories never enter the calculation. Conflicting duplicate rows fail coverage.
    """
    missing = PBP_REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"play-by-play missing required columns: {sorted(missing)}")
    if raw[["game_id", "play_id"]].isna().any().any():
        raise ValueError("play-by-play contains unresolved game/play identifiers")
    raw = raw.copy()
    for column in ("touchdown", "pass_touchdown", "two_point_attempt", "extra_point_attempt"):
        raw[column] = pd.to_numeric(raw[column], errors="raise")
        if not raw[column].dropna().isin([0, 1]).all():
            raise ValueError(f"invalid {column} flags in play-by-play")
    for column in ("play_id", "total_home_score", "total_away_score"):
        raw[column] = pd.to_numeric(raw[column], errors="raise")
        if not raw[column].dropna().mod(1).eq(0).all():
            raise ValueError(f"non-integer {column} in play-by-play")
    credits, results = [], []
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    scheduled = games.set_index("game_id")
    for gid, rows in raw.groupby("game_id", sort=False):
        if gid not in scheduled.index:
            continue  # postseason and games outside the requested season
        game = scheduled.loc[gid]
        relevant = sorted(PBP_REQUIRED) + [c for c in ("no_play", "td_team") if c in rows]
        distinct = rows.drop_duplicates(relevant)
        conflict = distinct.play_id.duplicated().any()
        plays = rows.drop_duplicates("play_id", keep="last").sort_values("play_id")
        td = plays.touchdown.eq(1)
        conversion = plays.two_point_attempt.eq(1) | plays.extra_point_attempt.eq(1)
        negated = plays.play_type.eq("no_play")
        if "no_play" in plays:
            negated |= plays.no_play.eq(1)
        unresolved = False
        for _, play in plays[td & ~conversion & ~negated].iterrows():
            kinds = [("scoring", "td_player_id", "td_player_name", "td_team")]
            if play.pass_touchdown == 1:
                kinds.append(("throwing", "passer_player_id", "passer_player_name", "posteam"))
            for kind, id_col, name_col, team_col in kinds:
                pid = play[id_col]
                if pd.isna(pid) or not str(pid).strip():
                    unresolved = True
                    continue
                credits.append(
                    dict(
                        game_id=gid,
                        play_id=int(play.play_id),
                        player_id=str(pid),
                        kind=kind,
                        player_name=play.get(name_col),
                        team=play.get(team_col),
                    )
                )
        ending = plays.desc.fillna("").str.contains(r"\bEND (?:OF )?GAME\b", case=False)
        if "game_end" in plays:
            ending |= plays.game_end.eq(1)
        terminal = plays.iloc[-1]
        home, away = terminal.total_home_score, terminal.total_away_score
        matches = (
            pd.notna(home)
            and pd.notna(away)
            and pd.notna(game.home_score)
            and pd.notna(game.away_score)
            and home == game.home_score
            and away == game.away_score
        )
        reasons = []
        if conflict:
            reasons.append("conflicting duplicate plays")
        if not ending.any():
            reasons.append("no end-of-game marker")
        if not matches:
            reasons.append("terminal scores do not match schedule")
        if unresolved:
            reasons.append("touchdown identities unresolved")
        results.append(
            dict(
                game_id=gid,
                season=int(game.season),
                week=int(game.week),
                complete=int(not reasons),
                reason="; ".join(reasons) or "complete",
                home_score=home,
                away_score=away,
                imported_at=stamp,
            )
        )
    return (
        pd.DataFrame(credits, columns=CREDIT_COLUMNS).drop_duplicates(
            ["game_id", "play_id", "player_id", "kind"]
        ),
        pd.DataFrame(results, columns=RESULT_COLUMNS),
    )


def import_touchdowns(conn: sqlite3.Connection, season: int, raw: pd.DataFrame) -> int:
    games = db.read_df(
        conn, "SELECT * FROM games WHERE season = ? AND game_type = 'REG'", (season,)
    )
    if games.empty:
        raise ValueError(f"no {season} schedule; refresh schedule before importing scoring")
    credits, results = transform_touchdowns(raw, games)
    if results.empty:
        raise ValueError("play-by-play has no matching regular-season games")
    # A successfully parsed season snapshot replaces coverage too: removed games become pending.
    with conn:
        conn.execute(
            "DELETE FROM touchdown_credits WHERE game_id IN "
            "(SELECT game_id FROM game_results WHERE season = ?)",
            (season,),
        )
        conn.execute("DELETE FROM game_results WHERE season = ?", (season,))
        for table, frame in (("touchdown_credits", credits), ("game_results", results)):
            conn.executemany(
                f"INSERT INTO {table} ({','.join(frame.columns)}) "
                f"VALUES ({','.join('?' for _ in frame.columns)})",
                db._rows(frame),
            )
    return len(credits)


def coverage(conn: sqlite3.Connection, season: int) -> pd.DataFrame:
    """Recheck schedule scores, so a schedule correction invalidates old coverage."""
    return db.read_df(
        conn,
        """SELECT g.game_id, g.season, g.week,
        CASE WHEN r.complete = 1 AND r.home_score = g.home_score
             AND r.away_score = g.away_score THEN 1 ELSE 0 END AS complete,
        CASE WHEN r.game_id IS NULL THEN 'scoring feed missing'
             WHEN r.complete = 0 THEN r.reason
             WHEN g.home_score IS NULL OR g.away_score IS NULL
               OR r.home_score != g.home_score OR r.away_score != g.away_score
             THEN 'terminal scores do not match schedule' ELSE 'complete' END AS reason
        FROM games g LEFT JOIN game_results r ON r.game_id = g.game_id
        WHERE g.season = ? AND g.game_type = 'REG'""",
        (season,),
    )


def touchdown_totals(conn: sqlite3.Connection, season: int) -> pd.DataFrame:
    """Shared aggregation; callers must establish coverage before treating absence as zero."""
    return db.read_df(
        conn,
        """SELECT g.game_id, g.week, t.player_id, COUNT(*) AS pool_td
        FROM touchdown_credits t JOIN games g ON g.game_id = t.game_id
        WHERE g.season = ? AND g.game_type = 'REG'
        GROUP BY g.game_id, g.week, t.player_id""",
        (season,),
    )


def pool_history(conn: sqlite3.Connection, season: int) -> pd.DataFrame:
    """Diagnostic stat rows plus eligible scorer-only rows, with nullable complete pool TDs."""
    pw = db.read_df(
        conn, "SELECT * FROM player_weeks WHERE season = ? AND season_type = 'REG'", (season,)
    )
    cov = coverage(conn, season).set_index("game_id")
    totals = touchdown_totals(conn, season)
    # Imported and legacy stats may still be unassociated. Resolve in memory, without a write.
    for i, row in pw[pw.game_id.isna()].iterrows():
        game = db.resolve_game(conn, season, int(row.week), row.team)
        if game:
            pw.at[i, "game_id"] = game["game_id"]
    extra = []
    existing = set(zip(pw.week, pw.player_id, strict=True))
    history = None
    for r in totals.itertuples():
        if (r.week, r.player_id) in existing:
            continue
        if history is None or history[0] != r.week:
            history = (r.week, state.historical_pool(conn, season, r.week).set_index("player_id"))
        if r.player_id not in history[1].index:
            continue
        identity = history[1].loc[r.player_id]
        if identity.position not in config.POSITIONS:
            continue
        game = conn.execute("SELECT * FROM games WHERE game_id = ?", (r.game_id,)).fetchone()
        credit_team = conn.execute(
            "SELECT team FROM touchdown_credits WHERE game_id = ? "
            "AND player_id = ? AND team IS NOT NULL LIMIT 1",
            (r.game_id, r.player_id),
        ).fetchone()
        team = credit_team[0] if credit_team else identity.team
        opponent = game["away_team"] if team == game["home_team"] else game["home_team"]
        extra.append(
            dict(
                season=season,
                week=r.week,
                season_type="REG",
                player_id=r.player_id,
                player_name=identity.player_name,
                position=identity.position,
                team=team,
                opponent=opponent,
                game_id=r.game_id,
                pass_td=0,
                rush_td=0,
                rec_td=0,
                attempts=0,
                carries=0,
                targets=0,
            )
        )
    if extra:
        pw = pd.concat([pw, pd.DataFrame(extra)], ignore_index=True)
    total_map = totals.set_index(["game_id", "player_id"]).pool_td.to_dict()
    pw["scoring_complete"] = pw.game_id.map(cov.complete).fillna(0).astype(bool)
    pw["pool_td"] = [
        total_map.get((r.game_id, r.player_id), 0) if r.scoring_complete else float("nan")
        for r in pw.itertuples()
    ]
    return pw


def estimated_tds(pw: pd.DataFrame) -> pd.Series:
    """Use complete scoring where known; the live model may fall back to legacy offensive TDs."""
    legacy = pw.pass_td + pw.rush_td + pw.rec_td
    return pw.pool_td.fillna(legacy) if "pool_td" in pw else legacy


def complete_weeks(conn: sqlite3.Connection, season: int) -> list[int]:
    cov = coverage(conn, season)
    return [int(w) for w, rows in cov.groupby("week") if rows.complete.all()]


def require_complete(conn: sqlite3.Connection, season: int, weeks: list[int]) -> None:
    if not weeks:
        raise ValueError(
            f"No complete regular-season coverage for {season}; "
            f"run `pool refresh --season {season}` before replay comparisons."
        )
    missing = set(weeks) - set(complete_weeks(conn, season))
    if missing:
        raise ValueError(
            f"Incomplete touchdown coverage for {season} weeks {sorted(missing)}; "
            f"run `pool refresh --season {season}` before replay comparisons."
        )


def pick_results(
    conn: sqlite3.Connection,
    season: int,
    week: int | None = None,
    recompute: bool = False,
    preserve_existing: bool = False,
) -> pd.DataFrame:
    if week is not None:
        state.validate_week(conn, season, week)
    picks = state.picks(conn, season)
    if week is not None:
        picks = picks[picks.week == week].copy()
    cov = coverage(conn, season).set_index("game_id")
    total_map = touchdown_totals(conn, season).set_index(["game_id", "player_id"]).pool_td.to_dict()
    updates, reasons, scores = [], [], []
    for row in picks.itertuples():
        state.validate_week(conn, season, int(row.week))
        stat = conn.execute(
            "SELECT game_id, team FROM player_weeks "
            "WHERE season = ? AND week = ? AND player_id = ?",
            (season, row.week, row.player_id),
        ).fetchone()
        gid = stat["game_id"] if stat else None
        if stat and not gid:
            actual_game = db.resolve_game(conn, season, row.week, stat["team"])
            gid = actual_game["game_id"] if actual_game else None
        gid = gid or row.game_id
        if gid not in cov.index or int(cov.loc[gid, "week"]) != row.week:
            reason = "game unresolved"
        else:
            reason = "" if cov.loc[gid, "complete"] else cov.loc[gid, "reason"]
        value = row.tds
        fresh_value = int(total_map.get((gid, row.player_id), 0))
        if not reason and recompute and not (preserve_existing and pd.notna(value)):
            value = fresh_value
            updates.append((value, gid, season, row.week, row.slot))
        elif not reason and pd.isna(value):
            reason = "not scored; run pool score"
        elif not reason and value != fresh_value:
            reason = "results changed; run pool score"
        reasons.append(reason)
        scores.append(value)
    if recompute:
        with conn:
            conn.executemany(
                "UPDATE my_picks SET tds = ?, game_id = ? "
                "WHERE season = ? AND week = ? AND slot = ?",
                updates,
            )
    picks["tds"] = scores
    picks["pending_reason"] = reasons
    return picks
