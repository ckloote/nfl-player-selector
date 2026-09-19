"""A finished synthetic season in a database: four teams, four weeks, and a full prior
season behind it, for the replay, archive and capture tests.
"""

from pool import db, snapshots

PRIOR, SEASON, WEEKS = 2023, 2024, [1, 2, 3, 4]
TEAMS = ["AAA", "BBB", "CCC", "DDD"]
PAIRS = [("AAA", "BBB"), ("CCC", "DDD")]


def players():
    """Two players per position per team, so every week has real alternatives."""
    out = []
    for team in TEAMS:
        for pos in ("QB", "RB", "WR"):
            for depth in (1, 2):
                pid = f"{team}-{pos}{depth}"
                out.append((pid, f"{team} {pos}{depth}", pos, team, depth))
    return out


def seed_season(conn, *, season_tds=None):
    """A four-week, four-team season with a full prior season behind it."""
    people = players()
    games, weeks_rows, rosters = [], [], []
    for season in (PRIOR, SEASON):
        for wk in WEEKS:
            for home, away in PAIRS:
                games.append(
                    (
                        f"g{season}-{wk}-{home}",
                        season,
                        wk,
                        "REG",
                        f"{season}-09-{10 + wk:02d}T13:00",
                        home,
                        away,
                        -3.0,
                        45.0,
                    )
                )
    opponent = {}
    for home, away in PAIRS:
        opponent[home], opponent[away] = away, home

    for pid, name, pos, team, depth in people:
        for wk in WEEKS:
            # Prior season: the starter scores, the backup does not.
            tds = 1 if depth == 1 else 0
            weeks_rows.append(
                (
                    PRIOR,
                    wk,
                    "REG",
                    pid,
                    name,
                    pos,
                    team,
                    opponent[team],
                    tds if pos == "QB" else 0,
                    0,
                    tds if pos != "QB" else 0,
                    30 if pos == "QB" else 0,
                    15 if pos == "RB" else 0,
                    8 if pos == "WR" else 0,
                )
            )
            actual = (season_tds or {}).get((wk, pid), 1 if depth == 1 else 0)
            weeks_rows.append(
                (
                    SEASON,
                    wk,
                    "REG",
                    pid,
                    name,
                    pos,
                    team,
                    opponent[team],
                    actual if pos == "QB" else 0,
                    0,
                    actual if pos != "QB" else 0,
                    30 if pos == "QB" else 0,
                    15 if pos == "RB" else 0,
                    8 if pos == "WR" else 0,
                )
            )
            for s in (PRIOR, SEASON):
                rosters.append((s, wk, pid, name, pos, team, "ACT", pos))

    with conn:
        conn.executemany(
            "INSERT INTO games(game_id, season, week, game_type, kickoff, home_team, away_team,"
            " spread_line, total_line) VALUES (?,?,?,?,?,?,?,?,?)",
            games,
        )
        conn.executemany(
            "INSERT INTO player_weeks(season, week, season_type, player_id, player_name, position,"
            " team, opponent, pass_td, rush_td, rec_td, attempts, carries, targets)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            weeks_rows,
        )
        conn.executemany(
            "INSERT INTO rosters(season, week, player_id, player_name, position, team, status,"
            " depth_chart_position) VALUES (?,?,?,?,?,?,?,?)",
            rosters,
        )
    # This synthetic fixture explicitly supplies complete scoring results. Legacy
    # databases in production must obtain these from the play-by-play importer.
    with conn:
        conn.execute("UPDATE games SET home_score = 21, away_score = 21, kickoff_known = 1")
        db.backfill_game_ids(conn)
        conn.execute(
            "INSERT INTO game_results SELECT game_id, season, week, 1, 'complete', "
            "home_score, away_score, '2026-09-01T00:00:00+00:00' FROM games"
        )
        credits = []
        for i, r in enumerate(conn.execute("SELECT * FROM player_weeks")):
            for j in range(r["pass_td"] + r["rush_td"] + r["rec_td"]):
                credits.append(
                    (
                        r["game_id"],
                        i * 100 + j,
                        r["player_id"],
                        "throwing" if r["position"] == "QB" else "scoring",
                    )
                )
        conn.executemany(
            "INSERT INTO touchdown_credits(game_id, play_id, player_id, kind) VALUES (?, ?, ?, ?)",
            credits,
        )
    return conn


def archive_all(conn, stamp="2024-09-01T00:00:00Z", optional=True):
    for season in (PRIOR, SEASON):
        for feed in snapshots.TABLES:
            if optional or feed in ("schedule", "player_stats", "touchdowns"):
                snapshots.archive(conn, season, feed, observed_at=stamp)
