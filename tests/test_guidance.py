"""Printed commands are literal shell tokens and target the selected records."""

import shlex

import pytest
from typer.testing import CliRunner

from pool import config, db, entrants, weekly
from pool.cli import app
from tests.support.season import SEASON, seed_season


@pytest.mark.parametrize("uv_launcher", [False, True])
@pytest.mark.parametrize(
    "value",
    [
        "two words",
        "Ja'Marr Chase",
        'say "hello"',
        "$HOME;`id` & | < > * ? ! \\ [red]",
        "",
    ],
)
def test_command_shell_tokens_round_trip(monkeypatch, uv_launcher, value):
    if uv_launcher:
        monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")
    else:
        monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)
    launcher = ["uv", "run", "pool"] if uv_launcher else ["pool"]
    args = ["record", "--rb", value, "--decision", value]
    assert shlex.split(weekly.command(args, season=2024, db_path=value)) == [
        *launcher,
        *args,
        "--season",
        "2024",
        "--db",
        value,
    ]
    assert shlex.split(weekly.command(args, season=config.DEFAULT_SEASON)) == [*launcher, *args]


@pytest.mark.parametrize("uv_launcher", [False, True])
def test_printed_template_check_uses_selected_records_without_archiving(
    tmp_path, monkeypatch, uv_launcher
):
    if uv_launcher:
        monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")
    else:
        monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)
    database = tmp_path / "my [red] pool's database.db"
    report = tmp_path / 'typed "weekly" report.csv'
    conn = db.connect(database)
    seed_season(conn)
    imported = entrants.import_report(
        conn,
        SEASON,
        1,
        b"week,entrant,QB,RB,FLEX\n1,Fixture Entrant,AAA QB1,AAA RB1,AAA WR1\n",
    )
    assert imported.written, imported.errors
    before = conn.iterdump()
    before = list(before)
    runner = CliRunner()
    written = runner.invoke(
        app,
        [
            "report",
            "template",
            "--week",
            "2",
            "--season",
            str(SEASON),
            "--db",
            str(database),
            "--out",
            str(report),
        ],
    )
    assert written.exit_code == 0, (written.output, written.exception)
    assert "Fixture Entrant" in report.read_text()
    report.write_text(
        report.read_text().replace("Fixture Entrant,,,", "Fixture Entrant,BBB QB1,BBB RB1,BBB WR1")
    )
    (line,) = [line for line in written.output.splitlines() if " report import " in line]
    words = shlex.split(line)
    args = words[words.index("report") :]
    assert args == [
        "report",
        "import",
        str(report),
        "--check",
        "--week",
        "2",
        "--season",
        str(SEASON),
        "--db",
        str(database),
    ]
    checked = runner.invoke(app, args)
    assert checked.exit_code == 0, (checked.output, checked.exception)
    assert "Parsed 2024 week 2: 1 entrants, 3 picks" in checked.output
    assert "Unresolved" not in checked.output
    assert list(conn.iterdump()) == before
    conn.close()
