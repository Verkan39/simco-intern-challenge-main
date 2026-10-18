"""Run log: one SQLite row per experiment.

Start a run before the experiment (expectation, inputs, versions), then finish it
after (outputs, status, conclusion). Rows are never deleted.

    python run_log.py list
    python run_log.py show 3
"""

import hashlib
import json
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).parent
DATABASE = ROOT / "runs.db"
STATUSES = ("success", "failed", "inconclusive")

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    expectation TEXT NOT NULL,
    inputs TEXT NOT NULL,
    model_version TEXT,
    uncommitted_changes INTEGER,
    data_version TEXT NOT NULL,
    outputs TEXT,
    status TEXT NOT NULL DEFAULT 'started',
    conclusion TEXT,
    backfilled INTEGER NOT NULL DEFAULT 0,
    note TEXT
)
"""


def connect():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    connection.execute(SCHEMA)
    return connection


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def data_version(data_files):
    """SHA-256 of each data file, keyed by its path relative to the project."""
    return {str(Path(name)): hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in data_files}


def start_run(name, expectation, inputs, data_files):
    """Record a run before it happens and return its ID."""
    if not expectation.strip():
        raise ValueError("Write the expectation before starting the run.")
    changes = git("status", "--porcelain", "--", ".", ":!runs.db")
    with connect() as connection:
        cursor = connection.execute(
            "INSERT INTO runs (name, started_at, expectation, inputs, model_version, uncommitted_changes, data_version)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, now(), expectation, json.dumps(inputs), git("rev-parse", "--short", "HEAD"),
             None if changes is None else int(bool(changes)), json.dumps(data_version(data_files))),
        )
        return cursor.lastrowid


def finish_run(run_id, outputs, status, conclusion=""):
    """Record what happened. A run can only be finished once."""
    if status not in STATUSES:
        raise ValueError(f"Status must be one of {STATUSES}.")
    with connect() as connection:
        row = connection.execute("SELECT status FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise ValueError(f"No run with ID {run_id}.")
        if row["status"] != "started":
            raise ValueError(f"Run {run_id} is already finished.")
        connection.execute(
            "UPDATE runs SET finished_at = ?, outputs = ?, status = ?, conclusion = ? WHERE id = ?",
            (now(), outputs, status, conclusion, run_id),
        )


def add_backfilled_run(name, expectation, inputs, data_files, model_version, outputs, status, conclusion, note):
    """Record a run that happened before this log existed. Its times are unknown, so they stay empty."""
    if status not in STATUSES:
        raise ValueError(f"Status must be one of {STATUSES}.")
    with connect() as connection:
        cursor = connection.execute(
            "INSERT INTO runs (name, expectation, inputs, model_version, data_version, outputs, status, conclusion,"
            " backfilled, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
            (name, expectation, json.dumps(inputs), model_version, json.dumps(data_version(data_files)),
             outputs, status, conclusion, note),
        )
        return cursor.lastrowid


def show(run_id):
    with connect() as connection:
        row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        print(f"No run with ID {run_id}.")
        return
    for key in row.keys():
        value = row[key]
        if key in ("inputs", "data_version") and value:
            value = json.dumps(json.loads(value), indent=2)
        print(f"--- {key}\n{value}")


def list_runs():
    with connect() as connection:
        rows = connection.execute("SELECT id, started_at, status, backfilled, name FROM runs ORDER BY id").fetchall()
    for row in rows:
        started = "(backfilled)" if row["backfilled"] else row["started_at"]
        print(f"{row['id']:>3}  {started:<25}  {row['status']:<12}  {row['name']}")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "show":
        show(int(sys.argv[2]))
    elif len(sys.argv) == 2 and sys.argv[1] == "list":
        list_runs()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
