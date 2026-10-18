"""Add the two runs that happened before run_log.py existed. Safe to run twice.

The data files have not changed since the initial commit (24924b8), so their
current hashes are the versions those runs used.
"""

import subprocess

from run_log import ROOT, add_backfilled_run, connect


BASELINE_OUTPUT = """Build visits: 1200
Top choice matched: 29.0%
Entrant chosen: 21.1%
Entrant expected: 15.9%
Check visits: 300
Top choice matched: 28.0%
Entrant chosen: 18.3%
Entrant expected: 15.4%"""


def section(text, start, end=None):
    """Return the lines after the heading `start`, up to the heading `end`."""
    body = text.split(start, 1)[1]
    return (body.split(end, 1)[0] if end else body).strip()


def main():
    with connect() as connection:
        if connection.execute("SELECT COUNT(*) FROM runs WHERE backfilled = 1").fetchone()[0]:
            print("Backfilled runs already exist. Nothing added.")
            return

    add_backfilled_run(
        name="Starter baseline",
        expectation="Not recorded before the run.",
        inputs={"command": "python run_baseline.py"},
        data_files=["data/households.csv", "data/visits.csv", "data/check_visits.csv"],
        model_version="24924b8",
        outputs=BASELINE_OUTPUT,
        status="success",
        conclusion=None,
        note="Output copied from the working session. Run on 2026-10-17 on the unchanged starter code; exact time"
        " unknown. Check-set results were seen in this run.",
    )

    expectation_file = subprocess.run(["git", "show", "abd7d7e:EXPECTATION.md"], cwd=ROOT,
                                      capture_output=True, text=True, check=True).stdout
    result_file = subprocess.run(["git", "show", "4fe17a8:EXPECTATION.md"], cwd=ROOT,
                                 capture_output=True, text=True, check=True).stdout
    add_backfilled_run(
        name="Test 1: label independence with repeat-weighted P(A)",
        expectation=section(expectation_file, "EXPECTATION:"),
        inputs={
            "command": "python label_probabilities.py",
            "definition": "For label(s) A, each household adds its entrant purchases from its first visit where A was"
            " shown and the entrant was bought, onward. P(A) = sum / 1200 rows.",
            "blank_message": "kept as its own value, (blank)",
        },
        data_files=["data/visits.csv"],
        model_version="3388664",
        outputs=(ROOT / "result_test1.txt").read_text(encoding="utf-8"),
        status="failed",
        conclusion=section(result_file, "RESULT & CONCLUSION :"),
        note="Expectation copied from commit abd7d7e (02:32:50), conclusion from commit 4fe17a8. The run happened"
        " between abd7d7e and 3388664 (02:34:02) on 2026-10-18; exact time unknown.",
    )
    print("Added 2 backfilled runs.")


if __name__ == "__main__":
    main()
