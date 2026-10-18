"""Task 1 in docs/DATA_PREPARATION.md: are the household columns related?

Uses only the 200 build households. Logs itself in runs.db: the expectation is read
from docs/DATA_PREPARATION.md when the run starts, and the outputs are saved when it ends.
The status and conclusion are added afterwards with `python run_log.py finish`.

Run from the project folder: python -m experiments.prepare_households
"""

import csv
import io
from collections import Counter
from contextlib import redirect_stdout
from itertools import combinations
from statistics import correlation

from run_log import ROOT, git, record_outputs, start_run


DATA = "data/train_household.csv"
INCOME = {"under_50k": 0, "50k_to_100k": 1, "over_100k": 2}
BUDGET = {"tight": 0, "middle": 1, "flexible": 2}
COLUMNS = {
    "income_band": lambda value: INCOME[value],
    "cleaning_spend_usd": float,
    "budget": lambda value: BUDGET[value],
    "bathrooms": int,
}


def expectation():
    text = (ROOT / "docs" / "DATA_PREPARATION.md").read_text(encoding="utf-8")
    task = text.split("TASK 1:", 1)[1].split("TASK 2:", 1)[0]
    return task.split("EXPECTATION:", 1)[1].strip()


def ranks(values):
    """Rank values from 1, giving tied values their average rank."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in range(start, end + 1):
            result[order[position]] = (start + end) / 2 + 1
        start = end + 1
    return result


def spearman(x, y):
    return correlation(ranks(x), ranks(y))


def cross_tab(rows, first, second):
    counts = Counter((row[first] or "(blank)", row[second] or "(blank)") for row in rows)
    columns = sorted({key[1] for key in counts})
    print(f"\n{first} x {second}")
    print(f"  {'':<14}" + "".join(f"{column:>12}" for column in columns))
    for value in sorted({key[0] for key in counts}):
        print(f"  {value:<14}" + "".join(f"{counts[(value, column)]:>12}" for column in columns))


def report(rows):
    print(f"{DATA}: {len(rows)} households")

    print("\nBlank values")
    for column in rows[0]:
        print(f"  {column:<20} {sum(not row[column] for row in rows):>4}")

    print("\nSpearman rank correlation (rows with a blank in either column are left out)")
    for first, second in combinations(COLUMNS, 2):
        pairs = [(COLUMNS[first](row[first]), COLUMNS[second](row[second]))
                 for row in rows if row[first] and row[second]]
        value = spearman([pair[0] for pair in pairs], [pair[1] for pair in pairs])
        print(f"  {first} x {second:<22} {value:>6.2f}   (n = {len(pairs)})")

    cross_tab(rows, "income_band", "budget")
    cross_tab(rows, "usual_brand", "budget")

    sentences = Counter()
    for row in rows:
        for sentence in row["note"].split(". "):
            if sentence.strip():
                sentences[sentence.strip().rstrip(".") + "."] += 1
    print(f"\nNote sentences ({sum(not row['note'] for row in rows)} households have a blank note)")
    for sentence, count in sentences.most_common():
        print(f"  {count:>4}  {sentence}")


def main():
    if git("status", "--porcelain", "--", ".", ":!runs.db"):
        print("Warning: uncommitted changes. The run will be logged with uncommitted_changes = 1.\n")
    run_id = start_run(
        name="Task 1: are household columns related?",
        expectation=expectation(),
        inputs={
            "command": "python -m experiments.prepare_households",
            "households": "200 build households (those in visits.csv)",
            "correlation": "Spearman on ranks, tied values get their average rank",
            "income_order": INCOME,
            "budget_order": BUDGET,
            "blanks": "left out pair by pair",
            "notes": "split into sentences on '. '",
        },
        data_files=[DATA],
    )
    with open(ROOT / DATA, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        report(rows)
    print(buffer.getvalue())
    record_outputs(run_id, buffer.getvalue())
    print(f"Logged as run {run_id}. Finish it with: python run_log.py finish {run_id} <status> \"<conclusion>\"")


if __name__ == "__main__":
    main()
