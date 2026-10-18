"""Test 2 in docs/EXPECTATION.md: entrant purchase rate by price, message and channel.

One lever at a time, on the build set. The 95% ranges come from resampling whole
households with replacement, so each household's six visits stay together.
A difference between two values counts as clear when its range excludes 0.

Attaches its outputs to a run already registered in runs.db.
Run from the project folder: python -m experiments.lever_rates 4
"""

import io
import random
import sys
from contextlib import redirect_stdout
from itertools import combinations

from run_baseline import read_rows
from run_log import connect, record_outputs


LEVERS = {"price": "entrant_price", "message": "entrant_message", "channel": "channel"}
RESAMPLES = 1000
SEED = 2026


def lever_value(row, lever):
    return row[LEVERS[lever]] or "(blank)"


def household_counts(rows):
    """For each household: {lever: {value: [visits, entrant purchases]}}."""
    households = {}
    for row in rows:
        counts = households.setdefault(row["household_id"], {lever: {} for lever in LEVERS})
        for lever in LEVERS:
            visits_and_purchases = counts[lever].setdefault(lever_value(row, lever), [0, 0])
            visits_and_purchases[0] += 1
            visits_and_purchases[1] += row["choice"] == "entrant"
    return households


def rates(sample, lever):
    """Purchase rate per value of one lever, pooled over the households in the sample."""
    totals = {}
    for counts in sample:
        for value, (visits, purchases) in counts[lever].items():
            total = totals.setdefault(value, [0, 0])
            total[0] += visits
            total[1] += purchases
    return {value: purchases / visits for value, (visits, purchases) in totals.items()}


def middle_95(values):
    values = sorted(values)
    return values[round(0.025 * (len(values) - 1))], values[round(0.975 * (len(values) - 1))]


def report(rows):
    households = list(household_counts(rows).values())
    generator = random.Random(SEED)
    resamples = [generator.choices(households, k=len(households)) for _ in range(RESAMPLES)]

    print(f"visits.csv: {len(rows)} visits, {len(households)} households, "
          f"{sum(row['choice'] == 'entrant' for row in rows)} entrant purchases")
    print(f"{RESAMPLES} household resamples, seed {SEED}. Blank messages shown as (blank).")
    print("A difference is clear when its 95% range excludes 0.")

    for lever in LEVERS:
        observed = rates(households, lever)
        resampled = [rates(sample, lever) for sample in resamples]
        totals = {}
        for row in rows:
            total = totals.setdefault(lever_value(row, lever), [0, 0])
            total[0] += 1
            total[1] += row["choice"] == "entrant"

        print(f"\n{lever}")
        print(f"  {'value':<14} {'visits':>6} {'bought':>6} {'rate':>7}   95% range")
        for value in sorted(observed):
            low, high = middle_95([sample[value] for sample in resampled if value in sample])
            visits, purchases = totals[value]
            print(f"  {value:<14} {visits:>6} {purchases:>6} {observed[value]:>7.1%}   {low:.1%} to {high:.1%}")

        print(f"  {'difference':<31} {'rate':>7}   95% range         verdict")
        for first, second in combinations(sorted(observed), 2):
            differences = [sample[first] - sample[second] for sample in resampled if first in sample and second in sample]
            low, high = middle_95(differences)
            verdict = "clear" if low > 0 or high < 0 else "within noise"
            label = f"{first} - {second}"
            print(f"  {label:<31} {observed[first] - observed[second]:>+7.1%}   "
                  f"{low:+.1%} to {high:+.1%}   {verdict}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return
    run_id = int(sys.argv[1])
    with connect() as connection:
        run = connection.execute("SELECT name, status FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None or run["status"] != "started" or not run["name"].startswith("Test 2"):
        print(f"Run {run_id} is not an open Test 2 run.")
        return

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        report(read_rows("visits.csv"))
    print(buffer.getvalue())
    record_outputs(run_id, buffer.getvalue())
    print(f"Saved to run {run_id}. Finish it with: python run_log.py finish {run_id} <status> \"<conclusion>\"")


if __name__ == "__main__":
    main()
