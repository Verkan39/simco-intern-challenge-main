"""One-time evaluation on check_visits.csv (50 households never used for fitting).

M3, M2, M0 and the starter baseline are fitted on all 200 build households, then scored
once on the check visits. Each visit sees only its household's earlier check visits as
history; the visit's own choice and later answers are hidden, as in evaluation.py. All
preparation statistics come from the build households. Uncertainty comes from
resampling the 50 check households with replacement.

Attaches its outputs to a run already registered in runs.db.
Run from the project folder: python -m experiments.check_set 9
"""

import csv
import io
import random
import sys
from contextlib import redirect_stdout
from math import log

import numpy as np

import baseline
from experiments import m0, m2, m3
from experiments.evaluation import HIDDEN, OPTIONS, SEED, SMALLEST, load, visits_by_household
from run_baseline import offers_for, read_rows
from run_log import ROOT, connect, record_outputs


RESAMPLES = 1000
MODELS = ("M3", "M2", "M0", "baseline")


def check_data(build_households):
    rows = read_rows("check_visits.csv")
    ids = {row["household_id"] for row in rows}
    overlap = ids & set(build_households)
    if overlap:
        raise ValueError(f"Check households also in the build set: {sorted(overlap)}")
    with open(ROOT / "data" / "households.csv", newline="", encoding="utf-8") as file:
        households = {row["household_id"]: row for row in csv.DictReader(file) if row["household_id"] in ids}
    missing = ids - set(households)
    if missing:
        raise ValueError(f"Check households missing from households.csv: {sorted(missing)}")
    return rows, households


def score_visits(predictors, rows, households):
    """One record per check visit: the actual choice, its levers, and each model's loss and entrant probability."""
    records = []
    for household_id, visits in visits_by_household(rows).items():
        history = []
        for visit in visits:
            visible = {key: value for key, value in visit.items() if key not in HIDDEN}
            record = {"household": household_id, "choice": visit["choice"], "price": visit["entrant_price"],
                      "message": visit["entrant_message"] or "(blank)", "channel": visit["channel"]}
            for name, predict in predictors.items():
                probabilities = predict(households[household_id], offers_for(visit), visible, history)["probabilities"]
                record[f"{name} loss"] = -log(max(probabilities[visit["choice"]], SMALLEST))
                record[f"{name} correct"] = max(probabilities, key=probabilities.get) == visit["choice"]
                record[f"{name} entrant"] = probabilities["entrant"]
            records.append(record)
            history.append(visit)
    return records


def mean_loss(records, name):
    return sum(record[f"{name} loss"] for record in records) / len(records)


def report(build_rows, build_households):
    rows, households = check_data(build_households)
    predictors = {
        "M3": m3.fit(build_rows, build_households),
        "M2": m2.fit(build_rows, build_households),
        "M0": m0.fit(build_rows, build_households),
        "baseline": baseline.predict,
    }
    records = score_visits(predictors, rows, households)
    by_household = {}
    for record in records:
        by_household.setdefault(record["household"], []).append(record)

    print(f"check_visits.csv: {len(records)} visits, {len(by_household)} households (none in the build set)")
    print(f"Models fitted on all {len(build_households)} build households; scored once on the check set.")
    print("Log loss = mean of -ln(probability given to the actual choice). Lower is better.")
    print(f"95% ranges from {RESAMPLES} resamples of the {len(by_household)} check households.\n")

    print(f"  {'model':<9} {'log loss':>8}   {'top choice matched':>18}   {'entrant expected':>16}   {'entrant chosen':>14}")
    chosen = sum(record["choice"] == "entrant" for record in records) / len(records)
    for name in MODELS:
        accuracy = sum(record[f"{name} correct"] for record in records) / len(records)
        expected = sum(record[f"{name} entrant"] for record in records) / len(records)
        print(f"  {name:<9} {mean_loss(records, name):>8.3f}   {accuracy:>18.1%}   {expected:>16.1%}   {chosen:>14.1%}")

    generator = random.Random(SEED)
    ids = sorted(by_household)
    samples = []
    for _ in range(RESAMPLES):
        sample = [record for household_id in generator.choices(ids, k=len(ids)) for record in by_household[household_id]]
        samples.append({name: mean_loss(sample, name) for name in MODELS})
    print("\n  difference          value   95% range         M3 lower in")
    for other in ("M2", "baseline", "M0"):
        values = np.array([sample["M3"] - sample[other] for sample in samples])
        low, high = np.percentile(values, [2.5, 97.5])
        lower_share = (values < 0).mean()
        verdict = "clear" if high < 0 or low > 0 else "within noise"
        print(f"  M3 - {other:<12} {mean_loss(records, 'M3') - mean_loss(records, other):>+7.3f}   "
              f"{low:+.3f} to {high:+.3f}   {lower_share:>5.0%} of resamples   {verdict}")

    print("\nEntrant share on check visits: each model's mean prediction vs what was chosen")
    print(f"  {'setting':<24} {'visits':>6} {'chosen':>7} " + "".join(f"{name:>9}" for name in MODELS))
    groups = ([("entrant price " + price, lambda r, p=price: r["price"] == p)
               for price in sorted({r["price"] for r in records}, key=float)]
              + [("message " + message, lambda r, m=message: r["message"] == m) for message in m2.MESSAGES]
              + [("channel " + channel, lambda r, c=channel: r["channel"] == c) for channel in m2.CHANNELS])
    for label, test in groups:
        matching = [record for record in records if test(record)]
        if not matching:
            continue
        actual = sum(record["choice"] == "entrant" for record in matching) / len(matching)
        print(f"  {label:<24} {len(matching):>6} {actual:>7.1%} "
              + "".join(f"{sum(r[f'{name} entrant'] for r in matching) / len(matching):>9.1%}" for name in MODELS))
    print("  (Small groups are noisy: about 75-150 visits per setting.)")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return
    run_id = int(sys.argv[1])
    with connect() as connection:
        run = connection.execute("SELECT name, status FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None or run["status"] != "started" or not run["name"].startswith("Check set"):
        print(f"Run {run_id} is not an open check-set run.")
        return

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        report(*load())
    print(buffer.getvalue())
    record_outputs(run_id, buffer.getvalue())
    print(f"Saved to run {run_id}. Finish it with: python run_log.py finish {run_id} <status> \"<conclusion>\"")


if __name__ == "__main__":
    main()
