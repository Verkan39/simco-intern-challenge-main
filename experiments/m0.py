"""M0 in docs/EXPECTATION.md: predict every visit with the training households' choice shares.

M0 ignores every input. It is the floor: a model that knows nothing about the household,
the offer or the history. It is scored with experiments/evaluation.py, next to the
starter baseline on the same held-out households.

Attaches its outputs to a run already registered in runs.db.
Run from the project folder: python -m experiments.m0 5
"""

import io
import sys
from collections import Counter
from contextlib import redirect_stdout

from experiments.evaluation import OPTIONS, cross_validate, fit_starter_baseline, load, spread
from run_log import connect, record_outputs


def fit(train_rows, households):
    counts = Counter(row["choice"] for row in train_rows)
    shares = {option: counts[option] / len(train_rows) for option in OPTIONS}

    def predict(household, offers, visit, history):
        return {"probabilities": dict(shares),
                "reason": "M0 ignores all inputs; these are the choice shares in the training households."}

    return predict


def report(rows, households):
    m0 = cross_validate(fit, rows, households)
    starter = cross_validate(fit_starter_baseline, rows, households)

    print(f"visits.csv: {len(rows)} visits, {len(households)} households, 5 folds of 40 households")
    print("Log loss = mean of -ln(probability given to the actual choice). Lower is better.")
    print(f"\n  {'fold':<6} {'visits':>6} {'M0':>8} {'baseline':>9} {'baseline - M0':>14}")
    for number, (a, b) in enumerate(zip(m0, starter), start=1):
        print(f"  {number:<6} {a['visits']:>6} {a['log_loss']:>8.3f} {b['log_loss']:>9.3f} "
              f"{b['log_loss'] - a['log_loss']:>+14.3f}")

    print("\nAcross folds")
    for name, results in (("M0", m0), ("baseline", starter)):
        s = spread([result["log_loss"] for result in results])
        print(f"  {name:<9} mean {s['mean']:.3f}   sd {s['sd']:.3f}   min {s['min']:.3f}   max {s['max']:.3f}")
    differences = [b["log_loss"] - a["log_loss"] for a, b in zip(m0, starter)]
    s = spread(differences)
    wins = sum(difference < 0 for difference in differences)
    print(f"  baseline - M0: mean {s['mean']:+.3f}   sd {s['sd']:.3f}   min {s['min']:+.3f}   max {s['max']:+.3f}"
          f"   (baseline lower in {wins} of {len(differences)} folds)")

    print("\nSupporting measures (all held-out visits)")
    for name, results in (("M0", m0), ("baseline", starter)):
        total = sum(result["visits"] for result in results)
        def pooled(key):
            return sum(result[key] * result["visits"] for result in results) / total
        print(f"  {name:<9} top choice matched {pooled('accuracy'):.1%}   "
              f"entrant expected {pooled('entrant_expected'):.1%}   entrant chosen {pooled('entrant_chosen'):.1%}")

    shares = fit(rows, households)(None, None, None, None)["probabilities"]
    print("\nM0 fitted on all 200 households")
    for option in OPTIONS:
        print(f"  {option:<9} {shares[option]:.1%}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return
    run_id = int(sys.argv[1])
    with connect() as connection:
        run = connection.execute("SELECT name, status FROM runs WHERE id = ?", (run_id,)).fetchone()
    if run is None or run["status"] != "started" or not run["name"].startswith("M0"):
        print(f"Run {run_id} is not an open M0 run.")
        return

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        report(*load())
    print(buffer.getvalue())
    record_outputs(run_id, buffer.getvalue())
    print(f"Saved to run {run_id}. Finish it with: python run_log.py finish {run_id} <status> \"<conclusion>\"")


if __name__ == "__main__":
    main()
