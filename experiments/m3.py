"""M3 in docs/EXPECTATION.md: M2 plus the household's purchase history.

All 21 inputs of M2 stay unchanged (experiments/m2.py). History is built only from the
household's earlier visits, and 'none' visits are ignored when tracking purchases:
  - times chosen before: per product, how many earlier visits bought it (shared weight);
  - most recent purchase: per product, 1 if it was the last product bought, skipping
    'none' visits (shared weight);
  - stocked up: 1 on every product if the immediately previous visit was a product
    purchase (shared weight);
  - first visit: 1 if there are no earlier visits, one weight per product.

The push-pull rule (reward the current purchase by +p, punish the previous one by -n)
equals (p - n) x times chosen + n x most recent purchase. Both weights are learned
freely, not constrained; the output reports which pattern the data supports.

Attaches its outputs to a run already registered in runs.db.
Run from the project folder: python -m experiments.m3 8

To save M3 fitted on all 200 build households (models/M3.json), from a finished run
with status success or inconclusive:
    python -m experiments.m3 save 8
"""

import io
import json
import random
import sys
from contextlib import redirect_stdout

import numpy as np

from experiments import m0, m2
from experiments.evaluation import (OPTIONS, SEED, cross_validate, fit_starter_baseline, load, spread,
                                    visits_by_household)
from experiments.logit import fit_weights, softmax
from run_baseline import offers_for
from run_log import ROOT, connect, data_version, record_outputs


PRODUCTS = OPTIONS[:4]
HISTORY_NAMES = (["times chosen before", "most recent purchase", "stocked up"]
                 + [f"first visit -> {product}" for product in PRODUCTS])
NAMES = m2.NAMES + HISTORY_NAMES
COLUMN = {name: index for index, name in enumerate(NAMES)}
PENALTY = 1.0
BOOTSTRAPS = 200
EXAMPLES = ("H001", "H002")
EXAMPLE_HISTORIES = {"first visit": [], "after one visit that bought the entrant": [{"choice": "entrant"}]}
MODEL_FILE = ROOT / "models" / "M3.json"


def history_features(history):
    """One row per option over HISTORY_NAMES, from earlier visits only; 'none' stays 0."""
    for visit in history:
        if visit["choice"] not in OPTIONS:
            raise ValueError(f"history choice must be one of {', '.join(OPTIONS)}, not {visit['choice']!r}.")
    block = np.zeros((len(OPTIONS), len(HISTORY_NAMES)))
    if not history:
        for index, product in enumerate(PRODUCTS):
            block[index, HISTORY_NAMES.index(f"first visit -> {product}")] = 1.0
        return block
    purchases = [visit["choice"] for visit in history if visit["choice"] != "none"]
    recent = purchases[-1] if purchases else None
    stocked = float(history[-1]["choice"] != "none")
    for index, product in enumerate(PRODUCTS):
        block[index, 0] = purchases.count(product)
        block[index, 1] = float(recent == product)
        block[index, 2] = stocked
    return block


def option_features(household, offers, channel, history, stats):
    return np.concatenate([m2.option_features(household, offers, channel, stats), history_features(history)], axis=1)


def train(train_rows, households, penalty=PENALTY):
    training_ids = list(dict.fromkeys(row["household_id"] for row in train_rows))
    stats = m2.statistics_from(train_rows, [households[household_id] for household_id in training_ids])
    features, chosen = [], []
    for household_id, visits in visits_by_household(train_rows).items():
        for position, row in enumerate(visits):
            features.append(option_features(households[household_id], offers_for(row), row["channel"],
                                            visits[:position], stats))
            chosen.append(OPTIONS.index(row["choice"]))
    penalised = np.array([0.0 if name.startswith("start: ") else 1.0 for name in NAMES])
    weights, steps = fit_weights(np.array(features), np.array(chosen), penalty, penalised)
    return {"names": NAMES, "weights": weights.tolist(), "stats": stats, "steps": steps}


def explain(features, weights, option):
    if option == "none":
        return "Every product scored below 'none' (the reference, score 0) on this visit."
    index = OPTIONS.index(option)
    contributions = [(features[index, k] * weights[k], name) for k, name in enumerate(NAMES)
                     if not name.startswith("start: ") and features[index, k] != 0]
    top = sorted(contributions, key=lambda item: -abs(item[0]))[:3]
    return f"{option}, largest contributions to its score: " + ", ".join(
        f"{name.split(' -> ')[0]} {value:+.2f}" for value, name in top) + "."


def predictor(model):
    weights, stats = np.array(model["weights"]), model["stats"]

    def predict(household, offers, visit, history):
        features = option_features(household, offers, visit["channel"], history, stats)
        probabilities = dict(zip(OPTIONS, softmax(features @ weights).tolist()))
        best = max(probabilities, key=probabilities.get)
        return {"probabilities": probabilities, "reason": explain(features, weights, best)}

    return predict


def fit(train_rows, households):
    return predictor(train(train_rows, households))


def bootstrap_weights(rows, households):
    by_household = visits_by_household(rows)
    ids = sorted(by_household)
    generator = random.Random(SEED)
    samples = []
    for _ in range(BOOTSTRAPS):
        sample_rows, sample_households = [], {}
        for copy, household_id in enumerate(generator.choices(ids, k=len(ids))):
            label = f"{household_id}#{copy}"
            sample_households[label] = households[household_id]
            sample_rows += [{**row, "household_id": label} for row in by_household[household_id]]
        samples.append(train(sample_rows, sample_households)["weights"])
    return np.array(samples)


def interval(values):
    return np.percentile(values, [2.5, 97.5])


def range_text(values):
    low, high = interval(values)
    return f"{low:+.2f} to {high:+.2f}{'  *' if low <= 0 <= high else ''}"


def push_pull_verdict(samples):
    """Which pattern the 95% ranges support, following the table agreed before the run."""
    recent_low, recent_high = interval(samples[:, COLUMN["most recent purchase"]])
    count_low, count_high = interval(samples[:, COLUMN["times chosen before"]])
    if recent_low > 0:
        if count_low > 0:
            return "loyalty builds: both weights positive (supports the push-pull rule, n < p)"
        if count_high < 0:
            return "variety-seeking / wear-out: most recent positive, times chosen negative (n > p)"
        return "recency only: most recent positive, times chosen not distinguishable from 0"
    if recent_high < 0:
        return "alternation: most recent purchase negative"
    return "can't tell: the most recent purchase weight's range crosses 0"


def print_comparison(results):
    names = list(results)
    pairs = [("M3", "M2"), ("M3", "baseline")]
    print(f"\n  {'fold':<5} {'visits':>6} " + "".join(f"{name:>9}" for name in names)
          + "".join(f"{f'{a} - {b}':>14}" for a, b in pairs))
    for fold, rows in enumerate(zip(*results.values()), start=1):
        scores = {name: row["log_loss"] for name, row in zip(names, rows)}
        print(f"  {fold:<5} {rows[0]['visits']:>6} " + "".join(f"{scores[name]:>9.3f}" for name in names)
              + "".join(f"{scores[a] - scores[b]:>+14.3f}" for a, b in pairs))

    print("\nAcross folds")
    for name in names:
        s = spread([row["log_loss"] for row in results[name]])
        print(f"  {name:<9} mean {s['mean']:.3f}   sd {s['sd']:.3f}   min {s['min']:.3f}   max {s['max']:.3f}")
    for a, b in pairs:
        differences = [x["log_loss"] - y["log_loss"] for x, y in zip(results[a], results[b])]
        s = spread(differences)
        wins = sum(difference < 0 for difference in differences)
        print(f"  {a} - {b:<9} mean {s['mean']:+.3f}   sd {s['sd']:.3f}   min {s['min']:+.3f}   "
              f"max {s['max']:+.3f}   ({a} lower in {wins} of {len(differences)} folds)")

    print("\nSupporting measures (all held-out visits)")
    for name in names:
        total = sum(row["visits"] for row in results[name])
        def pooled(key):
            return sum(row[key] * row["visits"] for row in results[name]) / total
        print(f"  {name:<9} top choice matched {pooled('accuracy'):.1%}   "
              f"entrant expected {pooled('entrant_expected'):.1%}   entrant chosen {pooled('entrant_chosen'):.1%}")


def report(rows, households):
    print(f"visits.csv: {len(rows)} visits, {len(households)} households, 5 folds of 40 households")
    print(f"M3: {len(NAMES)} weights (M2's {len(m2.NAMES)} unchanged + {len(HISTORY_NAMES)} history); "
          f"L2 penalty {PENALTY} (starting values not penalised)")
    print("History uses earlier visits only; 'none' visits are ignored when tracking purchases.")
    print("Inherits M2's household inputs, chosen from M1's results on the same data, so scores may be slightly optimistic.")
    print("History predicts choices; it cannot show cause (habit and preference look the same).")
    print("Log loss = mean of -ln(probability given to the actual choice). Lower is better.")
    results = {
        "M3": cross_validate(fit, rows, households),
        "M2": cross_validate(m2.fit, rows, households),
        "M0": cross_validate(m0.fit, rows, households),
        "baseline": cross_validate(fit_starter_baseline, rows, households),
    }
    print_comparison(results)

    model = train(rows, households)
    previous = m2.train(rows, households)
    samples = bootstrap_weights(rows, households)
    weights = np.array(model["weights"])
    print(f"\nM3 fitted on all 200 households (Newton's method converged in {model['steps']} steps)")
    print(f"95% ranges from {BOOTSTRAPS} household resamples; * = range crosses 0, so not a trustworthy reason")
    print(f"  {'weight':<34} {'M3':>7}   {'95% range':<18} {'M2':>7}")
    for index, name in enumerate(NAMES):
        before = f"{previous['weights'][index]:>+7.2f}" if index < len(m2.NAMES) else f"{'':>7}"
        print(f"  {name:<34} {weights[index]:>+7.2f}   {range_text(samples[:, index]):<18} {before}")

    print("\nPush-pull rule: (p - n) x times chosen + n x most recent purchase")
    n = weights[COLUMN["most recent purchase"]]
    count = weights[COLUMN["times chosen before"]]
    print(f"  n = most recent purchase weight  {n:+.2f}   {range_text(samples[:, COLUMN['most recent purchase']])}")
    print(f"  p - n = times chosen weight      {count:+.2f}   {range_text(samples[:, COLUMN['times chosen before']])}")
    print(f"  p = sum of both                  {n + count:+.2f}   "
          f"{range_text(samples[:, COLUMN['most recent purchase']] + samples[:, COLUMN['times chosen before']])}")
    print(f"  Verdict: {push_pull_verdict(samples)}")
    print("  'Times chosen' grows with visit number, so a negative weight could partly reflect households buying less over time.")

    carried = [name for name in HISTORY_NAMES if not (interval(samples[:, COLUMN[name]])[0] <= 0 <= interval(samples[:, COLUMN[name]])[1])]
    print(f"\nCarried from one visit to the next (history inputs with ranges clear of 0): {', '.join(carried) or 'none'}")

    predict = predictor(model)
    offer_text = ", ".join(f"{offer['product_id']} {offer['price']:.2f}" for offer in m2.EXAMPLE_OFFER)
    print(f"\nExample households (fitted on them too), M2's fixed offer: {offer_text}, "
          f"entrant message {m2.EXAMPLE_OFFER[0]['message']}, {m2.EXAMPLE_CHANNEL}")
    for household_id in EXAMPLES:
        household = households[household_id]
        summary = ", ".join(f"{key}={household[key] or '(blank)'}" for key in
                            ("budget", "income_band", "cleaning_spend_usd", "bathrooms", "usual_brand"))
        print(f"  {household_id}: {summary}")
        for label, history in EXAMPLE_HISTORIES.items():
            result = predict(household, m2.EXAMPLE_OFFER, {"channel": m2.EXAMPLE_CHANNEL}, history)
            print(f"    {label}: " + "  ".join(f"{option} {p:.1%}" for option, p in
                                              sorted(result["probabilities"].items(), key=lambda item: -item[1])))
            print(f"      reason: {result['reason']}")


def save(run_id):
    rows, households = load()
    model = train(rows, households)
    MODEL_FILE.parent.mkdir(exist_ok=True)
    MODEL_FILE.write_text(json.dumps({
        "model": "M3", "run_id": run_id, "fitted_on": f"all {len(households)} build households in visits.csv",
        "data_version": data_version(["data/visits.csv", "data/train_household.csv"]),
        "penalty": PENALTY, "names": model["names"], "weights": model["weights"], "stats": model["stats"],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {MODEL_FILE.relative_to(ROOT)}")


def run_status(run_id):
    with connect() as connection:
        return connection.execute("SELECT name, status FROM runs WHERE id = ?", (run_id,)).fetchone()


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "save":
        run = run_status(int(sys.argv[2]))
        if run is None or run["status"] not in ("success", "inconclusive") or not run["name"].startswith("M3"):
            print(f"Run {sys.argv[2]} is not a finished M3 run with status success or inconclusive.")
            return
        save(int(sys.argv[2]))
        return
    if len(sys.argv) != 2:
        print(__doc__)
        return
    run_id = int(sys.argv[1])
    run = run_status(run_id)
    if run is None or run["status"] != "started" or not run["name"].startswith("M3"):
        print(f"Run {run_id} is not an open M3 run.")
        return

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        report(*load())
    print(buffer.getvalue())
    record_outputs(run_id, buffer.getvalue())
    print(f"Saved to run {run_id}. Finish it with: python run_log.py finish {run_id} <status> \"<conclusion>\"")


if __name__ == "__main__":
    main()
