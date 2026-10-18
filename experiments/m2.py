"""M2 in docs/EXPECTATION.md: the levers plus the household signals that were reliable in M1.

Each option's score is its starting value plus:
  - usual brand: a shared weight when the option is the household's usual brand
    (a blank usual brand matches no option);
  - spend -> entrant: standardised cleaning spend, on the entrant only (blanks filled with
    the training households' median spend for the same bathroom count; no blank flag);
  - price, in dollars minus one average over all four products' prices in the training
    visits (the budget cleaner's constant 2.99 included), with three weights: the entrant's
    price, the price of the household's usual brand, and a shared weight for the others;
  - message, entrant only: cleaning / low_odor / environment / (blank), no reference;
  - channel, per product: online / retail, no reference.
'none' is the reference with score 0. The household inputs were chosen from M1's
results (run 6) on the same data, so M2's score may be slightly optimistic.

M1r is M2 without the levers (starting values, usual brand, spend -> entrant), scored
on the same folds so that M2 - M1r isolates what the levers add.

Attaches its outputs to a run already registered in runs.db.
Run from the project folder: python -m experiments.m2 7

To save M2 fitted on all 200 build households (models/M2.json), from a finished run
with status success or inconclusive:
    python -m experiments.m2 save 7
"""

import io
import json
import random
import statistics
import sys
from contextlib import redirect_stdout
from itertools import combinations

import numpy as np

from experiments import m0, m1
from experiments.evaluation import (OPTIONS, SEED, cross_validate, fit_starter_baseline, load, spread,
                                    visits_by_household)
from experiments.logit import fit_weights, softmax
from run_baseline import offers_for
from run_log import ROOT, connect, data_version, record_outputs


PRODUCTS = OPTIONS[:4]
MESSAGES = ("cleaning", "low_odor", "environment", "(blank)")
CHANNELS = ("online", "retail")
NAMES = ([f"start: {product}" for product in PRODUCTS] + ["usual brand", "spend -> entrant",
         "price: entrant", "price: usual brand", "price: other products"]
         + [f"message={message} -> entrant" for message in MESSAGES]
         + [f"channel={channel} -> {product}" for product in PRODUCTS for channel in CHANNELS])
HOUSEHOLD_ONLY = NAMES[:6]
PENALTY = 1.0
BOOTSTRAPS = 200
EXAMPLES = ("H001", "H002")
EXAMPLE_OFFER = [
    {"product_id": "entrant", "price": 4.49, "message": "environment"},
    {"product_id": "standard", "price": 3.99},
    {"product_id": "budget", "price": 2.99},
    {"product_id": "scented", "price": 4.29},
]
EXAMPLE_CHANNEL = "online"
MODEL_FILE = ROOT / "models" / "M2.json"
COLUMN = {name: index for index, name in enumerate(NAMES)}


def statistics_from(train_rows, training_households):
    """Spend medians and scaling from training households; the price average from training visits."""
    medians = {}
    for bathrooms in sorted({str(household["bathrooms"]) for household in training_households}):
        values = [float(household["cleaning_spend_usd"]) for household in training_households
                  if str(household["bathrooms"]) == bathrooms and household["cleaning_spend_usd"] not in ("", None)]
        if not values:
            raise ValueError(f"No training household with {bathrooms} bathrooms reported its spend.")
        medians[bathrooms] = statistics.median(values)
    spends = [spend_of(household, medians) for household in training_households]
    scale = float(np.std(spends))
    if scale == 0:
        raise ValueError("Spend does not vary across training households.")
    prices = [offer["price"] for row in train_rows for offer in offers_for(row)]
    return {"medians": medians, "spend_mean": float(np.mean(spends)), "spend_scale": scale,
            "price_average": float(np.mean(prices))}


def spend_of(household, medians):
    if household["cleaning_spend_usd"] not in ("", None):
        return float(household["cleaning_spend_usd"])
    bathrooms = str(household["bathrooms"])
    if bathrooms not in medians:
        raise ValueError(f"spend is blank and no training median exists for {bathrooms} bathrooms.")
    return medians[bathrooms]


def option_features(household, offers, channel, stats):
    """One row per option over all NAMES; 'none' stays 0."""
    prices = {offer["product_id"]: offer for offer in offers}
    missing = [product for product in PRODUCTS if product not in prices]
    if missing:
        raise ValueError(f"The offer has no price for {', '.join(missing)}.")
    message = prices["entrant"].get("message") or "(blank)"
    if message not in MESSAGES:
        raise ValueError(f"entrant message must be one of {', '.join(MESSAGES)}, not {message!r}.")
    if channel not in CHANNELS:
        raise ValueError(f"channel must be one of {', '.join(CHANNELS)}, not {channel!r}.")
    usual = household["usual_brand"]
    spend = (spend_of(household, stats["medians"]) - stats["spend_mean"]) / stats["spend_scale"]

    features = np.zeros((len(OPTIONS), len(NAMES)))
    for index, product in enumerate(PRODUCTS):
        row = features[index]
        row[COLUMN[f"start: {product}"]] = 1.0
        row[COLUMN["usual brand"]] = float(usual == product)
        price = float(prices[product]["price"]) - stats["price_average"]
        if product == "entrant":
            row[COLUMN["spend -> entrant"]] = spend
            row[COLUMN["price: entrant"]] = price
            row[COLUMN[f"message={message} -> entrant"]] = 1.0
        elif product == usual:
            row[COLUMN["price: usual brand"]] = price
        else:
            row[COLUMN["price: other products"]] = price
        row[COLUMN[f"channel={channel} -> {product}"]] = 1.0
    return features


def train(train_rows, households, names=NAMES, penalty=PENALTY):
    training_ids = list(dict.fromkeys(row["household_id"] for row in train_rows))
    stats = statistics_from(train_rows, [households[household_id] for household_id in training_ids])
    columns = [COLUMN[name] for name in names]
    features = np.array([option_features(households[row["household_id"]], offers_for(row), row["channel"], stats)
                         for row in train_rows])[:, :, columns]
    chosen = np.array([OPTIONS.index(row["choice"]) for row in train_rows])
    penalised = np.array([0.0 if name.startswith("start: ") else 1.0 for name in names])
    weights, steps = fit_weights(features, chosen, penalty, penalised)
    return {"names": list(names), "weights": weights.tolist(), "stats": stats, "steps": steps}


def explain(features, weights, names, option):
    if option == "none":
        return "Every product scored below 'none' (the reference, score 0) on this visit."
    index = OPTIONS.index(option)
    contributions = [(features[index, k] * weights[k], name) for k, name in enumerate(names)
                     if not name.startswith("start: ") and features[index, k] != 0]
    top = sorted(contributions, key=lambda item: -abs(item[0]))[:3]
    return f"{option}, largest contributions to its score: " + ", ".join(
        f"{name.split(' -> ')[0]} {value:+.2f}" for value, name in top) + "."


def predictor(model):
    names, weights, stats = model["names"], np.array(model["weights"]), model["stats"]
    columns = [COLUMN[name] for name in names]

    def predict(household, offers, visit, history):
        features = option_features(household, offers, visit["channel"], stats)[:, columns]
        probabilities = dict(zip(OPTIONS, softmax(features @ weights).tolist()))
        best = max(probabilities, key=probabilities.get)
        return {"probabilities": probabilities, "reason": explain(features, weights, names, best)}

    return predict


def fit(train_rows, households):
    return predictor(train(train_rows, households))


def fit_household_only(train_rows, households):
    return predictor(train(train_rows, households, names=HOUSEHOLD_ONLY))


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


def print_comparison(results):
    names = list(results)
    pairs = [("M2", "M1r"), ("M2", "M1"), ("M2", "baseline")]
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


def range_text(values):
    low, high = np.percentile(values, [2.5, 97.5])
    return f"{low:+.2f} to {high:+.2f}{'  *' if low <= 0 <= high else ''}"


def print_weights(model, samples):
    weights = np.array(model["weights"])
    print(f"\nM2 fitted on all 200 households (Newton's method converged in {model['steps']} steps)")
    stats = model["stats"]
    print(f"Price average used for centring: {stats['price_average']:.3f}; spend medians for blanks: {stats['medians']}")
    print(f"95% ranges from {BOOTSTRAPS} household resamples; * = range crosses 0, so not a trustworthy reason")
    print(f"  {'weight':<34} {'value':>7}   95% range")
    for index, name in enumerate(NAMES):
        print(f"  {name:<34} {weights[index]:>+7.2f}   {range_text(samples[:, index])}")

    print("\nLever effects (differences are what categories without a reference can be read by)")
    for name in ("price: entrant", "price: usual brand", "price: other products"):
        index = COLUMN[name]
        print(f"  {name:<34} {weights[index]:>+7.2f} per $1 (odds x{np.exp(weights[index]):.2f})   "
              f"{range_text(samples[:, index])}")
    print("  (starter baseline's hand-set price weight: -0.40 per $1, odds x0.67)")
    for first, second in combinations(MESSAGES, 2):
        a, b = COLUMN[f"message={first} -> entrant"], COLUMN[f"message={second} -> entrant"]
        label = f"message {first} - {second}"
        print(f"  {label:<34} {weights[a] - weights[b]:>+7.2f}   {range_text(samples[:, a] - samples[:, b])}")
    for product in PRODUCTS:
        a, b = COLUMN[f"channel=online -> {product}"], COLUMN[f"channel=retail -> {product}"]
        label = f"online - retail -> {product}"
        print(f"  {label:<34} {weights[a] - weights[b]:>+7.2f}   {range_text(samples[:, a] - samples[:, b])}")


def print_lever_settings(model, rows, households):
    """Mean predicted entrant probability with one lever set for every build visit, next to Test 2's rate."""
    predict = predictor(model)

    def predicted(change):
        total = 0.0
        for row in rows:
            offers, channel = change(offers_for(row), row["channel"])
            total += predict(households[row["household_id"]], offers, {"channel": channel}, [])["probabilities"]["entrant"]
        return total / len(rows)

    def observed(test):
        matching = [row for row in rows if test(row)]
        return sum(row["choice"] == "entrant" for row in matching) / len(matching), len(matching)

    print("\nEntrant share by lever setting: M2's mean prediction with the setting applied to every build visit,")
    print("next to the observed rate on visits that had that setting (Test 2's measure)")
    print(f"  {'setting':<28} {'M2 predicted':>12} {'observed':>9} {'visits':>7}")
    for price in sorted({row["entrant_price"] for row in rows}, key=float):
        def change(offers, channel, price=price):
            return [{**offer, "price": float(price)} if offer["product_id"] == "entrant" else offer for offer in offers], channel
        rate, count = observed(lambda row: row["entrant_price"] == price)
        print(f"  {'entrant price ' + price:<28} {predicted(change):>12.1%} {rate:>9.1%} {count:>7}")
    for message in MESSAGES:
        stored = "" if message == "(blank)" else message
        def change(offers, channel, stored=stored):
            return [{**offer, "message": stored} if offer["product_id"] == "entrant" else offer for offer in offers], channel
        rate, count = observed(lambda row: row["entrant_message"] == stored)
        print(f"  {'message ' + message:<28} {predicted(change):>12.1%} {rate:>9.1%} {count:>7}")
    for channel_value in CHANNELS:
        def change(offers, channel, channel_value=channel_value):
            return offers, channel_value
        rate, count = observed(lambda row: row["channel"] == channel_value)
        print(f"  {'channel ' + channel_value:<28} {predicted(change):>12.1%} {rate:>9.1%} {count:>7}")


def report(rows, households):
    print(f"visits.csv: {len(rows)} visits, {len(households)} households, 5 folds of 40 households")
    print(f"M2: {len(NAMES)} weights; M1r: {len(HOUSEHOLD_ONLY)} weights (M2 without the levers); "
          f"L2 penalty {PENALTY} (starting values not penalised)")
    print("Household inputs were chosen from M1's results (run 6) on the same data, so M2's score may be slightly optimistic.")
    print("The budget cleaner's price never varies (2.99), so no price effect is learned from it.")
    print("Log loss = mean of -ln(probability given to the actual choice). Lower is better.")
    results = {
        "M2": cross_validate(fit, rows, households),
        "M1r": cross_validate(fit_household_only, rows, households),
        "M1": cross_validate(m1.fit, rows, households),
        "M0": cross_validate(m0.fit, rows, households),
        "baseline": cross_validate(fit_starter_baseline, rows, households),
    }
    print_comparison(results)

    model = train(rows, households)
    print_weights(model, bootstrap_weights(rows, households))
    print_lever_settings(model, rows, households)

    predict = predictor(model)
    offer_text = ", ".join(f"{offer['product_id']} {offer['price']:.2f}" for offer in EXAMPLE_OFFER)
    print(f"\nExample households (fitted on them too), fixed offer: {offer_text}, "
          f"entrant message {EXAMPLE_OFFER[0]['message']}, {EXAMPLE_CHANNEL}")
    for household_id in EXAMPLES:
        household = households[household_id]
        result = predict(household, EXAMPLE_OFFER, {"channel": EXAMPLE_CHANNEL}, [])
        summary = ", ".join(f"{key}={household[key] or '(blank)'}" for key in
                            ("budget", "income_band", "cleaning_spend_usd", "bathrooms", "usual_brand"))
        print(f"  {household_id}: {summary}")
        print("    " + "  ".join(f"{option} {p:.1%}" for option, p in
                                 sorted(result["probabilities"].items(), key=lambda item: -item[1])))
        print(f"    reason: {result['reason']}")


def save(run_id):
    rows, households = load()
    model = train(rows, households)
    MODEL_FILE.parent.mkdir(exist_ok=True)
    MODEL_FILE.write_text(json.dumps({
        "model": "M2", "run_id": run_id, "fitted_on": f"all {len(households)} build households in visits.csv",
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
        if run is None or run["status"] not in ("success", "inconclusive") or not run["name"].startswith("M2"):
            print(f"Run {sys.argv[2]} is not a finished M2 run with status success or inconclusive.")
            return
        save(int(sys.argv[2]))
        return
    if len(sys.argv) != 2:
        print(__doc__)
        return
    run_id = int(sys.argv[1])
    run = run_status(run_id)
    if run is None or run["status"] != "started" or not run["name"].startswith("M2"):
        print(f"Run {run_id} is not an open M2 run.")
        return

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        report(*load())
    print(buffer.getvalue())
    record_outputs(run_id, buffer.getvalue())
    print(f"Saved to run {run_id}. Finish it with: python run_log.py finish {run_id} <status> \"<conclusion>\"")


if __name__ == "__main__":
    main()
