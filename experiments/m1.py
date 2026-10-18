"""M1 in docs/EXPECTATION.md: household traits in a multinomial logistic regression.

Each option's score is a starting value, plus a shared weight when the option is the
household's usual brand, plus per-option weights on the household's traits. 'none' is
the reference with score 0. Probabilities come from the softmax of the scores.

Traits: budget and income band as categories (one 0/1 column per category, no reference
category); cleaning spend (blanks filled with the median spend of training households
with the same number of bathrooms) and a 'spend was blank' flag; bathrooms. Spend and
bathrooms are standardised and every trait is centred with training-household
statistics only, so a contribution reads as 'compared with an average household'.

Weights maximise the likelihood of the training choices minus an L2 penalty
(PENALTY / 2 x sum of squared weights; starting values are not penalised), found with
Newton's method. Anything the plan does not cover raises an error instead of guessing.

Attaches its outputs to a run already registered in runs.db.
Run from the project folder: python -m experiments.m1 6

To save M1 fitted on all 200 build households (models/M1.json), from a finished run
with status success or inconclusive:
    python -m experiments.m1 save 6
"""

import io
import json
import random
import statistics
import sys
from contextlib import redirect_stdout

import numpy as np

from experiments import m0
from experiments.evaluation import (OPTIONS, SEED, cross_validate, fit_starter_baseline, load, spread,
                                    visits_by_household)
from run_log import ROOT, connect, data_version, record_outputs


PRODUCTS = OPTIONS[:4]
BUDGETS = ("tight", "middle", "flexible")
INCOMES = ("under_50k", "50k_to_100k", "over_100k")
TRAITS = ([f"budget={value}" for value in BUDGETS] + [f"income={value}" for value in INCOMES]
          + ["spend", "spend was blank", "bathrooms"])
STANDARDISED = ("spend", "bathrooms")
NAMES = ([f"start: {product}" for product in PRODUCTS] + ["usual brand"]
         + [f"{trait} -> {product}" for product in PRODUCTS for trait in TRAITS])
PENALISED = np.array([0.0] * len(PRODUCTS) + [1.0] * (len(NAMES) - len(PRODUCTS)))
PENALTY = 1.0
TOLERANCE = 1e-8
MAX_STEPS = 100
BOOTSTRAPS = 200
EXAMPLES = ("H001", "H002")
MODEL_FILE = ROOT / "models" / "M1.json"


def raw_traits(household, medians):
    where = f" for {household['household_id']}" if household.get("household_id") else ""
    if household["budget"] not in BUDGETS:
        raise ValueError(f"budget must be one of {', '.join(BUDGETS)}, not {household['budget']!r}{where}.")
    if household["income_band"] not in INCOMES:
        raise ValueError(f"income_band must be one of {', '.join(INCOMES)}, not {household['income_band']!r}{where}.")
    bathrooms = str(household["bathrooms"])
    if bathrooms not in medians:
        raise ValueError(f"bathrooms must be one seen in training ({', '.join(sorted(medians))}), not {bathrooms}{where}.")
    if household["cleaning_spend_usd"] in ("", None):
        spend, blank = medians[bathrooms], 1.0
    else:
        spend, blank = float(household["cleaning_spend_usd"]), 0.0
    return ([float(household["budget"] == value) for value in BUDGETS]
            + [float(household["income_band"] == value) for value in INCOMES]
            + [spend, blank, float(bathrooms)])


def trait_statistics(training_households):
    """Spend medians by bathroom count, and centring and scaling, from training households only."""
    medians = {}
    for bathrooms in sorted({str(household["bathrooms"]) for household in training_households}):
        values = [float(household["cleaning_spend_usd"]) for household in training_households
                  if str(household["bathrooms"]) == bathrooms and household["cleaning_spend_usd"] not in ("", None)]
        if not values:
            raise ValueError(f"No training household with {bathrooms} bathrooms reported its spend.")
        medians[bathrooms] = statistics.median(values)
    raw = np.array([raw_traits(household, medians) for household in training_households])
    scale = np.ones(len(TRAITS))
    for trait in STANDARDISED:
        index = TRAITS.index(trait)
        scale[index] = raw[:, index].std()
        if scale[index] == 0:
            raise ValueError(f"{trait} does not vary across training households.")
    return {"medians": medians, "mean": raw.mean(axis=0).tolist(), "scale": scale.tolist()}


def traits(household, stats):
    return (np.array(raw_traits(household, stats["medians"])) - stats["mean"]) / stats["scale"]


def option_features(household_traits, usual_brand):
    """One row per option; the score of each option is its row times the weights. 'none' stays 0."""
    features = np.zeros((len(OPTIONS), len(NAMES)))
    for index, product in enumerate(PRODUCTS):
        features[index, index] = 1.0
        features[index, len(PRODUCTS)] = float(usual_brand == product)
        start = len(PRODUCTS) + 1 + index * len(TRAITS)
        features[index, start:start + len(TRAITS)] = household_traits
    return features


def softmax(scores):
    shifted = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def objective(weights, features, chosen, penalty):
    scores = features @ weights
    peak = scores.max(axis=1, keepdims=True)
    log_totals = (peak + np.log(np.exp(scores - peak).sum(axis=1, keepdims=True)))[:, 0]
    loss = -(scores[np.arange(len(chosen)), chosen] - log_totals).sum()
    return loss + penalty / 2 * (PENALISED * weights ** 2).sum()


def gradient_and_hessian(weights, features, chosen, penalty):
    probabilities = softmax(features @ weights)
    expected = np.einsum("nj,njk->nk", probabilities, features)
    gradient = -(features[np.arange(len(chosen)), chosen] - expected).sum(axis=0) + penalty * PENALISED * weights
    hessian = (np.einsum("nj,njk,njl->kl", probabilities, features, features)
               - expected.T @ expected + penalty * np.diag(PENALISED))
    return gradient, hessian


def fit_weights(features, chosen, penalty=PENALTY):
    """Newton's method with step halving. Raises if the gradient does not reach TOLERANCE."""
    weights = np.zeros(features.shape[2])
    for step in range(1, MAX_STEPS + 1):
        gradient, hessian = gradient_and_hessian(weights, features, chosen, penalty)
        if np.abs(gradient).max() < TOLERANCE:
            return weights, step - 1
        direction = np.linalg.solve(hessian, gradient)
        size, current = 1.0, objective(weights, features, chosen, penalty)
        while objective(weights - size * direction, features, chosen, penalty) > current and size > 1e-10:
            size /= 2
        weights = weights - size * direction
    raise RuntimeError(f"Newton's method did not converge in {MAX_STEPS} steps.")


def train(train_rows, households, penalty=PENALTY):
    training_ids = list(dict.fromkeys(row["household_id"] for row in train_rows))
    stats = trait_statistics([households[household_id] for household_id in training_ids])
    rows_traits = {household_id: traits(households[household_id], stats) for household_id in training_ids}
    features = np.array([option_features(rows_traits[row["household_id"]], households[row["household_id"]]["usual_brand"])
                         for row in train_rows])
    chosen = np.array([OPTIONS.index(row["choice"]) for row in train_rows])
    weights, steps = fit_weights(features, chosen, penalty)
    return {"weights": weights.tolist(), "stats": stats, "steps": steps}


def explain(features, weights, option):
    index = OPTIONS.index(option)
    if option == "none":
        return "Every product scored below 'none' (the reference, score 0) for this household."
    contributions = [(features[index, k] * weights[k], NAMES[k]) for k in range(len(PRODUCTS), len(NAMES))
                     if features[index, k] != 0]
    top = sorted(contributions, key=lambda item: -abs(item[0]))[:3]
    parts = [f"{name.split(' -> ')[0]} {value:+.2f}" for value, name in top]
    return f"{option}, compared with an average household: " + ", ".join(parts) + "."


def predictor(model):
    weights, stats = np.array(model["weights"]), model["stats"]

    def predict(household, offers, visit, history):
        features = option_features(traits(household, stats), household["usual_brand"])
        probabilities = dict(zip(OPTIONS, softmax(features @ weights).tolist()))
        best = max(probabilities, key=probabilities.get)
        return {"probabilities": probabilities, "reason": explain(features, weights, best)}

    return predict


def fit(train_rows, households):
    return predictor(train(train_rows, households))


def bootstrap_ranges(rows, households):
    """95% range of each weight over refits on households resampled with replacement."""
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
    return np.percentile(np.array(samples), [2.5, 97.5], axis=0)


def print_comparison(results):
    names = list(results)
    print(f"\n  {'fold':<6} {'visits':>6} " + "".join(f"{name:>10}" for name in names)
          + f" {'M1 - M0':>10} {'M1 - baseline':>14}")
    for fold, rows in enumerate(zip(*results.values()), start=1):
        scores = {name: row["log_loss"] for name, row in zip(names, rows)}
        print(f"  {fold:<6} {rows[0]['visits']:>6} " + "".join(f"{scores[name]:>10.3f}" for name in names)
              + f" {scores['M1'] - scores['M0']:>+10.3f} {scores['M1'] - scores['baseline']:>+14.3f}")

    print("\nAcross folds")
    for name in names:
        s = spread([row["log_loss"] for row in results[name]])
        print(f"  {name:<9} mean {s['mean']:.3f}   sd {s['sd']:.3f}   min {s['min']:.3f}   max {s['max']:.3f}")
    for other in ("M0", "baseline"):
        differences = [a["log_loss"] - b["log_loss"] for a, b in zip(results["M1"], results[other])]
        s = spread(differences)
        wins = sum(difference < 0 for difference in differences)
        print(f"  M1 - {other:<8} mean {s['mean']:+.3f}   sd {s['sd']:.3f}   min {s['min']:+.3f}   "
              f"max {s['max']:+.3f}   (M1 lower in {wins} of {len(differences)} folds)")

    print("\nSupporting measures (all held-out visits)")
    for name in names:
        total = sum(row["visits"] for row in results[name])
        def pooled(key):
            return sum(row[key] * row["visits"] for row in results[name]) / total
        print(f"  {name:<9} top choice matched {pooled('accuracy'):.1%}   "
              f"entrant expected {pooled('entrant_expected'):.1%}   entrant chosen {pooled('entrant_chosen'):.1%}")


def report(rows, households):
    print(f"visits.csv: {len(rows)} visits, {len(households)} households, 5 folds of 40 households")
    print(f"M1: {len(NAMES)} weights, L2 penalty {PENALTY} (starting values not penalised)")
    print("Log loss = mean of -ln(probability given to the actual choice). Lower is better.")
    results = {"M1": cross_validate(fit, rows, households), "M0": cross_validate(m0.fit, rows, households),
               "baseline": cross_validate(fit_starter_baseline, rows, households)}
    print_comparison(results)

    model = train(rows, households)
    low, high = bootstrap_ranges(rows, households)
    weights = model["weights"]
    print(f"\nM1 fitted on all 200 households (Newton's method converged in {model['steps']} steps)")
    print(f"Spend medians used for blanks, by bathrooms: {model['stats']['medians']}")
    print(f"95% ranges from {BOOTSTRAPS} household resamples; * = range crosses 0, so not a trustworthy reason")
    print(f"  {'weight':<36} {'value':>7}   95% range")
    for name, value, a, b in zip(NAMES, weights, low, high):
        print(f"  {name:<36} {value:>+7.2f}   {a:+.2f} to {b:+.2f}{'  *' if a <= 0 <= b else ''}")
    usual = weights[NAMES.index("usual brand")]
    print(f"  Usual brand multiplies an option's odds by {np.exp(usual):.2f} (baseline's hand-set +0.55: x1.73)")

    predict = predictor(model)
    print("\nExample households (the model above was fitted on them too)")
    for household_id in EXAMPLES:
        household = households[household_id]
        result = predict(household, None, None, None)
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
        "model": "M1", "run_id": run_id, "fitted_on": f"all {len(households)} build households in visits.csv",
        "data_version": data_version(["data/visits.csv", "data/train_household.csv"]),
        "penalty": PENALTY, "names": NAMES, "weights": model["weights"], "stats": model["stats"],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {MODEL_FILE.relative_to(ROOT)}")


def run_status(run_id):
    with connect() as connection:
        return connection.execute("SELECT name, status FROM runs WHERE id = ?", (run_id,)).fetchone()


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "save":
        run = run_status(int(sys.argv[2]))
        if run is None or run["status"] not in ("success", "inconclusive") or not run["name"].startswith("M1"):
            print(f"Run {sys.argv[2]} is not a finished M1 run with status success or inconclusive.")
            return
        save(int(sys.argv[2]))
        return
    if len(sys.argv) != 2:
        print(__doc__)
        return
    run_id = int(sys.argv[1])
    run = run_status(run_id)
    if run is None or run["status"] != "started" or not run["name"].startswith("M1"):
        print(f"Run {run_id} is not an open M1 run.")
        return

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        report(*load())
    print(buffer.getvalue())
    record_outputs(run_id, buffer.getvalue())
    print(f"Saved to run {run_id}. Finish it with: python run_log.py finish {run_id} <status> \"<conclusion>\"")


if __name__ == "__main__":
    main()
