"""Shared scoring for every model, so each one is judged exactly like the last.

Households in visits.csv are split into 5 fixed groups of 40. For each group, a model
is fitted on the other 160 households and scored on the 40 held out, visit by visit in
time order. Before each prediction the visit's own choice and later answers are removed,
so a model cannot see what it is predicting. Earlier visits stay in the history.

A model is a function fit(train_rows, households) that returns a predict function with
the starter's signature: predict(household, offers, visit, history).
"""

import csv
import random
import statistics
from math import log

import baseline
from run_baseline import offers_for, read_rows
from run_log import ROOT


OPTIONS = ("entrant", "standard", "budget", "scented", "none")
FOLDS = 5
SEED = 2026
HIDDEN = ("choice", "said_why", "after_use_note")
SMALLEST = 1e-12


def load():
    """Build-set visits and the 200 build households."""
    with open(ROOT / "data" / "train_household.csv", newline="", encoding="utf-8") as file:
        households = {row["household_id"]: row for row in csv.DictReader(file)}
    return read_rows("visits.csv"), households


def folds(household_ids):
    """The fixed split: shuffle the sorted IDs with SEED and cut them into FOLDS groups."""
    ids = sorted(household_ids)
    random.Random(SEED).shuffle(ids)
    size = len(ids) // FOLDS
    return [ids[index * size:(index + 1) * size] for index in range(FOLDS)]


def visits_by_household(rows):
    households = {}
    for row in rows:
        households.setdefault(row["household_id"], []).append(row)
    for visits in households.values():
        visits.sort(key=lambda row: int(row["visit_number"]))
    return households


def score(predict, held_out, households):
    """Log loss, accuracy and entrant totals over held-out households' visits."""
    loss = correct = entrant_expected = entrant_chosen = count = 0
    for household_id, visits in held_out.items():
        history = []
        for visit in sorted(visits, key=lambda row: int(row["visit_number"])):
            visible = {key: value for key, value in visit.items() if key not in HIDDEN}
            probabilities = predict(households[household_id], offers_for(visit), visible, history)["probabilities"]
            loss -= log(max(probabilities[visit["choice"]], SMALLEST))
            correct += max(probabilities, key=probabilities.get) == visit["choice"]
            entrant_expected += probabilities["entrant"]
            entrant_chosen += visit["choice"] == "entrant"
            count += 1
            history.append(visit)
    return {"visits": count, "log_loss": loss / count, "accuracy": correct / count,
            "entrant_expected": entrant_expected / count, "entrant_chosen": entrant_chosen / count}


def cross_validate(fit, rows, households):
    """One result per fold, always on the same fixed split."""
    by_household = visits_by_household(rows)
    results = []
    for group in folds(by_household):
        held = set(group)
        train_rows = [row for row in rows if row["household_id"] not in held]
        predict = fit(train_rows, households)
        results.append(score(predict, {key: by_household[key] for key in group}, households))
    return results


def fit_starter_baseline(train_rows, households):
    """The starter baseline learns nothing, so fitting just returns it."""
    return baseline.predict


def spread(values):
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values), "min": min(values), "max": max(values)}
