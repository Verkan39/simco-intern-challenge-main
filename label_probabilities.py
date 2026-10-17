"""Label probabilities with repeat purchases, for the independence check in FIRST.md.

For label(s) A, each household contributes the number of entrant purchases from its
first visit where A was shown and the entrant was bought, onward (that visit included),
whatever the labels of the later visits. P(A) = sum over households / total rows.
Blank shelf messages are kept as their own value, "(blank)".
"""

import sys
from itertools import combinations, product

from run_baseline import read_rows


LABELS = {"price": "entrant_price", "channel": "channel", "message": "entrant_message"}


def label_values(row):
    values = {name: row[column] for name, column in LABELS.items()}
    if not values["message"]:
        values["message"] = "(blank)"
    return values


def probabilities(rows, sales, names):
    """Plain version: share of entrant sales with each combination of values. Used by chi_square.py."""
    options = [sorted({row[name] for row in rows}) for name in names]
    results = []
    for values in product(*options):
        count = sum(all(sale[name] == value for name, value in zip(names, values)) for sale in sales)
        results.append((values, count, count / len(sales)))
    return results


def households_in_order(rows):
    """Group visit rows by household, each household's visits in visit order."""
    households = {}
    for row in rows:
        households.setdefault(row["household_id"], []).append(row)
    for visits in households.values():
        visits.sort(key=lambda row: int(row["visit_number"]))
    return households


def repeat_probabilities(rows, names):
    """Return (values, count, probability) for every combination of values of the given labels."""
    counts = {}
    for visits in households_in_order(rows).values():
        purchases = [label_values(row) for row in visits if row["choice"] == "entrant"]
        seen = set()
        for index, values in enumerate(purchases):
            key = tuple(values[name] for name in names)
            if key not in seen:
                seen.add(key)
                counts[key] = counts.get(key, 0) + len(purchases) - index

    options = [sorted({label_values(row)[name] for row in rows}) for name in names]
    return [(values, counts.get(values, 0), counts.get(values, 0) / len(rows)) for values in product(*options)]


def main():
    file_name = sys.argv[1] if len(sys.argv) > 1 else "visits.csv"
    rows = read_rows(file_name)
    sales = sum(row["choice"] == "entrant" for row in rows)
    print(f"{file_name}: {len(rows)} rows, {sales} entrant sales")

    for size, title in [(1, "P(A)"), (2, "P(A and B)"), (3, "P(A and B and C)")]:
        print(f"\n{title}")
        for names in combinations(LABELS, size):
            results = repeat_probabilities(rows, names)
            for values, count, probability in results:
                label = " and ".join(f"{name}={value}" for name, value in zip(names, values))
                print(f"  {label:<55} {count:>4}  {probability:.3f}")
            print(f"  {'sum for ' + ' x '.join(names):<55} {sum(r[1] for r in results):>4}  {sum(r[2] for r in results):.3f}")


if __name__ == "__main__":
    main()
