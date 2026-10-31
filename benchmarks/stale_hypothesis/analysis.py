"""Scoring and summary for stale-hypothesis results. Standard library only."""
from collections import defaultdict
import math

# Fixed before any run. Change them only before looking at results.
MIN_CONSTRAINT_ACCURACY = 0.80
GO_LOWER_BOUND = 0.10
STOP_UPPER_BOUND = 0.05


def wilson(successes, total, z=1.96):
    """95% Wilson interval for a proportion."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def classify(scores, current, stale):
    """Highest-scoring candidate: 'current', 'stale' or 'other'. Ties count as other."""
    best = max(scores.values())
    winners = [name for name, score in scores.items() if score == best]
    if winners == [current]:
        return "current"
    if winners == [stale]:
        return "stale"
    return "other"


def rate(rows, key, value):
    if not rows:
        return {"rate": None, "low": 0.0, "high": 1.0, "n": 0}
    hits = sum(1 for row in rows if row[key] == value)
    low, high = wilson(hits, len(rows))
    return {"rate": round(hits / len(rows), 4), "low": round(low, 4), "high": round(high, 4), "n": len(rows)}


def attention_means(rows):
    totals, counts = defaultdict(float), defaultdict(int)
    for row in rows:
        for label, mass in (row.get("attention") or {}).items():
            totals[label] += mass
            counts[label] += 1
    return {label: round(totals[label] / counts[label], 4) for label in sorted(totals)}


def decide(constraint_accuracy, revised_stale):
    if constraint_accuracy < MIN_CONSTRAINT_ACCURACY:
        return "uninterpretable: model too weak on constraints, use a stronger model"
    if revised_stale["low"] >= GO_LOWER_BOUND:
        return "go: stale error is present, test focus zones"
    if revised_stale["high"] < STOP_UPPER_BOUND:
        return "stop: stale error is rare, focus is not the bottleneck"
    return "inconclusive: add episodes or use a harder task"


def summarize(rows):
    by_condition = {c: [r for r in rows if r["condition"] == c] for c in ("revised", "control")}
    revised, control = by_condition["revised"], by_condition["control"]
    constraint = rate(rows, "constraint_correct", True)
    revised_stale = rate(revised, "outcome", "stale")
    groups = {}
    for factor in ("style", "gap", "repeats"):
        for level in sorted({row[factor] for row in rows}, key=str):
            subset = [r for r in revised if r[factor] == level]
            groups[f"{factor}={level}"] = rate(subset, "outcome", "stale")
    return {
        "episodes": len(rows),
        "constraint_accuracy": constraint,
        "revised": {"current": rate(revised, "outcome", "current"), "stale": revised_stale,
                    "other": rate(revised, "outcome", "other")},
        # In control, "stale" means the model picked the name mentioned last.
        "control": {"current": rate(control, "outcome", "current"), "latest_name": rate(control, "outcome", "stale"),
                    "other": rate(control, "outcome", "other")},
        "revised_stale_by_factor": groups,
        "attention_per_token": {"revised": attention_means(revised), "control": attention_means(control)},
        "decision": decide(constraint["rate"], revised_stale),
    }
