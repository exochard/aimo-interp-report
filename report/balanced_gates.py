"""Prevalence and balanced accuracy of each tested signal, from saved run files (CPU, no models).

  python report/balanced_gates.py RUNS_DIR

The gates compared raw correct counts with the majority class (24 of 34 labelled rows are robust).
Balanced accuracy is the mean of the true-positive and true-negative rates; it is 0.5 for any constant
answer, so it does not depend on the share of robust rows. Only the hidden-activation run saved
per-row predictions, so its balanced accuracy is recomputed here from them; the other values are read
from the reports those runs wrote. The forced-answer test saved no held-out predictions.
"""
import json
import sys
from pathlib import Path


def balanced(truth, pred):
    pos = [p for t, p in zip(truth, pred) if t]
    neg = [p for t, p in zip(truth, pred) if not t]
    return (sum(pos) / len(pos) + sum(1 - p for p in neg) / len(neg)) / 2


def main(runs):
    runs = Path(runs)
    load = lambda p: json.loads((runs / p).read_text())
    out = []
    gate = load("probe-gate.json")
    out.append(("prevalence (robust share of 34 rows)", gate["true_rate"], None))
    for arm in ("majority", "capability", "sensitivity", "combined"):
        a = gate["arms"][arm]
        out.append((f"prompt-only: {arm}", a["correct"], a["balanced"]))
    cal = load("calibration-nested/report.json")["problem_holdout"]["selected"]
    out.append(("calibration (nested)", cal["correct"], cal["balanced"]))
    tr = load("trace-pilot/probe.json")["probe"]
    out.append(("trace sensitivity", tr["correct"], tr["balanced"]))
    act = load("activation-r3-yuris/probe.json")
    rows = act["predictions"]
    truth = [r["truth"] for r in rows]
    bal = balanced(truth, [r["probe"] for r in rows])
    assert abs(bal - act["probe"]["balanced_accuracy"]) < 1e-9, "saved balanced accuracy differs"
    out.append(("hidden-activation probe", sum(r["truth"] == r["probe"] for r in rows), bal))
    for name, correct, b in out:
        print(f"{name:40s} {correct if isinstance(correct, int) else round(correct, 3)!s:>6}  "
              f"{'' if b is None else round(b, 3)}")


if __name__ == "__main__":
    main(sys.argv[1])
