"""Paired comparisons for the report from saved features (no model inference).

  python report/intervals.py [ROOT] [--train pinned|1015|PATH] > report/intervals.json

ROOT is the folder holding runs/ (default: the lane folder). --train (default: $AIMO_TRAIN, else pinned)
chooses the label file every refit reads; a non-default file adds train_file and train_sha256 to the output.

Recomputes each method's problem-holdout predictions with the code that produced the reported counts
(fit_probe.py, sub/calibration.py, fit_trace_probe.py); the hidden-activation probe
is read from its saved predictions (runs/activation-r3-yuris/probe.json), checks that the counts reproduce and that
every method is scored on the same labelled (model, problem) rows, then reports for each method
against the majority baseline: correct counts, discordant rows (method right and majority wrong, and
the reverse), and a 95% interval for the difference in correct rows from 10,000 bootstrap resamples
of whole problems (all of a problem's rows move together).
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import trainset  # noqa: E402


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, HERE / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def keyed(df, pred):
    return {(m, p): (bool(y), bool(q)) for m, p, y, q in zip(df.model_id, df.problem, df.y, pred)}


def activation_keyed(predictions, problems, column):
    """Saved hidden-activation predictions (keyed by the sha256 of the problem text) as
    {(model, problem text): (truth, prediction)}; `column` is "probe" or "majority"."""
    text = {hashlib.sha256(p.encode()).hexdigest(): p for p in problems}
    return {(r["model_id"], text[r["problem_sha256"]]): (bool(r["truth"]), bool(r[column])) for r in predictions}


def main(root=None, train=None):
    root = Path(root) if root else HERE
    _, tname, tsha = trainset.resolve(train, HERE)
    print(f"train file: {tname} {tsha}", file=sys.stderr)
    fp = module("fit_probe.py", "fit_probe")
    cal = module("sub/calibration.py", "calibration")
    tp = module("fit_trace_probe.py", "fit_trace_probe")
    feats = root / "runs/feat/features.jsonl"
    methods, expected = {}, {}

    df = fp.table(feats, train)
    methods["majority"] = keyed(df, fp.oof(df, fp.majority))
    for arm, cols in fp.ARMS.items():
        methods[f"prompt:{arm}"] = keyed(df, fp.oof(df, fp.logistic(cols)))
    gate = json.loads((root / "runs/probe-gate.json").read_text())["arms"]
    expected.update({"majority": gate["majority"]["correct"],
                     **{f"prompt:{a}": gate[a]["correct"] for a in fp.ARMS}})

    cdf = cal.load(feats, train)
    sel, _, _ = cal.outer(cdf, "problem", False)
    methods["calibration (nested)"] = keyed(cdf, sel)
    expected["calibration (nested)"] = json.loads((root / "runs/calibration-nested/report.json").read_text())[
        "problem_holdout"]["selected"]["correct"]

    tdf = tp.table(root / "runs/trace-pilot/rows.jsonl", train=train)
    methods["trace sensitivity"] = keyed(tdf, tp.logo(tdf, tp.nested))
    methods["trace majority"] = keyed(tdf, tp.logo(tdf, tp.majority))
    pr = json.loads((root / "runs/trace-pilot/probe.json").read_text())
    expected["trace sensitivity"], expected["trace majority"] = pr["probe"]["correct"], pr["majority"]["correct"]

    act = json.loads((root / "runs/activation-r3-yuris/probe.json").read_text())
    problems = {p for _, p in methods["majority"]}
    methods["hidden-activation probe"] = activation_keyed(act["predictions"], problems, "probe")
    methods["activation majority"] = activation_keyed(act["predictions"], problems, "majority")
    expected["hidden-activation probe"], expected["activation majority"] = act["probe"]["correct"], act["majority"]["correct"]

    rows = sorted(methods["majority"])
    out = {"rows": len(rows), "problems": len({p for _, p in rows}), "same_rows": {}, "reproduced": {}, "vs_majority": {}}
    if tname != "pinned":
        out.update(trainset.provenance(tname, tsha))
    rng = np.random.default_rng(20261013)
    problems = sorted({p for _, p in rows})
    by_problem = {p: [r for r in rows if r[1] == p] for p in problems}
    draws = [rng.choice(len(problems), len(problems)) for _ in range(10_000)]
    base = methods["majority"]
    for name, m in methods.items():
        out["same_rows"][name] = sorted(m) == rows
        got = sum(y == q for y, q in m.values())
        out["reproduced"][name] = {"correct": got, "reported": expected.get(name), "match": got == expected.get(name)}
        if name == "majority" or sorted(m) != rows:
            continue
        diff = {p: sum((m[r][1] == m[r][0]) - (base[r][1] == base[r][0]) for r in by_problem[p]) for p in problems}
        d = np.array([diff[p] for p in problems])
        boot = np.array([d[idx].sum() for idx in draws])
        out["vs_majority"][name] = {
            "correct": got, "majority": sum(y == q for y, q in base.values()), "difference": int(d.sum()),
            "method_right_majority_wrong": sum(m[r][1] == m[r][0] and base[r][1] != base[r][0] for r in rows),
            "majority_right_method_wrong": sum(m[r][1] != m[r][0] and base[r][1] == base[r][0] for r in rows),
            "interval95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    choice, args = trainset.strip_flag(sys.argv[1:])
    main(args[0] if args else None, choice)
