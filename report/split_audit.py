"""Controlled split-leakage audit from saved prompt features (no model inference).

  python report/split_audit.py [ROOT] [--train pinned|1015|PATH] [--perms N] [--out FILE] > report/split-audit.json

ROOT is the folder holding runs/ (default: the lane folder). Compares two ways of holding data out
on the same labelled (model, problem) rows, with identical features and preprocessing fitted on the
training rows only:

  A  leave one (model, problem) row out; other models' rows for that problem stay in training
  B  leave the whole problem out (fit_probe.oof, the split the report uses)

Arms, fixed before any result: training-fold majority; logistic on the four problem-text length
features; the prompt-only combined probe (fit_probe.ARMS["combined"]); the nested calibration fitter
when sub/calibration.py can be called unchanged. The leakage control predicts a row from another
training row with the same normalized problem text (majority over them, else the training-fold
majority); it is a diagnostic, never an entry.

Primary endpoint: balanced accuracy of A minus B per arm, with a 95% interval from 10,000 bootstrap
draws (seed 0) over whole problems; both splits' fixed out-of-fold predictions are rescored on the
same draw. Sensitivity: the Lesson 9 cross-model label-agreement count under the pooled label shuffle
and under a shuffle within each model (model prevalence preserved).
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "report"))
import fit_probe as fp  # noqa: E402
import intervals  # noqa: E402
import length_baseline  # noqa: E402
import lesson9_counts  # noqa: E402
import lesson9_shuffle  # noqa: E402
import trainset  # noqa: E402

LENGTH = ["log_chars", "words", "digits", "dollars"]
DRAWS, SEED = 10_000, 0


def norm(text):
    return lesson9_counts._norm(text)


def with_length(df):
    """Adds the four length features, computed from the problem text as length_baseline.features does."""
    return pd.concat([df.reset_index(drop=True), length_baseline.features(df.problem.reset_index(drop=True))], axis=1)


def oof_row(df, predict):
    """Split A: each row is predicted by a model fitted on every other row."""
    out = np.zeros(len(df), dtype=bool)
    for i in range(len(df)):
        test = np.zeros(len(df), dtype=bool)
        test[i] = True
        out[i] = predict(df[~test], df[test])[0]
    return out


def leak_control(train, test):
    """Majority label of training rows with the same normalized problem text, else the fold majority."""
    seen = {}
    for p, y in zip(train.problem, train.y):
        seen.setdefault(norm(p), []).append(bool(y))
    fallback = train.y.mean() >= 0.5
    return np.array([np.mean(seen[norm(p)]) >= 0.5 if norm(p) in seen else fallback for p in test.problem])


def exposed(df):
    """Rows whose problem also belongs to another row, which split A leaves in training."""
    counts = df.problem.map(norm).value_counts()
    return int((df.problem.map(norm).map(counts) > 1).sum())


def arm_predictors():
    return {"majority": fp.majority,
            "length": fp.logistic(LENGTH),
            "prompt:combined": fp.logistic(fp.ARMS["combined"]),
            "leak_control": leak_control}


def calibration_preds(feats, train, df):
    """Nested calibration under both splits, calling sub/calibration.py's outer() unchanged."""
    cal = intervals.module("sub/calibration.py", "calibration")
    cdf = cal.load(feats, train)
    cdf["rowid"] = np.arange(len(cdf))
    key = {(m, p): i for i, (m, p) in enumerate(zip(cdf.model_id, cdf.problem))}
    order = [key[(m, p)] for m, p in zip(df.model_id, df.problem)]
    a, _, _ = cal.outer(cdf, "rowid", False)
    b, _, _ = cal.outer(cdf, "problem", False)
    return {"A": np.asarray(a)[order], "B": np.asarray(b)[order]}


def split_predictions(df, predictors):
    return {"A": {n: oof_row(df, f) for n, f in predictors.items()},
            "B": {n: fp.oof(df, f) for n, f in predictors.items()}}


def cell_counts(y, p):
    """tp, fn, tn, fp of one set of rows."""
    return [int((p & y).sum()), int((~p & y).sum()), int((~p & ~y).sum()), int((p & ~y).sum())]


def balanced_from(c):
    """Balanced accuracy from (..., 4) counts, with the zero-denominator rule of fit_probe.scores."""
    tpr = c[..., 0] / np.maximum(c[..., 0] + c[..., 1], 1)
    tnr = c[..., 2] / np.maximum(c[..., 2] + c[..., 3], 1)
    return (tpr + tnr) / 2


def problem_weights(n_problems, draws=DRAWS, seed=SEED):
    """(draws, n_problems) integer weights: how often each problem is drawn. Rows of a problem move together."""
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.choice(n_problems, n_problems), minlength=n_problems) for _ in range(draws)])


def metrics(y, p, base):
    s = fp.scores(y, p)
    s["wins_vs_majority"] = int(((p == y) & (base != y)).sum())
    s["losses_vs_majority"] = int(((p != y) & (base == y)).sum())
    return s


def audit(df, preds, draws=DRAWS, seed=SEED):
    y = df.y.to_numpy()
    problems = sorted(df.problem.unique())
    groups = [(df.problem == g).to_numpy() for g in problems]
    arms = list(preds["A"])
    # counts[problem, arm, split, cell]
    counts = np.array([[[cell_counts(y[m], preds[s][a][m]) for s in "AB"] for a in arms] for m in groups])
    w = problem_weights(len(problems), draws, seed)
    boot = balanced_from(np.einsum("dp,pasc->dasc", w, counts))
    out = {}
    for j, a in enumerate(arms):
        lo, hi = np.percentile(boot[:, j, 0] - boot[:, j, 1], [2.5, 97.5])
        full = {s: metrics(y, preds[s][a], preds[s]["majority"]) for s in "AB"}
        out[a] = {"A": full["A"], "B": full["B"],
                  "balanced_A_minus_B": full["A"]["balanced"] - full["B"]["balanced"],
                  "interval95": [float(lo), float(hi)],
                  "accuracy_A_minus_B": full["A"]["accuracy"] - full["B"]["accuracy"]}
    return out


def within_model_labels(rows, rng):
    """Labels shuffled inside each model, so every model keeps its own label counts."""
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["model_id"], []).append(i)
    labels = [bool(r["is_robust"]) for r in rows]
    for idx in by.values():
        vals = [labels[i] for i in idx]
        rng.shuffle(vals)
        for i, v in zip(idx, vals):
            labels[i] = v
    return labels


def shuffle_within_model(rows, perms=2000, seed=0):
    rows = lesson9_shuffle.labelled(rows)
    obs = lesson9_counts.agreement(rows, "model")
    rng = random.Random(seed)
    counts = sorted(lesson9_counts.agreement([dict(r, is_robust=v) for r, v in zip(rows, within_model_labels(rows, rng))],
                                             "model")["agree"] for _ in range(perms))
    return {"observed_multi": obs["multi"], "observed_agree": obs["agree"], "null_mean_agree": sum(counts) / perms,
            "null_range_95": [counts[int(0.025 * perms)], counts[min(perms - 1, int(0.975 * perms))]],
            "shuffles_reaching_observed": sum(c >= obs["agree"] for c in counts), "perms": perms}


def shuffle_comparison(labels, perms):
    rows = labels.astype(object).where(labels.notna(), None).to_dict("records")
    small = [r for r in rows if lesson9_shuffle.SMALL_EXCLUDE not in r["model_id"]]
    return {name: {"pooled": lesson9_shuffle.shuffle_null(rs, perms, 0), "within_model": shuffle_within_model(rs, perms, 0)}
            for name, rs in (("all_labelled", rows), ("small_track", small))}


def main(root=None, train=None, perms=2000):
    root = Path(root) if root else HERE
    _, tname, tsha = trainset.resolve(train, HERE)
    feats = root / "runs/feat/features.jsonl"
    df = with_length(fp.table(feats, train))
    preds = split_predictions(df, arm_predictors())
    out = {**trainset.provenance(tname, tsha), "rows": len(df), "problems": int(df.problem.nunique()),
           "true_rate": float(df.y.mean()), "test_rows_with_same_problem_in_training_A": exposed(df),
           "length_features": LENGTH, "prompt_features": fp.ARMS["combined"]}
    out["arms"] = audit(df, preds)
    try:
        cp = calibration_preds(feats, train, df)
        out["arms"].update({k: v for k, v in audit(df, {s: {"majority": preds[s]["majority"], "calibration (nested)": cp[s]}
                                                         for s in "AB"}).items() if k != "majority"})
    except Exception as e:  # recorded, never silent: the arm is left out of the table
        out["calibration_omitted"] = f"{type(e).__name__}: {e}"
    labels, _, _ = trainset.read(train, HERE)
    out["label_agreement_shuffle"] = shuffle_comparison(labels, perms)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?")
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--out")
    trainset.add_argument(ap)
    a = ap.parse_args()
    text = json.dumps(main(a.root, a.train, a.perms), indent=1)
    Path(a.out).write_text(text) if a.out else print(text)
