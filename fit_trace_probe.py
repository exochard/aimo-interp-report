"""Nested whole-problem validation of the trace-sensitivity probe on train-main-v2 (small track).

Frozen 2026-10-13 before any trace feature row exists (coordination/2026-10-13-r1/plan.md §4).
Rows: small-track models with a True/False label. Features: trace_features.FEATURES (eleven,
predeclared). Model: L2 logistic regression; missing values take the fitting rows' mean (the
missing-family and missing-control indicators are features themselves); standardisation and the
choice of C in {0.03, 0.1, 0.3} happen inside the outer training fold, by inner leave-one-problem-
out accuracy. Model-family intercepts enter as one-hot columns scaled by 0.5 after
standardisation, a shrinkage prior learned on fitting rows only.

Gate (all must hold): nested problem-holdout accuracy >= 0.80; at least 3 more correct rows than
the better of always-majority and nested calibration (25/34, runs/calibration-nested/report.json);
balanced accuracy >= 0.70; dropping any one problem group leaves a positive margin. Family
holdout (train on three models, predict the fourth) is reported separately; a loss below its
majority by more than one row blocks any cross-model claim.

  uv run --no-project --with pandas --with pyarrow --with scikit-learn python fit_trace_probe.py ROWS.jsonl OUT.json [--train pinned|1015|PATH]
ROWS.jsonl: {"model_id", "problem", "features": {...} | null} per line.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sub"))
from trace_features import FEATURES  # noqa: E402
import trainset  # noqa: E402

SMALL = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
         "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]
CS = (0.03, 0.1, 0.3)
FAMILY_SCALE = 0.5
CALIBRATION_CORRECT = 25  # nested calibration, runs/calibration-nested/report.json


def table(path: Path, labels: pd.DataFrame | None = None, train: str | None = None) -> pd.DataFrame:
    if labels is None:
        labels, _, _ = trainset.read(train, HERE)
    feats = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("features"):
            feats[(r["model_id"], r["problem"])] = r["features"]
    rows = []
    for _, lab in labels.iterrows():
        if lab.model_id in SMALL and lab.is_robust is not None:
            f = feats.get((lab.model_id, lab.problem))
            rows.append({"model_id": lab.model_id, "problem": lab.problem, "y": bool(lab.is_robust),
                         "has_features": f is not None, **{k: (f or {}).get(k, np.nan) for k in FEATURES}})
    return pd.DataFrame(rows)


def design(train: pd.DataFrame, rows: pd.DataFrame):
    X_tr = train[FEATURES].astype(float)
    mean = X_tr.mean().fillna(0.0)
    sc = StandardScaler().fit(X_tr.fillna(mean))

    def build(df):
        X = sc.transform(df[FEATURES].astype(float).fillna(mean))
        fam = np.stack([(df.model_id == m).to_numpy(float) * FAMILY_SCALE for m in SMALL], axis=1)
        return np.hstack([np.nan_to_num(X), fam])
    return build(train), build(rows)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, c: float) -> np.ndarray:
    if train.y.nunique() < 2:
        return np.full(len(test), bool(train.y.iloc[0]))
    Xtr, Xte = design(train, test)
    clf = LogisticRegression(C=c, max_iter=2000).fit(Xtr, train.y.to_numpy())
    return clf.predict(Xte).astype(bool)


def logo(df: pd.DataFrame, predict, by="problem") -> np.ndarray:
    out = np.zeros(len(df), dtype=bool)
    for g in df[by].unique():
        test = (df[by] == g).to_numpy()
        out[test] = predict(df[~test], df[test])
    return out


def nested(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    accs = {c: (logo(train, lambda a, b, c=c: fit_predict(a, b, c)) == train.y.to_numpy()).mean() for c in CS}
    c = max(CS, key=lambda k: (accs[k], -k))  # ties go to the stronger penalty
    return fit_predict(train, test, c)


def majority(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return np.full(len(test), bool(train.y.mean() >= 0.5))


def score(y, p) -> dict:
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tpr = (p & y).sum() / max(1, y.sum())
    tnr = (~p & ~y).sum() / max(1, (~y).sum())
    return {"correct": int((y == p).sum()), "n": int(len(y)), "accuracy": float((y == p).mean()),
            "balanced": float((tpr + tnr) / 2)}


def main(rows_path: str, out_path: str, train: str | None = None):
    _, tname, tsha = trainset.resolve(train, HERE)
    df = table(Path(rows_path), train=train)
    y = df.y.to_numpy()
    probe = logo(df, nested)
    base = logo(df, majority)
    s_probe, s_base = score(y, probe), score(y, base)
    best_base = max(s_base["correct"], CALIBRATION_CORRECT)
    drops = []
    for g in df.problem.unique():
        keep = (df.problem != g).to_numpy()
        # Calibration's per-row results are not stored: assume it kept all its correct rows (upper bound).
        drops.append(int((probe[keep] == y[keep]).sum()) - max(int((base[keep] == y[keep]).sum()),
                                                                 min(CALIBRATION_CORRECT, int(keep.sum()))))
    fam = logo(df, nested, by="model_id")
    fam_base = logo(df, majority, by="model_id")
    gate = {"accuracy": s_probe["accuracy"] >= 0.80, "margin": s_probe["correct"] - best_base >= 3,
            "balanced": s_probe["balanced"] >= 0.70, "drop_one_group": min(drops) > 0}
    report = {"rows": len(df), "with_features": int(df.has_features.sum()), "probe": s_probe,
              "majority": s_base, "calibration_correct": CALIBRATION_CORRECT, "drop_one_group_margin_min": min(drops),
              "family_holdout": score(y, fam), "family_majority": score(y, fam_base),
              "gate": gate, "pass": all(gate.values()), **trainset.provenance(tname, tsha)}
    Path(out_path).write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    choice, args = trainset.strip_flag(sys.argv[1:])
    main(*args[:2], train=choice)
