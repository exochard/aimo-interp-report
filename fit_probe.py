"""Grouped validation of the prompt-feature robustness predictor on train-main-v2.

Frozen 2026-10-11 before any feature row existed. Rows: small-track models with a True/False
label (34). Groups: the problem text, so every model's copy of a problem is held out together.
Arms, each a standardised L2 logistic regression (C=1) fit inside the training groups:

- capability: nll, log_tokens
- sensitivity: the six variant deltas
- combined: all eight

Baselines fit the same way: always-majority and per-model majority. Gate (Codex 2026-10-11):
combined pooled OOF accuracy >= 0.80, at least 3 more correct rows than the better baseline,
and the gain survives dropping any single problem group. Balanced accuracy is reported too: the
public scoring set's always-true accuracy is near 0.5, unlike the 0.71 here.

  uv run --no-project --with pandas --with pyarrow --with scikit-learn python fit_probe.py runs/feat/features.jsonl [--train pinned|1015|PATH]
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
from features import FEATURES  # noqa: E402
import trainset  # noqa: E402

SMALL = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
         "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]
ARMS = {"capability": ["nll", "log_tokens"], "sensitivity": FEATURES[2:], "combined": FEATURES}


def table(path: Path, train: str | None = None) -> pd.DataFrame:
    labels, _, _ = trainset.read(train, HERE)
    feats = {}
    for line in open(path):
        r = json.loads(line)
        if r["features"]:
            feats[(r["model_id"], r["problem"])] = r["features"]
    rows = []
    for _, lab in labels.iterrows():
        if lab.model_id in SMALL and lab.is_robust is not None and (lab.model_id, lab.problem) in feats:
            rows.append({"model_id": lab.model_id, "problem": lab.problem, "y": bool(lab.is_robust),
                         **feats[(lab.model_id, lab.problem)]})
    return pd.DataFrame(rows)


def oof(df: pd.DataFrame, predict) -> np.ndarray:
    out = np.zeros(len(df), dtype=bool)
    for g in df.problem.unique():
        test = (df.problem == g).to_numpy()
        out[test] = predict(df[~test], df[test])
    return out


def logistic(cols):
    def predict(train, test):
        if train.y.nunique() < 2:
            return np.full(len(test), bool(train.y.iloc[0]))
        sc = StandardScaler().fit(train[cols])
        m = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(train[cols]), train.y)
        return m.predict(sc.transform(test[cols])).astype(bool)
    return predict


def majority(train, test):
    return np.full(len(test), train.y.mean() >= 0.5)


def per_model(train, test):
    rate = train.groupby("model_id").y.mean()
    return np.array([rate.get(m, train.y.mean()) >= 0.5 for m in test.model_id])


def scores(y, p):
    tpr = (p & y).sum() / max(y.sum(), 1)
    tnr = (~p & ~y).sum() / max((~y).sum(), 1)
    return {"correct": int((p == y).sum()), "accuracy": float((p == y).mean()), "balanced": float((tpr + tnr) / 2)}


def main():
    choice, argv = trainset.strip_flag(sys.argv[1:])
    _, tname, tsha = trainset.resolve(choice, HERE)
    df = table(Path(argv[0]), choice)
    y = df.y.to_numpy()
    preds = {"majority": oof(df, majority), "per_model": oof(df, per_model),
             **{a: oof(df, logistic(c)) for a, c in ARMS.items()}}
    report = {"rows": len(df), "groups": int(df.problem.nunique()), "true_rate": float(y.mean()),
              "arms": {k: scores(y, p) for k, p in preds.items()},
              **trainset.provenance(tname, tsha)}
    base = max(report["arms"]["majority"]["correct"], report["arms"]["per_model"]["correct"])
    comb = preds["combined"]
    margin = report["arms"]["combined"]["correct"] - base
    better = preds["majority"] if report["arms"]["majority"]["correct"] >= report["arms"]["per_model"]["correct"] else preds["per_model"]
    # Gain without each problem group: the combined arm's net wins over the baseline elsewhere.
    net = [(int(((comb == y) & ~(df.problem == g).to_numpy()).sum()) - int(((better == y) & ~(df.problem == g).to_numpy()).sum()))
           for g in df.problem.unique()]
    report["gate"] = {"accuracy_ok": report["arms"]["combined"]["accuracy"] >= 0.80, "margin": margin,
                      "margin_ok": margin >= 3, "min_margin_drop_one_group": min(net) if net else None,
                      "robust_ok": bool(net) and min(net) > 0}
    report["gate"]["pass"] = all(report["gate"][k] for k in ("accuracy_ok", "margin_ok", "robust_ok"))
    sc = StandardScaler().fit(df[ARMS["combined"]])
    m = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(df[ARMS["combined"]]), df.y)
    rates = df.groupby("model_id").y.mean()
    artifact = {"features": ARMS["combined"], "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
                "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0]), "threshold": 0.5,
                "models": SMALL, "fallback": {k: bool(v >= 0.5) for k, v in rates.items()},
                "default": bool(df.y.mean() >= 0.5), "source": str(Path(argv[0]).name), "rows": len(df),
                **trainset.provenance(tname, tsha)}
    (HERE / "runs/probe.json").write_text(json.dumps(artifact, indent=1))
    out = HERE / "runs/probe-gate.json"
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
