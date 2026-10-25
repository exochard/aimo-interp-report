"""Calibration of robustness verdicts against model and problem effects, with grouped CV.

The target is whether a model's reasoning is robust on a problem, not whether the problem is
hard. Per (model, problem) row the feature blocks are:

- ability: the model's empirical-Bayes shrunken robust rate, (sum y + k p0) / (n + k) on the
  logit scale (a model random effect), computed without the row's own problem;
- difficulty: the problem's shrunken rate over the other models, computed without the row's
  own model (a problem random effect). A held-out problem has no other rows, so this block
  exists only when a whole model is held out; with a problem held out, difficulty has to come
  from covariates (prompt nll and length);
- sensitivity: prompt-only deltas from runs/feat (the failed 0.676 control uses all eight
  prompt features);
- stability: answer agreement, consistency and confidence across perturbed generations
  (analyze_gen.features). A feature is used only if present for 80% of rows; today it is not.

Validation is nested: the outer loop leaves out a whole problem (20 folds), or separately a
whole model family; an inner leave-one-problem-out loop on the training rows alone picks the
arm, the shrinkage k and the logistic C together. No arm is chosen by its outer score; the
per-arm outer numbers are descriptive only. Gate (Codex 2026-10-11): pooled OOF accuracy >= 0.80, at least 3
more correct rows than the 0.676 control (23 of 34), no held-out model below its majority
baseline.

  uv run --no-project --with pandas --with pyarrow --with scikit-learn python sub/calibration.py runs/feat/features.jsonl
"""

from __future__ import annotations

import json
import os
import sys
from itertools import product
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import trainset  # noqa: E402
SMALL = ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
         "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"]
FAMILY = {"Qwen/Qwen3.5-4B": "qwen3.5", "Skywork/Skywork-OR1-Math-7B": "skywork-or1",
          "allenai/Olmo-3-7B-Think": "olmo3", "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "deepseek-r1-qwen3"}
PROMPT = ["nll", "log_tokens"] + [f"{k}_{v}" for v in ("typos", "distract") for k in ("dnll", "dcos50", "dcos75")]
STABILITY = ["orig_consistency", "agree_typos", "agree_distract", "drop_typos", "drop_distract", "dlogprob_typos",
             "dlogprob_distract"]
GRID_K, GRID_C = (1.0, 4.0, 16.0), (0.1, 1.0, 10.0)


def load(feat_path: Path, train: str | None = None) -> pd.DataFrame:
    labels, _, _ = trainset.read(train, HERE)
    feats = {}
    for line in open(feat_path):
        r = json.loads(line)
        if r["features"]:
            feats[(r["model_id"], r["problem"])] = r["features"]
    rows = [{"model_id": r.model_id, "problem": r.problem, "y": bool(r.is_robust), **feats[(r.model_id, r.problem)]}
            for r in labels.itertuples() if r.model_id in SMALL and r.is_robust is not None and (r.model_id, r.problem) in feats]
    df = pd.DataFrame(rows)
    df["family"] = df.model_id.map(FAMILY)
    # runs/agreement-vs-labels.csv holds no problem text and the Qwen generations are 96% unanswered
    # (8,192-token cap), so stability stays missing until kaggle/perturb-gen has run.
    for c in STABILITY:
        df[c] = np.nan
    return df


def effect(train: pd.DataFrame, rows: pd.DataFrame, key: str, other: str, k: float) -> np.ndarray:
    """Shrunken logit rate of rows[key] in train, excluding train rows that share rows[other]."""
    p0 = min(max(train.y.mean(), 0.05), 0.95)
    out = np.empty(len(rows))
    for i, (kv, ov) in enumerate(zip(rows[key], rows[other])):
        t = train[(train[key] == kv) & (train[other] != ov)]
        p = (t.y.sum() + k * p0) / (len(t) + k)
        out[i] = np.log(p / (1 - p))
    return out


def design(train, rows, arm, k, with_difficulty):
    cols = []
    if "ability" in arm:
        cols.append(effect(train, rows, "model_id", "problem", k)[:, None])
    if "difficulty" in arm and with_difficulty:
        cols.append(effect(train, rows, "problem", "model_id", k)[:, None])
    if "covariates" in arm:
        cols.append(rows[["nll", "log_tokens"]].to_numpy(float))
    if "sensitivity" in arm:
        cols.append(rows[PROMPT[2:]].to_numpy(float))
    return np.hstack(cols) if cols else np.zeros((len(rows), 0))


def fit_predict(train, test, arm, k, c, with_difficulty):
    if train.y.nunique() < 2 or not arm:
        return np.full(len(test), train.y.mean() >= 0.5)
    xtr = design(train, train, arm, k, with_difficulty)
    xte = design(train, test, arm, k, with_difficulty)
    sc = StandardScaler().fit(xtr)
    m = LogisticRegression(C=c, max_iter=1000).fit(sc.transform(xtr), train.y)
    return m.predict(sc.transform(xte)).astype(bool)


def inner_table(train, with_difficulty):
    """Leave-one-problem-out correct count inside `train` for every (arm, k, C); the only place
    an arm or a hyper-parameter is chosen, so a test group never influences its own choice."""
    table = {}
    for name, arm in ARMS.items():
        for k, c in product(GRID_K, GRID_C):
            hit = 0
            for g in train.problem.unique():
                te = train.problem == g
                hit += int((fit_predict(train[~te], train[te], arm, k, c, with_difficulty) == train[te].y.to_numpy()).sum())
            table[(name, k, c)] = hit
    return table


def best(table, only=None):
    """Highest inner count; ties go to the earlier arm, then the smaller shrinkage and C."""
    keys = [key for key in table if only is None or key[0] == only]
    top = max(table[key] for key in keys)
    return next(key for key in keys if table[key] == top)


def outer(df, by, with_difficulty):
    """Outer folds are whole `by` groups. Each fold selects inside its training rows: `selected`
    picks arm, k and C jointly; `arms[name]` picks only k and C for a fixed arm (descriptive)."""
    selected, arms, chosen = np.zeros(len(df), dtype=bool), {n: np.zeros(len(df), dtype=bool) for n in ARMS}, []
    for g in df[by].unique():
        te = (df[by] == g).to_numpy()
        train = df[~te]
        table = inner_table(train, with_difficulty)
        name, k, c = best(table)
        chosen.append({"held_out": str(g), "arm": name, "k": k, "C": c, "inner_correct": table[(name, k, c)],
                       "inner_rows": len(train)})
        selected[te] = fit_predict(train, df[te], ARMS[name], k, c, with_difficulty)
        for n, arm in ARMS.items():
            _, kk, cc = best(table, n)
            arms[n][te] = fit_predict(train, df[te], arm, kk, cc, with_difficulty)
    return selected, arms, chosen


def control(df):
    """fit_probe.py combined arm: standardised logistic, C=1, all eight prompt features."""
    out = np.zeros(len(df), dtype=bool)
    for g in df.problem.unique():
        te = (df.problem == g).to_numpy()
        sc = StandardScaler().fit(df[~te][PROMPT])
        m = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(df[~te][PROMPT]), df[~te].y)
        out[te] = m.predict(sc.transform(df[te][PROMPT])).astype(bool)
    return out


def majority(df, by, per_model=False):
    out = np.zeros(len(df), dtype=bool)
    for g in df[by].unique():
        te = (df[by] == g).to_numpy()
        tr = df[~te]
        rate = tr.groupby("model_id").y.mean()
        out[te] = [rate.get(m, tr.y.mean()) >= 0.5 if per_model else tr.y.mean() >= 0.5 for m in df[te].model_id]
    return out


def score(y, p):
    tpr = (p & y).sum() / max(y.sum(), 1)
    tnr = (~p & ~y).sum() / max((~y).sum(), 1)
    return {"correct": int((p == y).sum()), "accuracy": round(float((p == y).mean()), 4),
            "balanced": round(float((tpr + tnr) / 2), 4)}


ARMS = {"ability": ("ability",), "ability+difficulty": ("ability", "difficulty"),
        "ability+covariates": ("ability", "covariates"), "ability+difficulty+covariates": ("ability", "difficulty", "covariates"),
        "ability+difficulty+sensitivity": ("ability", "difficulty", "covariates", "sensitivity")}


def paired(y, p, q):
    """Discordant counts of p against q and the exact two-sided sign-test p-value."""
    win, loss = int(((p == y) & (q != y)).sum()), int(((p != y) & (q == y)).sum())
    n = win + loss
    tail = sum(comb(n, i) for i in range(min(win, loss) + 1)) / 2 ** n if n else 1.0
    return {"wins": win, "losses": loss, "p_two_sided": round(min(1.0, 2 * tail), 4)}


def main():
    choice, argv = trainset.strip_flag(sys.argv[1:])
    _, tname, tsha = trainset.resolve(choice, HERE)
    df = load(Path(argv[0] if argv else HERE / "runs/feat/features.jsonl"), choice)
    y = df.y.to_numpy()
    usable = [c for c in STABILITY if df[c].notna().mean() >= 0.8]
    ctrl_pred = control(df)
    report = {"rows": len(df), "problems": int(df.problem.nunique()), "models": int(df.model_id.nunique()),
              "true_rate": round(float(y.mean()), 4), "stability_features_used": usable,
              "stability_features_missing": [c for c in STABILITY if c not in usable],
              "arms_offered": list(ARMS), "grid": {"k": GRID_K, "C": GRID_C},
              "control_combined_prompt": score(y, ctrl_pred),
              "problem_holdout": {"majority": score(y, majority(df, "problem")),
                                  "per_model_majority": score(y, majority(df, "problem", True))},
              "model_holdout": {"majority": score(y, majority(df, "family"))},
              **trainset.provenance(tname, tsha)}
    sel, arms, chosen = outer(df, "problem", False)
    report["problem_holdout"]["selected"] = score(y, sel)
    report["problem_holdout"]["selection_by_fold"] = chosen
    report["problem_holdout"]["arm_counts"] = {n: sum(c["arm"] == n for c in chosen) for n in ARMS}
    report["problem_holdout"]["descriptive_arms"] = {n: score(y, p) for n, p in arms.items()}
    fsel, farms, fchosen = outer(df, "family", True)
    per = {}
    for f in df.family.unique():
        te = (df.family == f).to_numpy()
        tr_major = bool(df[~te].y.mean() >= 0.5)
        per[f] = {"n": int(te.sum()), "correct": int((fsel[te] == y[te]).sum()),
                  "train_majority_correct": int((y[te] == tr_major).sum()),
                  "own_majority_correct": int(max(y[te].sum(), (~y[te]).sum()))}
    report["model_holdout"]["selected"] = {**score(y, fsel), "per_family": per, "selection_by_fold": fchosen}
    ph, mh = report["problem_holdout"]["selected"], report["model_holdout"]["selected"]
    ctrl = report["control_combined_prompt"]["correct"]
    report["gate"] = {"selector": "arm, k and C chosen by leave-one-problem-out inside each outer training fold",
                      "oof_accuracy": ph["accuracy"], "accuracy_ok": ph["accuracy"] >= 0.80,
                      "net_vs_control": ph["correct"] - ctrl, "margin_ok": ph["correct"] - ctrl >= 3,
                      "paired_vs_control": paired(y, sel, ctrl_pred),
                      "net_vs_majority": ph["correct"] - report["problem_holdout"]["majority"]["correct"],
                      "note": "model-holdout difficulty reads other models' labels on the same problem, which a new problem would not have",
                      "family_ok": all(v["correct"] >= v["train_majority_correct"] for v in mh["per_family"].values()),
                      "timing_path": "not measured: no perturbed-generation run exists yet"}
    report["gate"]["pass"] = all(report["gate"][k] for k in ("accuracy_ok", "margin_ok", "family_ok"))
    out = Path(os.environ.get("CAL_OUT", HERE / "runs/calibration-nested"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("model_holdout",)} | {"model_holdout_selected": {k: v for k, v in mh.items() if k != "selection_by_fold"}}, indent=1))


if __name__ == "__main__":
    main()
