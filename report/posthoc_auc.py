"""Post hoc ranking check for the five gated signals (not part of any predeclared gate).

  python report/posthoc_auc.py [ROOT] [--train pinned|1015|PATH] > report/posthoc_auc.json

ROOT is the folder holding runs/ (default: the lane folder). --train (default: $AIMO_TRAIN, else pinned)
chooses the label file; with a non-default file the reported-count assertions and the hidden-activation row
(saved against the pinned rows) are skipped, and the output gains train_file and train_sha256. For each gated signal it recomputes the
held-out score behind the Section 4 prediction (the decision value of the same model, fitted on the
same training rows, with the same inner selection), checks that thresholding it at the model's own
cut reproduces the reported correct count, and reports:

- the held-out AUC with robust as the positive class, and a 95 % interval from 10,000 resamples of
  whole problems (seed as in intervals.py);
- the accuracy when the pooled scores are cut so that the share predicted robust equals the mean
  training-fold robust share (the prior-matched threshold).
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import trainset  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))


def auc(y, s):
    """AUC with robust (True) as positive; ties count half. None when a class is empty."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean())


def bootstrap_auc(y, s, groups, draws=10_000, seed=20261013):
    """95 % interval of the AUC, resampling whole groups (problems); draws with one class are redrawn."""
    y, s, groups = np.asarray(y, bool), np.asarray(s, float), np.asarray(groups)
    names = sorted(set(groups.tolist()))
    index = {g: np.flatnonzero(groups == g) for g in names}
    rng = np.random.default_rng(seed)
    values = []
    while len(values) < draws:
        pick = np.concatenate([index[names[i]] for i in rng.choice(len(names), len(names))])
        a = auc(y[pick], s[pick])
        if a is not None:
            values.append(a)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def prior_matched(y, s, share):
    """Predict robust for the round(share * n) highest scores (ties at the cut all count as robust).

    Returns (correct, predicted robust count)."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    k = int(round(share * len(s)))
    if k <= 0:
        pred = np.zeros(len(s), bool)
    else:
        pred = s >= np.sort(s)[::-1][k - 1]
    return int((pred == y).sum()), int(pred.sum())


def training_share(y, groups):
    """Mean over rows of the robust share of the rows outside the row's own group."""
    y, groups = np.asarray(y, bool), np.asarray(groups)
    return float(np.mean([y[groups != g].mean() for g in groups]))


def summarise(y, s, groups, cut=0.0, draws=10_000):
    """Everything the report table needs for one signal; `cut` is the model's own decision threshold."""
    y, s = np.asarray(y, bool), np.asarray(s, float)
    share = training_share(y, groups)
    correct, predicted = prior_matched(y, s, share)
    lo, hi = bootstrap_auc(y, s, groups, draws=draws)
    return {"rows": int(len(y)), "own_cut_correct": int((y == (s >= cut)).sum()), "auc": auc(y, s),
            "auc_interval95": [lo, hi], "train_share": share, "prior_matched_predicted_robust": predicted,
            "prior_matched_correct": correct}


def _logit(train_x, train_y, test_x, c, max_iter):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=c, max_iter=max_iter).fit(train_x, train_y).decision_function(test_x)


def prompt_scores(df, cols):
    from sklearn.preprocessing import StandardScaler
    out = np.zeros(len(df))
    for g in df.problem.unique():
        te = (df.problem == g).to_numpy()
        sc = StandardScaler().fit(df[~te][cols])
        out[te] = _logit(sc.transform(df[~te][cols]), df[~te].y, sc.transform(df[te][cols]), 1.0, 1000)
    return out


def calibration_scores(cal, df):
    from sklearn.preprocessing import StandardScaler
    out = np.zeros(len(df))
    for g in df.problem.unique():
        te = (df.problem == g).to_numpy()
        train = df[~te]
        name, k, c = cal.best(cal.inner_table(train, False))
        arm = cal.ARMS[name]
        xtr, xte = cal.design(train, train, arm, k, False), cal.design(train, df[te], arm, k, False)
        sc = StandardScaler().fit(xtr)
        out[te] = _logit(sc.transform(xtr), train.y, sc.transform(xte), c, 1000)
    return out


def trace_scores(tp, df):
    out = np.zeros(len(df))
    for g in df.problem.unique():
        te = (df.problem == g).to_numpy()
        train, test = df[~te], df[te]
        accs = {c: (tp.logo(train, lambda a, b, c=c: tp.fit_predict(a, b, c)) == train.y.to_numpy()).mean()
                for c in tp.CS}
        c = max(tp.CS, key=lambda k: (accs[k], -k))
        xtr, xte = tp.design(train, test)
        out[te] = _logit(xtr, train.y.to_numpy(), xte, c, 2000)
    return out


def activation_scores(fa, folder, folds, labels):
    """Kernel-ridge decision values for the saved folds (penalty per fold from the saved report)."""
    import hashlib

    vectors = [np.load(folder / (fa.row_key(r.model_id, r.problem, r.reasoning_effort) + ".npz"))["activation"]
               for r in labels.itertuples()]
    groups = np.array([hashlib.sha256(p.encode()).hexdigest() for p in labels.problem])
    k = fa.native_kernel(vectors, labels.model_id.to_numpy())
    y = labels.is_robust.to_numpy(bool)
    out = np.zeros(len(y))
    for fold in folds:
        test = np.flatnonzero(groups == fold["problem_sha256"])
        train = np.flatnonzero(groups != fold["problem_sha256"])
        target = 2 * y[train].astype(float) - 1
        kt = k[np.ix_(train, train)]
        mean_col, mean_all = kt.mean(0), kt.mean()
        centered = kt - mean_col[None, :] - mean_col[:, None] + mean_all
        alpha = np.linalg.solve(centered + fold["penalty"] * np.eye(len(train)), target - target.mean())
        cross = k[np.ix_(test, train)]
        cross = cross - mean_col[None, :] - cross.mean(1, keepdims=True) + mean_all
        out[test] = cross @ alpha + target.mean()
    return out, y, groups


def activation_row(fa, root, reported, train=None):
    """The activation-probe summary, or None (with a stderr note) when the features folder is not shipped."""
    import pandas as pd

    folder = root / "runs/activation-r3-yuris/features"
    if not folder.is_dir():
        print(f"skipping the hidden-activation row: {folder} is missing (needs the GPU feature run)", file=sys.stderr)
        return None
    act = json.loads((root / "runs/activation-r3-yuris/probe.json").read_text())
    labels = pd.read_parquet(trainset.resolve(train, root)[0])
    labels = labels[labels.model_id.isin(fa.SMALL) & labels.is_robust.notna()].reset_index(drop=True)
    s, y, groups = activation_scores(fa, folder, act["folds"], labels)
    row = summarise(y, s, groups, 0.0)
    row["reported_correct"] = reported
    saved = np.array([r["probe"] for r in act["predictions"]], bool)
    assert (saved == (s >= 0)).all(), "activation scores do not reproduce the saved predictions"
    return row


def main(root=None, train=None):
    import intervals as iv

    root = Path(root) if root else HERE
    _, tname, tsha = trainset.resolve(train, HERE)
    print(f"train file: {tname} {tsha}", file=sys.stderr)
    fp = iv.module("fit_probe.py", "fit_probe")
    cal = iv.module("sub/calibration.py", "calibration")
    tp = iv.module("fit_trace_probe.py", "fit_trace_probe")
    fa = iv.module("fit_activation_probe.py", "fit_activation_probe")
    expected = json.loads((Path(__file__).resolve().parent / "intervals.json").read_text())["reproduced"]
    feats = root / "runs/feat/features.jsonl"
    signals = {}

    df = fp.table(feats, train)
    for arm, cols in fp.ARMS.items():
        signals[f"prompt:{arm}"] = (df, prompt_scores(df, cols), 0.0)
    cdf = cal.load(feats, train)
    signals["calibration (nested)"] = (cdf, calibration_scores(cal, cdf), 0.0)
    tdf = tp.table(root / "runs/trace-pilot/rows.jsonl", train=train)
    signals["trace sensitivity"] = (tdf, trace_scores(tp, tdf), 0.0)

    out = {}
    for name, (frame, s, cut) in signals.items():
        out[name] = summarise(frame.y.to_numpy(), s, frame.problem.to_numpy(), cut)
        out[name]["reported_correct"] = expected[name]["reported"] if tname == "pinned" else None

    if tname == "pinned":
        row = activation_row(fa, root, expected["hidden-activation probe"]["reported"], train)
        if row:
            out["hidden-activation probe"] = row
        for name, r in out.items():
            assert r["own_cut_correct"] == r["reported_correct"], f"{name}: scores do not reproduce the reported count"
    else:
        out.update(trainset.provenance(tname, tsha))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    choice, args = trainset.strip_flag(sys.argv[1:])
    main(args[0] if args else None, choice)
