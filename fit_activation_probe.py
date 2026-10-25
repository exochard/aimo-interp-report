"""Nested problem-held-out linear probes using a small native-activation Gram matrix."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import trainset
from run_activation_probe import HERE, SMALL, row_key

PENALTIES = (0.03, 0.1, 0.3)


def native_kernel(vectors, models):
    """Separate native coordinate spaces by family; equal L2 weight for each pool."""
    n = len(vectors)
    k = np.zeros((n, n), dtype=float)
    normal = [v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12) for v in vectors]
    for i in range(n):
        for j in range(i + 1):
            if models[i] == models[j]:
                k[i, j] = k[j, i] = float(np.mean(np.sum(normal[i] * normal[j], axis=1))) + 0.25
    return k


def predict(k, y, train, test, penalty):
    """Kernel ridge is a linear probe; centering is computed on fitting rows alone."""
    target = 2 * y[train].astype(float) - 1
    kt = k[np.ix_(train, train)]
    mean_col, mean_all = kt.mean(0), kt.mean()
    centered = kt - mean_col[None, :] - mean_col[:, None] + mean_all
    alpha = np.linalg.solve(centered + penalty * np.eye(len(train)), target - target.mean())
    cross = k[np.ix_(test, train)]
    cross = cross - mean_col[None, :] - cross.mean(1, keepdims=True) + mean_all
    return cross @ alpha + target.mean() >= 0


def held_out(k, y, groups, indices, penalty):
    result = np.zeros(len(indices), dtype=bool)
    for group in np.unique(groups[indices]):
        mask = groups[indices] == group
        result[mask] = predict(k, y, indices[~mask], indices[mask], penalty)
    return result


def nested(k, y, groups):
    result = np.zeros(len(y), dtype=bool)
    folds = []
    for group in np.unique(groups):
        test, train = np.flatnonzero(groups == group), np.flatnonzero(groups != group)
        accuracies = [float((held_out(k, y, groups, train, p) == y[train]).mean()) for p in PENALTIES]
        penalty = PENALTIES[int(np.argmax(accuracies))]
        result[test] = predict(k, y, train, test, penalty)
        folds.append(dict(problem_sha256=str(group), fit_rows=train.tolist(), held_out_rows=test.tolist(),
                          penalty=penalty, inner_accuracy=accuracies))
    return result, folds


def metrics(y, p):
    return dict(correct=int((y == p).sum()), n=len(y), accuracy=float((y == p).mean()),
                balanced_accuracy=float(np.mean([(y[y == value] == p[y == value]).mean()
                                                 for value in (False, True)])))


def evaluate(folder, train=None):
    manifest = json.loads((folder / "manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("incomplete extraction; quality gate remains unmeasured")
    labels_path, train_name, train_sha = trainset.resolve(train, HERE)
    if train_sha != manifest["label_sha256"]:
        raise ValueError("labels changed since extraction")
    labels = pd.read_parquet(labels_path)
    labels = labels[labels.model_id.isin(SMALL) & labels.is_robust.notna()].reset_index(drop=True)
    vectors = [np.load(folder / (row_key(r.model_id, r.problem, r.reasoning_effort) + ".npz"))["activation"]
               for r in labels.itertuples()]
    groups = np.array([hashlib.sha256(p.encode()).hexdigest() for p in labels.problem])
    k = native_kernel(vectors, labels.model_id.to_numpy())
    y = labels.is_robust.to_numpy(bool)
    probe, folds = nested(k, y, groups)
    majority = np.array([y[groups != g].mean() >= 0.5 for g in groups])
    drops = [int((probe[groups != g] == y[groups != g]).sum() -
                 (majority[groups != g] == y[groups != g]).sum()) for g in np.unique(groups)]
    m, b = metrics(y, probe), metrics(y, majority)
    gate = dict(coverage=manifest["completed_rows"] == len(labels), accuracy=m["accuracy"] >= 0.8,
                three_over_majority=m["correct"] - b["correct"] >= 3,
                three_over_calibration=m["correct"] - 25 >= 3,
                balanced=m["balanced_accuracy"] >= 0.7, drop_one_problem=min(drops) > 0)
    return dict(probe=m, majority=b, calibration_correct=25, gate=gate, passed=all(gate.values()),
                **trainset.provenance(train_name, train_sha), precision=manifest["models"], native_dimensions={r.model_id: int(v.size)
                    for r, v in zip(labels.itertuples(), vectors)}, folds=folds,
                predictions=[dict(problem_sha256=str(g), model_id=str(model), truth=bool(t),
                    probe=bool(p), majority=bool(base)) for g, model, t, p, base in
                    zip(groups, labels.model_id, y, probe, majority)],
                min_drop_margin=min(drops), claim="local mixed-precision screen; official precision untested")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    trainset.add_argument(ap)
    args = ap.parse_args()
    report = evaluate(args.features, args.train)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ("probe", "majority", "gate", "passed")}))
