import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import fit_probe as fp  # noqa: E402
import split_audit as S  # noqa: E402

MODELS = ["m1", "m2", "m3"]


def frame(n_problems=12, models=MODELS, seed=0):
    """Label fixed by the problem; features are noise, so only recognizing the problem can help."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_problems):
        for m in models:
            rows.append({"model_id": m, "problem": f"problem {k} " + "x" * (3 * k), "y": bool(k % 2),
                         **{c: rng.normal() for c in fp.ARMS["combined"]}})
    return S.with_length(pd.DataFrame(rows))


def preds_for(df):
    return S.split_predictions(df, S.arm_predictors())


def test_problem_determined_labels_split_a_wins_and_control_perfect():
    df = frame()
    p = preds_for(df)
    y = df.y.to_numpy()
    assert (p["A"]["leak_control"] == y).all()
    r = S.audit(df, p, draws=500)
    assert r["leak_control"]["A"]["accuracy"] == 1.0
    assert r["leak_control"]["B"]["accuracy"] < 1.0
    assert r["leak_control"]["balanced_A_minus_B"] > 0
    assert S.exposed(df) == len(df)


def test_no_repeated_problems_splits_identical():
    df = frame(models=["m1"])
    p = preds_for(df)
    for arm in p["A"]:
        assert (p["A"][arm] == p["B"][arm]).all()
    assert S.exposed(df) == 0
    r = S.audit(df, p, draws=200)
    assert all(v["balanced_A_minus_B"] == 0 and v["interval95"] == [0.0, 0.0] for v in r.values())


def test_within_model_shuffle_keeps_model_label_counts():
    rows = [{"model_id": m, "problem": f"p{k}", "is_robust": (k + i) % 3 == 0}
            for i, m in enumerate(MODELS) for k in range(9)]
    out = S.within_model_labels(rows, random.Random(1))
    for m in MODELS:
        idx = [i for i, r in enumerate(rows) if r["model_id"] == m]
        assert sum(out[i] for i in idx) == sum(rows[i]["is_robust"] for i in idx)
    assert out != [r["is_robust"] for r in rows]
    res = S.shuffle_within_model(rows, perms=50, seed=0)
    assert res["perms"] == 50 and 0 <= res["null_mean_agree"] <= res["observed_multi"]


def test_bootstrap_resamples_whole_problems():
    df = frame(6)
    p = preds_for(df)
    y = df.y.to_numpy()
    problems = sorted(df.problem.unique())
    w = S.problem_weights(len(problems), draws=20, seed=3)
    assert (w.sum(axis=1) == len(problems)).all()
    pred = p["B"]["prompt:combined"]
    for d in range(20):
        rows = np.concatenate([np.where(df.problem == g)[0] for g, k in zip(problems, w[d]) for _ in range(k)])
        direct = S.balanced_from(np.array(S.cell_counts(y[rows], pred[rows])))
        cells = sum(k * np.array(S.cell_counts(y[(df.problem == g).to_numpy()], pred[(df.problem == g).to_numpy()]))
                    for g, k in zip(problems, w[d]))
        assert direct == S.balanced_from(cells)
    assert len(rows) == sum(k * (df.problem == g).sum() for g, k in zip(problems, w[19]))


def test_preprocessing_fitted_on_training_rows_only():
    df = frame()
    cols = S.LENGTH
    train, test = df.iloc[3:], df.iloc[:3].copy()
    base = fp.logistic(cols)(train, test)
    skew = test.copy()
    skew.loc[skew.index[1:], cols] = 1e6  # extreme test rows must not move row 0's prediction
    assert fp.logistic(cols)(train, skew)[0] == base[0]
    held = S.oof_row(df, fp.logistic(cols))
    assert held[0] == fp.logistic(cols)(df.iloc[1:], df.iloc[:1])[0]
