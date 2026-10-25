"""Problem-length baseline for the artifact's robustness classifier (Lesson 8 follow-up).

  python report/length_baseline.py

Predicts `model_is_robust` on the 141 rows of augmented-sample-math-agg from the problem text alone:
log character count, word count, digit count and `$` count, standardized, logistic regression with
balanced class weights. 5-fold StratifiedGroupKFold by problem_id, seeds 0-19. The label-shuffle null
permutes labels across problems (rows of one problem stay together) and refits the 4-feature model.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

PARQUET = "data/augmented-sample-math-agg/validation-00000-of-00001.parquet"


def features(text):
    return pd.DataFrame({
        "log_chars": np.log1p(text.str.len()),
        "words": text.str.split().str.len(),
        "digits": text.str.count(r"\d"),
        "dollars": text.str.count(r"\$"),
    })


def cv_predict(X, y, groups, seed):
    pred = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, groups):
        sc = StandardScaler().fit(X.iloc[tr])
        m = LogisticRegression(C=1, class_weight="balanced").fit(sc.transform(X.iloc[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X.iloc[te]))
    return pred


def scores(X, y, groups, seeds):
    out = []
    for s in seeds:
        p = cv_predict(X, y, groups, s)
        out.append((balanced_accuracy_score(y, p), (p == y).mean()))
    return np.array(out)


def shuffle_null(X, y, groups, n=200):
    rng = np.random.default_rng(0)
    probs = pd.unique(groups)
    first = {p: np.where(groups == p)[0][0] for p in probs}
    nulls = []
    for i in range(n):
        perm = dict(zip(probs, rng.permutation(probs)))
        ys = np.array([y[first[perm[p]]] for p in groups])
        nulls.append(balanced_accuracy_score(ys, cv_predict(X, ys, groups, i)))
    return np.array(nulls)


def main():
    d = pd.read_parquet(PARQUET)
    X = features(d.original_problem)
    y = d.model_is_robust.astype(int).values
    g = d.problem_id.values
    print(f"{len(d)} rows, {len(pd.unique(g))} problems, {y.sum()} robust")
    for name, cols in (("4 length features", list(X.columns)), ("log length alone", ["log_chars"])):
        r = scores(X[cols], y, g, range(20))
        print(f"{name}: balanced mean {r[:,0].mean():.3f} (min {r[:,0].min():.3f}, max {r[:,0].max():.3f}), "
              f"accuracy mean {r[:,1].mean():.3f}")
    n = shuffle_null(X, y, g)
    print(f"label-shuffle null (200, 4 features): median {np.median(n):.3f}, "
          f"95th percentile {np.percentile(n, 95):.3f}, max {n.max():.3f}")
    print("artifact README (kit/solutions/uncertainty-profiling/README.md:127-128): balanced 0.6262, accuracy 0.6879")
    print(f"always non-robust accuracy: {(y == 0).mean():.3f} ({(y == 0).sum()}/{len(y)})")


if __name__ == "__main__":
    main()
