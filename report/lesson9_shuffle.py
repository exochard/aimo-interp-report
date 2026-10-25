"""Label-shuffle null for the Lesson 9 agreement count.

python report/lesson9_shuffle.py TRAIN.parquet [--perms 2000] [--seed 0]

Labels are permuted among the labelled rows (problems, models and row counts stay fixed); for each
shuffle the number of multi-model problems whose labels all agree is counted with
lesson9_counts.agreement. Runs for all labelled rows and for small-track rows (every model except
gpt-oss-120b). The shuffled count is compared with the observed one.
"""
import argparse
import json
import random

from lesson9_counts import _norm, agreement

SMALL_EXCLUDE = "gpt-oss"


def labelled(rows):
    return [r for r in rows if r["is_robust"] is not None and r["is_robust"] == r["is_robust"]]


def shuffle_null(rows, perms=2000, seed=0):
    rows = labelled(rows)
    obs = agreement(rows, "model")
    rng = random.Random(seed)
    labels = [bool(r["is_robust"]) for r in rows]
    counts = []
    for _ in range(perms):
        rng.shuffle(labels)
        counts.append(agreement([dict(r, is_robust=v) for r, v in zip(rows, labels)], "model")["agree"])
    counts.sort()
    lo, hi = counts[int(0.025 * perms)], counts[min(perms - 1, int(0.975 * perms))]
    return {"observed_multi": obs["multi"], "observed_agree": obs["agree"],
            "null_mean_agree": sum(counts) / perms, "null_range_95": [lo, hi],
            "shuffles_reaching_observed": sum(c >= obs["agree"] for c in counts), "perms": perms}


def disagreeing(rows):
    """Problems labelled for several models whose labels differ, with their (model, label) pairs."""
    groups = {}
    for r in labelled(rows):
        groups.setdefault(_norm(r["problem"]), []).append((r["model_id"], bool(r["is_robust"])))
    return [sorted(g) for g in groups.values()
            if len({m for m, _ in g}) > 1 and len({v for _, v in g}) > 1]


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = pd.read_parquet(a.train)
    rows = d.astype(object).where(d.notna(), None).to_dict("records")
    small = [r for r in rows if SMALL_EXCLUDE not in r["model_id"]]
    out = {"all_labelled": shuffle_null(rows, a.perms, a.seed),
           "small_track": shuffle_null(small, a.perms, a.seed),
           "disagreeing_problems": disagreeing(rows)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
