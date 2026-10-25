"""Forced-answer flip rates split by label source (Lesson 8), from the saved rows.

  python report/flips_by_source.py runs/flip-rows-v2.parquet runs/force-r4/*.jsonl

The parquet maps each row key to its label source ('train-main-v2-1015' or
'augmented-agg-filtered'). Prints (a) the AUC of flip_mean and flip_max by (model, source) and by
source, and (b) the problem-held-out rule search of answer_flips.py evaluate, run on each source
separately: held-out correct against the majority class of the training problems.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from per_model_auc import report  # noqa: E402

FEATS = ("flip_max", "flip_mean", "orig_consistency", "no_answer")


def heldout(rows):
    """Same fit loop as answer_flips.evaluate: best (feature, threshold, sign) on the other problems."""
    def fit(train):
        best = None
        for f in FEATS:
            for t in sorted({r[f] for r in train}):
                for sign in (1, -1):  # sign 1: robust when feature <= t
                    acc = sum(((r[f] <= t) if sign == 1 else (r[f] > t)) == r["robust"] for r in train)
                    if best is None or acc > best[0]:
                        best = (acc, f, t, sign)
        maj = Counter(r["robust"] for r in train).most_common(1)[0][0]
        return best, maj

    correct = maj_correct = 0
    for p in sorted({r["problem_id"] for r in rows}):
        train = [r for r in rows if r["problem_id"] != p]
        test = [r for r in rows if r["problem_id"] == p]
        (_, f, t, sign), maj = fit(train)
        for r in test:
            pred = (r[f] <= t) if sign == 1 else (r[f] > t)
            correct += pred == r["robust"]
            maj_correct += maj == r["robust"]
    return {"n": len(rows), "heldout_correct": correct, "majority_correct": maj_correct}


SURFACE = frozenset({"typos", "distract", "rename"})


def permutation_types(value):
    """Types in a `permutation_type` cell: a bare name or a JSON list of names."""
    value = value.strip()
    return set(json.loads(value)) if value.startswith("[") else {value}


def restrict_to_types(rows, permutations, allowed=SURFACE):
    """Keep the rows whose problem has only allowed permutation types in every one of its source rows.

    `permutations` is a list of (problem text, permutation_type cell). Rows are matched by problem text.
    """
    drop = {p for p, t in permutations if not permutation_types(t) <= allowed}
    keep = {p for p, _ in permutations} - drop
    return [r for r in rows if r["problem"] in keep]


def load(parquet, jsonls):
    import pandas as pd
    d = pd.read_parquet(parquet)
    src, text = dict(zip(d["key"], d["source"])), dict(zip(d["key"], d["problem"]))
    rows = [json.loads(line) for p in jsonls for line in open(p) if line.strip()]
    for r in rows:
        r["source"], r["problem"] = src[r["key"]], text[r["key"]]
    return rows


def auc_mean_flip(rows):
    """AUC of flip_mean with robust rows as the positives (ties count half); None if a class is empty."""
    pos = [r["flip_mean"] for r in rows if r["robust"]]
    neg = [r["flip_mean"] for r in rows if not r["robust"]]
    if not pos or not neg:
        return None
    return sum((x < z) + 0.5 * (x == z) for x in pos for z in neg) / (len(pos) * len(neg))


def _draw(by, probs, rng):
    """One problem-grouped resample whose AUC is defined (redrawn when a class is empty)."""
    while True:
        a = auc_mean_flip([r for p in rng.choices(probs, k=len(probs)) for r in by[p]])
        if a is not None:
            return a


def _group(rows):
    by = {}
    for r in rows:
        by.setdefault(r["problem_id"], []).append(r)
    return by, sorted(by)


def bootstrap_auc(rows, draws=4000, seed=0):
    """95 % interval of the mean-flip AUC, resampling problem ids with replacement."""
    rng = random.Random(seed)
    by, probs = _group(rows)
    v = sorted(_draw(by, probs, rng) for _ in range(draws))
    return v[int(0.025 * draws)], v[int(0.975 * draws)]


def bootstrap_diff(a_rows, b_rows, draws=4000, seed=0):
    """95 % interval of AUC(a) - AUC(b) with independent problem resamples of each set."""
    rng = random.Random(seed)
    (ba, pa), (bb, pb) = _group(a_rows), _group(b_rows)
    v = sorted(_draw(ba, pa, rng) - _draw(bb, pb, rng) for _ in range(draws))
    return v[int(0.025 * draws)], v[int(0.975 * draws)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--permutations", help="organisers' per-permutation parquet, for the surface-only restriction")
    a = ap.parse_args()
    rows = load(a.parquet, a.files)
    groups = sorted({(r["model_id"], r["source"]) for r in rows})
    print("AUC (robust rows flip less when above 0.5)")
    for m, s in groups:
        r = report([x for x in rows if x["model_id"] == m and x["source"] == s])
        print(f"  {s} / {m}: {r['rows']} rows ({r['robust']} robust)  flip_mean={r['flip_mean']:.3f}  flip_max={r['flip_max']:.3f}")
    for s in sorted({r["source"] for r in rows}):
        sub = [x for x in rows if x["source"] == s]
        r = report(sub)
        print(f"  {s} pooled: {r['rows']} rows ({r['robust']} robust)  flip_mean={r['flip_mean']:.3f}  flip_max={r['flip_max']:.3f}")
        h = heldout(sub)
        print(f"    problem-held-out rule: {h['heldout_correct']} correct of {h['n']}, majority {h['majority_correct']}")
    print("flip_mean AUC with problem-grouped bootstrap 95 % interval (4000 draws, seed 0)")
    train = [r for r in rows if r["source"] == "train-main-v2-1015"]
    other = [r for r in rows if r["source"] != "train-main-v2-1015"]
    subsets = [("train file, all rows", train), ("organisers' set, all rows", other),
               ("train file, DeepSeek rows", [r for r in train if "DeepSeek" in r["model_id"]])]
    for name, sub in subsets:
        lo, hi = bootstrap_auc(sub)
        print(f"  {name}: {len(sub)} rows, {len({r['problem_id'] for r in sub})} problems  "
              f"AUC {auc_mean_flip(sub):.3f} [{lo:.2f}, {hi:.2f}]")
    lo, hi = bootstrap_diff(train, other)
    print(f"  difference (train file minus organisers' set): {auc_mean_flip(train) - auc_mean_flip(other):.3f} "
          f"[{lo:.2f}, {hi:.2f}]")
    if a.permutations:
        import pandas as pd
        d = pd.read_parquet(a.permutations)
        sub = restrict_to_types(other, list(zip(d["original_problem"], d["permutation_type"])))
        n_robust = sum(r["robust"] for r in sub)
        lo, hi = bootstrap_auc(sub)
        print(f"  organisers' set, typos/distract/rename only: {len(sub)} rows ({n_robust} robust), "
              f"{len({r['problem_id'] for r in sub})} problems  AUC {auc_mean_flip(sub):.3f} [{lo:.2f}, {hi:.2f}]"
              + ("  (fewer than 5 robust rows: underpowered)" if n_robust < 5 else ""))


if __name__ == "__main__":
    main()
