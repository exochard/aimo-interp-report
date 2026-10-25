"""Counts quoted in Lesson 9 of the report, computed from the organisers' parquet files.

python report/lesson9_counts.py TRAIN.parquet [TRAIN2.parquet ...] --public PUBLIC.parquet
"""
import argparse
import json
from collections import defaultdict


def _norm(text):
    return " ".join(str(text).split())


def agreement(rows, by):
    """Among problems with a label, how many are labelled for more than one model
    (by="model") or model configuration (by="config", model plus reasoning effort),
    and for how many of those every label is the same."""
    groups = defaultdict(list)
    for r in rows:
        if r["is_robust"] is None or r["is_robust"] != r["is_robust"]:
            continue
        who = r["model_id"] if by == "model" else (r["model_id"], r["reasoning_effort"])
        groups[_norm(r["problem"])].append((who, bool(r["is_robust"])))
    multi = agree = 0
    for labels in groups.values():
        if len({w for w, _ in labels}) > 1:
            multi += 1
            agree += len({v for _, v in labels}) == 1
    return {"labelled_problems": len(groups), "multi": multi, "agree": agree}


def overlap(public_problems, train_rows):
    """Public problems that also appear in the training rows (any row / labelled row)."""
    seen = {_norm(r["problem"]) for r in train_rows}
    lab = {_norm(r["problem"]) for r in train_rows
           if r["is_robust"] is not None and r["is_robust"] == r["is_robust"]}
    pub = {_norm(p) for p in public_problems}
    return {"public_problems": len(pub), "in_train": len(pub & seen),
            "in_train_labelled": len(pub & lab)}


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("train", nargs="+")
    ap.add_argument("--public", required=True)
    a = ap.parse_args()
    pub = pd.read_parquet(a.public).original_problem.unique().tolist()
    out = {}
    for path in a.train:
        d = pd.read_parquet(path)
        d = d.astype(object).where(d.notna(), None)
        rows = d.to_dict("records")
        out[path] = {
            "rows": len(rows),
            "by_model": agreement(rows, "model"),
            "by_config": agreement(rows, "config"),
            "public_overlap": overlap(pub, rows),
        }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
