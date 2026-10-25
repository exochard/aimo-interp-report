"""Does the answer-changing control move the trace likelihood more than the safe variants? (Lesson 3)

  python report/trace_control.py runs/trace-pilot/rows.jsonl [--labels data/train-main-v2/train-00000-of-00001.parquet]

`neg_ratio` (sub/trace_features.py) is the mean answer-preserving log-likelihood change divided by
the answer-changing control's change, so a ratio below 1 means the control moved the likelihood
more. Rows without a control have neg_missing = 1. With --labels the count is repeated on the rows
that have a label (the 34 the probe is scored on).
"""
import argparse
import json
import math
import statistics


def summarise(rows):
    ratios = [r["features"]["neg_ratio"] for r in rows
              if not r["features"]["neg_missing"] and not math.isnan(r["features"]["neg_ratio"])]
    return {"rows": len(rows), "with_control": len(ratios), "control_moved_more": sum(x < 1 for x in ratios),
            "median_ratio": round(statistics.median(ratios), 2) if ratios else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--labels")
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.rows) if line.strip()]
    print("all traced rows:", summarise(rows))
    if a.labels:
        import pandas as pd
        t = pd.read_parquet(a.labels)
        lab = {(r.model_id, r.problem) for r in t.itertuples() if r.is_robust is not None and r.is_robust == r.is_robust}
        print("labelled rows:", summarise([r for r in rows if (r["model_id"], r["problem"]) in lab]))


if __name__ == "__main__":
    main()
