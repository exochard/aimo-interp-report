"""Counts quoted in Lesson 6 of the report: among organiser-augmented rows where every base
sample was correct (base_accuracy == 1.0), the share labelled non-robust, at any number of base
samples and restricted to n_base_predictions == 10. Also prints the unconditional non-robust
share over all rows and the number of unique problem_id (rows are per permutation type, so several
rows share one problem).

python report/lesson6_counts.py AGG_FILTERED.parquet AGG.parquet
"""
import json
import sys


def nonrobust_share(rows, n_base=None):
    sel = [r for r in rows if r["base_accuracy"] == 1.0
           and (n_base is None or r["n_base_predictions"] == n_base)]
    bad = sum(not r["model_is_robust"] for r in sel)
    return {"solved_rows": len(sel), "non_robust": bad, "share": bad / len(sel) if sel else None}


def overall(rows):
    """Unconditional non-robust share over all rows, and the unique problem count."""
    bad = sum(not r["model_is_robust"] for r in rows)
    return {"rows": len(rows), "non_robust": bad, "share": bad / len(rows) if rows else None,
            "solved_rows": sum(r["base_accuracy"] == 1.0 for r in rows),
            "unique_problems": len({r["problem_id"] for r in rows})}


def main(paths):
    import pandas as pd
    out = {}
    for p in paths:
        d = pd.read_parquet(p)
        rows = d.to_dict("records")
        out[p] = {"models": sorted(d.model_id.unique()), "rows": len(rows),
                  "base_samples_seen": sorted(int(n) for n in d.n_base_predictions.unique()),
                  "all_rows": overall(rows),
                  "any_sample_count": nonrobust_share(rows),
                  "ten_samples": nonrobust_share(rows, n_base=10)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
