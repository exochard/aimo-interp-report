"""Per-model AUC of the forced-answer flip rates (Lesson 8), from the saved rows.

  python report/per_model_auc.py runs/force-r4/*.jsonl [--pool Qwen3.5 --pool Skywork]

Convention as in answer_flips.py evaluate: AUC is the probability that a robust row has a lower
feature value than a non-robust row (ties count half). Above 0.5 means robust rows flip less.
--pool NAME (repeatable) adds one extra line pooling every model whose id contains any NAME.
"""
import argparse
import json


def auc(rows, feature):
    pos = [r[feature] for r in rows if r["robust"]]
    neg = [r[feature] for r in rows if not r["robust"]]
    if not pos or not neg:
        return None
    pairs = [(x < z) + 0.5 * (x == z) for x in pos for z in neg]
    return sum(pairs) / len(pairs)


def report(rows, features=("flip_mean", "flip_max")):
    feats = [f for f in features if rows and all(f in r for r in rows)]
    return {"rows": len(rows), "robust": sum(bool(r["robust"]) for r in rows),
            **{f: auc(rows, f) for f in feats}}


def by_model(rows, pools=()):
    models = sorted({r["model_id"] for r in rows})
    out = {m: report([r for r in rows if r["model_id"] == m]) for m in models}
    if pools:
        out["pooled: " + " + ".join(pools)] = report(
            [r for r in rows if any(p in r["model_id"] for p in pools)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--pool", action="append", default=[])
    a = ap.parse_args()
    rows = [json.loads(line) for p in a.files for line in open(p) if line.strip()]
    for name, r in by_model(rows, a.pool).items():
        vals = "  ".join(f"{k}={v:.3f}" for k, v in r.items() if k not in ("rows", "robust") and v is not None)
        print(f"{name}: {r['rows']} rows ({r['robust']} robust)  {vals}")


if __name__ == "__main__":
    main()
