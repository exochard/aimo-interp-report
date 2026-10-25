"""Does answer agreement under our perturbations predict the train-main-v2 robustness labels?

Reads runs/*/gen-*.jsonl from the Kaggle generation kernels (kaggle/gen/run.py) and joins each
(model, problem) with train-main-v2 (is_robust, max_drop). No reference answers exist, so every
feature is reference-free: agreement of perturbed samples with the original's modal answer.

  uv run --no-project --with pandas --with pyarrow --with scipy python analyze_gen.py
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent


def features(rows: list[dict]) -> dict:
    by = collections.defaultdict(list)
    for r in rows:
        by[r["variant"]].append(r)
    orig = [r["answer"] for r in by["original"]]
    counts = collections.Counter(a for a in orig if a is not None)
    mode, top = counts.most_common(1)[0] if counts else (None, 0)
    out = {"n_orig": len(orig), "orig_consistency": top / max(len(orig), 1),
           "orig_unanswered": sum(a is None for a in orig) / max(len(orig), 1),
           "orig_truncated": np.mean([r["finish"] == "length" for r in by["original"]]),
           "orig_tokens": np.mean([r["n_tokens"] for r in by["original"]])}
    for v in ("typos", "distract"):
        vs = by.get(v, [])
        agree = np.mean([r["answer"] == mode and mode is not None for r in vs]) if vs else np.nan
        out[f"agree_{v}"] = agree
        out[f"drop_{v}"] = out["orig_consistency"] - agree
        out[f"dtokens_{v}"] = np.mean([r["n_tokens"] for r in vs]) / max(out["orig_tokens"], 1) - 1 if vs else np.nan
        lp = [r["mean_logprob"] for r in vs if r["mean_logprob"] is not None]
        lo = [r["mean_logprob"] for r in by["original"] if r["mean_logprob"] is not None]
        out[f"dlogprob_{v}"] = np.mean(lp) - np.mean(lo) if lp and lo else np.nan
    out["max_drop_proxy"] = max(out["drop_typos"], out["drop_distract"])
    out["min_agree"] = min(out["agree_typos"], out["agree_distract"])
    return out


def loo_threshold_accuracy(x: np.ndarray, y: np.ndarray) -> float:
    """Leave-one-out: the rule (direction and threshold of x -> robust) is fit without the held row."""
    hits = 0
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        best = (-1.0, 1, 0.0)
        for sign in (1, -1):
            xs, ys = sign * x[keep], y[keep]
            for t in np.unique(np.concatenate([xs, [xs.max() + 1]])):
                acc = float(np.mean((xs <= t) == ys))
                if acc > best[0]:
                    best = (acc, sign, t)
        _, sign, t = best
        hits += int((sign * x[i] <= t) == y[i])
    return hits / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", action="store_true",
                    help="Spearman on the labelled rows only, print the largest |rho| and write no files")
    args = ap.parse_args()
    labels = pd.read_parquet(HERE / "data/train-main-v2/train-00000-of-00001.parquet")
    rows = [json.loads(line) for f in sorted(glob.glob(str(HERE / "runs/*/gen-*.jsonl"))) for line in open(f)]
    if not rows:
        raise SystemExit("no generations under runs/*/gen-*.jsonl yet")
    grouped = collections.defaultdict(list)
    for r in rows:
        grouped[(r["model_id"], r["problem"])].append(r)
    table = []
    for _, lab in labels.iterrows():
        key = (lab.model_id, lab.problem)
        if key in grouped:
            table.append({"model_id": lab.model_id, "is_robust": lab.is_robust, "max_drop": lab.max_drop,
                          **features(grouped[key])})
    df = pd.DataFrame(table)
    print("models with generations:", sorted({m for m, _ in grouped}))
    print(f"joined rows {len(df)}, labelled {df.is_robust.notna().sum()}")
    report = {"rows": len(df), "labelled": int(df.is_robust.notna().sum()), "features": {}}
    lab = df[df.is_robust.notna()]
    y = lab.is_robust.astype(bool).to_numpy()
    base = max(y.mean(), 1 - y.mean())
    report["majority_baseline"] = base
    for col in ["orig_consistency", "orig_unanswered", "orig_truncated", "agree_typos", "agree_distract",
                "drop_typos", "drop_distract", "max_drop_proxy", "min_agree", "dtokens_typos",
                "dtokens_distract", "dlogprob_typos", "dlogprob_distract"]:
        src = lab if args.labelled else df
        x = src[col].to_numpy(dtype=float)
        ok = ~np.isnan(x)
        rho, p = spearmanr(x[ok], src.max_drop.to_numpy()[ok]) if ok.sum() > 3 and np.nanstd(x[ok]) > 0 else (np.nan, np.nan)
        xl = lab[col].to_numpy(dtype=float)
        acc = loo_threshold_accuracy(xl, y) if len(y) > 3 and not np.isnan(xl).any() else np.nan
        report["features"][col] = {"spearman_vs_max_drop": rho, "p": p, "loo_threshold_acc": acc}
        print(f"{col:18s} rho={rho:+.2f} (p={p:.2f})  LOO acc={acc:.2f}  (majority {base:.2f})")
    if args.labelled:
        top = max(report["features"].items(), key=lambda kv: abs(kv[1]["spearman_vs_max_drop"])
                  if not np.isnan(kv[1]["spearman_vs_max_drop"]) else -1)
        print(f"largest |rho| on {len(lab)} labelled rows: {top[0]} rho={top[1]['spearman_vs_max_drop']:.2f} "
              f"p={top[1]['p']:.2f}")
        return
    print(df.groupby("model_id")[["max_drop", "orig_consistency", "max_drop_proxy", "min_agree"]].mean().round(2))
    out = HERE / "runs/agreement-vs-labels.json"
    out.write_text(json.dumps(report, indent=1, default=float))
    df.to_csv(HERE / "runs/agreement-vs-labels.csv", index=False)
    print("->", out)


if __name__ == "__main__":
    main()
