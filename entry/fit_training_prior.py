"""Fit the training-only hierarchical prior of the v3 entries.

  python entry/fit_training_prior.py --train PARQUET [--out entry/training-prior.json]

Reads the organisers' `train-main-v2` file (sha256 must be one of KNOWN), keeps rows with a True/False
`is_robust`, groups them by whole problem and counts robust / non-robust per (model_id,
reasoning_effort). A configuration's posterior is the beta-binomial mean
(robust + k * g) / (n + k) shrunk toward the track's global training rate g; k comes from a fixed
grid and is chosen by leave-one-problem-out log loss inside the training data (ties go to the larger k).
Configurations with fewer than MIN_N labelled rows use g directly. No public-set label, problem or
score enters; the JSON holds counts and probabilities only, never problem text.
"""

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

KNOWN = {
    "bedfb551ea0f4d95d5e70e1df7cf61395ce2d36eeba2869aba6a90fc3c198824": "pinned (revision 23791d9b, 82 rows)",
    "e18eb44207e7987781ec2ff214793196038442bdc1466fdf5498f3e87ec58285": "1015 (file of 2026-10-15, 94 rows)",
}
TRACK_MODELS = {
    "small": ["Qwen/Qwen3.5-4B", "Skywork/Skywork-OR1-Math-7B", "allenai/Olmo-3-7B-Think",
              "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"],
    "main": ["openai/gpt-oss-120b"],
}
GRID = [1, 2, 4, 8, 16, 32, 64]
MIN_N = 5
THRESHOLD = 0.5
EPS = 1e-6


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    """Labelled rows of one known train file as (frame, sha256)."""
    sha = sha256(path)
    if sha not in KNOWN:
        raise ValueError(f"{path} has sha256 {sha}, which matches no known train file")
    df = pd.read_parquet(path)
    df = df[df.is_robust.notna()].copy()
    df["y"] = df.is_robust.astype(bool)
    df["pid"] = df.problem.map(lambda t: re.sub(r"\s+", " ", t).strip().lower())
    return df.reset_index(drop=True), sha


def track_rows(df, track):
    return df[df.model_id.isin(TRACK_MODELS[track])].reset_index(drop=True)


def counts(df):
    """{(model_id, effort): [robust, non_robust]} over the rows."""
    out = {}
    for m, e, y in zip(df.model_id, df.reasoning_effort, df.y):
        out.setdefault((m, e), [0, 0])[0 if y else 1] += 1
    return out


def global_rate(df):
    return float(df.y.mean())


def posterior(c, g, k):
    n = c[0] + c[1]
    return g if n < MIN_N else (c[0] + k * g) / (n + k)


def decide(p):
    return p >= THRESHOLD


def inner_log_loss(df, k):
    """Leave-one-problem-out log loss of strength k on df (every fit inside the training folds)."""
    total = 0.0
    for pid in df.pid.unique():
        tr, te = df[df.pid != pid], df[df.pid == pid]
        g, cnt = global_rate(tr), counts(tr)
        for m, e, y in zip(te.model_id, te.reasoning_effort, te.y):
            p = min(max(posterior(cnt.get((m, e), [0, 0]), g, k), EPS), 1 - EPS)
            total -= math.log(p if y else 1 - p)
    return total / len(df)


def choose_k(df):
    losses = {k: inner_log_loss(df, k) for k in GRID}
    best = min(losses.values())
    return max(k for k in GRID if losses[k] <= best + 1e-12), losses


def fit_fold(train):
    """Everything a fold needs from its training rows: chosen k, global rate, counts."""
    k, _ = choose_k(train)
    return k, global_rate(train), counts(train)


def predict_prior(fit, test):
    k, g, cnt = fit
    return [decide(posterior(cnt.get((m, e), [0, 0]), g, k)) for m, e in zip(test.model_id, test.reasoning_effort)]


def predict_always(fit, test):
    return [True] * len(test)


def predict_per_model(fit, test):
    """Majority of the training fold per model_id (ties robust); unknown model: fold majority."""
    _, g, cnt = fit
    per = {}
    for (m, _e), (r, n) in cnt.items():
        per.setdefault(m, [0, 0])
        per[m][0] += r
        per[m][1] += n
    return [decide(per[m][0] / (per[m][0] + per[m][1])) if m in per else decide(g) for m in test.model_id]


ARMS = {"prior": predict_prior, "always_robust": predict_always, "per_model_majority": predict_per_model}


def lopo(df):
    """Outer leave-one-problem-out: per-row predictions of every arm, all choices inside the training fold."""
    preds = {a: pd.Series(False, index=df.index) for a in ARMS}
    for pid in df.pid.unique():
        te = df.pid == pid
        fit = fit_fold(df[~te])
        for a, f in ARMS.items():
            preds[a][te] = f(fit, df[te])
    return preds


def score(df, preds):
    """Correct counts overall and per model_id for each arm."""
    out = {}
    for a, p in preds.items():
        ok = p == df.y
        out[a] = {"correct": int(ok.sum()), "n": int(len(df)),
                  "by_model": {m: [int(ok[df.model_id == m].sum()), int((df.model_id == m).sum())]
                               for m in sorted(df.model_id.unique())}}
    return out


def gate(s):
    """Non-inferior within one row overall and per model against both baselines."""
    for base in ("always_robust", "per_model_majority"):
        if s["prior"]["correct"] < s[base]["correct"] - 1:
            return False
        for m, (c, _) in s["prior"]["by_model"].items():
            if c < s[base]["by_model"][m][0] - 1:
                return False
    return True


def evaluate(frames):
    """frames: {set name: (track, labelled frame)}. Returns {name: {scores, gate}}."""
    out = {}
    for name, (track, df) in frames.items():
        s = score(df, lopo(df))
        out[name] = {"track": track, "scores": s, "gate_passed": gate(s)}
    return out


def fit(paths):
    """paths: {"1015": file defining the shipped fit and the 43-row set, "pinned": file defining the 34-row set}."""
    full, sha = load(paths["1015"])
    pinned, pin_sha = load(paths["pinned"])
    if not (sha.startswith("e18eb442") and pin_sha.startswith("bedfb551")):
        raise ValueError("--train must be the 2026-10-15 file and --pinned the pinned revision")
    frames = {"small-34 (pinned)": ("small", track_rows(pinned, "small")),
              "small-43 (1015)": ("small", track_rows(full, "small")),
              "main-37 (pinned)": ("main", track_rows(pinned, "main")),
              "main-37 (1015)": ("main", track_rows(full, "main"))}
    ev = evaluate(frames)
    tracks = {}
    for track in TRACK_MODELS:
        df = track_rows(full, track)
        k, losses = choose_k(df)
        g, cnt = global_rate(df), counts(df)
        gate_ok = all(v["gate_passed"] for v in ev.values() if v["track"] == track)
        configs = []
        for (m, e), c in sorted(cnt.items()):
            p = posterior(c, g, k)
            configs.append({"model_id": m, "reasoning_effort": e, "robust": c[0], "non_robust": c[1],
                            "posterior": p, "used_global": c[0] + c[1] < MIN_N, "robust_decision": decide(p)})
        tracks[track] = {"rows": len(df), "problems": int(df.pid.nunique()), "robust": int(df.y.sum()),
                         "global_posterior": g, "global_decision": decide(g), "prior_strength": k,
                         "inner_log_loss": {str(a): b for a, b in losses.items()},
                         "gate_passed": gate_ok, "selected_rule": "hierarchical" if gate_ok else "global",
                         "configurations": configs}
    return {"input_sha256": sha, "input_file": KNOWN[sha], "pinned_sha256": pin_sha,
            "grid": GRID, "min_n": MIN_N, "threshold": THRESHOLD, "tracks": tracks}, ev


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="train-2026-10-15.parquet (sha256 e18eb442...)")
    ap.add_argument("--pinned", required=True, help="train-00000-of-00001.parquet (sha256 bedfb551...)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "training-prior.json"))
    a = ap.parse_args(argv)
    prior, _ = fit({"1015": a.train, "pinned": a.pinned})
    Path(a.out).write_text(json.dumps(prior, indent=2, sort_keys=True) + "\n")
    for t, v in prior["tracks"].items():
        print(t, "k =", v["prior_strength"], "global", round(v["global_posterior"], 4), v["selected_rule"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
